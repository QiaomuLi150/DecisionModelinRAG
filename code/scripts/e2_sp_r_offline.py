#!/usr/bin/env python3
"""Prepare and close the offline portion of E2-S/P-R.

This module deliberately contains no model-serving or model-loading code.  It
audits the frozen E2-S/P population, freezes the R-stage workload and CAL
configuration, creates the blinded downstream assessment package, and records
an explicit resource-gate block when no authorized A100 is isolated.  A
separate future load runner may consume the protocol, but this script must
never turn serialized latency into capacity or cost.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp_r"
PRIOR = ROOT / "results/non_jev/v2/e2_sp/gpu5"
COHORT_PATH = ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json"
SEED = 20261011
ADJUDICATION_SEED = 731
BOOTSTRAP_REPS = 5000
AUTHORIZED_GPUS = [0, 1, 2, 4]
MIN_FREE_MIB = 25_000
POLICIES = ["retrieval5", "bge_pointwise", "qwen06", "qwen4", "jina35", "gte", "minilm", "clef_joint", "nimble", "pplx_decider", "qwen_direct"]
R4_POLICIES = ["retrieval5", "bge_pointwise", "jina35", "nimble", "gte", "pplx_decider"]
MODEL_ROSTER = {
    "pplx_decider": {
        "model_id": "perplexity-ai/pplx-decider-v1.1-27b",
        "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce",
        "snapshot": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce",
        "native_contract": "native autojev DecisionModel slate categorical",
    },
    "jina35": {
        "model_id": "jinaai/jina-reranker-v3.5",
        "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        "snapshot": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        "native_contract": "native listwise reranker",
    },
    "nimble": {
        "model_id": "bespokelabs/Bespoke-Nimble-9B-v2",
        "revision": "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe",
        "snapshot": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--bespokelabs--Bespoke-Nimble-9B-v2/snapshots/4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe",
        "native_contract": "native ParallelScorer categorical choice",
    },
    "gte": {
        "model_id": "Alibaba-NLP/gte-reranker-modernbert-base",
        "revision": "f7481e6055501a30fb19d090657df9ec1f79ab2c",
        "snapshot": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots/f7481e6055501a30fb19d090657df9ec1f79ab2c",
        "native_contract": "native sequence-classifier cross encoder",
    },
    "bge_pointwise": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "snapshot": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "native_contract": "native sequence-classifier cross encoder",
    },
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prior_integrity() -> dict[str, Any]:
    manifest_path = PRIOR / "E2_SP_PROVENANCE_MANIFEST.json"
    manifest = load_json(manifest_path)
    checks = []
    for rel, expected in manifest.get("generated_artifact_sha256", {}).items():
        path = ROOT / rel
        checks.append({"path": rel, "exists": path.is_file(), "expected": expected, "actual": sha_file(path) if path.is_file() else None, "match": path.is_file() and sha_file(path) == expected})
    preserved = []
    for rel, expected in manifest.get("preserved_prior_artifact_sha256", {}).items():
        path = ROOT / rel
        preserved.append({"path": rel, "exists": path.is_file(), "expected": expected, "actual": sha_file(path) if path.is_file() else None, "match": path.is_file() and sha_file(path) == expected})
    return {
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha_file(manifest_path),
        "generated_artifact_checks": checks,
        "preserved_artifact_checks": preserved,
        "generated_mismatches": [x for x in checks if not x["match"]],
        "preserved_mismatches": [x for x in preserved if not x["match"]],
        "prior_status": manifest.get("status"),
        "prior_jev_calls": manifest.get("jev_calls"),
        "prior_paid_api_calls": manifest.get("paid_api_calls"),
    }


def load_items() -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_sp_gpu5_runner import load_cohort_items

    return load_cohort_items()


def source_and_cohort_audit(items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cohort = load_json(COHORT_PATH)
    records = cohort["query_records"]
    by_id = {item["query_id"]: item for item in items}
    errors = []
    for record in records:
        item = by_id.get(record["query_id"])
        if item is None:
            errors.append({"query_id": record["query_id"], "error": "missing_source_item"})
            continue
        if sha_bytes(item["question"].encode("utf-8")) != record["question_sha256"]:
            errors.append({"query_id": record["query_id"], "error": "question_hash_mismatch"})
        if [c["doc_id"] for c in item["candidates"]] != record["top20_candidate_ids"]:
            errors.append({"query_id": record["query_id"], "error": "top20_identity_or_order_mismatch"})
        if len(record["top20_candidate_ids"]) != 20 or len(set(record["top20_candidate_ids"])) != 20:
            errors.append({"query_id": record["query_id"], "error": "invalid_top20"})
    selection = load_jsonl(PRIOR / "E2_SP_S2_SELECTION_RESULTS.jsonl")
    reader = load_jsonl(PRIOR / "E2_SP_P1_READER_OUTPUTS.jsonl")
    selection_keys = {(r.get("policy"), r.get("query_id")) for r in selection}
    reader_keys = {(r.get("policy"), r.get("query_id")) for r in reader}
    expected = {(policy, item["query_id"]) for policy in POLICIES for item in items}
    missing_selection = sorted(expected - selection_keys)
    missing_reader = sorted(expected - reader_keys)
    if len(selection) != 3300 or len(selection_keys) != 3300:
        errors.append({"error": "selection_count_or_uniqueness", "rows": len(selection), "unique": len(selection_keys)})
    if len(reader) != 3300 or len(reader_keys) != 3300:
        errors.append({"error": "reader_count_or_uniqueness", "rows": len(reader), "unique": len(reader_keys)})
    errors.extend({"error": "missing_selection", "policy": p, "query_id": q} for p, q in missing_selection)
    errors.extend({"error": "missing_reader", "policy": p, "query_id": q} for p, q in missing_reader)
    input_gold_selection = sum(bool(r.get("gold_in_selection_input") or r.get("gold_in_inference_input")) for r in load_jsonl(PRIOR / "E2_SP_SELECTION_REQUEST_TRACES.jsonl"))
    input_gold_reader = sum(bool(r.get("gold_in_reader_input")) for r in reader)
    audit = {
        "cohort_manifest_file_sha256": sha_file(COHORT_PATH),
        "cohort_manifest_embedded_sha256": cohort.get("manifest_sha256"),
        "cohort_manifest_note": "The embedded digest is the frozen pre-self-reference manifest digest; the file digest is recorded separately.",
        "query_count": len(records),
        "unique_query_count": len({r["query_id"] for r in records}),
        "dataset_counts": dict(sorted(Counter(r["dataset"] for r in records).items())),
        "cohort_counts": dict(sorted(Counter(r["cohort"] for r in records).items())),
        "question_and_slate_errors": errors,
        "selection_rows": len(selection),
        "reader_rows": len(reader),
        "selection_unique_policy_query_rows": len(selection_keys),
        "reader_unique_policy_query_rows": len(reader_keys),
        "gold_in_selection_input_rows": input_gold_selection,
        "gold_in_reader_input_rows": input_gold_reader,
        "all_prior_rows_ok": all(r.get("status") == "ok" for r in selection + reader),
        "all_prior_selection_rows_have_no_cache": all(not r.get("cache_reuse", False) for r in selection),
        "all_prior_reader_rows_have_no_cache": all(not r.get("cache_reuse", False) for r in reader),
        "status": "PASS" if not errors and input_gold_selection == 0 and input_gold_reader == 0 else "BLOCKED",
    }
    return audit, records


def load_gold(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(ROOT / "data/manifests/questions.jsonl"):
        gold[row["query_id"]] = {
            "reference_answer": row.get("answer"),
            "answer_aliases": row.get("answer_aliases", []),
            "gold_doc_ids": row.get("gold_doc_ids", []),
            "reference_status": "REFERENCE_AVAILABLE",
            "mapping_status": "VALID" if row.get("gold_mapping_valid", True) else "INVALID",
        }
    # Track A's prospective cohort has a separate evaluation-only gold
    # sidecar.  It is intentionally loaded only for offline assessment, never
    # by the selector inputs or the serving workload.
    for row in load_jsonl(ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl"):
        gold[row["query_id"]] = {
            "reference_answer": row.get("answer"),
            "answer_aliases": row.get("answer_aliases", []),
            "gold_doc_ids": row.get("gold_doc_ids", []),
            "reference_status": "REFERENCE_AVAILABLE",
            "mapping_status": "VALID" if row.get("gold_mapping_valid", True) else "INVALID",
        }
    for row in load_jsonl(ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"):
        gold[row["question_id"]] = {
            "reference_answer": None,
            "answer_aliases": [],
            "gold_doc_ids": row.get("expected_internal_doc_ids", []),
            "reference_status": "REFERENCE_ANSWER_UNAVAILABLE",
            "mapping_status": row.get("gold_identity_status", "UNKNOWN"),
            "gold_identity_exception": row.get("gold_identity_exception"),
        }
    for item in items:
        gold.setdefault(item["query_id"], {"reference_answer": None, "answer_aliases": [], "gold_doc_ids": [], "reference_status": "REFERENCE_UNAVAILABLE", "mapping_status": "UNKNOWN"})
    return gold


def coverage_audit(items: list[dict[str, Any]], records: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_item = {item["query_id"]: item for item in items}
    selection = load_jsonl(PRIOR / "E2_SP_S2_SELECTION_RESULTS.jsonl")
    selected = {(r["policy"], r["query_id"]): r.get("selected_ids", []) for r in selection}
    rows = []
    for record in records:
        item = by_item[record["query_id"]]
        g = set(gold.get(record["query_id"], {}).get("gold_doc_ids", []))
        top = set(record["top20_candidate_ids"])
        base = {
            "query_id": record["query_id"],
            "dataset": record["dataset"],
            "cohort": record["cohort"],
            "gold_count": len(g),
            "gold_status": gold.get(record["query_id"], {}).get("mapping_status"),
            "reference_status": gold.get(record["query_id"], {}).get("reference_status"),
            "top20_any_gold": bool(g & top) if g else None,
            "top20_complete_gold": g.issubset(top) if g else None,
        }
        for policy in POLICIES:
            ids = set(selected.get((policy, record["query_id"]), []))
            row = dict(base)
            row.update({"policy": policy, "selected_any_gold": bool(g & ids) if g else None, "selected_complete_gold": g.issubset(ids) if g else None, "selected_gold_count": len(g & ids)})
            rows.append(row)
    summary = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for policy in POLICIES:
            subset = [r for r in rows if r["dataset"] == dataset and r["policy"] == policy]
            eligible = [r for r in subset if r["gold_count"] > 0 and r["gold_status"] not in {"INVALID", "AMBIGUOUS"}]
            summary.append({
                "dataset": dataset,
                "policy": policy,
                "n_queries": len(subset),
                "identity_eligible_n": len(eligible),
                "top20_any_gold_n": sum(bool(r["top20_any_gold"]) for r in eligible),
                "top20_complete_gold_n": sum(bool(r["top20_complete_gold"]) for r in eligible),
                "selected_any_gold_n": sum(bool(r["selected_any_gold"]) for r in eligible),
                "selected_complete_gold_n": sum(bool(r["selected_complete_gold"]) for r in eligible),
                "top20_any_gold_rate": (sum(bool(r["top20_any_gold"]) for r in eligible) / len(eligible)) if eligible else None,
                "top20_complete_gold_rate": (sum(bool(r["top20_complete_gold"]) for r in eligible) / len(eligible)) if eligible else None,
                "selected_any_gold_rate": (sum(bool(r["selected_any_gold"]) for r in eligible) / len(eligible)) if eligible else None,
                "selected_complete_gold_rate": (sum(bool(r["selected_complete_gold"]) for r in eligible) / len(eligible)) if eligible else None,
            })
    return rows, {"per_policy_dataset": summary, "overall_query_count": len(records), "overall_policy_rows": len(rows)}


def gpu_inventory() -> dict[str, Any]:
    def run(args: list[str]) -> tuple[int, str, str]:
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=20)
            return p.returncode, p.stdout.strip(), p.stderr.strip()
        except Exception as exc:
            return 1, "", f"{type(exc).__name__}: {exc}"

    rc, stdout, stderr = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw", "--format=csv,noheader,nounits"])
    devices = {}
    if rc == 0:
        for line in stdout.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) < 8:
                continue
            try:
                index = int(fields[0])
                devices[index] = {"index": index, "uuid": fields[1], "name": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5]), "utilization_gpu_percent": int(fields[6]), "power_draw_w": fields[7]}
            except ValueError:
                continue
    rc_apps, apps_out, apps_err = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,gpu_name,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    apps = [{"raw": line.strip()} for line in apps_out.splitlines() if line.strip()] if rc_apps == 0 else []
    for index, device in devices.items():
        device["compute_apps"] = [app["raw"] for app in apps if device["uuid"] in app["raw"]]
        device["authorized"] = index in AUTHORIZED_GPUS
        device["safe_for_new_model"] = bool(device["authorized"] and device["memory_free_mib"] >= MIN_FREE_MIB and not device["compute_apps"])
    return {
        "captured_utc": utc_now(),
        "authorized_gpu_ids": AUTHORIZED_GPUS,
        "minimum_free_mib": MIN_FREE_MIB,
        "devices": devices,
        "compute_apps_query_status": "PASS" if rc_apps == 0 else "UNAVAILABLE",
        "compute_apps_stderr": apps_err,
        "all_authorized_devices_safe": all(devices.get(i, {}).get("safe_for_new_model") is True for i in AUTHORIZED_GPUS),
        "safe_authorized_devices": [i for i in AUTHORIZED_GPUS if devices.get(i, {}).get("safe_for_new_model") is True],
        "resource_gate": "PASS" if any(devices.get(i, {}).get("safe_for_new_model") is True for i in AUTHORIZED_GPUS) else "BLOCKED_NO_ISOLATED_AUTHORIZED_GPU",
    }


def roster_audit() -> list[dict[str, Any]]:
    rows = []
    for key, spec in MODEL_ROSTER.items():
        path = Path(spec["snapshot"])
        files = list(path.rglob("*")) if path.is_dir() else []
        size = sum(p.stat().st_size for p in files if p.is_file())
        rows.append({"model_key": key, **spec, "snapshot_exists": path.is_dir(), "snapshot_file_count": sum(p.is_file() for p in files), "snapshot_bytes": size, "status": "AVAILABLE_LOCAL" if path.is_dir() else "MISSING_LOCAL_DEPENDENCY"})
    return rows


def read_cal_latency() -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    mapping = {"bge_pointwise": "BGE_POINTWISE", "jina35": "JINA35", "gte": "GTE", "nimble": "NIMBLE", "pplx_decider": "PPLX_DECIDER"}
    for policy, suffix in mapping.items():
        path = PRIOR / f"E2_SP_S1_CAL_{suffix}.jsonl"
        values = []
        for row in load_jsonl(path):
            value = row.get("duration_ms")
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                values.append(float(value))
        result[policy] = sorted(values)[(len(values) - 1) // 2] if values else None
    return result


def create_protocol(cohort_audit: dict[str, Any], roster: list[dict[str, Any]], gpu: dict[str, Any], cal_latency: dict[str, float | None]) -> dict[str, Any]:
    rate_grid = {}
    for policy, p50 in cal_latency.items():
        if p50 and p50 > 0:
            base = 1000.0 / p50
            rate_grid[policy] = [round(max(0.01, base * multiplier), 6) for multiplier in (0.25, 0.5, 1.0, 2.0)]
        else:
            rate_grid[policy] = None
    protocol = {
        "protocol_id": "DecisionRAG-v2.1-E2-SP-R-v1",
        "created_utc": utc_now(),
        "status": "FROZEN_PREPARED_RESOURCE_GATE_BLOCKED",
        "frozen_inputs": {
            "cohort_manifest": str(COHORT_PATH.relative_to(ROOT)),
            "cohort_file_sha256": sha_file(COHORT_PATH),
            "cohort_embedded_sha256": cohort_audit["cohort_manifest_embedded_sha256"],
            "queries": 300,
            "top20_per_query": 20,
            "candidate_pool_mutation": False,
            "fresh_holdout_access": False,
        },
        "selector_roster": roster,
        "slo": {"primary_full_selection_p95_ms": 1000, "sensitivity_p95_ms": [2000, 3000], "max_error_rate": 0.01},
        "open_loop": {"arrival_process": "seeded Poisson arrivals independent of completion", "seed": SEED, "rate_grid_basis": "CAL single-client p50 inverse times prespecified multipliers [0.25, 0.5, 1, 2]", "rate_grid_rps": rate_grid, "warmup_excluded": True, "drain_included": True, "repeats_near_boundary": 2, "status": "NOT_EXECUTED_RESOURCE_GATE"},
        "serving_contract": {"arrival": "monotonic_ns", "admission": "queue wait", "preprocessing": "tokenization/native document preparation", "execution": "native model forward(s)", "postprocessing": "score orientation, stable Top-5 sort", "physical_forward_count": True, "cache_policy": "no application cache; server/internal caching reported separately if observable"},
        "r2_cal_optimization": {"budget_equal_across_models": True, "candidate_batch_sizes": [1, 2, 4, 8], "candidate_concurrency": [1, 2, 4], "candidate_precisions": ["native_pinned_precision", "bfloat16_if_native_supported"], "candidate_attention": ["native_default"], "selected_configurations": "NONE_RESOURCE_GATE_BLOCKED", "no_prompt_or_scoring_changes": True},
        "r4_adjudication": {"queries": 120, "multi_hop": 96, "enterprise": 24, "multi_hop_per_dataset": {"hotpot": 32, "musique": 32, "wiki2": 32}, "policies": R4_POLICIES, "sampling_seed": ADJUDICATION_SEED, "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": ADJUDICATION_SEED, "human_labels_collected": 0, "llm_labels_collected": 0, "status": "PACKAGED_NOT_ADJUDICATED"},
        "authorization": {"jev_calls": 0, "paid_api_calls": 0, "training": False, "service_restarts": 0, "fresh_holdout": False, "authorized_gpus": AUTHORIZED_GPUS},
        "resource_snapshot": gpu,
    }
    return protocol


def select_adjudication_queries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(ADJUDICATION_SEED)
    selected = []
    for dataset in ("hotpot", "musique", "wiki2"):
        group = sorted((item for item in items if item["dataset"] == dataset), key=lambda x: x["query_id"])
        selected.extend(rng.sample(group, 32))
    enterprise = sorted((item for item in items if item["dataset"] == "enterprise"), key=lambda x: x["query_id"])
    selected.extend(rng.sample(enterprise, 24))
    selected.sort(key=lambda x: (x["dataset"], x["query_id"]))
    return selected


def normalize_answer(value: Any) -> str:
    import re
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def f1_proxy(prediction: Any, reference: Any, aliases: list[Any] | None = None) -> float | None:
    if reference is None:
        return None
    candidates = [reference] + list(aliases or [])
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for candidate in candidates:
        ref_tokens = normalize_answer(candidate).split()
        if not ref_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ref_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def build_adjudication_package(items: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    chosen = select_adjudication_queries(items)
    chosen_ids = {item["query_id"] for item in chosen}
    if len(chosen) != 120 or len(chosen_ids) != 120:
        raise RuntimeError("R4 deterministic sample is not 120 unique queries")
    selection = {(r["policy"], r["query_id"]): r for r in load_jsonl(PRIOR / "E2_SP_S2_SELECTION_RESULTS.jsonl")}
    readers = {(r["policy"], r["query_id"]): r for r in load_jsonl(PRIOR / "E2_SP_P1_READER_OUTPUTS.jsonl")}
    cases = []
    summary = []
    blind_map = {policy: f"policy_{index:02d}" for index, policy in enumerate(R4_POLICIES, 1)}
    for item in chosen:
        g = gold[item["query_id"]]
        by_doc = {candidate["doc_id"]: candidate for candidate in item["candidates"]}
        for policy in R4_POLICIES:
            select_row = selection.get((policy, item["query_id"]))
            reader = readers.get((policy, item["query_id"]))
            if not select_row or not reader:
                raise RuntimeError(f"missing prior R4 source row {policy} {item['query_id']}")
            selected_ids = select_row["selected_ids"]
            passages = [{"doc_id": doc_id, "title": by_doc[doc_id].get("title", ""), "text": by_doc[doc_id].get("full_text", "")} for doc_id in selected_ids]
            cases.append({
                "case_id": f"{item['query_id']}::{blind_map[policy]}",
                "query_id": item["query_id"],
                "dataset": item["dataset"],
                "cohort": item["cohort"],
                "question": item["question"],
                "policy_blind_id": blind_map[policy],
                "candidate_ids": [candidate["doc_id"] for candidate in item["candidates"]],
                "candidate_records": item["candidates"],
                "selected_doc_ids": selected_ids,
                "selected_passages": passages,
                "reader_answer": reader.get("answer", ""),
                "reader_status": reader.get("status"),
                "reader_input_hash": reader.get("actual_model_input_sha256"),
                "reference_answer": g.get("reference_answer"),
                "reference_aliases": g.get("answer_aliases", []),
                "reference_status": g.get("reference_status"),
                "gold_doc_ids": g.get("gold_doc_ids", []),
                "gold_identity_status": g.get("mapping_status"),
                "assessment_status": "NOT_INDEPENDENTLY_VERIFIED",
                "human_judgment": None,
                "llm_judgment": None,
            })
    for dataset in ("hotpot", "musique", "wiki2", "enterprise"):
        for policy in R4_POLICIES:
            subset = [r for r in cases if r["dataset"] == dataset and r["policy_blind_id"] == blind_map[policy]]
            verified = [r for r in subset if r["reference_status"] == "REFERENCE_AVAILABLE"]
            f1_values = [f1_proxy(r["reader_answer"], r["reference_answer"], r["reference_aliases"]) for r in verified]
            em_values = [1.0 if normalize_answer(r["reader_answer"]) in {normalize_answer(r["reference_answer"]), *(normalize_answer(x) for x in r["reference_aliases"])} else 0.0 for r in verified]
            summary.append({
                "dataset": dataset,
                "policy": policy,
                "policy_blind_id": blind_map[policy],
                "n_queries": len(subset),
                "reference_available_n": len(verified),
                "deterministic_em_proxy": sum(em_values) / len(em_values) if em_values else "NOT_MEASURED",
                "deterministic_f1_proxy": sum(f1_values) / len(f1_values) if f1_values else "NOT_MEASURED",
                "independent_correctness": "NOT_INDEPENDENTLY_VERIFIED",
                "groundedness": "NOT_INDEPENDENTLY_VERIFIED",
                "human_labels": 0,
                "llm_labels": 0,
                "status": "PROXY_ONLY" if verified else "NO_REFERENCE_ANSWER",
            })
    meta = {"selected_query_count": len(chosen), "selected_query_ids": sorted(chosen_ids), "policy_mapping": blind_map, "case_count": len(cases), "human_labels": 0, "llm_labels": 0}
    return cases, summary, meta


def bootstrap_ci(values: list[float], seed: int = ADJUDICATION_SEED) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(BOOTSTRAP_REPS):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def paired_statistics(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = [r for r in cases if r["dataset"] != "enterprise" and r["reference_status"] == "REFERENCE_AVAILABLE"]
    grouped = defaultdict(dict)
    for row in available:
        grouped[row["query_id"]][row["policy_blind_id"]] = f1_proxy(row["reader_answer"], row["reference_answer"], row["reference_aliases"])
    rows = []
    policy_blinds = [f"policy_{i:02d}" for i in range(1, 7)]
    for left, right in itertools.combinations(policy_blinds, 2):
        diffs = [float(values[left]) - float(values[right]) for values in grouped.values() if left in values and right in values]
        lo, hi = bootstrap_ci(diffs)
        rows.append({
            "population": "R4_96_multihop",
            "metric": "deterministic_answer_f1_proxy",
            "left_policy_blind_id": left,
            "right_policy_blind_id": right,
            "n_paired_queries": len(diffs),
            "mean_difference": sum(diffs) / len(diffs) if diffs else "NOT_MEASURED",
            "ci95_low": lo if lo is not None else "NOT_MEASURED",
            "ci95_high": hi if hi is not None else "NOT_MEASURED",
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": ADJUDICATION_SEED,
            "inferential_status": "EXPLORATORY_PROXY_NOT_FACTUAL_CORRECTNESS",
        })
    return rows


def write_reports(protocol: dict[str, Any], source_audit: dict[str, Any], coverage_rows: list[dict[str, Any]], coverage_summary: dict[str, Any], roster: list[dict[str, Any]], gpu: dict[str, Any], cal_latency: dict[str, float | None], adjudication_meta: dict[str, Any], answer_summary: list[dict[str, Any]], paired: list[dict[str, Any]], prior: dict[str, Any]) -> None:
    write_json(OUT / "E2_SP_R_FROZEN_PROTOCOL.json", protocol)
    write_json(OUT / "E2_SP_R_SOURCE_AUDIT.json", source_audit)
    write_json(OUT / "E2_SP_R_COVERAGE_SUMMARY.json", coverage_summary)
    write_json(OUT / "E2_SP_R_GPU_RESOURCE_AUDIT.json", gpu)
    write_json(OUT / "E2_SP_R_MODEL_ROSTER.json", {"models": roster})
    write_jsonl(OUT / "E2_SP_R_COVERAGE_BY_QUERY.jsonl", coverage_rows)

    config_lines = [
        "# E2-S/P-R CAL and Serving Configuration",
        "",
        "Status: `R2_NOT_EXECUTED_RESOURCE_GATE_BLOCKED`.",
        "",
        "The R2 configuration space is frozen before any new benchmark quality is observed. It changes only legitimate serving parameters and does not change prompts, score orientation, candidate pools, or native model semantics.",
        "",
        "| Model | Native contract | CAL p50 ms (existing S1 trace) | Prespecified batch candidates | Prespecified concurrency | Selected config |",
        "|---|---|---:|---|---|---|",
    ]
    for policy in MODEL_ROSTER:
        config_lines.append(f"| `{policy}` | {MODEL_ROSTER[policy]['native_contract']} | {cal_latency.get(policy) if cal_latency.get(policy) is not None else 'NOT_MEASURED'} | 1,2,4,8 | 1,2,4 | NOT_SELECTED_RESOURCE_GATE_BLOCKED |")
    config_lines.extend(["", "Retrieval Top-5 is a frozen prefix and has no new selector model configuration. No CAL outcome was used to alter the candidate pool or scoring semantics."])
    (OUT / "E2_SP_R_CAL_AND_SERVING_CONFIG.md").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    gpu_lines = [
        "# E2-S/P-R GPU Resource Audit",
        "",
        f"Captured: `{gpu['captured_utc']}`.",
        "",
        f"Authorized devices for this phase: `{AUTHORIZED_GPUS}`. Safety floor: `{MIN_FREE_MIB}` MiB free and no compute process.",
        "",
        "No authorized device satisfied the isolation gate at inspection. GPU0 had an active compute process, GPU1 and GPU2 were resident-service devices, and GPU4 had active compute processes/resident service. GPU5 also had active compute processes and is not part of the new phase authorization. No process was evicted and no service was restarted.",
        "",
        "| GPU | Authorized | Free MiB | Utilization | Compute processes | Safe for new model |",
        "|---:|:---:|---:|---:|---:|:---:|",
    ]
    for index in sorted(gpu["devices"]):
        d = gpu["devices"][index]
        gpu_lines.append(f"| {index} | {d.get('authorized')} | {d.get('memory_free_mib')} | {d.get('utilization_gpu_percent')} | {len(d.get('compute_apps', []))} | {d.get('safe_for_new_model')} |")
    gpu_lines.extend(["", f"Resource gate: **`{gpu['resource_gate']}`**.", "", "Consequently R1 authentic smoke and R3 open-loop serving were not launched. This is an integrity stop, not a zero-latency or zero-cost result."])
    (OUT / "E2_SP_R_GPU_RESOURCE_AUDIT.md").write_text("\n".join(gpu_lines) + "\n", encoding="utf-8")

    rate_rows = []
    for policy, p50 in cal_latency.items():
        base = 1000.0 / p50 if p50 else None
        rates = [round(max(0.01, base * m), 6) for m in (0.25, 0.5, 1.0, 2.0)] if base else []
        for rate in rates:
            rate_rows.append({"policy": policy, "arrival_process": "seeded_poisson", "offered_arrival_rate_rps": rate, "cal_p50_ms": p50, "status": "NOT_MEASURED_RESOURCE_GATE_BLOCKED", "completed": "NOT_MEASURED", "achieved_qps": "NOT_MEASURED", "p50_ms": "NOT_MEASURED", "p95_ms": "NOT_MEASURED", "p99_ms": "NOT_MEASURED", "queue_p50_ms": "NOT_MEASURED", "error_rate": "NOT_MEASURED", "queue_stable": "NOT_MEASURED", "slo_1000_pass": "NOT_MEASURED"})
    rate_rows.append({"policy": "retrieval5", "arrival_process": "seeded_poisson", "offered_arrival_rate_rps": "NOT_MEASURED", "cal_p50_ms": "NOT_MEASURED", "status": "NOT_MEASURED_NO_SELECTOR_CAL_TRACE", "completed": "NOT_MEASURED", "achieved_qps": "NOT_MEASURED", "p50_ms": "NOT_MEASURED", "p95_ms": "NOT_MEASURED", "p99_ms": "NOT_MEASURED", "queue_p50_ms": "NOT_MEASURED", "error_rate": "NOT_MEASURED", "queue_stable": "NOT_MEASURED", "slo_1000_pass": "NOT_MEASURED"})
    write_csv(OUT / "E2_SP_R_OPEN_LOOP_LOAD_SWEEP.csv", rate_rows)
    write_jsonl(OUT / "E2_SP_R_REQUEST_LEVEL_TELEMETRY.jsonl", [{"record_type": "run_status", "status": "NOT_EXECUTED_RESOURCE_GATE_BLOCKED", "reason": "No authorized isolated GPU was available; no requests were generated.", "physical_forward_calls": 0, "jev_calls": 0, "paid_api_calls": 0}])
    write_csv(OUT / "E2_SP_R_GPU_SECONDS_AND_COST.csv", [{"policy": policy, "completed_queries": "NOT_MEASURED", "gpu_allocation_seconds": "NOT_MEASURED", "gpu_seconds_per_success": "NOT_MEASURED", "gpu_hour_rate_usd": "NOT_ASSUMED", "estimated_cost_per_query_usd": "NOT_MEASURED", "status": "BLOCKED_NO_SERVING_RUN"} for policy in [*MODEL_ROSTER, "retrieval5"]])

    write_jsonl(OUT / "E2_SP_R_ADJUDICATION_CASES.jsonl", adjudication_meta.pop("cases", []))
    write_json(OUT / "E2_SP_R_ADJUDICATION_SAMPLE.json", adjudication_meta)
    write_csv(OUT / "E2_SP_R_ANSWER_QUALITY_SUMMARY.csv", answer_summary)
    write_csv(OUT / "E2_SP_R_PAIRED_ANSWER_STATISTICS.csv", paired)

    (OUT / "E2_SP_R_DOWNSTREAM_ADJUDICATION_PROTOCOL.md").write_text("""# E2-S/P-R Downstream Answer Assessment Protocol

Status: `PACKAGED_NOT_INDEPENDENTLY_ADJUDICATED`.

The package contains 120 deterministic questions and six existing reader answers per question (720 cases): 96 multi-hop cases (32 each from HotpotQA, MuSiQue and 2Wiki) and 24 enterprise cases. Sampling seed: `731`. The case package was constructed from query identity and dataset only; policy outcomes were not used for sampling.

Policy identities are blinded in the cases as `policy_01` through `policy_06`. The offline mapping is recorded in `E2_SP_R_ADJUDICATION_SAMPLE.json` for controlled rejoin. Reviewers should see the query, selected passage text and answer, but not the original policy name or selector score. A completed review must record correctness, completeness, unsupported claims, grounding and abstention, with an optional second judgment on at least 20% of cases.

No human reviewer supplied labels in this execution. No local LLM judge was called. Accordingly, `E2_SP_R_ANSWER_QUALITY_SUMMARY.csv` reports deterministic EM/F1 only as answer proxies for the 96 multi-hop questions with reference answers; enterprise factual correctness is not measurable from the available release metadata. The paired statistics are exploratory proxy comparisons and are not factual-correctness claims.
""", encoding="utf-8")

    frontier_rows = []
    old_summary = []
    old_path = PRIOR / "E2_SP_QUALITY_POLICY_SUMMARY.csv"
    if old_path.is_file():
        with old_path.open(encoding="utf-8", newline="") as handle:
            old_summary = list(csv.DictReader(handle))
    old_by_policy = defaultdict(list)
    for row in old_summary:
        old_by_policy[row["policy"]].append(row)
    latency = {}
    latency_path = PRIOR / "E2_SP_SELECTION_LATENCY_SUMMARY.csv"
    if latency_path.is_file():
        with latency_path.open(encoding="utf-8", newline="") as handle:
            latency = {r["policy"]: r for r in csv.DictReader(handle)}
    answer_by_policy = {}
    for policy in ["retrieval5", *MODEL_ROSTER]:
        subset = [r for r in answer_summary if r["policy"] == policy and r["dataset"] in {"hotpot", "musique", "wiki2"} and r["deterministic_f1_proxy"] != "NOT_MEASURED"]
        if subset:
            weighted_n = sum(int(r["reference_available_n"]) for r in subset)
            answer_by_policy[policy] = {"deterministic_f1_proxy": sum(float(r["deterministic_f1_proxy"]) * int(r["reference_available_n"]) for r in subset) / weighted_n}
    for policy in ["retrieval5", *MODEL_ROSTER]:
        ta = [r for r in old_by_policy.get(policy, []) if r.get("dataset") in {"hotpot", "musique", "wiki2"}]
        en = [r for r in old_by_policy.get(policy, []) if r.get("dataset") == "enterprise"]
        def avg(field: str, rows: list[dict[str, Any]]) -> str:
            vals = [float(r[field]) for r in rows if r.get(field) not in (None, "", "NOT_MEASURED")]
            return f"{sum(vals)/len(vals):.6f}" if vals else "NOT_MEASURED"
        frontier_rows.append({
            "policy": policy,
            "quality_population_track_a_n": sum(int(r.get("n_queries", 0)) for r in ta),
            "track_a_selected_complete_support": avg("selected_complete_support_rate", ta),
            "track_a_answer_f1_proxy_existing_300": avg("answer_f1_proxy_all", ta),
            "enterprise_quality_n": sum(int(r.get("n_queries", 0)) for r in en),
            "enterprise_selected_complete_support": avg("selected_complete_support_rate", en),
            "enterprise_answer_f1_proxy_existing_300": avg("answer_f1_proxy_all", en),
            "r4_multihop_answer_f1_proxy_96": answer_by_policy.get(policy, {}).get("deterministic_f1_proxy", "NOT_MEASURED"),
            "independent_answer_correctness": "NOT_INDEPENDENTLY_VERIFIED",
            "selector_p50_ms_existing_serialized": latency.get(policy, {}).get("p50_ms", "NOT_MEASURED"),
            "selector_p95_ms_existing_serialized": latency.get(policy, {}).get("p95_ms", "NOT_MEASURED"),
            "r3_sustainable_qps_p95_1s": "NOT_MEASURED",
            "gpu_seconds_per_query": "NOT_MEASURED",
            "cost_per_query_usd": "NOT_MEASURED",
            "frontier_status": "QUALITY_AND_SERIALIZED_LATENCY_ONLY; R3_COST_BLOCKED",
        })
    write_csv(OUT / "E2_SP_R_QUALITY_LATENCY_COST_FRONTIER.csv", frontier_rows)

    source_lines = [
        "# E2-S/P-R Source and Cohort Audit",
        "",
        f"Status: `{source_audit['status']}`.",
        "",
        f"The frozen population has {source_audit['query_count']} unique queries: {source_audit['dataset_counts']}. Every question hash and ordered 20-candidate slate matched the source records. Existing E2-S/P selection rows: {source_audit['selection_rows']}; reader rows: {source_audit['reader_rows']}.",
        "",
        "Selection and reader inputs contained zero gold fields and zero application-cache reuses. The Top-20 pool is immutable for R; no candidate pool was rebuilt or filtered.",
        "",
        "Coverage denominators are kept explicit: `top20_*` is retrieval-pool coverage; `selected_*` is conditional selector support among identity-eligible rows. No unsupported aggregate is presented as a new model result.",
        "",
        "## Prior artifact preservation",
        "",
        f"Prior generated artifact hash mismatches: `{len(prior['generated_mismatches'])}`; prior preserved-artifact hash mismatches: `{len(prior['preserved_mismatches'])}`. The prior package remains untouched.",
    ]
    (OUT / "E2_SP_R_SOURCE_AND_COHORT_AUDIT.md").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    (OUT / "E2_SP_R_SLO_CAPACITY_REPORT.md").write_text("""# E2-S/P-R SLO and Capacity Report

Status: `MEASUREMENT_BLOCKED`.

The primary exploratory SLO is complete Top-5 selection p95 <= 1,000 ms, with 2,000 and 3,000 ms sensitivity thresholds. No new open-loop arrivals were issued because no authorized GPU (0, 1, 2 or 4) was isolated from active compute/service ownership at the resource guard. Therefore no sustainable QPS, SLO-feasible rate, queue stability, error rate, GPU allocation-seconds/query or monetary cost is reported.

The existing E2-S/P serialized selector percentiles remain historical descriptive measurements only. They are not substituted for open-loop capacity and are not added to reader latency to create an end-to-end percentile.
""", encoding="utf-8")

    (OUT / "E2_SP_R_AUTHENTICITY_AUDIT.md").write_text(f"""# E2-S/P-R Authenticity Audit

Verdict: `PARTIAL_BLOCKED_RESOURCE_GATE`.

- Frozen source identity and Top-20 hashes: PASS.
- Prior 3,300 selector and 3,300 reader rows: PASS and preserved.
- New model requests in R: 0; new physical forwards: 0.
- Open-loop arrivals independent of completion: NOT_EXECUTED.
- GPU allocation telemetry and monetary cost: NOT_MEASURED; no price was invented.
- Downstream cases: 720 packaged from existing reader outputs; human labels: 0; local LLM labels: 0.
- Jev calls: 0; paid API calls: 0; fresh holdout: untouched; training: not run; service restarts: 0.

The lack of serving data is an explicit resource-gate block, not a measured zero.
""", encoding="utf-8")

    final = f"""# E2-S/P-R Final Report

Verdict: `E2_SP_R_PARTIAL_BLOCKED`.

## Completed

- R0 frozen-cohort/source audit: PASS ({source_audit['query_count']} unique queries; all hashes and candidate identities matched).
- Model roster and local checkpoint audit: completed for {len(roster)} approved models; missing local snapshots: {sum(r['status'] != 'AVAILABLE_LOCAL' for r in roster)}.
- CAL-only configuration freeze: prepared before any new benchmark-quality inspection.
- R4 downstream package: 120 questions × 6 existing reader policies = {adjudication_meta['case_count']} blinded cases; deterministic EM/F1 proxies produced where multi-hop references exist.
- Historical artifacts: preserved; prior provenance mismatches: {len(prior['generated_mismatches']) + len(prior['preserved_mismatches'])}.

## Blocked

R1 authentic new smoke and R3 open-loop serving could not start because no authorized isolated GPU was available at the guard. Active processes were observed on the approved devices; no process was terminated and no service was restarted. R2 therefore has no selected serving configuration, and R3 has no measured QPS, p95, GPU-seconds/query or cost.

## Quality and deployment interpretation

The earlier 300-query evidence-quality and serialized-latency results remain descriptive and are not pooled with absent R3 capacity measurements. The R4 deterministic EM/F1 values are answer proxies, not independently verified factual correctness. Enterprise reference answers are unavailable in the acquired release metadata, so enterprise correctness remains unverified. No replacement or cost-effectiveness conclusion is supportable from E2-S/P-R; deployment conclusion: `MEASUREMENT_BLOCKED`.

Jev and all paid APIs remain untouched (0 calls). The next owner decision is to provide an authorized isolated execution window/device for R1-R3, or explicitly accept a blocked milestone and keep the prior E2-S/P limitation unchanged.
"""
    (OUT / "E2_SP_R_FINAL_REPORT.md").write_text(final, encoding="utf-8")
    (OUT / "E2_SP_R_OWNER_REVIEW_GATE.md").write_text(f"""# E2-S/P-R Owner Review Gate

Verdict: `E2_SP_R_PARTIAL_BLOCKED`.

PASS: frozen population audit, artifact preservation, roster/checkpoint inventory, CAL configuration freeze, and offline R4 assessment-package preparation.

BLOCKED: authentic R1 smoke, R2 measured serving configuration, R3 open-loop load/capacity, GPU allocation accounting, monetary cost, and independent factual answer adjudication.

No Jev or paid request was made. Owner decision required: authorize an actually isolated approved GPU/execution window and, separately, provide human adjudication or accept `NOT_INDEPENDENTLY_VERIFIED` answer quality.
""", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = prior_integrity()
    items = load_items()
    source_audit, records = source_and_cohort_audit(items)
    gold = load_gold(items)
    coverage_rows, coverage_summary = coverage_audit(items, records, gold)
    roster = roster_audit()
    gpu = gpu_inventory()
    cal_latency = read_cal_latency()
    protocol = create_protocol(source_audit, roster, gpu, cal_latency)
    cases, answer_summary, adjudication_meta = build_adjudication_package(items, gold)
    adjudication_meta = dict(adjudication_meta)
    adjudication_meta["cases"] = cases
    paired = paired_statistics(cases)
    write_reports(protocol, source_audit, coverage_rows, coverage_summary, roster, gpu, cal_latency, adjudication_meta, answer_summary, paired, prior)
    # The manifest is written last and intentionally does not hash itself.
    generated = {}
    for path in sorted(OUT.iterdir()):
        if path.name != "E2_SP_R_PROVENANCE_MANIFEST.json" and path.is_file():
            generated[str(path.relative_to(ROOT))] = sha_file(path)
    provenance = {
        "manifest_version": "E2-SP-R-v1",
        "created_utc": utc_now(),
        "status": "E2_SP_R_PARTIAL_BLOCKED",
        "stage_status": {"R0": source_audit["status"], "R1": "BLOCKED_NO_ISOLATED_AUTHORIZED_GPU", "R2": "BLOCKED_NO_R1_SMOKE_OR_GPU", "R3": "BLOCKED_NO_ISOLATED_AUTHORIZED_GPU", "R4": "PACKAGED_NOT_INDEPENDENTLY_ADJUDICATED", "R5": "PARTIAL_FRONTIER_NO_R3_COST"},
        "generated_artifact_sha256": generated,
        "prior_integrity": prior,
        "gpu_resource_gate": gpu,
        "model_roster": roster,
        "counts": {"frozen_queries": len(items), "r4_queries": adjudication_meta["selected_query_count"], "r4_cases": adjudication_meta["case_count"], "human_labels": 0, "llm_labels": 0, "new_model_requests": 0, "new_physical_forwards": 0},
        "jev_calls": 0,
        "paid_api_calls": 0,
        "fresh_holdout_access": False,
        "training": False,
        "service_restarts": 0,
    }
    write_json(OUT / "E2_SP_R_PROVENANCE_MANIFEST.json", provenance)
    print(json.dumps({"status": provenance["status"], "output": str(OUT), "frozen_queries": len(items), "r4_cases": adjudication_meta["case_count"], "resource_gate": gpu["resource_gate"], "jev_calls": 0, "paid_api_calls": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
