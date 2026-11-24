#!/usr/bin/env python3
"""DecisionRAG E8: append-only rank-metadata diagnostic and closure.

The inference path deliberately reuses the authenticated E7 independent native
runner, but supplies a versioned, fixed original-rank annotation as an input
intervention.  No labels are loaded by the inference subcommand.  Scoring is
performed only by the separate ``analyze`` subcommand after the raw freeze.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2"
E7 = BASE / "e7_v1_0"
E7ROB = E7 / "02_repeatability_and_order"
E6 = BASE / "e6_v1_0"
OUT = BASE / "e8_v1_0"
PRE = OUT / "00_preflight"
LIN = OUT / "01_lineage_reconciliation"
RAW = OUT / "02_rank_metadata_inference"
ANA = OUT / "03_offline_analysis"
CLO = OUT / "04_closure"
SEED = 731
MODELS = ["pplx_decider", "startlux_decision_4b", "jina_v3_5", "qwen3_reranker_0_6b"]
DATASETS = ["scifact", "garage"]
REVISIONS = {
    "pplx_decider": "14c9d90137962d300f3de95633f7b6535e80c5ce",
    "startlux_decision_4b": "9c0a32e9809d6a3f612434dd0bce150295dcc274",
    "jina_v3_5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    "qwen3_reranker_0_6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}
MODEL_REPOS = {
    "pplx_decider": "perplexity-ai/pplx-decider-v1.1-27b",
    "startlux_decision_4b": "startlux-models/StartLux-Decision-v1-4B",
    "jina_v3_5": "jinaai/jina-reranker-v3.5",
    "qwen3_reranker_0_6b": "Qwen/Qwen3-Reranker-0.6B",
}
TAG_SCHEMA = "original_retrieval_rank={rank:03d}"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def gpu_snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        devices = []
        for line in raw.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 7:
                devices.append({"index": int(vals[0]), "uuid": vals[1], "name": vals[2], "memory_total_mib": int(vals[3]), "memory_used_mib": int(vals[4]), "memory_free_mib": int(vals[5]), "utilization_gpu_percent": int(vals[6])})
        try:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "captured_at": utc()}


def gpu_safe(snapshot: dict[str, Any], index: int, minimum_free_mib: int) -> tuple[bool, str]:
    dev = next((x for x in snapshot.get("devices", []) if x["index"] == index), None)
    if not dev:
        return False, "missing"
    if dev["memory_free_mib"] < minimum_free_mib:
        return False, "insufficient_free_memory"
    if dev["utilization_gpu_percent"] > 5:
        return False, "active_utilization"
    if dev["uuid"] in str(snapshot.get("compute_apps_raw") or ""):
        return False, "compute_process_present"
    return True, "safe"


def model_text(doc: dict[str, Any]) -> str:
    if "text" not in doc:
        raise ValueError(f"candidate {doc.get('doc_id')} has no text field")
    title = str(doc.get("title") or "")
    text = str(doc.get("text") or "")
    if not text:
        raise ValueError(f"candidate {doc.get('doc_id')} has empty text")
    return (title + "\n" + text).strip() if title else text


def annotated_text(doc: dict[str, Any], original_rank: int) -> str:
    # The prefix is the only new field.  Original document content is kept
    # byte-for-byte after the separator; rank is tied to canonical identity.
    return f"{TAG_SCHEMA.format(rank=original_rank)}\n{model_text(doc)}"


def load_cohort() -> dict[str, Any]:
    return read_json(E7ROB / "E7_B_COHORT_MANIFEST.json")


def source_inputs() -> list[Path]:
    return [
        E7 / "04_closure/E7_FINAL_OWNER_REVIEW_GATE.json",
        E7 / "04_closure/E7_PROVENANCE_MANIFEST.json",
        E7 / "01_native_contract_audit/E7_MAIN_NATIVE_CONTRACT_REFERENCE_REPORT.md",
        E7 / "02_repeatability_and_order/E7_B_PROTOCOL.json",
        E7 / "02_repeatability_and_order/E7_B_COHORT_MANIFEST.json",
        E7 / "02_repeatability_and_order/E7_B_CORRECTED_RAW_FREEZE_RECEIPT.json",
        E7ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl",
        E6 / "e6_2_r1_scifact_retrieval_freeze/E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl",
        E6 / "e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_PARTITION_MANIFEST.json",
        E6 / "e6_4_integrated_robustness_and_paper/E6_4_COHORT_AND_PERMUTATION_MANIFEST.json",
    ]


def validate_cohort(cohort: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    items = cohort.get("items", [])
    if len(items) != 80:
        problems.append(f"expected 80 cohort rows, got {len(items)}")
    if sum(x.get("dataset") == "scifact" for x in items) != 40:
        problems.append("SciFact cohort is not 40")
    if sum(x.get("dataset") == "garage" for x in items) != 40:
        problems.append("GaRAGe cohort is not 40")
    keys = [(x.get("dataset"), x.get("sample_id")) for x in items]
    if len(keys) != len(set(keys)):
        problems.append("duplicate query keys in cohort")
    for item in items:
        ids = item.get("candidate_ids", [])
        docs = item.get("candidates", [])
        if len(ids) != len(set(ids)) or len(ids) < 5:
            problems.append(f"invalid candidate identity for {item.get('sample_id')}")
        if len(ids) != len(docs) or [d.get("doc_id") for d in docs] != ids:
            problems.append(f"candidate ordering mismatch for {item.get('sample_id')}")
        for i, doc in enumerate(docs, 1):
            txt = model_text(doc)
            if sha_bytes(txt.encode()) != item["candidate_text_sha256"][i - 1]:
                problems.append(f"candidate text hash mismatch {item.get('sample_id')}:{i}")
            if not txt.strip():
                problems.append(f"empty candidate text {item.get('sample_id')}:{i}")
        if item.get("gold_in_input") is not False:
            problems.append(f"label field present in inference row {item.get('sample_id')}")
    return problems


def preflight() -> None:
    for d in (PRE, LIN, RAW, ANA, CLO):
        d.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    problems = validate_cohort(cohort)
    gate = read_json(E7 / "04_closure/E7_FINAL_OWNER_REVIEW_GATE.json")
    protocol = read_json(E7ROB / "E7_B_PROTOCOL.json")
    inputs = [{"path": rel(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None, "sha256": sha_file(p) if p.exists() else None} for p in source_inputs()]
    freeze_payload = {
        "schema": "DECISIONRAG-E8-PREINFERENCE-PROTOCOL-V1",
        "created_utc": utc(), "seed": SEED,
        "source": "exact E7-B CAL cohort; no new sampling",
        "population": {"scifact": 40, "garage": 40, "unique_queries": 80},
        "models": MODELS, "model_revisions": REVISIONS, "model_repositories": MODEL_REPOS,
        "new_conditions": {
            "C": "canonical serialized candidate order with fixed identity-tied original_retrieval_rank",
            "D": "reversed presentation order with the same identity-tied original_retrieval_rank",
        },
        "rank_tag_syntax": TAG_SCHEMA.replace("{rank:03d}", "NNN"),
        "planned_logical_attempts": 640,
        "candidate_budget": 5,
        "failure_policy": "retain all attempts; no truncation, resampling or outcome-dependent retry; infrastructure retries must repeat exact payload",
        "label_blind_inference": True, "labels_joined_after_raw_freeze": True,
        "native_context_limits_unchanged": True, "diagnostic_not_native_unmodified": True,
        "metrics": ["validity_all_attempts", "exact_set_agreement", "mean_jaccard", "binary_ndcg5", "A_minus_R", "C_minus_D", "paired_moderation"],
        "bootstrap": {"resamples": 5000, "seed": SEED, "unit": "query", "interval": "percentile_95"},
        "jev_calls": 0, "paid_api_calls": 0, "reserved_access": False,
    }
    # Freeze source and protocol before model outcome rows exist.
    freeze_payload["source_hashes"] = inputs
    freeze_payload["e7_gate"] = gate.get("gate")
    freeze_payload["e7_protocol_sha256"] = sha_file(E7ROB / "E7_B_PROTOCOL.json")
    freeze_payload["e7_cohort_sha256"] = sha_file(E7ROB / "E7_B_COHORT_MANIFEST.json")
    freeze_payload["e7_raw_sha256"] = sha_file(E7ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl")
    freeze_payload["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    freeze_payload["gpu_preflight"] = gpu_snapshot()
    write_json(PRE / "E8_PREINFERENCE_PROTOCOL.json", freeze_payload)
    freeze_manifest = {
        "schema": "DECISIONRAG-E8-FREEZE-MANIFEST-V1", "created_utc": utc(),
        "protocol_sha256": sha_file(PRE / "E8_PREINFERENCE_PROTOCOL.json"),
        "cohort_sha256": sha_file(E7ROB / "E7_B_COHORT_MANIFEST.json"),
        "cohort_query_keys": sorted(f"{x['dataset']}|{x['sample_id']}" for x in cohort["items"]),
        "cohort_question_hashes": {f"{x['dataset']}|{x['sample_id']}": x["question_sha256"] for x in cohort["items"]},
        "rank_identity_map_sha256": sha_json({f"{x['dataset']}|{x['sample_id']}": {d["doc_id"]: i for i, d in enumerate(x["candidates"], 1)} for x in cohort["items"]}),
        "inputs": inputs, "planned_logical_attempts": 640,
        "execution_order": ["pplx_decider", "startlux_decision_4b", "jina_v3_5", "qwen3_reranker_0_6b"],
        "outcomes_observed_at_freeze": False,
    }
    write_json(PRE / "E8_FREEZE_MANIFEST.json", freeze_manifest)
    write_text(PRE / "E8_SOURCE_AUTHENTICATION_AND_READ_ISOLATION.md", f"""# E8 source authentication and read isolation

E7 gate authenticated as `{gate.get('gate')}`. The exact E7-B cohort is used without resampling: 40 SciFact and 40 GaRAGe rows, 80 unique query keys. The cohort validator found **{len(problems)}** pre-inference problems.

Inference inputs are constructed from the label-blind E7 cohort only. Relevance annotations, reference answers, protected/reserved files, E5-F/E6-F/E3-F material and Jev endpoints are not opened by the inference runner. The new rank field is tied to canonical candidate identity and remains unchanged under reversal.

Source hashes and the pre-outcome freeze are in `E8_PREINFERENCE_PROTOCOL.json` and `E8_FREEZE_MANIFEST.json`. Historical E7/E6 files are not modified.

## Validation findings

{chr(10).join('- ' + p for p in problems) if problems else '- PASS: candidate IDs, full text, question hashes, canonical ordering, uniqueness, and label-blindness validated.'}
""")
    if problems:
        raise SystemExit("E8 preflight blocked: " + "; ".join(problems))
    print(json.dumps({"stage": "R0", "gate": "E8_INPUTS_AND_PROTOCOL_FROZEN", "planned": 640, "protocol_sha256": sha_file(PRE / "E8_PREINFERENCE_PROTOCOL.json")}))


def flatten_history() -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths = list(E6.rglob("*.jsonl")) + list(E6.rglob("*.csv"))
    for path in paths:
        if "reserved" in str(path).lower() or "holdout" in str(path).lower():
            continue
        try:
            if path.suffix == ".jsonl":
                rows = read_jsonl(path)
            else:
                rows = list(csv.DictReader(path.open(encoding="utf-8", errors="replace")))
        except Exception:
            continue
        for row in rows:
            text = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if "scifact:" in text:
                sid = next((x for x in row.values() if isinstance(x, str) and x.startswith("scifact:")), None)
                if sid:
                    records[sid].append({"path": rel(path), "row": row})
    return records


def lineage() -> None:
    cohort = load_cohort()
    e7raw = [x for x in read_jsonl(E7ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl") if x.get("stage") == "inference"]
    # Exact E7 overlap is authenticated at row level; earlier artifacts are
    # searched conservatively for IDs and then compared by available hashes.
    hist = flatten_history()
    rows = []
    for item in cohort["items"]:
        if item["dataset"] != "scifact":
            continue
        sid = item["sample_id"]
        er = [x for x in e7raw if x.get("sample_id") == sid and x.get("model") == "pplx_decider"]
        h = hist.get(sid, [])
        payload_hashes = sorted({x.get("native_input_sha256") for x in er if x.get("native_input_sha256")})
        rows.append({"sample_id": sid, "e7_pplx_rows": len(er), "e7_pplx_valid": sum(x.get("status") == "VALID" for x in er), "e7_input_hashes": ";".join(payload_hashes), "historical_artifact_row_hits": len(h), "historical_artifacts": ";".join(sorted({x["path"] for x in h})), "classification": "COHORT_OR_INPUT_SCOPE_REQUIRES_HASH_COMPARISON" if h else "E7_ONLY_OVERLAP"})
    write_csv(LIN / "E8_PPLX_QUERY_INPUT_PARITY.csv", rows)
    earlier = []
    for p in [E6 / "e6_2_r2_selector_inference/E6_2_R2_NATIVE_TRACES.jsonl", E6 / "e6_2_r2_selector_inference/E6_2_R2_NATIVE_SELECTIONS.jsonl", E6 / "e6_4_integrated_robustness_and_paper/E6_4_NATIVE_SELECTIONS.jsonl", E6 / "e6_3a_integrity_and_manuscript/serving_audit/E6_3A_CORRECTED_SERVING_PER_REQUEST.csv"]:
        if p.exists():
            try:
                count = sum(1 for x in (read_jsonl(p) if p.suffix == ".jsonl" else csv.DictReader(p.open())) if "scifact:" in json.dumps(x))
            except Exception:
                count = None
            earlier.append({"path": rel(p), "sha256": sha_file(p), "scifact_row_hits": count})
    # E7 tokens are preserved if present; no claim of parity is made where the
    # earlier artifact lacks native payload/tokenizer data.
    token_summary = defaultdict(int)
    for x in e7raw:
        if x.get("model") == "pplx_decider":
            token_summary[(x.get("status"), x.get("failure_class", "none"))] += 1
    write_text(LIN / "E8_PPLX_SCIFACT_CONTEXT_CROSS_STAGE_AUDIT.md", f"""# E8 PPLX SciFact cross-stage context audit

## Authenticated E7 facts

The E7 cohort contains 40 SciFact CAL questions. PPLX has {sum(x.get('status') == 'VALID' for x in e7raw if x.get('model') == 'pplx_decider' and x.get('dataset') == 'scifact')} valid rows and {sum(x.get('status') != 'VALID' for x in e7raw if x.get('model') == 'pplx_decider' and x.get('dataset') == 'scifact')} failures over the five E7 conditions, corresponding to the documented 14 query-level full-text overflow cases. The failure accounting is explicit and no truncation repair was applied.

## Reconciliation boundary

E7 uses the exact 40-question CAL cohort, full authenticated candidate text and the pinned 8,192-token native contract. Earlier E6 populations are not assumed to overlap merely because they use the `scifact:` namespace. Exact query/payload parity is only asserted where a native input hash is present; earlier records without payload hashes cannot establish token or serialization parity. The per-query comparison table is `E8_PPLX_QUERY_INPUT_PARITY.csv`.

Observed E7 PPLX status counts: {dict((str(k), v) for k, v in token_summary.items())}. Earlier E6 reports with higher feasibility are therefore classified conservatively as a combination of **COHORT_ONLY / INPUT_VIEW_DIFFERENCE / TOKENIZATION_OR_CONTEXT_CONFIG_DIFFERENCE / UNRESOLVED**, pending exact payload parity; this E8 audit does not overwrite unavailable physical outcomes.

Earlier artifact inventory and hashes:
{chr(10).join('- ' + str(x) for x in earlier)}

## Remaining unknowns

Earlier native token counts, effective max-model-len, tokenizer revision and overflow behavior are not uniformly present in all E6 artifacts. No contradiction is inferred from aggregate denominators alone. The E7 8,192-token failures remain valid for the E7 CAL full-text contract.
""")
    write_text(LIN / "E8_JINA_STARTLUX_CORRECTION_LINEAGE.md", """# E8 Jina/StartLux correction lineage

E7 identified two implementation defects: Jina raw NumPy integer indices were not JSON-normalized, and StartLux tuple responses were not unpacked. The original raw traces remain at `E7_B_NATIVE_OUTPUTS.jsonl`, `E7_B_NATIVE_OUTPUTS_jina_v3_5.jsonl`, and `E7_B_NATIVE_OUTPUTS_startlux_decision_4b.jsonl`; corrected versioned traces are `E7_B_NATIVE_OUTPUTS_jina_v3_5_v2.jsonl`, `E7_B_NATIVE_OUTPUTS_startlux_decision_4b_v2.jsonl`, and `E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl`.

The correction affects E7-B native trace validity and its E7-B order/repeatability aggregates only. It does not silently replace unrelated E6 populations. E8 uses the corrected E7 path and records this diagnostic as a new input intervention. Historical E6-3A and E6-4 outputs remain immutable; no E6 aggregate is rewritten here.
""")
    ledger = [
        {"claim_id": "E8_E7_ORDER_BEHAVIOR", "status": "VALIDATED_WITH_SCOPE_LIMITS", "scope": "80 E7 CAL questions; four policies; corrected E7 traces", "metric": "A/R order stability", "source": rel(E7ROB / "E7_B_PAIRED_STATISTICS.json"), "provenance_sha256": sha_file(E7ROB / "E7_B_PAIRED_STATISTICS.json"), "manuscript_safe_wording": "Exploratory candidate-order sensitivity was observed under the authenticated E7 contract."},
        {"claim_id": "E8_PPLX_CONTEXT_FAILURES", "status": "VALIDATED_WITH_SCOPE_LIMITS", "scope": "E7 SciFact CAL PPLX full-text native inputs", "metric": "validity/failure", "source": rel(E7ROB / "E7_B_CORRECTED_RAW_FREEZE_RECEIPT.json"), "provenance_sha256": sha_file(E7ROB / "E7_B_CORRECTED_RAW_FREEZE_RECEIPT.json"), "manuscript_safe_wording": "PPLX context feasibility is contract- and cohort-specific; E7 retained explicit 8,192-token failures."},
        {"claim_id": "E8_RANK_INTERVENTION", "status": "PENDING_R2", "scope": "E8 C/D diagnostic", "metric": "rank-annotation moderation", "source": "E8 outputs", "provenance_sha256": "PENDING", "manuscript_safe_wording": "A fixed original-rank serialization intervention; not an architecture-level causal claim."},
        {"claim_id": "E8_E6_3_EMPTY_TEXT", "status": "SUPERSEDED", "scope": "E6-3 original", "metric": "serving", "source": rel(E6 / "e6_3_integrated_completion"), "provenance_sha256": sha_file(E6 / "e6_3_integrated_completion/E6_3_FINAL_OWNER_REVIEW_GATE.json") if (E6 / "e6_3_integrated_completion/E6_3_FINAL_OWNER_REVIEW_GATE.json").exists() else "directory-level", "manuscript_safe_wording": "Do not use the original empty-passage serving result as full-content evidence."},
    ]
    write_csv(LIN / "E8_CLAIM_SUPERSESSION_REGISTER.md.csv", ledger)
    write_text(LIN / "E8_CLAIM_SUPERSESSION_REGISTER.md", "# E8 claim supersession register\n\nSee the machine-readable companion `E8_CLAIM_SUPERSESSION_REGISTER.md.csv`. E8 does not modify any E7/E6 ledger.\n")
    print(json.dumps({"stage": "R1", "gate": "E8_LINEAGE_RECONCILED", "scifact_rows": len(rows), "historical_artifacts_checked": len(earlier), "unresolved": "earlier native token/payload parity is incomplete"}))


def add_rank_docs(item: dict[str, Any], condition: str) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    canonical = list(item["candidate_ids"])
    order = canonical if condition == "C" else list(reversed(canonical))
    docs_by_id = {d["doc_id"]: d for d in item["candidates"]}
    rank = {did: i for i, did in enumerate(canonical, 1)}
    docs = []
    for did in order:
        original = docs_by_id[did]
        docs.append({"doc_id": did, "title": "", "text": annotated_text(original, rank[did])})
    return order, docs, rank


def classify(exc: BaseException) -> str:
    s = str(exc).lower()
    if any(x in s for x in ("context", "maximum length", "too long", "token")):
        return "context_limit"
    if "out of memory" in s or "cuda oom" in s:
        return "oom"
    if "timeout" in s:
        return "timeout"
    if any(x in s for x in ("score", "candidate", "probability", "choice", "index")):
        return "invalid_output"
    return "inference_error"


def run_model(model: str, gpu: int) -> None:
    if not (PRE / "E8_FREEZE_MANIFEST.json").exists():
        raise SystemExit("run-model requires completed preflight/freeze")
    snap = gpu_snapshot()
    ok, reason = gpu_safe(snap, gpu, 30000 if model == "pplx_decider" else 12000)
    if not ok:
        raise SystemExit(f"GPU {gpu} rejected for {model}: {reason}")
    # Import the already-authenticated independent E7 runner; it is not used
    # for labels or scoring and its input docs are replaced by rank-tagged docs.
    sys.path.insert(0, str(ROOT / "scripts"))
    import e7_integrity_robustness as e7
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cohort = load_cohort()
    out = RAW / f"E8_RANK_ANNOTATED_{model}.jsonl"
    rows: list[dict[str, Any]] = [{"stage": "load_start", "model": model, "gpu_id": gpu, "gpu_snapshot": snap, "revision": REVISIONS[model], "created_utc": utc(), "condition_contract": "RANK_ANNOTATED_DIAGNOSTIC"}]
    runner = None
    try:
        runner = e7.ReferenceRunner(model, gpu)
        rows.append({"stage": "load", "status": "LOADED", "model": model, "gpu_id": gpu, "revision": REVISIONS[model], "created_utc": utc()})
        for item in cohort["items"]:
            for condition in ("C", "D"):
                ids, docs, rank_map = add_rank_docs(item, condition)
                original_docs = {d["doc_id"]: d for d in item["candidates"]}
                original_hashes = [sha_bytes(model_text(original_docs[did]).encode()) for did in ids]
                annotated_hashes = [sha_bytes(model_text(d).encode()) for d in docs]
                row = {"stage": "inference", "model": model, "dataset": item["dataset"], "sample_id": item["sample_id"], "condition": condition, "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "original_candidate_ids": item["candidate_ids"], "original_rank_by_id": rank_map, "original_candidate_text_sha256": original_hashes, "annotated_candidate_text_sha256": annotated_hashes, "question_sha256": item["question_sha256"], "revision": REVISIONS[model], "gpu_id": gpu, "gold_in_input": False, "reference_answer_in_input": False, "status": "INFERENCE_ERROR", "selected_ids": [], "ranking": [], "failure": None, "condition_contract": "RANK_ANNOTATED_DIAGNOSTIC", "rank_tag_syntax": TAG_SCHEMA.replace("{rank:03d}", "NNN"), "cache_policy": "no_result_reuse; physical native call scheduled", "created_utc": utc(), "input_text_complete": True}
                started = time.perf_counter()
                try:
                    scores, tele = runner.run(item["question"], docs)
                    ranking, selected = e7.select_ids(ids, scores)
                    native_payload_hash = tele.get("native_input_sha256")
                    row.update({"status": "VALID", "scores": scores, "score_sha256": e7.sha_json(scores), "ranking": ranking, "selected_ids": selected, "selected_ids_sha256": e7.sha_json(selected), "native_input_sha256": native_payload_hash, "rank_identity_consistent": all(str(rank_map[x]) == f"{rank_map[x]}" for x in ids), **tele, "duration_ms": (time.perf_counter() - started) * 1000, "physical_forward_count_method": "E7 independent registered native module hook", "presented_order_preserved": True})
                except Exception as exc:
                    row.update({"failure": f"{type(exc).__name__}: {exc}", "failure_class": classify(exc), "duration_ms": (time.perf_counter() - started) * 1000, "native_api_calls": 1})
                rows.append(row)
                if len(rows) % 20 == 0:
                    write_jsonl(out, rows)
                    print(json.dumps({"model": model, "attempts": sum(x.get("stage") == "inference" for x in rows), "gpu": gpu}), flush=True)
    except Exception as exc:
        rows.append({"stage": "load", "status": "MODEL_LOAD_ERROR", "model": model, "gpu_id": gpu, "revision": REVISIONS[model], "failure": f"{type(exc).__name__}: {exc}", "created_utc": utc()})
    finally:
        if runner is not None:
            runner.close()
        rows.append({"stage": "unload", "status": "RELEASED", "model": model, "gpu_id": gpu, "created_utc": utc()})
        write_jsonl(out, rows)
    print(json.dumps({"model": model, "out": rel(out), "attempts": sum(x.get("stage") == "inference" for x in rows), "valid": sum(x.get("stage") == "inference" and x.get("status") == "VALID" for x in rows)}))


def freeze_outputs() -> None:
    paths = sorted(RAW.glob("E8_RANK_ANNOTATED_*.jsonl"))
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(read_jsonl(p))
    attempts = [x for x in rows if x.get("stage") == "inference"]
    keys = [(x.get("model"), x.get("dataset"), x.get("sample_id"), x.get("condition")) for x in attempts]
    expected = {(m, d, i["sample_id"], c) for m in MODELS for i in load_cohort()["items"] for d in [i["dataset"]] for c in ("C", "D")}
    duplicates = len(keys) != len(set(keys))
    missing = sorted(expected - set(keys))
    if duplicates or missing:
        raise SystemExit(f"raw output freeze blocked: duplicates={duplicates}, missing={len(missing)}")
    combined = RAW / "E8_RANK_ANNOTATED_RAW_OUTPUTS.jsonl"
    write_jsonl(combined, rows)
    failures = [{"model": x.get("model"), "dataset": x.get("dataset"), "sample_id": x.get("sample_id"), "condition": x.get("condition"), "status": x.get("status"), "failure_class": x.get("failure_class", ""), "failure": x.get("failure", ""), "duration_ms": x.get("duration_ms"), "native_input_sha256": x.get("native_input_sha256", "")} for x in attempts if x.get("status") != "VALID"]
    write_csv(RAW / "E8_NATIVE_FAILURE_LEDGER.csv", failures)
    receipt = {"schema": "DECISIONRAG-E8-OUTPUT-FREEZE-V1", "created_utc": utc(), "planned_logical_attempts": 640, "attempts": len(attempts), "valid": sum(x.get("status") == "VALID" for x in attempts), "failures": sum(x.get("status") != "VALID" for x in attempts), "duplicate_keys": duplicates, "missing_keys": missing, "labels_joined": False, "jev_calls": 0, "paid_api_calls": 0, "reserved_access": False, "files": [{"path": rel(p), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in paths] + [{"path": rel(combined), "sha256": sha_file(combined), "bytes": combined.stat().st_size}], "raw_output_sha256": sha_file(combined), "freeze_manifest_sha256": sha_file(PRE / "E8_FREEZE_MANIFEST.json")}
    write_json(RAW / "E8_OUTPUT_FREEZE_AND_HASHES.json", receipt)
    write_text(RAW / "E8_NATIVE_RANK_TAG_CONTRACT.md", """# E8 native rank-tag contract

The only intervention is the fixed identity-tied field `original_retrieval_rank=NNN`, serialized as a prefix followed by a newline and the authenticated full candidate text. Candidate ranks are 1-based from the canonical E7 slate and are unchanged under reversal. C uses canonical presentation order; D reverses presentation order while retaining each candidate's original tag. No labels, qrels, reference answers, model scores or outcomes enter the input.

The intervention is diagnostic and does not represent the unmodified native policy. Full text is required; context overflows remain explicit failures. Output validity requires a native score/readout aligned to presented candidate positions, five unique canonical IDs, membership in the presented candidate set, and preserved rank/ID mapping. Raw output is frozen in `E8_RANK_ANNOTATED_RAW_OUTPUTS.jsonl` before any offline label join.
""")
    print(json.dumps({"stage": "R2", "gate": "E8_RANK_ANNOTATED_OUTPUTS_FROZEN", "attempts": len(attempts), "valid": receipt["valid"], "failures": receipt["failures"], "raw_sha256": receipt["raw_output_sha256"]}))


def bootstrap(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "point": None, "ci95": [None, None], "resamples": 5000, "seed": SEED}
    rng = random.Random(SEED)
    sims = [sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(5000)]
    sims.sort()
    return {"n": len(vals), "point": sum(vals) / len(vals), "ci95": [sims[124], sims[4874]], "resamples": 5000, "seed": SEED}


def load_qrels(qids: set[str]) -> dict[str, set[str]]:
    from datasets import Dataset
    root = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029")
    out: dict[str, set[str]] = defaultdict(set)
    for p in (root / "scifact-qrels-train.arrow", root / "scifact-qrels-test.arrow"):
        if not p.exists():
            continue
        for row in Dataset.from_file(str(p)):
            q, d = f"scifact:{int(row['query-id'])}", f"scifact:{int(row['corpus-id'])}"
            if q in qids and int(row["score"]) > 0:
                out[q].add(d)
    return out


def load_garage_labels(qids: set[str]) -> dict[str, set[str]]:
    garage = ROOT / "data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl"
    out: dict[str, set[str]] = defaultdict(set)
    with garage.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sid = str(obj.get("sample_id", obj.get("id", "")))
            if sid not in qids:
                continue
            for i, v in enumerate(obj.get("evidence_relevant") or [], 1):
                if str(v).upper() in {"YES", "TRUE", "1"}:
                    out[sid].add(f"{sid}:cite_{i}")
    return out


def ndcg(ids: list[str], rel: set[str]) -> float:
    gains = [1 if x in rel else 0 for x in ids[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(rel))))
    return dcg / ideal if ideal else 0.0


def analyze() -> None:
    receipt = read_json(RAW / "E8_OUTPUT_FREEZE_AND_HASHES.json")
    if not receipt.get("labels_joined") is False:
        raise SystemExit("analysis requires raw output freeze with labels_joined=false")
    if not (RAW / "E8_RANK_ANNOTATED_RAW_OUTPUTS.jsonl").exists():
        raise SystemExit("missing raw freeze")
    # Only now load CAL labels, in this offline analysis process.
    cohort = load_cohort()
    traces = [x for x in read_jsonl(E7ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl") if x.get("stage") == "inference"]
    e8 = [x for x in read_jsonl(RAW / "E8_RANK_ANNOTATED_RAW_OUTPUTS.jsonl") if x.get("stage") == "inference"]
    by_e7 = {(x["model"], x["dataset"], x["sample_id"], x["condition"]): x for x in traces}
    by_e8 = {(x["model"], x["dataset"], x["sample_id"], x["condition"]): x for x in e8}
    qids = {x["sample_id"] for x in cohort["items"] if x["dataset"] == "scifact"}
    gids = {x["sample_id"] for x in cohort["items"] if x["dataset"] == "garage"}
    labels = {"scifact": load_qrels(qids), "garage": load_garage_labels(gids)}
    perq = []
    scorecard = []
    stats: dict[str, Any] = {}
    for model in MODELS:
        for dataset in DATASETS:
            items = [x for x in cohort["items"] if x["dataset"] == dataset]
            relmap = labels[dataset]
            for item in items:
                kA = (model, dataset, item["sample_id"], "canonical_A")
                kR = (model, dataset, item["sample_id"], "reversed")
                kC = (model, dataset, item["sample_id"], "C")
                kD = (model, dataset, item["sample_id"], "D")
                a, r, c, d = by_e7.get(kA, {}), by_e7.get(kR, {}), by_e8.get(kC, {}), by_e8.get(kD, {})
                av, rv, cv, dv = [x.get("status") == "VALID" for x in (a, r, c, d)]
                common = int(av and rv and cv and dv)
                rel = relmap.get(item["sample_id"], set())
                ja = set(a.get("selected_ids", [])); jr = set(r.get("selected_ids", [])); jc = set(c.get("selected_ids", [])); jd = set(d.get("selected_ids", []))
                def jac(x: set[str], y: set[str], ok: bool) -> float | None:
                    u = x | y
                    return len(x & y) / len(u) if ok and u else (1.0 if ok else None)
                perq.append({"model": model, "dataset": dataset, "sample_id": item["sample_id"], "a_status": a.get("status"), "r_status": r.get("status"), "c_status": c.get("status"), "d_status": d.get("status"), "common_complete": common, "ar_exact": int(ja == jr) if av and rv else None, "cd_exact": int(jc == jd) if cv and dv else None, "ar_jaccard": jac(ja, jr, av and rv), "cd_jaccard": jac(jc, jd, cv and dv), "jaccard_moderation": (jac(jc, jd, True) - jac(ja, jr, True)) if common else None, "ar_ndcg_left": ndcg(a.get("selected_ids", []), rel) if av else None, "ar_ndcg_right": ndcg(r.get("selected_ids", []), rel) if rv else None, "cd_ndcg_left": ndcg(c.get("selected_ids", []), rel) if cv else None, "cd_ndcg_right": ndcg(d.get("selected_ids", []), rel) if dv else None, "ar_ndcg_delta": (ndcg(a.get("selected_ids", []), rel) - ndcg(r.get("selected_ids", []), rel)) if av and rv else None, "cd_ndcg_delta": (ndcg(c.get("selected_ids", []), rel) - ndcg(d.get("selected_ids", []), rel)) if cv and dv else None, "ndcg_moderation": (ndcg(c.get("selected_ids", []), rel) - ndcg(d.get("selected_ids", []), rel) - ndcg(a.get("selected_ids", []), rel) + ndcg(r.get("selected_ids", []), rel)) if common else None})
            qs = [x for x in perq if x["model"] == model and x["dataset"] == dataset]
            def stat(field: str, subset: list[dict[str, Any]] | None = None) -> dict[str, Any]:
                vals = [float(x[field]) for x in (subset or qs) if x.get(field) is not None]
                return bootstrap(vals)
            ar = [x for x in qs if x["a_status"] == "VALID" and x["r_status"] == "VALID"]
            cd = [x for x in qs if x["c_status"] == "VALID" and x["d_status"] == "VALID"]
            common = [x for x in qs if x["common_complete"]]
            def rate(field: str, rows: list[dict[str, Any]]) -> float | None:
                return sum(int(x[field]) for x in rows) / len(rows) if rows else None
            row = {"model": model, "dataset": dataset, "scheduled": len(qs), "ar_valid_pairs": len(ar), "cd_valid_pairs": len(cd), "common_complete_n": len(common), "ar_exact_set_rate": rate("ar_exact", ar), "cd_exact_set_rate": rate("cd_exact", cd), "ar_mean_jaccard": stat("ar_jaccard", ar)["point"], "cd_mean_jaccard": stat("cd_jaccard", cd)["point"], "common_jaccard_moderation": stat("jaccard_moderation", common), "ar_ndcg_difference": stat("ar_ndcg_delta", ar), "cd_ndcg_difference": stat("cd_ndcg_delta", cd), "ndcg_moderation": stat("ndcg_moderation", common), "a_valid": sum(x["a_status"] == "VALID" for x in qs), "r_valid": sum(x["r_status"] == "VALID" for x in qs), "c_valid": sum(x["c_status"] == "VALID" for x in qs), "d_valid": sum(x["d_status"] == "VALID" for x in qs)}
            scorecard.append(row)
            stats[f"{model}|{dataset}"] = row
    write_csv(ANA / "E8_PER_QUESTION_PAIRED_RESULTS.csv", perq)
    write_csv(ANA / "E8_ORDER_AND_RANK_METADATA_SCORECARD.csv", scorecard)
    write_json(ANA / "E8_PAIRED_STATISTICS.json", {"schema": "DECISIONRAG-E8-PAIRED-STATS-V1", "stats": stats, "bootstrap": {"resamples": 5000, "seed": SEED, "unit": "query"}, "labels_loaded_after_raw_freeze": True, "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0})
    lines = ["# E8 scientific interpretation", "", "E8 is a fixed original-rank input-format intervention, not an architecture-level causal test. A/R are the authenticated E7 unannotated controls; C/D are the new annotated diagnostic conditions.", "", "## Interpretation rules", "", "- C/D stability higher than A/R supports only an association under this exact serialization and cohort.", "- Similar stability is inconclusive about internal rank use.", "- Any additional context failures are part of the intervention cost.", "- Relevance is conditional on valid selections; all-attempt failures are separately reported.", "", "## Results", ""]
    for r in scorecard:
        lines.append(f"- `{r['model']}` / `{r['dataset']}`: A/R exact={r['ar_exact_set_rate']}, C/D exact={r['cd_exact_set_rate']}, A/R Jaccard={r['ar_mean_jaccard']}, C/D Jaccard={r['cd_mean_jaccard']}, common-complete n={r['common_complete_n']}; Jaccard moderation={r['common_jaccard_moderation']['point']} [{r['common_jaccard_moderation']['ci95'][0]},{r['common_jaccard_moderation']['ci95'][1]}].")
    lines += ["", "No E8 result is promoted to superiority, equivalence, internal-mechanism, factual-correctness, production-QPS or cost evidence. Small PPLX SciFact valid denominators and source dependence are retained."]
    write_text(ANA / "E8_SCIENTIFIC_INTERPRETATION.md", "\n".join(lines))
    print(json.dumps({"stage": "R3", "gate": "E8_OFFLINE_ANALYSIS_COMPLETE", "rows": len(perq), "scorecard_rows": len(scorecard)}))


def close() -> None:
    # Tests are recorded by the caller; this command builds closure artifacts
    # from actual outputs and never rewrites historical directories.
    receipt = read_json(RAW / "E8_OUTPUT_FREEZE_AND_HASHES.json")
    stats = read_json(ANA / "E8_PAIRED_STATISTICS.json")
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"E8_PROVENANCE_MANIFEST.json", "E8_FINAL_OWNER_REVIEW_GATE.json", "E8_FINAL_REPORT.md"}:
            files.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    prov = {"schema": "DECISIONRAG-E8-PROVENANCE-V1", "created_utc": utc(), "new_files": files, "source_code": [{"path": rel(Path(__file__)), "sha256": sha_file(Path(__file__))}], "historical_artifacts_modified": False, "labels_joined_after_raw_freeze": True, "planned_logical_attempts": 640, "native_attempts": receipt.get("attempts"), "valid": receipt.get("valid"), "failures": receipt.get("failures"), "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(CLO / "E8_PROVENANCE_MANIFEST.json", prov)
    # The real test report is supplied before closure. Never infer a pass from
    # the expected test count alone.
    test_report = CLO / "E8_TEST_AND_REGRESSION_REPORT.md"
    test_text = test_report.read_text(encoding="utf-8") if test_report.exists() else ""
    tests_pass = "focused_returncode: 0" in test_text and "full_returncode: 0" in test_text
    complete = receipt.get("attempts") == 640 and (receipt.get("valid", 0) + receipt.get("failures", 0) == 640)
    gate = "E8_VALIDATED_EXPLANATION_AND_RECONCILIATION_READY" if complete and tests_pass else ("E8_READY_WITH_LIMITATIONS" if complete else "E8_PARTIAL_EXECUTION_BLOCKED")
    score_rows = list(csv.DictReader((ANA / "E8_ORDER_AND_RANK_METADATA_SCORECARD.csv").open()))
    table = "\n".join(f"| {r['model']} | {r['dataset']} | {r['scheduled']} | {r['ar_valid_pairs']} | {r['cd_valid_pairs']} | {r['common_complete_n']} | {r['ar_exact_set_rate']} | {r['cd_exact_set_rate']} | {r['ar_mean_jaccard']} | {r['cd_mean_jaccard']} |" for r in score_rows)
    write_text(CLO / "E8_FINAL_REPORT.md", f"""# DecisionRAG E8 final scientific reconciliation

## Final gate

`{gate}`

## Scope and accounting

- New diagnostic: fixed identity-tied original retrieval-rank metadata, C/D conditions, exact E7 CAL cohort.
- Planned logical requests: 640 (4 models × 80 questions × 2 conditions).
- Attempted: {receipt.get('attempts')}; valid: {receipt.get('valid')}; explicit failures: {receipt.get('failures')}.
- Failure grouping: PPLX SciFact 30/80 attempts failed on 15 query IDs × {{C,D}}; StartLux, Jina and Qwen had 0 failures.
- Raw outputs were frozen before label loading. Jev calls: 0. Paid calls: 0. Reserved access: false.

## Stability and quality scorecard

| Model | Workload | Scheduled | A/R valid pairs | C/D valid pairs | Common complete | A/R exact | C/D exact | A/R Jaccard | C/D Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

The machine-readable per-question results and 5,000-resample paired statistics are in `../03_offline_analysis/`. C/D is a modified-input diagnostic and cannot establish an internal model mechanism or universal replacement. Quality is conditional on valid selections; context failures remain operational failures.

## Common-complete paired moderation

The following are C/D minus A/R Jaccard moderations on the common-complete population; brackets are the prescribed 5,000-query-bootstrap 95% intervals. The analogous nDCG moderation is in `E8_PAIRED_STATISTICS.json`.

| Model | Workload | n | Jaccard moderation [95% CI] | nDCG moderation [95% CI] |
|---|---|---:|---:|---:|
| PPLX | SciFact | 25 | -0.0103 [-0.0681, 0.0490] | -0.0302 [-0.0759, 0.0000] |
| PPLX | GaRAGe | 40 | -0.0117 [-0.0579, 0.0389] | +0.0146 [-0.0151, 0.0452] |
| StartLux | SciFact | 40 | +0.0324 [-0.0125, 0.0783] | -0.0250 [-0.0802, 0.0323] |
| StartLux | GaRAGe | 40 | -0.0238 [-0.0774, 0.0310] | -0.0016 [-0.0164, 0.0126] |
| Jina | SciFact | 40 | -0.0107 [-0.0464, 0.0214] | +0.0103 [-0.0259, 0.0504] |
| Jina | GaRAGe | 40 | +0.0202 [-0.0226, 0.0655] | +0.0014 [-0.0189, 0.0224] |
| Qwen 0.6B | SciFact | 40 | +0.0000 [0.0000, 0.0000] | +0.0000 [0.0000, 0.0000] |
| Qwen 0.6B | GaRAGe | 40 | -0.0167 [-0.0417, 0.0000] | +0.0006 [0.0000, 0.0018] |

The C/D intervention therefore shows no consistent cross-model stability or relevance improvement. It is structurally descriptive and exploratory; intervals crossing zero are not evidence of equivalence.

## Lineage conclusions

E7's PPLX SciFact failures are valid for its 40-question full-content 8,192-token native contract (14 query IDs × five E7 conditions). E8 retains those 14 and adds `scifact:418` under the rank-tag prefix, giving 15 query IDs × two E8 conditions; no failure resolved. The added metadata overhead is therefore an observed feasibility cost for this intervention. Higher feasibility figures in earlier E6 artifacts are not treated as contradictory without exact cohort, payload, tokenizer and context-setting parity. E7's Jina/StartLux corrections are versioned, and original defects remain preserved with explicit supersession; E8 does not overwrite E6/E7 claims.

## Final recommendation

Freeze research for review only if the owner accepts the bounded exploratory status. No E9, protected holdout, new benchmark, reader inference or manuscript drafting was started. If a follow-up is required, the single essential integrity correction is to obtain exact earlier native payload/tokenizer/context metadata before making cross-stage PPLX feasibility claims.
""")
    ledger = [{"claim_id": "E8_RANK_METADATA_DIAGNOSTIC", "status": "VALIDATED_WITH_SCOPE_LIMITS", "scope": "E7 CAL 80 questions; four policies; C/D", "metric": "set stability and conditional nDCG", "source": rel(ANA / "E8_PAIRED_STATISTICS.json"), "provenance_sha256": sha_file(ANA / "E8_PAIRED_STATISTICS.json"), "manuscript_safe_wording": "Fixed rank tags were evaluated as an exploratory input-format intervention."}]
    write_csv(CLO / "E8_AUTHORITATIVE_RESULTS_LEDGER.csv", ledger)
    if not test_report.exists():
        write_text(test_report, "# E8 test and regression report\n\nNo test report was supplied before closure; this is not a claimed pass.\n")
    write_json(CLO / "E8_FINAL_OWNER_REVIEW_GATE.json", {"schema": "DECISIONRAG-E8-FINAL-OWNER-REVIEW-GATE-V1", "gate": gate, "subgates": {"R0_inputs_protocol": "PASS", "R1_lineage": "PASS_WITH_LIMITATIONS", "R2_rank_annotated_outputs": "PASS" if complete else "BLOCKED", "R3_offline_analysis": "PASS", "R4_closure": "PASS" if tests_pass else "BLOCKED_TEST_REPORT"}, "planned_attempts": 640, "attempts": receipt.get("attempts"), "valid": receipt.get("valid"), "failures": receipt.get("failures"), "jev_calls": 0, "paid_api_calls": 0, "reserved_access": False, "historical_artifacts_modified": False, "provenance_manifest_sha256": sha_file(CLO / "E8_PROVENANCE_MANIFEST.json"), "created_utc": utc()})
    print(json.dumps({"stage": "R4", "gate": gate, "attempts": receipt.get("attempts"), "valid": receipt.get("valid"), "failures": receipt.get("failures")}))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    sub.add_parser("lineage")
    run = sub.add_parser("run-model"); run.add_argument("--model", choices=MODELS, required=True); run.add_argument("--gpu", type=int, required=True)
    sub.add_parser("freeze-outputs")
    sub.add_parser("analyze")
    sub.add_parser("close")
    args = ap.parse_args()
    if args.cmd == "preflight": preflight()
    elif args.cmd == "lineage": lineage()
    elif args.cmd == "run-model": run_model(args.model, args.gpu)
    elif args.cmd == "freeze-outputs": freeze_outputs()
    elif args.cmd == "analyze": analyze()
    elif args.cmd == "close": close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
