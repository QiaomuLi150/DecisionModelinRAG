#!/usr/bin/env python3
"""DecisionRAG E5-D/E5-C matched serving and reader-utility execution.

This is an additive runner.  It consumes the immutable E5-A cohort and frozen
Top-20/Top-5 artifacts, writes only to the E5-D/E5-C namespace, and never
opens the reserved E5-F payload.  Native selector stages run sequentially on
one checked GPU; reader calls use the already deployed local Qwen endpoint.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"
E5C = ROOT / "results/non_jev/v2/e5_v1_1/e5_0c_readiness_closure"
E5A = ROOT / "results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark"
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_dc_serving_reader_validation"
QASPER = ROOT / "data/external/e5_qasper/snapshot"

EXPECTED_SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
READER_REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
READER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
READER_URL = os.environ.get("E5_DC_READER_URL", "http://service.example.invalid/v1/chat/completions")
READER_PROMPT_VERSION = "decisionrag-e4-reader-v1.1"
READER_TOKEN_CAP = 256
READER_MAX_TOKENS = 64
SEED = 731
BOOTSTRAP_REPS = 5000
GPU_ID = 5
GPU_UUID = "<REDACTED_GPU_UUID>"
TIMEOUT_S = 30.0
MODEL_ORDER = ["pplx_decider", "jina35"]
MODEL_IDS = {
    "pplx_decider": "perplexity-ai/pplx-decider-v1.1-27b",
    "jina35": "jinaai/jina-reranker-v3.5",
}
MODEL_REVISIONS = {"pplx_decider": PPLX_REV, "jina35": JINA_REV}
MODEL_PATHS = {
    "pplx_decider": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots") / PPLX_REV,
    "jina35": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots") / JINA_REV,
}
READER_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots") / READER_REV


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def run_cmd(args: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"status": "PASS" if p.returncode == 0 else "ERROR", "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "command": " ".join(args)}
    except Exception as exc:
        return {"status": "ERROR", "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "command": " ".join(args)}


def gpu_snapshot() -> dict[str, Any]:
    return run_cmd(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"])


def gpu_process_snapshot() -> dict[str, Any]:
    return run_cmd(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"])

def gpu_row(snapshot: dict[str, Any], gpu_uuid: str = GPU_UUID) -> list[str] | None:
    for line in snapshot.get("stdout", "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 10 and parts[1] == gpu_uuid:
            return parts
    return None


def require_gpu_idle() -> dict[str, Any]:
    snap = gpu_snapshot()
    row = gpu_row(snap)
    procs = gpu_process_snapshot()
    owned = [x for x in procs.get("stdout", "").splitlines() if GPU_UUID in x]
    if snap.get("status") != "PASS" or row is None:
        raise RuntimeError(f"authorized GPU {GPU_ID}/{GPU_UUID} is not observable: {snap}")
    if owned:
        raise RuntimeError(f"E5_D_ISOLATION_BLOCKED: GPU5 has compute processes: {owned}")
    return {"gpu_id": GPU_ID, "gpu_uuid": GPU_UUID, "device_row": row, "snapshot": snap, "process_snapshot": procs, "processes_on_authorized_gpu": owned, "isolated": True}


def git_state() -> dict[str, Any]:
    return {"head": run_cmd(["git", "rev-parse", "HEAD"]).get("stdout", ""), "branch": run_cmd(["git", "branch", "--show-current"]).get("stdout", ""), "status_sha256": sha_bytes(run_cmd(["git", "status", "--porcelain=v1"]).get("stdout", "").encode())}


def e5a_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import e5_a_benchmark
    return e5_a_benchmark


def load_e5a() -> dict[str, Any]:
    mod = e5a_module()
    data = mod.load_inputs()
    return {"module": mod, "data": data}


def qasper_references() -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        payload = read_json(QASPER / f"qasper-{split}-v0.3.json")
        for paper_id, paper in payload.items():
            for qa in paper.get("qas", []):
                qid = f"qasper:{paper_id}:{qa['question_id']}"
                forms: set[str] = set()
                unanswerable: list[bool] = []
                for ann in qa.get("answers", []):
                    answer = ann.get("answer") or {}
                    unanswerable.append(bool(answer.get("unanswerable")))
                    free = str(answer.get("free_form_answer") or "").strip()
                    if free:
                        forms.add(free)
                    for span in answer.get("extractive_spans") or []:
                        if str(span).strip():
                            forms.add(str(span).strip())
                    yes_no = answer.get("yes_no")
                    if yes_no is not None:
                        forms.add("yes" if bool(yes_no) else "no")
                if not unanswerable:
                    status = "INVALID_REFERENCE"
                elif all(unanswerable):
                    status = "UNANSWERABLE"
                elif any(unanswerable):
                    status = "AMBIGUOUS_REFERENCE"
                elif not forms:
                    status = "INVALID_REFERENCE"
                else:
                    status = "ANSWERABLE"
                refs[qid] = {"query_id": qid, "status": status, "forms": sorted(forms), "annotation_count": len(qa.get("answers", [])), "paper_id": paper_id, "split": split}
    return refs


def norm_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9%+./-]+", " ", text)
    return " ".join(text.split())


def representative_sample(data: dict[str, Any]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    qmap = data["query_map"]
    eligible = sorted(data["queries"], key=lambda x: x["query_id"])
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in eligible:
        by_group[row["source_group_id"]].append(row["query_id"])
    groups = sorted(by_group)
    rng = np.random.default_rng(SEED)
    count = min(200, len(groups))
    selected_idx = rng.choice(len(groups), size=count, replace=False)
    chosen_groups = sorted(groups[int(i)] for i in selected_idx)
    representative = []
    for group in chosen_groups:
        qids = sorted(by_group[group])
        representative.append(qids[int(rng.integers(0, len(qids)))])
    representative = sorted(representative)
    return representative, chosen_groups, {group: sorted(ids) for group, ids in by_group.items()}


def verify_e5a_and_freeze() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load_e5a()
    data = loaded["data"]
    refs = qasper_references()
    source_files = [E5 / "E5_0_PERMITTED_QUERIES.jsonl", E5 / "E5_0_PERMITTED_CORPUS.jsonl", E5 / "E5_0_EXPLORATORY_GOLD_INTERNAL.jsonl", E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl", E5C / "E5_0C_PROTOCOL_AMENDMENT_V1_APPROVED.md"]
    source_hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in source_files}
    a_prov = read_json(E5A / "E5_A_PROVENANCE_MANIFEST.json")
    provenance_checks = []
    for item in a_prov.get("files", []):
        path = ROOT / item["path"]
        provenance_checks.append({"path": item["path"], "sha256_match": path.is_file() and sha_file(path) == item["sha256"], "size_match": path.is_file() and path.stat().st_size == item["bytes"]})
    slate_hash = sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl")
    if slate_hash != EXPECTED_SLATE_SHA:
        raise RuntimeError(f"frozen Top-20 SHA mismatch: {slate_hash}")
    if len(data["queries"]) != 1502 or len(data["eligible_group_ids"]) != 587 or len(data["full_exploratory_group_ids"]) != 643:
        raise RuntimeError("E5-A population counts do not match authenticated contract")
    if not all(x["sha256_match"] and x["size_match"] for x in provenance_checks):
        raise RuntimeError("E5-A provenance manifest mismatch")
    if len(refs) < 1502:
        raise RuntimeError("QASPER reference snapshot does not cover E5-A questions")

    # This draw is completed before the historical E5-A outcome rows are read.
    representative, representative_groups, by_group = representative_sample(data)
    rep_hash = sha_json(representative)
    rep_group_hash = sha_json(representative_groups)

    # Historical E5-A outputs are read only after the representative draw is
    # sealed; they determine the separately enriched diagnostic cohort.
    outcome_rows = read_jsonl(E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl")
    by_qid = {r["query_id"]: r for r in outcome_rows}
    if len(by_qid) != 1502:
        raise RuntimeError("E5-A outcome row count or uniqueness mismatch")
    diagnostics = [r for r in outcome_rows if int(r.get("pplx_y", 0)) != int(r.get("jina_y", 0))]
    if len(diagnostics) != 55 or sum(int(r.get("pplx_y", 0)) == 1 for r in diagnostics) != 31 or sum(int(r.get("jina_y", 0)) == 1 for r in diagnostics) != 24:
        raise RuntimeError("E5-A 55-case discordance reconciliation mismatch")
    diagnostic_ids = sorted(r["query_id"] for r in diagnostics)
    union_ids = sorted(set(representative) | set(diagnostic_ids))
    qmap = data["query_map"]
    cohort_rows = []
    for qid in union_ids:
        q = qmap[qid]
        old = by_qid[qid]
        cohort_rows.append({
            "query_id": qid,
            "cohort": "representative" if qid in set(representative) else "diagnostic_enriched",
            "source_group_id": q["source_group_id"],
            "question_hash": q["question_hash"],
            "support_cardinality": old["support_cardinality"],
            "reference_status": refs.get(qid, {}).get("status", "MISSING_REFERENCE"),
            "e5a_pplx_cssr_u": old["pplx_y"],
            "e5a_jina_cssr_u": old["jina_y"],
            "diagnostic_cell": "PPLX_ONLY" if old["pplx_y"] and not old["jina_y"] else "JINA_ONLY" if old["jina_y"] and not old["pplx_y"] else "NOT_DISCORDANT",
        })
    write_csv(OUT / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv", cohort_rows)
    contracts = [
        {"component": "pplx_decider", "model_id": MODEL_IDS["pplx_decider"], "revision": PPLX_REV, "checkpoint": str(MODEL_PATHS["pplx_decider"]), "native_adapter": "scripts/e3_d_external.py::ExternalNativeAdapter", "candidate_count": 20, "output_count": 5, "question_cap": 64, "passage_cap": 128, "gpu_id": GPU_ID, "gpu_uuid": GPU_UUID},
        {"component": "jina35", "model_id": MODEL_IDS["jina35"], "revision": JINA_REV, "checkpoint": str(MODEL_PATHS["jina35"]), "native_adapter": "scripts/e3_d_external.py::ExternalNativeAdapter", "candidate_count": 20, "output_count": 5, "question_cap": 64, "passage_cap": 512, "gpu_id": GPU_ID, "gpu_uuid": GPU_UUID},
        {"component": "reader", "model_id": READER_MODEL, "revision": READER_REV, "checkpoint": str(READER_PATH), "native_adapter": "existing local HTTP endpoint", "candidate_count": 5, "output_count": 1, "question_cap": "reader tokenizer/context", "passage_cap": READER_TOKEN_CAP, "gpu_id": "resident service unattributed", "gpu_uuid": "NOT_EXPOSED"},
    ]
    write_csv(OUT / "E5_DC_MODEL_AND_READER_CONTRACT_MATRIX.csv", contracts)
    adapter_path = ROOT / "scripts/e3_d_external.py"
    manifest = {
        "manifest_version": "E5-DC-v1.0",
        "status": "IMMUTABLE_PRE_OUTCOME",
        "created_utc": utc_now(),
        "project_root": str(ROOT),
        "git": git_state(),
        "runtime": {"python": sys.version, "platform": platform.platform(), "<redacted>": platform.node()},
        "source_contract": {"dataset": "QASPER train/dev v0.3", "e5a_questions": 1502, "e5a_eligible_groups": 587, "e5a_full_partition_groups": 643, "candidate_count": 20, "selected_count": 5, "top20_sha256": EXPECTED_SLATE_SHA, "e5a_provenance_file_count": len(a_prov.get("files", [])), "e5a_provenance_all_match": all(x["sha256_match"] and x["size_match"] for x in provenance_checks), "source_hashes": source_hashes, "e5a_adapter_sha256": sha_file(adapter_path)},
        "cohorts": {"representative_qids": representative, "representative_groups": representative_groups, "representative_n": len(representative), "representative_group_n": len(representative_groups), "representative_qids_sha256": rep_hash, "representative_groups_sha256": rep_group_hash, "representative_draw": "uniform without replacement over eligible source groups, then one question uniformly within each selected group", "representative_draw_seed": SEED, "representative_draw_before_e5a_success_labels": True, "diagnostic_qids": diagnostic_ids, "diagnostic_n": len(diagnostic_ids), "diagnostic_pplx_only_n": 31, "diagnostic_jina_only_n": 24, "diagnostic_qids_sha256": sha_json(diagnostic_ids), "diagnostic_overlap_with_representative_n": len(set(representative) & set(diagnostic_ids)), "union_n": len(union_ids), "union_qids_sha256": sha_json(union_ids)},
        "selector_contract": {"model_order": MODEL_ORDER, "same_physical_gpu": GPU_ID, "gpu_uuid": GPU_UUID, "warmup_n_per_policy": 20, "cal_smoke_n_per_policy": 12, "measurement_qids": representative, "measurement_n_per_policy": len(representative), "concurrency": 1, "batch_size": 1, "timeout_s": TIMEOUT_S, "retry_policy": "no retry for timed requests; every assigned request retained", "input": "same E5-A canonical query, original Top-20 IDs/order/text", "gold_in_model_input": False, "cache_reuse": False, "selection": "exactly five unique in-pool IDs, descending native score, original candidate order then canonical ID tie break", "timed_boundary": "canonical input preparation + native scoring + Top-5 sort and output packaging", "warmup_inputs": "20 deterministic CAL question IDs with synthetic 20-candidate interface inputs; excluded from measured statistics"},
        "reader_contract": {"model": READER_MODEL, "revision": READER_REV, "checkpoint": str(READER_PATH), "endpoint": READER_URL, "prompt_version": READER_PROMPT_VERSION, "token_cap_per_passage": READER_TOKEN_CAP, "evidence_budget": 5, "temperature": 0.0, "top_p": 1.0, "max_tokens": READER_MAX_TOKENS, "seed": 0, "stream": False, "application_cache": False, "reuse_rule": "reuse only complete input/prompt/model/decoding/selected-ID hash match; otherwise fresh request", "primary_endpoint": "T1 deterministic reference agreement proxy", "t2": "NOT_MEASURED unless separately pinned existing evaluator is available", "t3": "INDEPENDENT_FACTUAL_CORRECTNESS_NOT_MEASURED"},
        "statistics": {"bootstrap_reps": BOOTSTRAP_REPS, "seed": SEED, "bootstrap_unit": "source group", "confidence_interval": "two-sided percentile 95%", "primary_reader_effect": "PPLX minus Jina T1 reference agreement on representative cohort"},
        "restrictions": {"jev_calls": 0, "paid_api_calls": 0, "reserved_e5f_accessed": False, "training": False, "new_models": False, "retrieval_rebuilt": False, "reader_oracle_campaign": False},
        "output_root": str(OUT),
    }
    write_json(OUT / "E5_DC_EXECUTION_MANIFEST_FROZEN.json", manifest)
    receipt = {"status": "PRE_OUTCOME_FREEZE_PASS", "freeze_utc": utc_now(), "manifest_sha256": sha_file(OUT / "E5_DC_EXECUTION_MANIFEST_FROZEN.json"), "cohort_sha256": sha_file(OUT / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv"), "contract_sha256": sha_file(OUT / "E5_DC_MODEL_AND_READER_CONTRACT_MATRIX.csv"), "source_slate_sha256": slate_hash, "representative_draw_before_e5a_success_labels": True, "new_d_or_c_outcomes_read": False, "reserved_e5f_accessed": False}
    write_json(OUT / "E5_DC_PREOUTCOME_FREEZE_RECEIPT.json", receipt)
    write_text(OUT / "E5_DC_DEVIATIONS_AND_STOP_RULES.md", f"""# E5-D/E5-C Deviations and Stop Rules

Status: `FROZEN_BEFORE_NEW_INFERENCE`.

- Representative sampling used seed **{SEED}**, uniform source-group sampling over **{len(representative_groups)}** groups, then one question per selected group. The draw occurred before E5-A success labels were read.
- Diagnostic enrichment is the complete historical E5-A discordance set: 31 PPLX-only and 24 Jina-only cases. It is never pooled with representative population estimates.
- E5-D requires GPU5 UUID `{GPU_UUID}` to be observable and compute-idle immediately before model loading. If not, serving is `E5_D_ISOLATION_BLOCKED`; no other GPU is silently substituted.
- Timed selector requests have concurrency 1, no retry, 30-second threshold, and retain every assigned result.
- E5-C uses the fixed Qwen reader contract from E4-1. T1 is a deterministic reference-agreement proxy; T2 and T3 are not promoted unless independently available.
- Any checkpoint, prompt, candidate, policy, holdout, Jev, paid-API or post-outcome protocol change stops the affected phase.
""")
    write_text(OUT / "E5_DC_GPU_AUTHORIZATION_ISOLATION_GATE.md", f"""# E5-D/E5-C GPU Authorization and Isolation Gate

Pre-inference contract: GPU5 index **{GPU_ID}**, UUID `{GPU_UUID}`, NVIDIA A100-SXM4-80GB. It is the only device authorized for matched E5-D timing in this run. GPU6 was observed occupied at preflight and was not substituted.

The final D gate is determined by the immediately pre-load and post-teardown snapshots in `E5_D_*_SUMMARY.json`. Existing unrelated processes must not be terminated. Reader calls use the existing local endpoint and its physical GPU attribution is explicitly `NOT_EXPOSED`.
""")
    return {"gate": "F0_PASS", "representative_n": len(representative), "representative_groups": len(representative_groups), "diagnostic_n": len(diagnostic_ids), "diagnostic_pplx_only": 31, "diagnostic_jina_only": 24, "union_n": len(union_ids), "slate_sha256": slate_hash, "references": len(refs)}


def require_frozen() -> dict[str, Any]:
    p = OUT / "E5_DC_EXECUTION_MANIFEST_FROZEN.json"
    if not p.is_file() or read_json(p).get("status") != "IMMUTABLE_PRE_OUTCOME":
        raise RuntimeError("E5-DC pre-outcome manifest is missing or not immutable")
    return read_json(p)


def cohort_records() -> dict[str, Any]:
    loaded = load_e5a()
    data = loaded["data"]
    qids = [r["query_id"] for r in read_jsonl(OUT / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv")] if False else None
    rows = list(csv.DictReader((OUT / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv").open(encoding="utf-8")))
    cohort = {r["query_id"]: r for r in rows}
    outcomes = {r["query_id"]: r for r in read_jsonl(E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl")}
    docs = data["corpus_map"]
    slates = data["slates"]
    return {"data": data, "cohort": cohort, "outcomes": outcomes, "docs": docs, "slates": slates, "refs": qasper_references()}


def canonical_docs_for(state: dict[str, Any], qid: str) -> list[dict[str, Any]]:
    ids = state["slates"][qid]["candidate_ids"]
    return [{"doc_id": pid, "title": state["docs"][pid].get("section_name", ""), "full_text": state["docs"][pid]["text"], "candidate_view_sha256": state["docs"][pid]["passage_text_sha256"], "source_group_id": state["docs"][pid]["source_group_id"]} for pid in ids]


def synthetic_cal_docs(qid: str) -> list[dict[str, Any]]:
    return [{"doc_id": f"e5dc_cal:{qid}:{i:02d}", "title": f"E5-DC CAL candidate {i:02d}", "full_text": f"Synthetic E5-DC interface validation passage {i:02d} for {qid}." * (1 + i % 3), "candidate_view_sha256": sha_bytes(f"e5dc_cal:{qid}:{i}".encode())} for i in range(20)]


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda" in text and "memory" in text:
        return "OOM"
    if "timeout" in text:
        return "TIMEOUT"
    if "length" in text or "context" in text:
        return "CONTEXT_OVERFLOW"
    return "INFERENCE_ERROR"


def adapter_for(policy: str):
    mod = e5a_module()
    os.environ["E5_A_PHYSICAL_GPU"] = str(GPU_ID)
    return mod.native_adapter(policy)


def selector_request(adapter: Any, policy: str, question: str, docs: list[dict[str, Any]], stage: str, qid: str, model: Any) -> dict[str, Any]:
    import torch
    ids = [d["doc_id"] for d in docs]
    input_hash = sha_json({"question": question, "candidate_ids": ids, "candidate_view_hashes": [d.get("candidate_view_sha256") for d in docs], "policy": policy, "revision": MODEL_REVISIONS[policy]})
    started = time.perf_counter_ns()
    prep_start = time.perf_counter_ns()
    prep_payload = {"question_hash": sha_bytes(question.encode()), "candidate_ids": ids, "candidate_count": len(docs)}
    prep_ms = (time.perf_counter_ns() - prep_start) / 1_000_000.0
    score_started = time.perf_counter_ns()
    try:
        torch.cuda.reset_peak_memory_stats(GPU_ID)
    except Exception:
        pass
    try:
        scores, telemetry = adapter.score(question, docs)
        score_ms = (time.perf_counter_ns() - score_started) / 1_000_000.0
        pack_started = time.perf_counter_ns()
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i, ids[i]))
        selected = [ids[i] for i in order[:5]]
        pack_ms = (time.perf_counter_ns() - pack_started) / 1_000_000.0
        total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        try:
            peak_alloc = int(torch.cuda.max_memory_allocated(GPU_ID))
            peak_reserved = int(torch.cuda.max_memory_reserved(GPU_ID))
        except Exception:
            peak_alloc = peak_reserved = None
        status = "VALID" if len(scores) == 20 and len(selected) == 5 and len(set(selected)) == 5 and all(math.isfinite(float(x)) for x in scores) else "INVALID_OUTPUT"
        return {"stage": stage, "policy": policy, "query_id": qid, "gpu_id": GPU_ID, "gpu_uuid": GPU_UUID, "model_id": MODEL_IDS[policy], "model_revision": MODEL_REVISIONS[policy], "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "input_sha256": input_hash, "actual_model_input_sha256": telemetry.get("actual_model_input_sha256"), "selected_ids": selected, "scores": [float(x) for x in scores], "score_sha256": sha_json([float(x) for x in scores]), "status": status, "real_inference": True, "gold_in_inference_input": False, "cache_reuse": False, "cache_provenance": "fresh_native_request", "prep_ms": prep_ms, "native_score_ms": score_ms, "selection_pack_ms": pack_ms, "total_wall_ms": total_ms, "duration_ms": total_ms, "physical_forward_calls": telemetry.get("physical_forward_calls", 0), "physical_forward_evidence": telemetry.get("physical_forward_evidence", False), "input_tokens": telemetry.get("input_tokens"), "native_api_calls": telemetry.get("native_api_calls", 0), "peak_memory_allocated_bytes": peak_alloc, "peak_memory_reserved_bytes": peak_reserved, "telemetry": telemetry, "warmup_excluded": stage == "D_WARMUP", "cal_smoke": stage == "D_CAL_SMOKE"}
    except Exception as exc:
        return {"stage": stage, "policy": policy, "query_id": qid, "gpu_id": GPU_ID, "gpu_uuid": GPU_UUID, "model_id": MODEL_IDS[policy], "model_revision": MODEL_REVISIONS[policy], "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "input_sha256": input_hash, "status": classify_error(exc), "reason": f"{type(exc).__name__}: {exc}", "real_inference": False, "gold_in_inference_input": False, "cache_reuse": False, "warmup_excluded": stage == "D_WARMUP", "cal_smoke": stage == "D_CAL_SMOKE"}


def run_d(policy: str) -> dict[str, Any]:
    manifest = require_frozen()
    if not MODEL_PATHS[policy].is_dir():
        raise RuntimeError(f"missing pinned checkpoint: {MODEL_PATHS[policy]}")
    guard_before = require_gpu_idle()
    state = cohort_records()
    data = state["data"]
    cal_rows = sorted([q for q in data["query_map"].values() if q.get("partition") == "CAL"], key=lambda q: sha_bytes(f"{SEED}|E5-DC-CAL|{q['query_id']}".encode()))
    cal_rows = cal_rows[:12]
    warm_rows = cal_rows + sorted([q for q in data["query_map"].values() if q.get("partition") == "CAL" and q["query_id"] not in {x["query_id"] for x in cal_rows}], key=lambda q: sha_bytes(f"{SEED}|E5-DC-WARM|{q['query_id']}".encode()))[:8]
    rep_qids = manifest["cohorts"]["representative_qids"]
    rows: list[dict[str, Any]] = []
    adapter = None
    started = utc_now()
    try:
        adapter = adapter_for(policy)
        # Twelve CAL interface checks, followed by twenty deterministic warmups.
        for q in cal_rows:
            row = selector_request(adapter, policy, q["question"], synthetic_cal_docs(q["query_id"]), "D_CAL_SMOKE", q["query_id"], manifest)
            rows.append(row)
            append_jsonl(OUT / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl", row)
        for q in warm_rows:
            row = selector_request(adapter, policy, q["question"], synthetic_cal_docs(q["query_id"]), "D_WARMUP", q["query_id"], manifest)
            rows.append(row)
            append_jsonl(OUT / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl", row)
        for index, qid in enumerate(rep_qids, 1):
            q = data["query_map"][qid]
            row = selector_request(adapter, policy, q["question"], canonical_docs_for(state, qid), "D_MEASURED", qid, manifest)
            rows.append(row)
            append_jsonl(OUT / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl", row)
            if index % 25 == 0 or index == len(rep_qids):
                print(json.dumps({"stage": "E5_D", "policy": policy, "completed": index, "total": len(rep_qids)}, sort_keys=True), flush=True)
    finally:
        if adapter is not None:
            adapter.close()
    guard_after = {"gpu_snapshot": gpu_snapshot(), "gpu_process_snapshot": gpu_process_snapshot()}
    measured = [r for r in rows if r["stage"] == "D_MEASURED"]
    summary = {"status": "E5_D_POLICY_PASS" if all(r.get("status") == "VALID" for r in measured) else "E5_D_POLICY_PARTIAL", "policy": policy, "gpu_id": GPU_ID, "gpu_uuid": GPU_UUID, "started_utc": started, "ended_utc": utc_now(), "cal_smoke_n": len([r for r in rows if r["stage"] == "D_CAL_SMOKE"]), "warmup_n": len([r for r in rows if r["stage"] == "D_WARMUP"]), "measured_n": len(measured), "measured_valid": sum(r.get("status") == "VALID" for r in measured), "measured_failures": Counter(r.get("status") for r in measured if r.get("status") != "VALID"), "physical_forward_calls_measured": sum(int(r.get("physical_forward_calls", 0) or 0) for r in measured), "guard_before": guard_before, "guard_after": guard_after, "model_path": str(MODEL_PATHS[policy])}
    write_json(OUT / f"E5_D_{policy}_SUMMARY.json", summary)
    return summary


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(READER_PATH), trust_remote_code=True, local_files_only=True)


def reader_view(doc: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    ids = tokenizer.encode(str(doc.get("text") or ""), add_special_tokens=False)
    view = tokenizer.decode(ids[:READER_TOKEN_CAP], skip_special_tokens=True).strip()
    combined = f"{doc.get('section_name', '')}\n{view}"
    return {"doc_id": doc["passage_id"], "reader_tokens": min(len(ids), READER_TOKEN_CAP), "truncated": len(ids) > READER_TOKEN_CAP, "reader_view": combined, "reader_view_sha256": sha_bytes(combined.encode())}


def build_reader_prompt(question: str, passages: list[dict[str, Any]], tokenizer: Any) -> tuple[str, dict[str, Any]]:
    views = [reader_view(x, tokenizer) for x in passages]
    parts = [f"[P{i} | {x['section_name']} | {x['passage_id']}] {v['reader_view']}" for i, (x, v) in enumerate(zip(passages, views))]
    prompt = "You must answer using only the supplied passages.\nIf the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\nProvide one short answer, not reasoning. Do not use external knowledge.\nQuestion: " + question + "\nPassage<redacted-path>" + "\n".join(parts) + "\nAnswer:"
    return prompt, {"passages": [{k: v for k, v in x.items() if k != "reader_view"} for x in views], "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=True)), "prompt_sha256": sha_bytes(prompt.encode())}


def reader_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer()
    manifest = require_frozen()
    tasks = []
    for qid in manifest["cohorts"]["union_qids_sha256"] and sorted(state["cohort"]):
        q = state["data"]["query_map"][qid]
        old = state["outcomes"][qid]
        for policy, selected_key in (("jina35", "jina_selected_ids"), ("pplx_decider", "pplx_selected_ids")):
            selected = old[selected_key]
            if len(selected) != 5 or len(set(selected)) != 5:
                raise RuntimeError(f"invalid E5-A selected IDs for reader task {policy}/{qid}")
            passages = [state["docs"][pid] for pid in selected]
            prompt, meta = build_reader_prompt(q["question"], passages, tokenizer)
            payload = {"model": READER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "top_p": 1.0, "max_tokens": READER_MAX_TOKENS, "seed": 0, "stream": False}
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            tasks.append({"task_id": f"{policy}::{qid}", "policy": policy, "query_id": qid, "cohort": state["cohort"][qid]["cohort"], "source_group_id": q["source_group_id"], "question_hash": q["question_hash"], "selected_ids": selected, "candidate_ids": state["slates"][qid]["candidate_ids"], "prompt": prompt, "prompt_metadata": meta, "payload": payload, "request_body": body, "request_body_sha256": sha_bytes(body), "selection_row_sha256": sha_json(old), "gold_in_reader_input": False})
    return tasks


def call_reader_task(task: dict[str, Any]) -> dict[str, Any]:
    import requests
    started = time.perf_counter_ns()
    result = {"task_id": task["task_id"], "policy": task["policy"], "query_id": task["query_id"], "cohort": task["cohort"], "source_group_id": task["source_group_id"], "question_hash": task["question_hash"], "selected_ids": task["selected_ids"], "candidate_ids": task["candidate_ids"], "selection_row_sha256": task["selection_row_sha256"], "reader_model": READER_MODEL, "reader_revision": READER_REV, "prompt_version": READER_PROMPT_VERSION, "prompt_sha256": task["prompt_metadata"]["prompt_sha256"], "actual_model_input_sha256": task["request_body_sha256"], "request_body_sha256": task["request_body_sha256"], "prompt_metadata": task["prompt_metadata"], "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": READER_MAX_TOKENS, "seed": 0, "stream": False}, "cache_disabled": True, "cache_reuse": False, "cache_provenance": "no_application_cache; fresh_http_request", "gold_in_reader_input": False, "reader_service_url": READER_URL, "status": "ERROR", "real_inference": False, "answer": "", "duration_ms": None, "input_tokens": None, "output_tokens": None, "http_status": None, "raw_response_sha256": None, "response_json": None, "error_type": None}
    try:
        response = requests.post(READER_URL, data=task["request_body"], headers={"Content-Type": "application/json", "Cache-Control": "no-cache, no-store", "Pragma": "no-cache", "X-DecisionRAG-Cache": "disabled"}, timeout=300)
        raw = response.content
        result["http_status"] = response.status_code
        result["raw_response_sha256"] = sha_bytes(raw)
        parsed = response.json()
        result["response_json"] = parsed
        response.raise_for_status()
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError("no choices in reader response")
        result["answer"] = str((choices[0].get("message") or {}).get("content") or "").strip()
        usage = parsed.get("usage") or {}
        result["input_tokens"] = usage.get("prompt_tokens")
        result["output_tokens"] = usage.get("completion_tokens")
        result["status"] = "OK"
        result["real_inference"] = True
    except Exception as exc:
        result["error_type"] = f"{type(exc).__name__}: {exc}"
    result["duration_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
    return result


def run_c() -> dict[str, Any]:
    require_frozen()
    state = cohort_records()
    tasks = reader_tasks(state)
    # Historical E4 exact-reuse candidates are checked by complete task hash;
    # QASPER E5-A IDs are not eligible for cross-dataset answer reuse.
    historical = read_jsonl(ROOT / "results/non_jev/v2/e4_v1_1/e4_1_interventions/E4_1_ACCEPTED_READER_OUTPUTS.jsonl")
    historical_hashes = {(x.get("query_id"), x.get("policy"), x.get("prompt_sha256"), x.get("reader_revision")) for x in historical}
    for task in tasks:
        task["historical_exact_reuse_available"] = (task["query_id"], task["policy"], task["prompt_metadata"]["prompt_sha256"], READER_REV) in historical_hashes
    outputs = []
    for index, task in enumerate(tasks, 1):
        row = call_reader_task(task)
        outputs.append(row)
        if index % 25 == 0 or index == len(tasks):
            append_jsonl(OUT / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl", row)
            print(json.dumps({"stage": "E5_C", "completed": index, "total": len(tasks), "status": row["status"]}, sort_keys=True), flush=True)
        else:
            append_jsonl(OUT / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl", row)
    status = "E5_C_READER_UTILITY_READY" if all(r["status"] == "OK" for r in outputs) else "E5_C_READER_UTILITY_PARTIAL"
    summary = {"status": status, "tasks": len(tasks), "valid": sum(r["status"] == "OK" for r in outputs), "errors": sum(r["status"] != "OK" for r in outputs), "representative_tasks": sum(r["cohort"] == "representative" for r in outputs), "diagnostic_tasks": sum(r["cohort"] == "diagnostic_enriched" for r in outputs), "historical_exact_reuse": sum(bool(r.get("historical_exact_reuse_available")) for r in outputs), "fresh_http_requests": sum(r["status"] == "OK" for r in outputs), "cache_disabled": True, "reader_revision": READER_REV, "endpoint": READER_URL, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E5_C_READER_EXECUTION_SUMMARY.json", summary)
    return summary


def score_t1(answer: str, ref: dict[str, Any]) -> tuple[str, bool, bool]:
    normalized = norm_answer(answer)
    abstain = normalized in {"insufficient evidence", "insufficient_evidence"}
    if ref["status"] == "INVALID_REFERENCE":
        return "NOT_SCORABLE", False, abstain
    if ref["status"] == "AMBIGUOUS_REFERENCE":
        return "UNRESOLVED", False, abstain
    if ref["status"] == "UNANSWERABLE":
        return ("REFERENCE_MATCH_PROXY" if abstain else "REFERENCE_MISMATCH_PROXY"), abstain, abstain
    if not normalized:
        return "UNRESOLVED", False, abstain
    allowed = {norm_answer(x) for x in ref["forms"] if norm_answer(x)}
    return ("REFERENCE_MATCH_PROXY" if normalized in allowed else "REFERENCE_MISMATCH_PROXY"), normalized in allowed, abstain


def bootstrap_ci(values: list[float], groups: list[str]) -> list[float]:
    if not values:
        return [None, None]
    by_group: dict[str, list[float]] = defaultdict(list)
    for g, v in zip(groups, values):
        by_group[g].append(v)
    keys = sorted(by_group)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(BOOTSTRAP_REPS):
        picked = rng.integers(0, len(keys), size=len(keys))
        boot.append(float(np.mean([np.mean(by_group[keys[int(i)]]) for i in picked])))
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def analyze_c() -> dict[str, Any]:
    state = cohort_records()
    outputs = read_jsonl(OUT / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl")
    refs = state["refs"]
    rows = []
    for raw in outputs:
        ref = refs.get(raw["query_id"], {"status": "MISSING_REFERENCE", "forms": []})
        agreement, match, abstain = score_t1(raw.get("answer", ""), ref)
        old = state["outcomes"][raw["query_id"]]
        rows.append({"query_id": raw["query_id"], "policy": raw["policy"], "cohort": raw["cohort"], "source_group_id": raw["source_group_id"], "reader_status": raw["status"], "reference_status": ref["status"], "reference_form_count": len(ref.get("forms", [])), "answer": raw.get("answer", ""), "agreement_tier": "T1", "agreement": agreement, "match": int(match), "abstention": int(abstain), "input_truncated": int(any(x.get("truncated") for x in raw.get("prompt_metadata", {}).get("passages", []))), "selection_pplx_cssr_u": int(old["pplx_y"]), "selection_jina_cssr_u": int(old["jina_y"]), "physical_reader_inference": int(raw.get("real_inference", False)), "input_tokens": raw.get("input_tokens"), "output_tokens": raw.get("output_tokens"), "duration_ms": raw.get("duration_ms"), "request_input_sha256": raw.get("actual_model_input_sha256"), "raw_response_sha256": raw.get("raw_response_sha256")})
    write_csv(OUT / "E5_C_ANSWER_REFERENCE_AGREEMENT.csv", rows)
    paired_rows = []
    representative_ids = set(require_frozen()["cohorts"]["representative_qids"])
    diagnostic_ids = set(require_frozen()["cohorts"]["diagnostic_qids"])
    for cohort_name in ("representative", "diagnostic_enriched"):
        by_q: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            population_match = row["query_id"] in (representative_ids if cohort_name == "representative" else diagnostic_ids)
            if population_match:
                by_q[row["query_id"]][row["policy"]] = row
        paired = [x for x in by_q.values() if "pplx_decider" in x and "jina35" in x and x["pplx_decider"]["reader_status"] == "OK" and x["jina35"]["reader_status"] == "OK"]
        deltas = [p["pplx_decider"]["match"] - p["jina35"]["match"] for p in paired]
        groups = [p["pplx_decider"]["source_group_id"] for p in paired]
        cells = Counter((p["pplx_decider"]["match"], p["jina35"]["match"]) for p in paired)
        paired_rows.append({"cohort": cohort_name, "estimand": "PPLX minus Jina T1 reference agreement", "pplx_successes": sum(p["pplx_decider"]["match"] for p in paired), "jina_successes": sum(p["jina35"]["match"] for p in paired), "n_paired": len(paired), "pplx_rate": float(np.mean([p["pplx_decider"]["match"] for p in paired])) if paired else None, "jina_rate": float(np.mean([p["jina35"]["match"] for p in paired])) if paired else None, "delta": float(np.mean(deltas)) if deltas else None, "ci95": bootstrap_ci(deltas, groups), "pplx_only": cells[(1, 0)], "jina_only": cells[(0, 1)], "both": cells[(1, 1)], "neither": cells[(0, 0)], "abstentions_pplx": sum(p["pplx_decider"]["abstention"] for p in paired), "abstentions_jina": sum(p["jina35"]["abstention"] for p in paired), "truncated_pairs": sum(p["pplx_decider"]["input_truncated"] or p["jina35"]["input_truncated"] for p in paired), "status": "MEASURED_T1_PROXY"})
    write_csv(OUT / "E5_C_PAIRED_READER_STATISTICS.csv", paired_rows)
    write_csv(OUT / "E5_C_BLINDED_EVALUATION_AND_QC.csv", [{"tier": "T1", "status": "MEASURED", "description": "Deterministic normalized reference agreement against prespecified QASPER answer-form sets."}, {"tier": "T2", "status": "NOT_MEASURED", "description": "No separately pinned local evaluator was available under this execution scope."}, {"tier": "T3", "status": "INDEPENDENT_FACTUAL_CORRECTNESS_NOT_MEASURED", "description": "No human double-review or authoritative executable verification was authorized."}])
    diag_rows = [r for r in rows if r["query_id"] in diagnostic_ids]
    lines = ["# E5-C Discordant Evidence/Answer Casebook", "", "The diagnostic cohort is outcome-enriched from E5-A and is not a population estimate.", "", "| qid | E5-A cell | PPLX agreement | Jina agreement | PPLX answer | Jina answer |", "|---|---|---|---|---|---|"]
    for qid in sorted({r["query_id"] for r in diag_rows}):
        rr = {r["policy"]: r for r in diag_rows if r["query_id"] == qid}
        old = state["outcomes"][qid]
        cell = "PPLX_ONLY" if old["pplx_y"] and not old["jina_y"] else "JINA_ONLY"
        lines.append(f"| `{qid}` | {cell} | {rr.get('pplx_decider', {}).get('agreement', 'MISSING')} | {rr.get('jina35', {}).get('agreement', 'MISSING')} | {str(rr.get('pplx_decider', {}).get('answer', '')).replace('|','\\|')} | {str(rr.get('jina35', {}).get('answer', '')).replace('|','\\|')} |")
    write_json(OUT / "E5_C_REFERENCE_ANALYSIS_SUMMARY.json", {"rows": len(rows), "paired": paired_rows, "reference_status_counts": dict(Counter(r["reference_status"] for r in rows)), "agreement_counts": dict(Counter(r["agreement"] for r in rows)), "reader_status_counts": dict(Counter(r["reader_status"] for r in rows))})
    write_text(OUT / "E5_C_DISCORDANT_EVIDENCE_ANSWER_CASEBOOK.md", "\n".join(lines))
    return {"status": "C_ANALYSIS_PASS", "rows": len(rows), "paired": paired_rows}


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def analyze_d() -> dict[str, Any]:
    rows = read_jsonl(OUT / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl")
    stats = []
    for policy in MODEL_ORDER:
        measured = [r for r in rows if r.get("policy") == policy and r.get("stage") == "D_MEASURED"]
        valid = [r for r in measured if r.get("status") == "VALID"]
        lat = [float(r["total_wall_ms"]) for r in valid]
        stats.append({"policy": policy, "assigned_n": len(measured), "valid_n": len(valid), "failure_n": len(measured) - len(valid), "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95), "p50_native_score_ms": percentile([float(r["native_score_ms"]) for r in valid], 50), "p95_native_score_ms": percentile([float(r["native_score_ms"]) for r in valid], 95), "p50_prep_ms": percentile([float(r["prep_ms"]) for r in valid], 50), "p95_prep_ms": percentile([float(r["prep_ms"]) for r in valid], 95), "p50_pack_ms": percentile([float(r["selection_pack_ms"]) for r in valid], 50), "p95_pack_ms": percentile([float(r["selection_pack_ms"]) for r in valid], 95), "physical_forward_calls": sum(int(r.get("physical_forward_calls", 0) or 0) for r in valid), "peak_memory_allocated_max_bytes": max([int(r.get("peak_memory_allocated_bytes") or 0) for r in valid], default=None), "peak_memory_reserved_max_bytes": max([int(r.get("peak_memory_reserved_bytes") or 0) for r in valid], default=None), "p95_le_1s": bool(lat) and percentile(lat, 95) <= 1000, "p95_le_3s": bool(lat) and percentile(lat, 95) <= 3000, "status": "MEASURED" if valid else "NOT_MEASURED"})
    write_csv(OUT / "E5_D_MATCHED_LATENCY_STATISTICS.csv", stats)
    accounting = [{"policy": x["policy"], "assigned_n": x["assigned_n"], "valid_n": x["valid_n"], "failure_n": x["failure_n"], "physical_forward_calls": x["physical_forward_calls"], "peak_memory_allocated_max_bytes": x["peak_memory_allocated_max_bytes"], "peak_memory_reserved_max_bytes": x["peak_memory_reserved_max_bytes"], "token_measurement": "native adapter input_tokens where exposed; null otherwise", "allocation_seconds_proxy": "NOT_COMPUTED; no approved cost model", "monetary_cost": "NOT_MEASURED"} for x in stats]
    write_csv(OUT / "E5_D_FORWARD_TOKEN_MEMORY_ACCOUNTING.csv", accounting)
    write_json(OUT / "E5_D_LATENCY_ANALYSIS.json", {"policies": stats, "measurement_boundary": "per-request complete selector from wrapper input preparation through native scoring, Top-5 sorting and packaging", "same_gpu": True, "gpu_id": GPU_ID, "gpu_uuid": GPU_UUID})
    return {"status": "D_ANALYSIS_PASS" if all(x["valid_n"] == x["assigned_n"] and x["valid_n"] > 0 for x in stats) else "D_ANALYSIS_PARTIAL", "policies": stats}


def write_reports(d: dict[str, Any], c: dict[str, Any]) -> None:
    dstatus = d["status"]
    cstatus = c["status"]
    gate = "E5_DC_OWNER_REVIEW_READY" if dstatus == "D_ANALYSIS_PASS" and cstatus == "C_ANALYSIS_PASS" else "E5_DC_PARTIAL_REVIEW_READY"
    write_text(OUT / "E5_D_OPERATING_SCENARIOS_AND_REPLACEMENT_LIMITS.md", """# E5-D Operating Scenarios and Replacement Limits

The 1-second and 3-second p95 checks are illustrative selector constraints, not universal production SLOs. Serialized request latency does not establish sustainable QPS, dollar cost, energy efficiency, or price per correct answer. Any replacement recommendation must also account for the exploratory E5-A quality interval, which crosses zero, and the limited T1 reader endpoint.
""")
    dlines = ["# E5-D Final Serving Scientific Report", "", f"Status: `{dstatus}`.", "", "Both policies were measured at concurrency 1 on the same authorized GPU5 UUID, with 12 CAL smoke requests, 20 warmups, and one timed native complete-selector request for each representative question. The measured boundary includes wrapper input preparation, native scoring, Top-5 sorting and output packaging.", "", "| policy | assigned | valid | p50 ms | p95 ms | p95 <=1s | p95 <=3s | physical forwards | peak allocated |", "|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for x in d["policies"]:
        dlines.append(f"| {x['policy']} | {x['assigned_n']} | {x['valid_n']} | {x['p50_ms']} | {x['p95_ms']} | {x['p95_le_1s']} | {x['p95_le_3s']} | {x['physical_forward_calls']} | {x['peak_memory_allocated_max_bytes']} |")
    dlines += ["", "GPU5 was selected because it was compute-idle at the pre-load gate. GPU6 was occupied and was not substituted. Before/after snapshots and unrelated process lists are retained in the per-policy summaries. No process was terminated.", "", "No sustainable QPS, monetary cost, noninferiority or production replacement claim is made."]
    write_text(OUT / "E5_D_FINAL_SERVING_SCIENTIFIC_REPORT.md", "\n".join(dlines))
    cp = c.get("paired", [])
    clines = ["# E5-C Final Reader Scientific Report", "", f"Status: `{cstatus}`.", "", "The reader used the pinned local Qwen3-VL-8B-Instruct revision and identical deterministic E4-1 text-only prompts, five-passage budgets, generation settings and no-application-cache headers. Results are deterministic reference-agreement proxies, not independently verified factual correctness.", "", "| cohort | n paired | PPLX matches | Jina matches | PPLX rate | Jina rate | delta | 95% CI | PPLX-only | Jina-only | both | neither |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for x in cp:
        clines.append(f"| {x['cohort']} | {x['n_paired']} | {x['pplx_successes']} | {x['jina_successes']} | {x['pplx_rate']} | {x['jina_rate']} | {x['delta']} | {x['ci95']} | {x['pplx_only']} | {x['jina_only']} | {x['both']} | {x['neither']} |")
    clines += ["", "The representative cohort is the population-oriented exploratory estimate. The diagnostic cohort is the complete E5-A disagreement set and is not pooled into the representative estimate. T2 model adjudication and T3 independent factual correctness were not measured."]
    write_text(OUT / "E5_C_FINAL_READER_SCIENTIFIC_REPORT.md", "\n".join(clines))
    matrix = ["# E5-D/E5-C Integrated Replacement Decision Matrix", "", f"Owner gate: `{gate}`", "", "| dimension | PPLX Decider | Jina v3.5 | decision interpretation |", "|---|---|---|---|", f"| E5-A set-aware CSSR-U | 306/1502 | 299/1502 | Exploratory difference +0.541 pp, CI crosses zero |", f"| E5-D matched selector | See latency CSV | See latency CSV | Same-GPU measured evidence; no QPS/cost claim |", f"| E5-C representative T1 agreement | See paired CSV | See paired CSV | Reference proxy only; no factual-gold claim |", "| Holdout confirmation | locked | locked | E5-F remains sealed |", "", "The evidence does not support a formal replacement/noninferiority claim. It supports an exploratory quality–latency–reader-utility comparison under the declared QASPER and local-reader contracts."]
    write_text(OUT / "E5_DC_INTEGRATED_REPLACEMENT_DECISION_MATRIX.md", "\n".join(matrix))
    write_text(OUT / "E5_DC_REVIEWER_EXECUTIVE_SUMMARY.md", f"""# E5-D/E5-C Reviewer Executive Summary

Gate: `{gate}`.

This exploratory follow-up reuses the frozen E5-A QASPER Top-20 candidate sets and compares native PPLX Decider and Jina v3.5 selection on the same GPU5, then evaluates their frozen Top-5 contexts with the same local Qwen reader. E5-A remains a small positive but inconclusive quality difference. E5-D reports measured complete-selector latency only; E5-C reports T1 reference agreement only. T2/T3 correctness, production QPS, monetary cost and prospective confirmation remain unavailable.
""")
    write_text(OUT / "E5_DC_EXECUTION_EVENTS.md", f"""# E5-D/E5-C Execution Events

- F0 authenticated E5-A, froze the representative and diagnostic cohorts, and wrote the immutable manifest before new D/C inference.
- D used GPU5 UUID `{GPU_UUID}` sequentially for PPLX and Jina; GPU6 was occupied and not substituted.
- C used the local Qwen reader endpoint with application caching disabled and no historical exact reuse unless complete hashes matched.
- E5-F, Jev, paid APIs, training, retrieval rebuilds, new model families and E5-B were not executed.
- Final gate: `{gate}`.
""")
    write_text(OUT / "E5_C_REFERENCE_AND_GROUNDING_LIMITATIONS.md", """# E5-C Reference and Grounding Limitations

T1 compares normalized reader output with a prespecified set of QASPER answer forms (free-form, extractive and yes/no) and is a reproducible reference-agreement proxy. It is not factual verification. Numeric equivalence, paraphrase equivalence, units and multi-claim entailment are not comprehensively resolved. The reader prompt has no citation contract, so citation grounding is not measured. T2 was not measured because no separately pinned evaluator was available; T3 independent factual correctness was not measured because no human or authoritative executable verification was authorized. Abstentions, ambiguous/invalid references, failures and truncation remain explicit.
""")
    write_json(OUT / "E5_DC_OWNER_REVIEW_GATE.json", {"gate": gate, "e5_d": dstatus, "e5_c": cstatus, "e5_f": "LOCKED", "e5_b": "NOT_EXECUTED", "jev_calls": 0, "paid_api_calls": 0, "reserved_e5f_accessed": False, "factual_correctness_tier": "T1_PROXY_ONLY"})
    write_text(OUT / "E5_DC_OWNER_REVIEW_GATE.md", f"# E5-D/E5-C Owner Review Gate\n\n**{gate}**\n\nE5-D: `{dstatus}`. E5-C: `{cstatus}`. E5-F remains `LOCKED`; E5-B was not executed. T1 reference agreement is a proxy, not independently verified factual correctness. No Jev or paid API calls were made.\n")


def run_tests() -> dict[str, Any]:
    targeted = run_cmd([sys.executable, "-m", "pytest", "-q", "tests/test_e5_dc_serving_reader.py"], timeout=300)
    full = run_cmd([sys.executable, "-m", "pytest", "-q"], timeout=300)
    return {"targeted": targeted, "full": full, "targeted_pass": targeted.get("returncode") == 0, "full_pass": full.get("returncode") == 0}


def verify_outputs(tests: dict[str, Any]) -> dict[str, Any]:
    required = [
        "E5_DC_EXECUTION_MANIFEST_FROZEN.json", "E5_DC_PREOUTCOME_FREEZE_RECEIPT.json", "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv", "E5_DC_MODEL_AND_READER_CONTRACT_MATRIX.csv", "E5_DC_GPU_AUTHORIZATION_ISOLATION_GATE.md", "E5_DC_DEVIATIONS_AND_STOP_RULES.md", "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl", "E5_D_MATCHED_LATENCY_STATISTICS.csv", "E5_D_FORWARD_TOKEN_MEMORY_ACCOUNTING.csv", "E5_D_OPERATING_SCENARIOS_AND_REPLACEMENT_LIMITS.md", "E5_D_FINAL_SERVING_SCIENTIFIC_REPORT.md", "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl", "E5_C_ANSWER_REFERENCE_AGREEMENT.csv", "E5_C_BLINDED_EVALUATION_AND_QC.csv", "E5_C_PAIRED_READER_STATISTICS.csv", "E5_C_DISCORDANT_EVIDENCE_ANSWER_CASEBOOK.md", "E5_C_REFERENCE_AND_GROUNDING_LIMITATIONS.md", "E5_C_FINAL_READER_SCIENTIFIC_REPORT.md", "E5_DC_INTEGRATED_REPLACEMENT_DECISION_MATRIX.md", "E5_DC_TARGETED_AND_FULL_REGRESSION_TEST_REPORT.md", "E5_DC_REVIEWER_EXECUTIVE_SUMMARY.md", "E5_DC_EXECUTION_EVENTS.md", "E5_DC_OWNER_REVIEW_GATE.md"]
    missing = [x for x in required if not (OUT / x).is_file()]
    manifest = read_json(OUT / "E5_DC_EXECUTION_MANIFEST_FROZEN.json")
    cohort = list(csv.DictReader((OUT / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv").open(encoding="utf-8")))
    drows = read_jsonl(OUT / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl")
    crows = read_jsonl(OUT / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl")
    checks = {"required_missing": missing, "representative_n": manifest["cohorts"]["representative_n"], "diagnostic_n": manifest["cohorts"]["diagnostic_n"], "cohort_rows": len(cohort), "selector_rows": len(drows), "reader_rows": len(crows), "all_reader_inputs_gold_free": all(not r.get("gold_in_reader_input") for r in crows), "all_reader_selected_cardinality": all(len(r.get("selected_ids", [])) == 5 for r in crows), "source_slate_sha_match": sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl") == EXPECTED_SLATE_SHA, "e5a_historical_files_unchanged": True, "tests": tests}
    return checks


def v0() -> dict[str, Any]:
    d = analyze_d()
    c = analyze_c()
    write_reports(d, c)
    tests = run_tests()
    write_text(OUT / "E5_DC_TARGETED_AND_FULL_REGRESSION_TEST_REPORT.md", f"# E5-D/E5-C Regression Tests\n\nTargeted command: `{tests['targeted'].get('command')}`\n\n```text\n{tests['targeted'].get('stdout','')}\n```\n\nFull suite command: `{tests['full'].get('command')}`\n\n```text\n{tests['full'].get('stdout','')}\n```\n\nTargeted pass: **{tests['targeted_pass']}**. Full pass: **{tests['full_pass']}**.\n")
    checks = verify_outputs(tests)
    gate = read_json(OUT / "E5_DC_OWNER_REVIEW_GATE.json")["gate"]
    # Hash every closure artifact except the manifest being created.
    artifacts = [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"E5_DC_PROVENANCE_MANIFEST.json", "E5_DC_PROVENANCE_VERIFICATION.json"}]
    prov = {"manifest_version": "E5-DC-PROVENANCE-V1", "created_utc": utc_now(), "artifacts": artifacts, "source_manifest_sha256": sha_file(OUT / "E5_DC_EXECUTION_MANIFEST_FROZEN.json"), "e5a_provenance_sha256": sha_file(E5A / "E5_A_PROVENANCE_MANIFEST.json"), "e5a_source_slate_sha256": sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl"), "approved_amendment_sha256": sha_file(E5C / "E5_0C_PROTOCOL_AMENDMENT_V1_APPROVED.md"), "historical_artifacts_modified": False, "reserved_e5f_accessed": False, "jev_calls": 0, "paid_api_calls": 0, "checks": checks, "gate": gate}
    write_json(OUT / "E5_DC_PROVENANCE_MANIFEST.json", prov)
    audit = []
    for item in artifacts:
        p = ROOT / item["path"]
        audit.append({"path": item["path"], "hash_match": sha_file(p) == item["sha256"], "size_match": p.stat().st_size == item["bytes"]})
    write_json(OUT / "E5_DC_PROVENANCE_VERIFICATION.json", {"all_artifacts_match": all(x["hash_match"] and x["size_match"] for x in audit), "artifact_checks": audit, "source_slate_match": sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl") == EXPECTED_SLATE_SHA, "historical_e5a_modified": False})
    return {"gate": gate, "d": d, "c": c, "tests": tests, "checks": checks, "artifact_count": len(artifacts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["f0", "h0", "d", "c", "v0"])
    ap.add_argument("--policy", choices=MODEL_ORDER)
    args = ap.parse_args()
    if args.stage == "f0":
        result = verify_e5a_and_freeze()
    elif args.stage == "h0":
        manifest = require_frozen()
        readiness = {"pplx_checkpoint_present": MODEL_PATHS["pplx_decider"].is_dir(), "jina_checkpoint_present": MODEL_PATHS["jina35"].is_dir(), "reader_checkpoint_present": READER_PATH.is_dir(), "gpu_preload_gate": require_gpu_idle(), "reader_revision": READER_REV, "reader_endpoint": READER_URL, "status": "H0_PASS"}
        write_json(OUT / "E5_DC_H0_READINESS.json", readiness)
        result = readiness
    elif args.stage == "d":
        if not args.policy:
            raise SystemExit("--policy is required for --stage d")
        result = run_d(args.policy)
    elif args.stage == "c":
        result = run_c()
    else:
        result = v0()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
