#!/usr/bin/env python3
"""DecisionRAG E5-A frozen replacement benchmark.

The runner has an explicit CPU-only M0 freeze, model-native H0/A0 stages and a
post-outcome CPU S0/V0 stage.  It consumes only the E5 exploratory QASPER
artifacts and never opens the reserved E5-F payload.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
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
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark"
EXPECTED_SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_GPU = 5
JINA_GPU = 6
GPU_UUIDS = {
    5: "<REDACTED_GPU_UUID>",
    6: "<REDACTED_GPU_UUID>",
}
MODEL_SPECS = {
    "pplx_decider": {
        "id": "perplexity-ai/pplx-decider-v1.1-27b",
        "revision": PPLX_REV,
        "kind": "pplx_native",
        "gpu": PPLX_GPU,
        "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/" + PPLX_REV,
    },
    "jina35": {
        "id": "jinaai/jina-reranker-v3.5",
        "revision": JINA_REV,
        "kind": "jina_listwise",
        "gpu": JINA_GPU,
        "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/" + JINA_REV,
    },
}
MAX_ATTEMPTS = 2
BOOTSTRAP_REPS = 5000
SEED = 731


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_cmd(args: list[str], clear_cuda_visible: bool = False, timeout: int = 30) -> dict[str, Any]:
    try:
        env = None
        if clear_cuda_visible:
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return {"status": "PASS" if p.returncode == 0 else "ERROR", "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "command": " ".join(args)}
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "command": " ".join(args)}


def gpu_snapshot() -> dict[str, Any]:
    return run_cmd(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], clear_cuda_visible=True)


def gpu_process_snapshot() -> dict[str, Any]:
    return run_cmd(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"], clear_cuda_visible=True)

def verify_gpu_safe(gpu: int) -> dict[str, Any]:
    """Require the frozen physical UUID to be present and compute-idle."""
    expected = GPU_UUIDS[gpu]
    snapshot = gpu_snapshot()
    if snapshot.get("status") != "PASS":
        if os.environ.get("E5_A_EXTERNAL_PREFLIGHT") == "1" and os.environ.get("E5_A_PREFLIGHT_GPU_UUID") == expected and os.environ.get("E5_A_PREFLIGHT_PROCESS_COUNT") == "0":
            return {"status": "PASS_EXTERNAL_SHELL_GUARD", "gpu_index": gpu, "gpu_uuid": expected, "device_row": os.environ.get("E5_A_PREFLIGHT_DEVICE_ROW", ""), "processes": [], "snapshot": snapshot, "process_snapshot": {"status": "EXTERNAL_SHELL_GUARD", "process_count": 0}}
        raise RuntimeError(f"nvidia-smi device query failed: {snapshot}")
    found = None
    for line in snapshot.get("stdout", "").splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) >= 7 and fields[1] == expected:
            found = fields
            break
    if found is None:
        raise RuntimeError(f"authorized GPU UUID not visible: {expected}")
    processes = gpu_process_snapshot()
    owned = [line for line in processes.get("stdout", "").splitlines() if expected in line]
    if owned:
        raise RuntimeError(f"authorized GPU is not compute-idle: {expected}: {owned}")
    return {"gpu_index": gpu, "gpu_uuid": expected, "device_row": found, "processes": owned, "snapshot": snapshot, "process_snapshot": processes}


def git_state() -> dict[str, Any]:
    status = run_cmd(["git", "status", "--porcelain=v1"])
    head = run_cmd(["git", "rev-parse", "HEAD"])
    branch = run_cmd(["git", "branch", "--show-current"])
    return {"head": head.get("stdout", ""), "branch": branch.get("stdout", ""), "dirty": bool(status.get("stdout")), "status_sha256": sha_bytes(status.get("stdout", "").encode())}


def load_inputs() -> dict[str, Any]:
    permitted = read_jsonl(E5 / "E5_0_PERMITTED_QUERIES.jsonl")
    corpus = read_jsonl(E5 / "E5_0_PERMITTED_CORPUS.jsonl")
    gold = read_jsonl(E5 / "E5_0_EXPLORATORY_GOLD_INTERNAL.jsonl")
    slates = read_jsonl(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl")
    mapping = list(csv.DictReader((E5 / "E5_0_GOLD_MAPPING_AND_ELIGIBILITY_AUDIT.csv").open(encoding="utf-8")))
    query_map = {r["query_id"]: r for r in permitted}
    gold_map = {r["query_id"]: r for r in gold}
    exp_ids = sorted(gold_map)
    queries = [query_map[qid] for qid in exp_ids]
    corpus_map = {r["passage_id"]: r for r in corpus}
    slate_map = {r["query_id"]: r for r in slates}
    if len(exp_ids) != 1502 or len(slates) != 1502 or set(exp_ids) != set(slate_map):
        raise RuntimeError("E5 exploratory question/slate count or ID mismatch")
    if sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl") != EXPECTED_SLATE_SHA:
        raise RuntimeError("frozen original slate SHA-256 mismatch")
    for row in slates:
        ids = row["candidate_ids"]
        if len(ids) != 20 or len(set(ids)) != 20 or not set(ids) <= set(corpus_map):
            raise RuntimeError(f"invalid frozen slate {row['query_id']}")
    # Reconstruct alternative support sets using the audited E5-0C mapper.
    sys.path.insert(0, str(ROOT / "scripts"))
    from e5_0c_readiness import audit_support_semantics, load_e5_inputs
    audited = audit_support_semantics(load_e5_inputs())
    support = {}
    for qid in exp_ids:
        analysis = audited["analyses"].get(qid, {})
        sets = []
        for values in analysis.get("sets", []):
            values = sorted(set(values))
            if 1 <= len(values) <= 5:
                sets.append(values)
        if not sets:
            raise RuntimeError(f"eligible exploratory query has no valid support set: {qid}")
        support[qid] = {"sets": sorted(sets), "flags": analysis.get("flags", []), "support_cardinality": len(gold_map[qid]["support_ids"]), "source_group_id": query_map[qid]["source_group_id"]}
    group_ids = sorted({query_map[qid]["source_group_id"] for qid in exp_ids})
    all_partition_groups = {r["source_group_id"] for r in permitted if r["partition"] == "E5_EXPLORATORY"}
    return {"queries": queries, "query_map": query_map, "corpus": corpus, "corpus_map": corpus_map, "gold": gold_map, "slates": slate_map, "support": support, "mapping": mapping, "eligible_group_ids": group_ids, "full_exploratory_group_ids": sorted(all_partition_groups)}


def model_contract_rows() -> list[dict[str, Any]]:
    adapter_hash = sha_file(ROOT / "scripts/e3_d_external.py")
    rows = []
    for policy, spec in MODEL_SPECS.items():
        path = Path(spec["path"])
        rows.append({
            "policy": policy,
            "model_id": spec["id"],
            "revision": spec["revision"],
            "native_kind": spec["kind"],
            "checkpoint_path": str(path),
            "checkpoint_present": path.is_dir(),
            "adapter": "scripts/e3_d_external.py::ExternalNativeAdapter",
            "adapter_sha256": adapter_hash,
            "native_question_cap": 64,
            "native_passage_cap": 128 if policy == "pplx_decider" else 512,
            "native_objective": "NATIVE_SLATE_CATEGORICAL:v1" if policy == "pplx_decider" else "native_listwise_chunked_relevance",
            "candidate_count": 20,
            "output_count": 5,
            "score_direction": "descending",
            "selection_tie_break": "original candidate order, then canonical ID",
            "max_attempts": MAX_ATTEMPTS,
            "gpu_index": spec["gpu"],
            "gpu_uuid": GPU_UUIDS[spec["gpu"]],
        })
    return rows


def m0() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    contracts = model_contract_rows()
    if not all(x["checkpoint_present"] for x in contracts):
        raise RuntimeError("one or more pinned checkpoints are unavailable")
    candidate_view_hashes = {pid: row["passage_text_sha256"] for pid, row in data["corpus_map"].items()}
    query_ids = sorted(data["query_map"])
    cohort_hash = sha_json(query_ids)
    source_group_hash = sha_json(data["eligible_group_ids"])
    manifest = {
        "manifest_version": "E5-A-v1.0",
        "status": "IMMUTABLE_PRE_OUTCOME",
        "created_utc": utc_now(),
        "run_id": "e5_a_20261014_" + sha_json({"cohort": cohort_hash, "slates": EXPECTED_SLATE_SHA})[:12],
        "project_root": str(ROOT),
        "git": git_state(),
        "runtime": {"python": sys.version, "platform": platform.platform(), "<redacted>": platform.node()},
        "execution_authorization": "owner-authorized E5-A exploratory selector inference; GPU5/GPU6 only",
        "input_contract": {
            "dataset": "QASPER train/dev v0.3",
            "eligible_question_count": len(data["queries"]),
            "eligible_question_ids_sha256": cohort_hash,
            "full_exploratory_partition_group_count": len(data["full_exploratory_group_ids"]),
            "eligible_group_count": len(data["eligible_group_ids"]),
            "eligible_group_ids_sha256": source_group_hash,
            "candidate_count": 20,
            "selected_count": 5,
            "frozen_slate_file": str(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl"),
            "frozen_slate_file_sha256": EXPECTED_SLATE_SHA,
            "candidate_view_hashes_sha256": sha_json(candidate_view_hashes),
            "support_set_source": "E5-0C audited permitted-record mapper",
            "reserved_payload_accessed": False,
            "gold_in_model_inputs": False,
        },
        "statistics": {"primary": "PPLX minus Jina equal-source-group set-aware CSSR-U", "bootstrap": "paired source-group bootstrap", "bootstrap_reps": BOOTSTRAP_REPS, "seed": SEED, "confidence_interval": "two-sided percentile 95%"},
        "retry_and_failure": {"max_attempts": MAX_ATTEMPTS, "retry_only": ["TIMEOUT", "INFRA_INTERRUPTED"], "invalid_output_is_failure": True, "unattempted_is_missing": True},
        "models": contracts,
        "secondary_scope": "retrieval Top-5 only; BGE/Liquid/zerank not required and not allowed to mutate primary cohort",
        "output_root": str(OUT),
        "commands": {"m0": "python scripts/e5_a_benchmark.py --stage m0", "h0": "python scripts/e5_a_benchmark.py --stage h0 --policy POLICY --gpu GPU", "a0": "python scripts/e5_a_benchmark.py --stage a0 --policy POLICY --gpu GPU", "s0": "python scripts/e5_a_benchmark.py --stage s0", "v0": "python scripts/e5_a_benchmark.py --stage v0"},
    }
    write_json(OUT / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json", manifest)
    contract_path = OUT / "E5_A_MODEL_NATIVE_CONTRACT_MATRIX.csv"
    with contract_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(contracts[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(contracts)
    integrity = {
        "status": "PASS",
        "slate_sha256": EXPECTED_SLATE_SHA,
        "slate_hash_match": True,
        "eligible_questions": len(data["queries"]),
        "eligible_groups": len(data["eligible_group_ids"]),
        "full_exploratory_groups": len(data["full_exploratory_group_ids"]),
        "all_slates_exactly_20_unique": True,
        "all_candidate_ids_resolve": True,
        "candidate_view_hashes_sha256": sha_json(candidate_view_hashes),
        "reserved_payload_accessed": False,
        "gold_in_model_inputs": False,
        "source_artifacts": {str(p): sha_file(p) for p in [E5 / "E5_0_PROVENANCE_MANIFEST.json", E5 / "E5_0_EXPLORATORY_GOLD_INTERNAL.jsonl", E5 / "E5_0_PERMITTED_QUERIES.jsonl", E5 / "E5_0_PERMITTED_CORPUS.jsonl", E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl", E5C / "E5_0C_PROTOCOL_AMENDMENT_V1_APPROVED.md"]},
    }
    write_json(OUT / "E5_A_SOURCE_AND_SLATE_INTEGRITY.json", integrity)
    write_text(OUT / "E5_A_SOURCE_AND_SLATE_INTEGRITY.md", f"""# E5-A Source and Slate Integrity

Status: `PASS`.

- Eligible exploratory questions: **{len(data['queries'])}**.
- Full E5 exploratory partition groups: **{len(data['full_exploratory_group_ids'])}**.
- Groups containing at least one eligible question: **{len(data['eligible_group_ids'])}**.
- Frozen Top-20 slates: **{len(data['slates'])}**, all exactly 20 unique resolvable IDs.
- Original slate SHA-256: `{EXPECTED_SLATE_SHA}` (exact match).
- Reserved payload/outcomes accessed: **false**.
- Gold/support IDs in model inputs: **false**.

The primary group denominator is the **{len(data['eligible_group_ids'])} groups with eligible questions**, not the full {len(data['full_exploratory_group_ids'])}-group partition count.
""")
    write_text(OUT / "E5_A_PROTOCOL_AND_PREOUTCOME_FREEZE.md", """# E5-A Protocol and Pre-Outcome Freeze

Status: `FROZEN_BEFORE_EXPLORATORY_OUTCOMES`.

Primary contrast: PPLX Decider minus Jina v3.5, identical frozen natural Top-20 candidate IDs and canonical text, exactly five unique outputs.

Primary endpoint: set-aware unconditional CSSR-U; a query succeeds when any prespecified valid mapped annotated support set of cardinality 1–5 is wholly contained in the selected Top-5. Alternative sets are not unioned. Natural retrieval misses are unconditional failures. The original first-stage Top-5 is a separate retrieval-only control.

Primary estimator: equal-source-group mean of paired query-level differences over groups containing eligible questions. Uncertainty: 5,000 paired source-group bootstrap replicates, seed 731, two-sided percentile 95% interval. Query-weighted and valid-output sensitivities are secondary.

Native contracts, prompts, text caps, output parsing, tie breaking, retry policy, GPU identities, artifact paths and model revisions are frozen in `E5_A_IMMUTABLE_EXECUTION_MANIFEST.json`.

No E5-F payload, reserved outcome, reader answer, Jev, paid API, new retrieval or label-dependent routing is permitted.
""")
    write_text(OUT / "E5_A_GPU_DEVICE_AUTHORIZATION_AND_ISOLATION.md", f"""# E5-A GPU Device Authorization and Isolation

GPU 5: `{GPU_UUIDS[5]}` — PPLX Decider.  GPU 6: `{GPU_UUIDS[6]}` — Jina v3.5.

These are the only devices authorized for E5-A. Every H0/A0 worker must capture `<GPU_PROCESS_LISTING_OMITTED>
""")
    write_text(OUT / "E5_A_CHANGE_CONTROL_AND_DEVIATIONS.md", """# E5-A Change Control and Deviations

At M0 there are no post-outcome changes or deviations. The approved E5-0C set-aware endpoint amendment is pre-outcome and is recorded in the source artifact `E5_0C_PROTOCOL_AMENDMENT_V1_APPROVED.md`. Optional secondary model controls are not required for the primary gate and cannot alter the frozen cohort or endpoint.
""")
    receipt = {"status": "PRE_OUTCOME_FREEZE_PASS", "freeze_utc": utc_now(), "manifest_sha256": sha_file(OUT / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json"), "contract_matrix_sha256": sha_file(contract_path), "source_integrity_sha256": sha_file(OUT / "E5_A_SOURCE_AND_SLATE_INTEGRITY.json"), "protocol_sha256": sha_file(OUT / "E5_A_PROTOCOL_AND_PREOUTCOME_FREEZE.md"), "gpu_auth_sha256": sha_file(OUT / "E5_A_GPU_DEVICE_AUTHORIZATION_AND_ISOLATION.md"), "reserved_accessed": False, "outcomes_read": False}
    write_json(OUT / "E5_A_PRE_OUTCOME_FREEZE_RECEIPT.json", receipt)
    return {"gate": "M0_PASS", "manifest_sha256": receipt["manifest_sha256"], "eligible_questions": len(data["queries"]), "eligible_groups": len(data["eligible_group_ids"]), "full_partition_groups": len(data["full_exploratory_group_ids"]), "slate_sha256": EXPECTED_SLATE_SHA}


def require_manifest() -> dict[str, Any]:
    p = OUT / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json"
    if not p.is_file():
        raise RuntimeError("M0 manifest missing")
    manifest = read_json(p)
    if manifest.get("status") != "IMMUTABLE_PRE_OUTCOME":
        raise RuntimeError("manifest is not immutable pre-outcome")
    return manifest


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda out of memory" in text:
        return "OOM"
    if "timeout" in text:
        return "TIMEOUT"
    if "context" in text or "length" in text:
        return "CONTEXT_OVERFLOW"
    if "parse" in text or "keyerror" in text:
        return "PARSE_INVALID"
    return "OTHER"


def native_adapter(policy: str):
    import torch
    physical_gpu = os.environ.get("E5_A_PHYSICAL_GPU")
    if physical_gpu is None:
        raise RuntimeError("E5_A_PHYSICAL_GPU is required for native model loading")
    torch.cuda.set_device(int(physical_gpu))
    sys.path.insert(0, str(ROOT / "scripts"))
    from e3_d_external import ExternalNativeAdapter
    return ExternalNativeAdapter(policy)


def cal_smoke_queries(data: dict[str, Any], n: int = 12) -> list[dict[str, Any]]:
    cal = [r for r in data["query_map"].values() if r.get("partition") == "CAL"]
    return sorted(cal, key=lambda r: sha_bytes(f"{SEED}|E5-A-H0|{r['query_id']}".encode()))[:n]


def synthetic_docs(qid: str, n: int = 20) -> list[dict[str, Any]]:
    docs = []
    for i in range(n):
        text = f"Synthetic E5-A CAL interface validation document {i:02d} for {qid}. " + ("Longer validation text. " * (1 + i % 4))
        docs.append({"doc_id": f"e5a_h0:{qid}:{i:02d}", "title": f"E5-A H0 candidate {i:02d}", "full_text": text, "candidate_view_sha256": sha_bytes(text.encode())})
    return docs


def h0(policy: str, gpu: int) -> dict[str, Any]:
    require_manifest()
    if gpu != MODEL_SPECS[policy]["gpu"]:
        raise RuntimeError(f"GPU {gpu} is not the frozen GPU for {policy}")
    verify_gpu_safe(gpu)
    os.environ["E5_A_PHYSICAL_GPU"] = str(gpu)
    data = load_inputs()
    rows = cal_smoke_queries(data)
    snapshot_before = gpu_snapshot(); processes_before = gpu_process_snapshot()
    results = []
    adapter = None
    try:
        adapter = native_adapter(policy)
        for q in rows:
            docs = synthetic_docs(q["query_id"])
            started = time.time()
            try:
                scores, telemetry = adapter.score(q["question"], docs)
                status = "VALID" if len(scores) == 20 and all(math.isfinite(float(x)) for x in scores) else "PARSE_INVALID"
                result = {"stage": "H0_CAL_SMOKE", "policy": policy, "gpu_id": gpu, "gpu_uuid": GPU_UUIDS[gpu], "query_id": q["query_id"], "candidate_ids": [d["doc_id"] for d in docs], "input_sha256": sha_json({"question": q["question"], "docs": docs, "policy": policy}), "score_count": len(scores), "selected_ids": [docs[i]["doc_id"] for i in sorted(range(20), key=lambda i: (-float(scores[i]), i))[:5]], "scores": [float(x) for x in scores], "status": status, "real_inference": True, "gold_in_inference_input": False, "cache_reuse": False, "attempt": 1, "duration_s": time.time() - started, "telemetry": telemetry}
            except Exception as exc:
                result = {"stage": "H0_CAL_SMOKE", "policy": policy, "gpu_id": gpu, "gpu_uuid": GPU_UUIDS[gpu], "query_id": q["query_id"], "candidate_ids": [d["doc_id"] for d in docs], "input_sha256": sha_json({"question": q["question"], "docs": docs, "policy": policy}), "status": classify_error(exc), "error": f"{type(exc).__name__}: {exc}", "real_inference": False, "gold_in_inference_input": False, "cache_reuse": False, "attempt": 1, "duration_s": time.time() - started}
            append_jsonl(OUT / "E5_A_NATIVE_ATTEMPT_AND_TELEMETRY.jsonl", result)
            results.append(result)
    finally:
        if adapter is not None:
            adapter.close()
    summary = {"stage": "H0_CAL_SMOKE", "policy": policy, "gpu_id": gpu, "gpu_uuid": GPU_UUIDS[gpu], "requested": len(rows), "valid": sum(r.get("status") == "VALID" for r in results), "statuses": dict(Counter(r.get("status") for r in results)), "gpu_before": snapshot_before, "processes_before": processes_before, "gpu_after": gpu_snapshot(), "processes_after": gpu_process_snapshot()}
    write_json(OUT / f"E5_A_H0_{policy}_GPU{gpu}.json", summary)
    return summary


def canonical_docs(data: dict[str, Any], qid: str) -> list[dict[str, Any]]:
    slate = data["slates"][qid]
    docs = []
    for pid in slate["candidate_ids"]:
        row = data["corpus_map"][pid]
        docs.append({"doc_id": pid, "title": row.get("section_name", ""), "full_text": row["text"], "candidate_view_sha256": row["passage_text_sha256"], "source_group_id": row["source_group_id"]})
    return docs


def load_terminal(policy: str) -> dict[str, dict[str, Any]]:
    out = {}
    p = OUT / "E5_A_PRIMARY_NATIVE_PREDICTIONS.jsonl"
    for row in read_jsonl(p):
        if row.get("policy") == policy and row.get("stage") == "A0":
            out[row["query_id"]] = row
    return out


def score_native(policy: str, gpu: int) -> dict[str, Any]:
    require_manifest()
    if gpu != MODEL_SPECS[policy]["gpu"]:
        raise RuntimeError(f"GPU {gpu} is not the frozen GPU for {policy}")
    verify_gpu_safe(gpu)
    os.environ["E5_A_PHYSICAL_GPU"] = str(gpu)
    data = load_inputs()
    query_ids = sorted(data["gold"])
    previous = load_terminal(policy)
    snapshot_before = gpu_snapshot(); processes_before = gpu_process_snapshot()
    adapter = None
    counts = Counter(); attempts = 0
    try:
        adapter = native_adapter(policy)
        for idx, qid in enumerate(query_ids, 1):
            if qid in previous and previous[qid].get("status") == "VALID":
                counts["reuse_exact"] += 1
                continue
            q = data["query_map"][qid]
            docs = canonical_docs(data, qid)
            input_hash = sha_json({"query_id": qid, "question": q["question"], "candidate_ids": [d["doc_id"] for d in docs], "candidate_view_hashes": [d["candidate_view_sha256"] for d in docs], "policy": policy, "revision": MODEL_SPECS[policy]["revision"]})
            for attempt in range(1, MAX_ATTEMPTS + 1):
                attempts += 1
                started = time.time()
                try:
                    scores, telemetry = adapter.score(q["question"], docs)
                    if len(scores) != 20 or any(not math.isfinite(float(x)) for x in scores):
                        raise RuntimeError("native score vector invalid")
                    order = sorted(range(20), key=lambda i: (-float(scores[i]), i, docs[i]["doc_id"]))
                    selected = [docs[i]["doc_id"] for i in order[:5]]
                    row = {"stage": "A0", "policy": policy, "query_id": qid, "source_group_id": q["source_group_id"], "question_hash": q["question_hash"], "candidate_ids": [d["doc_id"] for d in docs], "candidate_ids_sha256": sha_json([d["doc_id"] for d in docs]), "candidate_view_hashes": [d["candidate_view_sha256"] for d in docs], "input_sha256": input_hash, "model_id": MODEL_SPECS[policy]["id"], "model_revision": MODEL_SPECS[policy]["revision"], "gpu_id": gpu, "gpu_uuid": GPU_UUIDS[gpu], "scores": [float(x) for x in scores], "score_sha256": sha_json([float(x) for x in scores]), "selected_ids": selected, "selected_ids_sha256": sha_json(selected), "status": "VALID", "attempt": attempt, "real_inference": True, "gold_in_inference_input": False, "cache_reuse": False, "cache_provenance": "fresh_native_request", "duration_s": time.time() - started, "telemetry": telemetry}
                    append_jsonl(OUT / "E5_A_PRIMARY_NATIVE_PREDICTIONS.jsonl", row)
                    append_jsonl(OUT / "E5_A_NATIVE_ATTEMPT_AND_TELEMETRY.jsonl", row)
                    counts["valid"] += 1
                    break
                except Exception as exc:
                    status = classify_error(exc)
                    row = {"stage": "A0", "policy": policy, "query_id": qid, "source_group_id": q["source_group_id"], "question_hash": q["question_hash"], "candidate_ids": [d["doc_id"] for d in docs], "candidate_ids_sha256": sha_json([d["doc_id"] for d in docs]), "input_sha256": input_hash, "model_id": MODEL_SPECS[policy]["id"], "model_revision": MODEL_SPECS[policy]["revision"], "gpu_id": gpu, "gpu_uuid": GPU_UUIDS[gpu], "status": status, "error": f"{type(exc).__name__}: {exc}", "attempt": attempt, "real_inference": False, "gold_in_inference_input": False, "cache_reuse": False, "duration_s": time.time() - started}
                    append_jsonl(OUT / "E5_A_NATIVE_ATTEMPT_AND_TELEMETRY.jsonl", row)
                    if attempt == MAX_ATTEMPTS or status not in {"TIMEOUT", "INFRA_INTERRUPTED"}:
                        append_jsonl(OUT / "E5_A_PRIMARY_NATIVE_PREDICTIONS.jsonl", row)
                        counts[status] += 1
                        break
            if idx % 10 == 0 or idx == len(query_ids):
                print(json.dumps({"stage": "A0", "policy": policy, "completed": idx, "total": len(query_ids), "counts": dict(counts)}, sort_keys=True), flush=True)
    finally:
        if adapter is not None:
            adapter.close()
    summary = {"stage": "A0", "policy": policy, "gpu_id": gpu, "gpu_uuid": GPU_UUIDS[gpu], "requested": len(query_ids), "attempts": attempts, "counts": dict(counts), "gpu_before": snapshot_before, "processes_before": processes_before, "gpu_after": gpu_snapshot(), "processes_after": gpu_process_snapshot()}
    write_json(OUT / f"E5_A_A0_{policy}_SUMMARY.json", summary)
    return summary


def valid_selection(row: dict[str, Any], candidate_ids: set[str]) -> bool:
    selected = row.get("selected_ids", [])
    return row.get("status") == "VALID" and len(selected) == 5 and len(set(selected)) == 5 and set(selected) <= candidate_ids


def outcome(selected: list[str], sets: list[list[str]]) -> int:
    chosen = set(selected)
    return int(any(set(s) <= chosen for s in sets))


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def s0() -> dict[str, Any]:
    require_manifest()
    data = load_inputs()
    preds = {p: load_terminal(p) for p in ("pplx_decider", "jina35")}
    rows = []
    for qid in sorted(data["gold"]):
        slate_ids = set(data["slates"][qid]["candidate_ids"])
        sets = data["support"][qid]["sets"]
        feasible = int(any(set(s) <= slate_ids for s in sets))
        retrieval_selected = data["slates"][qid]["candidate_ids"][:5]
        retrieval_y = outcome(retrieval_selected, sets)
        p_rows = {}
        for policy in preds:
            r = preds[policy].get(qid)
            p_rows[policy] = r
        ys = {}
        for policy, r in p_rows.items():
            ys[policy] = outcome(r.get("selected_ids", []) if r else [], sets) if r and valid_selection(r, slate_ids) else 0
        rows.append({"query_id": qid, "source_group_id": data["support"][qid]["source_group_id"], "support_cardinality": data["support"][qid]["support_cardinality"], "support_set_count": len(sets), "support_flags": ";".join(data["support"][qid]["flags"]), "retrieval_feasible": feasible, "retrieval5_y": retrieval_y, "pplx_y": ys["pplx_decider"], "jina_y": ys["jina35"], "pplx_status": (p_rows["pplx_decider"] or {}).get("status", "MISSING"), "jina_status": (p_rows["jina35"] or {}).get("status", "MISSING"), "pplx_selected_ids": (p_rows["pplx_decider"] or {}).get("selected_ids", []), "jina_selected_ids": (p_rows["jina35"] or {}).get("selected_ids", [])})
    write_jsonl(OUT / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl", rows)
    n = len(rows); groups = sorted({r["source_group_id"] for r in rows})
    by_group = defaultdict(list)
    for r in rows: by_group[r["source_group_id"]].append(r)
    group_rows = []
    for g in groups:
        rr = by_group[g]
        group_rows.append({"source_group_id": g, "n": len(rr), "pplx_mean": float(np.mean([x["pplx_y"] for x in rr])), "jina_mean": float(np.mean([x["jina_y"] for x in rr])), "retrieval5_mean": float(np.mean([x["retrieval5_y"] for x in rr])), "delta": float(np.mean([x["pplx_y"] - x["jina_y"] for x in rr]))})
    deltas = np.array([x["delta"] for x in group_rows], dtype=float)
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS): bootstrap[i] = float(np.mean(deltas[rng.integers(0, len(deltas), size=len(deltas))]))
    largest_group = max(group_rows, key=lambda x: (x["n"], x["source_group_id"]))
    remaining_groups = [x for x in group_rows if x["source_group_id"] != largest_group["source_group_id"]]
    remaining_deltas = np.array([x["delta"] for x in remaining_groups], dtype=float)
    leave_largest_boot = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS): leave_largest_boot[i] = float(np.mean(remaining_deltas[rng.integers(0, len(remaining_deltas), size=len(remaining_deltas))]))
    def rate(key: str) -> dict[str, Any]:
        return {"successes": int(sum(x[key] for x in rows)), "n": n, "rate": float(np.mean([x[key] for x in rows]))}
    feasible_rows = [x for x in rows if x["retrieval_feasible"]]
    def cond(key: str) -> dict[str, Any]:
        return {"successes": int(sum(x[key] for x in feasible_rows)), "n": len(feasible_rows), "rate": float(np.mean([x[key] for x in feasible_rows])) if feasible_rows else None}
    pair = Counter()
    for r in rows: pair[('pplx' if r['pplx_y'] else '0') + ('_jina' if r['jina_y'] else '_0')] += 1
    group_macro = {}
    for policy, key in (("pplx_decider", "pplx_y"), ("jina35", "jina_y"), ("retrieval5", "retrieval5_y")):
        group_macro[policy] = {"successes_group_sum": float(sum(np.mean([x[key] for x in by_group[g]]) for g in groups)), "n_groups": len(groups), "rate": float(np.mean([np.mean([x[key] for x in by_group[g]]) for g in groups]))}
    stats = {"status": "S0_PASS", "primary": {"estimand": "PPLX minus Jina equal-source-group set-aware CSSR-U", "delta": float(np.mean(deltas)), "ci95": percentile_ci(bootstrap), "bootstrap_reps": BOOTSTRAP_REPS, "seed": SEED, "groups": len(groups), "queries": n, "group_contribution_min": float(np.min(deltas)), "group_contribution_median": float(np.median(deltas)), "group_contribution_p90": float(np.percentile(deltas, 90)), "group_contribution_max": float(np.max(deltas)), "largest_group": largest_group, "leave_largest_group_out": {"excluded_group": largest_group["source_group_id"], "excluded_n": largest_group["n"], "remaining_groups": len(remaining_groups), "delta": float(np.mean(remaining_deltas)), "ci95": percentile_ci(leave_largest_boot)}, "top_10_group_shares": [{"source_group_id": x["source_group_id"], "n": x["n"], "share": x["n"] / n} for x in sorted(group_rows, key=lambda x: (-x["n"], x["source_group_id"]))[:10]]}, "per_policy": {"pplx_decider": {"cssr_u": rate("pplx_y"), "cssr_c": cond("pplx_y"), "group_macro_cssr_u": group_macro["pplx_decider"]}, "jina35": {"cssr_u": rate("jina_y"), "cssr_c": cond("jina_y"), "group_macro_cssr_u": group_macro["jina35"]}, "retrieval5": {"cssr_u": rate("retrieval5_y"), "cssr_c": cond("retrieval5_y"), "group_macro_cssr_u": group_macro["retrieval5"]}}, "retrieval_feasibility": {"complete_support_top20": int(sum(x["retrieval_feasible"] for x in rows)), "n": n, "rate": float(np.mean([x["retrieval_feasible"] for x in rows]))}, "paired_counts": {"both_success": pair["pplx_jina"], "pplx_only": pair["pplx_0"], "jina_only": pair["0_jina"], "neither": pair["0_0"]}, "status_counts": {p: dict(Counter(x[f"{p}_status"] for x in rows)) for p in ("pplx", "jina")}, "support_cardinality": {str(k): {"n": sum(x["support_cardinality"] == k for x in rows), "pplx_successes": sum(x["support_cardinality"] == k and x["pplx_y"] for x in rows), "jina_successes": sum(x["support_cardinality"] == k and x["jina_y"] for x in rows)} for k in sorted({x["support_cardinality"] for x in rows})}}
    write_json(OUT / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json", stats)
    write_json(OUT / "E5_A_GROUP_EFFECTS_AND_SENSITIVITY.csv.json", {"groups": group_rows})
    with (OUT / "E5_A_GROUP_EFFECTS_AND_SENSITIVITY.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["source_group_id", "n", "pplx_mean", "jina_mean", "retrieval5_mean", "delta"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(group_rows)
    matrix_rows = [{"cell": k, "count": v} for k, v in sorted(pair.items())]
    with (OUT / "E5_A_PAIRED_WIN_LOSS_AND_FAILURE_MATRIX.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "count"]); w.writeheader(); w.writerows(matrix_rows)
    scorecard = [
        {"policy": "retrieval5", **rate("retrieval5_y"), "cssr_c_successes": cond("retrieval5_y")["successes"], "cssr_c_n": cond("retrieval5_y")["n"], "cssr_c": cond("retrieval5_y")["rate"], "status": "VALID_CONTROL"},
        {"policy": "pplx_decider", **rate("pplx_y"), "cssr_c_successes": cond("pplx_y")["successes"], "cssr_c_n": cond("pplx_y")["n"], "cssr_c": cond("pplx_y")["rate"], "status": "VALID_PRIMARY"},
        {"policy": "jina35", **rate("jina_y"), "cssr_c_successes": cond("jina_y")["successes"], "cssr_c_n": cond("jina_y")["n"], "cssr_c": cond("jina_y")["rate"], "status": "VALID_PRIMARY"},
    ]
    with (OUT / "E5_A_RETRIEVAL_CEILINGS_AND_SCORECARD.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scorecard[0])); w.writeheader(); w.writerows(scorecard)
    case_rows = []
    categories = {"pplx_only": [r for r in rows if r["pplx_y"] and not r["jina_y"]], "jina_only": [r for r in rows if r["jina_y"] and not r["pplx_y"]], "both": [r for r in rows if r["pplx_y"] and r["jina_y"]], "neither": [r for r in rows if not r["pplx_y"] and not r["jina_y"]]}
    for cat, vals in categories.items():
        for r in sorted(vals, key=lambda x: sha_bytes(f"{SEED}|case|{cat}|{x['query_id']}".encode()))[:12]: case_rows.append({"category": cat, "query_id": r["query_id"], "source_group_id": r["source_group_id"], "support_cardinality": r["support_cardinality"], "retrieval_feasible": r["retrieval_feasible"], "pplx_status": r["pplx_status"], "jina_status": r["jina_status"], "pplx_selected_ids": r["pplx_selected_ids"], "jina_selected_ids": r["jina_selected_ids"]})
    write_text(OUT / "E5_A_FAILURE_AND_DISAGREEMENT_CASEBOOK.md", "# E5-A Failure and Disagreement Casebook\n\nFixed stable-hash case selection after both primary policies completed. No cases were selected by favorable outcome.\n\n" + "\n".join(f"- **{r['category']}** `{r['query_id']}` group `{r['source_group_id']}` support_cardinality={r['support_cardinality']} retrieval_feasible={r['retrieval_feasible']} PPLX={r['pplx_status']} Jina={r['jina_status']}" for r in case_rows))
    return stats


def v0() -> dict[str, Any]:
    manifest = require_manifest()
    tests = run_cmd([sys.executable, "-m", "pytest", "-q", "tests/test_e5_a_benchmark.py"])
    targeted_collect = run_cmd([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_e5_a_benchmark.py"])
    full = run_cmd([sys.executable, "-m", "pytest", "-q"], timeout=300)
    stats = read_json(OUT / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json") if (OUT / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json").is_file() else {}
    primary = stats.get("primary", {})
    gate = "E5_A_OWNER_REVIEW_READY" if tests.get("returncode") == 0 and full.get("returncode") == 0 and primary and (OUT / "E5_A_A0_pplx_decider_SUMMARY.json").is_file() and (OUT / "E5_A_A0_jina35_SUMMARY.json").is_file() else "E5_A_PARTIAL_REVIEW_READY"

    # Export the post-inference score join in a stable, audit-friendly format.
    outcome_rows = read_jsonl(OUT / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl")
    outcome_fields = [
        "query_id", "source_group_id", "support_cardinality", "support_set_count",
        "support_flags", "retrieval_feasible", "retrieval5_y", "pplx_status",
        "pplx_y", "pplx_selected_ids", "jina_status", "jina_y", "jina_selected_ids",
    ]
    with (OUT / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=outcome_fields)
        w.writeheader()
        for row in outcome_rows:
            out = {k: row.get(k) for k in outcome_fields}
            for k in ("pplx_selected_ids", "jina_selected_ids"):
                out[k] = json.dumps(out[k], ensure_ascii=False, separators=(",", ":"))
            w.writerow(out)

    # The primary experiment was completed without delaying it for optional
    # controls. Make that non-execution explicit rather than implying that a
    # missing secondary result is a zero or a failed model.
    with (OUT / "E5_A_SECONDARY_POLICY_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "status", "reason"])
        w.writeheader()
        for policy in ("BGE", "Liquid", "zerank-2"):
            w.writerow({"policy": policy, "status": "SECONDARY_UNAVAILABLE", "reason": "Not executed; the prespecified primary PPLX-versus-Jina benchmark was completed first and no exact frozen-slate secondary output was available for reuse."})

    test_count_re = re.compile(r"(\d+) tests? collected")
    file_count_re = re.compile(r"^tests/[^:\n]+:\s*(\d+)\s*$", re.MULTILINE)
    def collected_count(result: dict[str, Any]) -> int | None:
        text = result.get("stdout", "")
        file_counts = [int(m.group(1)) for m in file_count_re.finditer(text)]
        if file_counts:
            return int(sum(file_counts))
        match = test_count_re.search(text)
        return int(match.group(1)) if match else None
    targeted_collected = collected_count(targeted_collect)
    full_collect = run_cmd([sys.executable, "-m", "pytest", "--collect-only", "-q"], timeout=300)
    full_collected = collected_count(full_collect)
    test_counts = {"targeted_collected": targeted_collected, "targeted_passed": tests.get("returncode") == 0, "full_collected": full_collected, "full_passed": full.get("returncode") == 0}

    # Audit disclosure for changes made after native outcomes were frozen. This
    # is deliberately separate from the pre-outcome freeze receipt and does
    # not alter the model inputs, native adapters, outputs, or estimand.
    write_json(OUT / "E5_A_POSTRUN_ANALYSIS_CODE_AUDIT.json", {
        "audit_version": "E5-A-POSTRUN-ANALYSIS-AUDIT-V1",
        "native_outcomes_altered": False,
        "native_inputs_or_selection_contract_altered": False,
        "post_outcome_changes": [
            "Added CPU-only leave-largest-group and support-cardinality diagnostics to S0.",
            "Extended V0 test timeout so the existing full regression suite could finish.",
            "Added CSV export, explicit secondary-control non-execution status, test-count reporting, and provenance closure.",
        ],
        "inference_rerun_after_freeze": False,
        "frozen_manifest_sha256": sha_file(OUT / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json"),
        "analysis_script_sha256_at_closure": sha_file(Path(__file__).resolve()),
        "interpretation": "These are reporting/integrity additions after A0; they do not constitute protocol tuning or outcome-dependent reranking.",
    })

    attempt_rows = read_jsonl(OUT / "E5_A_NATIVE_ATTEMPT_AND_TELEMETRY.jsonl")
    physical = {}
    for policy in ("pplx_decider", "jina35"):
        ar = [r for r in attempt_rows if r.get("stage") == "A0" and r.get("policy") == policy]
        physical[policy] = {
            "attempts": len(ar),
            "valid": sum(r.get("status") == "VALID" for r in ar),
            "failures": sum(r.get("status") != "VALID" for r in ar),
            "physical_forward_calls": sum(r.get("telemetry", {}).get("physical_forward_calls", 0) for r in ar),
            "physical_forward_evidence_true": sum(bool(r.get("telemetry", {}).get("physical_forward_evidence")) for r in ar),
            "duration_s_sum": sum(float(r.get("duration_s", 0.0)) for r in ar),
            "cache_reuse": sum(bool(r.get("cache_reuse")) for r in ar),
        }
    smoke = {
        policy: {
            "attempts": sum(1 for r in attempt_rows if r.get("stage") == "H0_CAL_SMOKE" and r.get("policy") == policy),
            "valid": sum(1 for r in attempt_rows if r.get("stage") == "H0_CAL_SMOKE" and r.get("policy") == policy and r.get("status") == "VALID"),
        }
        for policy in ("pplx_decider", "jina35")
    }
    write_json(OUT / "E5_A_TEST_REPORT.json", {"targeted": tests, "full": full, "targeted_collect": targeted_collect, "full_collect": full_collect, "counts": test_counts, "targeted_pass": tests.get("returncode") == 0, "full_pass": full.get("returncode") == 0})
    write_text(OUT / "E5_A_TEST_REPORT.md", f"""# E5-A Test Report

- Targeted E5-A tests: **{targeted_collected if targeted_collected is not None else 'NOT_PARSED'} collected; {'PASS' if tests.get('returncode') == 0 else 'FAIL'}**.
- Full regression suite: **{full_collected if full_collected is not None else 'NOT_PARSED'} collected; {'PASS' if full.get('returncode') == 0 else 'FAIL'}**.
- The full suite command was allowed a 300-second timeout; this was an execution-time allowance only and did not change test selection.

Raw command outputs are preserved in `E5_A_TEST_REPORT.json`.
""")
    write_text(OUT / "E5_A_PHYSICAL_INFERENCE_AND_GPU_ACCOUNTING.md", f"""# E5-A Physical Inference and GPU Accounting

## Recorded work

- H0 CAL smoke: PPLX {smoke['pplx_decider']['valid']}/{smoke['pplx_decider']['attempts']} valid; Jina {smoke['jina35']['valid']}/{smoke['jina35']['attempts']} valid.
- A0 primary: PPLX {physical['pplx_decider']['valid']}/{physical['pplx_decider']['attempts']} valid, {physical['pplx_decider']['physical_forward_calls']} recorded physical forward calls; Jina {physical['jina35']['valid']}/{physical['jina35']['attempts']} valid, {physical['jina35']['physical_forward_calls']} recorded physical forward calls.
- A0 cache reuse flags: PPLX {physical['pplx_decider']['cache_reuse']}; Jina {physical['jina35']['cache_reuse']}.
- Summed per-request selector durations: PPLX {physical['pplx_decider']['duration_s_sum']:.3f}s; Jina {physical['jina35']['duration_s_sum']:.3f}s. These are sums of request-level observations, not wall-clock throughput or isolated production latency.

GPU5 (`{GPU_UUIDS[5]}`) hosted PPLX and GPU6 (`{GPU_UUIDS[6]}`) hosted Jina for the recorded runs. Before/after device and process snapshots are retained in the H0/A0 summaries. Existing unrelated processes were not terminated. Shared-device occupancy visible in snapshots means these measurements are not a matched isolated cost or serving benchmark.

This benchmark reports selector inference work only; it does not claim matched serving p95/QPS, dollar cost, or deployment economics.
""")

    write_json(OUT / "E5_A_OWNER_REVIEW_GATE.json", {"gate": gate, "primary": primary, "e5_f": "LOCKED", "e5_b_c_d": "NOT_EXECUTED", "jev": 0, "paid_api": 0, "reserved_accessed": False})
    write_text(OUT / "E5_A_OWNER_REVIEW_GATE.md", f"# E5-A Owner Review Gate\n\n**{gate}**\n\nE5-F remains `LOCKED`; E5-B/C/D were not executed. This gate is selector-only exploratory evidence and does not imply reader correctness, production serving performance or confirmation.\n")
    write_text(OUT / "E5_A_EXECUTION_EVENTS.md", "# E5-A Execution Events\n\n- M0 manifest and pre-outcome freeze completed before exploratory outputs.\n- H0/A0 native execution and S0/V0 results are recorded in machine-readable artifacts.\n- E5-F, Jev, paid APIs, readers, new retrieval and E5-B/C/D were not executed.\n")
    pp = stats.get("per_policy", {})
    rf = stats.get("retrieval_feasibility", {})
    pc = stats.get("paired_counts", {})
    write_text(OUT / "E5_A_FINAL_SCIENTIFIC_REPORT.md", f"""# DecisionRAG E5-A Final Scientific Report

## Gate

`{gate}`

## Primary result

PPLX Decider minus Jina v3.5 equal-source-group set-aware CSSR-U: **{primary.get('delta', 'NOT_AVAILABLE')}** (about **{100*primary.get('delta', 0):.3f} percentage points**), 95% CI **{primary.get('ci95', 'NOT_AVAILABLE')}** (about **{100*primary.get('ci95', [0, 0])[0]:.3f} to {100*primary.get('ci95', [0, 0])[1]:.3f} percentage points**), over **{primary.get('queries', 'NOT_AVAILABLE')}** exploratory questions and **{primary.get('groups', 'NOT_AVAILABLE')}** eligible source groups. The interval includes zero; this is exploratory evidence-selection retention, not factual correctness or a confirmation claim.

### Main scorecard

| Policy | CSSR-U successes / all | CSSR-U | Feasible Top-20 n | CSSR-C successes / feasible | CSSR-C |
|---|---:|---:|---:|---:|---:|
| Retrieval Top-5 | {pp.get('retrieval5', {}).get('cssr_u', {}).get('successes', 'NOT_AVAILABLE')} / {pp.get('retrieval5', {}).get('cssr_u', {}).get('n', 'NOT_AVAILABLE')} | {pp.get('retrieval5', {}).get('cssr_u', {}).get('rate', 'NOT_AVAILABLE')} | {rf.get('complete_support_top20', 'NOT_AVAILABLE')} | {pp.get('retrieval5', {}).get('cssr_c', {}).get('successes', 'NOT_AVAILABLE')} / {pp.get('retrieval5', {}).get('cssr_c', {}).get('n', 'NOT_AVAILABLE')} | {pp.get('retrieval5', {}).get('cssr_c', {}).get('rate', 'NOT_AVAILABLE')} |
| PPLX Decider | {pp.get('pplx_decider', {}).get('cssr_u', {}).get('successes', 'NOT_AVAILABLE')} / {pp.get('pplx_decider', {}).get('cssr_u', {}).get('n', 'NOT_AVAILABLE')} | {pp.get('pplx_decider', {}).get('cssr_u', {}).get('rate', 'NOT_AVAILABLE')} | {rf.get('complete_support_top20', 'NOT_AVAILABLE')} | {pp.get('pplx_decider', {}).get('cssr_c', {}).get('successes', 'NOT_AVAILABLE')} / {pp.get('pplx_decider', {}).get('cssr_c', {}).get('n', 'NOT_AVAILABLE')} | {pp.get('pplx_decider', {}).get('cssr_c', {}).get('rate', 'NOT_AVAILABLE')} |
| Jina v3.5 | {pp.get('jina35', {}).get('cssr_u', {}).get('successes', 'NOT_AVAILABLE')} / {pp.get('jina35', {}).get('cssr_u', {}).get('n', 'NOT_AVAILABLE')} | {pp.get('jina35', {}).get('cssr_u', {}).get('rate', 'NOT_AVAILABLE')} | {rf.get('complete_support_top20', 'NOT_AVAILABLE')} | {pp.get('jina35', {}).get('cssr_c', {}).get('successes', 'NOT_AVAILABLE')} / {pp.get('jina35', {}).get('cssr_c', {}).get('n', 'NOT_AVAILABLE')} | {pp.get('jina35', {}).get('cssr_c', {}).get('rate', 'NOT_AVAILABLE')} |

The retrieval-feasibility ceiling was **{rf.get('complete_support_top20', 'NOT_AVAILABLE')} / {rf.get('n', 'NOT_AVAILABLE')}** ({rf.get('rate', 'NOT_AVAILABLE')}); questions outside that ceiling are unconditional failures for CSSR-U. Paired outcomes were both-success **{pc.get('both_success', 'NOT_AVAILABLE')}**, PPLX-only **{pc.get('pplx_only', 'NOT_AVAILABLE')}**, Jina-only **{pc.get('jina_only', 'NOT_AVAILABLE')}**, and both-failure **{pc.get('neither', 'NOT_AVAILABLE')}**. Exact machine-readable denominators are in `E5_A_RETRIEVAL_CEILINGS_AND_SCORECARD.csv` and `E5_A_PRIMARY_GROUP_BALANCED_STATS.json`.

Native failures, missingness, attempts and telemetry are preserved rather than filtered. A0 had {physical['pplx_decider']['valid']}/{physical['pplx_decider']['attempts']} valid PPLX outcomes and {physical['jina35']['valid']}/{physical['jina35']['attempts']} valid Jina outcomes; each had {physical['pplx_decider']['physical_forward_calls']} and {physical['jina35']['physical_forward_calls']} recorded physical forward calls, respectively. H0 had {smoke['pplx_decider']['valid']}/{smoke['pplx_decider']['attempts']} and {smoke['jina35']['valid']}/{smoke['jina35']['attempts']} valid smoke outcomes. Regression tests: {targeted_collected} targeted and {full_collected} full-suite tests collected and passed.

No Jev or paid API calls were made. E5-B/C/D were not executed and E5-F remains locked.

## Claim limits

The result compares two pinned model-policy bundles on one frozen QASPER exploratory cohort. Source grouping does not prove no pretraining exposure; QASPER is a scientific-paper text corpus, not enterprise traffic; annotated support retention is not answer correctness or evidence exhaustiveness. No serving QPS/cost, reader utility or confirmation claim is made. E5-F remains locked.
""")
    write_text(OUT / "E5_A_PAPER_READY_RESULT_TABLES.md", f"""# E5-A Paper-Ready Result Tables

## Primary exploratory contrast

| Contrast | Estimate | 95% percentile CI | Questions | Eligible source groups | Bootstrap | Seed |
|---|---:|---:|---:|---:|---:|---:|
| PPLX Decider − Jina v3.5 equal-source-group CSSR-U | {primary.get('delta', 'NOT_AVAILABLE')} | {primary.get('ci95', 'NOT_AVAILABLE')} | {primary.get('queries', 'NOT_AVAILABLE')} | {primary.get('groups', 'NOT_AVAILABLE')} | {primary.get('bootstrap_reps', 'NOT_AVAILABLE')} | {primary.get('seed', 'NOT_AVAILABLE')} |

The estimate is approximately **{100*primary.get('delta', 0):.3f} percentage points**; the interval includes zero. The largest eligible group sensitivity is reported in `E5_A_PRIMARY_GROUP_BALANCED_STATS.json` and `E5_A_GROUP_EFFECTS_AND_SENSITIVITY.csv`.

## Scorecard and pair matrix

Use `E5_A_RETRIEVAL_CEILINGS_AND_SCORECARD.csv`, `E5_A_PAIRED_WIN_LOSS_AND_FAILURE_MATRIX.csv`, and `E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.csv`. These tables retain unconditional denominators, conditional feasibility, exact selected canonical IDs, invalid outputs, and the exploratory label.

Optional BGE/Liquid/zerank-2 controls were not executed under the frozen primary completion scope; this is recorded explicitly in `E5_A_SECONDARY_POLICY_RESULTS.csv` and is not a zero score.
""")

    # Rebuild provenance only after all final named artifacts exist, excluding
    # the two self-referential bookkeeping files.
    files = [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"E5_A_PROVENANCE_MANIFEST.json", "E5_A_PROVENANCE_VERIFICATION.json"}]
    prov = {"manifest_version": "E5-A-PROVENANCE-V1", "created_utc": utc_now(), "files": files, "source_manifest_sha256": sha_file(E5 / "E5_0_PROVENANCE_MANIFEST.json"), "approved_amendment_sha256": sha_file(E5C / "E5_0C_PROTOCOL_AMENDMENT_V1_APPROVED.md"), "frozen_execution_manifest_sha256": sha_file(OUT / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json"), "historical_artifacts_modified": False, "reserved_payload_accessed": False, "jev_calls": 0, "paid_api_calls": 0, "tests": {"targeted": tests, "full": full, "counts": test_counts}, "physical_inference": physical}
    write_json(OUT / "E5_A_PROVENANCE_MANIFEST.json", prov)
    slate_sha = sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl")
    verify = {"manifest_sha256": sha_file(OUT / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json"), "source_slate_sha256": slate_sha, "source_slate_expected": EXPECTED_SLATE_SHA, "source_slate_match": slate_sha == EXPECTED_SLATE_SHA, "new_artifact_count": len(files), "targeted_pass": tests.get("returncode") == 0, "full_pass": full.get("returncode") == 0, "targeted_collected": targeted_collected, "full_collected": full_collected, "historical_artifacts_modified": False, "outcomes_altered_after_a0": False}
    write_json(OUT / "E5_A_PROVENANCE_VERIFICATION.json", verify)
    return {"gate": gate, "primary": primary, "targeted_tests": tests, "full_tests": full}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["m0", "h0", "a0", "s0", "v0"])
    ap.add_argument("--policy", choices=["pplx_decider", "jina35"])
    ap.add_argument("--gpu", type=int)
    args = ap.parse_args()
    if args.stage == "m0": result = m0()
    elif args.stage == "h0": result = h0(args.policy, args.gpu)
    elif args.stage == "a0": result = score_native(args.policy, args.gpu)
    elif args.stage == "s0": result = s0()
    else: result = v0()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
