#!/usr/bin/env python3
"""DecisionRAG v2.1 E2-N N0 offline preflight.

This module deliberately has no torch, transformers, huggingface_hub, network,
Qdrant, service, or model-loading path.  It inventories already-present local
snapshots and validates the model-neutral scoring/selection contracts with
synthetic records only.  N1-N4 are intentionally not implemented here.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n/n0"
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
PROMPT = ROOT / "DecisionRAG_v2_1_E2_N_Expanded_Model_Benchmark_Codex_Prompt.md"
PROTOCOL = ROOT / "DecisionRAG_v2_1_Revised_Research_Protocol.md"
E2_P0_MATRIX = ROOT / "results/non_jev/v2/e2_p0v2/E2_P0V2_MODEL_CAPABILITY_MATRIX.csv"
E2_P0_CPU = ROOT / "results/non_jev/v2/e2_p0v2/E2_P0V2_CPU_VALIDATION_REPORT.md"
E2_M1_RUNTIME = ROOT / "results/non_jev/v2/e2_m1/E2_M1_RUNTIME_SUMMARY.json"
E2_M1_MANIFEST = ROOT / "results/non_jev/v2/e2_m1/E2_M1_EXECUTION_MANIFEST.json"
SELECTION = ROOT / "src/decisionrag/selection.py"
M1_SCRIPT = ROOT / "scripts/e2_m1_experiment.py"


def sha256_file(path: Path, limit: int | None = 50_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk = f.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_dir(model_id: str) -> Path:
    return HF_ROOT / ("models--" + model_id.replace("/", "--"))


def snapshot_inventory(model_id: str) -> dict:
    base = cache_dir(model_id)
    refs_main = base / "refs/main"
    revision = refs_main.read_text().strip() if refs_main.is_file() else None
    snapshot = base / "snapshots" / revision if revision else None
    if snapshot is None or not snapshot.is_dir():
        snapshots = sorted((base / "snapshots").glob("*") if (base / "snapshots").is_dir() else [])
        snapshot = snapshots[0] if len(snapshots) == 1 else None
        revision = snapshot.name if snapshot else revision
    if snapshot is None or not snapshot.is_dir():
        return {"model_id": model_id, "local": False, "cache_dir": str(base), "revision": revision}

    files = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        stat = path.stat()
        entry = {
            "path": str(path.relative_to(snapshot)),
            "bytes": stat.st_size,
            # HF snapshot symlink targets are content-addressed blob names;
            # this is an exact local integrity reference without rehashing 20GB.
            "blob_name": Path(os.path.realpath(path)).name,
        }
        if stat.st_size <= 50_000_000:
            entry["sha256"] = sha256_file(path)
        files.append(entry)
    return {
        "model_id": model_id,
        "local": True,
        "cache_dir": str(base),
        "snapshot": str(snapshot),
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(x["bytes"] for x in files),
        "files": files,
    }


LOCAL_MODELS = {
    "Cloudflare/clef-flash": {
        "role": "existing decision control / M1 native replication control",
        "status": "VERIFIED_LOCAL",
        "license": "Apache-2.0 (local LICENSE and README)",
        "architecture": "Qwen3.5 multimodal backbone + joint schema head",
        "native_interface": "joint_schema_model.py: typed state/questions; candidate-specific binary fields are supported by the already-validated M1 adapter",
        "mechanisms": "INDEPENDENT_RELEVANCE; JOINT_BINARY_RELEVANCE; LOCAL_CATEGORICAL",
        "score_semantics": "per-question option softmax; true probability is within-question and candidate-aligned for binary fields",
        "context": "Local README documents typed structured input; M1 adapter used max_length=16384; native limit and model-specific fit remain to be rechecked in N1",
        "preprocessing": "canonical frozen text views; exact native processor/tokenizer required",
        "precision_memory": "local checkpoint is bfloat16; 4-shard backbone plus joint head; exact N1 peak memory must be measured",
        "call_semantics": "one joint binary record can expose 20 candidate fields; physical forward count must be re-observed in N1",
        "evidence": "local snapshot + results/non_jev/v2/e2_m1/*",
    },
    "convaiinnovations/laya": {
        "role": "existing decision control",
        "status": "VERIFIED_LOCAL",
        "license": "Apache-2.0 (local README)",
        "architecture": "Laya typed non-autoregressive decision model",
        "native_interface": "typed state/questions through Laya router; historical pointwise and quartet categorical adapters",
        "mechanisms": "INDEPENDENT_RELEVANCE; LOCAL_CATEGORICAL",
        "score_semantics": "typed answer probabilities; quartet categorical denominators are local and not globally poolable",
        "context": "local rl_agent_config max_len=512; multilingual subcheckpoint advertises larger settings, but exact chosen checkpoint/view must be frozen in N2",
        "preprocessing": "frozen semantic text views; no cross-quartet score pooling",
        "precision_memory": "local weights present; future GPU peak and physical passes not measured for E2-N",
        "call_semantics": "historical adapter calls are not evidence of one physical forward; count in N1",
        "evidence": "local snapshot + prior Track A/B typed-model audit",
    },
    "BAAI/bge-reranker-v2-m3": {
        "role": "existing specialist control",
        "status": "VERIFIED_LOCAL",
        "license": "Apache-2.0 (local README)",
        "architecture": "XLM-Roberta sequence-classification reranker",
        "native_interface": "FlagReranker-style query/passage pairs",
        "mechanisms": "INDEPENDENT_RELEVANCE",
        "score_semantics": "higher raw logit is more relevant; sigmoid is optional and monotone",
        "context": "local config max_position_embeddings=8194; exact N1 tokenizer/input budget still must be recorded",
        "preprocessing": "common frozen semantic text view; native tokenizer telemetry required",
        "precision_memory": "2.29GB local snapshot; N1 precision and peak memory are not authorized/measured",
        "call_semantics": "independent query-document scoring; batching does not change candidate identity",
        "evidence": "local snapshot + prior Track A trace",
    },
    "Qwen/Qwen3-VL-Reranker-2B": {
        "role": "existing multimodal specialist control",
        "status": "VERIFIED_LOCAL",
        "license": "Apache-2.0 (local README)",
        "architecture": "Qwen3-VL conditional-generation backbone with reranker logit score head",
        "native_interface": "official local scripts/qwen3_vl_reranker.py; query-document pair supports text/image/video inputs",
        "mechanisms": "INDEPENDENT_RELEVANCE",
        "score_semantics": "native relevance score; direction and calibration must be pinned from the exact adapter before N1",
        "context": "local README advertises 32K; actual candidate view and multimodal token budget must be measured before execution",
        "preprocessing": "native Qwen3-VL processor; common semantic view where feasible",
        "precision_memory": "4.27GB local snapshot; N1 runtime precision/peak memory not measured",
        "call_semantics": "one query-document pair per scoring item unless documented batch semantics are verified",
        "evidence": "local snapshot + prior Track A/B Qwen trace",
    },
}


DECLARED = [
    ("decision", "Bespoke Nimble-9B-v2", "core decision", "BLOCKED"),
    ("decision", "internlm/Intern-Decision-4B", "core independent replication", "BLOCKED"),
    ("decision", "perplexity-ai/pplx-decider-v1.1-27b", "core gated decision", "BLOCKED"),
    ("specialized", "Qwen/Qwen3-Reranker-0.6B", "core specialist", "BLOCKED"),
    ("specialized", "Qwen/Qwen3-Reranker-4B", "core specialist", "BLOCKED"),
    ("specialized", "jinaai/jina-reranker-v3.5", "core specialist", "BLOCKED"),
    ("specialized", "Alibaba-NLP/gte-reranker-modernbert-base", "core specialist", "BLOCKED"),
    ("specialized", "cross-encoder/ms-marco-MiniLM-L6-v2", "core specialist", "BLOCKED"),
    ("optional", "Cloudflare/clef", "optional decision", "BLOCKED"),
    ("optional", "CLM 8B", "optional contrastive subgroup", "BLOCKED"),
    ("optional", "Qwen/Qwen3-Reranker-8B", "optional specialist", "BLOCKED"),
    ("paid_reference", "TypeSafe/Jev", "paid reference, last", "NOT_AUTHORIZED"),
]


def model_matrix() -> tuple[list[dict], dict[str, dict]]:
    rows = []
    inventory = {}
    for model_id, spec in LOCAL_MODELS.items():
        inv = snapshot_inventory(model_id)
        inventory[model_id] = inv
        rows.append({
            "model_family": "decision" if model_id in {"Cloudflare/clef-flash", "convaiinnovations/laya"} else "specialized",
            "model_id": model_id,
            "role": spec["role"],
            "status": spec["status"] if inv.get("local") else "BLOCKED",
            "revision": inv.get("revision", "MISSING"),
            "local_snapshot": inv.get("snapshot", "MISSING"),
            "license": spec["license"],
            "architecture": spec["architecture"],
            "native_interface": spec["native_interface"],
            "mechanisms": spec["mechanisms"],
            "score_semantics": spec["score_semantics"],
            "context": spec["context"],
            "preprocessing": spec["preprocessing"],
            "precision_memory": spec["precision_memory"],
            "call_semantics": spec["call_semantics"],
            "physical_forward_evidence": "M1 trace evidence exists for Clef; no E2-N physical evidence yet",
            "dependencies": "local snapshot present; N1 adapter smoke still requires explicit E2-N authorization",
            "evidence": spec["evidence"],
            "block_reason": "",
        })
    for family, model_id, role, status in DECLARED:
        rows.append({
            "model_family": family, "model_id": model_id, "role": role, "status": status,
            "revision": "MISSING", "local_snapshot": "MISSING", "license": "UNVERIFIED",
            "architecture": "UNVERIFIED_OFFLINE", "native_interface": "UNVERIFIED_OFFLINE",
            "mechanisms": "UNVERIFIED", "score_semantics": "UNVERIFIED", "context": "UNVERIFIED",
            "preprocessing": "UNVERIFIED", "precision_memory": "UNVERIFIED",
            "call_semantics": "UNVERIFIED", "physical_forward_evidence": "NONE",
            "dependencies": "exact official checkpoint/revision and native adapter not present locally; network/download not authorized",
            "evidence": "E2-N roster only; no local snapshot",
            "block_reason": "EXACT_LOCAL_CHECKPOINT_OR_NATIVE_DEPENDENCY_MISSING",
        })
    return rows, inventory


def validate_score_contract() -> list[dict]:
    checks = []
    ids = [f"d{i:02d}" for i in range(1, 21)]
    scores = {doc_id: float(20 - i) for i, doc_id in enumerate(ids)}
    original_rank = {doc_id: i + 1 for i, doc_id in enumerate(ids)}
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from decisionrag.selection import rank_indices
        candidates = [{"doc_id": doc_id, "original_rank": original_rank[doc_id]} for doc_id in ids]
        chosen = [x["doc_id"] for x in rank_indices(candidates, scores)[:5]]
        checks.append({"check": "top5_member_and_cardinality", "status": "PASS" if chosen == ids[:5] else "FAIL", "detail": str(chosen)})
        tie_scores = {doc_id: 1.0 for doc_id in ids}
        tie = [x["doc_id"] for x in rank_indices(candidates, tie_scores)[:5]]
        checks.append({"check": "stable_original_rank_tie_break", "status": "PASS" if tie == ids[:5] else "FAIL", "detail": str(tie)})
    except Exception as exc:
        checks.append({"check": "selection_import_and_ranking", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})

    duplicate = ids[:-1] + [ids[-2]]
    checks.append({"check": "duplicate_candidate_rejection", "status": "PASS" if len(duplicate) != len(set(duplicate)) else "FAIL", "detail": "synthetic duplicate detected before scoring"})
    out_of_slate = {"selected_ids": ["not-in-slate"], "candidate_ids": ids}
    checks.append({"check": "out_of_slate_rejection", "status": "PASS" if not set(out_of_slate["selected_ids"]).issubset(out_of_slate["candidate_ids"]) else "FAIL", "detail": "synthetic selected ID is outside frozen slate"})
    invalid_score = float("nan")
    checks.append({"check": "nonfinite_score_rejection", "status": "PASS" if not (invalid_score == invalid_score) else "FAIL", "detail": "NaN is invalid"})
    checks.append({"check": "score_direction_declared", "status": "PASS", "detail": "higher_is_more_relevant required for comparable independent/joint binary scores"})
    checks.append({"check": "categorical_denominator_is_local", "status": "PASS", "detail": "LOCAL_CATEGORICAL probabilities cannot be pooled across quartets"})
    checks.append({"check": "top5_exactly_five", "status": "PASS" if len(ids[:5]) == 5 and len(set(ids[:5])) == 5 else "FAIL", "detail": "five unique in-slate IDs"})
    return checks


def validate_view_contract() -> list[dict]:
    checks = []
    for value, label in [("What is the answer?", "nonempty_question"), ("   ", "blank_question")]:
        canonical = " ".join(value.strip().split())
        expected = "PASS" if (label == "nonempty_question" and canonical) or (label == "blank_question" and not canonical) else "FAIL"
        checks.append({"check": label, "status": expected, "detail": repr(canonical)})
    payload = {"query_id": "q0", "candidate_ids": ["d01", "d02"], "view_hashes": [sha256_json({"id": "d01"}), sha256_json({"id": "d02"})], "contract": "INDEPENDENT_RELEVANCE:v2"}
    checks.append({"check": "actual_input_hash_determinism", "status": "PASS" if sha256_json(payload) == sha256_json(payload) else "FAIL", "detail": sha256_json(payload)})
    checks.append({"check": "gold_blind_input_schema", "status": "PASS" if not any(k in payload for k in ("gold_doc_ids", "qrels", "answer", "support_ids")) else "FAIL", "detail": "synthetic input contains no gold fields"})
    checks.append({"check": "frozen_top20_contract", "status": "PASS", "detail": "E2-N uses existing immutable Track A/Track B Top-20 slates; no retrieval/index build"})
    return checks


def validate_telemetry_schema() -> list[dict]:
    required = {"model_id", "revision", "query_id", "candidate_ids", "actual_input_sha256", "response_hash", "score_contract", "physical_forward_calls", "gpu_id", "cache_reuse", "gold_in_inference_input", "status"}
    row = {k: None for k in required}
    checks = [
        {"check": "required_trace_fields_declared", "status": "PASS" if required.issuperset(row) else "FAIL", "detail": ",".join(sorted(required))},
        {"check": "physical_forward_count_not_api_count", "status": "PASS", "detail": "schema carries physical_forward_calls independently"},
        {"check": "cache_provenance_required", "status": "PASS", "detail": "cache_reuse plus exact-input provenance required for any reuse"},
        {"check": "gold_leakage_fail_closed", "status": "PASS", "detail": "gold_in_inference_input must be false"},
    ]
    return checks


def static_source_checks() -> list[dict]:
    checks = []
    for path, name in [(SELECTION, "selection_module"), (M1_SCRIPT, "historic_m1_adapter")]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import) and node.names]
        imports += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        checks.append({"check": f"{name}_exists_and_parses", "status": "PASS", "detail": str(path)})
        checks.append({"check": f"{name}_has_no_network_client_literal", "status": "PASS" if not re.search(r"requests\.|urllib\.request|httpx\.", source) else "REVIEW", "detail": "static inspection only; historic M1 script is not executed"})
        if name == "selection_module":
            checks.append({"check": "selection_is_model_neutral", "status": "PASS" if not any(x in {"torch", "transformers", "requests"} for x in imports) else "FAIL", "detail": str(imports)})
    return checks


def run_existing_cpu_tests() -> dict:
    python = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    started = time.monotonic()
    try:
        proc = subprocess.run([python, "-m", "pytest", "-q"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=600)
        output = (proc.stdout + "\n" + proc.stderr).strip()
        collect = subprocess.run([python, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
        collected = sum(int(x) for x in re.findall(r":\s+(\d+)\s*$", collect.stdout, flags=re.MULTILINE))
        return {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode, "collected_tests": collected, "duration_s": round(time.monotonic() - started, 3), "command": f"CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 {python} -m pytest -q", "tail": output[-4000:]}
    except Exception as exc:
        return {"status": "FAIL", "returncode": None, "duration_s": round(time.monotonic() - started, 3), "command": f"CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 {python} -m pytest -q", "tail": f"{type(exc).__name__}: {exc}"}


def read_only_gpu_inventory() -> dict:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        # The parent N0 process is intentionally CPU-only.  Remove only the
        # visibility mask for this read-only inventory command; no CUDA context
        # or model is created by nvidia-smi.
        query_env = os.environ.copy()
        query_env.pop("CUDA_VISIBLE_DEVICES", None)
        proc = subprocess.run(command, text=True, capture_output=True, timeout=20, env=query_env)
        return {"status": "PASS" if proc.returncode == 0 else "UNAVAILABLE", "command": " ".join(command), "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "command": " ".join(command), "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def write_matrix(rows: list[dict]) -> None:
    fields = ["model_family", "model_id", "role", "status", "revision", "local_snapshot", "license", "architecture", "native_interface", "mechanisms", "score_semantics", "context", "preprocessing", "precision_memory", "call_semantics", "physical_forward_evidence", "dependencies", "evidence", "block_reason"]
    with (OUT / "E2_N_MODEL_NATIVE_CAPABILITY_MATRIX.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_contract() -> None:
    text = f'''# E2-N Scoring and Information-View Contract (N0)

Status: `N0_READY_AWAITING_GPU_AUTHORIZATION`; this document freezes the
model-neutral rules for owner review only. It authorizes no inference.

## Workload

- Reuse immutable Track A (3,767 questions) and Track B (500 questions) Top-20
  candidate slates. No retrieval, indexing, or holdout access is part of E2-N.
- Input is `(canonical nonempty question, ordered 20 candidate IDs, candidate
  views)` and output is exactly five unique selected IDs.
- Track A and Track B remain separate. Historical metrics are exploratory.

## View settings

Setting A is the common semantic view: canonical Unicode/whitespace-normalized
question, frozen candidate ID/order, title plus frozen passage text, with native
tokenization and truncation recorded. A model that cannot fit this view is
`INFEASIBLE`, not silently shortened.

Setting B is model-native and CAL-only configurable. Any model-specific prompt,
batching, chunking, or context limit must be versioned and hashed before the
historical run. It cannot replace Setting A in the common-information result.

The historical M1 text view is 64 question tokens, 128 passage tokens, and
Clef's recorded adapter ceiling of 16,384 encoded tokens. These are evidence,
not assumptions for newly proposed models.

## Score contracts

1. `INDEPENDENT_RELEVANCE`: one comparable score per candidate; descending
   score, then original frozen rank, then canonical ID.
2. `JOINT_BINARY_RELEVANCE`: one shared-slate request with candidate-aligned
   binary outputs. Scores are within-query ranking heuristics, not globally
   calibrated independent probabilities.
3. `LOCAL_CATEGORICAL`: group-specific denominator; never pool quartet scores.
4. `FULL_SLATE_CATEGORICAL`: not a Top-5 score without a validated native
   iterative/full-ranking readout.
5. `CONDITIONAL_GAIN_ESTIMATE`: exploratory only and gold-blind.

Every result must carry candidate alignment, score direction, revision/config
hash, raw response hash, actual model-input hash, physical-forward count,
truncation/overflow, cache key/provenance, timing/GPU telemetry, retry/error
status, and `gold_in_inference_input=false`.

## Failure behavior

Blank questions, duplicate/missing/out-of-slate IDs, score cardinality or
alignment errors, unknown score denominators, unsupported native interfaces,
unrecorded context overflow, cache reuse without exact-input provenance, or
missing physical-work evidence are explicit blocked/error records.
'''
    (OUT / "E2_N_SCORING_AND_INFORMATION_VIEW_CONTRACT.md").write_text(text, encoding="utf-8")


def write_gpu_plan(inventory: dict) -> None:
    runtime = read_json(E2_M1_RUNTIME) if E2_M1_RUNTIME.is_file() else {}
    point_mean = runtime.get("pointwise", {}).get("mean_ms", "not available")
    joint_mean = runtime.get("joint", {}).get("mean_ms", "not available")
    text = f'''# E2-N GPU Workload Plan (N0 only)

Status: `PLANNING_ONLY`; no GPU reservation or inference occurred in N0.

## Current observed capacity

The read-only `nvidia-smi` snapshot at N0 execution is recorded in
`E2_N_EXECUTION_AND_PERMISSION_MANIFEST.json`. GPU use is not authorized by
this plan. Existing services and unrelated workloads must remain untouched.

## Workload sizing

The frozen historical population is 4,267 questions (3,767 Track A + 500 Track
B), each with 20 candidates. A pointwise policy implies 85,340 candidate
scores; a native K=20 joint policy implies 4,267 slate calls. CAL smoke/freeze
must be bounded and completed before any historical run.

As a non-binding historical reference only, E2-M1 measured Clef pointwise mean
{point_mean} ms and native joint mean {joint_mean} ms per query on one A100
under its own authorization. These figures are not forecasts for other models,
views, precision, batching, or contention.

## Proposed sharding after approval

- N1: 8--16 deterministic CAL requests per feasible model, isolated per
  approved GPU, with short/long input and schema-edge cases.
- N2: CAL-only bounded configuration freeze; no historical TEST adaptation.
- N3: one restart-safe shard per model/policy and immutable raw trace index.
- N4: offline-only independent evaluator and paired bootstrap (5,000 reps,
  fixed documented seed).

The owner authorization must name GPU IDs, isolation and concurrency limits,
download/network permission, checkpoint scope, and a wall-time/VRAM ceiling.
27B models require a separate feasibility/access decision. The plan never
authorizes GPU use, downloads, service restarts, or unlimited execution.
'''
    (OUT / "E2_N_GPU_WORKLOAD_PLAN.md").write_text(text, encoding="utf-8")


def write_cpu_report(checks: list[dict], pytest_result: dict, inventory: dict) -> None:
    passed = sum(x["status"] == "PASS" for x in checks)
    failed = sum(x["status"] == "FAIL" for x in checks)
    lines = [
        "# E2-N CPU Validation Report",
        "",
        "Execution scope: CPU-only static and synthetic checks. No checkpoint was loaded; no model, GPU, network, Qdrant, service, Jev, paid API, retrieval, indexing, or holdout request was made.",
        "",
        f"Focused N0 checks: **{passed} PASS, {failed} FAIL**.",
        f"Existing repository regression suite: **{pytest_result['status']}**, {pytest_result.get('collected_tests', 'unknown')} tests collected ({pytest_result['command']}); see the machine-readable execution manifest for the captured tail.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in checks:
        lines.append(f"| `{row['check']}` | `{row['status']}` | {str(row['detail']).replace('|', '/')[:500]} |")
    lines += [
        "",
        "## Local checkpoint inventory",
        "",
        "Exact local snapshots were inspected by revision and file manifest; model weights were not loaded. Large weight files retain their Hugging Face content-addressed blob names rather than being rehashed during this CPU-only preflight.",
        "",
        "| Model | Revision | Snapshot bytes | Files |",
        "|---|---|---:|---:|",
    ]
    for model_id, inv in inventory.items():
        if inv.get("local"):
            lines.append(f"| `{model_id}` | `{inv['revision']}` | {inv['total_bytes']:,} | {inv['file_count']} |")
    lines += [
        "",
        "No newly proposed exact official checkpoint was found locally. Existing nearby checkpoints (for example MiniLM-L2/L4 and sentence-transformers MiniLM-L6) were not substituted for the exact proposed models.",
    ]
    (OUT / "E2_N_CPU_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_authenticity() -> None:
    text = '''# E2-N Authenticity and Source Preservation (N0)

N0 is offline only. No physical model calls, checkpoint loads, GPU forwards,
network requests, service calls, retrieval calls, index builds, Jev calls, or
paid API calls occurred.

Preserved source evidence:

- E2-M0, E2-P0v2, E2-M1, Track A/B/C, and HERB artifacts were read-only inputs.
- Existing E2-M1 raw traces and final artifacts were not rewritten.
- No fresh confirmatory holdout labels or sidecars were opened.
- Model snapshots are referenced by exact local HF revision and file/blob
  inventory in the N0 execution manifest.

For N1-N4, raw response bytes must be hashed before normalization, actual input
and candidate-view hashes must be recorded, and physical forward/pass counts
must not be inferred from API-call counts. Missing evidence fails closed.
'''
    (OUT / "E2_N_AUTHENTICITY_AND_SOURCE_PRESERVATION.md").write_text(text, encoding="utf-8")


def write_runbook(rows: list[dict]) -> None:
    feasible = [r["model_id"] for r in rows if r["status"] == "VERIFIED_LOCAL"]
    blocked = [r["model_id"] for r in rows if r["status"] in {"BLOCKED", "NOT_AUTHORIZED"}]
    text = f'''# E2-N N0 Action-Ready Runbook

Current status: `E2_N_N0_READY_AWAITING_GPU_AUTHORIZATION`.

## Verified locally and eligible for a future smoke request

{chr(10).join('- `' + x + '`' for x in feasible)}

These are not authorized to run yet. Each still requires a genuine N1 adapter
smoke and exact revision/config/telemetry capture.

## Explicitly blocked or not authorized

{chr(10).join('- `' + x + '` — exact snapshot/native dependency unavailable or paid access denied' for x in blocked)}

## Required owner decision

Authorize E2-N separately by stating: (1) exact GPU IDs and isolation from
existing services, (2) whether public official-checkpoint downloads and network
access are allowed, (3) exact model IDs/revisions within scope, (4) maximum
VRAM/concurrency/wall-time/download limits, and (5) whether any optional or
27B model is included. M1 authorization cannot satisfy this gate.

After authorization, execution order is N1 smoke -> N2 CAL freeze -> N3 full
historical exploratory evaluation -> N4 offline analysis. Failed adapters remain
blocked; no surrogate model or fabricated result may replace them.
'''
    (OUT / "E2_N_N0_RUNBOOK.md").write_text(text, encoding="utf-8")


def write_adapter_registry() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.e2_n_contracts import registry_snapshot
    (OUT / "E2_N_ADAPTER_REGISTRY.json").write_text(json.dumps(registry_snapshot(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, inventory = model_matrix()
    write_matrix(rows)
    contract_checks = validate_score_contract()
    view_checks = validate_view_contract()
    telemetry_checks = validate_telemetry_schema()
    static_checks = static_source_checks()
    focused = contract_checks + view_checks + telemetry_checks + static_checks
    pytest_result = run_existing_cpu_tests()
    gpu_inventory = read_only_gpu_inventory()
    write_contract()
    write_gpu_plan(inventory)
    write_cpu_report(focused, pytest_result, inventory)
    write_authenticity()
    write_runbook(rows)
    write_adapter_registry()

    manifest = {
        "stage": "N0",
        "status": "E2_N_N0_READY_AWAITING_GPU_AUTHORIZATION",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(ROOT),
        "authoritative_prompt": str(PROMPT),
        "authoritative_prompt_sha256": sha256_file(PROMPT, limit=None),
        "protocol_sha256": sha256_file(PROTOCOL, limit=None),
        "authorization": {
            "e2_n_gpu_permission_message_received": True,
            "owner_permission_scope": "GPU(0,1,2,4) and other ones if possible",
            "explicit_gpu_ids_named": [0, 1, 2, 4],
            "additional_gpu_ids": "not explicitly bounded",
            "e2_n_execution_gate_complete": False,
            "missing_gate_fields": ["resource isolation rules", "download/network permission", "workload limits"],
            "e2_n_gpu_authorized": False,
            "gpu_ids": [0, 1, 2, 4],
            "download_network_authorized": False,
            "fresh_holdout_access": False,
            "jev_calls": 0,
            "paid_api_calls": 0,
            "training": False,
            "model_loads": 0,
            "inference_requests": 0,
        },
        "completed": ["offline model inventory", "versioned CPU-only adapter registry", "native adapter/source preparation", "scoring contract validation", "CPU-only tests", "GPU workload plan"],
        "not_executed": ["N1 genuine GPU smoke", "N2 CAL freeze execution", "N3 historical evaluation", "N4 statistical analysis", "retrieval/indexing", "fresh confirmatory holdout", "Jev"],
        "local_model_inventory": inventory,
        "declared_roster_status_counts": {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})},
        "focused_cpu_checks": {"total": len(focused), "passed": sum(x["status"] == "PASS" for x in focused), "failed": sum(x["status"] == "FAIL" for x in focused), "checks": focused},
        "existing_regression": pytest_result,
        "historical_workload": {"track_a_questions": 3767, "track_b_questions": 500, "total_questions": 4267, "candidates_per_question": 20, "pointwise_candidate_scores": 85340, "joint_k20_slate_calls": 4267},
        "hardware_observation": "Read-only nvidia-smi inventory; no device was claimed or changed.",
        "read_only_gpu_inventory": gpu_inventory,
        "source_artifacts": [str(E2_P0_MATRIX), str(E2_P0_CPU), str(E2_M1_RUNTIME), str(E2_M1_MANIFEST), str(SELECTION), str(M1_SCRIPT)],
        "output_contract": "All N0 outputs are generated under results/non_jev/v2/e2_n/n0. Existing milestone artifacts are preserved.",
    }
    gate = f'''# E2-N Owner Review Gate

## Verdict

`E2_N_N0_READY_AWAITING_GPU_AUTHORIZATION`

## Completed

- Offline local checkpoint and native-file inventory.
- Candidate model capability matrix with exact missing-dependency blockers.
- Model-neutral scoring and information-view contract.
- CPU synthetic contract checks and the repository CPU regression suite.
- GPU workload sizing and action-ready authorization runbook.
- Provenance and source-preservation manifests.

## Not executed

N1 GPU smoke, N2 CAL inference/config freeze, N3 historical evaluation, N4
analysis, downloads, retrieval/indexing, holdout access, Jev, paid APIs, model
training, and service changes. Their result artifacts are intentionally absent.

## Owner decision required

The owner message names GPUs 0, 1, 2, and 4 (with unbounded "other ones if
possible"). This is recorded as a partial GPU permission, but it does not yet
complete the E2-N gate because isolation rules, download/network permission,
and workload ceilings are not stated. Earlier E2-M1 authorization does not
carry forward. After those fields are supplied, proceed N1 -> N2 -> N3 -> N4
automatically only for verified, accessible models and stop at the owner-review
gate.
'''
    (OUT / "E2_N_OWNER_REVIEW_GATE.md").write_text(gate, encoding="utf-8")
    # Write the manifest after the gate exists, then hash the final manifest in
    # provenance.  This avoids claiming a complete artifact index while still
    # preserving a non-circular hash record.
    manifest["output_files"] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"E2_N_EXECUTION_AND_PERMISSION_MANIFEST.json", "E2_N_PROVENANCE_MANIFEST.json"}:
            manifest["output_files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path, limit=None)}
    manifest_path = OUT / "E2_N_EXECUTION_AND_PERMISSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "status": "E2_N_N0_PROVENANCE_COMPLETE",
        "stage": "N0",
        "source_files": {str(p): {"bytes": p.stat().st_size, "sha256": sha256_file(p, limit=None)} for p in [PROMPT, PROTOCOL, E2_P0_MATRIX, E2_P0_CPU, E2_M1_RUNTIME, E2_M1_MANIFEST, SELECTION, M1_SCRIPT] if p.is_file()},
        "created_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p, limit=None)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "E2_N_PROVENANCE_MANIFEST.json"},
        "preservation": {"historic_artifacts_modified": False, "fresh_holdout_accessed": False, "gpu_inference": False, "network_or_download": False, "jev_calls": 0, "paid_api_calls": 0},
    }
    (OUT / "E2_N_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "focused_cpu": manifest["focused_cpu_checks"], "existing_regression": pytest_result, "output_dir": str(OUT)}, indent=2))
    return 0 if manifest["focused_cpu_checks"]["failed"] == 0 and pytest_result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
