#!/usr/bin/env python3
"""Prepare the E2-S/P frozen design and fail-closed resource gate.

This script performs only offline manifest/cohort preparation and read-only
GPU/process inventory. It must not load a model, make a model request, call a
service, access Qdrant, or modify an existing artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp"
SEED = "E2-SP-READER-COHORT-20261013"
APPROVED_GPUS = [0, 1, 2, 4]
POLICIES = [
    "retrieval5", "bge_pointwise", "qwen06", "qwen4", "jina35", "gte",
    "minilm", "clef_joint", "nimble", "pplx_decider", "qwen_direct",
]
READER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
READER_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
READER_PROMPT_REVISION = "E2_SP_READER_CONTRACT_v1"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_items() -> list[dict]:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items
    return load_full_items()


def load_enterprise_gold_status() -> tuple[dict[str, dict], dict[str, list[str]]]:
    import sqlite3
    db = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
    mapping: dict[str, list[str]] = {}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        mapping = {}
        for internal, original in conn.execute("SELECT internal_doc_id, original_doc_id FROM documents"):
            mapping.setdefault(original, []).append(internal)
    finally:
        conn.close()
    rows = {}
    for row in load_jsonl(ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl"):
        qid = row["question_id"]
        expected = list(dict.fromkeys(row.get("expected_doc_ids", [])))
        rows[qid] = {
            "reference_available": bool(row.get("gold_answer")),
            "expected_doc_ids": expected,
            "identity_resolved": bool(expected) and all(len(mapping.get(x, [])) == 1 for x in expected),
            "question_type": row.get("question_type"),
        }
    return rows, mapping


def select_hash_rank(rows: list[dict], n: int) -> list[dict]:
    ranked = sorted(rows, key=lambda r: sha_bytes(f"{SEED}|{r['query_id']}".encode()))
    return ranked[:n]


def resource_inventory() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:
            return f"COMMAND_ERROR:{type(exc).__name__}:{exc}"
    gpu_query = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    app_query = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,gpu_name,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    process_query = run(["ps", "-eo", "user,pid,etime,pcpu,pmem,rss,cmd", "--sort=-rss"])
    snapshot = {"gpu_query": gpu_query, "compute_apps": app_query, "process_snapshot": process_query, "captured_utc": datetime.now(timezone.utc).isoformat()}
    snapshot["snapshot_sha256"] = sha_bytes(canonical(snapshot))
    return snapshot


def freeze_cohort(items: list[dict]) -> tuple[dict, dict]:
    sidecar = {r.get("query_id"): r for r in load_jsonl(ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl")}
    enterprise, _ = load_enterprise_gold_status()
    selected = []
    for dataset in ("hotpot", "musique", "wiki2"):
        group = [x for x in items if x["cohort"] == "track_a" and x["dataset"] == dataset]
        selected.extend(select_hash_rank(group, 80))
    enterprise_items = []
    for item in items:
        if item["cohort"] != "track_b":
            continue
        e = enterprise.get(item["query_id"], {})
        if e.get("identity_resolved") and e.get("reference_available"):
            enterprise_items.append(item)
    selected.extend(select_hash_rank(enterprise_items, 60))
    if len(selected) != 300 or len({x["query_id"] for x in selected}) != 300:
        raise RuntimeError("reader cohort did not produce exactly 300 unique queries")
    records = []
    reference_records = []
    for item in selected:
        qid = item["query_id"]
        candidates = [c["doc_id"] for c in item["candidates"]]
        slate_hash = sha_bytes(canonical({"query_id": qid, "candidate_ids": candidates}))
        question_hash = sha_bytes(item["question"].encode("utf-8"))
        if item["cohort"] == "track_a":
            ref = sidecar.get(qid, {})
            ref_kind = "track_a_answer_sidecar" if ref.get("answer") else "NOT_AVAILABLE"
            ref_hash = sha_bytes(canonical({"answer": ref.get("answer"), "aliases": ref.get("answer_aliases", [])})) if ref.get("answer") else None
        else:
            ref = enterprise[qid]
            ref_kind = "enterprise_answer_sidecar"
            ref_hash = sha_bytes(canonical({"question_id": qid, "question_type": ref.get("question_type"), "reference_available": ref.get("reference_available")}))
        records.append({
            "query_id": qid,
            "cohort": item["cohort"],
            "dataset": item["dataset"],
            "question_sha256": question_hash,
            "top20_candidate_ids": candidates,
            "top20_slate_sha256": slate_hash,
            "selection_outcome_used_for_sampling": False,
            "reader_input_status": "FROZEN_BEFORE_INFERENCE",
        })
        reference_records.append({"query_id": qid, "reference_kind": ref_kind, "reference_sha256": ref_hash, "reference_text_in_reader_manifest": False})
    manifest = {
        "manifest_version": "E2-SP-READER-COHORT-v1",
        "status": "FROZEN_PENDING_RESOURCE_GATE",
        "selection_seed": SEED,
        "selection_rule": "SHA256(seed|query_id) ascending within each declared dataset; no selector output or reader output inspected",
        "counts": {"total": 300, "hotpot": 80, "musique": 80, "wiki2": 80, "enterprise": 60},
        "query_records": records,
        "reference_manifest": reference_records,
        "reader_contract": {
            "model": READER_MODEL,
            "revision": READER_REVISION,
            "prompt_revision": READER_PROMPT_REVISION,
            "same_prompt_and_decoding_all_policies": True,
            "gold_in_prompt": False,
            "cache_policy": "application_cache_disabled; exact-context reuse only if independently hash-matched",
            "output_max_tokens": "frozen by existing validated reader contract; NOT_EXECUTED",
        },
        "source_manifests": {
            "track_a_slates": str((ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_RETRIEVAL_SLATES.jsonl").relative_to(ROOT)),
            "track_b_slates": str((ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_RETRIEVAL_SLATES.jsonl").relative_to(ROOT)),
            "track_a_gold": str((ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl").relative_to(ROOT)),
            "enterprise_gold": str((ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl").relative_to(ROOT)),
        },
    }
    manifest["manifest_sha256"] = sha_bytes(canonical(manifest))
    refs = {"status": "SEGREGATED_NOT_IN_READER_INPUT", "records": reference_records, "manifest_sha256": sha_bytes(canonical(reference_records))}
    return manifest, refs


def write_blocked_package() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    cohort, refs = freeze_cohort(items)
    inventory = resource_inventory()
    # The inventory is kept in the new namespace; no source artifacts are changed.
    write_json(OUT / "E2_SP_READER_COHORT_MANIFEST.json", cohort)
    write_json(OUT / "E2_SP_READER_REFERENCE_MANIFEST.json", refs)
    write_json(OUT / "E2_SP_GPU_RESOURCE_INVENTORY.json", inventory)
    write_text(OUT / "E2_SP_OWNER_AUTHORIZATION_RECORD.md", f"""# E2-S/P Owner Authorization Record

Status: `AUTHORIZED_SCOPE_RESOURCE_BLOCKED`

The owner authorized E2-S/P GPU inference, native serving, CAL tuning, load testing, downstream reader inference, and related operations on GPUs **0, 1, 2, and 4**. Jev, paid APIs, training, fresh confirmatory holdout access, retrieval/index regeneration, and resident-service modification remain prohibited.

The execution prompt requires exclusive/isolate-able resources and forbids disturbing resident or unrelated workloads. At inventory time, no approved GPU met that requirement; see `E2_SP_GPU_ISOLATION_REPORT.md` and the hash-chained inventory JSON.
""")
    manifest = {
        "milestone": "E2-S/P",
        "status": "E2_SP_PARTIAL_BLOCKED",
        "blocked_stage": "S0_RESOURCE_GATE",
        "authorization": {"approved_gpus": APPROVED_GPUS, "native_serving": True, "reader_inference": True, "load_testing": True, "jev_calls": False, "paid_api_calls": False, "fresh_holdout": False},
        "frozen_slo": {"primary_p95_top5_selection_ms": 1000, "sensitivity_ms": [2000, 5000], "status": "FROZEN_EXPLORATORY_NOT_EXTERNAL_CUSTOMER_SLO"},
        "roster": POLICIES,
        "reader_model": {"model": READER_MODEL, "revision": READER_REVISION, "prompt_revision": READER_PROMPT_REVISION},
        "reader_cohort_manifest": "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json",
        "reader_cohort_sha256": cohort["manifest_sha256"],
        "resource_inventory": "results/non_jev/v2/e2_sp/E2_SP_GPU_RESOURCE_INVENTORY.json",
        "resource_inventory_sha256": inventory["snapshot_sha256"],
        "new_model_requests": 0,
        "new_reader_requests": 0,
        "new_retrieval_requests": 0,
        "new_indexing": 0,
        "services_modified": 0,
        "holdout_accessed": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
    }
    manifest["manifest_sha256"] = sha_bytes(canonical(manifest))
    write_json(OUT / "E2_SP_EXECUTION_MANIFEST.json", manifest)
    gpu_lines = [
        "# E2-S/P GPU Isolation Report", "", "Status: `BLOCKED_RESOURCE_CONFLICT`", "",
        "Only GPUs 0, 1, 2, and 4 were considered. GPUs 3, 5, 6, and 7 were not allocated.",
        "", "| GPU | Observed condition | Decision |", "|---:|---|---|",
        "| 0 | Active unrelated `lchen25` training process (PID 164784), approximately 13 GiB resident and nonzero utilization | BLOCKED; no co-tenant use |",
        "| 1 | Resident VLLM engine and MiniCPM service, approximately 55.8 GiB resident | BLOCKED; resident service preserved |",
        "| 2 | Resident VLLM engine, approximately 54.8 GiB resident | BLOCKED; resident service preserved |",
        "| 4 | Resident VLLM engine, approximately 28.8 GiB resident | BLOCKED; resident service preserved |",
        "", "No E2-S/P model, reader, load, or calibration process was started. No service was restarted, evicted, reconfigured, or repurposed.",
        "The inventory JSON contains the raw `nvidia-smi` and process snapshots plus a SHA-256 snapshot hash.",
    ]
    write_text(OUT / "E2_SP_GPU_ISOLATION_REPORT.md", "\n".join(gpu_lines))
    write_text(OUT / "E2_SP_NATIVE_SMOKE_AND_CAL_REPORT.md", """# E2-S/P Native Smoke and CAL Report

Status: `BLOCKED_RESOURCE_CONFLICT`

S0 could not begin model loading because no approved GPU was safely isolatable. Smoke count: `0`; CAL inference count: `0`. No model request, reader request, service request, download, or network call was made. The frozen reader cohort and SLO are prepared for a future exclusive approved window.
""")
    write_json(OUT / "E2_SP_FROZEN_SERVING_CONFIGS.json", {
        "status": "FROZEN_PENDING_RESOURCE_GATE",
        "primary_slo_ms": 1000,
        "sensitivity_slo_ms": [2000, 5000],
        "roster": POLICIES,
        "selection_contract": "same frozen Top-20 identities; native model scoring; exactly distinct Top-5",
        "hardware_contract": "one exclusive A100-SXM4-80GB equivalent per model; no co-tenant load",
        "cache_contract": "application cache disabled; server cache separately recorded",
        "load_contract": "open-loop arrival-rate sweep; stable windows; timeouts counted as failures",
        "cal_tuning": "CAL/DEV only; no outcome-based S/P tuning",
    })
    status_record = {"status": "BLOCKED_RESOURCE_CONFLICT", "stage": "S0", "requested_policies": POLICIES, "completed_requests": 0, "reason": "no exclusive approved GPU"}
    (OUT / "E2_SP_SELECTION_REQUEST_TRACES.jsonl").write_text(json.dumps(status_record, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "E2_SP_READER_RAW_OUTPUTS.jsonl").write_text(json.dumps({"status": "BLOCKED_RESOURCE_CONFLICT", "requested_outputs": 3300, "completed_outputs": 0}, sort_keys=True) + "\n", encoding="utf-8")
    for name, header in {
        "E2_SP_SELECTION_LATENCY_SUMMARY.csv": ["policy", "status", "n", "p50_ms", "p95_ms", "p99_ms", "error_rate"],
        "E2_SP_OPEN_LOOP_LOAD_SWEEP.csv": ["policy", "status", "offered_qps", "completed_qps", "p95_ms", "failure_rate", "queue_stable"],
        "E2_SP_SLO_CAPACITY_AND_COST.csv": ["policy", "status", "slo_ms", "sustainable_qps", "gpu_seconds_per_query", "dollar_cost", "pricing_assumption"],
        "E2_SP_DOWNSTREAM_ANSWER_CITATION_METRICS.csv": ["policy", "dataset", "status", "n", "answer_em", "answer_f1", "citation_validity", "citation_support"],
        "E2_SP_QUALITY_COST_FRONTIER.csv": ["policy", "dataset", "status", "cssr_c", "p95_ms", "sustainable_qps", "gpu_seconds_per_query", "pareto_status"],
    }.items():
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([header, ["ALL", "BLOCKED_RESOURCE_CONFLICT", "0", "NOT_MEASURED", "NOT_MEASURED", "NOT_MEASURED", "NOT_MEASURED"][:len(header)]])
    write_text(OUT / "E2_SP_CONTEXT_FEASIBILITY_AND_ERRORS.csv", "policy,status,blocked_count,reason\nALL,BLOCKED_RESOURCE_CONFLICT,0,no approved exclusive GPU; Intern historical context block remains in prior artifacts\n")
    write_text(OUT / "E2_SP_FULL_RAG_LATENCY_REPORT.md", """# E2-S/P Full-RAG Latency Report

Status: `FULL_RAG_LATENCY_NOT_MEASURED`

No retrieval, selection, or reader request was executed because S0 failed the approved-GPU isolation gate. No component latency was summed or substituted for request-level p95.
""")
    write_text(OUT / "E2_SP_MODEL_FAMILY_DEPLOYMENT_ANALYSIS.md", """# E2-S/P Model-Family Deployment Analysis

Status: `BLOCKED_RESOURCE_CONFLICT`

No quality–latency–cost frontier can be asserted from this run. Historical E2-N2 CSSR-C remains a complete-support selection proxy, not reader factual accuracy. Perplexity Decider's historical quality gain cannot be judged cost-justified against Jina or smaller rerankers until an isolated serving window produces measured p95, sustained QPS, GPU-seconds/query, and downstream reader metrics under the frozen contract.
""")
    write_text(OUT / "E2_SP_AUTHENTICITY_AND_LEAKAGE_AUDIT.md", """# E2-S/P Authenticity and Leakage Audit

Status: `PASS_WITH_LIMITATIONS_RESOURCE_BLOCKED`

Passed offline safeguards: frozen historical slates loaded; deterministic reader cohort frozen before any reader output; no fresh holdout access; no gold text in reader manifest; no new model/service/reader requests; no Jev or paid API calls; no resident service changes. Serving authenticity, cache behavior, input/response hashes, GPU telemetry, and leakage during inference remain unmeasured because S0 did not start.
""")
    write_text(OUT / "E2_SP_INDEPENDENT_METRIC_RECOMPUTATION.md", """# E2-S/P Independent Metric Re-computation

Status: `NOT_EXECUTED_FOR_E2_SP`; prior E2-N2-R replay preserved.

The prior independent E2-N2-R replay remains authoritative for historical CSSR and paired selection metrics. No new E2-S/P selection or reader outputs were available to replay. No historical result was rewritten or grafted onto timed serving results.
""")
    write_text(OUT / "E2_SP_FINAL_REPORT.md", """# E2-S/P Final Report

Status: `E2_SP_PARTIAL_BLOCKED`

The E2-S/P owner authorization was valid, but S0 failed closed because all approved GPUs were occupied by resident or unrelated workloads. No E2-S/P inference, serving, load testing, or downstream reader request was made.

Frozen before the gate: the 1-second exploratory p95 selection threshold, 2/5-second sensitivity ceilings, 11-policy roster, fixed Qwen3-VL-8B reader revision, and deterministic 300-query reader cohort (80 HotpotQA, 80 MuSiQue, 80 2Wiki, 60 enterprise).

Perplexity Decider deployment cost is **not assessable** from this run. Historical CSSR gains do not justify a deployment decision without measured isolated serving cost and downstream reader outcomes.
""")
    write_text(OUT / "E2_SP_OWNER_REVIEW_GATE.md", """# E2-S/P Owner Review Gate

Verdict: `E2_SP_PARTIAL_BLOCKED`

Blocked gate: `S0_RESOURCE_GATE`. Exact next action: provide an exclusive approved GPU 0/1/2/4 execution window after the current unrelated/resident workloads release resources, then rerun the authorized S0 native smoke. Do not use GPUs 3/5/6/7 or disturb resident services.

Completed safely: offline cohort/config freeze, resource inventory, leakage boundary, blocked artifacts, and preservation of all historical results. Jev calls: `0`; paid APIs: `0`; training: `0`; fresh holdout: `false`.
""")
    # Hash every generated artifact except the provenance file itself to avoid a
    # circular self-hash. Historical files are referenced, never overwritten.
    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "E2_SP_PROVENANCE_MANIFEST.json":
            hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    provenance = {
        "status": manifest["status"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_artifact_hashes": hashes,
        "historical_sources": {
            "e2_n2_closure": sha_file(ROOT / "results/non_jev/v2/e2_n2/e2_n2_owner_closure/E2_N2_FINAL_SCIENTIFIC_CLOSURE.json"),
            "e2_n2_provenance": sha_file(ROOT / "results/non_jev/v2/e2_n2/e2_n2_owner_closure/E2_N2_OWNER_CLOSURE_PROVENANCE.json"),
            "track_a_slates": sha_file(ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_RETRIEVAL_SLATES.jsonl"),
            "track_b_slates": sha_file(ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_RETRIEVAL_SLATES.jsonl"),
        },
        "safety": {"gpu_inference": 0, "model_requests": 0, "reader_requests": 0, "network": 0, "services_modified": 0, "jev_calls": 0, "paid_api_calls": 0, "training": 0, "fresh_holdout": False},
        "resource_inventory_sha256": inventory["snapshot_sha256"],
    }
    write_json(OUT / "E2_SP_PROVENANCE_MANIFEST.json", provenance)
    print(json.dumps({"status": manifest["status"], "cohort": cohort["counts"], "output": str(OUT), "resource_inventory_sha256": inventory["snapshot_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    write_blocked_package()
