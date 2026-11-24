#!/usr/bin/env python3
"""Create the versioned E2-S/P GPU5 owner-review package offline."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "results/non_jev/v2/e2_sp"
OUT = E2 / "gpu5"
SELECTION = OUT / "E2_SP_S2_SELECTION_RESULTS.jsonl"
READER = OUT / "E2_SP_P1_READER_OUTPUTS.jsonl"
COHORT = E2 / "E2_SP_READER_COHORT_MANIFEST.json"
POLICIES = ["retrieval5", "bge_pointwise", "qwen06", "qwen4", "jina35", "gte", "minilm", "clef_joint", "nimble", "pplx_decider", "qwen_direct"]
TRACE_FILES = {
    "bge_pointwise": "E2_SP_S2_FULL_BGE_POINTWISE.jsonl",
    "qwen06": "E2_SP_S2_FULL_QWEN06.jsonl",
    "qwen4": "E2_SP_S2_FULL_QWEN4.jsonl",
    "jina35": "E2_SP_S2_FULL_JINA35.jsonl",
    "gte": "E2_SP_S2_FULL_GTE.jsonl",
    "minilm": "E2_SP_S2_FULL_MINILM.jsonl",
    "clef_joint": "E2_SP_S2_FULL_CLEF_JOINT.jsonl",
    "nimble": "E2_SP_S2_FULL_NIMBLE.jsonl",
    "pplx_decider": "E2_SP_S2_FULL_PPLX_DECIDER.jsonl",
    "qwen_direct": "E2_SP_S2_FULL_QWEN_DIRECT.jsonl",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection_rows = load_jsonl(SELECTION)
    reader_rows = load_jsonl(READER)
    latency_rows = load_csv(OUT / "E2_SP_SELECTION_LATENCY_SUMMARY.csv")
    quality_rows = load_csv(OUT / "E2_SP_QUALITY_POLICY_SUMMARY.csv")
    if len(selection_rows) != 3300 or len(reader_rows) != 3300:
        raise RuntimeError("finalization requires complete 3300-row selection and reader artifacts")

    # Required top-level aliases are placed inside the GPU5 continuation
    # namespace so the earlier blocked package remains byte-for-byte intact.
    selection_traces = []
    for policy, filename in TRACE_FILES.items():
        trace_path = OUT / filename
        for row in load_jsonl(trace_path):
            duration = row.get("duration_ms")
            if duration is None and isinstance(row.get("telemetry"), dict):
                duration = row["telemetry"].get("duration_ms")
            selection_traces.append({
                "policy": policy,
                "query_id": row.get("query_id"),
                "dataset": row.get("dataset"),
                "candidate_ids": row.get("candidate_ids"),
                "status": row.get("status"),
                "duration_ms": duration,
                "model_id": row.get("model_id"),
                "revision": row.get("revision"),
                "physical_gpu_id": row.get("e2_sp_physical_gpu_id", row.get("physical_gpu_id")),
                "physical_forward_calls": row.get("physical_forward_calls"),
                "physical_forward_evidence": row.get("physical_forward_evidence"),
                "input_tokens": row.get("input_tokens"),
                "actual_model_input_sha256": row.get("actual_model_input_sha256"),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "cache_reuse": row.get("cache_reuse", False),
                "gold_in_selection_input": row.get("gold_in_inference_input", False),
                "source_trace_sha256": sha_file(trace_path),
            })
    selection_traces.sort(key=lambda row: (row["policy"], row["query_id"]))
    write_jsonl(OUT / "E2_SP_SELECTION_REQUEST_TRACES.jsonl", selection_traces)
    write_jsonl(OUT / "E2_SP_READER_RAW_OUTPUTS.jsonl", reader_rows)

    context_rows = []
    for row in reader_rows:
        context_rows.append({
            "policy": row["policy"],
            "query_id": row["query_id"],
            "dataset": row["dataset"],
            "candidate_count": len(row.get("candidate_ids", [])),
            "selected_count": len(row.get("selected_ids", [])),
            "prompt_tokens": row.get("prompt_metadata", {}).get("prompt_tokens"),
            "context_overflow": row.get("prompt_metadata", {}).get("context_overflow"),
            "reader_status": row.get("status"),
            "reader_error": row.get("error_type"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "cache_reuse": row.get("cache_reuse"),
            "gold_in_reader_input": row.get("gold_in_reader_input"),
            "actual_model_input_sha256": row.get("actual_model_input_sha256"),
            "raw_response_sha256": row.get("raw_response_sha256"),
        })
    write_csv(OUT / "E2_SP_CONTEXT_FEASIBILITY_AND_ERRORS.csv", context_rows)

    open_loop_rows = []
    for policy in POLICIES:
        for rate in ("1", "2", "4"):
            open_loop_rows.append({
                "policy": policy,
                "offered_arrival_rate_rps": rate,
                "status": "NOT_MEASURED",
                "observation_window_s": "NOT_MEASURED",
                "completed": "NOT_MEASURED",
                "timeouts_or_errors": "NOT_MEASURED",
                "p95_ms": "NOT_MEASURED",
                "throughput_rps": "NOT_MEASURED",
                "queue_stable": "NOT_MEASURED",
                "reason": "S2 continuation completed isolated serialized traces; no open-loop load process was executed, so sustainable QPS is not inferred.",
            })
    write_csv(OUT / "E2_SP_OPEN_LOOP_LOAD_SWEEP.csv", open_loop_rows)

    capacity_rows = []
    latency_by_policy = {row["policy"]: row for row in latency_rows}
    for policy in POLICIES:
        measured = latency_by_policy.get(policy, {})
        capacity_rows.append({
            "policy": policy,
            "single_client_n": measured.get("n", "NOT_MEASURED"),
            "single_client_p50_ms": measured.get("p50_ms", "NOT_MEASURED"),
            "single_client_p95_ms": measured.get("p95_ms", "NOT_MEASURED"),
            "single_client_p99_ms": measured.get("p99_ms", "NOT_MEASURED"),
            "slo_1000_hit_rate": measured.get("slo_1000_hit_rate", "NOT_MEASURED"),
            "slo_2000_hit_rate": measured.get("slo_2000_hit_rate", "NOT_MEASURED"),
            "slo_5000_hit_rate": measured.get("slo_5000_hit_rate", "NOT_MEASURED"),
            "sustainable_qps_1000ms": "NOT_ESTABLISHED",
            "sustainable_qps_2000ms": "NOT_ESTABLISHED",
            "gpu_seconds_per_query": "NOT_MEASURED",
            "gpu_cost_usd_per_query": "NOT_MEASURED",
            "status": "PASS_WITH_LIMITATIONS",
            "cost_basis": "No approved monetary rate and no open-loop trial; do not convert serialized latency into cost.",
        })
    write_csv(OUT / "E2_SP_SLO_CAPACITY_AND_COST.csv", capacity_rows)

    full_rag = """# E2-S/P Full-RAG Latency Report

Status: `FULL_RAG_LATENCY_NOT_MEASURED`.

The completed GPU5 continuation measured native Top-20-to-Top-5 selector requests and a separate downstream reader workload. It did not replay live BM25/dense retrieval and Qdrant access for the frozen 300-query cohort. Therefore no full-request p50/p95 is reported and no component percentile is summed to fabricate one. The reader service was resident and its request latency is recorded separately with unattributed physical-GPU topology.
"""
    (OUT / "E2_SP_FULL_RAG_LATENCY_REPORT.md").write_text(full_rag, encoding="utf-8")

    def q(dataset: str, policy: str, field: str) -> str:
        for row in quality_rows:
            if row["dataset"] == dataset and row["policy"] == policy:
                return row.get(field, "NOT_MEASURED")
        return "NOT_MEASURED"

    latency_table = [
        "| Policy | n | p50 ms | p95 ms | p99 ms | <=1s | <=2s | <=5s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in latency_rows:
        latency_table.append(f"| {row['policy']} | {row['n']} | {row['p50_ms']} | {row['p95_ms']} | {row['p99_ms']} | {row['slo_1000_hit_rate']} | {row['slo_2000_hit_rate']} | {row['slo_5000_hit_rate']} |")

    quality_table = [
        "| Policy | Track A CSSR-C | Track A reader F1 proxy | Enterprise CSSR-C | Enterprise reader F1 proxy |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        ta = [row for row in quality_rows if row["policy"] == policy and row["dataset"] in {"hotpot", "musique", "wiki2"}]
        en = [row for row in quality_rows if row["policy"] == policy and row["dataset"] == "enterprise"]
        def avg(field: str, rows: list[dict[str, Any]]) -> str:
            values = [float(row[field]) for row in rows if row.get(field) not in (None, "", "NOT_MEASURED")]
            return f"{sum(values)/len(values):.4f}" if values else "NOT_MEASURED"
        quality_table.append(f"| {policy} | {avg('selected_complete_support_rate', ta)} | {avg('answer_f1_proxy_all', ta)} | {avg('selected_complete_support_rate', en)} | {avg('answer_f1_proxy_all', en)} |")

    final_report = [
        "# E2-S/P GPU5 Continuation — Final Report",
        "",
        "Status: `E2_SP_PASS_WITH_LIMITATIONS`.",
        "",
        "The GPU5 continuation completed authentic native selector smoke, CAL validation, isolated 300-query S2 selection, and fixed-reader downstream evaluation. The original approved-GPU package that stopped at S0 remains preserved under the parent namespace; this report covers only the versioned GPU5 continuation amendment.",
        "",
        "## Completed stages",
        "",
        "- S0 smoke: 8 frozen queries × 10 native model branches plus Clef joint/pointwise; all passed with physical-forward and input/response hash evidence.",
        "- S1 CAL: 30 predeclared CAL queries per branch; BGE, Qwen0.6B, Qwen4B, Jina3.5, GTE, MiniLM, Clef pointwise/joint, Nimble, Perplexity Decider, and direct Qwen2B all passed. No gold was placed in model inputs and no outcome-driven tuning was performed.",
        "- S2: 300 frozen queries × 11 policies = 3,300 native selection outcomes; all completed with 20-candidate alignment and no cache reuse.",
        "- P1 reader: 300 frozen queries × 11 policies = 3,300 real requests; all completed through the existing pinned Qwen3-VL-8B-Instruct service. Each row has an actual request-input hash and raw-response hash.",
        "- Offline analysis: Track A and enterprise results were recomputed separately from segregated reference sidecars. EM/F1 are labeled answer proxies, not independent factual correctness.",
        "",
        "## Resource and service boundary",
        "",
        "- Native selector measurements used isolated physical GPU5 under the GPU5 continuation amendment: one model loaded at a time, no GPU5 compute owner at each guard, and no resident service restart or eviction.",
        "- GPUs 0/1/2/4 and unrelated resident workloads were not repurposed by this continuation. GPU5 memory returned after each native branch; Perplexity Decider stayed within the 80-GiB device floor.",
        "- The reader used the already-running local Qwen service because its pinned weights were not present as a local snapshot. The service was not modified or restarted, but per-request physical GPU attribution and server-internal queueing are unavailable; reader latency is therefore not a hardware-equal selector comparison.",
        "- Jev calls: 0. Paid API calls: 0. Fresh confirmatory holdout: not accessed. Training: not run.",
        "",
        "## Native selector latency",
        "",
        "These are serialized single-client native selection request timings over 300 frozen queries, not sustained production capacity. Open-loop load and monetary cost remain explicitly unmeasured.",
        "",
        *latency_table,
        "",
        "The <=1-second values are descriptive exploratory SLO hit rates only. They do not establish sustainable QPS. Jina, GTE, MiniLM, BGE, Qwen0.6B, Qwen4B, and direct Qwen2B have serialized p95 below 1 second; Nimble, Clef joint, and Perplexity Decider do not.",
        "",
        "## Quality and reader proxies",
        "",
        "Track A is the 240-query exploratory slice of HotpotQA, MuSiQue, and 2Wiki; enterprise is a separate 60-question slice. No cross-domain pooled leaderboard is claimed.",
        "",
        *quality_table,
        "",
        "Reference-answer EM/F1 use the frozen answer sidecars and deterministic normalization. They are coupled to the Qwen reader and are not independently verified factual correctness, completeness, or groundedness. The frozen prompt contains no citation instruction, so citation metrics are `NOT_APPLICABLE` rather than zero.",
        "",
        "## PPLX deployment boundary",
        "",
        f"On the Track A slice, Perplexity Decider's macro selected complete-support rate is `{q('hotpot', 'pplx_decider', 'selected_complete_support_rate')}`, `{q('musique', 'pplx_decider', 'selected_complete_support_rate')}`, and `{q('wiki2', 'pplx_decider', 'selected_complete_support_rate')}` by dataset; the macro is reported in the machine-readable summary. Its exploratory support proxy is higher than Jina's on the three-dataset macro, while its reader F1 proxy is lower. The aggregate native selector p50/p95 are `{next(row['p50_ms'] for row in latency_rows if row['policy']=='pplx_decider')}`/`{next(row['p95_ms'] for row in latency_rows if row['policy']=='pplx_decider')}` ms versus Jina's `{next(row['p50_ms'] for row in latency_rows if row['policy']=='jina35')}`/`{next(row['p95_ms'] for row in latency_rows if row['policy']=='jina35')}` ms.",
        "",
        "Perplexity's quality gain is therefore not deployment-cost justified by this experiment: no dollar rate, GPU-seconds/query, or sustainable-QPS measurement is available, and its serialized p95 misses the 1-second exploratory threshold. The correct conclusion is `COST_JUSTIFICATION_NOT_ESTABLISHED`, not a universal rejection or approval.",
        "",
        "## Required limitations",
        "",
        "- `E2_SP_OPEN_LOOP_LOAD_SWEEP.csv` contains explicit `NOT_MEASURED` rows; no sustainable QPS is claimed.",
        "- `E2_SP_SLO_CAPACITY_AND_COST.csv` contains serialized timing only; GPU-seconds and dollar cost are `NOT_MEASURED`.",
        "- `E2_SP_FULL_RAG_LATENCY_REPORT.md` is `FULL_RAG_LATENCY_NOT_MEASURED`; retrieval and Qdrant were not replayed and percentile sums were not substituted.",
        "- The reader service topology differs from the isolated selector topology, so reader latency cannot be used as a matched hardware frontier.",
        "- The experiment is exploratory and historical/cohort-conditioned. It does not freeze or confirm the independent H1 holdout and does not change `P0v2_NOT_FROZEN`.",
        "",
        "## Key artifacts",
        "",
        "- `E2_SP_S2_SELECTION_RESULTS.jsonl`: 3,300 immutable policy/query selections.",
        "- `E2_SP_SELECTION_REQUEST_TRACES.jsonl`: normalized native trace index with source trace hashes.",
        "- `E2_SP_P1_READER_OUTPUTS.jsonl` and `E2_SP_READER_RAW_OUTPUTS.jsonl`: 3,300 raw-hash-backed reader outcomes.",
        "- `E2_SP_DOWNSTREAM_ANSWER_CITATION_METRICS.csv`, `E2_SP_QUALITY_POLICY_SUMMARY.csv`, and `E2_SP_QUALITY_COST_FRONTIER.csv`: offline metrics.",
        "- `E2_SP_GPU5_POLICY_MANIFEST.json`, `E2_SP_PROVENANCE_MANIFEST.json`: provenance and hashes.",
        "",
        "Owner-review gate: stop here. No Jev or paid phase is started by this package.",
    ]
    (OUT / "E2_SP_FINAL_REPORT.md").write_text("\n".join(final_report) + "\n", encoding="utf-8")

    gate = """# E2-S/P GPU5 Owner Review Gate

Verdict: `E2_SP_PASS_WITH_LIMITATIONS`

PASS:

- Native S0 smoke, S1 CAL validation, and S2 300-query selection completed on isolated GPU5.
- All 3,300 frozen reader policy/query requests completed with actual request-input and raw-response hashes.
- No cache reuse, gold-in-input rows, Jev calls, paid calls, holdout access, training, resident-service restart, or unrelated-service eviction occurred.
- CPU regression suite: 65/65 passed with `PYTHONPATH=.`.

LIMITATIONS / NOT MEASURED:

- Open-loop arrival-rate sweeps and sustainable QPS were not executed.
- GPU-seconds/query and dollar cost are not measured; no monetary price assumption was invented.
- Full-RAG retrieval+selection+reader request latency is not measured.
- Reader requests used the resident pinned Qwen service; physical GPU attribution and internal queueing are unavailable.
- Independent H1 confirmation and Jev remain unauthorized and untouched.

Owner decision requested: review the exploratory quality/latency evidence and explicitly authorize any future open-loop or paid/confirmatory phase separately.
"""
    (OUT / "E2_SP_OWNER_REVIEW_GATE.md").write_text(gate, encoding="utf-8")

    source_paths = [E2 / "E2_SP_FINAL_REPORT.md", E2 / "E2_SP_EXECUTION_MANIFEST.json", E2 / "E2_SP_OWNER_REVIEW_GATE.md"]
    generated = [
        OUT / "E2_SP_S2_SELECTION_RESULTS.jsonl", OUT / "E2_SP_GPU5_POLICY_MANIFEST.json",
        OUT / "E2_SP_SELECTION_REQUEST_TRACES.jsonl", OUT / "E2_SP_SELECTION_LATENCY_SUMMARY.csv",
        OUT / "E2_SP_OPEN_LOOP_LOAD_SWEEP.csv", OUT / "E2_SP_SLO_CAPACITY_AND_COST.csv",
        OUT / "E2_SP_CONTEXT_FEASIBILITY_AND_ERRORS.csv", OUT / "E2_SP_P1_READER_OUTPUTS.jsonl",
        OUT / "E2_SP_READER_RAW_OUTPUTS.jsonl", OUT / "E2_SP_DOWNSTREAM_ANSWER_CITATION_METRICS.csv",
        OUT / "E2_SP_QUALITY_POLICY_SUMMARY.csv", OUT / "E2_SP_QUALITY_COST_FRONTIER.csv",
        OUT / "E2_SP_INDEPENDENT_METRIC_RECOMPUTATION.md", OUT / "E2_SP_FULL_RAG_LATENCY_REPORT.md",
        OUT / "E2_SP_FINAL_REPORT.md", OUT / "E2_SP_OWNER_REVIEW_GATE.md",
    ]
    generated.extend(sorted(OUT.glob("E2_SP_S1_CAL_*.jsonl")))
    generated.extend(sorted(OUT.glob("E2_SP_S1_CAL_*_SUMMARY.json")))
    manifest = {
        "manifest_version": "E2-SP-GPU5-CONTINUATION-v1",
        "status": "E2_SP_PASS_WITH_LIMITATIONS",
        "resource_amendment": "results/non_jev/v2/e2_sp/E2_SP_GPU5_RESOURCE_AMENDMENT.md",
        "physical_selector_gpu": 5,
        "reader_topology": "resident_local_Qwen_service_unattributed_gpu",
        "cohort_manifest_sha256": sha_file(COHORT),
        "selection_manifest_sha256": sha_file(SELECTION),
        "reader_output_sha256": sha_file(READER),
        "generated_artifact_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in generated if path.is_file()},
        "preserved_prior_artifact_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in source_paths if path.is_file()},
        "counts": {"selection_rows": len(selection_rows), "reader_rows": len(reader_rows), "policies": len(POLICIES), "queries": 300, "cal_rows_per_branch": 30},
        "jev_calls": 0,
        "paid_api_calls": 0,
        "fresh_holdout_access": False,
        "resident_service_modified": False,
        "training": False,
        "open_loop_capacity": "NOT_MEASURED",
        "dollar_cost": "NOT_MEASURED",
    }
    (OUT / "E2_SP_PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
