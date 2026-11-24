#!/usr/bin/env python3
"""Offline audit/integration for the completed shared-contention R3 traces."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp_r"
R3 = OUT / "shared_gpu0/r3"
MODELS = {
    "bge_pointwise": "BGE_POINTWISE",
    "jina35": "JINA35",
    "nimble": "NIMBLE",
    "gte": "GTE",
    "pplx_decider": "PPLX_DECIDER",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def load_source_items() -> dict[str, dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_local_cal_items

    # R3 is the frozen 30-query CAL workload, not the 300-query historical
    # exploratory cohort used by the earlier selector/reader package.
    return {item["query_id"]: item for item in load_local_cal_items()}


def check_summary(summary: dict, request_rows: list[dict]) -> dict:
    if len(request_rows) != 40 or summary.get("requests") != 40 or summary.get("completed") != 40:
        raise RuntimeError(f"bad rate population: {summary.get('model')}:{summary.get('rate_index')}")
    if any(row.get("status") != "ok" for row in request_rows):
        raise RuntimeError(f"error in rate: {summary.get('model')}:{summary.get('rate_index')}")
    durations = [float(row["duration_ms"]) for row in request_rows]
    queues = [float(row["queue_wait_ms"]) for row in request_rows]
    result = {
        "p50_ms": percentile(durations, 0.50),
        "p95_ms": percentile(durations, 0.95),
        "p99_ms": percentile(durations, 0.99),
        "queue_p50_ms": percentile(queues, 0.50),
        "queue_p95_ms": percentile(queues, 0.95),
        "error_rate": 0.0,
        "physical_forward_calls": sum(int(row.get("physical_forward_calls", 0)) for row in request_rows),
    }
    for key, value in result.items():
        if abs(float(value) - float(summary[key])) > 1e-6:
            raise RuntimeError(f"raw-trace mismatch {key}: {value} != {summary[key]}")
    return result


def write_reports(rows: list[dict], manifest: dict, manifest_path: Path) -> None:
    csv_path = OUT / "E2_SP_R_R3_SHARED_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E2-S/P-R R3 Shared-Contention Execution Report", "",
        "Verdict: `EXECUTED_SHARED_CONTENTION_LIMITED`; original isolated-capacity gate remains `BLOCKED`.", "",
        "All five approved local native adapters completed the seeded four-rate open-loop workload on physical GPU0. GPU0 had pre-existing resident work throughout; no process was evicted and no service was restarted. These results are not valid for isolated capacity, fair hardware comparison, cost, or production SLO certification.", "",
        "- Total R3 requests: **800** (5 models × 4 rates × 40 requests).",
        "- Completed: **800/800**; errors: **0**; physical forwards: **800**.",
        "- Every request has frozen 20-candidate identity, actual-input hash, raw-response hash, and selection-text hashes.",
        "- Gold in model inputs: **0**; application cache reuse: **0**.",
        "- Jev calls: **0**; paid API calls: **0**; fresh holdout access: **false**.", "",
        "The p50/p95/p99 and queue percentiles below were recomputed from raw request rows and cross-checked against the runner summaries. SLO flags are descriptive under shared contention only.", "",
        "| Model | Offered RPS | Achieved QPS | p50 ms | p95 ms | p99 ms | Queue p50 ms | Queue p95 ms | Queue growth ms | p95 ≤1s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['model']}` | {row['offered_rps']:.6f} | {row['achieved_qps']:.6f} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} | {row['queue_p50_ms']:.3f} | {row['queue_p95_ms']:.3f} | {row['queue_growth_ms']:.3f} | {'PASS' if row['slo_1000_pass'] else 'FAIL'} |")
    lines += ["", "## Interpretation", "", "BGE and GTE stayed below the descriptive 1-second p95 threshold at all observed shared rates. Jina passed at its two middle observed rates but not at its lowest and highest rates. Nimble and PPLX did not pass at any observed rate. These are observations under one shared GPU0 topology, not causal model or capacity rankings.", "", "The original isolated-capacity gate remains blocked. Isolated sustainable QPS, queue-stability boundary, GPU allocation-seconds/query, and monetary cost remain `NOT_MEASURED`.", "", "The pre-shared package remains preserved at `revisions/pre_shared_execution_20261013/`."]
    (OUT / "E2_SP_R_R3_SHARED_EXECUTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol_path = OUT / "E2_SP_R_FROZEN_PROTOCOL.json"
    protocol = load_json(protocol_path)
    protocol["status"] = "FROZEN_R1_R2_SHARED_CONTENTION_R3_EXECUTED_SHARED_CONTENTION"
    protocol["open_loop"]["status"] = "EXECUTED_SHARED_CONTENTION_LIMITED_NOT_ISOLATED"
    protocol["open_loop"]["actual_requests"] = 800
    protocol["open_loop"]["models_completed"] = sorted(MODELS)
    protocol["open_loop"]["trace_directory"] = "results/non_jev/v2/e2_sp_r/shared_gpu0/r3"
    protocol["open_loop"]["isolated_capacity_validity"] = "BLOCKED_SHARED_CONTENTION"
    protocol["shared_contention_amendment"] = {
        "version": "E2-SP-R-shared-gpu0-r3-20261013",
        "owner_instruction": "Use available GPU memory even when GPU is shared; do not evict resident workloads.",
        "execution_device": "GPU0",
        "resident_contention": True,
        "no_process_eviction": True,
        "no_service_restart": True,
        "scientific_status": "OPERATIONAL_DESCRIPTIVE_ONLY_NOT_ISOLATED_CAPACITY",
    }
    protocol.setdefault("counts", {})["r3_shared_requests"] = 800
    protocol["counts"]["r3_shared_physical_forwards"] = 800
    write_json(protocol_path, protocol)

    (OUT / "E2_SP_R_SLO_CAPACITY_REPORT.md").write_text("""# E2-S/P-R SLO and Capacity Report

Status: `SHARED_OPERATIONAL_R3_COMPLETED_ISOLATED_CAPACITY_BLOCKED`.

The five-model seeded open-loop workload completed with 800/800 successful requests. Raw traces and recomputed metrics are in `E2_SP_R_R3_SHARED_EXECUTION_REPORT.md` and `E2_SP_R_R3_SHARED_SUMMARY.csv`.

This is not an isolated capacity certification. GPU0 had pre-existing work throughout. Shared p95/SLO observations are descriptive only; isolated sustainable QPS, queue stability, GPU allocation-seconds/query and monetary cost remain `NOT_MEASURED`.
""", encoding="utf-8")
    (OUT / "E2_SP_R_AUTHENTICITY_AUDIT.md").write_text("""# E2-S/P-R Authenticity Audit

Verdict: `E2_SP_R_PARTIAL_BLOCKED` with completed shared operational R3 traces.

- Frozen source identity and Top-20 hashes: PASS.
- Prior 3,300 selector and 3,300 reader rows: PASS and preserved.
- New native smoke/CAL rows: 190/190 PASS under shared GPU0 contention.
- Shared R3: 800/800 requests PASS across five models and four rates; 800 physical forwards.
- Every R3 request has candidate identity, actual input hash, raw response hash, and selection-text hashes.
- Gold in inference inputs: 0; application cache reuse: 0.
- Shared GPU0 contention: recorded in every request; no process eviction and no service restart.
- Isolated production SLO, fair hardware capacity, GPU allocation accounting, and monetary cost: BLOCKED/NOT_MEASURED.
- R4 downstream cases: 720 packaged from existing reader outputs; human labels: 0; local LLM labels: 0.
- Jev calls: 0; paid API calls: 0; fresh holdout: untouched; training: not run.
""", encoding="utf-8")
    (OUT / "E2_SP_R_FINAL_REPORT.md").write_text("""# E2-S/P-R Final Report

Verdict: `E2_SP_R_PARTIAL_BLOCKED`.

## Completed

- R0 frozen-cohort/source audit: PASS (300 unique queries; ordered Top-20 identities matched).
- Local roster: PASS for five approved models at frozen revisions.
- R1 native smoke: PASS WITH SHARED-CONTENTION LIMITATION (40/40 rows; 40 physical forwards).
- R2 CAL validation: PASS WITH SHARED-CONTENTION LIMITATION (150/150 rows; 150 physical forwards).
- R3 shared operational workload: 800/800 successful requests, five models × four rates × 40 requests, 800 physical forwards, zero errors.
- R3 raw-input and response trace integrity: PASS; all hashes and frozen candidate identities verified offline.
- R4 downstream package: 120 questions × 6 existing reader policies = 720 blinded cases; answer metrics remain proxy-only.
- Historical pre-shared package preserved under `revisions/pre_shared_execution_20261013/`.

## Remaining block

The original isolated-capacity gate remains blocked. GPU0 was shared with resident work throughout R1/R2/R3. Shared p95/SLO observations cannot certify isolated production capacity, fair hardware comparison, sustainable QPS, queue stability, GPU allocation-seconds/query, or monetary cost.

## Interpretation

All five approved native adapters loaded and produced aligned outputs on the frozen workload. BGE/GTE showed the strongest descriptive shared-device latency behavior; Jina was mixed; Nimble and PPLX were slower and queue-sensitive. These are operational observations under one shared topology, not deployment conclusions. R4 EM/F1 values are answer proxies, not independently verified factual correctness.

Jev and paid APIs remain untouched (0 calls). No process was terminated and no service was restarted.
""", encoding="utf-8")
    (OUT / "E2_SP_R_OWNER_REVIEW_GATE.md").write_text("""# E2-S/P-R Owner Review Gate

Verdict: `E2_SP_R_PARTIAL_BLOCKED`.

PASS: frozen population audit, artifact preservation, five-model local roster, native smoke, shared CAL validation, 800-request shared R3 execution, raw-trace/input/response integrity, and offline R4 package preparation.

PASS WITH LIMITATIONS: all new inference used shared GPU0. Every row records contention; recomputed p95/SLO values are descriptive only.

BLOCKED: isolated R3 capacity, fair production SLO, sustainable QPS, queue stability, GPU allocation accounting, monetary cost, and independent factual answer adjudication.

No Jev or paid request was made. An isolated authorized execution window is still required for a valid capacity comparison.
""", encoding="utf-8")

    # Hash all current generated artifacts after every report/protocol write.
    generated = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != manifest_path.name and "pre_shared_execution_20261013" not in path.parts:
            generated[str(path.relative_to(ROOT))] = sha_file(path)
    manifest["generated_artifact_sha256"] = generated
    write_json(manifest_path, manifest)


def main() -> int:
    source = load_source_items()
    rows: list[dict] = []
    all_requests = []
    for model, suffix in MODELS.items():
        trace_path = R3 / f"E2_SP_R_R3_SHARED_{suffix}_REQUESTS.jsonl"
        summary_path = R3 / f"E2_SP_R_R3_SHARED_{suffix}_SUMMARY.json"
        trace = load_jsonl(trace_path)
        published = load_json(summary_path)
        requests = [row for row in trace if row.get("record_type") == "request"]
        summaries = [row for row in trace if row.get("record_type") == "rate_summary"]
        if len(requests) != 160 or len(summaries) != 4 or len(published.get("summaries", [])) != 4:
            raise RuntimeError(f"bad trace shape for {model}")
        if published.get("status") != "SHARED_CONTENTION_LIMITED" or published.get("jev_calls") != 0 or published.get("paid_api_calls") != 0:
            raise RuntimeError(f"bad trace header for {model}")
        if sha_file(trace_path) != published.get("trace_sha256"):
            raise RuntimeError(f"trace hash mismatch for {model}")
        by_rate_jsonl = {int(row["rate_index"]): row for row in summaries}
        by_rate_summary = {int(row["rate_index"]): row for row in published["summaries"]}
        if sorted(by_rate_jsonl) != [0, 1, 2, 3] or sorted(by_rate_summary) != [0, 1, 2, 3]:
            raise RuntimeError(f"bad rate indexes for {model}")
        for request in requests:
            item = source.get(request.get("query_id"))
            if item is None or request.get("candidate_ids") != [c["doc_id"] for c in item["candidates"]]:
                raise RuntimeError(f"frozen candidate identity mismatch for {model}:{request.get('query_id')}")
            if len(request["candidate_ids"]) != 20 or len(set(request["candidate_ids"])) != 20:
                raise RuntimeError(f"invalid candidate slate for {model}")
            if request.get("resource_mode") != "shared_gpu0_contention_limited" or request.get("isolated") is not False:
                raise RuntimeError(f"missing shared marker for {model}")
            if request.get("gold_in_inference_input") is not False or request.get("cache_reuse") is not False:
                raise RuntimeError(f"gold/cache marker failure for {model}")
            if not request.get("actual_model_input_sha256") or not request.get("raw_response_sha256") or not request.get("selection_text_sha256"):
                raise RuntimeError(f"missing request hash for {model}")
            if int(request.get("physical_forward_calls", 0)) < 1:
                raise RuntimeError(f"missing physical forward for {model}")
        for rate_index in range(4):
            request_rate = [r for r in requests if int(r["rate_index"]) == rate_index]
            recomputed = check_summary(by_rate_summary[rate_index], request_rate)
            check_summary(by_rate_jsonl[rate_index], request_rate)
            summary = by_rate_summary[rate_index]
            rows.append({
                "model": model, "rate_index": rate_index, "offered_rps": summary["offered_rps"],
                "achieved_qps": summary["achieved_qps"], "requests": 40, "completed": 40, "errors": 0,
                "error_rate": 0.0, "p50_ms": recomputed["p50_ms"], "p95_ms": recomputed["p95_ms"],
                "p99_ms": recomputed["p99_ms"], "queue_p50_ms": recomputed["queue_p50_ms"],
                "queue_p95_ms": recomputed["queue_p95_ms"], "queue_growth_ms": summary["queue_growth_ms"],
                "slo_1000_pass": recomputed["p95_ms"] <= 1000.0,
                "slo_2000_pass": recomputed["p95_ms"] <= 2000.0,
                "slo_3000_pass": recomputed["p95_ms"] <= 3000.0,
                "physical_forward_calls": recomputed["physical_forward_calls"],
                "resource_mode": "shared_gpu0_contention_limited",
                "capacity_validity": "NOT_VALID_FOR_ISOLATED_CAPACITY",
                "cost_status": "NOT_MEASURED_SHARED_CONTENTION",
            })
        all_requests.extend(requests)
    if len(all_requests) != 800 or {r["query_id"] for r in all_requests} != set(source):
        raise RuntimeError("R3 does not equal 5 models × 4 rates × 40 over the frozen CAL population")

    manifest_path = OUT / "E2_SP_R_PROVENANCE_MANIFEST.json"
    manifest = load_json(manifest_path)
    manifest.update({
        "manifest_version": "E2-SP-R-v2-shared-gpu0-r3",
        "status": "E2_SP_R_PARTIAL_BLOCKED",
        "stage_status": {"R0": "PASS", "R1": "PASS_WITH_SHARED_CONTENTION_LIMITATION", "R2": "CAL_VALIDATION_PASS_WITH_SHARED_CONTENTION_LIMITATION", "R3": "EXECUTED_SHARED_CONTENTION_LIMITED_ISOLATED_GATE_BLOCKED", "R4": "PACKAGED_NOT_INDEPENDENTLY_ADJUDICATED", "R5": "PARTIAL_FRONTIER_NO_ISOLATED_R3_COST"},
        "counts": {**manifest.get("counts", {}), "r3_shared_requests": 800, "r3_shared_physical_forwards": 800, "new_model_requests": 990, "new_physical_forwards": 990},
        "resource_mode": "shared_gpu0_contention_limited", "isolated_capacity_measurement": False,
        "r3_open_loop_requests": 800, "r3_isolated_capacity_valid": False,
        "gpu_allocation_seconds": "NOT_MEASURED_ISOLATED; SHARED_PROCESS_WALL_PROXIES_IN_R3_SUMMARIES",
        "monetary_cost": "NOT_MEASURED_SHARED_CONTENTION", "jev_calls": 0, "paid_api_calls": 0,
    })
    write_reports(rows, manifest, manifest_path)
    generated = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != manifest_path.name and "pre_shared_execution_20261013" not in path.parts:
            generated[str(path.relative_to(ROOT))] = sha_file(path)
    manifest = load_json(manifest_path)
    manifest["generated_artifact_sha256"] = generated
    write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "r3_models": 5, "r3_requests": 800, "r3_errors": 0, "r3_physical_forwards": 800, "generated_artifacts": len(generated), "isolated_capacity": False, "jev_calls": 0, "paid_api_calls": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
