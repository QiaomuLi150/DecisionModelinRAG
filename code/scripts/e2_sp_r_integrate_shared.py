#!/usr/bin/env python3
"""Integrate the authorized shared-GPU0 E2-S/P-R R1/R2 traces.

The script is offline: it reads completed native traces and writes versioned
reports/manifests.  It never loads a checkpoint, sends a request, or changes
the frozen cohort.  R3 remains blocked because GPU0 was shared with another
active process throughout the new runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp_r"
SHARED = OUT / "shared_gpu0"
REVISION = OUT / "revisions/pre_shared_execution_20261013"

MODEL_FILES = {
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
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def load_source_items() -> dict[str, dict]:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_sp_gpu5_runner import load_cohort_items
    from e2_m1_experiment import load_local_cal_items
    rows = load_cohort_items() + load_local_cal_items()
    return {item["query_id"]: item for item in rows}


def main() -> int:
    summaries = []
    traces = {}
    for model, suffix in MODEL_FILES.items():
        for stage, expected in (("S0_SMOKE", 8), ("S1_CAL", 30)):
            summary_path = SHARED / f"E2_SP_R_{stage}_{suffix}_SHARED_GPU0_SUMMARY.json"
            trace_path = SHARED / f"E2_SP_{stage}_{suffix if suffix != 'PPLX_DECIDER' else 'PPLX_DECIDER'}.jsonl"
            summary = load_json(summary_path)
            rows = load_jsonl(trace_path)
            if summary.get("status") != "PASS" or summary.get("rows") != expected or summary.get("ok_rows") != expected:
                raise RuntimeError(f"shared trace failed validation: {summary_path}")
            if len(rows) != expected or any(row.get("status") != "ok" for row in rows):
                raise RuntimeError(f"shared trace row validation failed: {trace_path}")
            for row in rows:
                if row.get("e2_sp_resource_mode") != "shared_gpu0_contention_limited":
                    raise RuntimeError(f"missing shared resource marker: {trace_path}")
                if row.get("e2_sp_isolated") is not False:
                    raise RuntimeError(f"trace incorrectly marked isolated: {trace_path}")
                if row.get("gold_in_inference_input") or row.get("e2_sp_gold_in_inference_input"):
                    raise RuntimeError(f"gold leakage marker in trace: {trace_path}")
                if row.get("cache_reuse") or row.get("e2_sp_cache_reuse"):
                    raise RuntimeError(f"cache reuse marker in trace: {trace_path}")
                if not row.get("actual_model_input_sha256") or not row.get("raw_response_sha256"):
                    raise RuntimeError(f"missing input/response hash: {trace_path}")
                if len(row.get("candidate_ids", [])) != 20 or len(set(row.get("candidate_ids", []))) != 20:
                    raise RuntimeError(f"invalid candidate slate in trace: {trace_path}")
            summaries.append(summary)
            traces[(model, stage)] = rows

    source = load_source_items()
    for rows in traces.values():
        for row in rows:
            item = source.get(row["query_id"])
            if item is None or [c["doc_id"] for c in item["candidates"]] != row["candidate_ids"]:
                raise RuntimeError(f"frozen candidate identity mismatch: {row.get('query_id')}")

    cal_rows = []
    for model in MODEL_FILES:
        rows = traces[(model, "S1_CAL")]
        durations = [float(row["duration_ms"]) for row in rows if isinstance(row.get("duration_ms"), (int, float))]
        tokens = [row.get("input_tokens") for row in rows if isinstance(row.get("input_tokens"), int)]
        cal_rows.append({
            "model": model,
            "resource_mode": "shared_gpu0_contention_limited",
            "n": len(rows),
            "ok": sum(row.get("status") == "ok" for row in rows),
            "physical_forward_calls": sum(int(row.get("physical_forward_calls", 0)) for row in rows),
            "p50_duration_ms": percentile(durations, 0.50),
            "p95_duration_ms": percentile(durations, 0.95),
            "mean_duration_ms": sum(durations) / len(durations) if durations else None,
            "min_duration_ms": min(durations) if durations else None,
            "max_duration_ms": max(durations) if durations else None,
            "mean_input_tokens": sum(tokens) / len(tokens) if tokens else "NOT_RECORDED_NATIVE_ADAPTER",
            "gold_inference_rows": sum(bool(row.get("gold_in_inference_input") or row.get("e2_sp_gold_in_inference_input")) for row in rows),
            "cache_reuse_rows": sum(bool(row.get("cache_reuse") or row.get("e2_sp_cache_reuse")) for row in rows),
            "capacity_interpretation": "NOT_VALID_FOR_ISOLATED_CAPACITY",
        })

    csv_path = OUT / "E2_SP_R_SHARED_CAL_TELEMETRY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cal_rows[0]))
        writer.writeheader()
        writer.writerows(cal_rows)

    smoke_ok = sum(s["ok_rows"] for s in summaries if s["e2_sp_stage"] == "S0_SMOKE")
    cal_ok = sum(s["ok_rows"] for s in summaries if s["e2_sp_stage"] == "S1_CAL")
    smoke_forwards = sum(s["physical_forward_calls"] for s in summaries if s["e2_sp_stage"] == "S0_SMOKE")
    cal_forwards = sum(s["physical_forward_calls"] for s in summaries if s["e2_sp_stage"] == "S1_CAL")

    config = OUT / "E2_SP_R_CAL_AND_SERVING_CONFIG.md"
    lines = [
        "# E2-S/P-R CAL and Serving Configuration",
        "",
        "Status: `R1_PASS_R2_CAL_VALIDATION_SHARED_CONTENTION_LIMITED_R3_BLOCKED`.",
        "",
        "The approved native smoke and 30-query CAL validation were executed on shared GPU0. The native adapters, frozen 20-candidate inputs, score contracts and trace hashes are valid. Because another process remained attached to GPU0, these runs are not isolated serving measurements and were not used to claim a production SLO or cost.",
        "",
        "| Model | CAL n | CAL p50 ms | CAL p95 ms | Physical forwards | Selected optimization |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in cal_rows:
        lines.append(f"| `{row['model']}` | {row['n']} | {row['p50_duration_ms']:.3f} | {row['p95_duration_ms']:.3f} | {row['physical_forward_calls']} | NOT_SELECTED_SHARED_CONTENTION_LIMITED |")
    lines.extend([
        "",
        "The R2 result is CAL validation, not tuning: no prompt, candidate pool, scoring contract, or native model semantics were changed. R3 open-loop capacity remains blocked until an isolated authorized device/window is available.",
    ])
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = OUT / "E2_SP_R_SHARED_EXECUTION_AUDIT.md"
    audit.write_text(f"""# E2-S/P-R Shared-GPU Execution Audit

Verdict: `R1_PASS_R2_CAL_VALIDATION_PASS_WITH_LIMITATIONS_R3_BLOCKED`.

- Native smoke: `{smoke_ok}/40` rows passed with `{smoke_forwards}` physical forwards.
- CAL validation: `{cal_ok}/150` rows passed with `{cal_forwards}` physical forwards.
- Models: BGE, Jina v3.5, Nimble, GTE, and PPLX Decider at the frozen local revisions.
- Every row has a 20-candidate frozen slate, actual model-input hash, raw response hash, and no gold/cache markers.
- GPU: physical GPU0, shared with an existing process; no process was evicted and no resident service was restarted.
- The shared condition is a scientific limitation: duration values are descriptive CAL traces only. They cannot establish isolated p95, sustainable QPS, queue stability, GPU-seconds/query, or monetary cost.
- Jev calls: 0. Paid API calls: 0. Fresh holdout access: false. Training: false.

The pre-execution blocked package is preserved under `revisions/pre_shared_execution_20261013/`.
""", encoding="utf-8")

    resource_lines = [
        "# E2-S/P-R Shared GPU Resource Audit",
        "",
        "All new native runs used physical GPU0 sequentially. GPU0 already had an attached process; no process was evicted and no service was restarted. The snapshots below are descriptive resource telemetry, not isolated capacity measurements.",
        "",
        "| Model | Stage | GPU0 free before (MiB) | GPU0 utilization before (%) | Compute records before | GPU snapshot after |",
        "|---|---|---:|---:|---:|---|",
    ]
    for summary in summaries:
        before = summary["gpu_guard_before"]["gpu_before"]
        after = summary.get("gpu_snapshot_after", {}).get("stdout", "").replace("\n", "\\n")
        resource_lines.append(f"| `{summary['model_key']}` | `{summary['e2_sp_stage']}` | {before['memory_free_mib']} | {before['utilization_gpu_percent']} | {len(summary['gpu_guard_before'].get('compute_apps_before', []))} | `{after}` |")
    resource_lines.extend(["", "Interpretation: `SHARED_CONTENTION_LIMITED`; R3 open-loop SLO, queue stability, QPS and cost remain blocked."])
    (OUT / "E2_SP_R_SHARED_GPU_RESOURCE_AUDIT.md").write_text("\n".join(resource_lines) + "\n", encoding="utf-8")

    # Keep the original source/protocol inputs intact while replacing only the
    # R-stage status language in the owner-facing reports.
    (OUT / "E2_SP_R_SLO_CAPACITY_REPORT.md").write_text("""# E2-S/P-R SLO and Capacity Report

Status: `MEASUREMENT_BLOCKED_SHARED_CONTENTION`.

Native smoke and CAL traces completed on shared GPU0. They validate the adapters and provide descriptive single-client CAL timings, but GPU0 had another active process throughout. No isolated open-loop arrivals were issued. Therefore sustainable QPS, isolated 1-second p95 feasibility, queue stability, GPU allocation-seconds/query and monetary cost remain `NOT_MEASURED`.

The shared CAL timing table is in `E2_SP_R_SHARED_CAL_TELEMETRY.csv`. It must not be substituted for isolated capacity or added to reader latency to form an end-to-end percentile.
""", encoding="utf-8")

    (OUT / "E2_SP_R_AUTHENTICITY_AUDIT.md").write_text(f"""# E2-S/P-R Authenticity Audit

Verdict: `R1_PASS_R2_CAL_VALIDATION_PASS_WITH_LIMITATIONS_R3_BLOCKED`.

- Frozen source identity and Top-20 hashes: PASS.
- Prior 3,300 selector and 3,300 reader rows: PASS and preserved.
- New native smoke rows: `{smoke_ok}/40` PASS; new CAL rows: `{cal_ok}/150` PASS.
- Physical forwards: `{smoke_forwards + cal_forwards}`; all new rows have input and response hashes.
- Gold in inference inputs: 0; application cache reuse: 0.
- Shared GPU0 contention: recorded in every row; no process eviction and no service restart.
- R3 open-loop capacity, isolated SLO, GPU allocation accounting and monetary cost: NOT_MEASURED.
- R4 downstream cases: 720 packaged from existing reader outputs; human labels: 0; local LLM labels: 0.
- Jev calls: 0; paid API calls: 0; fresh holdout: untouched; training: not run.

Shared execution demonstrates native adapter functionality, not isolated production capacity.
""", encoding="utf-8")

    (OUT / "E2_SP_R_FINAL_REPORT.md").write_text(f"""# E2-S/P-R Final Report

Verdict: `E2_SP_R_PARTIAL_BLOCKED`.

## Completed

- R0 frozen-cohort/source audit: PASS (300 unique queries; all question and ordered Top-20 identities matched).
- Local roster: PASS for five approved models at frozen revisions.
- R1 native smoke: PASS WITH SHARED-CONTENTION LIMITATION (`{smoke_ok}/40` rows, `{smoke_forwards}` physical forwards).
- R2 CAL validation: PASS WITH SHARED-CONTENTION LIMITATION (`{cal_ok}/150` rows, `{cal_forwards}` physical forwards). No serving optimization was selected or tuned.
- R4 downstream package: 120 questions × 6 existing reader policies = 720 blinded cases; deterministic EM/F1 remains proxy-only.
- Historical blocked package preserved under `revisions/pre_shared_execution_20261013/`.

## Remaining block

R3 open-loop capacity and production SLO are blocked. GPU0 was usable for sequential model calls but was not isolated: an existing process remained attached, and other GPUs were occupied by resident services or unrelated workloads. Shared traces cannot support a fair isolated p95/QPS/cost comparison. No process was terminated and no service was restarted.

## Interpretation

The new traces establish that all five approved native adapters can load and produce aligned 20-candidate outputs with physical-forward evidence on the frozen cohort. The CAL duration values are descriptive shared-device observations only. They do not establish a deployment frontier. R4 EM/F1 values are answer proxies, not independently verified factual correctness; enterprise reference answers remain unavailable.

Jev and paid APIs remain untouched (`0` calls). Full R-stage completion requires a genuinely isolated authorized execution window for R3. Deployment/cost conclusion: `MEASUREMENT_BLOCKED`.
""", encoding="utf-8")

    (OUT / "E2_SP_R_OWNER_REVIEW_GATE.md").write_text("""# E2-S/P-R Owner Review Gate

Verdict: `E2_SP_R_PARTIAL_BLOCKED`.

PASS: frozen population audit, artifact preservation, five-model local roster, native smoke, shared-device CAL validation, trace/input/response integrity, and offline R4 assessment-package preparation.

PASS WITH LIMITATIONS: R1/R2 execution used shared GPU0. All rows record the contention condition; no isolated latency or cost claim is made.

BLOCKED: R3 open-loop capacity, isolated 1-second p95 SLO, sustainable QPS, queue stability, GPU allocation accounting, monetary cost, and independent factual answer adjudication.

No Jev or paid request was made. Next owner decision: provide an isolated approved GPU/window for R3, or accept the remaining partial block.
""", encoding="utf-8")

    protocol_path = OUT / "E2_SP_R_FROZEN_PROTOCOL.json"
    protocol = load_json(protocol_path)
    protocol["status"] = "FROZEN_R1_R2_SHARED_CONTENTION_R3_BLOCKED"
    protocol["open_loop"]["status"] = "NOT_EXECUTED_SHARED_CONTENTION_INTEGRITY_BLOCK"
    protocol["r2_cal_optimization"]["selected_configurations"] = "CAL_VALIDATED_SHARED_CONTENTION; NO_OPTIMIZATION_SELECTED"
    protocol["resource_snapshot"]["resource_gate"] = "SHARED_CONTENTION_LIMITED_FOR_R1_R2; R3_BLOCKED"
    protocol["authorization"]["shared_gpu0_execution"] = True
    protocol["authorization"]["no_process_eviction"] = True
    protocol["authorization"]["no_service_restart"] = True
    write_json(protocol_path, protocol)

    # Manifest all generated R files recursively, including the shared traces.
    generated = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "E2_SP_R_PROVENANCE_MANIFEST.json" and "pre_shared_execution_20261013" not in path.parts:
            generated[str(path.relative_to(ROOT))] = sha_file(path)
    manifest = load_json(OUT / "E2_SP_R_PROVENANCE_MANIFEST.json")
    manifest.update({
        "manifest_version": "E2-SP-R-v2-shared-gpu0",
        "status": "E2_SP_R_PARTIAL_BLOCKED",
        "stage_status": {"R0": "PASS", "R1": "PASS_WITH_SHARED_CONTENTION_LIMITATION", "R2": "CAL_VALIDATION_PASS_WITH_SHARED_CONTENTION_LIMITATION", "R3": "BLOCKED_SHARED_CONTENTION", "R4": "PACKAGED_NOT_INDEPENDENTLY_ADJUDICATED", "R5": "PARTIAL_FRONTIER_NO_R3_COST"},
        "generated_artifact_sha256": generated,
        "counts": {**manifest.get("counts", {}), "r1_smoke_rows": smoke_ok, "r1_smoke_physical_forwards": smoke_forwards, "r2_cal_rows": cal_ok, "r2_cal_physical_forwards": cal_forwards, "new_model_requests": smoke_ok + cal_ok, "new_physical_forwards": smoke_forwards + cal_forwards},
        "resource_mode": "shared_gpu0_contention_limited",
        "isolated_capacity_measurement": False,
        "r3_open_loop_requests": 0,
        "gpu_allocation_seconds": "NOT_MEASURED",
        "monetary_cost": "NOT_MEASURED",
        "jev_calls": 0,
        "paid_api_calls": 0,
    })
    write_json(OUT / "E2_SP_R_PROVENANCE_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "smoke_rows": smoke_ok, "cal_rows": cal_ok, "smoke_forwards": smoke_forwards, "cal_forwards": cal_forwards, "r3": "BLOCKED_SHARED_CONTENTION", "jev_calls": 0, "paid_api_calls": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
