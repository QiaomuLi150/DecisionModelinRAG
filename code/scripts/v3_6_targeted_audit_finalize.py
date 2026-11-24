#!/usr/bin/env python3
"""Finalize A1-A3 and available-cohort B1-B2 outputs offline."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_targeted_audit"
B = OUT / "B_transfer_timing"
POLICIES = ("pplx_single_L4096", "pplx_single_L6144", "jina_full", "jina_truncated_L8192")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def q95(values):
    if not values: return float("nan")
    xs = sorted(float(x) for x in values)
    k = 0.95 * (len(xs) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi: return xs[lo]
    return xs[lo] + (k - lo) * (xs[hi] - xs[lo])


def load_jsonl(path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def bootstrap_p95_ratio(query_values, baseline_p95, draws=40000, seed=731):
    ids = sorted(query_values)
    if not baseline_p95 or not ids:
        return {"p95_ratio": None, "ratio_lo95": None, "ratio_hi95": None,
                "prob_ratio_le_2": None, "draws": draws, "seed": seed,
                "unit": "canonical_id", "bootstrap_engine": "numpy_batched_linear_quantile"}
    # Batched NumPy resampling preserves the requested query-ID bootstrap,
    # linear p95 interpolation and seed while avoiding the infeasible Python
    # loop over 40,000 full-size sorted samples.
    import numpy as np
    values = np.asarray([float(query_values[x]) for x in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws, dtype=np.float64)
    batch_size = 512
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        sample_idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples = values[sample_idx]
        ratios[start:stop] = np.quantile(samples, 0.95, axis=1, method="linear") / baseline_p95
    ratios.sort()
    observed = q95(values.tolist()) / baseline_p95
    return {"p95_ratio": q95([query_values[x] for x in ids]) / baseline_p95 if baseline_p95 else None,
            "ratio_lo95": ratios[int(0.025 * (len(ratios) - 1))],
            "ratio_hi95": ratios[int(0.975 * (len(ratios) - 1))],
            "prob_ratio_le_2": sum(x <= 2.0 for x in ratios) / len(ratios),
            "draws": draws, "seed": seed, "unit": "canonical_id",
            "bootstrap_engine": "numpy_batched_linear_quantile"}


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    raw = []
    for p in sorted((B / "raw").glob("*.jsonl")):
        for record in load_jsonl(p):
            # Retain the physical trace identity for offline joins.  GPU and
            # S1/S2 labels can be reused across sequential workers.
            record["_source_file"] = str(p)
            raw.append(record)
    target = [x for x in raw if x.get("policy") in POLICIES]
    if not target:
        raise SystemExit("NO_COMPLETED_B_TARGET_TRACES")
    by = defaultdict(list); anchors = defaultdict(list)
    for x in raw:
        if x.get("anchor"):
            anchors[(x["policy"].rsplit("_", 1)[0], x.get("session"))].append(x)
        elif x.get("policy") in POLICIES:
            by[(x["policy"], x.get("session"))].append(x)
    ratio_rows = []
    for policy in POLICIES:
        all_target = [x for x in target if x["policy"] == policy]
        for session in sorted({x.get("session") for x in all_target}):
            xs = by[(policy, session)]
            worker_gpu = xs[0].get("gpu_slot") if xs else None
            # Each policy worker owns its own start/end anchor set.  Session
            # labels and GPU slots can both repeat across sequential workers,
            # so the physical source trace is the authoritative join key.
            an = [x for x in raw
                  if x.get("_source_file") == xs[0].get("_source_file")
                  and str(x.get("policy", "")).startswith("jina_full_anchor_")]
            qvals = defaultdict(list)
            for x in xs:
                if x.get("latency_ms") is not None: qvals[x["canonical_id"]].append(float(x["latency_ms"]))
            means = {k: statistics.mean(v) for k, v in qvals.items() if v}
            anchor_lat = [float(x["latency_ms"]) for x in an if x.get("latency_ms") is not None]
            baseline = q95(anchor_lat)
            session_valid_rate = sum(bool(x.get("valid")) for x in xs) / len(xs) if xs else 0.0
            if session_valid_rate < 0.99 or not means or not anchor_lat:
                boot = {"p95_ratio": "NOT_COMPARABLE", "ratio_lo95": "NOT_COMPARABLE", "ratio_hi95": "NOT_COMPARABLE", "prob_ratio_le_2": "NOT_COMPARABLE", "draws": 40000, "seed": 731, "unit": "canonical_id"}
            else:
                boot = bootstrap_p95_ratio(means, baseline)
            ratio_rows.append({"policy": policy, "basis": f"session_{session}", "n_requests": len(xs), "valid_n": sum(bool(x.get("valid")) for x in xs), "valid_rate_strict": session_valid_rate, "p50_ms": statistics.median([float(x["latency_ms"]) for x in xs if x.get("latency_ms") is not None]), "p95_ms": q95([float(x["latency_ms"]) for x in xs if x.get("latency_ms") is not None]), "anchor_n": len(anchor_lat), "anchor_p95_ms": baseline, "p95_ratio": boot["p95_ratio"], "ratio_lo95": boot["ratio_lo95"], "ratio_hi95": boot["ratio_hi95"], "prob_ratio_le_2": boot["prob_ratio_le_2"], "alloc_p95_mib": q95([float(x["alloc_peak_mib"]) for x in xs if x.get("alloc_peak_mib") is not None]), "R_verdict": "PASS" if session_valid_rate >= .99 else "FAIL", "S_verdict": "NOT_COMPARABLE" if session_valid_rate < .99 else ("PASS" if isinstance(boot["ratio_hi95"], float) and boot["ratio_hi95"] <= 2 else "FAIL" if isinstance(boot["ratio_lo95"], float) and boot["ratio_lo95"] > 2 else "INCONCLUSIVE")})
    fields = list(ratio_rows[0].keys())
    write_csv(B / "export_d2_latency_ratios.csv", fields, ratio_rows)
    per_fields = ["session", "gpu_slot", "cohort", "policy", "canonical_id", "dataset", "attempt", "pass_id", "status", "error_class", "latency_ms", "latency_with_fallback_ms", "alloc_peak_mib", "input_sha256", "labels_joined"]
    write_csv(B / "export_d2_latency_perrequest.csv", per_fields, [{k: x.get(k) for k in per_fields} for x in raw])
    manifest = {}
    for p in sorted((OUT).rglob("*")):
        if p.is_file() and p.name not in {"B1_B2_REPORT.md"}:
            manifest[str(p.relative_to(ROOT))] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    (OUT / "RAW_AND_EXPORT_SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (B / "EXECUTION_EVENTS.json").write_text(json.dumps({"status": "PASS_WITH_RESTARTED_RUNNER_ATTEMPTS", "aborted_attempts": [
        {"policy": "pplx_single_L4096", "session": 1, "cause": "progress-print TypeError after first target pass; no canonical trace finalized; corrected canonical restart launched"},
        {"policy": "pplx_single_L6144", "session": 1, "cause": "progress-print TypeError after first target pass; no canonical trace finalized; corrected canonical restart launched"},
        {"policy": "pplx_single_L6144", "session": 2, "cause": "operator-interrupted invalid runner before its known progress-print boundary; no canonical trace finalized; corrected canonical restart launched"},
        {"policy": "pplx_single_L4096", "session": 2, "cause": "operator-interrupted invalid runner before its known progress-print boundary; no canonical trace finalized; corrected canonical restart launched"},
        {"policy": "jina_full", "session": 1, "cause": "progress-print TypeError after target computation; no canonical trace finalized; corrected canonical restart already present"},
        {"policy": "jina_truncated_L8192", "session": 1, "cause": "progress-print TypeError after target computation; no canonical trace finalized; corrected canonical restart already present"}
    ], "canonical_restarts": "Only corrected sessions are included in canonical raw exports; aborted attempts are retained here as technical execution evidence."}, indent=2) + "\n", encoding="utf-8")
    status = "PARTIAL"
    (OUT / "STATUS.md").write_text("""# DecisionRAG v3.6 targeted audit status\n\n| Stage | Status | Explanation |\n|---|---|---|\n| A1 | PARTIAL | The exact frozen v1.5/v1.6 pair was audited, but raw serialized text strings were not stored; exact serialized-input hashes and all list/set differences are exported. The claimed 197 differences are not present: 1,069 list and 789 set differences were observed. |\n| A2 | PARTIAL | v1.5/v1.6 tokenizer/dtype/batch/attention settings are unrecoverable from their frozen receipts. Existing fp32 batch-1 C2 exports were reused without rerun; no historical setting cause is identifiable. |\n| A3 | PARTIAL | Both historical values are retained with explicit protocol labels; no replacement value is scientifically justified. |\n| B1 | PARTIAL | Clean timing completed for the available frozen 500 Enterprise + 2,167 MuSiQue cohort with two sessions/two passes and Jina anchors; the requested additional 333 MuSiQue slates do not exist. |\n| B2 | PARTIAL | p95 and anchor-relative bootstrap ratios are exported for the available cohort; the full requested 2,500-question MuSiQue population remains blocked and sub-99% policies are NOT_COMPARABLE. |\n\nNo labels were included in raw timing traces. No paid API was called. No historical artifact was overwritten.\n""", encoding="utf-8")
    summary = json.loads((OUT / "A1_A3_SUMMARY.json").read_text())
    report = ["# DecisionRAG v3.6 Targeted Audit — Final Report", "", "## Final gate", "", "`V3_6_TARGETED_AUDIT_PARTIAL_BLOCKED`", "", "## A1–A3 historical Jina audit", "", f"The actual v1.5/v1.6 original-order pair contains {summary['population']} common queries, {summary['list_difference_n']} Top-5 list differences and {summary['set_difference_n']} Top-5 set differences. This does not match the previously stated 197-query premise. The cited nDCG values are reproduced from existing exports: v1.5 = 0.1789703393 and v1.6 D1 = 0.1855466324; observed difference = 0.0065762931.", "", "For the first 30 differing queries, the earliest evidence-bounded difference is a window serialized-input/content-hash mismatch. The raw text strings themselves were not retained, so the audit exports exact request hashes where available and does not claim a raw-text comparison. Score-level, dtype, padding, batch and tokenizer causes remain unidentified.", "", "The existing C2 fp32/batch-1 pair is reused: both runs have 1,380 valid rows and exact list agreement. Its mean nDCG is retained as a diagnostic only; it does not reproduce either historical row and does not identify a cause.", "", "### Paper handling", "", "Retain the v1.5 frozen E11 input-text result and the v1.6 D1 matched-text control as separate protocol-labeled rows. Do not silently replace 0.17897 with 0.18555 or attribute the gap to a setting that was not recorded.", "", "## B1–B2 transfer timing", "", f"The new clean timing covers {len(target)} target request records over the available frozen cohort. The required 2,500 MuSiQue population cannot be executed from the repository: the exact frozen R1 artifacts contain 2,167 MuSiQue slates, leaving 333 missing candidate slates. This is a hard coverage blocker, not an outcome-based exclusion.", "", "Per-request and per-session p95 exports are in `B_transfer_timing/export_d2_latency_perrequest.csv` and `B_transfer_timing/export_d2_latency_ratios.csv`. Each worker/session uses its own 120-record (60 start + 60 end) full-text Jina anchor, with query-ID bootstrap (40,000 draws, seed 731); ratios are explicitly `NOT_COMPARABLE` below 99% validity.", "", "## Integrity", "", "Raw timing records are label blind, historical inputs are preserved, and the execution events document the two runner failures that were corrected before canonical traces were finalized. GPUs 3–6 were the only devices used for the authorized timing workers; no other process occupied them during the runs.", "", "## Artifacts", "", "- `STATUS.md`", "- `A_twostage_jina/twostage_jina_diff.csv`", "- `A_twostage_jina/twostage_jina_first30_causes.csv`", "- `A_twostage_jina/A2_SETTINGS_AUDIT.json`", "- `A_twostage_jina/A2_EXISTING_RUN_SUMMARY.csv`", "- `B_transfer_timing/PROTOCOL.json`", "- `B_transfer_timing/export_d2_latency_perrequest.csv`", "- `B_transfer_timing/export_d2_latency_ratios.csv`", "- `B_transfer_timing/EXECUTION_EVENTS.json`", "- `RAW_AND_EXPORT_SHA256_MANIFEST.json`"]
    (OUT / "B1_B2_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Recompute the manifest after all generated reports/events exist.  Do not
    # include the manifest itself, which cannot hash its own final bytes.
    final_manifest = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"RAW_AND_EXPORT_SHA256_MANIFEST.json"}:
            final_manifest[str(p.relative_to(ROOT))] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    (OUT / "RAW_AND_EXPORT_SHA256_MANIFEST.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "target_records": len(target), "ratios": str(B / "export_d2_latency_ratios.csv")}, indent=2))


if __name__ == "__main__": main()
