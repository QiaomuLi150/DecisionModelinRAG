#!/usr/bin/env python3
"""CPU-only D2 analysis and export generation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_6/d2_matched_timing"
RAW = OUT / "raw/d2_latency_perrequest_raw.jsonl"
FALLBACK = ROOT / "results/v1_5/c2_latency_uncertainty/export/fallback_cost.csv"
POLICIES = ["jina_full", "jina_two_stage_full", "jina_two_stage_pplxwindow", "jina_pplx_budget", "pplx_budget", "pplx_two_stage"]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos); hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def rows() -> list[dict[str, Any]]:
    return [json.loads(x) for x in RAW.open(encoding="utf-8") if x.strip()]


def main() -> None:
    rs = rows()
    if len(rs) != 2160 or {p: sum(x["policy"] == p for x in rs) for p in POLICIES} != {p: 360 for p in POLICIES}:
        raise RuntimeError("D2_RAW_COVERAGE_INVALID")
    fallback: dict[tuple[str, int], float] = {}
    with FALLBACK.open(newline="", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            fallback[(x["canonical_id"], int(x["attempt"]))] = float(x["fallback_ms"])
    ids = sorted({x["canonical_id"] for x in rs})
    if len(ids) != 120 or len(fallback) != 360 or not all((cid, a) in fallback for cid in ids for a in (1, 2, 3)):
        raise RuntimeError("D2_FALLBACK_COHORT_MISMATCH")
    for x in rs:
        x["fallback_ms"] = fallback[(x["canonical_id"], int(x["attempt"]))]
        x["effective_latency_ms"] = float(x["latency_ms"]) + (x["fallback_ms"] if not x.get("valid") else 0.0)
    # Required compact per-request export.  Keep the raw trace separate.
    per_path = OUT / "export/d2_latency_perrequest.csv"
    fields = ["policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"]
    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for x in rs:
            w.writerow({k: x.get(k) for k in fields})

    def values(policy: str, basis: str) -> list[float]:
        return [float(x["effective_latency_ms"] if basis == "all_requests_with_failure_fallback" else x["latency_ms"])
                for x in rs if x["policy"] == policy and (basis == "all_requests_with_failure_fallback" or x.get("valid"))]

    # Query-cluster bootstrap: sample the 120 canonical IDs with replacement,
    # then retain all three attempts for each sampled cluster.
    by = {(p, cid): [x for x in rs if x["policy"] == p and x["canonical_id"] == cid] for p in POLICIES for cid in ids}
    def bootstrap_ratio(policy: str, basis: str, draws: int = 10000, seed: int = 731) -> dict[str, float]:
        rng = random.Random(seed); out = []
        for _ in range(draws):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            pv = []; bv = []
            for cid in sample:
                pv.extend(float(x["effective_latency_ms"] if basis == "all_requests_with_failure_fallback" else x["latency_ms"]) for x in by[(policy, cid)] if basis == "all_requests_with_failure_fallback" or x.get("valid"))
                bv.extend(float(x["effective_latency_ms"] if basis == "all_requests_with_failure_fallback" else x["latency_ms"]) for x in by[("jina_full", cid)] if basis == "all_requests_with_failure_fallback" or x.get("valid"))
            if pv and bv and pct(bv, .95) not in (None, 0):
                out.append(pct(pv, .95) / pct(bv, .95))
        return {"p95_ratio": pct(out, .5), "ratio_lo95": pct(out, .025), "ratio_hi95": pct(out, .975),
                "prob_ratio_le_2": sum(x <= 2.0 for x in out) / len(out), "bootstrap_draws": len(out), "bootstrap_seed": seed}

    ratio_rows = []
    for p in POLICIES:
        basis = "all_requests_with_failure_fallback"
        vals = values(p, basis)
        all_rows = [x for x in rs if x["policy"] == p]
        alloc_all = [float(x["alloc_peak_mib"]) for x in all_rows if x.get("alloc_peak_mib") is not None]
        alloc_success = [float(x["alloc_peak_mib"]) for x in all_rows if x.get("valid") and x.get("alloc_peak_mib") is not None]
        b = bootstrap_ratio(p, basis)
        ratio_rows.append({"policy": p, "basis": basis, "p50_ms": pct(vals, .5), "p95_ms": pct(vals, .95),
                           "p95_ratio": 1.0 if p == "jina_full" else b["p95_ratio"],
                           "ratio_lo95": 1.0 if p == "jina_full" else b["ratio_lo95"],
                           "ratio_hi95": 1.0 if p == "jina_full" else b["ratio_hi95"],
                           "prob_ratio_le_2": 1.0 if p == "jina_full" else b["prob_ratio_le_2"],
                           "alloc_p95_mib": pct(alloc_all, .95),
                           "alloc_success_p95_mib": pct(alloc_success, .95),
                           "attempted": len(all_rows), "valid": sum(bool(x.get("valid")) for x in all_rows),
                           "failed": sum(not bool(x.get("valid")) for x in all_rows),
                           "failure_classes": dict(Counter(x.get("error_class") for x in all_rows if not x.get("valid"))),
                           "bootstrap_draws": b["bootstrap_draws"], "bootstrap_seed": b["bootstrap_seed"]})
    ratio_path = OUT / "export/d2_latency_ratios.csv"
    ratio_fields = ["policy", "basis", "p50_ms", "p95_ms", "p95_ratio", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib"]
    with ratio_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ratio_fields); w.writeheader(); w.writerows({k: x.get(k) for k in ratio_fields} for x in ratio_rows)
    (OUT / "D2_LATENCY_STATS_V2.json").write_text(json.dumps({"status": "PASS", "basis": "all_requests_with_failure_fallback", "fallback_source": str(FALLBACK.relative_to(ROOT)), "fallback_source_sha256": file_sha(FALLBACK), "rows": ratio_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "D2_RAW_TRACE_SHA256_MANIFEST_V2.json").write_text(json.dumps({"status": "PASS", "traces": [{"path": str(RAW.relative_to(ROOT)), "sha256": file_sha(RAW), "records": len(rs), "labels_joined": False}, {"path": str(FALLBACK.relative_to(ROOT)), "sha256": file_sha(FALLBACK), "records": len(fallback), "description": "authenticated frozen BM25 fallback timing reused offline"}]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Reports are deliberately explicit about the already-authenticated fallback
    # timing and about inference order/model-family loading.
    md = ["# DecisionRAG v1.6 D2 Matched Timing Report", "", "Status: `D2_COMPLETE`", "",
          "The frozen 120-ID cohort was evaluated three times per policy, one request at a time, in one host-capable Python session on dedicated GPU3. The old blocked D2 protocol and gate are preserved; this report is the versioned continuation after the GPU3 gate cleared.", "",
          "## Frozen execution", "", f"- Policies: {', '.join(POLICIES)}", "- Records: 2,160 (360 per policy)", "- Valid: 2,148; failures: 12", "- Labels joined: no", "- Protected holdouts/API calls: no/zero", "- Timing boundary: view preparation, native calls, score parsing, ID mapping and Top-5 packaging; model loading excluded and reported in the session metadata.", "- Jina and PPLX model families were loaded sequentially in the same process because both cannot be resident on an 80-GiB device; all requests used GPU3.", "", "## Timing and ratio table", "", "| policy | p50 ms | p95 ms | p95 / Jina-full | 95% ratio CI | Pr(ratio ≤ 2) | alloc p95 MiB | valid/attempted |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for x in ratio_rows:
        md.append(f"| {x['policy']} | {x['p50_ms']:.3f} | {x['p95_ms']:.3f} | {x['p95_ratio']:.4f} | [{x['ratio_lo95']:.4f}, {x['ratio_hi95']:.4f}] | {x['prob_ratio_le_2']:.4f} | {x['alloc_p95_mib']:.3f} | {x['valid']}/{x['attempted']} |")
    md += ["", "## Failure accounting", "", "Every failed request remains in the raw trace. The 12 failures are the same frozen `robotics:89` cohort record across the relevant policies: three exact D1 PPLX-source failures for `jina_two_stage_pplxwindow`, three unavailable E11 budget views for `jina_pplx_budget`, three unavailable E11 budget views for `pplx_budget`, and three dynamic-budget failures for `pplx_two_stage`. No retry or fill-in was used.", "", "## Fallback basis", "", "The all-request ratio basis adds the previously measured, frozen CPU BM25 Top-5 fallback time to failed request times only. That fallback trace is not a new inference run and is hash-recorded in `D2_RAW_TRACE_SHA256_MANIFEST_V2.json`. Successful request times are unchanged. The fallback measurement covers list extraction and first-five packaging, not a fresh BM25 search or model work.", "", "## Limitations", "", "- GPU3 was isolated at the recorded preflight (4 MiB used, 81,149 MiB free, 0% utilization); the session was not randomized across time and model families were sequentially loaded.", "- Native runtimes do not expose a portable forward counter; `n_calls` and native-call stage records are retained, while physical forward counts are not claimed.", "- These are matched selector timings, not sustainable production QPS or monetary cost estimates.", "- The ratio bootstrap is a 10,000-draw query-cluster bootstrap with seed 731 over 120 IDs and three attempts per ID."]
    (OUT / "D2_MATCHED_TIMING_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "D2_COMPLETE", "records": len(rs), "valid": sum(x.get("valid") for x in rs), "ratio_rows": ratio_rows}, indent=2))


if __name__ == "__main__":
    main()
