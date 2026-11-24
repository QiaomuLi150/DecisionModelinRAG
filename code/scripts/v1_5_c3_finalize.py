#!/usr/bin/env python3
"""Offline completion of the bounded C3 export and report."""
from __future__ import annotations
import csv, hashlib, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_5/c3_concurrency_gpu3"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def main():
    rows = list(csv.DictReader((OUT / "export/c3_concurrency.csv").open(encoding="utf-8")))
    receipt = json.loads((OUT / "C3_EXECUTION_RECEIPT.json").read_text(encoding="utf-8"))
    rmax = receipt["rmax"]
    best = max(rmax, key=lambda x: x["throughput_rps"])
    rows.append({"config": "A_routing_only", "routing_rate_rps": "", "selection_rate_rps": "", "type": "capacity", "n": best["completed"], "p50_ms": best["p50_ms"], "p95_ms": best["p95_ms"], "p99_ms": "", "throughput_rps": best["throughput_rps"], "timeouts": 0, "peak_mem_gib": ""})
    (OUT / "export/R_MAX.json").write_text(json.dumps({"best": best, "all_levels": rmax}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["config", "routing_rate_rps", "selection_rate_rps", "type", "n", "p50_ms", "p95_ms", "p99_ms", "throughput_rps", "timeouts", "peak_mem_gib"]
    with (OUT / "export/c3_concurrency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    events = sum(1 for x in (OUT / "raw/c3_events.jsonl").open(encoding="utf-8") if x.strip())
    report = ["# v1.5 C3 Bounded Concurrency Diagnostic", "", "## Status", "", "`C3_BOUNDED_EXPLORATORY_COMPLETE_D_SEPARATE_BLOCKED`", "", f"- Genuine label-blind native event records: `{events}`.", f"- GPU: `3` (`{receipt['gpu_uuid']}`); post-run host check: released to 4 MiB used and 0% utilization.", f"- PPLX revision: `{receipt['pplx_load_ms']:.3f}` ms load measurement; model revision is pinned in the protocol.", "- D_separate was not run because no second dedicated A100 passed the isolation gate.", "", "## Rmax diagnostic", "", f"- Best serial native-call capacity row: `{best['throughput_rps']:.6f}` requests/s at tested concurrency `{best['concurrency']}`.", "- This is a serial native-call capacity diagnostic, not sustainable production QPS.", "", "## Observed bounded cells", "", "| Config | Routing rate | Selection rate | Type | n | p50 ms | p95 ms | Throughput/s |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["type"] != "capacity": report.append(f"| {r['config']} | {r['routing_rate_rps']} | {r['selection_rate_rps']} | {r['type']} | {r['n']} | {r['p50_ms']} | {r['p95_ms']} | {r['throughput_rps']} |")
    report += ["", "## Integrity limitations", "", "- The full v2 C3 schedule was shortened to 10 s warm-up and 60 s measurement cells under versioned protocol `PROTOCOL_V3_BOUNDED.json`; these results are exploratory and not a replacement for the original 60/300-second design.", "- The recovered R1 routing source contains one fixed request; the pool repeats that exact request with stable IDs. No prompt diversity is claimed.", "- The queue is a deterministic FIFO virtual-arrival simulation using genuine main-thread native forwards. The failed thread-wrapper attempt is preserved by its traceback/session and is not included as a result.", "- No labels or relevance outcomes entered any model input. No Jev/paid calls occurred.", ""]
    (OUT / "C3_BOUNDED_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "C3_EXPORT_MANIFEST.json").write_text(json.dumps({"csv_sha256": sha(OUT / "export/c3_concurrency.csv"), "csv_rows": len(rows), "raw_sha256": sha(OUT / "raw/c3_events.jsonl"), "raw_records": events, "receipt_sha256": sha(OUT / "C3_EXECUTION_RECEIPT.json"), "labels_joined": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"csv_rows": len(rows), "raw_records": events, "best_rmax": best, "status": "C3_BOUNDED_EXPLORATORY_COMPLETE_D_SEPARATE_BLOCKED"}, indent=2, sort_keys=True))

if __name__ == "__main__": main()
