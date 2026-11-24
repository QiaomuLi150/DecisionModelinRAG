#!/usr/bin/env python3
"""Export 97.5% two-sided p95 latency-ratio CIs from clean Pass-B traces."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/v3_6_essential/frozen_prospective_validation_v2"
OUT = V2 / "final_closure_v1"
SEED = 731
DRAWS = 10_000
POLICIES = ("pplx_single_4096", "pplx_single_6144")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main() -> None:
    passes = [
        read(V2 / "raw/native_passB_paired_clean_pass1_gpu3_shard0_of1_pass1.jsonl"),
        read(V2 / "raw/native_passB_paired_clean_pass2_gpu3_shard0_of1_pass2.jsonl"),
    ]
    maps = []
    for rows in passes:
        maps.append({(r["canonical_id"], r["policy"]): r for r in rows if r.get("anchor_stage") is None})
    ids = sorted({cid for cid, _ in maps[0]} & {cid for cid, _ in maps[1]})
    rows = []
    for policy in POLICIES:
        # Match the existing 95% analyzer: each policy receives the same
        # query-cluster draw stream initialized from the frozen seed.
        rng = np.random.default_rng(SEED)
        left = np.asarray([[float(m[(cid, policy)]["latency_ms"]) for m in maps] for cid in ids], dtype=float)
        right = np.asarray([[float(m[(cid, "jina_full_text")]["latency_ms"]) for m in maps] for cid in ids], dtype=float)
        observed = float(np.percentile(left.reshape(-1), 95, method="linear") / np.percentile(right.reshape(-1), 95, method="linear"))
        boot = np.empty(DRAWS, dtype=float)
        for i in range(DRAWS):
            sample = rng.integers(0, len(ids), size=len(ids))
            boot[i] = np.percentile(left[sample].reshape(-1), 95, method="linear") / np.percentile(right[sample].reshape(-1), 95, method="linear")
        rows.append({
            "policy": policy,
            "basis": "pooled_clean_serial_passes",
            "n_query_ids": len(ids),
            "p95_ratio": observed,
            "ratio_lo97_5": float(np.quantile(boot, 0.0125)),
            "ratio_hi97_5": float(np.quantile(boot, 0.9875)),
            "bootstrap_draws": DRAWS,
            "bootstrap_seed": SEED,
            "bootstrap_unit": "canonical_id_query_cluster",
            "interval": "two-sided_97.5_percent",
            "latency_basis": "all_requests_including_failure_latency; no fallback latency substituted",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUT / "PASSB_P95_RATIOS_97_5.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    md = [
        "# Pass-B p95 Ratio 97.5% Bootstrap Intervals",
        "",
        "Two-sided 97.5% query-cluster bootstrap intervals, using 10,000 draws and seed 731. Each query contributes both clean serial passes; no fallback latency is substituted.",
        "",
        "| Policy | p95 ratio vs full-text Jina | 97.5% interval |",
        "|---|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['policy']} | {r['p95_ratio']:.6f} | [{r['ratio_lo97_5']:.6f}, {r['ratio_hi97_5']:.6f}] |")
    (OUT / "PASSB_P95_RATIOS_97_5.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
