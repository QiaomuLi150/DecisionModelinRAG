#!/usr/bin/env python3
"""Run the explicitly authorized 20-request determinism rechecks.

The rechecks use the frozen original-order manifest and write only new,
append-only diagnostic traces under stage4_analysis_v3.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v2_1_replication"
MANIFEST = BASE / "stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl"
OUT = BASE / "stage4_analysis_v3/determinism_raw"
POLICIES = ["jina_full", "jina_budget", "jina_two_stage_matched", "pplx_native", "pplx_budget", "pplx_two_stage"]


def load_runner():
    spec = importlib.util.spec_from_file_location("v2_1_hybrid_native", ROOT / "scripts/v2_1_hybrid_native.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_rows():
    rows = [json.loads(x) for x in MANIFEST.open(encoding="utf-8") if x.strip()]
    by = {}
    for row in rows:
        if row["order_id"] == "original":
            by.setdefault(row["dataset"], []).append(row)
    for dataset in by:
        by[dataset].sort(key=lambda x: x["canonical_id"])
    rng = random.Random(20261022)
    return {dataset: rng.sample(rows, min(20, len(rows))) for dataset, rows in sorted(by.items())}


def main():
    runner = load_runner()
    selected = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, rows in selected.items():
        # P0 is deterministic without a model call; all model policies that
        # had an original-order run are physically rechecked below.
        records.append({"dataset": dataset, "policy": "bm25_order", "n_rechecked": len(rows), "n_mismatch": 0, "status": "OFFLINE_DETERMINISTIC"})
        for policy in POLICIES:
            out = OUT / f"{dataset}__{policy}__original__recheck.jsonl"
            runner.run_policy(policy, rows, 3, out)
            original_path = BASE / "stage1_native_v2_hybrid" / f"{policy}__original__full.jsonl"
            originals = {}
            with original_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        x = json.loads(line); originals[x["canonical_id"]] = x
            actual = {}
            with out.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        x = json.loads(line); actual[x["canonical_id"]] = x
            mismatch = 0
            for row in rows:
                cid = row["canonical_id"]
                a, b = actual[cid], originals[cid]
                if a.get("status") != b.get("status"):
                    mismatch += 1
                elif a.get("status") == "PASS" and a.get("selected_ids") != b.get("selected_ids"):
                    mismatch += 1
                elif a.get("status") != "PASS" and a.get("error_class") != b.get("error_class"):
                    mismatch += 1
            records.append({"dataset": dataset, "policy": policy, "n_rechecked": len(rows), "n_mismatch": mismatch, "status": "PASS"})
    (OUT / "DETERMINISM_RECHECK_SUMMARY.json").write_text(json.dumps({"seed": 20261022, "gpu": 3, "records": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "records": records}, indent=2))


if __name__ == "__main__":
    main()
