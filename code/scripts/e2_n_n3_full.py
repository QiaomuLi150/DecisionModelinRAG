#!/usr/bin/env python3
"""E2-N N3 full historical exploratory scoring.

Only frozen Track A/B slates are consumed.  Gold sidecars are deliberately not
loaded in this inference script.  Clef is reused only from the compatible,
already-completed E2-M1 traces; the source hashes are recorded explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n/n3"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def full_items() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items

    rows = load_full_items()
    if len(rows) != 4267 or len({x["query_id"] for x in rows}) != len(rows):
        raise RuntimeError(f"unexpected frozen Track A/B population: {len(rows)}")
    return rows


def run_local(model: str, items: list[dict]) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_n_n1_smoke import run_bge, run_laya

    rows = run_bge(items) if model == "bge" else run_laya(items)
    for row in rows:
        row["stage"] = "N3_FULL_HISTORICAL"
    return rows


def reuse_clef(items: list[dict]) -> tuple[list[dict], dict]:
    source_point = ROOT / "results/non_jev/v2/e2_m1/E2_M1_POINTWISE_SCORES.jsonl"
    source_joint = ROOT / "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl"
    source_rows = load_jsonl(source_point) + load_jsonl(source_joint)
    if len(source_rows) != 8534:
        raise RuntimeError(f"expected 8534 compatible Clef source rows, got {len(source_rows)}")
    expected_ids = {x["query_id"] for x in items}
    if {x.get("query_id") for x in source_rows} != expected_ids:
        raise RuntimeError("Clef source trace query population does not match frozen Track A/B items")
    rows = []
    for source in source_rows:
        if source.get("status") != "ok" or source.get("gold_in_inference_input") is True:
            raise RuntimeError(f"invalid source Clef row {source.get('query_id')}")
        row = dict(source)
        row["stage"] = "N3_FULL_HISTORICAL_REUSED"
        row["physical_forward_evidence"] = True
        row["actual_model_input_sha256"] = source.get("actual_input_sha256")
        row["raw_response_sha256"] = source.get("response_hash")
        row["trace_origin"] = "compatible_prior_e2_m1_trace"
        row["trace_origin_path"] = str((ROOT / "results/non_jev/v2/e2_m1" / ("E2_M1_POINTWISE_SCORES.jsonl" if source.get("formulation") == "pointwise_binary" else "E2_M1_JOINT_BINARY_SCORES.jsonl")).relative_to(ROOT))
        rows.append(row)
    return rows, {
        "source_point_path": str(source_point.relative_to(ROOT)),
        "source_joint_path": str(source_joint.relative_to(ROOT)),
        "source_point_sha256": sha_file(source_point),
        "source_joint_sha256": sha_file(source_joint),
        "source_rows": len(source_rows),
        "reuse_reason": "same model revision, frozen views, score contracts, candidate slates, and input/response hashes; no incompatible cache reuse",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("bge", "laya", "clef"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = full_items()
    reuse = None
    if args.model == "clef":
        rows, reuse = reuse_clef(items)
    else:
        rows = run_local(args.model, items)
    path = OUT / f"E2_N_N3_{args.model.upper()}_FULL.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ok = [r for r in rows if r.get("status") == "ok"]
    physical = [r for r in ok if r.get("physical_forward_evidence") is True]
    hashes = [r for r in ok if r.get("actual_model_input_sha256") and r.get("raw_response_sha256")]
    candidate_ok = [r for r in ok if len(r.get("candidate_ids", [])) == 20 and len(set(r.get("candidate_ids", []))) == 20]
    expected_rows = 8534 if args.model == "clef" else 4267
    summary = {
        "stage": "N3_FULL_HISTORICAL",
        "model": args.model,
        "frozen_queries": 4267,
        "track_a_queries": 3767,
        "track_b_queries": 500,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "unique_queries": len({r.get("query_id") for r in rows}),
        "ok_rows": len(ok),
        "physical_forward_evidence_rows": len(physical),
        "input_and_response_hash_rows": len(hashes),
        "valid_unique_20_candidate_rows": len(candidate_ok),
        "gold_in_inference_input_rows": sum(r.get("gold_in_inference_input") is True for r in rows),
        "cache_reuse_rows": sum(r.get("cache_reuse") is True for r in rows),
        "trace_reused_from_e2_m1": args.model == "clef",
        "trace_sha256": sha_file(path),
        "jev_calls": 0,
        "paid_api_calls": 0,
        "downloads": 0,
        "status": "PASS" if len(rows) == expected_rows and len(ok) == len(rows) and len(physical) == len(rows) and len(hashes) == len(rows) and len(candidate_ok) == len(rows) else "BLOCKED",
    }
    if reuse:
        summary["reuse_provenance"] = reuse
    (OUT / f"E2_N_N3_{args.model.upper()}_FULL_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "rows": len(rows), "status": summary["status"], "trace_reused_from_e2_m1": summary["trace_reused_from_e2_m1"]}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
