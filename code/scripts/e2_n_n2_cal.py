#!/usr/bin/env python3
"""E2-N N2 CAL validation for verified local, instrumented branches.

The N2 configuration is fixed and recorded before any N3 historical run.  No
gold labels are loaded and no model-selection tuning is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n/n2"
SEED = 20261011


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_items() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_local_cal_items

    items = load_local_cal_items()
    if len(items) != 30 or {x["dataset"] for x in items} != {"hotpot", "musique", "wiki2"}:
        raise RuntimeError(f"expected 30 CAL rows across three datasets, got {len(items)}")
    if len({x["query_id"] for x in items}) != len(items):
        raise RuntimeError("CAL query IDs are not unique")
    return items


def config_for(model: str) -> dict:
    configs = {
        "bge": {
            "model_id": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "adapter": "transformers AutoModelForSequenceClassification",
            "score_contract": "INDEPENDENT_RELEVANCE:v2",
            "precision": "bfloat16",
            "token_max_length": 512,
            "batch_size_candidates": 20,
            "score_direction": "higher_is_more_relevant",
            "tie_break": "original_candidate_order_then_doc_id",
        },
        "laya": {
            "model_id": "convaiinnovations/laya",
            "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
            "adapter": "native laya Router typed relevance",
            "score_contract": "INDEPENDENT_RELEVANCE:v2",
            "precision": "checkpoint native",
            "token_max_length": 512,
            "batch_size_candidates": 4,
            "sort_by_length": True,
            "score_field": "answers.relevance.noul",
            "score_direction": "higher_is_more_relevant",
            "tie_break": "original_candidate_order_then_doc_id",
            "temperature_note": "checkpoint emits invalid/out-of-range temperature metadata; confidence calibration is uncalibrated, ordering retained",
        },
        "clef": {
            "model_id": "Cloudflare/clef-flash",
            "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
            "adapter": "native joint_schema_model",
            "score_contracts": ["INDEPENDENT_RELEVANCE:v2", "JOINT_BINARY_RELEVANCE:v2"],
            "precision": "bfloat16",
            "token_max_length": 16384,
            "pointwise_batch_size_candidates": 4,
            "joint_batch_size": 1,
            "joint_candidate_count": 20,
            "score_direction": "higher_is_more_relevant",
            "tie_break": "original_candidate_order_then_doc_id",
        },
    }
    value = configs[model]
    value["config_sha256"] = sha_json(value)
    return value


def run(model: str, items: list[dict]) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_n_n1_smoke import run_bge, run_clef, run_laya

    if model == "bge":
        return run_bge(items)
    if model == "laya":
        return run_laya(items)
    if model == "clef":
        return run_clef(items)
    raise ValueError(model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("bge", "laya", "clef"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    config = config_for(args.model)
    rows = run(args.model, items)
    trace_path = OUT / f"E2_N_N2_{args.model.upper()}_CAL.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output_rows = [r for r in rows if r.get("status") == "ok"]
    physical_rows = [r for r in output_rows if r.get("physical_forward_evidence") is True]
    hash_rows = [r for r in output_rows if r.get("actual_model_input_sha256")]
    candidate_rows = [r for r in output_rows if len(r.get("candidate_ids", [])) == 20 and len(set(r.get("candidate_ids", []))) == 20]
    expected_rows = 60 if args.model == "clef" else 30
    summary = {
        "stage": "N2_CAL",
        "model": args.model,
        "config_sha256": config["config_sha256"],
        "rows": len(rows),
        "unique_queries": len({r.get("query_id") for r in rows}),
        "ok_rows": len(output_rows),
        "physical_forward_evidence_rows": len(physical_rows),
        "actual_model_input_hash_rows": len(hash_rows),
        "valid_unique_20_candidate_rows": len(candidate_rows),
        "gold_in_inference_input_rows": sum(r.get("gold_in_inference_input") is True for r in rows),
        "cache_reuse_rows": sum(r.get("cache_reuse") is True for r in rows),
        "jev_calls": 0,
        "paid_api_calls": 0,
        "downloads": 0,
        "expected_rows": expected_rows,
        "status": "PASS" if len(rows) == expected_rows and len(output_rows) == len(rows) and len(physical_rows) == len(rows) and len(hash_rows) == len(rows) and len(candidate_rows) == len(rows) else "BLOCKED",
    }
    (OUT / f"E2_N_N2_{args.model.upper()}_CAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / f"E2_N_N2_{args.model.upper()}_CONFIG.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
