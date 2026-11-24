#!/usr/bin/env python3
"""Freeze the disjoint suffix after the R1 boundary partition.

No model is loaded.  Existing flushed records are copied to a new prefix
namespace, two interrupted in-flight requests are recorded as failures, and
only unseen L=6144/8192 rows are emitted for the three-way continuation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "v3_6_essential" / "followup_v3_repro_fixes"
PREP = BASE / "R1_transfer_budget_grid"
RAW = BASE / "raw"
OUT = BASE / "pplx_suffix"


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def req(q, docs):
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    return {"state": f"Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {q}", "criteria": {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    partials = [RAW / "pplx_shard0_of2.jsonl.part", RAW / "pplx_shard1_of2.jsonl.part"]
    done = {}
    for p in partials:
        rows = [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
        for r in rows:
            done[(r["dataset"], r["canonical_id"], int(r["budget_L"]))] = r
        prefix = RAW / p.name.replace("pplx_", "pplx_prefix_").replace(".jsonl.part", ".jsonl")
        if not prefix.exists():
            prefix.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    all_rows = []
    for L in (6144, 8192):
        all_rows.extend(json.loads(x) for x in (PREP / f"prepared_L{L}.jsonl").open(encoding="utf-8") if x.strip())
    selected_by_shard = []
    cancelled = []
    for shard, p in enumerate(partials):
        ordered = [x for i, x in enumerate([json.loads(y) for y in sum(([z for z in open(PREP / f'prepared_L{L}.jsonl', encoding='utf-8') if z.strip()] for L in (2048,4096,6144,8192)), [])]) if i % 2 == shard]
        # The first unrecorded row in each preserved prefix is the call that
        # received Ctrl-C at the partition boundary.  Never issue it again.
        n = sum(1 for _ in p.open(encoding="utf-8"))
        if n >= len(ordered):
            continue
        x = ordered[n]
        if int(x["length_limit"]) < 6144:
            raise RuntimeError(f"boundary unexpectedly before L6144: shard={shard} L={x['length_limit']}")
        docs = x.get("budgeted_candidates") or []
        cancelled.append({
            "dataset": x["dataset"], "canonical_id": x["canonical_id"], "query_id": x["query_id"], "group_id": x.get("group_id", ""),
            "budget_L": x["length_limit"], "policy": "pplx_single_corrected_builder", "order_id": "original",
            "candidate_ids": [d["candidate_id"] for d in docs], "source_slate_sha256": x["source_slate_sha256"],
            "input_sha256": sha(req(str(x["query"]), docs)), "serialized_tokens": x.get("serialized_tokens"),
            "builder_status": x["budget_info"].get("builder_status"), "labels_joined": False, "input_label_blind": True,
            "model_revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "gpu_index_requested": 3 if shard == 0 else 6,
            "status": "FAIL", "valid": False, "selected_ids": [], "scores": [], "n_calls": 1,
            "physical_forward_count": None, "error_class": "CANCELLED_IN_FLIGHT_FOR_BOUNDARY_PARTITION",
            "error": "Owned worker was stopped at a predeclared disjoint length-partition boundary; this request was not retried.",
            "latency_ms": None, "peak_memory_allocated_mib": None,
        })
        done[(x["dataset"], x["canonical_id"], int(x["length_limit"]))] = cancelled[-1]
    for x in all_rows:
        key = (x["dataset"], x["canonical_id"], int(x["length_limit"]))
        if key not in done:
            selected_by_shard.append(x)
    suffix_path = OUT / "prepared_L6144_L8192_unseen.jsonl"
    suffix_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in selected_by_shard), encoding="utf-8")
    (RAW / "pplx_boundary_cancelled.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in cancelled), encoding="utf-8")
    receipt = {"status": "FROZEN", "prefix_records": {str(p): sum(1 for _ in p.open(encoding='utf-8')) for p in partials}, "cancelled_records": len(cancelled), "unseen_suffix_records": len(selected_by_shard), "suffix_path": str(suffix_path.relative_to(ROOT)), "no_retries": True, "label_blind": True}
    (OUT / "SUFFIX_FREEZE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
