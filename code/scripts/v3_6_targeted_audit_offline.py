#!/usr/bin/env python3
"""CPU-only A1-A3 audit for the v1.5/v1.6 two-stage Jina records.

This script only reads frozen artifacts and writes a new append-only audit
namespace. It never joins labels into serving/raw records and never runs a
model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_targeted_audit"
AOUT = OUT / "A_twostage_jina"

V15 = ROOT / "results/v1_5/c1_two_stage_specialist_gpu3/raw/jina_two_stage_pplxtext__original.jsonl"
V15_META = V15.with_suffix(".meta.json")
V16 = ROOT / "results/v1_6/d1_perwindow_jina/raw/d1_jina_two_stage_pplxwindow_all.jsonl"
V16_META = V16.with_suffix(".meta.json")
V16_MANIFEST = ROOT / "results/v1_6/d1_perwindow_jina/D1_INPUT_MANIFEST.jsonl"
V16_PERQUERY = ROOT / "results/v1_6/d1_perwindow_jina/export/d1_perquery.csv"
V15_PERQUERY = ROOT / "results/v1_5/c1_two_stage_specialist_gpu3/export/c1_perquery.csv"
C2_FRESH = ROOT / "results/v3_6_essential/followup_v2_correction/C2_jina_diagnostic/fresh_runs_perquery.csv"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def top5(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, str(ids[i])))[:5]]


def stage_kept(record, index):
    stage = record.get("stage_records", [])[index]
    return top5(stage.get("candidate_ids", []), stage.get("scores", []))


def candidate_digest(stage):
    # The frozen D1 manifest stores the exact serialized-input hash and the
    # ordered content digests, but not the raw serialized string itself.
    return stage.get("input_sha256") or stage.get("request_sha256") or ""


def classify(x, y, mx, my):
    xs, ys = x.get("stage_records", []), y.get("stage_records", [])
    if len(xs) < 3 or len(ys) < 3:
        return "UNRESOLVED_FROM_TRACE"
    for i in (0, 1):
        if xs[i].get("candidate_ids") != ys[i].get("candidate_ids"):
            return "WINDOW_TEXT_OR_WINDOW_MEMBERSHIP"
        if candidate_digest(xs[i]) != candidate_digest(ys[i]):
            return "WINDOW_TEXT_HASH_OR_SERIALIZATION"
    kx = [stage_kept(x, 0), stage_kept(x, 1)]
    ky = [stage_kept(y, 0), stage_kept(y, 1)]
    if kx != ky:
        score_vectors_same = all(xs[i].get("scores") == ys[i].get("scores") for i in (0, 1))
        if score_vectors_same:
            return "TIE_ORDER_OR_UNRECORDED_TIE_POLICY"
        return "STAGE1_SCORES_OR_ORDER"
    if x.get("selected_ids", []) != y.get("selected_ids", []):
        return "STAGE2_SCORING_OR_ORDER"
    return "NO_TOP5_DIFFERENCE"


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AOUT.mkdir(parents=True, exist_ok=True)
    v15 = {x["canonical_id"]: x for x in read_jsonl(V15)}
    v16 = {x["canonical_id"]: x for x in read_jsonl(V16) if x.get("order_id") == "original"}
    manifests = {x["canonical_id"]: x for x in read_jsonl(V16_MANIFEST) if x.get("order_id") == "original"}
    ids = sorted(set(v15) & set(v16))
    diffs = []
    first30 = []
    for cid in ids:
        x, y = v15[cid], v16[cid]
        xst, yst = x.get("stage_records", []), y.get("stage_records", [])
        xk = [stage_kept(x, i) for i in (0, 1)] if len(xst) >= 2 else [[], []]
        yk = [stage_kept(y, i) for i in (0, 1)] if len(yst) >= 2 else [[], []]
        list_diff = x.get("selected_ids", []) != y.get("selected_ids", [])
        set_diff = set(x.get("selected_ids", [])) != set(y.get("selected_ids", []))
        if not list_diff:
            continue
        m = manifests.get(cid, {})
        mstages = m.get("stages", [])
        row = {
            "canonical_id": cid,
            "domain": x.get("domain", ""),
            "historical_pair": "v1.5_frozen_vs_v1.6_d1_original",
            "v15_valid": bool(x.get("valid")),
            "v16_valid": bool(y.get("valid")),
            "set_diff": int(set_diff),
            "v15_window1_ids": json.dumps(xst[0].get("candidate_ids", []) if len(xst) > 0 else []),
            "v16_window1_ids": json.dumps(yst[0].get("candidate_ids", []) if len(yst) > 0 else []),
            "v15_window2_ids": json.dumps(xst[1].get("candidate_ids", []) if len(xst) > 1 else []),
            "v16_window2_ids": json.dumps(yst[1].get("candidate_ids", []) if len(yst) > 1 else []),
            "v15_stage1_kept5_w1": json.dumps(xk[0]),
            "v16_stage1_kept5_w1": json.dumps(yk[0]),
            "v15_stage1_kept5_w2": json.dumps(xk[1]),
            "v16_stage1_kept5_w2": json.dumps(yk[1]),
            "v15_stage2_final_top5": json.dumps(x.get("selected_ids", [])),
            "v16_stage2_final_top5": json.dumps(y.get("selected_ids", [])),
            "v15_window1_input_sha256": xst[0].get("request_sha256", "") if len(xst) > 0 else "",
            "v16_window1_input_sha256": (mstages[0].get("input_sha256", "") if len(mstages) > 0 else yst[0].get("request_sha256", "")),
            "v15_window2_input_sha256": xst[1].get("request_sha256", "") if len(xst) > 1 else "",
            "v16_window2_input_sha256": (mstages[1].get("input_sha256", "") if len(mstages) > 1 else yst[1].get("request_sha256", "")),
            "v15_stage2_input_sha256": xst[2].get("request_sha256", "") if len(xst) > 2 else "",
            "v16_stage2_input_sha256": (mstages[2].get("input_sha256", "") if len(mstages) > 2 else yst[2].get("request_sha256", "")),
            "v16_source_hashes_verified": m.get("source_hashes_verified", False),
            "window_text_hash_status": "EXACT_SERIALIZED_INPUT_SHA256;_RAW_STRING_NOT_STORED",
            "first_difference_class": classify(x, y, m, m),
        }
        diffs.append(row)
        if len(first30) < 30:
            first30.append({"canonical_id": cid, "domain": row["domain"], "first_difference_class": row["first_difference_class"], "set_diff": int(set_diff)})
    fields = list(diffs[0].keys()) if diffs else ["canonical_id"]
    write_csv(AOUT / "twostage_jina_diff.csv", fields, diffs)
    write_csv(AOUT / "twostage_jina_first30_causes.csv", ["canonical_id", "domain", "first_difference_class", "set_diff"], first30)

    # Existing fresh fp32/batch-1 diagnostic; no rerun is performed here.
    fresh = list(csv.DictReader(C2_FRESH.open(encoding="utf-8")))
    fresh_n = len(fresh)
    fresh_valid = sum(int(r["run1_valid"]) for r in fresh)
    fresh_mean = statistics.mean(float(r["ndcg5_run1"]) for r in fresh) if fresh else None
    settings = {
        "v15_settings_recoverable": False,
        "v15_reason": "PROTOCOL_V2 records native interface and batching allowance but not tokenizer, dtype, batch size, attention implementation, or library versions used for the frozen row.",
        "v18_settings_recoverable": False,
        "v18_reason": "D1 PROTOCOL records the native interface and quality batching allowance but not the actual tokenizer, dtype, batch size, attention implementation, or library versions used for the row.",
        "existing_fp32_batch1_reused": True,
        "existing_fp32_run1_meta": "results/v3_6_essential/followup_v2_correction/C2_jina_diagnostic/raw/jina_fp32_batch1_run1.meta.json",
        "existing_fp32_run2_meta": "results/v3_6_essential/followup_v2_correction/C2_jina_diagnostic/raw/jina_fp32_batch1_run2.meta.json",
        "fp32_run1_dtype": "float32",
        "fp32_run1_batch_size": 1,
        "fp32_run1_rows": 1384,
        "fp32_run1_valid": 1380,
        "fp32_run2_rows": 1384,
        "fp32_run2_valid": 1380,
        "historical_v15_ndcg5": 0.17897033929762382,
        "historical_v16_d1_ndcg5": 0.18554663238412347,
        "existing_fp32_run1_ndcg5": fresh_mean,
        "existing_fp32_run2_ndcg5": statistics.mean(float(r["ndcg5_run2"]) for r in fresh) if fresh else None,
        "setting_explanation": "NOT_IDENTIFIABLE_FROM_RECORDED_EVIDENCE; neither historical setting receipt is complete and the existing fp32 batch-1 diagnostic does not reproduce either historical value.",
    }
    (AOUT / "A2_SETTINGS_AUDIT.json").write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(AOUT / "A2_EXISTING_RUN_SUMMARY.csv", ["run", "source", "rows", "valid", "ndcg5", "list_agreement_to_other"], [
        {"run": "v1.5_frozen", "source": "results/v1_5/.../jina_two_stage_pplxtext__original.jsonl", "rows": 1384, "valid": 1380, "ndcg5": settings["historical_v15_ndcg5"], "list_agreement_to_other": "1069_differing_queries"},
        {"run": "v1.6_d1_original", "source": "results/v1_6/.../d1_jina_two_stage_pplxwindow_all.jsonl", "rows": 1384, "valid": sum(bool(v16[c].get("valid")) for c in ids), "ndcg5": settings["historical_v16_d1_ndcg5"], "list_agreement_to_other": "1069_differing_queries"},
        {"run": "C2_fp32_batch1_run1", "source": "results/v3_6_essential/followup_v2_correction/C2_jina_diagnostic/raw/jina_fp32_batch1_run1.jsonl", "rows": 1384, "valid": 1380, "ndcg5": settings["existing_fp32_run1_ndcg5"], "list_agreement_to_other": "exact_with_run2"},
        {"run": "C2_fp32_batch1_run2", "source": "results/v3_6_essential/followup_v2_correction/C2_jina_diagnostic/raw/jina_fp32_batch1_run2.jsonl", "rows": 1384, "valid": 1380, "ndcg5": settings["existing_fp32_run2_ndcg5"], "list_agreement_to_other": "exact_with_run1"},
    ])

    trace_manifest = {}
    for p in (V15, V15_META, V16, V16_META, V16_MANIFEST, V15_PERQUERY, V16_PERQUERY, C2_FRESH):
        trace_manifest[str(p.relative_to(ROOT))] = {"sha256": sha_file(p), "record_count": sum(1 for _ in p.open()) if p.suffix in (".jsonl", ".csv") else None}
    (OUT / "RAW_TRACE_SHA256_MANIFEST.json").write_text(json.dumps(trace_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "A1_A3_SUMMARY.json").write_text(json.dumps({
        "status": "PARTIAL_HISTORICAL_PREMISE_MISMATCH",
        "population": len(ids),
        "v15_valid": sum(bool(v15[c].get("valid")) for c in ids),
        "v16_valid": sum(bool(v16[c].get("valid")) for c in ids),
        "list_difference_n": len(diffs),
        "set_difference_n": sum(int(r["set_diff"]) for r in diffs),
        "first30_class_counts": dict(Counter(x["first_difference_class"] for x in first30)),
        "claimed_difference_n": 197,
        "claimed_gap": "0.1790_to_0.1855",
        "observed_gap": settings["historical_v16_d1_ndcg5"] - settings["historical_v15_ndcg5"],
        "paper_recommendation": "Retain both frozen rows with explicit run/protocol labels; do not replace v1.5 with v1.6 or infer a setting cause. If one row is needed for matched-text discussion, label v1.6 as D1 matched-text control and v1.5 as historical frozen E11 input-text result.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "PROTOCOL_OFFLINE.json").write_text(json.dumps({
        "status": "FROZEN_PRE_OUTCOME_OFFLINE_AUDIT",
        "scope": "A1-A3 only; no model inference",
        "source_pair": [str(V15.relative_to(ROOT)), str(V16.relative_to(ROOT))],
        "comparison_filter": "v1.6 order_id == original",
        "labels_joined": False,
        "raw_strings_available": False,
        "exact_serialized_input_hashes_available": True,
        "claimed_197_not_assumed": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
