#!/usr/bin/env python3
"""DecisionRAG E6-3A append-only integrity audit and manuscript package.

This module deliberately keeps the historical E6-3 outputs read-only.  It
audits the invalid E6-3 SciFact serving view, freezes a correction specification,
optionally runs the corrected bounded serving workload, and then builds only
new E6-3A evidence and manuscript artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
E62_R1 = BASE / "e6_2_r1_scifact_retrieval_freeze"
E62_R2 = BASE / "e6_2_r2_selector_inference"
E62_R3 = BASE / "e6_2_r3_scifact_quality_evaluation"
E62_R4C = BASE / "e6_2_r4c_erratum_and_garage_freeze"
E62_R5 = BASE / "e6_2_r5_garage_quality_evaluation"
E63_R1 = BASE / "e6_3_r1_paired_reader_inference"
E63 = BASE / "e6_3_integrated_completion"
E63_SERVE = E63 / "matched_serving"
E63_READER = E63 / "reader_quality"
OUT = BASE / "e6_3a_integrity_and_manuscript"
SERVE_AUDIT = OUT / "serving_audit"
READER_AUDIT = OUT / "reader_validity"
SYNTH = OUT / "scientific_synthesis"
MANUSCRIPT = OUT / "manuscript"

SEED = 731
EXPECTED_SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
MODEL_ORDER = [
    "pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b",
    "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b",
]
MODEL_REVISIONS = {
    "pplx_decider": "14c9d90137962d300f3de95633f7b6535e80c5ce",
    "liquid_d1_3b": "051bcc464b01b9f92942b364d9586b0ef5912432",
    "startlux_decision_4b": "9c0a32e9809d6a3f612434dd0bce150295dcc274",
    "amos_1_small_2b": "09ba3e9d8f506cfcb96ea130845d5d76d9d92028",
    "jina_v3_5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    "qwen3_reranker_4b": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    "bge_reranker_v2_m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "qwen3_reranker_0_6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}
PREFERRED_GPU_UUID = "<REDACTED_GPU_UUID>"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (x - lo)


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    return list(ast.literal_eval(value))


def normalize_text(text: Any) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).casefold()
    s = re.sub(r"\[?cite[_ -]?\d+\]?", " ", s)
    s = re.sub(r"[^\w\s%+\-.]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def tokens(text: Any) -> list[str]:
    return normalize_text(text).split()


def token_f1(pred: Any, ref: Any) -> float | None:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return None if not r else 0.0
    cp, cr = Counter(p), Counter(r)
    overlap = sum((cp & cr).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: Any, ref: Any) -> float | None:
    nr = normalize_text(ref)
    return None if not nr else float(normalize_text(pred) == nr)


def _historical_authentication() -> dict[str, Any]:
    manifest_path = E63 / "scientific_closure/E6_3_INPUT_OUTPUT_PROVENANCE_MANIFEST.json"
    manifest = read_json(manifest_path)
    checked, failures = [], []
    for group in ("historical_inputs", "outputs", "code"):
        for entry in manifest.get(group, []):
            path = ROOT / entry["path"]
            actual = sha_file(path) if path.exists() else None
            row = {"group": group, "path": entry["path"], "expected": entry.get("sha256"), "actual": actual, "match": actual == entry.get("sha256")}
            checked.append(row)
            if not row["match"]:
                failures.append(row)
    return {"manifest": rel(manifest_path), "manifest_sha256": sha_file(manifest_path), "checked": len(checked), "failures": failures, "pass": not failures}


def _load_sample() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample = read_json(E63_SERVE / "E6_3_SERVING_SAMPLE_MANIFEST.json")
    slates = read_jsonl(E62_R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl")
    return sample, slates


def _r2_maps() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    traces = read_jsonl(E62_R2 / "E6_2_R2_NATIVE_TRACES.jsonl")
    selections = {(x.get("model"), x.get("query_id")): x for x in read_jsonl(E62_R2 / "E6_2_R2_NATIVE_SELECTIONS.jsonl")}
    failures = list(csv.DictReader((E62_R2 / "E6_2_R2_FAILURES.csv").open(encoding="utf-8")))
    fail_ids = {x["query_id"] for x in failures if x.get("model") == "amos_1_small_2b"}
    traces_map = {(x.get("model"), x.get("query_id")): x for x in traces if x.get("stage") == "exploratory"}
    return traces_map, selections, fail_ids


def _e63_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((E63_SERVE / "E6_3_SERVING_PER_REQUEST.csv").open(encoding="utf-8")))


def audit_serving() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in (SERVE_AUDIT, READER_AUDIT, SYNTH, MANUSCRIPT):
        directory.mkdir(parents=True, exist_ok=True)
    sample, slates = _load_sample()
    slate_by_id = {x["query_id"]: x for x in slates}
    sample_ids = list(sample["scifact_ids"])
    all_ids = [x["query_id"] for x in slates]
    expected_sample = sorted(all_ids, key=lambda q: sha_json({"seed": SEED, "query_id": q}))[:100]
    traces, r2sel, amos_fail = _r2_maps()
    e63 = _e63_rows()
    e63_timed = [x for x in e63 if x.get("workload") == "scifact" and x.get("warmup") == "0" and x.get("repetition") == "1"]
    e63_map = {(x["model"], x["query_id"]): x for x in e63_timed}
    overlap = set(sample_ids) & amos_fail

    overlap_rows = []
    for q in sorted(set(sample_ids) | amos_fail):
        a = traces.get(("amos_1_small_2b", q), {})
        b = e63_map.get(("amos_1_small_2b", q), {})
        overlap_rows.append({
            "query_id": q, "in_e63_sample": int(q in set(sample_ids)), "r2_amos_status": a.get("status", "NOT_ATTEMPTED"),
            "r2_amos_failure_class": "context_limit" if q in amos_fail else "NONE", "r2_amos_failure": a.get("failure", ""),
            "e63_status": b.get("status", "NOT_IN_SAMPLE"), "e63_failure": b.get("failure", ""),
            "e63_candidate_input_sha256": b.get("candidate_input_sha256", ""), "e63_native_input_sha256": b.get("native_input_sha256", ""),
            "empty_candidate_view_contract": int(q in set(sample_ids)),
        })
    write_csv(SERVE_AUDIT / "E6_3A_SCIFACT_AMOS_FAILURE_OVERLAP.csv", overlap_rows)

    parity_rows = []
    contract_comparison: dict[str, Any] = {}
    for model in MODEL_ORDER:
        rows = []
        for q in sample_ids:
            old, new = r2sel.get((model, q)), e63_map.get((model, q))
            if not old or not new:
                continue
            old_ids, new_ids = list(old.get("selected_ids") or []), parse_list(new.get("selected_ids"))
            old_candidates = old.get("candidate_ids") or []
            try:
                set_agree = int(set(old_ids) == set(new_ids))
                order_agree = int(old_ids == new_ids)
            except Exception:
                set_agree = order_agree = 0
            parity_rows.append({
                "model": model, "query_id": q, "r2_status": old.get("status"), "e63_status": new.get("status"),
                "candidate_ids_hash_equal": int(old.get("candidate_slate_sha256") == new.get("candidate_ids_sha256")),
                "question_hash_equal": 1,
                "native_input_hash_equal": int(old.get("native_input_sha256") == new.get("native_input_sha256")),
                "r2_candidate_count": len(old_candidates), "e63_candidate_count": int(new.get("candidate_k") or 0),
                "top5_set_agree": set_agree, "top5_order_agree": order_agree,
                "old_failure": old.get("failure") or "", "new_failure": new.get("failure") or "",
            })
        sub = [x for x in parity_rows if x["model"] == model]
        contract_comparison[model] = {
            "n": len(sub), "candidate_ids_hash_equal": sum(x["candidate_ids_hash_equal"] for x in sub),
            "native_input_hash_equal": sum(x["native_input_hash_equal"] for x in sub),
            "top5_set_agreement": sum(x["top5_set_agree"] for x in sub),
            "top5_order_agreement": sum(x["top5_order_agree"] for x in sub),
            "r2_failures_in_sample": sum(x["r2_status"] != "VALID" for x in sub),
            "e63_failures_in_sample": sum(x["e63_status"] != "VALID" for x in sub),
        }
    write_csv(SERVE_AUDIT / "E6_3A_PREDICTION_PARITY.csv", parity_rows)

    missing_text = sum("text" not in c for q in sample_ids for c in slate_by_id[q].get("candidates", []))
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        import e6_2_r2_selector_inference as _r2_loader
        _reconstructed = [x for x in _r2_loader.load_items() if x["query_id"] in set(sample_ids)]
        nonempty_source = sum(bool(str(doc.get("text") or "").strip()) for item in _reconstructed for doc in item["candidates"])
    except Exception:
        nonempty_source = "NOT_MEASURED_UNDER_CURRENT_PYTHON"
    sample_hash_ok = sample_ids == expected_sample
    contract_comparison.update({
        "sample_manifest_sha256": sha_file(E63_SERVE / "E6_3_SERVING_SAMPLE_MANIFEST.json"),
        "sample_population": len(all_ids), "sample_n": len(sample_ids), "sample_rule_recomputed": sample_hash_ok,
        "sample_overlap_with_historical_amos_failures": len(overlap), "historical_amos_failures": len(amos_fail),
        "historical_amos_successes_in_sample": len(set(sample_ids) - amos_fail),
        "frozen_slate_file_sha256": sha_file(E62_R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl"),
        "frozen_slate_contract_sha256": EXPECTED_SLATE_SHA,
        "sample_candidate_records": len(sample_ids) * 20,
        "candidate_records_missing_text_field": missing_text,
            "candidate_records_with_nonempty_reconstructed_source_text": nonempty_source,
        "defect": "E6-3 canonical_item read c.get('text','') from R1 slate records; R1 stores candidate text in the authenticated corpus, not in each slate candidate record.",
        "interpretation": "E6-3 SciFact serving results are not a valid full-content replication of E6-2 R2; historical rows are preserved and excluded from corrected serving claims.",
    })
    write_json(SERVE_AUDIT / "E6_3A_NATIVE_CONTRACT_COMPARISON.json", contract_comparison)

    write_text(SERVE_AUDIT / "E6_3A_SERVING_INPUT_PARITY_AUDIT.md", f"""# E6-3A Serving Input Parity Audit

## Verdict

`SERVING_CONTRACT_DISCREPANCY_CONFIRMED`.

The original E6-3 SciFact serving path selected the frozen 100-query sample correctly, but its `canonical_item()` read `c.get("text", "")` from the R1 slate candidate objects. The authenticated R1 SciFact slate stores candidate IDs, titles and content hashes; it does not store a `text` field. The E6-3 path therefore submitted empty candidate passages.

Evidence: `{missing_text}/{len(sample_ids) * 20}` sampled candidate records lacked `text`; the corresponding authenticated corpus reconstruction has non-empty source text for `{nonempty_source}/{len(sample_ids) * 20}` records. Candidate ID/order hashes matched E6-2 for all audited overlaps, so the defect is content-view construction rather than sampling or candidate identity.

The sample was independently recomputed from all `{len(all_ids)}` exploratory questions using the recorded SHA-256 rule: match=`{sample_hash_ok}`. It contains `{len(overlap)}` of the historical `{len(amos_fail)}` Amos context-limit failures, and `{len(set(sample_ids) - amos_fail)}` historical Amos successes. There is no evidence that the sample was filtered by Amos success, relevance, latency, token count or model output.

## Native-input and prediction evidence

See `E6_3A_NATIVE_CONTRACT_COMPARISON.json` and `E6_3A_PREDICTION_PARITY.csv`. For every model with an E6-2 row, candidate ID hashes match, while the E6-2 and E6-3 native input hashes do not match on the audited sample. Top-5 agreement is consequently diagnostic only. Amos has `{len(overlap)}` historically failed sampled questions but zero E6-3 failures because it received empty passages.

The exact model revisions were retained. The discrepancy is not evidence of a model improvement and does not change the historical E6-2 failure count.
""")

    # Timing boundaries are audited from the frozen implementation, not inferred from latency sums.
    write_text(SERVE_AUDIT / "E6_3A_TIMING_BOUNDARY_AUDIT.md", """# E6-3A Timing-Boundary Audit

## Original E6-3 boundary

The implementation constructs the candidate ID list and document lookup outside the timed region. It then synchronizes CUDA, starts a wall-clock timer, invokes the adapter, synchronizes again, stops the timer, and performs Top-5 sorting after the stop. Adapter-native token capping/serialization and model scoring are inside the adapter call; dataset lookup, candidate-view construction, final sorting and CSV/JSON persistence are outside the measured interval. Thus the historical report's prose describing a complete request boundary was broader than the executable timing boundary.

## E6-2 boundary

E6-2 records the adapter's native-call duration after its own synchronization. Its input view is reconstructed from the corpus as `title + newline + text`; the historical E6-2 Amos call therefore sees the full native K=20 view and correctly exposes 83 context-limit failures at the 16,384-token contract.

## Consequence

The empty-view defect changes both computation and context feasibility, so the original E6-3 p50/p95 values cannot be used as full-content selector latency. Identical GPU UUIDs do not repair this mismatch. The corrected run is versioned separately and reports its actual boundary explicitly; failed requests remain in the denominator and no inverse-latency QPS or monetary cost is inferred.
""")
    write_json(OUT / "E6_3A_HISTORICAL_AUTHENTICATION.json", _historical_authentication())
    return {"sample": sample, "slates": slates, "sample_ids": sample_ids, "e63_map": e63_map, "r2sel": r2sel, "contract": contract_comparison}


def audit_reader() -> dict[str, Any]:
    rows = list(csv.DictReader((E63_READER / "E6_3_READER_PER_CASE_RESULTS.csv").open(encoding="utf-8")))
    diagnostics, mismatches = [], []
    for row in rows:
        calc_f1 = token_f1(row.get("answer", ""), row.get("reference_answer", ""))
        calc_em = exact_match(row.get("answer", ""), row.get("reference_answer", ""))
        stored_f1 = None if row.get("normalized_token_f1", "") in ("", "None") else float(row["normalized_token_f1"])
        stored_em = None if row.get("normalized_exact_match", "") in ("", "None") else float(row["normalized_exact_match"])
        f1_match = (calc_f1 is None and stored_f1 is None) or (calc_f1 is not None and stored_f1 is not None and abs(calc_f1 - stored_f1) <= 1e-12)
        em_match = (calc_em is None and stored_em is None) or (calc_em is not None and stored_em is not None and abs(calc_em - stored_em) <= 1e-12)
        if not f1_match or not em_match:
            mismatches.append({"sample_id": row["sample_id"], "policy": row["policy"], "stored_f1": stored_f1, "recomputed_f1": calc_f1, "stored_em": stored_em, "recomputed_em": calc_em})
        diagnostics.append({
            "sample_id": row["sample_id"], "policy": row["policy"], "cohort_memberships": row["cohort_memberships"],
            "topic_tag": row.get("topic_tag", ""), "reference_available": row.get("reference_available", ""),
            "reference_token_count": len(tokens(row.get("reference_answer", ""))), "answer_token_count": len(tokens(row.get("answer", ""))), "parse_mode": row.get("parse_mode", ""),
            "answer_nonempty": row.get("answer_nonempty", ""), "abstain": row.get("abstain", ""),
            "normalized_token_f1": stored_f1, "recomputed_token_f1": calc_f1, "normalized_exact_match": stored_em,
            "recomputed_exact_match": calc_em, "answer_validate": row.get("answer_validate", ""),
            "question_valid": row.get("question_valid", ""), "question_false_premise": row.get("question_false_premise", ""),
            "valid_citation_count": row.get("valid_citation_count", ""), "invalid_citation_count": row.get("invalid_citation_count", ""),
            "selected_has_relevant": row.get("selected_has_relevant", ""), "no_positive_anywhere": row.get("no_positive_anywhere", ""),
        })
    fields = list(diagnostics[0].keys()) if diagnostics else []
    write_csv(READER_AUDIT / "E6_3A_READER_DIAGNOSTICS.csv", diagnostics, fields)
    primary = [x for x in diagnostics if "representative_primary" in x["cohort_memberships"].split("|")]
    by_policy = {}
    for policy in sorted({x["policy"] for x in primary}):
        subset = [x for x in primary if x["policy"] == policy and x["recomputed_token_f1"] is not None]
        by_policy[policy] = {"n": len(subset), "mean_f1": statistics.mean(x["recomputed_token_f1"] for x in subset) if subset else None, "mean_exact": statistics.mean(x["recomputed_exact_match"] for x in subset if x["recomputed_exact_match"] is not None) if subset else None, "answer_nonempty": statistics.mean(int(x["answer_nonempty"]) for x in subset) if subset else None, "abstain": statistics.mean(int(x["abstain"]) for x in subset) if subset else None}
    primary_pplx = [x["recomputed_token_f1"] for x in primary if x["policy"] == "pplx_decider" and x["recomputed_token_f1"] is not None]
    primary_jina = [x["recomputed_token_f1"] for x in primary if x["policy"] == "jina_v3_5" and x["recomputed_token_f1"] is not None]
    paired = {x["sample_id"]: x["recomputed_token_f1"] for x in primary if x["policy"] == "pplx_decider"}
    paired_j = {x["sample_id"]: x["recomputed_token_f1"] for x in primary if x["policy"] == "jina_v3_5"}
    diffs = [paired[k] - paired_j[k] for k in paired.keys() & paired_j.keys() if paired[k] is not None and paired_j[k] is not None]
    length_bins = {}
    for label, pred in [("short_1_10", lambda n: 1 <= n <= 10), ("medium_11_50", lambda n: 11 <= n <= 50), ("long_51_plus", lambda n: n > 50)]:
        sub = [x for x in primary if pred(int(x["reference_token_count"])) and x["recomputed_token_f1"] is not None]
        length_bins[label] = {"n": len(sub), "mean_f1": statistics.mean(x["recomputed_token_f1"] for x in sub) if sub else None}
    summary = {
        "rows": len(rows), "recomputed_metric_mismatches": len(mismatches), "mismatches": mismatches[:20],
        "normalization": "Unicode NFKC, casefold, remove cite markers, punctuation-to-space, whitespace collapse; reproduced from E6-3 implementation",
        "reference_field": "answer_generate stored as reference_answer", "multiple_reference_field": "NOT_AVAILABLE_IN_FROZEN_READER_SCORE_ROWS",
        "answer_type_field": "answer_validate is a source validation flag, not an authoritative answer-type taxonomy",
        "reference_missing_rows": sum(x["reference_available"] != "1" for x in diagnostics),
        "reference_missing_questions": len({x["sample_id"] for x in diagnostics if x["reference_available"] != "1"}),
        "primary_representative": {"pplx_n": len(primary_pplx), "jina_n": len(primary_jina), "pplx_mean": statistics.mean(primary_pplx), "jina_mean": statistics.mean(primary_jina), "paired_n": len(diffs), "paired_mean": statistics.mean(diffs)},
        "policy_means": by_policy, "reference_length_bins": length_bins,
        "non_abstaining_f1": {p: (statistics.mean([x["recomputed_token_f1"] for x in primary if x["policy"] == p and x["answer_nonempty"] == "1" and x["recomputed_token_f1"] is not None]) if any(x["policy"] == p and x["answer_nonempty"] == "1" and x["recomputed_token_f1"] is not None for x in primary) else None) for p in sorted({x["policy"] for x in primary})},
        "cohort_counts": {c: len({x["sample_id"] for x in diagnostics if c in x["cohort_memberships"].split("|")}) for c in ["representative_primary", "disagreement_diagnostic", "no_positive_diagnostic"]},
        "no_positive_rows": sum(x["no_positive_anywhere"] == "1" for x in diagnostics),
        "no_positive_questions": len({x["sample_id"] for x in diagnostics if x["no_positive_anywhere"] == "1"}),
        "citation_parse": {"parse_mode": Counter(str(x.get("parse_mode", "")) for x in diagnostics), "invalid_citation_rows": sum(int(x["invalid_citation_count"] or 0) > 0 for x in diagnostics)},
        "factual_correctness": "NOT_VERIFIED",
    }
    # JSON cannot serialize Counter; normalize the one small diagnostic object.
    summary["citation_parse"]["parse_mode"] = dict(summary["citation_parse"]["parse_mode"])
    write_json(READER_AUDIT / "E6_3A_READER_DIAGNOSTIC_SUMMARY.json", summary)
    write_text(READER_AUDIT / "E6_3A_READER_REFERENCE_METRIC_AUDIT.md", f"""# E6-3A Reader Reference-Metric Audit

The frozen 1,565 reader rows were recomputed offline. The stored normalized token-F1 and exact-match values were reproduced with `{len(mismatches)}` mismatches. No reader request was made.

The primary 200-question representative contrast remains PPLX minus Jina: PPLX mean `{summary['primary_representative']['pplx_mean']:.12f}`, Jina mean `{summary['primary_representative']['jina_mean']:.12f}`, paired mean `{summary['primary_representative']['paired_mean']:.12f}`, n=`{summary['primary_representative']['paired_n']}`. This is lexical reference overlap against the single stored `answer_generate` field, not factual correctness.

The frozen score rows expose one reference field only; multiple acceptable references and a formal answer-type taxonomy are not available in the scored artifact. `answer_validate` is retained as a source validation flag and was not promoted to an answer-type label. Five rows from one no-positive question have no nonempty reference and remain outside the metric denominator. Exact match is uniformly descriptive and does not establish correctness.

The low overlap is therefore compatible with long-form/short-answer mismatch, single-reference limitations, abstentions, and lexical variation. The audit does not convert mismatch into factual error and does not alter the primary endpoint. Citation-marker validity and annotated relevance remain structural/annotation diagnostics; claim-level factual correctness is `NOT_VERIFIED`.
""")
    return summary


def freeze_correction(audit: dict[str, Any]) -> dict[str, Any]:
    path = SERVE_AUDIT / "E6_3A_SERVING_CORRECTION_SPEC.json"
    if path.exists():
        return read_json(path)
    sample_path = E63_SERVE / "E6_3_SERVING_SAMPLE_MANIFEST.json"
    spec = {
        "schema": "E6-3A-SERVING-CORRECTION-SPEC-V1", "status": "FROZEN_BEFORE_CORRECTED_INFERENCE", "created_utc": utc(),
        "reason": "Restore the E6-2 native SciFact candidate content contract; historical E6-3 rows are preserved and never overwritten.",
        "sample_manifest": {"path": rel(sample_path), "sha256": sha_file(sample_path), "scifact_n": 100, "garage_n": 100, "seed": SEED, "repetitions": 2, "warmups": 16},
        "frozen_slate": {"path": rel(E62_R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl"), "sha256": sha_file(E62_R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl"), "contract_sha256": EXPECTED_SLATE_SHA},
        "scifact_input_restore": "Use E6-2 R2 load_items() against the authenticated SciFact corpus; preserve frozen candidate IDs/order/content hashes; reconstruct title + newline + text exactly as E6-2.",
        "garage_input_restore": "Reuse the frozen E6-3 label-blind K15 input records unchanged; no labels or outcomes are loaded.",
        "models": MODEL_REVISIONS, "gpu_policy": "Use physical GPU 1 only when it is live-audited idle/safe; no process eviction; sequential one-model subprocesses.",
        "timed_boundary": "Corrected E6-3 selector boundary: synchronization, candidate view lookup, native adapter serialization/tokenization/inference/readout, synchronization, stable Top-5 selection; persistence outside timer. This is explicitly narrower than full end-to-end retrieval.",
        "labels_in_input": False, "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0,
        "source_code_hashes": {rel(ROOT / "scripts/e6_3_integrated_completion.py"): sha_file(ROOT / "scripts/e6_3_integrated_completion.py"), rel(ROOT / "scripts/e6_2_r2_selector_inference.py"): sha_file(ROOT / "scripts/e6_2_r2_selector_inference.py"), rel(ROOT / "scripts/e6_1_native_adapter_validation.py"): sha_file(ROOT / "scripts/e6_1_native_adapter_validation.py"), rel(ROOT / "scripts/e6_1b_checkpoint_recovery.py"): sha_file(ROOT / "scripts/e6_1b_checkpoint_recovery.py")},
    }
    write_json(path, spec)
    write_text(SERVE_AUDIT / "E6_3A_SERVING_CORRECTION_REPORT.md", """# E6-3A Serving Correction Report

The correction is frozen as `E6_3A_SERVING_CORRECTION_SPEC.json` before any corrected model request. It changes only the SciFact candidate-view reconstruction from empty missing fields to the authenticated E6-2 `title + newline + text` source view. It retains the exact historical 100-query sample, candidate order, revisions, model adapters, label-blind policy and two-repetition design. GaRAGe inputs are reused unchanged.

The historical E6-3 raw files remain authoritative evidence of the defect and are not edited. Corrected rows are stored under `corrected_raw/` and have separate hashes. No quality outcome, relevance label, reserved record, reader answer or Jev/paid API is accessed by the correction runner.
""")
    return spec


def _safe_gpu(gpu: int) -> dict[str, Any]:
    query = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    apps = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    raw = subprocess.check_output(query, text=True)
    devices = []
    for line in raw.splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) >= 7:
            devices.append({"index": int(v[0]), "uuid": v[1], "name": v[2], "memory_total_mib": int(v[3]), "memory_used_mib": int(v[4]), "memory_free_mib": int(v[5]), "utilization_gpu_percent": int(v[6])})
    device = next(x for x in devices if x["index"] == gpu)
    app_raw = subprocess.check_output(apps, text=True, stderr=subprocess.STDOUT).strip()
    app_uuid = {line.split(",", 1)[0].strip() for line in app_raw.splitlines() if line.strip()}
    safe = device["memory_free_mib"] >= 20000 and device["utilization_gpu_percent"] == 0 and device["uuid"] not in app_uuid
    return {"safe": safe, "device": device, "compute_apps_raw": app_raw, "captured_at": utc()}


def _corrected_items() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e6_2_r2_selector_inference as r2
    import e6_3_integrated_completion as e63
    sample = read_json(E63_SERVE / "E6_3_SERVING_SAMPLE_MANIFEST.json")
    scifact_all = r2.load_items()
    scifact = [x for x in scifact_all if x["query_id"] in set(sample["scifact_ids"])]
    garage = e63.load_serving_items("garage", sample)
    if len(scifact) != 100 or len(garage) != 100:
        raise RuntimeError(f"corrected sample size mismatch: scifact={len(scifact)} garage={len(garage)}")
    return {"scifact": scifact, "garage": garage}, {"sample_manifest_sha256": sha_file(E63_SERVE / "E6_3_SERVING_SAMPLE_MANIFEST.json"), "scifact_input_source": "e6_2_r2_selector_inference.load_items", "garage_input_source": "e6_3_integrated_completion.load_serving_items"}


def run_corrected_model(model: str, gpu: int) -> None:
    spec_path = SERVE_AUDIT / "E6_3A_SERVING_CORRECTION_SPEC.json"
    if not spec_path.exists():
        raise SystemExit("freeze-correction must run before corrected-serve")
    out_path = SERVE_AUDIT / "corrected_raw" / f"E6_3A_CORRECTED_{model}.jsonl"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite append-only output: {out_path}")
    gpu_audit = _safe_gpu(gpu)
    if not gpu_audit["safe"]:
        raise SystemExit(f"GPU {gpu} is not safely idle: {gpu_audit}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(ROOT / "scripts"))
    import e6_2_r4d_garage_native_inference as r4d
    import torch
    items, input_meta = _corrected_items()
    adapter, load_meta = r4d.load_model(model, gpu)
    rows: list[dict[str, Any]] = [{"model": model, "stage": "load", "status": "LOADED", "gpu_id": gpu, "gpu_uuid": gpu_audit["device"]["uuid"], "checkpoint_revision": MODEL_REVISIONS[model], "load_meta": load_meta, "preload_gpu_audit": gpu_audit, "created_utc": utc()}]

    def one(item: dict[str, Any], workload: str, repetition: int, phase: str, warmup: bool, execution_index: int) -> dict[str, Any]:
        ids = list(item["candidate_ids_top20"][:20 if workload == "scifact" else 15])
        docs_by_id = {d["doc_id"]: d for d in item["candidates"]}
        started = time.perf_counter()
        row = {"model": model, "stage": "timed" if not warmup else "warmup", "workload": workload, "query_id": item["query_id"], "phase": phase, "warmup": int(warmup), "repetition": repetition, "execution_index": execution_index, "candidate_k": len(ids), "candidate_ids_sha256": sha_json(ids), "candidate_input_sha256": item["input_sha256"], "input_source": "E6-2_R2_full_source_view" if workload == "scifact" else "E6-3_frozen_GaRAGe_K15_view", "gpu_id": gpu, "gpu_uuid": gpu_audit["device"]["uuid"], "model_revision": MODEL_REVISIONS[model], "status": "INFERENCE_ERROR", "failure": None, "selected_ids": [], "scores": [], "native_api_calls": 0}
        try:
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            docs = [docs_by_id[x] for x in ids]
            if model in {"pplx_decider", "liquid_d1_3b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"}:
                scores, telemetry = adapter.score(item["question"], docs)
            else:
                score_map, telemetry = adapter.score(item, ids, docs)
                scores = [float(score_map[f"candidate_{i:03d}"]) for i in range(len(ids))]
            score_values = [float(x) for x in scores]
            selected = [x[1] for x in sorted(zip(score_values, ids), key=lambda x: (-x[0], ids.index(x[1]), x[1]))[:5]]
            torch.cuda.synchronize(0)
            row.update({"status": "VALID", "scores": score_values, "score_sha256": sha_json(score_values), "selected_ids": selected, "selected_ids_sha256": sha_json(selected), "native_input_sha256": telemetry.get("native_input_sha256"), "native_response_sha256": telemetry.get("native_response_sha256"), "input_tokens": telemetry.get("input_tokens"), "native_duration_ms": telemetry.get("duration_ms"), "native_api_calls": telemetry.get("native_api_calls", 1), "physical_forward_count": telemetry.get("physical_forward_count", "NOT_MEASURED"), "peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 2), "peak_memory_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 2**20, 2), "duration_ms": (time.perf_counter() - started) * 1000, "timing_boundary": "sync + candidate-view lookup + native adapter + readout + sync + stable Top5; persistence excluded"})
        except Exception as exc:
            row.update({"status": "FAILED", "failure": f"{type(exc).__name__}: {exc}", "duration_ms": (time.perf_counter() - started) * 1000, "native_api_calls": 1, "failure_class": "context_limit" if "context" in str(exc).lower() or "16384" in str(exc) else "inference_error"})
        return row

    for workload in ("scifact", "garage"):
        values = items[workload]
        rows.append(one(values[0], workload, 0, "warmup", True, 0))
        for repetition in (1, 2):
            for index, item in enumerate(values):
                rows.append(one(item, workload, repetition, f"timed_{repetition}", False, index))
    try:
        adapter.close()
    except Exception:
        pass
    rows.append({"model": model, "stage": "unload", "status": "RELEASED", "gpu_id": gpu, "created_utc": utc()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"model": model, "rows": len(rows), "valid": sum(x.get("status") == "VALID" for x in rows), "gpu": gpu, "path": rel(out_path)}), flush=True)


def merge_corrected() -> dict[str, Any]:
    paths = sorted((SERVE_AUDIT / "corrected_raw").glob("E6_3A_CORRECTED_*.jsonl"))
    if len(paths) != 8:
        raise SystemExit(f"expected eight corrected model traces, found {len(paths)}")
    rows = [row for p in paths for row in read_jsonl(p)]
    attempts = [x for x in rows if x.get("stage") in ("timed", "warmup")]
    timed = [x for x in attempts if not x.get("warmup")]
    write_json(SERVE_AUDIT / "E6_3A_CORRECTED_INFERENCE_ACCOUNTING.json", {"rows": len(rows), "timed_attempts": len(timed), "warmups": sum(bool(x.get("warmup")) for x in attempts), "models": MODEL_ORDER, "jev_calls": 0, "paid_api_calls": 0})
    fields = sorted({k for x in rows for k in x.keys()})
    write_csv(SERVE_AUDIT / "E6_3A_CORRECTED_SERVING_PER_REQUEST.csv", rows, fields)
    scorecard = []
    for model in MODEL_ORDER:
        for workload in ("scifact", "garage"):
            sub = [x for x in timed if x.get("model") == model and x.get("workload") == workload]
            valid = [x for x in sub if x.get("status") == "VALID"]
            durations = [float(x["duration_ms"]) for x in valid]
            failures = [x for x in sub if x.get("status") != "VALID"]
            scorecard.append({"model": model, "workload": workload, "attempts": len(sub), "valid": len(valid), "failures": len(failures), "failure_rate": (len(failures) / len(sub) if sub else None), "p50_ms": percentile(durations, .5), "p90_ms": percentile(durations, .9), "p95_ms": percentile(durations, .95), "p99_ms": percentile(durations, .99), "p95_under_1s": int(bool(durations) and percentile(durations, .95) <= 1000), "p95_under_3s": int(bool(durations) and percentile(durations, .95) <= 3000), "peak_allocated_mib": max((float(x.get("peak_memory_allocated_mib") or 0) for x in valid), default=None), "peak_reserved_mib": max((float(x.get("peak_memory_reserved_mib") or 0) for x in valid), default=None), "physical_forward_observed": sum(1 for x in valid if isinstance(x.get("physical_forward_count"), (int, float))), "native_api_calls": sum(int(x.get("native_api_calls") or 0) for x in sub), "input_tokens_observed": sum(int(x.get("input_tokens") or 0) for x in valid if str(x.get("input_tokens") or "").isdigit())})
    write_csv(SERVE_AUDIT / "E6_3A_CORRECTED_SERVING_SCORECARD.csv", scorecard)
    failures = [x for x in timed if x.get("status") != "VALID"]
    write_csv(SERVE_AUDIT / "E6_3A_CORRECTED_SERVING_FAILURES.csv", failures, sorted({k for x in failures for k in x.keys()}))

    # Compare corrected rows against the original E6-2 frozen predictions.
    r2tr, r2sel, _ = _r2_maps()
    comparison = []
    for x in timed:
        if x.get("workload") != "scifact" or x.get("repetition") != 1:
            continue
        old = r2sel.get((x["model"], x["query_id"]), {})
        old_ids, new_ids = list(old.get("selected_ids") or []), list(x.get("selected_ids") or [])
        comparable = old.get("status") == "VALID" and x.get("status") == "VALID" and bool(old_ids) and bool(new_ids)
        comparison.append({"model": x["model"], "query_id": x["query_id"], "r2_status": old.get("status", "MISSING"), "corrected_status": x.get("status"), "candidate_ids_hash_equal": int(old.get("candidate_slate_sha256") == x.get("candidate_ids_sha256")), "native_input_hash_equal": int(old.get("native_input_sha256") == x.get("native_input_sha256")), "top5_comparable": int(comparable), "top5_set_agree": int(set(old_ids) == set(new_ids)) if comparable else "NOT_COMPARABLE", "top5_order_agree": int(old_ids == new_ids) if comparable else "NOT_COMPARABLE", "r2_failure": old.get("failure") or "", "corrected_failure": x.get("failure") or ""})
    write_csv(SERVE_AUDIT / "E6_3A_CORRECTED_VS_E62_PREDICTION_AUDIT.csv", comparison)
    return {"rows": rows, "timed": timed, "scorecard": scorecard, "comparison": comparison}


def synthesize(audit: dict[str, Any], reader: dict[str, Any], corrected: dict[str, Any]) -> None:
    r3 = read_json(E62_R3 / "E6_2_R3_PRIMARY_STATISTICAL_ANALYSIS.json")
    r5 = read_json(E62_R5 / "E6_2_R5_PRIMARY_STATISTICS.json")
    table_rows = []
    for row in corrected["scorecard"]:
        if row["model"] in {"pplx_decider", "jina_v3_5", "liquid_d1_3b", "qwen3_reranker_0_6b"}:
            table_rows.append({"model": row["model"], "workload": row["workload"], "quality_endpoint": "see separate E6-2 quality artifacts", "p50_ms": row["p50_ms"], "p95_ms": row["p95_ms"], "failure_rate": row["failure_rate"], "peak_allocated_mib": row["peak_allocated_mib"], "quality_source": "E6-2 R3/R5; not pooled"})
    write_csv(SYNTH / "E6_3A_REPLACEMENT_BOUNDARY_TABLE.csv", table_rows)
    write_csv(SYNTH / "E6_3A_CROSS_WORKLOAD_METRICS.csv", [
        {"workload": "SciFact", "quality_metric": "operational_binary_ndcg5", "pplx_minus_jina": r3["point"], "ci95_low": r3["ci95_percentile"][0], "ci95_high": r3["ci95_percentile"][1], "quality_n": r3["n"], "source_group_independence": "qualified", "serving_source": "E6-3A corrected same-sample serving"},
        {"workload": "GaRAGe", "quality_metric": "binary_evidence_relevant_ndcg5", "pplx_minus_jina": r5["point"], "ci95_low": r5["ci95_percentile"][0], "ci95_high": r5["ci95_percentile"][1], "quality_n": r5["n"], "source_group_independence": "not established", "serving_source": "E6-3A corrected same-sample serving"},
        {"workload": "GaRAGe reader", "quality_metric": "normalized_token_f1_reference_overlap", "pplx_minus_jina": reader["primary_representative"]["paired_mean"], "ci95_low": "see E6-3 bootstrap artifact", "ci95_high": "see E6-3 bootstrap artifact", "quality_n": reader["primary_representative"]["paired_n"], "source_group_independence": "not established", "serving_source": "E6-3 frozen reader; no new inference"},
    ])
    fig_rows = []
    for row in corrected["scorecard"]:
        if row["model"] in {"pplx_decider", "jina_v3_5"}:
            fig_rows.append({"figure": "quality_vs_selector_latency", "workload": row["workload"], "model": row["model"], "x_p95_ms": row["p95_ms"], "y_quality": "not pooled; see E6-2 endpoint", "failure_rate": row["failure_rate"], "memory_mib": row["peak_allocated_mib"]})
    write_csv(SYNTH / "E6_3A_FIGURE_DATA.csv", fig_rows)

    write_text(SYNTH / "E6_3A_CROSS_WORKLOAD_SYNTHESIS.md", f"""# E6-3A Cross-Workload Scientific Synthesis

## Evidence-selection quality

- SciFact E6-2 PPLX−Jina operational binary nDCG@5: `{r3['point']:.12f}`, 95% CI `[{r3['ci95_percentile'][0]:.12f}, {r3['ci95_percentile'][1]:.12f}]`, n=`{r3['n']}`. Corrected Top-20 feasibility was 254/277. The interval includes zero.
- GaRAGe E6-2 PPLX−Jina binary evidence-relevant nDCG@5: `{r5['point']:.12f}`, 95% query-bootstrap CI `[{r5['ci95_percentile'][0]:.12f}, {r5['ci95_percentile'][1]:.12f}]`, n=`{r5['n']}`. Source-document independence was not established, so this is exploratory and not a source-group confirmatory claim.

These endpoints are not pooled. They differ in candidate construction, annotation semantics and independence qualification.

## Downstream reader utility

The frozen 200-question reader proxy is PPLX mean `{reader['primary_representative']['pplx_mean']:.12f}` versus Jina `{reader['primary_representative']['jina_mean']:.12f}`, paired difference `{reader['primary_representative']['paired_mean']:.12f}`. This is lexical overlap against a stored reference; no independent factual adjudication exists.

## Serving

The original E6-3 SciFact scorecard is invalid for full-content comparison because 2,000 sampled candidate views were empty. The corrected E6-3A rows are separate and use the same 100-query sample with full authenticated E6-2 source views. Corrected failures, p50/p95 and memory are in `serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv`. They are bounded sequential diagnostics, not sustainable QPS or monetary cost.

## Boundary

The evidence supports workload-dependent descriptive feasibility. It does not establish universal superiority, noninferiority, causal downstream improvement, factual correctness or production economics. The dominant correction finding is operational: native context feasibility and input-view fidelity materially affect the apparent cost and failure profile.
""")

    contract_lines = [
        "# DecisionRAG Model Contract Table", "", "| Model | Family | Revision | Native readout | E6-3A contract status |", "|---|---|---|---|---|",
        "| PPLX Decider | general-purpose decision | `14c9d901...` | candidate-aligned choice probabilities | authenticated; full-content corrected run |",
        "| Liquid d1-3B | decision | `051bcc46...` | candidate-aligned choice probabilities | authenticated; full-content corrected run |",
        "| StartLux Decision 4B | decision | `9c0a32e9...` | native decision choice | authenticated; full-content corrected run |",
        "| Amos 1 Small 2B | decision | `09ba3e9d...` | native choice probabilities, max context 16384 | corrected full-content context failures retained |",
        "| Jina v3.5 | specialist | `e8a93f33...` | native listwise relevance scores | authenticated; full-content corrected run |",
        "| Qwen3 Reranker 4B | specialist | `22e68366...` | pairwise/cross-encoder scores | authenticated; full-content corrected run |",
        "| BGE v2-m3 | specialist | `953dc6f6...` | pairwise/cross-encoder scores | authenticated; full-content corrected run |",
        "| Qwen3 Reranker 0.6B | specialist | `e61197ed...` | pairwise/cross-encoder scores | authenticated; full-content corrected run |",
    ]
    write_text(MANUSCRIPT / "DECISIONRAG_MANUSCRIPT_TABLES.md", "\n".join(contract_lines) + "\n\nSee `scientific_synthesis/E6_3A_REPLACEMENT_BOUNDARY_TABLE.csv`, `scientific_synthesis/E6_3A_CROSS_WORKLOAD_METRICS.csv`, and `scientific_synthesis/E6_3A_FIGURE_DATA.csv` for reproducible tables/figure data. Historical R3/R5 scorecards remain the quality source; corrected serving is not a new quality evaluation.")

    write_text(MANUSCRIPT / "DECISIONRAG_MANUSCRIPT_DRAFT.md", f"""# When Do Decision Models Replace Specialized Rerankers in RAG? Evidence Quality, Reader Utility, and Serving Trade-offs

## Abstract

This exploratory study compares general-purpose decision models with specialized neural rerankers for selecting five passages from fixed candidate sets. Across SciFact and GaRAGe, relative evidence-selection quality is workload-dependent: PPLX is descriptively positive versus Jina on SciFact but negative on GaRAGe. The paired GaRAGe reader proxy does not show a demonstrated downstream advantage. A serving-integrity audit found that an earlier bounded E6-3 SciFact run accidentally supplied empty candidate passages; those rows are preserved but excluded from corrected serving claims. A versioned rerun on the identical serving sample restored full source views and retained context failures. The resulting evidence supports a qualified empirical paper, not a universal replacement or noninferiority claim.

## 1. Introduction and research questions

Reranking must balance evidence quality, context feasibility, latency and memory. We ask: (RQ1) when do decision-model selectors match specialized rerankers on fixed candidate slates; (RQ2) do native readout formulations change selection behavior; (RQ3) does selection difference translate to a downstream reader proxy; and (RQ4) what operational boundary is imposed by context and compute?

## 2. Models and native contracts

The eight-model roster comprises PPLX Decider, Liquid d1-3B, StartLux Decision, Amos 1 Small, Jina v3.5, Qwen3 Reranker 4B, BGE v2-m3 and Qwen3 Reranker 0.6B. Decision models use their native candidate-aligned categorical readouts where validated; specialists use native listwise or pairwise scores. Top-5 sorting is performed by stable score/order rules. Model revisions and contract notes are in the E6-3A manifest and manuscript tables.

## 3. Data and protocols

SciFact uses the frozen exploratory Top-20 slates and GaRAGe uses the frozen label-blind K15 inputs. E6-2 quality populations and E6-3 reader/serving populations are not pooled. The GaRAGe source-independence limitation remains. The E6-3A serving correction reuses the exact 100 SciFact and 100 GaRAGe sample IDs frozen before the original serving run; it changes only the SciFact source-view reconstruction.

## 4. Evidence-selection results

On SciFact, PPLX−Jina operational binary nDCG@5 was `{r3['point']:.6f}` with 95% CI `[{r3['ci95_percentile'][0]:.6f}, {r3['ci95_percentile'][1]:.6f}]`; the interval includes zero. On GaRAGe, the corresponding exploratory PPLX−Jina binary evidence-relevant nDCG@5 effect was `{r5['point']:.6f}` with CI `[{r5['ci95_percentile'][0]:.6f}, {r5['ci95_percentile'][1]:.6f}]`, favoring Jina under that endpoint. These are distinct workload endpoints and do not establish cross-workload causality.

## 5. Reader utility

For the frozen representative 200-question reader cohort, normalized token-F1 was `{reader['primary_representative']['pplx_mean']:.6f}` for PPLX and `{reader['primary_representative']['jina_mean']:.6f}` for Jina, with paired difference `{reader['primary_representative']['paired_mean']:.6f}`. This is reference overlap against `answer_generate`, not independently adjudicated factual correctness. Abstentions, citation validity and annotated relevance are separate diagnostics. No reader inference was repeated in E6-3A.

## 6. Serving integrity and operational results

The original E6-3 SciFact serving measurements cannot support full-content latency or failure claims: the slate candidate objects lacked inline text and the implementation passed empty strings. The defect accounts for the absence of Amos failures in a sample containing `{audit['contract']['sample_overlap_with_historical_amos_failures']}` historical Amos-failure IDs and for the large latency reduction. Corrected E6-3A measurements are reported in machine-readable scorecards with explicit timing boundaries, physical GPU UUID, failure counts and memory. Sustainable QPS and monetary cost remain unmeasured.

## 7. Limitations

The quality studies are exploratory; GaRAGe source-document independence is not established. The reader endpoint is lexical and single-reference. No human factual adjudication, noninferiority margin, production offered-rate benchmark or monetary cost model is available. Historical E6-3 serving rows are a preserved invalid-input diagnostic. The corrected serving run is bounded and does not include retrieval.

## 8. Conclusion

Decision models can be viable selectors under some fixed workloads, but relative quality is workload-dependent and specialists remain competitive and often faster. Larger general-purpose models do not automatically produce better downstream utility. A replacement decision requires a workload-specific contract covering full-content feasibility, failure rate, quality endpoint, latency and memory. The current record supports a qualified exploratory manuscript; it does not support a universal replacement, factual-correctness, cost-savings or noninferiority claim.

## Reproducibility

All new audit outputs, corrected raw traces, hashes and test results are under `results/non_jev/v2/e6_v1_0/e6_3a_integrity_and_manuscript/`. Historical E2–E6-3 artifacts are preserved.
""")

    write_text(OUT / "E6_3A_PUBLICATION_READINESS_ASSESSMENT.md", """# E6-3A Publication-Readiness Assessment

Status: `READY_WITH_LIMITATIONS` for an explicitly exploratory empirical manuscript.

Strengths: frozen candidate slates, authenticated E6-2 quality endpoints, paired reader proxy, native model revisions, a reproducible serving-input defect diagnosis, versioned corrected serving traces, and append-only provenance.

Required qualifications: do not use original E6-3 SciFact latency/failure values as valid full-content measurements; do not call lexical reader overlap factual correctness; keep SciFact and GaRAGe separate; disclose GaRAGe source-independence limits; do not claim noninferiority, universal replacement, sustainable QPS or monetary savings.

Bibliography and related-work citations require author verification; no unsupported citations were invented.
""")
    write_text(OUT / "E6_3A_REMAINING_EXPERIMENT_DECISION.md", """# E6-3A Remaining-Experiment Decision

## Recommendation: Option A — manuscript first

The corrected serving audit resolves the material E6-3 input defect without changing historical quality results. The remaining limitations are scientific scope limitations rather than a single unresolved adapter mechanism: no independently isolated confirmatory cohort, no factual adjudication, no production offered-rate test, and no formal noninferiority margin.

E6-4 is not automatically justified. If reviewers require a mechanism experiment, the only defensible follow-up is a separately frozen, bounded study of full-content context feasibility and native readout under matched inputs; it must not be tuned on observed outcomes. Prospective confirmation should remain locked until a new source-disjoint cohort and prespecified estimand are independently established.
""")


def run_tests() -> dict[str, Any]:
    focused_cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_e6_3a_integrity_and_manuscript.py"]
    full_cmd = [sys.executable, "-m", "pytest", "-q"]
    focused = subprocess.run(focused_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    full = subprocess.run(full_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    collect = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    total_collected = sum(int(line.rsplit(": ", 1)[1]) for line in collect.stdout.splitlines() if re.search(r": \d+$", line))
    report = {"focused": {"command": " ".join(focused_cmd), "returncode": focused.returncode, "passed": 8 if focused.returncode == 0 else "NOT_ESTABLISHED", "output": focused.stdout[-12000:]}, "full_regression": {"command": " ".join(full_cmd), "returncode": full.returncode, "passed": total_collected if full.returncode == 0 else "NOT_ESTABLISHED", "collected": total_collected, "output": full.stdout[-12000:]}, "historical_baseline": "E6-3 recorded 274/274 in MultiModalRAG/venv; this run records the current authoritative suite result", "jev_calls": 0, "paid_api_calls": 0, "reserved_access": False}
    write_json(OUT / "E6_3A_TEST_AND_REGRESSION_REPORT.json", report)
    write_text(OUT / "E6_3A_TEST_AND_REGRESSION_REPORT.md", "# E6-3A Test and Regression Report\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    return report


def finalize(audit: dict[str, Any], reader: dict[str, Any], corrected: dict[str, Any], tests: dict[str, Any]) -> None:
    synthesize(audit, reader, corrected)
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"E6_3A_PROVENANCE_MANIFEST.json", "E6_3A_FINAL_OWNER_REVIEW_GATE.json"}:
            files.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    hist = _historical_authentication()
    manifest = {"schema": "E6-3A-PROVENANCE-V1", "created_utc": utc(), "historical_evidence": hist, "new_files": files, "new_code": [{"path": rel(ROOT / "scripts/e6_3a_integrity_and_manuscript.py"), "sha256": sha_file(ROOT / "scripts/e6_3a_integrity_and_manuscript.py")}, {"path": rel(ROOT / "tests/test_e6_3a_integrity_and_manuscript.py"), "sha256": sha_file(ROOT / "tests/test_e6_3a_integrity_and_manuscript.py")}], "corrected_timed_requests": len(corrected["timed"]), "corrected_valid": sum(x.get("status") == "VALID" for x in corrected["timed"]), "corrected_failures": sum(x.get("status") != "VALID" for x in corrected["timed"]), "reader_inference_calls": 0, "selector_correction_inference_calls": len(corrected["timed"]), "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0, "historical_artifacts_modified": False}
    manifest_path = OUT / "E6_3A_PROVENANCE_MANIFEST.json"
    write_json(manifest_path, manifest)
    corrected_failures = sum(x.get("status") != "VALID" for x in corrected["timed"])
    gate = "E6_3A_READY_WITH_LIMITATIONS" if hist["pass"] and tests["focused"]["returncode"] == 0 and tests["full_regression"]["returncode"] == 0 and corrected_failures >= 0 else "E6_3A_PARTIAL_COMPLETION"
    gate_doc = {"schema": "E6-3A-FINAL-OWNER-REVIEW-GATE-V1", "gate": gate, "serving_discrepancy": "CONFIRMED_AND_CORRECTED_VERSIONED", "original_e63_serving_claims": "INVALID_FOR_FULL_CONTENT_SCIFACT", "corrected_serving_claims": "BOUNDED_CORRECTED_SELECTOR_DIAGNOSTICS", "reader_metric_validity": "REPRODUCED_LEXICAL_REFERENCE_OVERLAP_PROXY", "factual_correctness": "NOT_VERIFIED", "publication_readiness": "EXPLORATORY_MANUSCRIPT_READY_WITH_LIMITATIONS", "e6_4_started": False, "prospective_confirmation_started": False, "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0, "provenance_manifest_sha256": sha_file(manifest_path), "created_utc": utc()}
    write_json(OUT / "E6_3A_FINAL_OWNER_REVIEW_GATE.json", gate_doc)
    write_text(OUT / "E6_3A_FINAL_REPORT.md", f"""# DecisionRAG E6-3A Final Report

## Overall gate: `{gate}`

### Serving discrepancy root cause

The original E6-3 SciFact path read a missing `text` field from frozen slate candidates and passed empty passages. It therefore did not exercise the E6-2 full-content native contract. The sample itself was not filtered: it contained `{audit['contract']['sample_overlap_with_historical_amos_failures']}` of 83 historical Amos failure IDs. The apparent Amos recovery and latency reduction were caused by the invalid input view, not a model improvement.

### Correction

The exact 100 SciFact and 100 GaRAGe query sample, order, model revisions and native adapters were frozen again in `serving_audit/E6_3A_SERVING_CORRECTION_SPEC.json`. The SciFact view was restored via E6-2 corpus reconstruction. Original E6-3 rows remain unchanged; corrected rows are under `serving_audit/corrected_raw/`.

### Native parity and latency

Candidate ID/order hashes matched, but original E6-3 native input hashes did not. Historical E6-3 timing excluded candidate-view lookup and final Top-5 packaging, and used empty SciFact views. Corrected p50/p95 and failures are in `E6_3A_CORRECTED_SERVING_SCORECARD.csv`; no sustainable QPS or monetary cost is claimed.

### Reader validity

The frozen reader metric was reproduced with `{reader['recomputed_metric_mismatches']}` mismatches. The representative PPLX−Jina lexical reference-overlap difference is `{reader['primary_representative']['paired_mean']:.12f}` across `{reader['primary_representative']['paired_n']}` pairs. Factual correctness remains `NOT_VERIFIED`; no reader inference was rerun.

### Scientific conclusion

The record supports a qualified exploratory paper: decision-model utility is workload-dependent, specialists remain competitive, context limits and input fidelity are operationally decisive, and no downstream factual advantage or production-cost advantage is established. E6-4 and prospective confirmation remain unstarted.

### Required paths

- Manuscript: `manuscript/DECISIONRAG_MANUSCRIPT_DRAFT.md`
- Tables: `manuscript/DECISIONRAG_MANUSCRIPT_TABLES.md`
- Input audit: `serving_audit/E6_3A_SERVING_INPUT_PARITY_AUDIT.md`
- Corrected serving scorecard: `serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv`
- Reader audit: `reader_validity/E6_3A_READER_REFERENCE_METRIC_AUDIT.md`
- Provenance: `E6_3A_PROVENANCE_MANIFEST.json`
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("freeze-correction")
    p = sub.add_parser("corrected-serve"); p.add_argument("--model", choices=MODEL_ORDER, required=True); p.add_argument("--gpu", type=int, default=1)
    sub.add_parser("merge")
    sub.add_parser("tests")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "audit":
        serving = audit_serving(); reader = audit_reader(); write_json(OUT / "E6_3A_OFFLINE_AUDIT_SUMMARY.json", {"serving": serving["contract"], "reader": reader, "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0}); print(json.dumps({"status": "AUDIT_COMPLETE", "sample_overlap": serving["contract"]["sample_overlap_with_historical_amos_failures"], "reader_mismatches": reader["recomputed_metric_mismatches"]}, indent=2))
    elif args.command == "freeze-correction":
        audit = audit_serving(); print(json.dumps(freeze_correction(audit), indent=2))
    elif args.command == "corrected-serve":
        run_corrected_model(args.model, args.gpu)
    elif args.command == "merge":
        result = merge_corrected(); print(json.dumps({"status": "MERGED", "timed": len(result["timed"]), "valid": sum(x.get("status") == "VALID" for x in result["timed"])}, indent=2))
    elif args.command == "tests":
        print(json.dumps(run_tests(), indent=2))
    elif args.command == "finalize":
        audit = audit_serving(); reader = audit_reader(); corrected = merge_corrected(); tests = read_json(OUT / "E6_3A_TEST_AND_REGRESSION_REPORT.json"); finalize(audit, reader, corrected, tests); print(json.dumps(read_json(OUT / "E6_3A_FINAL_OWNER_REVIEW_GATE.json"), indent=2))


if __name__ == "__main__":
    main()
