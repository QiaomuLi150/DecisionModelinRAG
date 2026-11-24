#!/usr/bin/env python3
"""Offline E2-S/P reader, selection, and integrity analysis.

This script reads only completed E2-S/P traces plus segregated evaluation
sidecars.  It makes no model or service request.  Reference-answer EM/F1 are
reported as deterministic answer proxies, never as independent factual truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "results/non_jev/v2/e2_sp"
OUT = E2 / "gpu5"
SELECTION = OUT / "E2_SP_S2_SELECTION_RESULTS.jsonl"
READER = OUT / "E2_SP_P1_READER_OUTPUTS.jsonl"
COHORT = E2 / "E2_SP_READER_COHORT_MANIFEST.json"
TRACK_A_GOLD = ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl"
ENTERPRISE_GOLD = ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl"
ENTERPRISE_DB = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
POLICIES = ["retrieval5", "bge_pointwise", "qwen06", "qwen4", "jina35", "gte", "minilm", "clef_joint", "nimble", "pplx_decider", "qwen_direct"]
DATASETS = ["hotpot", "musique", "wiki2", "enterprise"]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: Any) -> str:
    text = str(value or "").lower()
    return " ".join(token for token in re.sub(r"[^a-z0-9\s]", " ", text).split() if token not in {"a", "an", "the"})


def f1(prediction: str, truth: str) -> float:
    pred = norm(prediction).split()
    gold = norm(truth).split()
    if not pred or not gold:
        return float(pred == gold)
    remaining = list(gold)
    common = 0
    for token in pred:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, reference: dict[str, Any], dataset: str) -> tuple[float | None, float | None]:
    if dataset != "hotpot" and dataset != "musique" and dataset != "wiki2":
        truth = reference.get("gold_answer")
        if not truth:
            return None, None
        return float(norm(prediction) == norm(truth)), f1(prediction, truth)
    truths = [reference.get("answer", ""), *reference.get("answer_aliases", [])]
    truths = [truth for truth in truths if truth]
    if not truths:
        return None, None
    return float(any(norm(prediction) == norm(truth) for truth in truths)), max(f1(prediction, truth) for truth in truths)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def extract_original_id(value: str) -> str | None:
    match = re.search(r"(dsid_[0-9a-f]{32})", value)
    return match.group(1) if match else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))["query_records"]
    if len(cohort) != 300 or len({row["query_id"] for row in cohort}) != 300:
        raise RuntimeError("invalid frozen cohort")
    selection_rows = load_jsonl(SELECTION)
    reader_rows = load_jsonl(READER)
    if len(selection_rows) != 3300 or len(reader_rows) != 3300:
        raise RuntimeError(f"unexpected E2-S/P row counts selection={len(selection_rows)} reader={len(reader_rows)}")
    selection = {(row["policy"], row["query_id"]): row for row in selection_rows}
    reader = {(row["policy"], row["query_id"]): row for row in reader_rows}
    if len(selection) != 3300 or len(reader) != 3300:
        raise RuntimeError("duplicate policy/query rows")
    if any(row.get("status") != "ok" for row in reader_rows):
        raise RuntimeError("reader output contains errors")

    track_a_gold = {row["query_id"]: row for row in load_jsonl(TRACK_A_GOLD)}
    enterprise_gold = {row["question_id"]: row for row in load_jsonl(ENTERPRISE_GOLD)}
    conn = sqlite3.connect(f"file:{ENTERPRISE_DB}?mode=ro", uri=True)
    internal_to_original = {row[0]: row[1] for row in conn.execute("SELECT internal_doc_id, original_doc_id FROM documents")}
    conn.close()

    metric_rows: list[dict[str, Any]] = []
    for cohort_row in cohort:
        qid = cohort_row["query_id"]
        dataset = cohort_row["dataset"]
        if dataset == "enterprise":
            reference = enterprise_gold[qid]
            gold_ids = set(reference.get("expected_doc_ids", []))
            identity_eligible = qid != "qst_0413" and bool(gold_ids)
        else:
            reference = track_a_gold[qid]
            gold_ids = set(reference.get("gold_doc_ids", []))
            identity_eligible = bool(reference.get("gold_mapping_valid", True)) and bool(gold_ids)
        top20_ids = cohort_row["top20_candidate_ids"]
        top20_original = {internal_to_original.get(doc_id, extract_original_id(doc_id)) for doc_id in top20_ids} if dataset == "enterprise" else set(top20_ids)
        top20_original.discard(None)
        top20_any = bool(gold_ids & top20_original) if identity_eligible else None
        top20_complete = gold_ids.issubset(top20_original) if identity_eligible else None
        for policy in POLICIES:
            selection_row = selection[(policy, qid)]
            reader_row = reader[(policy, qid)]
            selected_ids = selection_row["selected_ids"]
            selected_original = {internal_to_original.get(doc_id, extract_original_id(doc_id)) for doc_id in selected_ids} if dataset == "enterprise" else set(selected_ids)
            selected_original.discard(None)
            selected_any = bool(gold_ids & selected_original) if identity_eligible else None
            selected_complete = gold_ids.issubset(selected_original) if identity_eligible else None
            recall = (len(gold_ids & selected_original) / len(gold_ids)) if identity_eligible else None
            answer_em, answer_f1 = answer_metrics(reader_row.get("answer", ""), reference, dataset)
            answer = str(reader_row.get("answer") or "").strip()
            citation_ids = sorted(set(re.findall(r"\bP([0-9]+)\b", answer, flags=re.IGNORECASE)))
            latency = float(reader_row["duration_ms"]) if reader_row.get("duration_ms") is not None else None
            metric_rows.append({
                "policy": policy,
                "query_id": qid,
                "dataset": dataset,
                "cohort": cohort_row["cohort"],
                "identity_eligible": identity_eligible,
                "top20_any_gold": top20_any,
                "top20_complete_gold": top20_complete,
                "selected_any_gold": selected_any,
                "selected_complete_gold": selected_complete,
                "selected_gold_document_recall": recall,
                "reader_status": reader_row.get("status"),
                "real_inference": reader_row.get("real_inference"),
                "answer": answer,
                "answer_nonempty": bool(answer),
                "abstention": norm(answer) == norm("INSUFFICIENT_EVIDENCE"),
                "answer_em_proxy": answer_em,
                "answer_f1_proxy": answer_f1,
                "citation_prompt_status": "NOT_APPLICABLE_prompt_has_no_citation_instruction",
                "citation_ids_detected": ",".join(citation_ids),
                "reader_duration_ms": latency,
                "input_tokens": reader_row.get("input_tokens"),
                "output_tokens": reader_row.get("output_tokens"),
                "prompt_sha256": reader_row.get("prompt_sha256"),
                "actual_model_input_sha256": reader_row.get("actual_model_input_sha256"),
                "raw_response_sha256": reader_row.get("raw_response_sha256"),
            })

    write_csv(OUT / "E2_SP_DOWNSTREAM_ANSWER_CITATION_METRICS.csv", metric_rows)

    summary_rows = []
    for dataset in DATASETS:
        for policy in POLICIES:
            subset = [row for row in metric_rows if row["dataset"] == dataset and row["policy"] == policy]
            eligible = [row for row in subset if row["identity_eligible"]]
            answer_rows = [row for row in subset if row["answer_f1_proxy"] is not None]
            latencies = [row["reader_duration_ms"] for row in subset if row["reader_duration_ms"] is not None]
            summary_rows.append({
                "dataset": dataset,
                "policy": policy,
                "n_queries": len(subset),
                "identity_eligible_n": len(eligible),
                "reader_ok_n": sum(row["reader_status"] == "ok" for row in subset),
                "real_inference_n": sum(bool(row["real_inference"]) for row in subset),
                "nonempty_n": sum(row["answer_nonempty"] for row in subset),
                "abstention_n": sum(row["abstention"] for row in subset),
                "top20_any_gold_rate": mean([float(row["top20_any_gold"]) for row in eligible if row["top20_any_gold"] is not None]),
                "top20_complete_gold_rate": mean([float(row["top20_complete_gold"]) for row in eligible if row["top20_complete_gold"] is not None]),
                "selected_any_gold_rate": mean([float(row["selected_any_gold"]) for row in eligible if row["selected_any_gold"] is not None]),
                "selected_complete_support_rate": mean([float(row["selected_complete_gold"]) for row in eligible if row["selected_complete_gold"] is not None]),
                "selected_complete_support_n": sum(bool(row["selected_complete_gold"]) for row in eligible),
                "selected_gold_document_recall": mean([float(row["selected_gold_document_recall"]) for row in eligible if row["selected_gold_document_recall"] is not None]),
                "answer_em_proxy_all": mean([float(row["answer_em_proxy"]) for row in answer_rows]),
                "answer_f1_proxy_all": mean([float(row["answer_f1_proxy"]) for row in answer_rows]),
                "answer_f1_proxy_reader_ok": mean([float(row["answer_f1_proxy"]) for row in answer_rows if row["reader_status"] == "ok"]),
                "reader_p50_ms": percentile(latencies, 0.50),
                "reader_p95_ms": percentile(latencies, 0.95),
                "mean_input_tokens": mean([float(row["input_tokens"]) for row in subset if row["input_tokens"] is not None]),
                "mean_output_tokens": mean([float(row["output_tokens"]) for row in subset if row["output_tokens"] is not None]),
                "citation_metric": "NOT_APPLICABLE_prompt_has_no_citation_instruction",
            })
    write_csv(OUT / "E2_SP_QUALITY_POLICY_SUMMARY.csv", summary_rows)

    # A compact comparable frontier table.  It deliberately reports measured
    # selector latency and reader proxies separately from dollar cost; no
    # monetary rate was authorized or established here.
    latency_by_policy = {row["policy"]: row for row in load_csv(OUT / "E2_SP_SELECTION_LATENCY_SUMMARY.csv")}
    frontier = []
    for policy in POLICIES:
        rows = [row for row in summary_rows if row["policy"] == policy]
        track_a = [row for row in rows if row["dataset"] in {"hotpot", "musique", "wiki2"}]
        enterprise = [row for row in rows if row["dataset"] == "enterprise"]
        frontier.append({
            "policy": policy,
            "track_a_n": sum(row["n_queries"] for row in track_a),
            "track_a_selected_complete_support": mean([row["selected_complete_support_rate"] for row in track_a if row["selected_complete_support_rate"] is not None]),
            "track_a_answer_f1_proxy": mean([row["answer_f1_proxy_all"] for row in track_a if row["answer_f1_proxy_all"] is not None]),
            "enterprise_n": sum(row["n_queries"] for row in enterprise),
            "enterprise_selected_complete_support": enterprise[0]["selected_complete_support_rate"],
            "enterprise_answer_f1_proxy": enterprise[0]["answer_f1_proxy_all"],
            "selector_p50_ms": latency_by_policy.get(policy, {}).get("p50_ms", "NOT_MEASURED"),
            "selector_p95_ms": latency_by_policy.get(policy, {}).get("p95_ms", "NOT_MEASURED"),
            "selector_slo_1000_hit_rate": latency_by_policy.get(policy, {}).get("slo_1000_hit_rate", "NOT_MEASURED"),
            "gpu_seconds_per_query": "NOT_MEASURED",
            "gpu_cost_usd_per_query": "NOT_MEASURED",
            "frontier_status": "DESCRIPTIVE_ONLY_no_comparable_cost_or_open_loop_capacity",
        })
    write_csv(OUT / "E2_SP_QUALITY_COST_FRONTIER.csv", frontier)

    checks = {
        "cohort_rows": len(cohort),
        "selection_rows": len(selection_rows),
        "reader_rows": len(reader_rows),
        "all_reader_ok": all(row["reader_status"] == "ok" for row in metric_rows),
        "all_real_inference": all(row["real_inference"] is True for row in metric_rows),
        "all_actual_input_hashes": all(bool(row["actual_model_input_sha256"]) for row in metric_rows),
        "all_raw_response_hashes": all(bool(row["raw_response_sha256"]) for row in metric_rows),
        "gold_in_reader_rows": sum(bool(row.get("gold_in_reader_input")) for row in load_jsonl(READER)),
        "cache_reuse_rows": sum(bool(row.get("cache_reuse")) for row in load_jsonl(READER)),
        "qst_0413_in_cohort": any(row["query_id"] == "qst_0413" for row in cohort),
        "track_a_metric_rows": sum(row["dataset"] != "enterprise" for row in metric_rows),
        "enterprise_metric_rows": sum(row["dataset"] == "enterprise" for row in metric_rows),
        "enterprise_identity_ambiguous_rows": sum(row["dataset"] == "enterprise" and not row["identity_eligible"] for row in metric_rows),
        "selection_sha256": sha_file(SELECTION),
        "reader_sha256": sha_file(READER),
    }
    report = [
        "# E2-S/P Independent Metric Recomputation",
        "",
        "Status: `PASS_WITH_LIMITATIONS`.",
        "",
        "All rows were recomputed offline from the immutable Top-20/Top-5 selections, the completed reader traces, the segregated Track A answer sidecar, and the segregated EnterpriseRAG-Bench answer sidecar. No model or service request was made by this analysis.",
        "",
        "## Interpretation",
        "",
        "- Track A and enterprise are reported separately; they are not pooled into one quality estimate.",
        "- Reference-answer EM/F1 are deterministic answer proxies. They are not independently adjudicated factual correctness, completeness, or groundedness.",
        "- The frozen reader prompt contains no citation instruction, so citation scoring is `NOT_APPLICABLE_prompt_has_no_citation_instruction`; detected `P#` tokens are retained in the machine-readable rows only.",
        "- Enterprise identity-dependent support metrics use the path-qualified internal-to-original mapping. The sampled 60-question cohort does not contain `qst_0413`.",
        "- Dollar cost, GPU-seconds/query, and open-loop capacity remain `NOT_MEASURED`; serialized selector latency must not be converted into cost or sustainable QPS.",
        "",
        "## Integrity checks",
        "",
        "```json",
        json.dumps(checks, indent=2, sort_keys=True),
        "```",
        "",
        "Machine-readable outputs: `E2_SP_DOWNSTREAM_ANSWER_CITATION_METRICS.csv`, `E2_SP_QUALITY_POLICY_SUMMARY.csv`, and `E2_SP_QUALITY_COST_FRONTIER.csv`.",
    ]
    (OUT / "E2_SP_INDEPENDENT_METRIC_RECOMPUTATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, sort_keys=True))


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
