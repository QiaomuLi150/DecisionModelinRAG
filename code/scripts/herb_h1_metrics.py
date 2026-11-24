#!/usr/bin/env python3
"""Compute deterministic, gold-aware HERB H1 metrics after frozen inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/herb_h1"
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]
PRIMARY = [("bge_p5", "clef_g5"), ("qwen_p5", "clef_g5"), ("clef_g5", "retrieval5")]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def entity_tokens(answer: str) -> set[str]:
    return set(re.findall(r"eid_[0-9a-f]+", answer.casefold()))


def listify(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def f1(pred: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    if not pred:
        return 0.0
    precision = len(pred & gold) / len(pred)
    recall = len(pred & gold) / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def reference_metric(answer: str, gold: dict) -> tuple[float | None, float | None, str]:
    typ = gold.get("question_type")
    values = listify(gold.get("ground_truth"))
    if typ == "person":
        predicted, expected = entity_tokens(answer), {x.casefold() for x in values}
        return (1.0 if predicted == expected else 0.0, f1(predicted, expected), "entity_set_em_f1")
    if typ in {"company", "pr", "url"}:
        expected = {norm(x) for x in values}
        answer_norm = norm(answer)
        predicted = {value for value in expected if value and value in answer_norm}
        # Matching against exact reference strings is deliberately reported as
        # an answer proxy, not independently verified factual correctness.
        return (1.0 if predicted == expected else 0.0, f1(predicted, expected), "normalized_reference_proxy")
    return None, None, "NOT_MEASURED_CONTENT"


def bootstrap(values_a: list[float], values_b: list[float], clusters: list[str], seed: int = 731, reps: int = 5000) -> tuple[float, float, float]:
    deltas = [a - b for a, b in zip(values_a, values_b)]
    observed = sum(deltas) / len(deltas) if deltas else float("nan")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        groups[cluster].append(index)
    keys = list(groups)
    rng = random.Random(seed)
    sampled = []
    for _ in range(reps):
        indices = [index for key in (rng.choice(keys) for _ in keys) for index in groups[key]]
        sampled.append(sum(deltas[i] for i in indices) / len(indices) if indices else float("nan"))
    sampled.sort()
    lo = sampled[int(0.025 * (len(sampled) - 1))]
    hi = sampled[int(0.975 * (len(sampled) - 1))]
    return observed, lo, hi


def main() -> int:
    sidecar = {row["query_id"]: row for row in load(OUT / "protected/HERB_H1_GOLD_SIDECAR.jsonl")}
    artifacts = {row["physical_artifact_id"]: row for row in load(OUT / "HERB_H1_ARTIFACT_MANIFEST.jsonl")}
    readers = load(OUT / "HERB_H1_READER_RESULTS.jsonl")
    if len(readers) != 1514 * len(POLICIES):
        raise RuntimeError(f"expected {1514 * len(POLICIES)} reader results, got {len(readers)}")
    metrics = []
    reader_hash = digest(OUT / "HERB_H1_READER_RESULTS.jsonl")
    by_policy_query = {}
    for row in readers:
        qid = row["query_id"]
        gold = sidecar[qid]
        originals = {artifacts[doc]["original_source_id"] for doc in row["selected_doc_ids"] if doc in artifacts}
        citations = set(gold.get("citations") or [])
        answerable = gold.get("answerability") == "answerable"
        cite_recall = len(originals & citations) / len(citations) if answerable and citations else None
        cite_hit = 1.0 if answerable and originals & citations else 0.0 if answerable else None
        answer = str(row.get("answer") or "")
        abstained = norm(answer) == "insufficient_evidence."
        em, ref_f1, ref_kind = reference_metric(answer, gold) if answerable else (None, None, "NOT_APPLICABLE_UNANSWERABLE")
        record = {"query_id": qid, "product": gold.get("product"), "answerability": gold.get("answerability"), "question_type": gold.get("question_type"), "policy_id": row["policy_id"], "reader_status": row.get("reader_status"), "selected_count": len(row["selected_doc_ids"]), "citation_recall": cite_recall, "citation_hit": cite_hit, "answer_nonempty": 1.0 if answer.strip() else 0.0, "abstained": 1.0 if abstained else 0.0, "reference_em": em, "reference_f1": ref_f1, "reference_metric_kind": ref_kind, "gold_accessed": True, "reader_result_sha256": reader_hash}
        metrics.append(record)
        by_policy_query[(qid, row["policy_id"])] = record
    metric_path = OUT / "HERB_H1_METRICS.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(metrics[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(metrics)
    summary_rows = []
    for policy in POLICIES:
        rows = [row for row in metrics if row["policy_id"] == policy]
        for cohort in ("answerable", "unanswerable", "all"):
            subset = [row for row in rows if cohort == "all" or row["answerability"] == cohort]
            def mean(key: str):
                vals = [float(row[key]) for row in subset if row[key] is not None]
                return sum(vals) / len(vals) if vals else "NOT_MEASURED"
            summary_rows.append({"policy_id": policy, "cohort": cohort, "n": len(subset), "citation_recall": mean("citation_recall"), "citation_hit": mean("citation_hit"), "answer_nonempty": mean("answer_nonempty"), "abstained": mean("abstained"), "reference_em": mean("reference_em"), "reference_f1": mean("reference_f1"), "reader_errors": sum(row["reader_status"] != "ok" for row in subset)})
    summary_path = OUT / "HERB_H1_POLICY_SUMMARY.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary_rows[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary_rows)
    paired = []
    answerable_qids = [qid for qid, row in sidecar.items() if row.get("answerability") == "answerable"]
    for left, right in PRIMARY:
        for metric in ("citation_recall", "citation_hit"):
            a = [by_policy_query[(qid, left)][metric] for qid in answerable_qids]
            b = [by_policy_query[(qid, right)][metric] for qid in answerable_qids]
            if any(value is None for value in a + b):
                continue
            products = [sidecar[qid].get("product") for qid in answerable_qids]
            observed, lo, hi = bootstrap(a, b, products)
            paired.append({"comparison": f"{left}_vs_{right}", "metric": metric, "population": "answerable", "n": len(a), "difference_left_minus_right": observed, "ci95_low": lo, "ci95_high": hi, "bootstrap_replicates": 5000, "seed": 731, "cluster": "product"})
    paired_path = OUT / "HERB_H1_PRIMARY_PAIRED_STATISTICS.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(paired[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    answerable = [row for row in metrics if row["answerability"] == "answerable"]
    unanswerable = [row for row in metrics if row["answerability"] == "unanswerable"]
    (OUT / "HERB_H1_ANSWERABLE_BREAKDOWN.md").write_text("# HERB H1 Answerable Breakdown\n\nCitation metrics are original-artifact-ID coverage diagnostics, not complete-support or CSSR claims. Reference EM/F1 is a deterministic answer proxy for structured types; content is `NOT_MEASURED`.\n\n- Query count: 815\n- Policy outcomes: 9,780\n- Gold sidecar joined after inference: yes\n\nSee `HERB_H1_POLICY_SUMMARY.csv` and `HERB_H1_PRIMARY_PAIRED_STATISTICS.csv`.\n", encoding="utf-8")
    (OUT / "HERB_H1_UNANSWERABLE_BREAKDOWN.md").write_text("# HERB H1 Unanswerable Breakdown\n\nUnanswerable queries are not pooled with answerable queries. We report exact abstention and nonempty-answer rates only; a nonempty answer is not treated as factual correctness.\n\n- Query count: 699\n- Policy outcomes: 8,388\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "reader_rows": len(readers), "answerable_rows": len(answerable), "unanswerable_rows": len(unanswerable), "metrics_sha256": digest(metric_path), "paired_sha256": digest(paired_path), "jev_calls": 0, "hosted_cloudflare_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
