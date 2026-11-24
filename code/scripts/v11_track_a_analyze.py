#!/usr/bin/env python3
"""Offline Track A metric, paired-statistics, and provenance analysis.

This script is intentionally separate from retrieval and selector execution:
it is the first Track A step that reads the evaluation-only gold sidecar.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v11/track_a"
PREFLIGHT = ROOT / "results/preflight/v11"
DATASETS = ("hotpot", "musique", "wiki2")
POLICIES = ("raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5")
MODELS = ("bge", "qwen", "laya", "clef")
PRIMARY = (("bge_p5", "clef_g5"), ("qwen_p5", "clef_g5"), ("clef_g5", "retrieval5"))
B = 5000
SEED = 731


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(float(value) for value in values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def norm(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(token for token in value.split() if token not in {"a", "an", "the"})


def f1(prediction: str, truth: str) -> float:
    pred, gold = norm(prediction).split(), norm(truth).split()
    if not pred or not gold:
        return float(pred == gold)
    remaining = list(gold)
    common = 0
    for token in pred:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def answer_scores(prediction: str, question: dict) -> tuple[float, float]:
    golds = [question["answer"], *question.get("answer_aliases", [])]
    return (float(any(norm(prediction) == norm(gold) for gold in golds)), max(f1(prediction, gold) for gold in golds))


def bootstrap(values: list[float], seed: int = SEED) -> dict:
    if not values:
        return {"status": "NOT_MEASURED", "n": 0}
    rng = random.Random(seed)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(B)]
    return {"status": "OK" if len(values) >= 40 else "UNDERPOWERED_DESCRIPTIVE", "n": len(values), "delta": statistics.mean(values), "ci95_low": pct(samples, 0.025), "ci95_high": pct(samples, 0.975), "bootstrap_replicates": B, "bootstrap_seed": seed}


def paired(a: list[dict], b: list[dict], field: str, eligibility: str | None) -> dict:
    bm = {row["query_id"]: row for row in b}
    values = []
    for left in a:
        right = bm.get(left["query_id"])
        if right is None:
            continue
        if eligibility and not left[eligibility]:
            continue
        values.append(float(left[field]) - float(right[field]))
    return bootstrap(values)


def write_csv(path: Path, values: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main() -> int:
    cohort = rows(PREFLIGHT / "NEW_TEST_IDS.jsonl")
    gold = {row["query_id"]: row for row in rows(PREFLIGHT / "NEW_TEST_GOLD_SIDECAR.jsonl")}
    retrieval_rows = rows(OUT / "V11_TRACK_A_RETRIEVAL_SLATES.jsonl")
    selection_rows = rows(OUT / "V11_TRACK_A_SELECTOR_RESULTS_FULL.jsonl")
    reader_rows = rows(OUT / "V11_TRACK_A_READER_RESULTS_FULL.jsonl")
    if len(cohort) != 3767 or len(retrieval_rows) != 3767 or len(selection_rows) != 3767 * 12 or len(reader_rows) != 3767 * 12:
        raise RuntimeError(f"unexpected Track A sizes cohort={len(cohort)} retrieval={len(retrieval_rows)} selectors={len(selection_rows)} readers={len(reader_rows)}")
    if set(gold) != {row["query_id"] for row in cohort}:
        raise RuntimeError("gold sidecar does not exactly cover the frozen cohort")
    retrieval = {row["query_id"]: row for row in retrieval_rows}
    selections = {(row["query_id"], row["policy_id"]): row for row in selection_rows}
    readers = {(row["query_id"], row["policy_id"]): row for row in reader_rows}
    if len(selections) != len(selection_rows) or len(readers) != len(reader_rows):
        raise RuntimeError("duplicate query/policy rows")

    metric_rows = []
    for query in cohort:
        qid = query["query_id"]
        retrieval_row = retrieval[qid]
        top20 = list(retrieval_row["candidate_ids"])
        gold_docs = set(gold[qid]["gold_doc_ids"])
        if not gold_docs:
            raise RuntimeError(f"empty gold set: {qid}")
        complete_top20 = gold_docs.issubset(top20)
        any_top20 = bool(gold_docs.intersection(top20))
        for policy in POLICIES:
            selection = selections[(qid, policy)]
            reader = readers[(qid, policy)]
            selected = list(selection["selected_doc_ids"])
            reader_ok = reader.get("reader_status") == "ok"
            em, answer_f1 = answer_scores(reader.get("answer", ""), gold[qid]) if reader_ok else (0.0, 0.0)
            passages = reader.get("view_info", {}).get("passages", [])
            gold_selected = gold_docs.intersection(selected)
            truncated_gold = sum(1 for passage in passages if passage.get("doc_id") in gold_docs and passage.get("truncated"))
            metric_rows.append({
                "query_id": qid, "dataset": query["dataset"], "policy_id": policy,
                "top20_any_gold": any_top20, "top20_complete_gold": complete_top20,
                "support_recall": len(gold_selected) / len(gold_docs),
                "complete_support": bool(gold_docs.issubset(selected)),
                "reader_ok": reader_ok, "answer_em": em, "answer_f1": answer_f1,
                "reader_duration_ms": float(reader.get("duration_ms") or 0.0),
                "reader_input_tokens": reader.get("input_tokens"),
                "reader_output_tokens": reader.get("output_tokens"),
                "selection_wall_ms": float(selection.get("selection_wall_ms") or 0.0),
                "truncated_gold_count": truncated_gold,
                "gold_count": len(gold_docs), "selected_count": len(selected),
                "selection_status": selection.get("status"), "reader_status": reader.get("reader_status"),
            })

    metric_map = {(row["query_id"], row["policy_id"]): row for row in metric_rows}
    summary = {}
    for policy in POLICIES:
        policy_summary = {"overall": {}, "by_dataset": {}}
        for label, subset in [("overall", [row for row in metric_rows if row["policy_id"] == policy])] + [(dataset, [row for row in metric_rows if row["policy_id"] == policy and row["dataset"] == dataset]) for dataset in DATASETS]:
            reader_ok = [row for row in subset if row["reader_ok"]]
            eligible = [row for row in subset if row["top20_complete_gold"]]
            target = {
                "n": len(subset), "selection_ok": sum(row["selection_status"] == "ok" for row in subset),
                "reader_ok": len(reader_ok), "reader_error": len(subset) - len(reader_ok),
                "any_gold_top20_n": sum(row["top20_any_gold"] for row in subset),
                "any_gold_top20_rate": sum(row["top20_any_gold"] for row in subset) / len(subset) if subset else None,
                "complete_gold_top20_n": len(eligible), "complete_gold_top20_rate": len(eligible) / len(subset) if subset else None,
                "cssr_unconditional": statistics.mean(row["complete_support"] for row in subset) if subset else None,
                "cssr_conditional_complete_gold": statistics.mean(row["complete_support"] for row in eligible) if eligible else None,
                "support_recall": statistics.mean(row["support_recall"] for row in subset) if subset else None,
                "answer_em_all": statistics.mean(row["answer_em"] for row in subset) if subset else None,
                "answer_f1_all": statistics.mean(row["answer_f1"] for row in subset) if subset else None,
                "answer_em_reader_ok": statistics.mean(row["answer_em"] for row in reader_ok) if reader_ok else None,
                "answer_f1_reader_ok": statistics.mean(row["answer_f1"] for row in reader_ok) if reader_ok else None,
                "reader_duration_p50_ms": pct([row["reader_duration_ms"] for row in subset], 0.5),
                "reader_duration_p95_ms": pct([row["reader_duration_ms"] for row in subset], 0.95),
                "reader_input_tokens_mean": statistics.mean([row["reader_input_tokens"] for row in subset if row["reader_input_tokens"] is not None]) if any(row["reader_input_tokens"] is not None for row in subset) else None,
                "truncated_gold_count": sum(row["truncated_gold_count"] for row in subset),
            }
            if label == "overall":
                policy_summary["overall"] = target
            else:
                policy_summary["by_dataset"][label] = target
        summary[policy] = policy_summary

    metric_path = OUT / "V11_TRACK_A_METRICS.csv"
    write_csv(metric_path, metric_rows, list(metric_rows[0]))
    primary_rows = []
    for left, right in PRIMARY:
        for dataset in (*DATASETS, "pooled"):
            left_rows = [row for row in metric_rows if row["policy_id"] == left and (dataset == "pooled" or row["dataset"] == dataset)]
            right_rows = [row for row in metric_rows if row["policy_id"] == right and (dataset == "pooled" or row["dataset"] == dataset)]
            for field, eligibility in (("complete_support", "top20_complete_gold"), ("support_recall", "top20_complete_gold"), ("answer_f1", None), ("answer_em", None)):
                result = paired(left_rows, right_rows, field, eligibility)
                primary_rows.append({"comparison": f"{left}_vs_{right}", "left_policy": left, "right_policy": right, "dataset": dataset, "metric": field, **result})
    write_csv(OUT / "V11_TRACK_A_PRIMARY_PAIRED_STATISTICS.csv", primary_rows, list(primary_rows[0]))

    rescue_rows = []
    for model in MODELS:
        base = {row["query_id"]: row for row in metric_rows if row["policy_id"] == f"{model}_p5"}
        rescue = {row["query_id"]: row for row in metric_rows if row["policy_id"] == f"{model}_rescue5"}
        for dataset in (*DATASETS, "pooled"):
            qids = [row["query_id"] for row in metric_rows if row["policy_id"] == f"{model}_p5" and (dataset == "pooled" or row["dataset"] == dataset)]
            benefit = sum(not base[qid]["complete_support"] and rescue[qid]["complete_support"] for qid in qids)
            harm = sum(base[qid]["complete_support"] and not rescue[qid]["complete_support"] for qid in qids)
            unchanged = len(qids) - benefit - harm
            rescue_rows.append({"model": model, "dataset": dataset, "n": len(qids), "benefit_complete_support": benefit, "harm_complete_support": harm, "unchanged_complete_support": unchanged})
    write_csv(OUT / "V11_TRACK_A_RESCUE_AUDIT.csv", rescue_rows, list(rescue_rows[0]))

    collision_rows = []
    for model in ("laya", "clef"):
        policy = f"{model}_g5"
        for dataset in (*DATASETS, "pooled"):
            subset = [row for row in cohort if dataset == "pooled" or row["dataset"] == dataset]
            total_groups = collision_groups = exact_one_groups = 0
            for query in subset:
                selection = selections[(query["query_id"], policy)]
                gold_docs = set(gold[query["query_id"]]["gold_doc_ids"])
                for group in (selection.get("groupwise") or {}).get("groups", []):
                    total_groups += 1
                    count = len(gold_docs.intersection(group.get("group_doc_ids", [])))
                    collision_groups += count >= 2
                    exact_one_groups += count == 1
            collision_rows.append({"model": model, "dataset": dataset, "queries": len(subset), "quartets": total_groups, "quartets_with_two_or_more_gold": collision_groups, "quartets_with_exactly_one_gold": exact_one_groups})
    write_csv(OUT / "V11_TRACK_A_GROUPWISE_COLLISIONS.csv", collision_rows, list(collision_rows[0]))

    all_gold = {row["query_id"]: set(gold[row["query_id"]]["gold_doc_ids"]) for row in cohort}
    stage_values = {
        "retrieval_ms": [float(row.get("retrieval_ms") or 0) for row in retrieval_rows],
        "bm25_ms": [float(row.get("bm25_ms") or 0) for row in retrieval_rows],
        "embedding_ms": [float(row.get("embedding_ms") or 0) for row in retrieval_rows],
        "qdrant_ms": [float(row.get("qdrant_ms") or 0) for row in retrieval_rows],
        "rrf_ms": [float(row.get("rrf_ms") or 0) for row in retrieval_rows],
    }
    runtime = {name: {"n": len(values), "p50_ms": pct(values, 0.5), "p95_ms": pct(values, 0.95), "mean_ms": statistics.mean(values) if values else None, "sum_ms": sum(values)} for name, values in stage_values.items()}
    for model in MODELS:
        trace_path = OUT / f"V11_TRACK_A_{model.upper()}_POINTWISE_FULL.jsonl"
        trace = rows(trace_path)
        values = [float(row.get("batch_duration_ms") or 0) for row in trace]
        runtime[f"selector_{model}_pointwise_ms"] = {"n": len(values), "p50_ms": pct(values, 0.5), "p95_ms": pct(values, 0.95), "mean_ms": statistics.mean(values) if values else None, "sum_ms": sum(values)}
    for policy in POLICIES:
        values = [row["reader_duration_ms"] for row in metric_rows if row["policy_id"] == policy]
        runtime[f"reader_{policy}_measured_ms"] = {"n": len(values), "p50_ms": pct(values, 0.5), "p95_ms": pct(values, 0.95), "mean_ms": statistics.mean(values) if values else None, "sum_ms": sum(values)}
    (OUT / "V11_TRACK_A_RUNTIME_SUMMARY.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # A concise human-readable dataset report keeps the machine-readable CSVs primary.
    report = ["# Track A Dataset Breakdown", "", f"Confirmation cohort: {len(cohort)} questions.", "", "| Policy | Dataset | n | Any-gold@20 | Complete-gold@20 | CSSR conditional | Support recall | Reader F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        for dataset in DATASETS:
            value = summary[policy]["by_dataset"][dataset]
            report.append(f"| {policy} | {dataset} | {value['n']} | {value['any_gold_top20_rate']:.4f} | {value['complete_gold_top20_rate']:.4f} | {value['cssr_conditional_complete_gold'] if value['cssr_conditional_complete_gold'] is not None else 'NOT_MEASURED'} | {value['support_recall']:.4f} | {value['answer_f1_all']:.4f} |")
    report.extend(["", "CSSR conditional denominator is the immutable query set whose complete gold document set is present in the frozen Top-20. Reader failures are counted as zero in `answer_*_all`; reader-success-only values are also present in the machine-readable summary."])
    (OUT / "V11_TRACK_A_DATASET_BREAKDOWN.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    failure = ["# Track A Failure Attribution", "", "All selector and reader rows were required to be explicit. The table below attributes loss at the retrieval, selection, and reader layers.", "", "| Dataset | n | Retrieval any-gold miss | Complete-gold miss | Reader errors | Gold truncation events |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        subset = [row for row in metric_rows if row["dataset"] == dataset and row["policy_id"] == "raw20"]
        failure.append(f"| {dataset} | {len(subset)} | {sum(not row['top20_any_gold'] for row in subset)} | {sum(not row['top20_complete_gold'] for row in subset)} | {sum(not row['reader_ok'] for row in subset)} | {sum(row['truncated_gold_count'] for row in subset)} |")
    failure.extend(["", "Rescue benefit/harm/unchanged counts are in `V11_TRACK_A_RESCUE_AUDIT.csv`. Groupwise quartet collision counts are in `V11_TRACK_A_GROUPWISE_COLLISIONS.csv`."])
    (OUT / "V11_TRACK_A_FAILURE_ATTRIBUTION.md").write_text("\n".join(failure) + "\n", encoding="utf-8")

    manifest = {
        "study": "DecisionRAG v1.1 Track A prospective multi-hop confirmation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "query_count": len(cohort), "policies": list(POLICIES),
        "datasets": Counter(row["dataset"] for row in cohort), "gold_read_only_analysis": True, "jev_calls": 0, "hosted_cloudflare_calls": 0,
        "input_hashes": {
            "cohort": digest(PREFLIGHT / "NEW_TEST_IDS.jsonl"), "gold_sidecar": digest(PREFLIGHT / "NEW_TEST_GOLD_SIDECAR.jsonl"),
            "retrieval_slates": digest(OUT / "V11_TRACK_A_RETRIEVAL_SLATES.jsonl"), "selector_results": digest(OUT / "V11_TRACK_A_SELECTOR_RESULTS_FULL.jsonl"),
            "reader_results": digest(OUT / "V11_TRACK_A_READER_RESULTS_FULL.jsonl"),
        },
        "output_hashes": {"metrics": digest(metric_path), "primary_statistics": digest(OUT / "V11_TRACK_A_PRIMARY_PAIRED_STATISTICS.csv"), "rescue_audit": digest(OUT / "V11_TRACK_A_RESCUE_AUDIT.csv"), "groupwise_collisions": digest(OUT / "V11_TRACK_A_GROUPWISE_COLLISIONS.csv")},
        "definitions": {"complete_gold": "all gold_doc_ids are in the immutable Top-20", "cssr_conditional": "complete selected gold_doc_ids divided by complete_gold queries", "bootstrap": f"paired query bootstrap, B={B}, seed={SEED}"},
    }
    (OUT / "V11_TRACK_A_PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "V11_TRACK_A_ANALYSIS_SUMMARY.json").write_text(json.dumps({"metrics": summary, "runtime": runtime, "jev_calls": 0, "hosted_cloudflare_calls": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "queries": len(cohort), "policies": len(POLICIES), "metric_rows": len(metric_rows), "jev_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
