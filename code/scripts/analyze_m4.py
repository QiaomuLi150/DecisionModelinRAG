#!/usr/bin/env python3
"""M4 offline metrics, CAL comparator selection and paired bootstrap analysis."""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]
TYPED = ["laya_p5", "clef_p5", "laya_g5", "clef_g5"]
LOCAL_BASELINES = ["bge_p5", "qwen_p5"]
B = 5000
SEED = 731


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = " ".join(token for token in value.split() if token not in {"a", "an", "the"})
    return value


def f1_score(prediction: str, gold: str) -> float:
    pred = normalize(prediction).split()
    truth = normalize(gold).split()
    if not pred or not truth:
        return float(pred == truth)
    common = 0
    remaining = list(truth)
    for token in pred:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if not common:
        return 0.0
    precision = common / len(pred)
    recall = common / len(truth)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, question: dict) -> tuple[float, float]:
    golds = [question["answer"], *question.get("answer_aliases", [])]
    pred_norm = normalize(prediction)
    em = float(any(pred_norm == normalize(gold) for gold in golds))
    f1 = max(f1_score(prediction, gold) for gold in golds)
    return em, f1


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = (len(values) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def bootstrap_delta(a: list[float], b: list[float], seed: int = SEED) -> dict:
    if len(a) != len(b) or not a:
        return {"status": "NOT_MEASURED", "n": 0}
    delta = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    samples = []
    for _ in range(B):
        samples.append(sum(delta[rng.randrange(len(delta))] for _ in delta) / len(delta))
    return {"status": "OK" if len(delta) >= 40 else "UNDERPOWERED_DESCRIPTIVE", "n": len(delta), "delta": sum(delta) / len(delta), "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)], "bootstrap_replicates": B, "bootstrap_seed": seed}


def main() -> int:
    question_rows = {row["query_id"]: row for row in rows(ROOT / "data/manifests/questions.jsonl")}
    selection_rows = rows(ROOT / "data/results/m3/selection_results.jsonl")
    reader_rows = rows(ROOT / "data/results/m4/reader_results.jsonl")
    selections = {(row["policy_id"], row["query_id"]): row for row in selection_rows}
    metric_rows = []
    for reader in reader_rows:
        question = question_rows[reader["query_id"]]
        gold = set(question["gold_doc_ids"])
        top20 = set(reader["top20_doc_ids"])
        selected = set(reader["selected_doc_ids"])
        gold_present = gold.issubset(top20)
        support_recall = len(gold & selected) / len(gold)
        em, f1 = answer_metrics(reader["answer"], question) if reader["reader_status"] == "ok" else (0.0, 0.0)
        metric_rows.append({"query_id": reader["query_id"], "dataset": reader["dataset"], "split": question["split"], "query_type": question.get("query_type"), "hop_count": question.get("hop_count"), "policy_id": reader["policy_id"], "selected_size": len(reader["selected_doc_ids"]), "gold_present_top20": gold_present, "complete_support": bool(gold.issubset(selected)), "support_recall": support_recall, "answer_em": em, "answer_f1": f1, "reader_status": reader["reader_status"], "reader_input_tokens": reader.get("input_tokens"), "selection_wall_ms": selections[(reader["policy_id"], reader["query_id"])].get("selection_wall_ms", 0.0), "truncated_support_count": sum(1 for p in reader.get("view_info", {}).get("passages", []) if p.get("truncated"))})
    by_key = defaultdict(list)
    for row in metric_rows:
        by_key[(row["policy_id"], row["dataset"], row["split"])].append(row)
    metrics = {}
    for policy in POLICIES:
        policy_rows = [row for row in metric_rows if row["policy_id"] == policy]
        entry = {"n": len(policy_rows), "overall": {}, "by_dataset": {}, "by_split": {}}
        for label, subset in [("overall", policy_rows)] + [(dataset, [row for row in policy_rows if row["dataset"] == dataset]) for dataset in ("hotpot", "musique", "wiki2")] + [(split, [row for row in policy_rows if row["split"] == split]) for split in ("cal", "test")]:
            valid = [row for row in subset if row["reader_status"] == "ok"]
            eligible = [row for row in valid if row["gold_present_top20"]]
            target = entry["overall"] if label == "overall" else entry["by_dataset"].setdefault(label, {}) if label in ("hotpot", "musique", "wiki2") else entry["by_split"].setdefault(label, {})
            target.update({"n": len(subset), "reader_success": len(valid), "cssr_unconditional": sum(row["complete_support"] for row in valid) / len(subset) if subset else None, "support_recall": statistics.mean(row["support_recall"] for row in valid) if valid else None, "cssr_gold_present": sum(row["complete_support"] for row in eligible) / len(eligible) if eligible else None, "gold_present_n": len(eligible), "answer_em": statistics.mean(row["answer_em"] for row in valid) if valid else None, "answer_f1": statistics.mean(row["answer_f1"] for row in valid) if valid else None, "reader_input_tokens_mean": statistics.mean(row["reader_input_tokens"] for row in valid if row.get("reader_input_tokens") is not None) if any(row.get("reader_input_tokens") is not None for row in valid) else None, "selection_p50_ms": percentile([row["selection_wall_ms"] for row in subset], 0.5) if subset else None, "selection_p95_ms": percentile([row["selection_wall_ms"] for row in subset], 0.95) if subset else None})
        metrics[policy] = entry

    # CAL-only comparator freezing.  CSSR uses the common complete-support
    # eligible set; ties then use answer F1 and lower selection p95.
    cal_scores = {}
    for policy in LOCAL_BASELINES:
        cal = metrics[policy]["by_split"]["cal"]
        cal_scores[policy] = {"cssr_gold_present": cal["cssr_gold_present"], "answer_f1": cal["answer_f1"], "selection_p95_ms": cal["selection_p95_ms"]}
    best_local = sorted(LOCAL_BASELINES, key=lambda p: (-cal_scores[p]["cssr_gold_present"], -cal_scores[p]["answer_f1"], cal_scores[p]["selection_p95_ms"], p))[0]

    stats = {"bootstrap_replicates": B, "bootstrap_seed": SEED, "comparisons": {}}
    for policy in TYPED:
        for comparator in ["bge_p5", "qwen_p5", best_local]:
            comparisons = {}
            for dataset in ("hotpot", "musique", "wiki2"):
                a = [row["complete_support"] for row in metric_rows if row["policy_id"] == policy and row["dataset"] == dataset and row["split"] == "test" and row["gold_present_top20"]]
                b = [row["complete_support"] for row in metric_rows if row["policy_id"] == comparator and row["dataset"] == dataset and row["split"] == "test" and row["gold_present_top20"]]
                comparisons[dataset] = bootstrap_delta([float(x) for x in a], [float(x) for x in b])
            macro_values = []
            rng = random.Random(SEED)
            per_dataset = {}
            for dataset in ("hotpot", "musique", "wiki2"):
                a = [float(row["complete_support"]) for row in metric_rows if row["policy_id"] == policy and row["dataset"] == dataset and row["split"] == "test" and row["gold_present_top20"]]
                b = [float(row["complete_support"]) for row in metric_rows if row["policy_id"] == comparator and row["dataset"] == dataset and row["split"] == "test" and row["gold_present_top20"]]
                per_dataset[dataset] = (a, b)
            if all(per_dataset[d][0] and len(per_dataset[d][0]) == len(per_dataset[d][1]) for d in per_dataset):
                observed = statistics.mean(statistics.mean(a[i] - b[i] for i in range(len(a))) for a, b in per_dataset.values())
                for _ in range(B):
                    values = []
                    for dataset in ("hotpot", "musique", "wiki2"):
                        a, b = per_dataset[dataset]
                        values.append(sum((lambda index: a[index] - b[index])(rng.randrange(len(a))) for _ in a) / len(a))
                    macro_values.append(statistics.mean(values))
                comparisons["equal_dataset_macro"] = {"status": "OK" if min(len(a) for a, _ in per_dataset.values()) >= 40 else "UNDERPOWERED_DESCRIPTIVE", "n_by_dataset": {d: len(a) for d, (a, _) in per_dataset.items()}, "delta": observed, "ci95": [percentile(macro_values, 0.025), percentile(macro_values, 0.975)], "bootstrap_replicates": B, "bootstrap_seed": SEED}
            stats["comparisons"][f"{policy}_vs_{comparator}"] = comparisons
    output = {"study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2", "created_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics, "cal_comparator_scores": cal_scores, "best_local_baseline": best_local, "metric_definition": {"answer_normalization": "lowercase, remove punctuation/articles, whitespace normalization; MuSiQue answer aliases accepted", "binary_support": "complete gold_doc_ids set retained", "conditional_denominator": "gold_doc_ids subset of immutable Top-20", "failure_inclusive_answer_metrics": True}, "statistics": stats, "reader_status_counts": {status: sum(1 for row in metric_rows if row["reader_status"] == status) for status in {row["reader_status"] for row in metric_rows}}, "metric_rows": len(metric_rows), "jev_calls": 0, "hosted_cloudflare_calls": 0}
    path = ROOT / "data/results/m4/m4_metrics.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "results/audit/phase_state.json").write_text(json.dumps({"current_milestone": "M4", "status": "DONE", "jev_calls": 0, "jev_execution_allowed": False, "m4_metrics": str(path), "best_local_baseline": best_local}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reader_rows": len(reader_rows), "best_local_baseline": best_local, "cal_comparator_scores": cal_scores, "jev_calls": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
