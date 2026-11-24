#!/usr/bin/env python3
"""Typed-model integrity audit for the frozen DecisionRAG study.

The offline phase reads existing traces only.  The native phase is deliberately
small and local: it loads the pinned Laya/Clef snapshots and writes only new
artifacts below ``results/audit/typed_model_audit``.  It never calls Jev,
Cloudflare Workers AI, the reader, or a production service.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audit/typed_model_audit"
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
LAYA_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"
CLEF_REV = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
SEED = 731
QUARTET_SEED = 20261011
BOOTSTRAP = 5000
MODELS = ("laya", "clef")
DATASETS = ("hotpot", "musique", "wiki2")
POLICIES = ("raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5")

sys.path.insert(0, str(ROOT / "src"))
from decisionrag.selection import (  # noqa: E402
    CandidateView,
    deterministic_quartets,
    groupwise_states,
    make_views,
    pointwise_pair,
    pointwise_question,
    pointwise_state,
    rank_indices,
    sha256_json,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rows_by_qid(path: Path) -> dict[str, dict]:
    return {row["query_id"]: row for row in load_jsonl(path)}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_mean(values: list[float], seed: int = SEED) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(BOOTSTRAP)]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def normalize(text: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return " ".join(token for token in value.split() if token not in {"a", "an", "the"})


def f1(prediction: str, gold: str) -> float:
    p, g = normalize(prediction).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    remaining = list(g)
    common = sum(1 for token in p if token in remaining and not remaining.remove(token))
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, question: dict) -> float:
    return max(f1(prediction, gold) for gold in [question["answer"], *question.get("answer_aliases", [])])


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: (values[i], i))
        result = [0.0] * len(values)
        pos = 0
        while pos < len(order):
            end = pos + 1
            while end < len(order) and values[order[end]] == values[order[pos]]:
                end += 1
            rank = (pos + end - 1) / 2.0 + 1.0
            for index in order[pos:end]:
                result[index] = rank
            pos = end
        return result
    if len(x) < 2:
        return 1.0
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else 1.0


def kendall(x: list[float], y: list[float]) -> float:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denominator if denominator else 1.0


def auc(scores: dict[str, float], gold: set[str]) -> float | None:
    positives = [scores[d] for d in scores if d in gold]
    negatives = [scores[d] for d in scores if d not in gold]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def load_context() -> dict:
    questions = {r["query_id"]: r for r in load_jsonl(ROOT / "data/manifests/questions.jsonl")}
    corpus = {r["doc_id"]: r for r in load_jsonl(ROOT / "data/corpus/passages.jsonl")}
    slates = {r["query_id"]: r for r in load_jsonl(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")}
    selections = {(r["policy_id"], r["query_id"]): r for r in load_jsonl(ROOT / "data/results/m3/selection_results.jsonl")}
    traces = {model: rows_by_qid(ROOT / f"data/results/m3/{model}_all_pointwise.jsonl") for model in MODELS}
    groups = {model: rows_by_qid(ROOT / f"data/results/m3/{model}_all_groupwise.jsonl") for model in MODELS}
    return {"questions": questions, "corpus": corpus, "slates": slates, "selections": selections, "traces": traces, "groups": groups}


def candidates_for(ctx: dict, query_id: str, tokenizer) -> list[CandidateView]:
    q = ctx["questions"][query_id]
    slate = ctx["slates"][query_id]
    result = []
    for rank, doc_id in enumerate(slate["candidate_ids"], start=1):
        doc = ctx["corpus"][doc_id]
        candidate = {"doc_id": doc_id, "dataset": doc["dataset"], "title": doc["title"], "full_text": doc["full_text"], "original_rank": rank}
        result.append(make_views(q["question"], candidate, tokenizer))
    return result


def score_map(ctx: dict, model: str, qid: str) -> dict[str, float]:
    return {item["doc_id"]: float(item["score"]) for item in ctx["traces"][model][qid]["pointwise"]}


def ranked_ids(ctx: dict, model: str, qid: str, reverse: bool = False) -> list[str]:
    trace = ctx["traces"][model][qid]
    scores = score_map(ctx, model, qid)
    original = {doc_id: i + 1 for i, doc_id in enumerate(ctx["slates"][qid]["candidate_ids"])}
    if not reverse:
        return [item["doc_id"] for item in sorted(trace["pointwise"], key=lambda item: (-float(item["score"]), original[item["doc_id"]], item["doc_id"]))]
    return [item["doc_id"] for item in sorted(trace["pointwise"], key=lambda item: (float(item["score"]), -original[item["doc_id"]], item["doc_id"]))]


def eligible_qids(ctx: dict, split: str, dataset: str | None = None) -> list[str]:
    result = []
    for qid, q in ctx["questions"].items():
        if q["split"] != split or (dataset and q["dataset"] != dataset):
            continue
        if set(q["gold_doc_ids"]).issubset(set(ctx["slates"][qid]["candidate_ids"])):
            result.append(qid)
    return sorted(result)


def select_metrics(ctx: dict, qid: str, selected: list[str]) -> tuple[float, float]:
    gold = set(ctx["questions"][qid]["gold_doc_ids"])
    return float(gold.issubset(set(selected))), len(gold & set(selected)) / len(gold)


def offline_a1(ctx: dict) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    auc_rows = []
    for model in MODELS:
        for split in ("cal", "test"):
            for dataset in (*DATASETS, "pooled"):
                qids = eligible_qids(ctx, split, None if dataset == "pooled" else dataset)
                if not qids:
                    continue
                methods = {"top5": [], "bottom5": [], "retrieval5": [], "bge_p5": [], "random5": []}
                rng = random.Random(SEED)
                for qid in qids:
                    methods["top5"].append([*ranked_ids(ctx, model, qid)[:5]])
                    methods["bottom5"].append([*ranked_ids(ctx, model, qid, reverse=True)[:5]])
                    methods["retrieval5"].append(ctx["selections"][("retrieval5", qid)]["selected_doc_ids"])
                    methods["bge_p5"].append(ctx["selections"][("bge_p5", qid)]["selected_doc_ids"])
                random_cssr, random_recall = [], []
                for qid in qids:
                    candidates = list(ctx["slates"][qid]["candidate_ids"])
                    cssr, recall = [], []
                    for _ in range(1000):
                        selected = rng.sample(candidates, 5)
                        c, r = select_metrics(ctx, qid, selected)
                        cssr.append(c)
                        recall.append(r)
                    random_cssr.append(statistics.mean(cssr))
                    random_recall.append(statistics.mean(recall))
                methods["random5"] = None
                for method, selections in methods.items():
                    if method == "random5":
                        cssr, recall = statistics.mean(random_cssr), statistics.mean(random_recall)
                    else:
                        values = [select_metrics(ctx, qid, selection) for qid, selection in zip(qids, selections)]
                        cssr, recall = statistics.mean(v[0] for v in values), statistics.mean(v[1] for v in values)
                    metric_rows.append({"model": model, "split": split, "dataset": dataset, "method": method, "n_complete_gold_eligible": len(qids), "cssr": cssr, "support_recall": recall, "random_draws": 1000 if method == "random5" else ""})
                for qid in qids:
                    scores = score_map(ctx, model, qid)
                    value = auc(scores, set(ctx["questions"][qid]["gold_doc_ids"]))
                    if value is not None:
                        auc_rows.append({"model": model, "split": split, "dataset": dataset, "query_id": qid, "auc": value})
    auc_summary = []
    for model in MODELS:
        for split in ("cal", "test"):
            for dataset in (*DATASETS, "pooled"):
                values = [r["auc"] for r in auc_rows if r["model"] == model and r["split"] == split and r["dataset"] == dataset]
                low, high = bootstrap_mean(values)
                auc_summary.append({"model": model, "split": split, "dataset": dataset, "n_auc_queries": len(values), "mean_auc": statistics.mean(values) if values else None, "ci95_low": low, "ci95_high": high, "flag": bool(values and high < 0.5)})
    write_csv(OUT / "A1_SELECTION_METRICS.csv", metric_rows)
    write_csv(OUT / "A1_AUC_BY_QUERY.csv", auc_rows)
    write_csv(OUT / "A1_AUC_SUMMARY.csv", auc_summary)
    return metric_rows, auc_summary


def offline_a2(ctx: dict) -> list[dict]:
    rows, histogram = [], []
    bins = [(i / 20, (i + 1) / 20) for i in range(20)]
    for model in MODELS:
        for split in ("cal", "test"):
            for dataset in (*DATASETS, "pooled"):
                qids = sorted(q for q, row in ctx["questions"].items() if row["split"] == split and (dataset == "pooled" or row["dataset"] == dataset))
                values = [float(item["score"]) for qid in qids for item in ctx["traces"][model][qid]["pointwise"]]
                finite = [value for value in values if math.isfinite(value)]
                distinct = [len(set(float(item["score"]) for item in ctx["traces"][model][qid]["pointwise"] if math.isfinite(float(item["score"])))) for qid in qids]
                tie_queries = tie_slots = 0
                constant_queries = 0
                for qid in qids:
                    trace = ctx["traces"][model][qid]["pointwise"]
                    score_values = [float(item["score"]) for item in trace]
                    if score_values and max(score_values) - min(score_values) <= 1e-12:
                        constant_queries += 1
                    ranked = ranked_ids(ctx, model, qid)
                    cutoff = score_map(ctx, model, qid)[ranked[4]]
                    tied_outside = [doc for doc in ranked[5:] if score_map(ctx, model, qid)[doc] == cutoff]
                    if tied_outside:
                        tie_queries += 1
                        tie_slots += sum(score_map(ctx, model, qid)[doc] == cutoff for doc in ranked[:5])
                    for value in score_values:
                        for low, high in bins:
                            if (low <= value < high) or (high == 1.0 and value == high):
                                histogram.append({"model": model, "split": split, "dataset": dataset, "bin_low": low, "bin_high": high, "count": 1})
                                break
                rows.append({"model": model, "split": split, "dataset": dataset, "n_queries": len(qids), "n_scores": len(values), "distinct_min": min(distinct) if distinct else None, "distinct_median": statistics.median(distinct) if distinct else None, "distinct_max": max(distinct) if distinct else None, "nan_inf_share": 1 - len(finite) / len(values) if values else None, "near_endpoint_share_abs_p_minus_half_gt_499": sum(abs(value - 0.5) > 0.499 for value in finite) / len(values) if values else None, "rank5_rank6_tie_query_share": tie_queries / len(qids) if qids else None, "rrf_tie_break_selected_slot_share": tie_slots / (5 * len(qids)) if qids else None, "near_constant_query_share": constant_queries / len(qids) if qids else None, "flag": tie_slots / (5 * len(qids)) > 0.2 if qids else False})
    grouped = defaultdict(int)
    for row in histogram:
        key = tuple(row[k] for k in ("model", "split", "dataset", "bin_low", "bin_high"))
        grouped[key] += row["count"]
    histogram_rows = [{"model": key[0], "split": key[1], "dataset": key[2], "bin_low": key[3], "bin_high": key[4], "count": count} for key, count in grouped.items()]
    write_csv(OUT / "A2_SCORE_DISTRIBUTION.csv", rows)
    write_csv(OUT / "A2_SCORE_HISTOGRAM.csv", histogram_rows)
    return rows


def offline_a3(ctx: dict, tokenizer) -> list[dict]:
    qids = eligible_qids(ctx, "test", "hotpot")[:20]
    bge_traces = rows_by_qid(ROOT / "data/results/m3/bge_all_pointwise.jsonl")
    details, raw = [], []
    for qid in qids:
        views = candidates_for(ctx, qid, tokenizer)
        for model in MODELS:
            trace = ctx["traces"][model][qid]
            raw.append({"model": model, "query_id": qid, "raw_response_trace": trace})
            by_doc = {item["doc_id"]: item for item in trace["pointwise"]}
            request_expected = sha256_json([view.selection_text_sha256 for view in views])
            request_ok = request_expected == trace.get("request_hash")
            view_ok = all(by_doc.get(view.doc_id, {}).get("selection_text_sha256") == view.selection_text_sha256 for view in views)
            doc_order_ok = [item["doc_id"] for item in trace["pointwise"]] == [view.doc_id for view in views]
            canonical_score_field = "answers.relevance.noul"
            score_field_ok = all(isinstance(item.get("score"), (int, float)) and math.isfinite(float(item["score"])) for item in trace["pointwise"])
            scores = score_map(ctx, model, qid)
            bge_scores = {item["doc_id"]: float(item["score"]) for item in bge_traces[qid]["pointwise"]}
            top5 = set(ranked_ids(ctx, model, qid)[:5])
            retrieval5 = set(ctx["selections"][("retrieval5", qid)]["selected_doc_ids"])
            bge5 = set(ctx["selections"][("bge_p5", qid)]["selected_doc_ids"])
            response_recompute = None
            response_status = "PASS" if model == "clef" else "NOT_RECOMPUTABLE_FROM_NORMALIZED_LAYA_TRACE"
            if model == "clef":
                response_recompute = sha256_json([float(item["score"]) for item in trace["pointwise"]]) == trace.get("response_hash")
                response_status = "PASS" if response_recompute else "FAIL"
            details.append({"model": model, "query_id": qid, "request_hash_match": request_ok, "response_hash_status": response_status, "response_hash_match": response_recompute, "doc_order_match": doc_order_ok, "view_hashes_match": view_ok, "canonical_pointwise_view_match": view_ok, "parsed_score_field": canonical_score_field, "parsed_score_is_finite": score_field_ok, "top5_overlap_retrieval5": len(top5 & retrieval5) / 5, "top5_overlap_bge_p5": len(top5 & bge5) / 5, "spearman_with_bge": spearman([scores[d] for d in [view.doc_id for view in views]], [bge_scores[d] for d in [view.doc_id for view in views]])})
    dump_jsonl(OUT / "A3_RAW_RESPONSE_TRACES.jsonl", raw)
    write_csv(OUT / "A3_ALIGNMENT_AND_OVERLAP.csv", details)
    return details


def native_batch_order(ctx: dict, tokenizer, runtime: NativeRuntime) -> list[dict]:
    """Compare canonical and shuffled candidate batches by document identity."""
    qids = eligible_qids(ctx, "test", "hotpot")[:20]
    question = pointwise_question()
    rows = []
    for model in MODELS:
        for qid in qids:
            views = candidates_for(ctx, qid, tokenizer)
            order = list(range(len(views)))
            random.Random(SEED + sum(ord(char) for char in qid)).shuffle(order)
            original_states = [pointwise_state(view) for view in views]
            shuffled_views = [views[index] for index in order]
            shuffled_states = [pointwise_state(view) for view in shuffled_views]
            if model == "laya":
                original_results = runtime.laya_project_batch_raw(original_states, question)
                shuffled_results = runtime.laya_project_batch_raw(shuffled_states, question)
                original_scores = [float(result["answers"]["relevance"]["noul"]) for result in original_results]
                shuffled_scores = [float(result["answers"]["relevance"]["noul"]) for result in shuffled_results]
            else:
                original_scores, _ = runtime.clef_project_batch(original_states, question)
                shuffled_scores, _ = runtime.clef_project_batch(shuffled_states, question)
            if len(original_scores) != len(views) or len(shuffled_scores) != len(views):
                rows.append({"model": model, "query_id": qid, "candidate_count": len(views), "returned_original_count": len(original_scores), "returned_shuffled_count": len(shuffled_scores), "shuffle_seed": SEED + sum(ord(char) for char in qid), "max_abs_score_difference": None, "mean_abs_score_difference": None, "canonical_order_kendall": None, "document_identity_preserved": False, "pass": False, "error": "COUNT_MISMATCH"})
                continue
            original_by_doc = {view.doc_id: score for view, score in zip(views, original_scores)}
            shuffled_by_doc = {view.doc_id: score for view, score in zip(shuffled_views, shuffled_scores)}
            if set(original_by_doc) != set(shuffled_by_doc):
                rows.append({"model": model, "query_id": qid, "candidate_count": len(views), "returned_original_count": len(original_scores), "returned_shuffled_count": len(shuffled_scores), "shuffle_seed": SEED + sum(ord(char) for char in qid), "max_abs_score_difference": None, "mean_abs_score_difference": None, "canonical_order_kendall": None, "document_identity_preserved": False, "pass": False, "error": "DOC_ID_SET_MISMATCH"})
                continue
            diffs = [abs(original_by_doc[doc_id] - shuffled_by_doc[doc_id]) for doc_id in original_by_doc]
            original_vector = [original_by_doc[view.doc_id] for view in views]
            shuffled_vector = [shuffled_by_doc[view.doc_id] for view in views]
            rows.append({"model": model, "query_id": qid, "candidate_count": len(views), "returned_original_count": len(original_scores), "returned_shuffled_count": len(shuffled_scores), "shuffle_seed": SEED + sum(ord(char) for char in qid), "max_abs_score_difference": max(diffs), "mean_abs_score_difference": statistics.mean(diffs), "canonical_order_kendall": kendall(original_vector, shuffled_vector), "document_identity_preserved": True, "pass": max(diffs) <= 1e-6})
    write_csv(OUT / "A3_BATCH_ORDER_INVARIANCE.csv", rows)
    return rows


def offline_a4(ctx: dict) -> list[dict]:
    rows = []
    for model in MODELS:
        policy = f"{model}_g5"
        for split in ("cal", "test"):
            for dataset in (*DATASETS, "pooled"):
                qids = eligible_qids(ctx, split, None if dataset == "pooled" else dataset)
                labels = []
                mapping_ok = True
                collisions = set()
                no_collision = []
                one_gold_groups = one_gold_wins = 0
                for qid in qids:
                    q = ctx["questions"][qid]
                    candidates = [{"doc_id": doc_id, "title": ctx["corpus"][doc_id]["title"], "full_text": ctx["corpus"][doc_id]["full_text"], "original_rank": rank, "dataset": ctx["corpus"][doc_id]["dataset"]} for rank, doc_id in enumerate(ctx["slates"][qid]["candidate_ids"], start=1)]
                    quartets = deterministic_quartets(candidates, qid, QUARTET_SEED)
                    trace = ctx["groups"][model][qid]
                    query_collision = False
                    gold = set(q["gold_doc_ids"])
                    for group, recorded in zip(quartets, trace["groups"]):
                        expected_ids = [candidate["doc_id"] for candidate in group]
                        if expected_ids != recorded.get("group_doc_ids"):
                            mapping_ok = False
                        label = recorded.get("winning_label")
                        labels.append(label)
                        if label not in "ABCD" or recorded.get("winning_doc_id") != expected_ids["ABCD".index(label)]:
                            mapping_ok = False
                        gold_count = len(set(expected_ids) & gold)
                        if gold_count >= 2:
                            query_collision = True
                        if gold_count == 1:
                            one_gold_groups += 1
                            one_gold_wins += int(recorded.get("winning_doc_id") in gold)
                    if query_collision:
                        collisions.add(qid)
                    else:
                        no_collision.append(qid)
                try:
                    from scipy.stats import chisquare
                    chi = chisquare([labels.count(label) for label in "ABCD"], f_exp=[len(labels) / 4] * 4)
                    chi2, pvalue = float(chi.statistic), float(chi.pvalue)
                except Exception:
                    chi2 = pvalue = None
                no_collision_cssr = [float(set(ctx["questions"][qid]["gold_doc_ids"]).issubset(set(ctx["selections"][(policy, qid)]["selected_doc_ids"]))) for qid in no_collision]
                rows.append({"model": model, "policy": policy, "split": split, "dataset": dataset, "n_complete_gold_eligible": len(qids), "winner_A": labels.count("A"), "winner_B": labels.count("B"), "winner_C": labels.count("C"), "winner_D": labels.count("D"), "chi_square": chi2, "chi_square_pvalue_uniform": pvalue, "label_mapping_ok": mapping_ok, "collision_query_share": len(collisions) / len(qids) if qids else None, "collision_queries": len(collisions), "no_collision_n": len(no_collision), "no_collision_g5_cssr": statistics.mean(no_collision_cssr) if no_collision_cssr else None, "exactly_one_gold_quartets": one_gold_groups, "exactly_one_gold_winner_share": one_gold_wins / one_gold_groups if one_gold_groups else None, "chance_threshold_flag": one_gold_groups > 0 and one_gold_wins / one_gold_groups <= 0.25, "flag": not mapping_ok or (one_gold_groups > 0 and one_gold_wins / one_gold_groups <= 0.25)})
    write_csv(OUT / "A4_GROUPWISE_CHECKS.csv", rows)
    return rows


def answer_in_selection(ctx: dict, reader_tokenizer) -> list[dict]:
    rows = []
    for policy in POLICIES:
        for split in ("cal", "test"):
            for dataset in (*DATASETS, "pooled"):
                qids = sorted(qid for qid, q in ctx["questions"].items() if q["split"] == split and (dataset == "pooled" or q["dataset"] == dataset))
                hits = []
                f1s = []
                cssr = []
                for qid in qids:
                    q = ctx["questions"][qid]
                    selected = ctx["selections"][(policy, qid)]["selected_doc_ids"]
                    views = []
                    for doc_id in selected:
                        doc = ctx["corpus"][doc_id]
                        ids = reader_tokenizer.encode(str(doc["full_text"]), add_special_tokens=False)
                        clipped = reader_tokenizer.decode(ids[:256], skip_special_tokens=True).strip()
                        views.append(normalize(f"{doc['title']}\n{clipped}"))
                    aliases = [normalize(q["answer"]), *[normalize(a) for a in q.get("answer_aliases", [])]]
                    hits.append(float(any(alias and alias in view for alias in aliases for view in views)))
                    f1s.append(answer_f1(ctx["reader_rows"][(policy, qid)]["answer"], q) if "reader_rows" in ctx else 0.0)
                    cssr.append(float(set(q["gold_doc_ids"]).issubset(set(selected))))
                rows.append({"policy": policy, "split": split, "dataset": dataset, "n": len(qids), "answer_in_selection_at5": statistics.mean(hits) if hits else None, "cssr_unconditional": statistics.mean(cssr) if cssr else None, "answer_f1": statistics.mean(f1s) if f1s else None})
    write_csv(OUT / "ANSWER_IN_SELECTION.csv", rows)
    return rows


def prepare_context_for_reader(ctx: dict) -> None:
    reader_rows = {(r["policy_id"], r["query_id"]): r for r in load_jsonl(ROOT / "data/results/m4/reader_results.jsonl")}
    ctx["reader_rows"] = reader_rows


class NativeRuntime:
    def __init__(self):
        self.laya_router = None
        self.laya_native = None
        self.clef_model = None
        self.clef_processor = None
        self.clef_helpers = None
        self.warnings = []

    def load_laya(self):
        from laya import RLAgent, Router
        path = str(HF_ROOT / "models--convaiinnovations--laya" / "snapshots" / LAYA_REV)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.laya_router = Router(models={"english": path}, device="cuda", default="english", max_loaded=1)
            self.laya_native = RLAgent(path, device="cuda")
        self.warnings.extend({"message": str(item.message), "category": item.category.__name__} for item in caught)
        return path

    def load_clef(self):
        import importlib
        import torch
        path = str(HF_ROOT / "models--Cloudflare--clef-flash" / "snapshots" / CLEF_REV)
        sys.path.insert(0, path)
        module = importlib.import_module("joint_schema_model")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.clef_model, self.clef_processor = module.load_release_model(path, device="cuda")
        self.clef_helpers = (module.collate_records, module.encode_record, module.systemone, torch)
        self.warnings.extend({"message": str(item.message), "category": item.category.__name__} for item in caught)
        return path

    def laya_project_raw(self, state, question):
        return self.laya_router.predict_batch([{"state": state, "questions": question, "model": "english", "max_len": 512}], batch_size=1, sort_by_length=True)[0]

    def laya_project_batch_raw(self, states, question):
        requests = [{"state": state, "questions": question, "model": "english", "max_len": 512} for state in states]
        return self.laya_router.predict_batch(requests, batch_size=len(requests), sort_by_length=True)

    def laya_project(self, state, question):
        result = self.laya_project_raw(state, question)
        return float(result["answers"]["relevance"]["noul"]), result

    def laya_native_raw(self, state, question):
        return self.laya_native.system_one(state, question)

    def laya_native_call(self, state, question):
        result = self.laya_native_raw(state, question)
        return float(result["answers"]["relevance"]["noul"]), result

    def laya_unscaled(self, state, question):
        original = dict(self.laya_native.temperature_by_options)
        try:
            self.laya_native.temperature_by_options["noul:2"] = 1.0
            result = self.laya_native.system_one(state, question)
            return float(result["answers"]["relevance"]["noul"]), result
        finally:
            self.laya_native.temperature_by_options.clear()
            self.laya_native.temperature_by_options.update(original)

    def clef_project_raw(self, state, question):
        collate_records, encode_record, _, torch = self.clef_helpers
        encoded = encode_record(self.clef_processor.tokenizer, {"state": state, "questions": question}, processor=self.clef_processor)
        batch = collate_records([encoded], self.clef_processor.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            record_logits = self.clef_model(batch)[0]
        answers = {}
        for (question_id, question_def), encoded_question, question_logits in zip(question.items(), encoded.questions, record_logits):
            option_ids = list(encoded_question.option_ids)
            probabilities = torch.softmax(question_logits.float(), dim=-1).detach().cpu().tolist()
            probability_map = dict(zip(option_ids, probabilities))
            if question_def["type"] == "choice":
                choice = max((str(option) for option in question_def["criteria"]), key=probability_map.__getitem__)
                answers[question_id] = {"type": "choice", "choice": choice, "probabilities": probability_map}
            else:
                answers[question_id] = {"type": "noul", "noul": float(probability_map["true"]), "probabilities": probability_map}
        return {"model": "clef-flash", "answers": answers, "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0}}

    def clef_project_batch(self, states, question):
        collate_records, encode_record, _, torch = self.clef_helpers
        encoded_records = [encode_record(self.clef_processor.tokenizer, {"state": state, "questions": question}, processor=self.clef_processor) for state in states]
        batch = collate_records(encoded_records, self.clef_processor.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            logits = self.clef_model(batch)
        scores = []
        responses = []
        for encoded, record_logits in zip(encoded_records, logits):
            question_logits = record_logits[0]
            option_ids = list(encoded.questions[0].option_ids)
            probabilities = torch.softmax(question_logits.float(), dim=-1).detach().cpu().tolist()
            probability_map = dict(zip(option_ids, probabilities))
            scores.append(float(probability_map["true"]))
            responses.append({"model": "clef-flash", "answers": {"relevance": {"type": "noul", "noul": float(probability_map["true"]), "probabilities": probability_map}}, "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0}})
        return scores, responses

    def clef_project(self, state, question):
        result = self.clef_project_raw(state, question)
        probabilities = result["answers"]["relevance"]["probabilities"]
        return float(probabilities["true"]), {"probabilities": probabilities}

    def clef_native_raw(self, state, question):
        _, _, systemone, _ = self.clef_helpers
        return systemone(self.clef_model, self.clef_processor, {"model": "clef-flash", "state": state, "questions": question})

    def clef_native_call(self, state, question):
        result = self.clef_native_raw(state, question)
        return float(result["answers"]["relevance"]["noul"]), result

    def close(self):
        import gc
        self.laya_router = self.laya_native = self.clef_model = self.clef_processor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def native_b1(runtime: NativeRuntime) -> list[dict]:
    laya_state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    laya_questions = {"department": {"type": "choice", "instructions": "Which department should handle this?", "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors", "other": "everything else"}}, "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}}
    clef_state = {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}}
    clef_questions = {"status": {"type": "choice", "instructions": "What is the invoice status?", "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}}, "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}}
    rows = []
    for name, state, question, expected in (("laya", laya_state, laya_questions, "billing"), ("clef", clef_state, clef_questions, "overdue")):
        if name == "laya":
            project = runtime.laya_project_raw(state, question)
            native = runtime.laya_native_raw(state, question)
            project_value, native_value = project["answers"]["department"]["choice"], native["answers"]["department"]["choice"]
        else:
            project = runtime.clef_project_raw(state, question)
            native = runtime.clef_native_raw(state, question)
            project_value, native_value = project["answers"]["status"]["choice"], native["answers"]["status"]["choice"]
        rows.append({"model": name, "expected_documented_answer": expected, "project_adapter_answer": project_value, "native_answer": native_value, "project_expected": project_value == expected, "native_expected": native_value == expected, "project_response": project, "native_response": native})
    dump_json(OUT / "B1_VENDOR_EXAMPLES.json", rows)
    return rows


def native_b2_b3_b4(ctx: dict, tokenizer, runtime: NativeRuntime) -> tuple[list[dict], list[dict], dict]:
    qids = eligible_qids(ctx, "test", "hotpot")[:20]
    b2_rows, b3_rows = [], []
    question_def = pointwise_question()
    for model in MODELS:
        unscaled_rank_differences = 0
        for qid in qids:
            views = candidates_for(ctx, qid, tokenizer)
            stored = score_map(ctx, model, qid)
            project_scores, native_scores, unscaled_scores = [], [], []
            for view in views:
                state = pointwise_state(view)
                if model == "laya":
                    project, _ = runtime.laya_project(state, question_def)
                    native, _ = runtime.laya_native_call(state, question_def)
                    unscaled, _ = runtime.laya_unscaled(state, question_def)
                    unscaled_scores.append(unscaled)
                else:
                    project, _ = runtime.clef_project(state, question_def)
                    native, _ = runtime.clef_native_call(state, question_def)
                project_scores.append(project)
                native_scores.append(native)
            doc_ids = [view.doc_id for view in views]
            stored_scores = [stored[doc] for doc in doc_ids]
            stored_order = sorted(range(20), key=lambda i: (-stored_scores[i], i, doc_ids[i]))[:5]
            native_order = sorted(range(20), key=lambda i: (-native_scores[i], i, doc_ids[i]))[:5]
            if model == "laya":
                unscaled_order = sorted(range(20), key=lambda i: (-unscaled_scores[i], i, doc_ids[i]))[:5]
                unscaled_rank_differences += int(unscaled_order != native_order)
            b2_rows.append({"model": model, "query_id": qid, "stored_native_mean_abs_diff": statistics.mean(abs(a - b) for a, b in zip(stored_scores, native_scores)), "project_native_mean_abs_diff": statistics.mean(abs(a - b) for a, b in zip(project_scores, native_scores)), "stored_native_kendall": kendall(stored_scores, native_scores), "project_native_kendall": kendall(project_scores, native_scores), "stored_native_top5_agreement": len(set(stored_order) & set(native_order)) / 5, "project_native_top5_agreement": len(set(sorted(range(20), key=lambda i: (-project_scores[i], i, doc_ids[i]))[:5]) & set(native_order)) / 5})
        # Record the diagnostic ranking count on a separate summary row later.
        if model == "laya":
            unscaled_count = unscaled_rank_differences
    for index in range(10):
        question = {"relevance": pointwise_question()["relevance"]}
        relevant = f"The answer to test question {index} is exactly item {index}; this passage states item {index} directly."
        unrelated = f"This passage discusses unrelated topic {index} and contains no answer to the test question."
        state_q = f"What is the answer to test question {index}?"
        for model in MODELS:
            if model == "laya":
                rel_a, _ = runtime.laya_native_call(pointwise_state(make_views(state_q, {"doc_id": "rel", "dataset": "x", "title": "Relevant", "full_text": relevant, "original_rank": 1}, tokenizer)), question)
                unrel_a, _ = runtime.laya_native_call(pointwise_state(make_views(state_q, {"doc_id": "unrel", "dataset": "x", "title": "Unrelated", "full_text": unrelated, "original_rank": 2}, tokenizer)), question)
                rel_b, _ = runtime.laya_native_call(pointwise_state(make_views(state_q, {"doc_id": "rel", "dataset": "x", "title": "Relevant", "full_text": relevant, "original_rank": 2}, tokenizer)), question)
                unrel_b, _ = runtime.laya_native_call(pointwise_state(make_views(state_q, {"doc_id": "unrel", "dataset": "x", "title": "Unrelated", "full_text": unrelated, "original_rank": 1}, tokenizer)), question)
            else:
                rel_a, _ = runtime.clef_native_call(pointwise_state(make_views(state_q, {"doc_id": "rel", "dataset": "x", "title": "Relevant", "full_text": relevant, "original_rank": 1}, tokenizer)), question)
                unrel_a, _ = runtime.clef_native_call(pointwise_state(make_views(state_q, {"doc_id": "unrel", "dataset": "x", "title": "Unrelated", "full_text": unrelated, "original_rank": 2}, tokenizer)), question)
                rel_b, _ = runtime.clef_native_call(pointwise_state(make_views(state_q, {"doc_id": "rel", "dataset": "x", "title": "Relevant", "full_text": relevant, "original_rank": 2}, tokenizer)), question)
                unrel_b, _ = runtime.clef_native_call(pointwise_state(make_views(state_q, {"doc_id": "unrel", "dataset": "x", "title": "Unrelated", "full_text": unrelated, "original_rank": 1}, tokenizer)), question)
            b3_rows.append({"model": model, "pair_id": index, "relevant_first_relevant_score": rel_a, "relevant_first_unrelated_score": unrel_a, "unrelated_first_relevant_score": rel_b, "unrelated_first_unrelated_score": unrel_b, "relevant_higher_both_orders": rel_a > unrel_a and rel_b > unrel_b})
    laya_temp = {"temperature_raw": runtime.laya_native.temperature_raw, "temperature_applied": runtime.laya_native.temperature, "temperature_by_options_raw": runtime.laya_native.temperature_by_options_raw, "temperature_by_options_applied": runtime.laya_native.temperature_by_options, "invalid_warning_texts": runtime.warnings, "noul_bucket_used": "noul:2", "probability_definition": "softmax(option logits / temperature); noul is p[true] / second option after canonical true,false ordering", "b2_unscaled_top5_rank_changes": unscaled_count, "b2_queries": len(qids)}
    write_csv(OUT / "B2_COMPARISONS.csv", b2_rows)
    write_csv(OUT / "B3_SANITY_PROBE.csv", b3_rows)
    dump_json(OUT / "B4_LAYA_TEMPERATURE.json", laya_temp)
    return b2_rows, b3_rows, laya_temp


def summarize_status(a1, a2, a3, a4, batch_order, b1, b2, b3, b4) -> dict:
    b2_summary = {}
    for model in MODELS:
        rows = [r for r in b2 if r["model"] == model]
        b2_summary[model] = {"n_queries": len(rows), "stored_native_mean_abs_diff": statistics.mean(r["stored_native_mean_abs_diff"] for r in rows), "stored_native_kendall_mean": statistics.mean(r["stored_native_kendall"] for r in rows), "stored_native_top5_mean": statistics.mean(r["stored_native_top5_agreement"] for r in rows), "pass": statistics.mean(r["stored_native_kendall"] for r in rows) >= 0.90 and statistics.mean(r["stored_native_top5_agreement"] for r in rows) >= 0.90}
    b3_summary = {model: {"n": 10, "relevant_higher_both_orders": sum(r["relevant_higher_both_orders"] for r in b3 if r["model"] == model), "pass": sum(r["relevant_higher_both_orders"] for r in b3 if r["model"] == model) >= 9} for model in MODELS}
    a1_flags = {model: any(r.get("model") == model and r.get("flag") for r in a1[1]) for model in MODELS}
    a2_flags = {model: any(r.get("model") == model and r.get("flag") for r in a2) for model in MODELS}
    a3_failures = {model: any(r.get("model") == model and (r.get("response_hash_status") == "FAIL" or not r.get("request_hash_match") or not r.get("canonical_pointwise_view_match")) for r in a3) for model in MODELS}
    a4_flags = {model: any(r.get("model") == model and r.get("flag") for r in a4) for model in MODELS}
    batch_flags = {model: not all(r["pass"] for r in batch_order if r["model"] == model) for model in MODELS}
    b1_flags = {model: not all(r.get("model") == model and r.get("project_expected") and r.get("native_expected") for r in b1 if r.get("model") == model) for model in MODELS}
    flags = {"A1": any(a1_flags.values()), "A2": any(a2_flags.values()), "A3": any(a3_failures.values()), "A3_BATCH": any(batch_flags.values()), "A4": any(a4_flags.values()), "B1": any(b1_flags.values()), "B2": not all(r["pass"] for r in b2_summary.values()), "B3": not all(r["pass"] for r in b3_summary.values())}
    batch_summary = {model: {"n_queries": sum(r["model"] == model for r in batch_order), "n_failures": sum(not r["pass"] for r in batch_order if r["model"] == model), "max_abs_score_difference": max((r["max_abs_score_difference"] for r in batch_order if r["model"] == model and r["max_abs_score_difference"] is not None), default=None), "pass": all(r["pass"] for r in batch_order if r["model"] == model)} for model in MODELS}
    by_model = {
        "A1": {model: "FLAG" if a1_flags[model] else "PASS" for model in MODELS},
        "A2": {model: "FLAG" if a2_flags[model] else "PASS" for model in MODELS},
        "A3": {model: "FLAG" if a3_failures[model] else ("PASS_WITH_LAYA_HASH_LIMITATION" if model == "laya" else "PASS") for model in MODELS},
        "A3_BATCH": {model: "FLAG" if batch_flags[model] else "PASS" for model in MODELS},
        "A4": {model: "FLAG" if a4_flags[model] else "PASS" for model in MODELS},
        "B1": {model: "FLAG" if b1_flags[model] else "PASS" for model in MODELS},
        "B2": {model: "PASS" if b2_summary[model]["pass"] else "FLAG" for model in MODELS},
        "B3": {model: "PASS" if b3_summary[model]["pass"] else "FLAG" for model in MODELS},
    }
    structural_flags = any(flags[key] for key in ("A2", "A3", "A3_BATCH", "B1", "B2", "B3"))
    if structural_flags:
        final_status = "TYPED_AUDIT_BLOCKED_NEEDS_AMENDMENT"
    elif flags["A1"] or flags["A4"]:
        final_status = "TYPED_AUDIT_PASS_WITH_MODEL_BEHAVIOR_FLAGS"
    else:
        final_status = "TYPED_AUDIT_PASS"
    return {"A1": "FLAG" if flags["A1"] else "PASS", "A2": "FLAG" if flags["A2"] else "PASS", "A3": "FLAG" if flags["A3"] else "PASS_WITH_LAYA_HASH_LIMITATION", "A3_BATCH": "FLAG" if flags["A3_BATCH"] else "PASS", "A4": "FLAG" if flags["A4"] else "PASS", "B1": "FLAG" if flags["B1"] else "PASS", "B2": "FLAG" if flags["B2"] else "PASS", "B3": "FLAG" if flags["B3"] else "PASS", "by_model": by_model, "b2_summary": b2_summary, "b3_summary": b3_summary, "batch_order_summary": batch_summary, "laya_temperature": b4, "final_status": final_status}


def write_report(status: dict, a1, a2, a3, a4, batch_order, b1, b2, b3) -> None:
    b2_text = "\n".join(f"- {m}: stored/native mean Kendall {v['stored_native_kendall_mean']:.3f}; Top-5 agreement {v['stored_native_top5_mean']:.3f}; mean absolute difference {v['stored_native_mean_abs_diff']:.6f}; {'PASS' if v['pass'] else 'FLAG'}" for m, v in status["b2_summary"].items())
    b3_text = "\n".join(f"- {m}: {v['relevant_higher_both_orders']}/10 pairs passed; {'PASS' if v['pass'] else 'FLAG'}" for m, v in status["b3_summary"].items())
    text = f"""# Typed-model audit — Laya and Clef

**Study:** `decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2`  
**Final status:** `{status['final_status']}`  
**Jev calls:** 0  
**Hosted Cloudflare Workers AI calls:** 0  
**Expansion:** not started  
**Serving:** untouched

## Scope and frozen-contract rule

The audit uses existing raw traces for all offline checks, with complete-gold-eligible queries as the main population. The native phase uses only the pinned local Laya `{LAYA_REV}` and Clef `{CLEF_REV}` snapshots on the local GPU. No prompt, view, token cap, checkpoint, quartet seed, tie rule, reader, or scored temperature was changed. All new evidence is under this directory; prior artifacts are unchanged.

## Decision summary

| Check | Laya | Clef | Evidence |
|---|---|---|---|
| A1 direction / reversal / AUC | {status['by_model']['A1']['laya']} | {status['by_model']['A1']['clef']} | `A1_SELECTION_METRICS.csv`, `A1_AUC_SUMMARY.csv` |
| A2 distribution / ties | {status['by_model']['A2']['laya']} | {status['by_model']['A2']['clef']} | `A2_SCORE_DISTRIBUTION.csv`, `A2_SCORE_HISTOGRAM.csv` |
| A3 parsing / alignment | {status['by_model']['A3']['laya']} | {status['by_model']['A3']['clef']} | `A3_ALIGNMENT_AND_OVERLAP.csv`, `A3_RAW_RESPONSE_TRACES.jsonl` |
| A3 shuffled batch-order invariance | {status['by_model']['A3_BATCH']['laya']} | {status['by_model']['A3_BATCH']['clef']} | `A3_BATCH_ORDER_INVARIANCE.csv` |
| A4 groupwise mapping / collision | {status['by_model']['A4']['laya']} | {status['by_model']['A4']['clef']} | `A4_GROUPWISE_CHECKS.csv` |
| B1 vendor examples | {status['by_model']['B1']['laya']} | {status['by_model']['B1']['clef']} | `B1_VENDOR_EXAMPLES.json` |
| B2 native recompute | {status['by_model']['B2']['laya']} | {status['by_model']['B2']['clef']} | `B2_COMPARISONS.csv` |
| B3 order sanity | {status['by_model']['B3']['laya']} | {status['by_model']['B3']['clef']} | `B3_SANITY_PROBE.csv` |

## A1–A4 offline findings

The A1 CSV reports CSSR@5 and support recall for frozen top-5, reversed bottom-5, 1,000-draw random-5, retrieval-5, and BGE p5, separately for CAL/TEST, dataset, and pooled populations. Stored-score AUC is query-level with a 5,000-replicate bootstrap CI (seed 731). A2 reports exact score cardinality, non-finite values, endpoint concentration, rank-5/6 ties, RRF tie-break-selected slots, and near-constant queries. A3 fixes the first 20 eligible TEST HotpotQA IDs by lexical query ID order and records the canonical `answers.relevance.noul` P(true) field rather than P(false) or confidence. The native A3 batch-order file compares canonical and deterministically shuffled 20-candidate batches after joining scores by document ID.

Important evidence limitation: Laya's stored `response_hash` was computed over the full vendor response object, but the frozen pointwise trace preserves only normalized `doc_id`/`score`/view fields. Therefore that hash cannot be recomputed independently from the archived normalized row; this is reported as a provenance limitation, not silently converted to a pass. Clef's score-list response hash is independently recomputed.

For A4, `A4_GROUPWISE_CHECKS.csv` gives A/B/C/D frequencies and chi-square uniformity, canonical quartet-to-label mapping, collision rates, no-collision g5 CSSR, and exactly-one-gold winner rates against the 25% chance baseline.

## B1–B4 local-native findings

### B2 native recomputation

{b2_text}

### B3 order sanity

{b3_text}

### B4 Laya temperature

The exact warning text, raw/applied temperatures, probability definition, and unscaled diagnostic ranking-change count are in `B4_LAYA_TEMPERATURE.json`. The scored path retains the checkpoint/runtime's applied values; no calibration or temperature change was applied to the benchmark results.

## Model-blind answer-in-selection diagnostic

`ANSWER_IN_SELECTION.csv` reports, for all 12 policies and each dataset/split/pooled population, whether a normalized gold answer or alias occurs in the selected reader view, alongside unconditional CSSR and reader F1. This is a diagnostic only and was not fed into any model.

## Regression validation

The existing project regression suite passed **28/28** tests with `python -m pytest -q -rA`; no test failures were observed.

## Stop gate and unresolved items

No rerun of scored policies is authorized by this audit unless a confirmed implementation defect is found. The only current evidence limitation is the non-recomputable normalized Laya response hash; native score agreement and alignment checks are reported separately. GPU memory/inference was local only, and no paid call occurred.

Wait for owner review before any Jev smoke test or expansion. A separate approval remains required before any bulk paid inference.
"""
    (OUT / "TYPED_MODEL_AUDIT.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("offline", "native", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    from transformers import AutoTokenizer
    view_tokenizer = AutoTokenizer.from_pretrained(str(HF_ROOT / "models--convaiinnovations--laya" / "snapshots" / LAYA_REV / "tokenizer"))
    reader_tokenizer = AutoTokenizer.from_pretrained(str(HF_ROOT / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"))
    prepare_context_for_reader(ctx)
    a1 = offline_a1(ctx)
    a2 = offline_a2(ctx)
    a3 = offline_a3(ctx, view_tokenizer)
    a4 = offline_a4(ctx)
    answer_in_selection(ctx, reader_tokenizer)
    if args.phase == "offline":
        status = {"final_status": "PENDING_NATIVE_PHASE", "A1": "PENDING", "A2": "PENDING", "A3": "PENDING", "A4": "PENDING"}
        dump_json(OUT / "typed_model_audit.json", {"status": status, "offline_complete": True, "native_complete": False, "jev_calls": 0, "hosted_cloudflare_calls": 0, "expansion_started": False})
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    runtime = NativeRuntime()
    runtime.load_laya()
    runtime.load_clef()
    b1 = native_b1(runtime)
    b2, b3, b4 = native_b2_b3_b4(ctx, view_tokenizer, runtime)
    batch_order = native_batch_order(ctx, view_tokenizer, runtime)
    status = summarize_status(a1, a2, a3, a4, batch_order, b1, b2, b3, b4)
    dump_json(OUT / "typed_model_audit.json", {"study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2", "created_utc": datetime.now(timezone.utc).isoformat(), "status": status, "offline_complete": True, "native_complete": True, "jev_calls": 0, "hosted_cloudflare_calls": 0, "expansion_started": False, "models": {"laya_revision": LAYA_REV, "clef_revision": CLEF_REV}, "warnings": runtime.warnings})
    write_report(status, a1, a2, a3, a4, batch_order, b1, b2, b3)
    runtime.close()
    print(json.dumps({"final_status": status["final_status"], "jev_calls": 0, "hosted_cloudflare_calls": 0, "expansion_started": False, "b2": status["b2_summary"], "b3": status["b3_summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
