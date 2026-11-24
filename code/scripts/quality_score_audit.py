#!/usr/bin/env python3
"""Frozen 20-query semantic correctness audit for the non-Jev experiment.

This script deliberately does not write any frozen M3/M4/M5 artifact.  It
replays the stored BGE path (including the suspected shared-tokenizer path),
replays BGE with the checkpoint's native tokenizer, and queries the local Qwen
reranker in original and shuffled candidate order.  The shared semantic views
remain produced by the frozen Laya tokenizer in both BGE replays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decisionrag.selection import make_views, pointwise_pair, rank_indices, sha256_json  # noqa: E402

HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
BGE_REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
QWEN_REV = "4bd860ac4f15ad1897a214615cccc700f8f71818"
LAYA_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"
QWEN_URL = "http://service.example.invalid/v1/rerank"
QWEN_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
SEED = 20261011


def snapshot(owner: str, repo: str, revision: str) -> str:
    return str(HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / revision)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def pairwise_auc(gold_ids: set[str], scores: dict[str, float]) -> tuple[float | None, int, int]:
    positives = [(doc_id, scores[doc_id]) for doc_id in scores if doc_id in gold_ids]
    negatives = [(doc_id, scores[doc_id]) for doc_id in scores if doc_id not in gold_ids]
    if not positives or not negatives:
        return None, len(positives), len(negatives)
    wins = 0.0
    for _, positive in positives:
        for _, negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives)), len(positives), len(negatives)


def kendall(order_a: list[str], order_b: list[str]) -> float:
    rank_b = {doc_id: index for index, doc_id in enumerate(order_b)}
    concordant = discordant = 0
    for left in range(len(order_a)):
        for right in range(left + 1, len(order_a)):
            delta = rank_b[order_a[left]] - rank_b[order_a[right]]
            if delta == 0:
                continue
            if delta < 0:
                concordant += 1
            else:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 1.0


def deterministic_sample(question_rows: list[dict]) -> list[dict]:
    by_dataset = {dataset: sorted((row for row in question_rows if row["dataset"] == dataset), key=lambda row: row["query_id"]) for dataset in ("hotpot", "musique", "wiki2")}
    # Seven/seven/six gives all three frozen datasets representation without
    # selecting by any model score or gold-presence outcome.
    return by_dataset["hotpot"][:7] + by_dataset["musique"][:7] + by_dataset["wiki2"][:6]


def build_items(question_rows: list[dict], slate_rows: list[dict], corpus_rows: list[dict], selected: list[dict]) -> list[dict]:
    qmap = {row["query_id"]: row for row in question_rows}
    smap = {row["query_id"]: row for row in slate_rows}
    corpus = {row["doc_id"]: row for row in corpus_rows}
    output = []
    for question in selected:
        slate = smap[question["query_id"]]
        candidates = []
        for original_rank, doc_id in enumerate(slate["candidate_ids"], start=1):
            source = corpus[doc_id]
            candidates.append({"doc_id": doc_id, "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": original_rank})
        output.append({"query_id": question["query_id"], "dataset": question["dataset"], "question": question["question"], "gold_doc_ids": question["gold_doc_ids"], "candidates": candidates})
    return output


def load_stored(model_key: str, path: Path | None = None) -> dict[str, dict]:
    path = path or (ROOT / "data/results/m3" / f"{model_key}_all_pointwise.jsonl")
    return {row["query_id"]: row for row in rows(path)}


def score_map(row: dict) -> dict[str, float]:
    return {point["doc_id"]: float(point["score"]) for point in row["pointwise"]}


def compare_stored(row: dict, views: list, scores: list[float]) -> dict:
    stored = score_map(row)
    replay = {view.doc_id: float(score) for view, score in zip(views, scores)}
    differences = [abs(stored[doc_id] - replay[doc_id]) for doc_id in stored]
    stored_order = [view["doc_id"] for view in rank_indices([dict(view.__dict__) for view in views], stored)]
    replay_order = [view["doc_id"] for view in rank_indices([dict(view.__dict__) for view in views], replay)]
    return {
        "max_abs_diff": max(differences) if differences else None,
        "mean_abs_diff": sum(differences) / len(differences) if differences else None,
        "exact_score_match": all(value == 0.0 for value in differences),
        "kendall_tau": kendall(stored_order, replay_order),
        "stored_ranking": stored_order,
        "replay_ranking": replay_order,
    }


def load_bge(torch, use_native_tokenizer: bool):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_path = snapshot("BAAI", "bge-reranker-v2-m3", BGE_REV)
    view_tokenizer = AutoTokenizer.from_pretrained(snapshot("convaiinnovations", "laya", LAYA_REV) + "/tokenizer")
    model_tokenizer = AutoTokenizer.from_pretrained(model_path) if use_native_tokenizer else view_tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda").eval()
    return model, view_tokenizer, model_tokenizer


def replay_bge(items: list[dict], use_native_tokenizer: bool) -> tuple[dict[str, dict], dict[str, list]]:
    import torch
    model, view_tokenizer, model_tokenizer = load_bge(torch, use_native_tokenizer)
    results = {}
    views_by_query = {}
    for item in items:
        views = [make_views(item["question"], candidate, view_tokenizer) for candidate in item["candidates"]]
        pairs = [pointwise_pair(view) for view in views]
        encoded = model_tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        torch.cuda.synchronize()
        with torch.inference_mode():
            values = model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
        torch.cuda.synchronize()
        results[item["query_id"]] = {"scores": [float(value) for value in values], "input_tokens": int(encoded["attention_mask"].sum().item()), "tokenizer_mode": "native_bge" if use_native_tokenizer else "shared_laya"}
        views_by_query[item["query_id"]] = views
    del model
    torch.cuda.empty_cache()
    return results, views_by_query


def qwen_scores(session: requests.Session, query: str, views: list, order: list[int]) -> tuple[dict[str, float], dict]:
    ordered_views = [views[index] for index in order]
    payload = {"model": QWEN_MODEL, "query": query, "documents": [pointwise_pair(view)[1] for view in ordered_views]}
    response = session.post(QWEN_URL, json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    if len(body.get("results", [])) != len(views):
        raise RuntimeError(f"expected {len(views)} Qwen scores, got {len(body.get('results', []))}")
    by_doc = {}
    for result in body["results"]:
        index = int(result["index"])
        if index < 0 or index >= len(ordered_views):
            raise RuntimeError(f"Qwen returned invalid index {index}")
        doc_id = ordered_views[index].doc_id
        if doc_id in by_doc:
            raise RuntimeError(f"Qwen returned duplicate mapped doc_id {doc_id}")
        by_doc[doc_id] = float(result["relevance_score"])
    return by_doc, {"payload": payload, "payload_hash": sha256_json(payload), "body": body, "order": order}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/results/audit/quality")
    parser.add_argument("--baseline-bge", default="data/results/audit/original_non_jev/m3/bge_all_pointwise.jsonl")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    question_rows = rows(ROOT / "data/manifests/questions.jsonl")
    selected = deterministic_sample(question_rows)
    slate_rows = rows(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")
    corpus_rows = rows(ROOT / "data/corpus/passages.jsonl")
    items = build_items(question_rows, slate_rows, corpus_rows, selected)
    write_jsonl(out / "audit_queries.jsonl", [{"query_id": item["query_id"], "dataset": item["dataset"], "selection_rule": "sorted_frozen_dataset_prefix_7_7_6", "seed": SEED} for item in items])

    stored_bge = load_stored("bge", ROOT / args.baseline_bge)
    corrected_bge = load_stored("bge")
    stored_qwen = load_stored("qwen")
    stored_laya = load_stored("laya")
    stored_clef = load_stored("clef")
    native, native_views = replay_bge(items, use_native_tokenizer=True)
    shared, shared_views = replay_bge(items, use_native_tokenizer=False)
    if set(native_views) != set(shared_views):
        raise RuntimeError("BGE replay query sets differ")

    session = requests.Session()
    qwen_audit = {}
    rng_seed = SEED
    for item_index, item in enumerate(items):
        query_id = item["query_id"]
        views = shared_views[query_id]
        original_order = list(range(len(views)))
        shuffled_order = list(reversed(original_order)) if item_index % 2 == 0 else original_order[::2] + original_order[1::2]
        original_scores, original_meta = qwen_scores(session, item["question"], views, original_order)
        shuffled_scores, shuffled_meta = qwen_scores(session, item["question"], views, shuffled_order)
        qwen_audit[query_id] = {"original": original_scores, "shuffled": shuffled_scores, "original_meta": original_meta, "shuffled_meta": shuffled_meta, "max_abs_diff": max(abs(original_scores[doc_id] - shuffled_scores[doc_id]) for doc_id in original_scores)}

    score_rows = []
    semantic_rows = []
    representative = None
    for item in items:
        query_id = item["query_id"]
        views = shared_views[query_id]
        native_cmp = compare_stored(stored_bge[query_id], views, native[query_id]["scores"])
        shared_cmp = compare_stored(stored_bge[query_id], views, shared[query_id]["scores"])
        corrected_cmp = compare_stored(corrected_bge[query_id], views, native[query_id]["scores"])
        stored_q = score_map(stored_qwen[query_id])
        qaudit = qwen_audit[query_id]
        q_order = [view["doc_id"] for view in rank_indices([dict(view.__dict__) for view in views], stored_q)]
        q_replay_order = [view["doc_id"] for view in rank_indices([dict(view.__dict__) for view in views], qaudit["original"])]
        q_shuffle_order = [view["doc_id"] for view in rank_indices([dict(view.__dict__) for view in views], qaudit["shuffled"])]
        qdiff = [abs(stored_q[doc_id] - qaudit["original"][doc_id]) for doc_id in stored_q]
        for model_name, values, ranking in [
            ("bge_original_stored", stored_bge[query_id], native_cmp["stored_ranking"]),
            ("bge_native_replay", {"pointwise": [{"doc_id": v.doc_id, "score": s} for v, s in zip(views, native[query_id]["scores"])]}, native_cmp["replay_ranking"]),
            ("bge_shared_tokenizer_replay", {"pointwise": [{"doc_id": v.doc_id, "score": s} for v, s in zip(views, shared[query_id]["scores"])]}, shared_cmp["replay_ranking"]),
            ("bge_corrected_stored", corrected_bge[query_id], corrected_cmp["stored_ranking"]),
            ("qwen_stored", stored_qwen[query_id], q_order),
            ("qwen_replay", {"pointwise": [{"doc_id": v.doc_id, "score": qaudit["original"][v.doc_id]} for v in views]}, q_replay_order),
            ("qwen_shuffled_replay", {"pointwise": [{"doc_id": v.doc_id, "score": qaudit["shuffled"][v.doc_id]} for v in views]}, q_shuffle_order),
        ]:
            scores = score_map(values)
            auc, positive_count, negative_count = pairwise_auc(set(item["gold_doc_ids"]), scores)
            score_rows.append({"query_id": query_id, "dataset": item["dataset"], "model": model_name, "gold_count_in_slate": positive_count, "negative_count": negative_count, "gold_negative_auc": auc, "ranking": ranking})
        semantic_rows.append({"query_id": query_id, "dataset": item["dataset"], "bge_native": native_cmp, "bge_shared_tokenizer": shared_cmp, "bge_corrected": corrected_cmp, "bge_original_vs_corrected_kendall": kendall(native_cmp["stored_ranking"], corrected_cmp["stored_ranking"]), "qwen_request_hash_matches_stored": qaudit["original_meta"]["payload_hash"] == stored_qwen[query_id]["request_hash"], "qwen_max_abs_diff_original_vs_stored": max(qdiff), "qwen_mean_abs_diff_original_vs_stored": sum(qdiff) / len(qdiff), "qwen_kendall_stored_vs_replay": kendall(q_order, q_replay_order), "qwen_kendall_original_vs_shuffled": kendall(q_replay_order, q_shuffle_order), "qwen_batch_order_invariant": max(qaudit["max_abs_diff"], 0.0) <= 1e-5})
        if representative is None:
            representative = {"query_id": query_id, "question": item["question"], "gold_doc_ids": item["gold_doc_ids"], "views": [{"doc_id": view.doc_id, "question_view": view.question_view, "passage_view": view.passage_view, "pointwise_pair": pointwise_pair(view), "selection_text_sha256": view.selection_text_sha256, "original_rank": view.original_rank} for view in views], "original_stored_bge_scores": score_map(stored_bge[query_id]), "corrected_stored_bge_scores": score_map(corrected_bge[query_id]), "native_bge_scores": {view.doc_id: score for view, score in zip(views, native[query_id]["scores"])}, "shared_tokenizer_bge_scores": {view.doc_id: score for view, score in zip(views, shared[query_id]["scores"])}, "stored_qwen_scores": stored_q, "qwen_replay_scores": qaudit["original"], "qwen_shuffled_scores": qaudit["shuffled"]}

    write_jsonl(out / "M3_SCORE_ALIGNMENT_AUDIT.csv.jsonl", score_rows)
    alignment_lines = ["query_id,dataset,model,doc_id,baseline_score,replay_score,abs_diff,baseline_rank,replay_rank"]
    for item in items:
        query_id = item["query_id"]
        views = shared_views[query_id]
        view_ids = [view.doc_id for view in views]
        original = score_map(stored_bge[query_id])
        corrected = score_map(corrected_bge[query_id])
        native_scores = {view.doc_id: score for view, score in zip(views, native[query_id]["scores"])}
        shared_scores = {view.doc_id: score for view, score in zip(views, shared[query_id]["scores"])}
        qstored = score_map(stored_qwen[query_id])
        qreplay = qwen_audit[query_id]["original"]
        qshuffle = qwen_audit[query_id]["shuffled"]
        score_sets = [("bge_original_vs_native", original, native_scores), ("bge_original_vs_shared", original, shared_scores), ("bge_corrected_vs_native", corrected, native_scores), ("qwen_stored_vs_replay", qstored, qreplay), ("qwen_replay_vs_shuffled", qreplay, qshuffle)]
        for model_name, baseline, replay in score_sets:
            base_order = [row["doc_id"] for row in rank_indices([dict(view.__dict__) for view in views], baseline)]
            replay_order = [row["doc_id"] for row in rank_indices([dict(view.__dict__) for view in views], replay)]
            base_rank = {doc_id: index + 1 for index, doc_id in enumerate(base_order)}
            replay_rank = {doc_id: index + 1 for index, doc_id in enumerate(replay_order)}
            for doc_id in view_ids:
                alignment_lines.append(",".join([query_id, item["dataset"], model_name, doc_id, str(baseline[doc_id]), str(replay[doc_id]), str(abs(baseline[doc_id] - replay[doc_id])), str(base_rank[doc_id]), str(replay_rank[doc_id])]))
    (out / "M3_SCORE_ALIGNMENT_AUDIT.csv").write_text("\n".join(alignment_lines) + "\n", encoding="utf-8")
    write_jsonl(out / "M3_AUROC_BY_QUERY.jsonl", score_rows)
    (out / "semantic_alignment_summary.json").write_text(json.dumps(semantic_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "representative_inputs.json").write_text(json.dumps(representative, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "qwen_raw_replays.json").write_text(json.dumps(qwen_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Static common-view and score-join audit across all four M3 model traces.
    join_rows = []
    for item in items:
        query_id = item["query_id"]
        expected = [view.doc_id for view in shared_views[query_id]]
        expected_hashes = [view.selection_text_sha256 for view in shared_views[query_id]]
        record = {"query_id": query_id, "expected_doc_order": expected, "expected_selection_hashes": expected_hashes}
        for model_key, stored in (("bge", stored_bge), ("qwen", stored_qwen), ("laya", stored_laya), ("clef", stored_clef)):
            trace = stored[query_id]
            points = trace["pointwise"]
            ids = [point["doc_id"] for point in points]
            hashes = [point.get("selection_text_sha256") for point in points]
            record[f"{model_key}_doc_order_match"] = ids == expected
            record[f"{model_key}_hash_order_match"] = hashes == expected_hashes
            record[f"{model_key}_unique_doc_ids"] = len(ids) == len(set(ids)) == 20
        join_rows.append(record)
    write_jsonl(out / "shared_view_join_audit.jsonl", join_rows)
    (out / "audit_summary.json").write_text(json.dumps({"query_count": len(items), "datasets": {dataset: sum(item["dataset"] == dataset for item in items) for dataset in ("hotpot", "musique", "wiki2")}, "bge_revision": BGE_REV, "qwen_revision": QWEN_REV, "laya_view_tokenizer_revision": LAYA_REV, "baseline_bge": args.baseline_bge, "bge_native_vs_original_stored_exact_queries": sum(row["bge_native"]["exact_score_match"] for row in semantic_rows), "bge_shared_replay_vs_original_stored_exact_queries": sum(row["bge_shared_tokenizer"]["exact_score_match"] for row in semantic_rows), "bge_corrected_stored_vs_native_exact_queries": sum(row["bge_corrected"]["exact_score_match"] for row in semantic_rows), "qwen_original_vs_stored_queries_within_1e-5": sum(row["qwen_max_abs_diff_original_vs_stored"] <= 1e-5 for row in semantic_rows), "qwen_request_hash_matches": sum(row["qwen_request_hash_matches_stored"] for row in semantic_rows), "qwen_order_invariant_queries": sum(row["qwen_batch_order_invariant"] for row in semantic_rows), "all_shared_view_join_rows_valid": all(all(value for key, value in row.items() if key.endswith(("doc_order_match", "hash_order_match", "unique_doc_ids"))) for row in join_rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "query_count": len(items), "summary": json.loads((out / "audit_summary.json").read_text())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
