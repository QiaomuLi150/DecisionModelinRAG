#!/usr/bin/env python3
"""Offline retrieval/evaluation integrity audit for the frozen non-Jev study."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decisionrag.selection import deterministic_quartets, make_views  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(n: int, d: int) -> str:
    return "NOT_MEASURED" if not d else f"{100.0 * n / d:.2f}%"


def table(headers: list[str], body: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in body)
    return "\n".join(out)


def main() -> int:
    questions = rows(ROOT / "data/manifests/questions.jsonl")
    slates = {row["query_id"]: row for row in rows(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")}
    corpus = {row["doc_id"]: row for row in rows(ROOT / "data/corpus/passages.jsonl")}
    qmap = {row["query_id"]: row for row in questions}
    selections = rows(ROOT / "data/results/m3/selection_results.jsonl")
    selection_map = {(row["policy_id"], row["query_id"]): row for row in selections}
    reader_rows = {(row["policy_id"], row["query_id"]): row for row in rows(ROOT / "data/results/m4/reader_results.jsonl")}

    coverage = []
    for question in questions:
        gold = set(question["gold_doc_ids"])
        top20 = set(slates[question["query_id"]]["candidate_ids"])
        top5 = set(slates[question["query_id"]]["candidate_ids"][:5])
        coverage.append({"query_id": question["query_id"], "dataset": question["dataset"], "split": question["split"], "gold_count": len(gold), "any_gold_top20": bool(gold & top20), "complete_gold_top20": gold <= top20, "any_gold_retrieval5": bool(gold & top5), "complete_gold_retrieval5": gold <= top5})

    out = ROOT / "data/results/audit/quality"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gold_coverage_by_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in coverage:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    coverage_body = []
    for split in ("cal", "test"):
        for dataset in ("hotpot", "musique", "wiki2"):
            cohort = [row for row in coverage if row["split"] == split and row["dataset"] == dataset]
            coverage_body.append([split.upper(), dataset, len(cohort), sum(row["any_gold_top20"] for row in cohort), pct(sum(row["any_gold_top20"] for row in cohort), len(cohort)), sum(row["complete_gold_top20"] for row in cohort), pct(sum(row["complete_gold_top20"] for row in cohort), len(cohort)), sum(row["any_gold_retrieval5"] for row in cohort), pct(sum(row["any_gold_retrieval5"] for row in cohort), len(cohort)), sum(row["complete_gold_retrieval5"] for row in cohort), pct(sum(row["complete_gold_retrieval5"] for row in cohort), len(cohort))])
        cohort = [row for row in coverage if row["split"] == split]
        coverage_body.append([split.upper(), "ALL", len(cohort), sum(row["any_gold_top20"] for row in cohort), pct(sum(row["any_gold_top20"] for row in cohort), len(cohort)), sum(row["complete_gold_top20"] for row in cohort), pct(sum(row["complete_gold_top20"] for row in cohort), len(cohort)), sum(row["any_gold_retrieval5"] for row in cohort), pct(sum(row["any_gold_retrieval5"] for row in cohort), len(cohort)), sum(row["complete_gold_retrieval5"] for row in cohort), pct(sum(row["complete_gold_retrieval5"] for row in cohort), len(cohort))])

    # Support visibility: the manifest preserves passage-level gold IDs only;
    # sentence offsets are not available in the frozen inference manifest, so
    # sentence-level survival is explicitly NOT_MEASURED rather than inferred.
    from transformers import AutoTokenizer
    view_tokenizer = AutoTokenizer.from_pretrained(str(Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148/tokenizer")))
    selector_visibility = []
    for question in questions:
        slate = slates[question["query_id"]]
        candidates = []
        for rank, doc_id in enumerate(slate["candidate_ids"], 1):
            source = corpus[doc_id]
            candidates.append({"doc_id": doc_id, "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": rank})
        views = {view.doc_id: view for view in [make_views(question["question"], candidate, view_tokenizer) for candidate in candidates]}
        for doc_id in question["gold_doc_ids"]:
            if doc_id in views:
                source = corpus[doc_id]
                full_tokens = len(view_tokenizer.encode(f"{source['title']}\n{source['full_text']}", add_special_tokens=False))
                budget = 128 + len(view_tokenizer.encode(str(source["title"]), add_special_tokens=False))
                selector_visibility.append({"query_id": question["query_id"], "split": question["split"], "dataset": question["dataset"], "doc_id": doc_id, "gold_in_top20": True, "selector_view_truncated": full_tokens > budget})
            else:
                selector_visibility.append({"query_id": question["query_id"], "split": question["split"], "dataset": question["dataset"], "doc_id": doc_id, "gold_in_top20": False, "selector_view_truncated": None})
    with (out / "support_visibility_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in selector_visibility:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    visibility_body = []
    for split in ("cal", "test"):
        cohort = [row for row in selector_visibility if row["split"] == split]
        present = [row for row in cohort if row["gold_in_top20"]]
        visibility_body.append([split.upper(), len(cohort), len(present), sum(row["selector_view_truncated"] for row in present), pct(sum(row["selector_view_truncated"] for row in present), len(present)), "NOT_MEASURED", "sentence offsets absent from frozen manifest"])
    reader_visibility_body = []
    for policy in sorted({row["policy_id"] for row in reader_rows.values()}):
        for split in ("cal", "test"):
            selected_gold = 0
            selected_gold_truncated = 0
            for (policy_id, query_id), reader in reader_rows.items():
                question = qmap[query_id]
                if policy_id != policy or question["split"] != split:
                    continue
                gold = set(question["gold_doc_ids"])
                for passage in reader.get("view_info", {}).get("passages", []):
                    if passage.get("doc_id") in gold:
                        selected_gold += 1
                        selected_gold_truncated += int(bool(passage.get("truncated")))
            reader_visibility_body.append([policy, split.upper(), selected_gold, selected_gold_truncated, pct(selected_gold_truncated, selected_gold), "NOT_MEASURED", "sentence offsets absent from frozen manifest"])

    collision_body = []
    collision_rows = []
    for question in questions:
        slate = slates[question["query_id"]]
        candidates = []
        for rank, doc_id in enumerate(slate["candidate_ids"], 1):
            source = corpus[doc_id]
            candidates.append({"doc_id": doc_id, "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": rank})
        gold = set(question["gold_doc_ids"])
        groups = deterministic_quartets([dict(make_views(question["question"], candidate, view_tokenizer).__dict__) for candidate in candidates], question["query_id"], 20261011)
        collisions = [sum(view["doc_id"] in gold for view in group) for group in groups]
        collision_rows.append({"query_id": question["query_id"], "dataset": question["dataset"], "split": question["split"], "gold_count": len(gold), "groups_with_two_or_more_gold": sum(value >= 2 for value in collisions), "gold_docs_in_same_quartet": max(collisions) >= 2, "gold_counts_by_group": collisions})
    for split in ("cal", "test"):
        for dataset in ("hotpot", "musique", "wiki2"):
            cohort = [row for row in collision_rows if row["split"] == split and row["dataset"] == dataset]
            collision_body.append([split.upper(), dataset, len(cohort), sum(row["gold_docs_in_same_quartet"] for row in cohort), sum(row["groups_with_two_or_more_gold"] for row in cohort), sum(sum(row["gold_counts_by_group"]) for row in cohort), 5 * len(cohort)])
        cohort = [row for row in collision_rows if row["split"] == split]
        collision_body.append([split.upper(), "ALL", len(cohort), sum(row["gold_docs_in_same_quartet"] for row in cohort), sum(row["groups_with_two_or_more_gold"] for row in cohort), sum(sum(row["gold_counts_by_group"]) for row in cohort), 5 * len(cohort)])
    with (out / "groupwise_collision_by_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in collision_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rescue_rows = []
    rescue_body = []
    for model in ("bge", "qwen", "laya", "clef"):
        for split in ("cal", "test"):
            cohort = []
            for question in questions:
                if question["split"] != split:
                    continue
                p5 = selection_map[(f"{model}_p5", question["query_id"])]
                rescue = selection_map[(f"{model}_rescue5", question["query_id"])]
                gold = set(question["gold_doc_ids"])
                p5_complete = gold <= set(p5["selected_doc_ids"])
                rescue_complete = gold <= set(rescue["selected_doc_ids"])
                p5_recall = len(gold & set(p5["selected_doc_ids"])) / len(gold)
                rescue_recall = len(gold & set(rescue["selected_doc_ids"])) / len(gold)
                row = {"query_id": question["query_id"], "dataset": question["dataset"], "split": split, "model": model, "p5_complete": p5_complete, "rescue_complete": rescue_complete, "outcome": "benefit" if rescue_complete and not p5_complete else "harm" if p5_complete and not rescue_complete else "unchanged", "support_effect": "improved" if rescue_recall > p5_recall else "degraded" if rescue_recall < p5_recall else "unchanged", "p5_support_recall": p5_recall, "rescue_support_recall": rescue_recall, "swapped": bool(rescue.get("rescue", {}).get("swapped"))}
                rescue_rows.append(row)
                cohort.append(row)
            rescue_body.append([model, split.upper(), len(cohort), sum(row["outcome"] == "benefit" for row in cohort), sum(row["outcome"] == "harm" for row in cohort), sum(row["outcome"] == "unchanged" for row in cohort), sum(row["support_effect"] == "improved" for row in cohort), sum(row["support_effect"] == "degraded" for row in cohort), sum(row["support_effect"] == "unchanged" for row in cohort), sum(row["swapped"] for row in cohort)])
    with (out / "rescue_outcomes_by_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in rescue_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    (out / "GOLD_COVERAGE_RECONCILIATION.md").write_text("""# Gold coverage reconciliation\n\nThe frozen report's M2 `gold presence` is reported here as **any-gold Top-20 coverage**. M4's conditional CSSR denominator is **complete-gold Top-20 coverage** (`gold_doc_ids ⊆ Top-20`), not any-gold presence. The tables below recompute both over the same frozen CAL/TEST query populations. `retrieval5` denotes the first five entries of the frozen Top-20 slate.\n\n""" + table(["Split", "Dataset", "N", "Any-gold Top-20 n", "Any-gold rate", "Complete-gold Top-20 n", "Complete-gold rate", "Any-gold Top-5 n", "Any-gold Top-5 rate", "Complete-gold Top-5 n", "Complete-gold Top-5 rate"], coverage_body) + "\n\nThe complete-gold TEST Top-20 denominator used by corrected M4 remains 291/600. Any-gold and complete-gold are intentionally not interchangeable.\n", encoding="utf-8")
    (out / "GROUPWISE_COLLISION_AND_RESCUE_AUDIT.md").write_text("""# Groupwise collision and rescue audit\n\n## Complete-gold collisions\n\nA collision is a deterministic four-way tournament partition containing at least two annotated gold passages. Counts are reported per query and per quartet; these are structural diagnostics, not model failures.\n\n""" + table(["Split", "Dataset", "N queries", "Queries with collision", "Collision quartets", "Gold assignments", "Quartet slots"], collision_body) + "\n\n## Rescue outcomes\n\nBenefit means rescue changes incomplete Top-5 support to complete support; harm means the reverse; unchanged covers both unchanged complete and unchanged incomplete outcomes. Support-effect counts use graded passage recall.\n\n""" + table(["Model", "Split", "N", "Benefit", "Harm", "Unchanged", "Recall improved", "Recall degraded", "Recall unchanged", "Swapped"], rescue_body) + "\n", encoding="utf-8")
    (out / "support_visibility_summary.json").write_text(json.dumps({"selector": selector_visibility, "reader": reader_visibility_body}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "support_visibility_summary.md").write_text("""# Support visibility and truncation audit\n\nThe frozen question manifest has passage-level `gold_doc_ids` but no supporting-sentence offsets. Therefore passage-level inclusion and tokenizer truncation are measured; sentence-level survival is `NOT_MEASURED`, not assumed.\n\n## Selector views\n\n""" + table(["Split", "Gold passage instances", "In Top-20", "Selector-truncated", "Truncated rate", "Sentence survival", "Note"], visibility_body) + "\n\n## Reader views\n\n" + table(["Policy", "Split", "Selected gold passage instances", "Reader-truncated", "Truncated rate", "Sentence survival", "Note"], reader_visibility_body) + "\n", encoding="utf-8")
    print(json.dumps({"coverage_rows": len(coverage), "collision_rows": len(collision_rows), "rescue_rows": len(rescue_rows), "test_complete_gold_top20": sum(row["complete_gold_top20"] for row in coverage if row["split"] == "test"), "test_any_gold_top20": sum(row["any_gold_top20"] for row in coverage if row["split"] == "test")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
