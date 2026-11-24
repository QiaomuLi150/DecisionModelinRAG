import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_m1_counts_and_mapping_integrity():
    manifest = json.loads((ROOT / "data/manifests/m1_data_manifest.json").read_text())
    assert manifest["mapping"]["selected"] == 750
    assert manifest["mapping"]["invalid"] == 0
    assert manifest["mapping"]["counts"] == {
        "hotpot": {"cal": 50, "test": 200},
        "musique": {"cal": 50, "test": 200},
        "wiki2": {"cal": 50, "test": 200},
    }
    rows = load_jsonl(ROOT / "data/manifests/questions.jsonl")
    assert len(rows) == 750
    assert all(row["gold_mapping_valid"] and row["gold_doc_ids"] for row in rows)


def test_m1_split_and_document_isolation():
    rows = load_jsonl(ROOT / "data/manifests/questions.jsonl")
    assert len({row["query_id"] for row in rows}) == len(rows)
    assert not ({row["query_id"] for row in rows if row["split"] == "cal"} & {row["query_id"] for row in rows if row["split"] == "test"})
    # Every gold ID must be a passage in the record's released context.
    for row in rows:
        assert set(row["gold_doc_ids"]).issubset(set(row["context_doc_ids"]))


def test_m1_corpus_contains_no_gold_or_answer_fields():
    passages = load_jsonl(ROOT / "data/corpus/passages.jsonl")
    assert len(passages) == 144650
    forbidden = {"answer", "gold_doc_ids", "supporting_facts", "is_supporting"}
    assert all(not forbidden.intersection(row) for row in passages)
    assert all(row["full_text"].strip() for row in passages)


def test_m1_gold_audit_is_stratified_20_per_dataset():
    audit = load_jsonl(ROOT / "data/manifests/gold_mapping_audit.jsonl")
    assert len(audit) == 60
    assert {dataset: sum(row["dataset"] == dataset for row in audit) for dataset in ("hotpot", "musique", "wiki2")} == {"hotpot": 20, "musique": 20, "wiki2": 20}
    assert all(row["gold_mapping_valid"] for row in audit)
