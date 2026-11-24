import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_m2_frozen_slates_are_complete_and_unique():
    questions = rows(ROOT / "data/manifests/questions.jsonl")
    slates = rows(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")
    assert len(questions) == len(slates) == 750
    assert {row["query_id"] for row in questions} == {row["query_id"] for row in slates}
    assert len({row["slate_id"] for row in slates}) == 750
    assert all(len(row["candidate_ids"]) == 20 for row in slates)
    assert all(len(row["candidate_ids"]) == len(set(row["candidate_ids"])) for row in slates)
    assert all(row["qrels_accessed_during_generation"] is False for row in slates)


def test_m2_candidates_resolve_to_frozen_corpus():
    corpus_ids = {row["doc_id"] for row in rows(ROOT / "data/corpus/passages.jsonl")}
    for slate in rows(ROOT / "data/frozen_slates/m2_top20_slates.jsonl"):
        assert set(slate["candidate_ids"]).issubset(corpus_ids)


def test_m2_manifest_integrity():
    manifest = json.loads((ROOT / "data/manifests/m2_retrieval_manifest.json").read_text())
    assert manifest["collection"] == "decisionrag_m2_qwen_embedding_v1"
    assert manifest["embedding_dimension"] == 2048
    assert manifest["corpus_passages"] == manifest["qdrant_points"] == 144650
    assert manifest["queries"] == 750
    assert manifest["qrels_accessed_during_generation"] is False
