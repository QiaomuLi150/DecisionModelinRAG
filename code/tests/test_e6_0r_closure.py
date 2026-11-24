from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import e6_0r_closure as closure


def test_dochop_is_not_natural_complete_support():
    audit = closure.audit_dochop()
    assert audit["rows"] == 11379
    assert audit["natural_corpus_present"] is False
    assert audit["answer_field_present_rows"] == 0
    assert audit["evidence_label_field_present_rows"] == 0
    assert audit["admission"].startswith("NOT_ADMISSIBLE")


def test_garage_fixed_pool_k_accounting_and_alignment():
    audit = closure.audit_garage()
    assert audit["rows"] == 2366
    assert audit["exact_k15_rows"] == 2329
    assert audit["aligned_annotation_lengths"] == 2366
    assert audit["complete_support_set_field_present"] is False
    assert audit["admission"].startswith("FIXED_POOL_GROUNDING")


def test_scifact_natural_relevance_identity_and_sparse_qrels():
    audit = closure.audit_scifact()
    assert audit["query_rows"] == 1109
    assert audit["corpus_rows"] == 5183
    assert audit["qrel_query_ids"] == 1109
    assert audit["qrel_queries_missing_from_query_file"] == 0
    assert audit["qrel_docs_missing_from_corpus"] == 0
    assert audit["complete_support_sets"] is False
    assert audit["admission"].startswith("NATURAL_RELEVANCE")


def test_scifact_component_split_is_disjoint_and_reserved_is_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setattr(closure, "OUT", tmp_path)
    split = closure.freeze_scifact_splits()
    parts = split["partitions"]
    assert sum(parts[k]["query_count"] for k in parts) == 1109
    assert parts["RESERVED"].get("query_ids") is None
    assert parts["CAL"].get("query_ids")
    assert parts["EXPLORATORY"].get("query_ids")
    all_visible = parts["CAL"]["query_ids"] + parts["EXPLORATORY"]["query_ids"]
    assert len(all_visible) == len(set(all_visible))


def test_frozen_e5_slate_and_historical_roster_unchanged():
    slate = closure.E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    assert closure.sha_file(slate) == closure.EXPECTED_SLATE_SHA
    registry = closure.E6 / "E6_0_EIGHT_MODEL_VERSION_LICENSE_REGISTRY.csv"
    text = registry.read_text(encoding="utf-8")
    for name in ("pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b", "jina35", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"):
        assert name in text


def test_endpoint_separation_and_unjudged_policy():
    assert "N-REL" != "F-GRD"
    assert "UNKNOWN" not in {"ANSWER-THE-QUESTION", "RELATED-INFORMATION", "OUTDATED"}
    schema = {
        "primary": "BEIR/SciFact",
        "secondary": "GaRAGe K15",
        "historical": "QASPER CSSR-U",
    }
    assert len(set(schema.values())) == 3


def test_no_inference_gpu_or_paid_scope_in_closure_source():
    source = Path(closure.__file__).read_text(encoding="utf-8").lower()
    assert "import torch" not in source
    assert "transformers" not in source
    assert "nvidia-smi" not in source
    assert "requests.post" not in source
    assert "model.forward" not in source


def test_historical_manifest_verifier_does_not_accept_missing_files():
    missing = Path("<REDACTED_PATH>")
    missing.write_text(json.dumps({"files": [{"path": "does/not/exist", "sha256": "x"}]}), encoding="utf-8")
    result = closure.verify_manifest(missing)
    assert result["failed"] == 1
