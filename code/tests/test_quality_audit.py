from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bge_model_input_uses_native_tokenizer_while_views_remain_common():
    m3 = (ROOT / "scripts/run_m3_selectors.py").read_text(encoding="utf-8")
    m5 = (ROOT / "scripts/run_m5_serving.py").read_text(encoding="utf-8")
    assert 'model_tokenizer = AutoTokenizer.from_pretrained(path)' in m3
    assert 'encoded = model_tokenizer(' in m3
    assert 'self.model_tokenizer = AutoTokenizer.from_pretrained(path)' in m5
    assert 'encoded = self.model_tokenizer(' in m5
    # The shared Laya tokenizer remains the view tokenizer; the old defect was
    # using that tokenizer for BGE model IDs as well.
    assert 'views = [make_views(item["question"], c, view_tokenizer)' in m3
    assert 'views = [make_views(question, candidate, self.common_tokenizer)' in m5


def test_original_non_jev_artifacts_are_hash_manifested():
    root = ROOT / "data/results/audit/original_non_jev"
    manifest = root / "ORIGINAL_ARTIFACT_HASHES.json"
    assert manifest.exists()
    import json
    values = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(values) >= 30
    assert "m3/bge_all_pointwise.jsonl" in values
    assert "m4/reader_results.jsonl" in values


def test_quality_audit_stop_gate_and_interleaved_traces():
    import json

    audit = ROOT / "data/results/audit/quality"
    manifest = json.loads((audit / "QUALITY_AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "NON_JEV_COMPLETE_AUDITED_JEV_PENDING_APPROVAL"
    assert manifest["jev_calls"] == 0
    assert manifest["full_expansion_started"] is False
    policies = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]
    for policy in policies:
        traces = [json.loads(line) for line in (audit / "m5_interleaved" / f"{policy}_traces.jsonl").read_text().splitlines() if line.strip()]
        assert len(traces) == 100
        assert all(row["status"] == "ok" and row["paid_api_usage"] is False for row in traces)
        assert all(all(key in row for key in ("request_duration_ms", "retrieval_ms", "reader_ms", "embedding_ms", "qdrant_ms", "bm25_ms", "rrf_ms")) for row in traces)
    assert (audit / "QUALITY_CORRECTNESS_AUDIT.md").exists()
    assert (audit / "M3_SCORE_ALIGNMENT_AUDIT.csv").exists()
    assert (audit / "M5_INTERLEAVED_SERVING_REPORT.md").exists()


def test_qwen_corrected_contract_and_audit_artifacts():
    import json

    m3 = (ROOT / "scripts/run_m3_selectors.py").read_text(encoding="utf-8")
    m5 = (ROOT / "scripts/run_m5_serving.py").read_text(encoding="utf-8")
    assert '"chat_template_kwargs": {"add_generation_prompt": True}' in m3
    assert '"chat_template_kwargs": {"add_generation_prompt": True}' in m5

    audit = ROOT / "data/results/audit/quality"
    required = [
        "QWEN_SEMANTIC_CORRECTNESS_AUDIT.md",
        "QWEN_REFERENCE_VS_SERVICE.csv",
        "QWEN_SERVING_CONFIGURATION.json",
        "QWEN_SCORE_ALIGNMENT.jsonl",
        "QWEN_REGRESSION_TEST_REPORT.md",
        "QWEN_ERRATUM.md",
        "QWEN_CORRECTED_M5_REPORT.md",
    ]
    assert all((audit / name).exists() for name in required)
    summary = json.loads((audit / "QWEN_AUDIT_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["query_count"] == 20
    assert summary["candidate_count"] == 400
    assert summary["jev_calls"] == 0
    assert summary["full_expansion_started"] is False
    assert json.loads((audit / "qwen_original_service/ORIGINAL_QWEN_SERVICE_ARTIFACT_HASHES.json").read_text(encoding="utf-8"))
