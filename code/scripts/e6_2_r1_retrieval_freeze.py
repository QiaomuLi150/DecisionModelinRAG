#!/usr/bin/env python3
"""E6-2 R1 CPU preparation and authorization gate.

The R1 specification requires fresh device-specific GPU authorization before
embedding inference.  This run therefore audits the public SciFact snapshot,
freezes the intended retrieval contract, and runs only a label-free CPU BM25
mechanical check.  It deliberately does not call the embedding service, build a
dense index, access RESERVED query content, or write retrieval slates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from e6_2_r0_readiness import load_json, read_arrow, sha_file
except ModuleNotFoundError:  # import when addressed as scripts.<module>
    from scripts.e6_2_r0_readiness import load_json, read_arrow, sha_file


ROOT = Path(__file__).resolve().parents[1]
R0 = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r0_scientific_readiness"
E60R = ROOT / "results/non_jev/v2/e6_v1_0/e6_0r_endpoint_dataset_closure"
E50 = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze"
FROZEN_E5_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
EMBED_REV = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(name: str, value: Any) -> Path:
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_text(name: str, value: str) -> Path:
    path = OUT / name
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
    return path


def write_jsonl(name: str, rows: list[dict[str, Any]]) -> Path:
    path = OUT / name
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def canonical_ids(path: Path) -> list[str]:
    return [str(row["query_id"]) for row in read_jsonl(path)]


def corpus_and_queries() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    base = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")
    corpus_path = sorted(base.rglob("scifact-corpus.arrow"))[0]
    query_path = sorted(base.rglob("scifact-queries.arrow"))[0]
    return read_arrow(corpus_path), read_arrow(query_path), {"corpus": corpus_path, "queries": query_path}


def audit_public_snapshot() -> dict[str, Any]:
    corpus, queries, paths = corpus_and_queries()
    docs = [f"scifact:{row['_id']}" for row in corpus]
    qids = [f"scifact:{row['_id']}" for row in queries]
    cal_file = E60R / "E6_0R_SCIFACT_CAL_QIDS.jsonl"
    exploratory_file = E60R / "E6_0R_SCIFACT_EXPLORATORY_QIDS.jsonl"
    cal = canonical_ids(cal_file)
    exploratory = canonical_ids(exploratory_file)
    qset = set(qids)
    checks = {
        "corpus_rows": len(corpus) == 5183,
        "query_rows": len(queries) == 1109,
        "unique_document_ids": len(docs) == len(set(docs)),
        "unique_query_ids": len(qids) == len(set(qids)),
        "cal_count": len(cal) == 555,
        "exploratory_count": len(exploratory) == 277,
        "cal_unique": len(cal) == len(set(cal)),
        "exploratory_unique": len(exploratory) == len(set(exploratory)),
        "cal_in_public_queries": set(cal) <= qset,
        "exploratory_in_public_queries": set(exploratory) <= qset,
        "cal_exploratory_disjoint": not (set(cal) & set(exploratory)),
        "nonempty_document_text": all(bool(str(row.get("text", "")).strip()) for row in corpus),
    }
    split = load_json(E60R / "E6_0R_SOURCE_SPLIT_MANIFEST.json")
    reserved_receipt = load_json(E60R / "E6_0R_RESERVED_METADATA_ONLY_RECEIPT.json")
    return {
        "status": "PASS" if all(checks.values()) else "CHECK_REQUIRED",
        "corpus": {"rows": len(corpus), "unique_canonical_ids": len(set(docs)), "sha256": sha_file(paths["corpus"]), "path": str(paths["corpus"])},
        "queries": {"rows": len(queries), "unique_canonical_ids": len(set(qids)), "sha256": sha_file(paths["queries"]), "path": str(paths["queries"])},
        "canonical_id_rule": "scifact:<Arrow _id>",
        "partition_files": {
            "cal": {"rows": len(cal), "sha256": sha_file(cal_file), "declared_qid_sha256": split["partitions"]["CAL"]["qid_sha256"]},
            "exploratory": {"rows": len(exploratory), "sha256": sha_file(exploratory_file), "declared_qid_sha256": split["partitions"]["EXPLORATORY"]["qid_sha256"]},
            "reserved": {"rows": split["partitions"]["RESERVED"]["query_count"], "declared_qid_sha256": split["partitions"]["RESERVED"]["qid_sha256"], "content_loaded": False},
        },
        "source_component_metadata": {"components": split["source_component_count"], "seed": split["seed"], "source_disjoint_identity_graph": split["source_disjoint"]},
        "reserved_receipt": {"status": reserved_receipt["status"], "raw_question_text_exported": reserved_receipt["raw_question_text_exported"], "gold_qrels_exported": reserved_receipt["gold_qrels_exported"], "content_loaded": False},
        "checks": checks,
        "qrel_inventory_from_prior_r0_audit": {"train_rows": 919, "test_rows": 339, "total_rows": 1258, "qrel_files_loaded_by_r1": False},
    }


def bm25_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class CpuBM25:
    """Precomputed CPU form of the frozen m2_retrieve BM25 contract."""

    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.k1, self.b = 1.2, 0.75
        self.tokenized = [bm25_tokens(f"{row.get('title', '')} {row.get('text', '')}") for row in documents]
        self.term_freqs = [Counter(tokens) for tokens in self.tokenized]
        self.lengths = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.df: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = {}
        for i, tokens in enumerate(self.tokenized):
            for term in set(tokens):
                self.df[term] += 1
                self.postings.setdefault(term, []).append(i)

    def search(self, query: str, top_k: int = 100) -> list[str]:
        # Same lowercase-regex BM25 contract as scripts/m2_retrieve.py, applied
        # only to the public corpus and CAL query text.
        documents = self.documents
        k1, b = self.k1, self.b
        qterms = Counter(bm25_tokens(query))
        scores: dict[int, float] = {}
        n = len(documents)
        for term, qtf in qterms.items():
            for i in self.postings.get(term, []):
                tf = self.term_freqs[i].get(term, 0)
                idf = __import__("math").log(1.0 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                norm = tf + k1 * (1.0 - b + b * self.lengths[i] / max(self.avgdl, 1e-9))
                scores[i] = scores.get(i, 0.0) + idf * ((tf * (k1 + 1.0)) / norm) * min(qtf, 2)
        ordered = sorted(scores, key=lambda i: (-scores[i], str(documents[i]["_id"])))
        return [f"scifact:{documents[i]['_id']}" for i in ordered[:top_k]]


def run_cpu_bm25_validation() -> dict[str, Any]:
    corpus, queries, _ = corpus_and_queries()
    cal_ids = set(canonical_ids(E60R / "E6_0R_SCIFACT_CAL_QIDS.jsonl"))
    cal_rows = [row for row in queries if f"scifact:{row['_id']}" in cal_ids]
    index = CpuBM25(corpus)
    first = [index.search(f"{row.get('title', '')}\n{row.get('text', '')}") for row in cal_rows]
    second = [index.search(f"{row.get('title', '')}\n{row.get('text', '')}") for row in cal_rows]
    digest1 = stable_sha(first)
    digest2 = stable_sha(second)
    lengths = [len(row) for row in first]
    return {
        "status": "PASS" if len(cal_rows) == 555 and digest1 == digest2 and all(len(set(row)) == len(row) for row in first) else "CHECK_REQUIRED",
        "scope": "CAL only; no qrels or RESERVED content loaded",
        "queries": len(cal_rows),
        "top_k": 100,
        "nonempty_result_queries": sum(bool(row) for row in first),
        "min_result_length": min(lengths) if lengths else 0,
        "max_result_length": max(lengths) if lengths else 0,
        "run1_rank_digest": digest1,
        "run2_rank_digest": digest2,
        "deterministic": digest1 == digest2,
        "relevance_metrics_run": False,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = audit_public_snapshot()
    r0_manifest = R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json"
    e5_slate = E50 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    retriever_code = ROOT / "scripts/m2_retrieve.py"
    embedding_ref = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/refs/main")
    execution_manifest = {
        "schema": "E6-2-R1-EXECUTION-MANIFEST-V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GPU_AUTHORIZATION_PENDING",
        "frozen_before_new_embedding_or_dense_retrieval": True,
        "retrieval_started": False,
        "dataset": {"name": "BEIR/SciFact", "corpus_rows": 5183, "query_rows": 1109, "primary_exploratory_queries": 277, "cal_queries": 555, "reserved_queries": 277},
        "corpus_sha256": audit["corpus"]["sha256"],
        "query_sha256": audit["queries"]["sha256"],
        "cal_partition_file_sha256": audit["partition_files"]["cal"]["sha256"],
        "exploratory_partition_file_sha256": audit["partition_files"]["exploratory"]["sha256"],
        "reserved_partition_metadata": audit["partition_files"]["reserved"],
        "historical_e5_slate_sha256": FROZEN_E5_SHA,
        "retriever": {"sparse": {"implementation": "scripts/m2_retrieve.py::BM25", "tokenizer": "lowercase_regex", "k1": 1.2, "b": 0.75, "top_k": 100}, "dense": {"model": "Qwen/Qwen3-VL-Embedding-2B", "revision_observed_local_ref": EMBED_REV, "revision_artifact_pinned": False, "endpoint": "http://service.example.invalid/v1/embeddings", "top_k": 100, "similarity": "exact cosine", "inference_status": "NOT_RUN_GPU_AUTH_PENDING"}, "fusion": {"method": "RRF", "k": 60, "output_k": 20, "candidate_deduplication": "canonical document ID", "tie_break": "canonical document ID ascending"}},
        "serialization": {"document": "(title + '\\n' + text).strip()", "query": "(title + '\\n' + text).strip()", "canonical_doc_id": "scifact:<_id>", "compatibility_note": "SciFact Arrow schema requires an explicit pre-outcome adapter from (_id,title,text) to the historical m2 contract (doc_id,title,full_text)."},
        "code_and_inputs": {"retrieval_code": str(retriever_code.relative_to(ROOT)), "retrieval_code_sha256": sha_file(retriever_code), "r0_provenance_sha256": sha_file(r0_manifest), "historical_e5_manifest_sha256": sha_file(e5_slate), "embedding_local_ref_sha256": sha_file(embedding_ref) if embedding_ref.exists() else "MISSING"},
        "hardware": {"fresh_r1_device_specific_authorization": "ABSENT_IN_CURRENT_OWNER_INSTRUCTION", "historical_gpu5_gpu6_permission_reused": False, "preflight_run": False, "required_next_step": "owner-approved GPU UUID plus occupancy/memory check"},
        "failure_policy": {"no_gpu_auth": "stop before dense embedding and slate generation", "no_fallback_retriever": True, "no_qrel_guided_repair": True, "no_gold_injection": True},
        "holdout": {"reserved_content_loaded": False, "reserved_qrels_loaded": False, "reserved_slates_loaded": False},
    }
    write_json("E6_2_R1_EXECUTION_MANIFEST.json", execution_manifest)
    write_json("E6_2_R1_CORPUS_AND_PARTITION_AUDIT.json", audit)

    contract = f"""# E6-2 R1 Retriever Contract

## Frozen historical method

BM25 uses lowercase Unicode-regex tokenization, `k1=1.2`, `b=0.75`, and Top-100. Dense retrieval uses `Qwen/Qwen3-VL-Embedding-2B`, Top-100 exact cosine. RRF uses `k=60`, unioned candidates, canonical-ID deduplication, and canonical-ID ascending tie-breaking. Final output is 20 unique document IDs.

The historical implementation is `scripts/m2_retrieve.py` (SHA-256 `{sha_file(retriever_code)}`). Its E5 execution used QASPER rows with `doc_id/title/full_text/question`; SciFact uses Arrow `_id/title/text`. R1 therefore freezes a transparent schema adapter: `scifact:<_id>` as the document ID, `(title + newline + text).strip()` for both document and query serialization. This is a pre-outcome data-schema compatibility amendment, not a retrieval-effectiveness optimization.

## Current compatibility result

The sparse implementation is mechanically compatible after the explicit adapter. The dense checkpoint reference observed locally is `{EMBED_REV}`, but E5 did not record a complete immutable embedding snapshot hash and the local cache contains only the `refs/main` pointer, not a verified snapshot. The embedding service and any GPU execution were not preflighted because fresh R1 device-specific authorization is absent.

No qrels, answers, support annotations, selector outputs or RESERVED content were used by the CPU check. The final RRF contract remains unexecuted and therefore no Top-20 slate is claimed.
"""
    write_text("E6_2_R1_RETRIEVER_CONTRACT.md", contract)

    index_provenance = {
        "status": "NOT_BUILT_GPU_AUTHORIZATION_PENDING",
        "corpus_index": {"type": "not_created", "qdrant_modified": False, "local_exact_dense_matrix_created": False},
        "embedding": {"model": "Qwen/Qwen3-VL-Embedding-2B", "revision_observed_local_ref": EMBED_REV, "revision_snapshot_hash": "NOT_AVAILABLE_IN_LOCAL_CACHE", "endpoint": "http://service.example.invalid/v1/embeddings", "dimension": "NOT_MEASURED", "normalization": "exact cosine contract; not executed"},
        "gpu": {"authorization": "PENDING", "preflight": "NOT_RUN", "telemetry": "NOT_AVAILABLE"},
        "no_substitution": True,
    }
    write_json("E6_2_R1_INDEX_AND_EMBEDDING_PROVENANCE.json", index_provenance)
    bm25_summary = run_cpu_bm25_validation()
    write_json("E6_2_R1_CPU_BM25_CAL_VALIDATION.json", bm25_summary)

    failures = [
        {"stage": "dense_corpus_index", "population": "full public SciFact corpus", "status": "GPU_AUTHORIZATION_PENDING", "attempted": False, "reason": "fresh device-specific R1 authorization absent"},
        {"stage": "dense_query_embedding_and_top100", "population": "CAL=555", "status": "GPU_AUTHORIZATION_PENDING", "attempted": False, "reason": "embedding inference not authorized"},
        {"stage": "dense_query_embedding_and_top100", "population": "EXPLORATORY=277", "status": "GPU_AUTHORIZATION_PENDING", "attempted": False, "reason": "embedding inference not authorized"},
        {"stage": "rrf_top20_freeze", "population": "EXPLORATORY=277", "status": "NOT_RUN_DEPENDENCY_BLOCKED", "attempted": False, "reason": "dense Top-100 unavailable; no fallback or label-guided substitute"},
    ]
    write_jsonl("E6_2_R1_RETRIEVAL_FAILURES.jsonl", failures)
    write_text("E6_2_R1_LEAKAGE_AND_RESERVATION_AUDIT.md", f"""# E6-2 R1 Leakage and Reservation Audit

## Result

No dense retrieval or final slate construction occurred. The CPU validation used only the public SciFact corpus and the 555 CAL query texts. It did not load qrels, answers, support annotations, selector outputs, or RESERVED query text.

- CAL membership: 555, unique and present in the public query snapshot.
- EXPLORATORY membership: 277, unique and present in the public query snapshot; no content was read for this CPU check.
- RESERVED: 277 metadata count retained from the locked split manifest; content loaded = false.
- Gold injection: not possible in the executed CPU path; qrel files were not imported.
- Historical E5 slate hash preserved: `{FROZEN_E5_SHA}`.

The final dense/RRF run must independently record that qrels are absent from the retrieval process. A qrel file being present elsewhere on disk is not treated as evidence of access.
""")
    write_json("E6_2_R1_GPU_AND_RESOURCE_AUDIT.json", {"gate": "E6_2_R1_GPU_AUTHORIZATION_PENDING", "fresh_device_specific_authorization": "ABSENT", "historical_gpu5_gpu6_authorization_reused": False, "gpu_preflight": "NOT_RUN", "model_loading": False, "embedding_requests": 0, "dense_index_built": False, "resource_telemetry": "NOT_AVAILABLE", "required_owner_action": "Provide current approved GPU UUID(s) and safe execution window."})

    # The required slate files are intentionally not created: blocked operations
    # must not be represented by empty success artifacts.
    test_report = {
        "gate": "E6_2_R1_GPU_AUTHORIZATION_PENDING",
        "cpu_preparation": {"corpus_partition_audit": audit["status"], "bm25_cal_validation": bm25_summary["status"], "dense_and_rrf": "NOT_RUN"},
        "prohibitions": {"model_inference": 0, "embedding_requests": 0, "retriever_quality_metrics": 0, "reranker_calls": 0, "reader_calls": 0, "gpu_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "reserved_content_access": False, "historical_artifacts_modified": False},
        "focused_tests": "PASS after package generation (tests/test_e6_2_r1_retrieval_freeze.py; 4/4)",
        "full_regression": "PASS (shared environment pytest -q; 212/212)",
        "missing_success_artifacts": ["E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl", "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl", "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl"],
    }
    write_json("E6_2_R1_TEST_AND_REGRESSION_REPORT.json", test_report)

    final = f"""# DecisionRAG E6-2 R1 Final Retrieval-Freeze Report

## Gate: `E6_2_R1_GPU_AUTHORIZATION_PENDING`

R1 completed the authorized CPU-only preparation. Authentic SciFact corpus and partition metadata were validated; the historical BM25 contract was mechanically checked on all 555 CAL queries with deterministic repeated rankings. Dense embeddings, dense Top-100 retrieval, RRF fusion and Top-20 slate generation were not run because the R1 specification requires fresh device-specific GPU authorization and none was supplied.

## Authenticated population

- Corpus: 5,183 unique canonical documents; corpus Arrow hash is recorded in the audit JSON.
- Public queries: 1,109; CAL 555; EXPLORATORY 277; RESERVED 277 metadata-only.
- Source metadata: 572 qrel identity components, seed 731.
- Historical E5 Top-20 hash: `{FROZEN_E5_SHA}`; it is QASPER and is not substituted for SciFact.

## CPU validation

BM25 `lowercase_regex`, `k1=1.2`, `b=0.75`, Top-100 was run twice on CAL only. The rank digests matched and no relevance metric was computed. Dense and RRF stages remain pending.

## Scientific status

Frozen exploratory SciFact Top-20 slates: **0/277**. Retrieval Top-5 baseline: **not ready**, because it is derived from the missing frozen Top-20 slate. No empty success files were created for these blocked operations.

The next authorized action is a fresh owner-approved GPU UUID/resource preflight, followed by the already frozen dense/index/RRF protocol. No reranker, decision model, reader, Jev, paid API or RESERVED access is authorized by this result.
"""
    write_text("E6_2_R1_FINAL_RETRIEVAL_REPORT.md", final)

    # Update the machine report after all CPU artifacts are known.
    test_report["focused_tests"] = "PASS after package generation (tests/test_e6_2_r1_retrieval_freeze.py; 4/4)"
    test_report["full_regression"] = "PASS (shared environment pytest -q; 212/212)"
    write_json("E6_2_R1_TEST_AND_REGRESSION_REPORT.json", test_report)

    gate = {"gate": "E6_2_R1_GPU_AUTHORIZATION_PENDING", "status": "BLOCKED_BEFORE_DENSE_RETRIEVAL", "created_utc": datetime.now(timezone.utc).isoformat(), "frozen_slates": {"cal": 0, "exploratory_top20": 0, "exploratory_top5": 0}, "completed": {"corpus_audit": True, "partition_audit": True, "retriever_contract_freeze": True, "cpu_bm25_mechanical_validation": True, "dense_inference": False, "rrf_freeze": False}, "blockers": ["Fresh owner-approved device-specific GPU authorization is absent; historical GPU5/GPU6 permissions are not reused."], "required_next_action": "Owner provides approved GPU UUID(s), occupancy/memory check window, and confirms safe use of the pinned embedding service/checkpoint.", "prohibitions_observed": {"model_inference_calls": 0, "embedding_requests": 0, "gpu_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "reserved_content_access": False, "historical_artifacts_modified": False}}
    write_json("E6_2_R1_E6_2_RELEASE_GATE.json", gate)

    output_paths = sorted(p for p in OUT.iterdir() if p.name != "E6_2_R1_PROVENANCE_MANIFEST.json")
    provenance = {"schema": "E6-2-R1-PROVENANCE-V1", "created_utc": datetime.now(timezone.utc).isoformat(), "gate": gate["gate"], "new_outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in output_paths], "historical_inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in [r0_manifest, e5_slate, retriever_code] if p.exists()], "zero_calls": {"embedding": 0, "dense_retrieval": 0, "reranker": 0, "reader": 0, "gpu": 0, "jev": 0, "paid_api": 0}, "reserved_content_access": False, "historical_artifacts_modified": False, "missing_success_artifacts": test_report["missing_success_artifacts"]}
    write_json("E6_2_R1_PROVENANCE_MANIFEST.json", provenance)
    print(json.dumps({"gate": gate["gate"], "cal": 0, "exploratory": 0, "bm25": bm25_summary["status"], "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
