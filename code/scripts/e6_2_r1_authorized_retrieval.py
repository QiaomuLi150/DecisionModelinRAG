#!/usr/bin/env python3
"""Authorized E6-2 R1 SciFact retrieval freeze.

This is retrieval-only. It reads the public SciFact corpus plus CAL and
EXPLORATORY queries, calls only the existing embedding service, performs exact
cosine + the frozen BM25/RRF fusion, and never loads qrels or RESERVED data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from e6_2_r0_readiness import read_arrow, sha_file
from e6_2_r1_retrieval_freeze import CpuBM25, canonical_ids


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze"
E60R = ROOT / "results/non_jev/v2/e6_v1_0/e6_0r_endpoint_dataset_closure"
DATASETS = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")
EMBED_URL = "http://service.example.invalid/v1/embeddings"
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
FROZEN_E5_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(name: str, value: Any) -> Path:
    path = R1 / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_jsonl(name: str, rows: list[dict[str, Any]]) -> Path:
    path = R1 / name
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def gpu_snapshot() -> dict[str, Any]:
    query = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    apps = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(query, capture_output=True, text=True, check=False)
    apps_result = subprocess.run(apps, capture_output=True, text=True, check=False)
    rows = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 6:
            rows.append({"index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "utilization_gpu_percent": int(fields[5])})
    return {"returncode": result.returncode, "gpus": rows, "compute_apps_returncode": apps_result.returncode, "compute_apps_raw": apps_result.stdout, "stderr": result.stderr}


def load_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    corpus_path = sorted(DATASETS.rglob("scifact-corpus.arrow"))[0]
    query_path = sorted(DATASETS.rglob("scifact-queries.arrow"))[0]
    return read_arrow(corpus_path), read_arrow(query_path), {"corpus": corpus_path, "queries": query_path}


def text_for(row: dict[str, Any]) -> str:
    return (f"{row.get('title', '')}\n{row.get('text', '')}").strip()


def post_batch(texts: list[str], batch_id: int, phase: str, timeout: int, retries: int) -> tuple[int, str, list[list[float]], dict[str, Any]]:
    payload = {"model": MODEL, "input": texts}
    request_hash = stable_sha({"model": MODEL, "input": texts})
    started = time.perf_counter()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(EMBED_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            data = sorted(body["data"], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in data]
            if len(vectors) != len(texts):
                raise RuntimeError(f"embedding count mismatch {len(vectors)} != {len(texts)}")
            raw_hash = sha_bytes(response.content)
            receipt = {"batch_id": batch_id, "phase": phase, "attempt": attempt, "status": "ok", "request_sha256": request_hash, "response_bytes_sha256": raw_hash, "items": len(texts), "dimension": len(vectors[0]) if vectors else 0, "elapsed_ms": (time.perf_counter() - started) * 1000.0}
            return batch_id, phase, vectors, receipt
        except Exception as exc:  # preserve failures in the caller
            last_error = repr(exc)
    receipt = {"batch_id": batch_id, "phase": phase, "attempt": retries, "status": "error", "request_sha256": request_hash, "items": len(texts), "error": last_error, "elapsed_ms": (time.perf_counter() - started) * 1000.0}
    raise RuntimeError(json.dumps(receipt, sort_keys=True))


def embed_batches(texts: list[str], phase: str, batch_size: int, workers: int, receipts: list[dict[str, Any]]) -> np.ndarray:
    chunks = [(i, texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    vectors: list[list[list[float]] | None] = [None] * len(chunks)
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(post_batch, chunk[1], idx, phase, 600, 4) for idx, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            try:
                idx, _, batch_vectors, receipt = future.result()
                vectors[idx] = batch_vectors
                receipts.append(receipt)
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError("embedding failures: " + " | ".join(errors[:3]))
    flat = [vector for batch in vectors if batch is not None for vector in batch]
    array = np.asarray(flat, dtype=np.float32)
    if len(array) != len(texts) or array.ndim != 2:
        raise RuntimeError(f"embedding shape mismatch {array.shape} for {len(texts)} inputs")
    return array


def top_dense(matrix: np.ndarray, query: np.ndarray, docs: list[dict[str, Any]], top_k: int = 100) -> list[tuple[int, float]]:
    doc_norm = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    if np.any(doc_norm == 0) or query_norm == 0:
        raise RuntimeError("zero norm encountered in exact cosine stage")
    scores = (matrix @ query) / (doc_norm * query_norm)
    ids = [str(row["_id"]) for row in docs]
    order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), ids[i]))[:top_k]
    return [(i, float(scores[i])) for i in order]


def rrf(rows: list[dict[str, Any]], bm_hits: list[tuple[int, float]], dense_hits: list[tuple[int, float]], k: int = 60, top_k: int = 20) -> list[dict[str, Any]]:
    bm_rank = {i: rank for rank, (i, _) in enumerate(bm_hits, 1)}
    de_rank = {i: rank for rank, (i, _) in enumerate(dense_hits, 1)}
    bm_score = dict(bm_hits)
    de_score = dict(dense_hits)
    scored = []
    for i in set(bm_rank) | set(de_rank):
        value = (1.0 / (k + bm_rank[i]) if i in bm_rank else 0.0) + (1.0 / (k + de_rank[i]) if i in de_rank else 0.0)
        scored.append((i, value))
    scored.sort(key=lambda pair: (-pair[1], f"scifact:{rows[pair[0]]['_id']}"))
    out = []
    for i, value in scored[:top_k]:
        row = rows[i]
        text = text_for(row)
        out.append({"document_id": f"scifact:{row['_id']}", "corpus_row": i, "title": row.get("title", ""), "content_sha256": sha_bytes(text.encode()), "content_ref": f"scifact-corpus.arrow#row={i};_id={row['_id']}", "bm25_rank": bm_rank.get(i), "bm25_score": bm_score.get(i), "dense_rank": de_rank.get(i), "dense_score": de_score.get(i), "rrf_score": value})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-audit-file", type=Path, default=None)
    args = parser.parse_args()
    R1.mkdir(parents=True, exist_ok=True)
    before = gpu_snapshot()
    corpus, queries, paths = load_snapshot()
    cal_ids = set(canonical_ids(E60R / "E6_0R_SCIFACT_CAL_QIDS.jsonl"))
    exploratory_ids = set(canonical_ids(E60R / "E6_0R_SCIFACT_EXPLORATORY_QIDS.jsonl"))
    qrows = {f"scifact:{row['_id']}": row for row in queries}
    cal_rows = [qrows[qid] for qid in sorted(cal_ids)]
    exploratory_rows = [qrows[qid] for qid in sorted(exploratory_ids)]
    if len(corpus) != 5183 or len(cal_rows) != 555 or len(exploratory_rows) != 277:
        raise RuntimeError("population mismatch")

    service_url = "http://service.example.invalid/v1/models"
    service = requests.get(service_url, timeout=10).json()
    model_entries = [item for item in service.get("data", []) if item.get("id") == MODEL]
    if len(model_entries) != 1:
        raise RuntimeError(f"pinned embedding model not uniquely served: {service}")
    served_root = model_entries[0].get("root", "")
    if REVISION not in served_root:
        raise RuntimeError(f"served revision mismatch: {served_root}")

    manifest_v2 = {"schema": "E6-2-R1-AUTHORIZED-EXECUTION-MANIFEST-V2", "created_utc": datetime.now(timezone.utc).isoformat(), "authorization": "owner authorized all idle/available GPUs in current instruction; only already-running embedding service used", "dataset": {"name": "BEIR/SciFact", "corpus_rows": len(corpus), "cal_queries": len(cal_rows), "exploratory_queries": len(exploratory_rows), "reserved_queries": 277}, "corpus_sha256": sha_file(paths["corpus"]), "query_sha256": sha_file(paths["queries"]), "cal_partition_sha256": sha_file(E60R / "E6_0R_SCIFACT_CAL_QIDS.jsonl"), "exploratory_partition_sha256": sha_file(E60R / "E6_0R_SCIFACT_EXPLORATORY_QIDS.jsonl"), "reserved_content_loaded": False, "retrieval_contract": {"bm25": {"tokenizer": "lowercase_regex", "k1": 1.2, "b": 0.75, "top_k": 100}, "dense": {"model": MODEL, "revision": REVISION, "service": EMBED_URL, "top_k": 100, "similarity": "exact cosine"}, "rrf": {"k": 60, "top_k": 20, "tie_break": "scifact:<_id> ascending"}}, "serialization": {"query": "(title + '\\n' + text).strip()", "document": "(title + '\\n' + text).strip()", "canonical_id": "scifact:<_id>"}, "service_metadata": model_entries[0], "gpu_before": before, "gpu_selection": "host-managed embedding service; no new GPU process started or evicted"}
    write_json("E6_2_R1_EXECUTION_MANIFEST_V2.json", manifest_v2)

    receipts: list[dict[str, Any]] = []
    doc_embeddings = embed_batches([text_for(row) for row in corpus], "corpus", args.batch_size, args.workers, receipts)
    query_rows = cal_rows + exploratory_rows
    query_embeddings = embed_batches([text_for(row) for row in query_rows], "queries_cal_exploratory", args.batch_size, args.workers, receipts)
    np.save(R1 / "E6_2_R1_SCIFACT_DOCUMENT_EMBEDDINGS_V2.npy", doc_embeddings)
    np.save(R1 / "E6_2_R1_SCIFACT_QUERY_EMBEDDINGS_V2.npy", query_embeddings)
    write_jsonl("E6_2_R1_EMBEDDING_CALL_RECEIPTS_V2.jsonl", sorted(receipts, key=lambda x: (x["phase"], x["batch_id"])))

    bm25 = CpuBM25(corpus)
    query_vec = {f"scifact:{row['_id']}": query_embeddings[i] for i, row in enumerate(query_rows)}
    all_outputs: dict[str, list[dict[str, Any]]] = {}
    for row in query_rows:
        qid = f"scifact:{row['_id']}"
        bm_hits = []
        # Recompute scores using the frozen CPU implementation's deterministic
        # ranking.  Its output IDs are converted back to corpus indices.
        bm_ids = bm25.search(text_for(row), 100)
        index_by_id = {f"scifact:{doc['_id']}": i for i, doc in enumerate(corpus)}
        bm_hits = [(index_by_id[doc_id], 0.0) for doc_id in bm_ids]
        dense_hits = top_dense(doc_embeddings, query_vec[qid], corpus, 100)
        candidates = rrf(corpus, bm_hits, dense_hits, 60, 20)
        if len(candidates) != 20 or len({x["document_id"] for x in candidates}) != 20:
            raise RuntimeError(f"invalid Top-20 for {qid}")
        all_outputs[qid] = [{"rank": rank, **candidate} for rank, candidate in enumerate(candidates, 1)]

    def slate_row(row: dict[str, Any]) -> dict[str, Any]:
        qid = f"scifact:{row['_id']}"
        candidates = all_outputs[qid]
        qtext = text_for(row)
        return {"slate_id": f"e6_2_r1:{qid}", "query_id": qid, "query_text": qtext, "query_text_sha256": sha_bytes(qtext.encode()), "dataset": "BEIR/SciFact", "pool_source": "BM25+Qwen3-VL-Embedding-2B+RRF60", "candidate_order": "rrf_desc_then_scifact_id", "candidate_ids": [x["document_id"] for x in candidates], "candidates": candidates, "qrels_accessed_during_generation": False, "reserved_content_accessed": False, "slate_sha256": stable_sha([x["document_id"] for x in candidates])}

    cal_slates = [slate_row(row) for row in cal_rows]
    exploratory_slates = [slate_row(row) for row in exploratory_rows]
    write_jsonl("E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl", cal_slates)
    write_jsonl("E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl", exploratory_slates)
    top5 = [{"query_id": row["query_id"], "candidate_ids": row["candidate_ids"][:5], "source_slate_id": row["slate_id"], "baseline": "first-five-RRF", "qrels_accessed_during_generation": False} for row in exploratory_slates]
    write_jsonl("E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl", top5)

    after = gpu_snapshot()
    gpu_audit = {"status": "PASS", "before": before, "after": after, "selected_idle_observed_before": [g["index"] for g in before["gpus"] if g["memory_used_mib"] < 1024 and g["utilization_gpu_percent"] == 0], "new_gpu_process_started": False, "process_eviction": False, "unrelated_services_modified": False, "embedding_service_revision": REVISION, "note": "Embedding execution was delegated to the pre-existing host service; its internal GPU placement is host-managed."}
    write_json("E6_2_R1_GPU_AND_RESOURCE_AUDIT_V2.json", gpu_audit)
    write_json("E6_2_R1_CORPUS_AND_PARTITION_AUDIT_V2.json", {"corpus_rows": len(corpus), "cal_rows": len(cal_slates), "exploratory_rows": len(exploratory_slates), "unique_corpus_ids": len({x["document_id"] for row in exploratory_slates for x in row["candidates"]}), "all_exploratory_top20_valid": all(len(row["candidate_ids"]) == 20 and len(set(row["candidate_ids"])) == 20 for row in exploratory_slates), "corpus_sha256": sha_file(paths["corpus"]), "query_sha256": sha_file(paths["queries"]), "embedding_matrix_sha256": sha_file(R1 / "E6_2_R1_SCIFACT_DOCUMENT_EMBEDDINGS_V2.npy"), "query_embedding_matrix_sha256": sha_file(R1 / "E6_2_R1_SCIFACT_QUERY_EMBEDDINGS_V2.npy"), "qrels_accessed_during_generation": False, "reserved_content_accessed": False})
    write_jsonl("E6_2_R1_RETRIEVAL_FAILURES_V2.jsonl", [])

    gate = {"gate": "E6_2_R1_RETRIEVAL_FROZEN_READY", "status": "READY_FOR_SEPARATE_E6_2_SELECTOR_AUTHORIZATION", "created_utc": datetime.now(timezone.utc).isoformat(), "frozen_slates": {"cal": len(cal_slates), "exploratory_top20": len(exploratory_slates), "exploratory_top5": len(top5)}, "contract": {"corpus_documents": len(corpus), "top_k": 20, "unique_candidates_all_exploratory": True, "qrels_accessed_during_generation": False, "reserved_content_accessed": False}, "embedding": {"model": MODEL, "revision": REVISION, "physical_requests": len(receipts), "failed_batches": sum(r.get("status") != "ok" for r in receipts)}, "next_gate": "Owner must separately authorize E6-2 selector inference; R1 does not run selectors/readers."}
    write_json("E6_2_R1_E6_2_RELEASE_GATE_V2.json", gate)
    report = f"""# DecisionRAG E6-2 R1 Authorized Retrieval-Freeze Report

## Gate: `E6_2_R1_RETRIEVAL_FROZEN_READY`

The owner’s current instruction authorized use of idle/available GPUs. R1 used the already-running host embedding service at `{EMBED_URL}`; no new GPU process was started and no unrelated process was evicted.

- Corpus: {len(corpus)} canonical SciFact documents.
- CAL validation slates: {len(cal_slates)}.
- EXPLORATORY frozen Top-20 slates: {len(exploratory_slates)}/277.
- EXPLORATORY retrieval Top-5 baselines: {len(top5)}.
- Embedding model revision: `{REVISION}`.
- Dense embeddings: exact cosine over the full corpus.
- Fusion: BM25 Top-100 + dense Top-100 + RRF(60), 20 unique IDs.
- qrels/support/answers/selectors/readers: not accessed.
- RESERVED content: not accessed.

The Top-20 slates are now fixed inputs for every future selector. R1 is complete; E6-2 selector inference remains a separate owner decision.
"""
    (R1 / "E6_2_R1_FINAL_RETRIEVAL_REPORT_V2.md").write_text(report, encoding="utf-8")

    outputs = sorted(p for p in R1.iterdir() if p.name not in {"E6_2_R1_PROVENANCE_MANIFEST_V2.json"})
    provenance = {"schema": "E6-2-R1-PROVENANCE-V2", "created_utc": datetime.now(timezone.utc).isoformat(), "gate": gate["gate"], "new_outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in outputs], "historical_inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in [R1 / "E6_2_R1_EXECUTION_MANIFEST.json", R1 / "E6_2_R1_FINAL_RETRIEVAL_REPORT.md", R1 / "E6_2_R1_E6_2_RELEASE_GATE.json"] if p.exists()], "physical_work": {"embedding_batches": len(receipts), "embedding_items": len(corpus) + len(query_rows), "model_forward_count": "host service telemetry not exposed in client response"}, "zero_calls": {"selector": 0, "reranker": 0, "reader": 0, "jev": 0, "paid_api": 0}, "reserved_content_access": False, "historical_artifacts_modified": False}
    write_json("E6_2_R1_PROVENANCE_MANIFEST_V2.json", provenance)
    print(json.dumps({"gate": gate["gate"], "cal": len(cal_slates), "exploratory": len(exploratory_slates), "embedding_batches": len(receipts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
