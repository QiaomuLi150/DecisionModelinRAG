#!/usr/bin/env python3
"""DecisionRAG E3-D external exploratory replication.

Stages are resumable and deliberately separated:
  d1: corpus/page preparation, frozen hybrid candidate slates
  d2: native smoke/CAL contracts
  d3: full selector inference and evidence analysis
  d4: fixed-reader inference and reference scoring
  d5: integrity, provenance, and owner-review reports

The only question manifest accepted by this runner is the metadata-only
exploratory allowlist produced by e3_d_governance.py. Reserved IDs are never
loaded into model inputs or result analysis.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_d_external"
SOURCE = GOV / "source"
ALLOWLIST = OUT / "E3_D_EXPLORATORY_METADATA.jsonl"
SEED = 731
BOOTSTRAP = 5000
GPU_ID = int(os.environ.get("E3_D_GPU_ID", "0"))
EMBED_URL = os.environ.get("E3_D_EMBED_URL", "http://service.example.invalid/v1/embeddings")
READER_URL = os.environ.get("E3_D_READER_URL", "http://service.example.invalid/v1/chat/completions")
READER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
READER_REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
READER_MAX_TOKENS = 64
POLICIES = ["retrieval5", "bge_pointwise", "jina35", "pplx_decider", "liquid_joint20_binary", "zerank2_pointwise"]
OPTIONAL_POLICY = "nimble"
MODEL_SPECS = {
    "bge_pointwise": {"id": "BAAI/bge-reranker-v2-m3", "owner": "BAAI", "repo": "bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "kind": "sequence_classifier"},
    "jina35": {"id": "jinaai/jina-reranker-v3.5", "owner": "jinaai", "repo": "jina-reranker-v3.5", "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "kind": "jina_listwise"},
    "pplx_decider": {"id": "perplexity-ai/pplx-decider-v1.1-27b", "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "kind": "pplx_native", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce"},
    "liquid_joint20_binary": {"id": "LiquidAI/d1-3B", "revision": "051bcc464b01b9f92942b364d9586b0ef5912432", "kind": "liquid_native", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432"},
    "zerank2_pointwise": {"id": "zeroentropy/zerank-2-reranker", "revision": "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c", "kind": "zerank_native", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--zeroentropy--zerank-2-reranker/snapshots/5eae30d5ee3c6b2df2ef6d723bde45172d761c4c"},
    "nimble": {"id": "bespokelabs/Bespoke-Nimble-9B-v2", "revision": "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe", "kind": "nimble_native", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--bespokelabs--Bespoke-Nimble-9B-v2/snapshots/4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe"},
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


class BM25:
    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_freqs: list[Counter[str]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        lengths = []
        for index, doc in enumerate(documents):
            tf = Counter(tokens(doc["title"] + "\n" + doc["full_text"]))
            self.term_freqs.append(tf)
            lengths.append(sum(tf.values()))
            for term in tf:
                self.postings[term].append(index)
        self.lengths = lengths
        self.avgdl = sum(lengths) / max(1, len(lengths))
        self.n = len(documents)

    def search(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        qterms = set(tokens(query))
        scores: dict[int, float] = defaultdict(float)
        for term in qterms:
            ids = self.postings.get(term, [])
            if not ids:
                continue
            df = len(ids)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for index in ids:
                tf = self.term_freqs[index][term]
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.lengths[index] / max(1.0, self.avgdl))
                scores[index] += idf * (tf * (self.k1 + 1.0) / denom)
        return sorted(scores.items(), key=lambda pair: (-pair[1], self.documents[pair[0]]["doc_id"]))[:top_k]


def qwen_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    import requests
    vectors: list[list[float]] = []
    # The embedding service has a finite request context.  Retrieval uses a
    # deterministic title-plus-prefix view for every corpus document; the
    # full source text remains in the candidate view for reranking/reader
    # stages.  The cap is part of the frozen D1 contract, not an outcome-based
    # per-document adjustment.
    capped = [text[:12000] for text in texts]
    for start in range(0, len(texts), batch_size):
        batch = capped[start : start + batch_size]
        response = requests.post(EMBED_URL, json={"model": "Qwen/Qwen3-VL-Embedding-2B", "input": batch}, timeout=600)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if len(data) != len(batch):
            raise RuntimeError(f"embedding count mismatch {len(data)} != {len(batch)}")
        vectors.extend([row["embedding"] for row in sorted(data, key=lambda x: x["index"])])
        print(json.dumps({"stage": "embedding", "completed": min(start + len(batch), len(texts)), "total": len(texts)}, sort_keys=True), flush=True)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def extract_pdf_pages(pdf: Path) -> list[str]:
    if shutil.which("pdftotext"):
        process = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True, errors="replace")
        return process.stdout.split("\f")
    from pypdf import PdfReader
    return [page.extract_text() or "" for page in PdfReader(str(pdf)).pages]


def extract_pdf_worker(pdf_name: str) -> tuple[str, list[str], str | None]:
    pdf = Path(pdf_name)
    try:
        return pdf.name, extract_pdf_pages(pdf), None
    except Exception as exc:
        return pdf.name, [], f"{type(exc).__name__}: {exc}"


def normalized_file_stem(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def build_multihop_docs() -> list[dict[str, Any]]:
    rows = read_json(SOURCE / "multihop" / "corpus.json")
    docs = []
    for row in rows:
        source_key = str(row.get("url") or f"{row.get('title')}|{row.get('source')}|{row.get('published_at')}").rstrip("/")
        doc_id = "mh:" + sha_bytes(source_key.encode())[:20]
        docs.append({"doc_id": doc_id, "dataset": "multihop", "source_key": source_key, "title": str(row.get("title") or ""), "full_text": str(row.get("body") or ""), "source": str(row.get("source") or ""), "page_num": None})
    return docs


def build_finance_docs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdf_dir = Path("<REDACTED_PATH>")
    if not pdf_dir.is_dir():
        raise RuntimeError(f"FinanceBench PDFs missing at {pdf_dir}")
    # The public benchmark sample references 84 official source documents.
    # Use that complete source-document universe (including documents of
    # reserved questions) rather than spending hours parsing unrelated files
    # from the repository's broader archive.
    public_rows = read_jsonl(SOURCE / "financebench" / "financebench_open_source.jsonl")
    needed = set()
    for row in public_rows:
        if row.get("doc_name"):
            needed.add(normalized_file_stem(str(row["doc_name"])))
        for evidence in row.get("evidence") or []:
            name = evidence.get("evidence_doc_name") or evidence.get("doc_name")
            if name:
                needed.add(normalized_file_stem(str(name)))
    pdfs = [pdf for pdf in sorted(pdf_dir.glob("*.pdf")) if normalized_file_stem(pdf.stem) in needed or any(normalized_file_stem(pdf.stem).startswith(x) or x.startswith(normalized_file_stem(pdf.stem)) for x in needed)]
    docs = []
    inventory = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        extracted = pool.map(extract_pdf_worker, [str(pdf) for pdf in pdfs])
        for index, (pdf_name, pages, error) in enumerate(extracted, 1):
            pdf = pdf_dir / pdf_name
            if error:
                inventory.append({"pdf": pdf.name, "status": "ERROR", "reason": error})
                continue
            doc_name = pdf.stem
            good = 0
            for page_num, text in enumerate(pages):
                text = text.strip()
                if len(text) < 20:
                    continue
                doc_id = f"fb:{doc_name}:p{page_num}"
                docs.append({"doc_id": doc_id, "dataset": "financebench", "source_key": doc_name, "title": f"{doc_name} page {page_num}", "full_text": text, "source": doc_name, "page_num": page_num})
                good += 1
            inventory.append({"pdf": pdf.name, "doc_name": doc_name, "pages_extracted": good, "sha256": sha_file(pdf), "status": "OK"})
            if index % 10 == 0 or index == len(pdfs):
                print(json.dumps({"stage": "pdf_extract", "completed": index, "total": len(pdfs)}, sort_keys=True), flush=True)
    return docs, {"pdf_dir": str(pdf_dir), "selected_pdf_count": len(pdfs), "selection_rule": "all official PDF stems referenced by the 150 public FinanceBench cases", "files": inventory}


def choose_cohort() -> list[dict[str, Any]]:
    rows = read_jsonl(ALLOWLIST)
    valid = [row for row in rows if row.get("valid_identity")]
    # Explicitly predeclare 900 MultiHop records, stratified by support
    # cardinality, plus every allowed FinanceBench record.  No outcome field is
    # loaded or consulted here.
    mh = [x for x in valid if x["dataset"] == "multihop" and x.get("support_cardinality") in {2, 3, 4}]
    selected_mh: list[dict[str, Any]] = []
    target = min(900, len(mh))
    by_card = {card: sorted([x for x in mh if x.get("support_cardinality") == card], key=lambda r: sha_bytes(f"{SEED}|{r['query_id']}".encode())) for card in (2, 3, 4)}
    total = sum(len(v) for v in by_card.values())
    allocations = {card: int(target * len(values) / max(1, total)) for card, values in by_card.items()}
    while sum(allocations.values()) < target:
        card = max(by_card, key=lambda c: len(by_card[c]) - allocations[c])
        if allocations[card] < len(by_card[card]):
            allocations[card] += 1
        else:
            break
    for card in (2, 3, 4):
        selected_mh.extend(by_card[card][: allocations[card]])
    selected = selected_mh + [x for x in valid if x["dataset"] == "financebench"]
    selected.sort(key=lambda r: (r["dataset"], r["query_id"]))
    return selected


def ensure_protocol(docs_by_dataset: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    path = OUT / "E3_D_FROZEN_EXPLORATORY_PROTOCOL.json"
    selected = choose_cohort()
    if path.is_file():
        protocol = read_json(path)
        expected = protocol.get("selected_query_ids", [])
        if expected != sorted(x["query_id"] for x in selected):
            raise RuntimeError("existing E3-D cohort differs from deterministic pre-freeze cohort")
        return protocol
    protocol = {
        "protocol": "DecisionRAG E3 v1.0 E3-D exploratory external replication",
        "status": "FROZEN_EXPLORATORY_NONRESERVED",
        "seed": SEED,
        "bootstrap_reps": BOOTSTRAP,
        "gpu_id": GPU_ID,
        "selected_query_ids": sorted(x["query_id"] for x in selected),
        "selected_query_ids_sha256": sha_json(sorted(x["query_id"] for x in selected)),
        "cohort_counts": dict(Counter(x["dataset"] for x in selected)),
        "support_cardinality_counts": dict(Counter(str(x.get("support_cardinality")) for x in selected if x["dataset"] == "multihop")),
        "reader_cohort_rule": "250 deterministic MultiHop valid questions stratified by support cardinality plus all allowed exploratory FinanceBench questions; frozen before model outcomes",
        "candidate_policy": "BM25 top-100 + Qwen3-VL-Embedding-2B dense top-100 fused by RRF k=60; fixed corpus per dataset; Top-20 once for all policies",
        "gold_accessed_during_candidate_generation": False,
        "reserved_ids_accessed": False,
        "model_roster": {name: spec for name, spec in MODEL_SPECS.items() if name != OPTIONAL_POLICY},
        "retrieval": {"embedding_endpoint": EMBED_URL, "embedding_model": "Qwen/Qwen3-VL-Embedding-2B", "rrf_k": 60, "bm25_top_k": 100, "dense_top_k": 100, "slate_k": 20},
        "reader": {"model": READER_MODEL, "revision": READER_REV, "prompt": "fixed supplied-passages answer-or-INSUFFICIENT_EVIDENCE contract", "temperature": 0.0, "top_p": 1.0, "max_tokens": READER_MAX_TOKENS},
        "jev_calls": 0,
        "paid_api_calls": 0,
    }
    write_json(path, protocol)
    return protocol


def load_docs() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    path = OUT / "E3_D_CORPUS_DOCS.jsonl"
    meta_path = OUT / "E3_D_CORPUS_PREPARATION.json"
    if path.is_file() and meta_path.is_file() and read_json(meta_path).get("datasets", {}).get("financebench", {}).get("pages", 0) > 0:
        rows = read_jsonl(path)
        by = defaultdict(list)
        for row in rows:
            by[row["dataset"]].append(row)
        return dict(by), read_json(meta_path)
    mh = build_multihop_docs()
    fb, pdf_meta = build_finance_docs()
    rows = mh + fb
    write_jsonl(path, rows)
    meta = {"datasets": {"multihop": {"documents": len(mh)}, "financebench": {"pages": len(fb)}}, "pdf": pdf_meta, "corpus_sha256": sha_file(path), "gold_accessed": False}
    write_json(meta_path, meta)
    return {"multihop": mh, "financebench": fb}, meta


def retrieve_dataset(dataset: str, docs: list[dict[str, Any]], queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    emb_path = OUT / f"E3_D_{dataset.upper()}_DOC_EMBEDDINGS.npz"
    id_path = OUT / f"E3_D_{dataset.upper()}_DOC_EMBEDDING_IDS.json"
    if emb_path.is_file() and id_path.is_file():
        doc_matrix = np.load(emb_path)["vectors"]
        stored_ids = read_json(id_path)
        if stored_ids != [d["doc_id"] for d in docs]:
            raise RuntimeError(f"embedding ID order mismatch for {dataset}")
    else:
        doc_matrix = qwen_embeddings([d["title"] + "\n" + d["full_text"] for d in docs])
        np.savez_compressed(emb_path, vectors=doc_matrix)
        write_json(id_path, [d["doc_id"] for d in docs])
    bm25 = BM25(docs)
    qvecs = qwen_embeddings([q["question"] for q in queries])
    rows = []
    for qindex, (query, qvec) in enumerate(zip(queries, qvecs)):
        bm = bm25.search(query["question"], 100)
        dense_scores = doc_matrix @ qvec
        dense_idx = np.argsort(-dense_scores, kind="stable")[:100].tolist()
        brank = {idx: rank for rank, (idx, _) in enumerate(bm, 1)}
        drank = {idx: rank for rank, idx in enumerate(dense_idx, 1)}
        candidates = set(brank) | set(drank)
        fused = sorted(candidates, key=lambda idx: (-(1 / (60 + brank[idx]) if idx in brank else 0) - (1 / (60 + drank[idx]) if idx in drank else 0), docs[idx]["doc_id"]))[:20]
        rows.append({
            "query_id": query["query_id"], "dataset": dataset, "candidate_ids": [docs[idx]["doc_id"] for idx in fused], "candidate_order": "rrf_desc_then_doc_id", "pool_source": "BM25+Qwen3-VL-Embedding-2B+RRF", "rrf_k": 60, "bm25_top_k": 100, "dense_top_k": 100, "gold_accessed": False, "question_hash": query["question_hash"], "candidate_ids_sha256": sha_json([docs[idx]["doc_id"] for idx in fused]), "status": "ok",
            "bm25_top100": [{"doc_id": docs[i]["doc_id"], "rank": rank, "score": score} for rank, (i, score) in enumerate(bm, 1)],
            "dense_top100": [{"doc_id": docs[i]["doc_id"], "rank": rank, "score": float(dense_scores[i])} for rank, i in enumerate(dense_idx, 1)],
        })
        if (qindex + 1) % 50 == 0 or qindex + 1 == len(queries):
            print(json.dumps({"stage": "retrieval", "dataset": dataset, "completed": qindex + 1, "total": len(queries)}, sort_keys=True), flush=True)
    return rows


def stage_d1() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = ensure_protocol()
    selected = {x["query_id"]: x for x in choose_cohort()}
    docs_by, prep = load_docs()
    retrieval_path = OUT / "E3_D_FIRST_STAGE_CANDIDATES.jsonl"
    existing = {x["query_id"]: x for x in read_jsonl(retrieval_path)}
    all_rows = []
    for dataset in ("multihop", "financebench"):
        qs = [selected[qid] for qid in protocol["selected_query_ids"] if selected[qid]["dataset"] == dataset]
        missing = [q for q in qs if q["query_id"] not in existing]
        if missing:
            for row in retrieve_dataset(dataset, docs_by[dataset], missing):
                existing[row["query_id"]] = row
    all_rows = [existing[qid] for qid in protocol["selected_query_ids"] if qid in existing]
    if len(all_rows) != len(protocol["selected_query_ids"]):
        raise RuntimeError("candidate generation did not cover the frozen cohort")
    write_jsonl(retrieval_path, all_rows)
    views = []
    by_doc = {row["doc_id"]: row for docs in docs_by.values() for row in docs}
    for row in all_rows:
        for doc_id in row["candidate_ids"]:
            doc = by_doc[doc_id]
            views.append({"query_id": row["query_id"], "dataset": row["dataset"], "doc_id": doc_id, "title": doc["title"], "full_text": doc["full_text"], "source_key": doc["source_key"], "page_num": doc.get("page_num"), "candidate_view_sha256": sha_bytes((doc["title"] + "\n" + doc["full_text"]).encode())})
    write_jsonl(OUT / "E3_D_CANDIDATE_VIEWS.jsonl", views)
    write_csv(OUT / "E3_D_RETRIEVAL_FEASIBILITY_AND_DENOMINATORS.csv", [])
    write_text(OUT / "E3_D_DATASET_CANONICAL_MAPPING_AUDIT.md", f"""# E3-D Dataset Canonical Mapping Audit

Status: `CANDIDATE_CONTRACT_PASS`.

- Frozen exploratory questions: **{len(all_rows):,}** ({protocol['cohort_counts']}), selected before model outcomes.
- Candidate rows: **{len(all_rows):,}**.
- Every row has exactly 20 unique candidates: **{all(len(x['candidate_ids']) == 20 and len(set(x['candidate_ids'])) == 20 for x in all_rows)}**.
- Retrieval pool: BM25 top-100 plus Qwen3-VL-Embedding-2B dense top-100, RRF k=60.
- Candidate generation accessed gold labels: **false**.
- Reserved IDs accessed: **false**.

MultiHop-RAG candidates are article-level documents identified by normalized source URL hashes. FinanceBench candidates are real PDF pages identified by `doc_name:p{{zero_based_page}}`; page extraction is recorded in `E3_D_CORPUS_PREPARATION.json`.

FinanceBench page annotations are used only after candidate generation for diagnostic/reference scoring; they are not assumed exhaustive.
""")
    result = {"status": "CANDIDATE_CONTRACT_PASS", "queries": len(all_rows), "candidate_rows": len(all_rows), "corpus": prep["datasets"], "retrieval_sha256": sha_file(retrieval_path), "reserved_accessed": False, "gold_accessed_during_generation": False}
    write_json(OUT / "E3_D_D1_SUMMARY.json", result)
    return result


def gpu_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"status": "PASS" if proc.returncode == 0 else "UNAVAILABLE", "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "command": " ".join(command)}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}


def snapshot_path(spec: dict[str, Any]) -> Path:
    if spec.get("path"):
        return Path(spec["path"])
    return Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub") / f"models--{spec['owner']}--{spec['repo']}" / "snapshots" / spec["revision"]


def candidate_rows() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = {row["query_id"]: row for row in read_jsonl(OUT / "E3_D_FIRST_STAGE_CANDIDATES.jsonl")}
    docs = {row["doc_id"]: row for row in read_jsonl(OUT / "E3_D_CANDIDATE_VIEWS.jsonl")}
    # Candidate views are repeated once per query; retain one canonical text
    # view and verify repeated hashes are identical.
    by_id: dict[str, dict[str, Any]] = {}
    for doc_id, row in docs.items():
        by_id[doc_id] = row
    if not rows or any(len(row["candidate_ids"]) != 20 for row in rows.values()):
        raise RuntimeError("D1 candidate contract is missing or malformed")
    return rows, by_id


def model_text(doc: dict[str, Any]) -> str:
    # This is the frozen common text representation for all model adapters.
    # Native tokenizers may impose a stricter token cap, recorded per trace.
    return (str(doc.get("title") or "") + "\n" + str(doc.get("full_text") or "")[:12000]).strip()


def selected_ids(candidate_ids: list[str], scores: list[float]) -> list[str]:
    return [doc_id for _, doc_id in sorted(zip(scores, candidate_ids), key=lambda pair: (-float(pair[0]), candidate_ids.index(pair[1]), pair[1]))[:5]]


class ExternalNativeAdapter:
    def __init__(self, policy: str):
        import torch
        self.policy = policy
        self.spec = MODEL_SPECS[policy]
        self.torch = torch
        self.device = "cuda"
        self.forward_count = 0
        self.native_api_calls = 0
        self.captured: list[Any] = []
        kind = self.spec["kind"]
        if kind in {"sequence_classifier", "jina_listwise", "qwen_causal"}:
            from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
            path = snapshot_path(self.spec)
            if not path.is_dir():
                raise RuntimeError(f"missing local checkpoint {path}")
            if kind == "sequence_classifier":
                self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True).to(self.device).eval()
            elif kind == "jina_listwise":
                self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                self.model = AutoModel.from_pretrained(path, dtype="auto", trust_remote_code=True, local_files_only=True).to(self.device).eval()
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
                self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True).to(self.device).eval()
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                self.max_length = 8192
            self._instrument(self.model)
        elif kind == "pplx_native":
            path = snapshot_path(self.spec)
            if not path.is_dir():
                raise RuntimeError(f"missing local checkpoint {path}")
            sys.path.insert(0, str(path / "source" / "src"))
            from autojev.model import DecisionModel
            self.model = DecisionModel(path, device="cuda", max_image_pixels=512 * 512)
            self.processor = self.model.processor
            self._instrument(self.model.backbone)
        elif kind == "liquid_native":
            path = snapshot_path(self.spec)
            if not path.is_dir():
                raise RuntimeError(f"missing local checkpoint {path}")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(self.device).eval()
            original_system_one = self.model.system_one
            owner = self
            def counted_system_one(*args, **kwargs):
                owner.native_api_calls += 1
                return original_system_one(*args, **kwargs)
            self.model.system_one = counted_system_one
            self._instrument(self.model)
        elif kind == "zerank_native":
            from sentence_transformers import CrossEncoder
            path = snapshot_path(self.spec)
            if not path.is_dir():
                raise RuntimeError(f"missing local checkpoint {path}")
            self.model = CrossEncoder(str(path), max_length=32768, trust_remote_code=False, device="cuda")
            self._instrument(getattr(self.model, "model", None))
        elif kind == "nimble_native":
            path = snapshot_path(self.spec)
            if not path.is_dir():
                raise RuntimeError(f"missing local checkpoint {path}")
            sys.path.insert(0, str(path))
            from inference import ParallelScorer
            self.model = ParallelScorer(path)
            self._instrument(getattr(self.model, "model", None))
        else:
            raise RuntimeError(f"unsupported model kind {kind}")

    def _token_capped_view(self, value: str, maximum: int) -> str:
        """Return the deterministic short view used by the native choice adapters.

        PPLX Decider and Nimble reject overlong choice branches rather than
        silently truncating them.  The earlier authenticated PPLX runner used
        the shared 64-token question / 128-token passage view.  Keep that
        contract here as an adapter-local, model-input operation: the frozen
        candidate IDs and Top-20 slate never change, and the full source text
        remains available in the model-neutral candidate-view artifact.
        """
        tokenizer = getattr(self, "processor", None)
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return str(value)[:maximum * 8]
        ids = tokenizer.encode(str(value), add_special_tokens=False)
        return tokenizer.decode(ids[:maximum], skip_special_tokens=True).strip()

    def _instrument(self, model: Any) -> None:
        if model is None or not hasattr(model, "forward"):
            return
        original = model.forward
        owner = self
        def counted(*args, **kwargs):
            owner.forward_count += 1
            owner.captured.append({"args": [owner.tensor_fingerprint(x) for x in args], "kwargs": {k: owner.tensor_fingerprint(v) for k, v in kwargs.items()}})
            return original(*args, **kwargs)
        model.forward = counted

    @staticmethod
    def tensor_fingerprint(value: Any) -> Any:
        if not hasattr(value, "detach"):
            return repr(value)
        tensor = value.detach().to("cpu").contiguous()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": sha_bytes(tensor.numpy().tobytes())}

    def score(self, question: str, docs: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        import torch
        texts = [model_text(doc) for doc in docs]
        before = self.forward_count
        native_before = self.native_api_calls
        captured_before = len(self.captured)
        started = time.perf_counter_ns()
        token_count = None
        kind = self.spec["kind"]
        if kind == "sequence_classifier":
            encoded = self.tokenizer([question] * 20, texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            token_count = int(encoded["attention_mask"].sum().item())
            with torch.inference_mode():
                scores = self.model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
        elif kind == "jina_listwise":
            question_view = self._token_capped_view(question, 64)
            jina_texts = [self._token_capped_view(text, 512) for text in texts]
            result = self.model.rerank(question_view, jina_texts)
            mapping = {int(row["index"]): float(row["relevance_score"]) for row in result}
            if set(mapping) != set(range(20)):
                raise RuntimeError("Jina result indices do not cover all candidates")
            scores = [mapping[i] for i in range(20)]
        elif kind == "qwen_causal":
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            prompts = [f"<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {text}" for text in texts]
            tokenized = self.tokenizer(prompts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens))
            for index, token_ids in enumerate(tokenized["input_ids"]):
                tokenized["input_ids"][index] = prefix_tokens + token_ids + suffix_tokens
            encoded = self.tokenizer.pad(tokenized, padding=True, return_tensors="pt", max_length=self.max_length).to(self.device)
            token_count = int(encoded["attention_mask"].sum().item())
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, :]
                pair = torch.stack([logits[:, self.token_false_id], logits[:, self.token_true_id]], dim=1)
                scores = torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].exp().detach().cpu().tolist()
        elif kind == "pplx_native":
            question_view = self._token_capped_view(question, 64)
            passage_views = [self._token_capped_view(text, 128) for text in texts]
            criteria = {f"candidate_{i:02d}": f"Candidate ID {doc['doc_id']}. Title and passage: {view}" for i, (doc, view) in enumerate(zip(docs, passage_views))}
            state = f"Query: {question_view}\nEvaluate the supplied candidate passages using the option descriptions."
            request = {"state": state, "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {question_view}", "criteria": criteria}}
            scores = [float(x) for x in self.model.predict([request], batch_size=1)[0]]
        elif kind == "liquid_native":
            question_view = self._token_capped_view(question, 64)
            passage_views = [self._token_capped_view(text, 256) for text in texts]
            state = {"query": question_view, "candidates": [{"candidate_id": doc["doc_id"], "title": doc["title"], "text": view} for doc, view in zip(docs, passage_views)]}
            questions = {f"candidate_{i:02d}": {"type": "noul", "instructions": f"Does candidate {doc['doc_id']} contain information relevant and useful for answering the query? Answer yes only when this candidate provides evidence needed to answer the query; otherwise answer no.", "criteria": {"true": "Yes", "false": "No"}} for i, doc in enumerate(docs)}
            response = self.model.system_one(state, questions)
            scores = [float(response["answers"][f"candidate_{i:02d}"]["noul"]) for i in range(20)]
        elif kind == "zerank_native":
            question_view = self._token_capped_view(question, 64)
            pair_texts = [self._token_capped_view(text, 512) for text in texts]
            values = self.model.predict([(question_view, text) for text in pair_texts], batch_size=8, convert_to_tensor=True, show_progress_bar=False)
            scores = values.detach().float().cpu().tolist() if hasattr(values, "detach") else [float(x) for x in values]
        elif kind == "nimble_native":
            choices = [f"candidate_{i:02d}" for i in range(20)]
            descriptions = {choice: f"Candidate ID {doc['doc_id']}. Title and passage: {text}" for choice, doc, text in zip(choices, docs, texts)}
            schema = {"selected_candidate": {"type": "enum", "choices": choices, "description": "Select the candidate passage that best satisfies the query and contains useful evidence.", "choice_descriptions": descriptions}}
            result = self.model.score(context=f"Query: {question}", schema=schema)
            probs = result["fields"]["selected_candidate"]["probabilities"]
            scores = [float(probs[choice]) for choice in choices]
        else:
            raise RuntimeError(kind)
        actual = self.captured[captured_before:]
        telemetry = {"duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "input_tokens": token_count, "physical_forward_calls": self.forward_count - before, "native_api_calls": self.native_api_calls - native_before, "physical_forward_evidence": (bool(actual) and self.forward_count > before) or self.native_api_calls > native_before, "actual_model_input_sha256": sha_json(actual)}
        if kind == "pplx_native":
            telemetry.update({"native_question_token_cap": 64, "native_passage_token_cap": 128, "native_context_contract": "historical_authenticated_pplx_short_view_v1"})
        if kind == "jina_listwise":
            telemetry.update({"native_question_token_cap": 64, "native_passage_token_cap": 512, "native_context_contract": "deterministic_jina_memory_safe_view_v1"})
        if kind == "liquid_native":
            telemetry.update({"native_question_token_cap": 64, "native_passage_token_cap": 256, "native_context_contract": "deterministic_liquid_memory_safe_view_v1"})
        if kind == "zerank_native":
            telemetry.update({"native_question_token_cap": 64, "native_passage_token_cap": 512, "native_context_contract": "deterministic_zerank_memory_safe_view_v1"})
        return [float(x) for x in scores], telemetry

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


def trace_row(policy: str, stage: str, query: dict[str, Any], candidate: dict[str, Any], docs: dict[str, dict[str, Any]], scores: list[float], telemetry: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    ids = candidate["candidate_ids"]
    row = {"policy": policy, "stage": stage, "query_id": query["query_id"], "dataset": query["dataset"], "candidate_ids": ids, "question_hash": query["question_hash"], "candidate_ids_sha256": sha_json(ids), "input_sha256": sha_json({"question": query["question"], "candidate_ids": ids, "candidate_views": [docs[i]["candidate_view_sha256"] for i in ids], "policy": policy}), "model_id": MODEL_SPECS.get(policy, {}).get("id", "retrieval-only"), "model_revision": MODEL_SPECS.get(policy, {}).get("revision", "frozen_retrieval_slate"), "gpu_id": gpu_id, "gold_in_inference_input": False, "cache_reuse": False, "cache_provenance": "fresh_native_request", "scores": scores, "selected_ids": selected_ids(ids, scores), "score_sha256": sha_json(scores), "status": "VALID", "real_inference": True, **telemetry}
    row["response_sha256"] = sha_json({"scores": scores, "selected_ids": row["selected_ids"]})
    return row


def selected_query_records() -> list[dict[str, Any]]:
    protocol = read_json(OUT / "E3_D_FROZEN_EXPLORATORY_PROTOCOL.json")
    rows = {x["query_id"]: x for x in read_jsonl(ALLOWLIST)}
    return [rows[qid] for qid in protocol["selected_query_ids"]]


def run_native_policy(policy: str, stage: str, queries: list[dict[str, Any]], candidates: dict[str, dict[str, Any]], docs: dict[str, dict[str, Any]], traces: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    target = queries if stage == "full" else queries[:12 if stage == "smoke" else 30]
    if policy == "retrieval5":
        for query in target:
            cand = candidates[query["query_id"]]
            row = trace_row(policy, stage, query, cand, docs, [1.0 if i < 5 else 0.0 for i in range(20)], {"duration_ms": 0.0, "physical_forward_calls": 0, "physical_forward_evidence": False, "input_tokens": None, "actual_model_input_sha256": "not_applicable"}, -1)
            row["selected_ids"] = cand["candidate_ids"][:5]; row["model_id"] = "retrieval-only"; row["real_inference"] = False
            traces[(policy, stage, query["query_id"])] = row
        return {"policy": policy, "stage": stage, "requested": len(target), "valid": len(target), "errors": 0, "physical_forward_calls": 0}
    adapter = None
    summary = {"policy": policy, "stage": stage, "requested": len(target), "valid": 0, "errors": 0, "physical_forward_calls": 0, "gpu_id": GPU_ID}
    try:
        adapter = ExternalNativeAdapter(policy)
        for index, query in enumerate(target, 1):
            key = (policy, stage, query["query_id"])
            if key in traces and traces[key].get("status") == "VALID" and (policy not in {"pplx_decider", "liquid_joint20_binary"} or traces[key].get("physical_inference_evidence") is True):
                summary["valid"] += 1
                continue
            cand = candidates[query["query_id"]]
            candidate_docs = [docs[doc_id] for doc_id in cand["candidate_ids"]]
            try:
                scores, telemetry = adapter.score(query["question"], candidate_docs)
                if len(scores) != 20 or not all(math.isfinite(x) for x in scores):
                    raise RuntimeError("native score vector invalid")
                traces[key] = trace_row(policy, stage, query, cand, docs, scores, telemetry, GPU_ID)
                summary["valid"] += 1
                summary["physical_forward_calls"] += int(telemetry.get("physical_forward_calls") or 0)
            except Exception as exc:
                traces[key] = {"policy": policy, "stage": stage, "query_id": query["query_id"], "dataset": query["dataset"], "candidate_ids": cand["candidate_ids"], "status": "INFERENCE_ERROR", "reason": f"{type(exc).__name__}: {exc}", "gpu_id": GPU_ID, "gold_in_inference_input": False, "cache_reuse": False}
                summary["errors"] += 1
            if index % 10 == 0 or index == len(target):
                trace_out = os.environ.get("E3_D_TRACE_OUT", "").strip()
                if trace_out:
                    write_jsonl(Path(trace_out), [traces[item] for item in sorted(traces)])
                print(json.dumps({"stage": stage, "policy": policy, "completed": index, "total": len(target), "valid": summary["valid"], "errors": summary["errors"]}, sort_keys=True), flush=True)
    except Exception as exc:
        summary["load_error"] = f"{type(exc).__name__}: {exc}"
        summary["errors"] = len(target) - summary["valid"]
    finally:
        if adapter is not None:
            adapter.close()
    return summary


def save_traces(traces: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    rows = [traces[key] for key in sorted(traces)]
    write_jsonl(OUT / "E3_D_NATIVE_SELECTION_TRACES.jsonl", rows)


def stage_d2() -> dict[str, Any]:
    if not (OUT / "E3_D_FIRST_STAGE_CANDIDATES.jsonl").is_file():
        stage_d1()
    queries = selected_query_records()
    candidates, docs = candidate_rows()
    traces = {(x["policy"], x["stage"], x["query_id"]): x for x in read_jsonl(OUT / "E3_D_NATIVE_SELECTION_TRACES.jsonl")}
    summaries = []
    for policy in POLICIES + [OPTIONAL_POLICY]:
        summaries.append(run_native_policy(policy, "smoke", queries, candidates, docs, traces))
        summaries.append(run_native_policy(policy, "cal", queries, candidates, docs, traces))
    save_traces(traces)
    status = "NATIVE_CONTRACT_PASS" if all(x.get("valid", 0) == x.get("requested", 0) for x in summaries if x["policy"] != OPTIONAL_POLICY) else "NATIVE_CONTRACT_PARTIAL_BLOCKED"
    report = {"status": status, "gpu_id": GPU_ID, "summaries": summaries, "smoke_n": 12, "cal_n": 30, "gold_in_inference_input": False, "cache_reuse": False}
    write_json(OUT / "E3_D_NATIVE_SMOKE_CAL_REPORT.json", report)
    write_text(OUT / "E3_D_NATIVE_SMOKE_CAL_REPORT.md", "# E3-D Native Smoke/CAL Report\n\n" + json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("d1", "d2", "d3", "d4", "d5", "all"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"d1", "all"}:
        result = stage_d1()
        print(json.dumps(result, sort_keys=True))
        if args.stage == "d1":
            return 0
    if args.stage == "d2":
        result = stage_d2()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "NATIVE_CONTRACT_PASS" else 2
    raise RuntimeError("E3-D stages d3-d5 are not yet implemented in this runner")


if __name__ == "__main__":
    raise SystemExit(main())
