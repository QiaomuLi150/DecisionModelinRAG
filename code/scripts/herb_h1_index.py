#!/usr/bin/env python3
"""Build and validate the dedicated HERB H1 retrieval corpus.

This is a deterministic H1 adapter.  It uses the existing Qwen embedding
service on the already-running service GPU and a new Qdrant collection.  It
never reads the protected gold sidecar and never modifies Track A/B/C/R2
collections or indexes.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/vendor/herb_h1/official_dataset_repo"
OUT = ROOT / "results/non_jev/herb_h1"
DB = OUT / "HERB_H1_CORPUS.sqlite3"
CHUNK_MANIFEST = OUT / "HERB_H1_CHUNK_MANIFEST.jsonl"
BUILD_REPORT = OUT / "HERB_H1_INDEX_BUILD_REPORT.json"
INTEGRITY_REPORT = OUT / "HERB_H1_INDEX_INTEGRITY_REPORT.json"
EMBED_PROBE = OUT / "HERB_H1_EMBEDDING_PROBE.json"
EMBED_PROGRESS = OUT / "HERB_H1_EMBEDDING_PROGRESS.json"
RETRIEVAL = OUT / "HERB_H1_RETRIEVAL_SLATES.jsonl"
RETRIEVAL_REPORT = OUT / "HERB_H1_RETRIEVAL_REPORT.json"
EMBED_URL = "http://service.example.invalid/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QDRANT_URL = "http://service.example.invalid"
COLLECTION = "decisionrag_h1_herb_qwen_embedding_r1"
EMBED_DIMENSION = 2048
CHUNK_SIZE = 384
OVERLAP = 64
BM25_TOP_K = 100
DENSE_TOP_K = 100
RRF_K = 60
SLATE_K = 20
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
WS_RE = re.compile(r"\S+", re.UNICODE)


def digest(value: Any) -> str:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def flat(value: Any) -> str:
    if isinstance(value, str):
        return normalize_text(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def artifact_text(product: str, source_type: str, item: dict[str, Any]) -> tuple[str, str]:
    """Return title and serving text, excluding product-level oracle fields."""
    original_id = str(item.get("id", "UNKNOWN"))
    if source_type == "slack":
        message = item.get("Message", {}).get("User", {})
        replies = item.get("ThreadReplies", [])
        title = f"Slack {original_id}"
        body = "\n".join([
            f"Product: {product}", f"Source type: {source_type}", f"Artifact ID: {original_id}",
            f"Channel: {item.get('Channel', {}).get('name', '')}",
            f"Channel ID: {item.get('Channel', {}).get('channelID', '')}",
            f"User: {message.get('userId', '')}", f"Timestamp: {message.get('timestamp', '')}",
            f"Message: {message.get('text', '')}", f"Thread replies: {flat(replies)}",
        ])
    elif source_type == "documents":
        title = str(item.get("type") or item.get("id") or "Document")
        body = "\n".join([
            f"Product: {product}", f"Source type: {source_type}", f"Artifact ID: {original_id}",
            f"Document type: {item.get('type', '')}", f"Date: {item.get('date', '')}",
            f"Author: {item.get('author', '')}", f"Link: {item.get('document_link', '')}",
            f"Content: {item.get('content', '')}", f"Feedback: {flat(item.get('feedback', []))}",
        ])
    elif source_type == "meeting_transcripts":
        title = f"Meeting {original_id}"
        body = "\n".join([
            f"Product: {product}", f"Source type: {source_type}", f"Artifact ID: {original_id}",
            f"Date: {item.get('date', '')}", f"Document type: {item.get('document_type', '')}",
            f"Participants: {flat(item.get('participants', []))}", f"Transcript: {item.get('transcript', '')}",
        ])
    elif source_type == "meeting_chats":
        title = f"Meeting chat {original_id}"
        body = "\n".join([
            f"Product: {product}", f"Source type: {source_type}", f"Artifact ID: {original_id}",
            f"Chat: {item.get('text', '')}",
        ])
    elif source_type == "urls":
        title = str(item.get("id") or item.get("link") or "URL")
        body = "\n".join([
            f"Product: {product}", f"Source type: {source_type}", f"Artifact ID: {original_id}",
            f"URL: {item.get('link', '')}", f"Description: {item.get('description', '')}",
        ])
    elif source_type == "prs":
        title = str(item.get("title") or item.get("id") or "Pull request")
        body = "\n".join([
            f"Product: {product}", f"Source type: {source_type}", f"Artifact ID: {original_id}",
            f"Title: {item.get('title', '')}", f"Summary: {item.get('summary', '')}",
            f"Link: {item.get('link', '')}", f"State: {item.get('state', '')}",
            f"Merged: {item.get('merged', '')}", f"Mergeable: {item.get('mergeable', '')}",
            f"Number: {item.get('number', '')}", f"User: {flat(item.get('user', {}))}",
            f"Reviews: {flat(item.get('reviews', []))}", f"Created at: {item.get('created_at', '')}",
        ])
    else:  # pragma: no cover - source types are frozen above
        raise ValueError(f"unsupported HERB source type: {source_type}")
    return title, normalize_text(body)


def make_chunks(text: str) -> list[tuple[int, str, int]]:
    tokens = WS_RE.findall(text)
    if not tokens:
        return [(0, "", 0)]
    step = CHUNK_SIZE - OVERLAP
    chunks: list[tuple[int, str, int]] = []
    seen: set[str] = set()
    start = 0
    while start < len(tokens):
        value = " ".join(tokens[start : start + CHUNK_SIZE])
        if value not in seen:
            chunks.append((len(chunks), value, min(CHUNK_SIZE, len(tokens) - start)))
            seen.add(value)
        if start + CHUNK_SIZE >= len(tokens):
            break
        start += step
    return chunks


def search_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def db_connect() -> sqlite3.Connection:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            internal_doc_id TEXT PRIMARY KEY,
            original_source_id TEXT NOT NULL,
            product TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_record_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            body_sha256 TEXT NOT NULL,
            text TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            UNIQUE(product, source_type, source_record_index)
        );
        CREATE TABLE IF NOT EXISTS chunks (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            internal_doc_id TEXT NOT NULL,
            original_source_id TEXT NOT NULL,
            product TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_record_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            text_sha256 TEXT NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(internal_doc_id) REFERENCES documents(internal_doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(internal_doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(search_text, chunk_id UNINDEXED, content='');
        """
    )
    conn.commit()


def build() -> dict[str, Any]:
    products = sorted((RAW / "products").glob("*.json"))
    conn = db_connect()
    init_db(conn)
    existing = {row[0] for row in conn.execute("SELECT internal_doc_id FROM documents")}
    rows: list[dict[str, Any]] = []
    source_counts: collections.Counter[str] = collections.Counter()
    chunk_counts: collections.Counter[str] = collections.Counter()
    started = time.perf_counter()
    for product_path in products:
        product = product_path.stem
        data = json.loads(product_path.read_text(encoding="utf-8"))
        for source_type in ("slack", "documents", "meeting_transcripts", "meeting_chats", "urls", "prs"):
            for index, item in enumerate(data.get(source_type, [])):
                internal_id = f"herb:{product}:{source_type}:{index:05d}"
                if internal_id in existing:
                    continue
                original_id = str(item.get("id", "UNKNOWN"))
                title, text = artifact_text(product, source_type, item)
                chunks = make_chunks(text)
                conn.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (internal_id, original_id, product, source_type, index, title, digest(text), text, len(chunks)),
                )
                for chunk_index, chunk_text, token_count in chunks:
                    chunk_id = f"{internal_id}#chunk-{chunk_index}"
                    row = {
                        "chunk_id": chunk_id, "internal_doc_id": internal_id,
                        "original_source_id": original_id, "product": product,
                        "source_type": source_type, "source_record_index": index,
                        "title": title, "chunk_index": chunk_index, "token_count": token_count,
                        "text_sha256": digest(chunk_text),
                    }
                    cursor = conn.execute(
                        "INSERT INTO chunks(chunk_id, internal_doc_id, original_source_id, product, source_type, source_record_index, title, chunk_index, token_count, text_sha256, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (chunk_id, internal_id, original_id, product, source_type, index, title, chunk_index, token_count, row["text_sha256"], chunk_text),
                    )
                    rowid = int(cursor.lastrowid)
                    fts_text = " ".join(search_tokens(f"{title} {chunk_text}"))
                    conn.execute("INSERT INTO chunks_fts(rowid, search_text, chunk_id) VALUES (?, ?, ?)", (rowid, fts_text, chunk_id))
                    rows.append(row)
                    chunk_counts[source_type] += 1
                source_counts[source_type] += 1
                existing.add(internal_id)
                if len(existing) % 250 == 0:
                    conn.commit()
    conn.commit()
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db_counts = dict(conn.execute("SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"))
    db_chunk_counts = dict(conn.execute("SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type"))
    conn.close()
    if rows:
        append_jsonl(CHUNK_MANIFEST, rows, "a" if CHUNK_MANIFEST.exists() else "w")
    summary = {
        "status": "PASS",
        "created_utc": time.time(),
        "product_files": len(products),
        "physical_documents": document_count,
        "chunks": chunk_count,
        "document_counts_by_source_type": db_counts,
        "chunk_counts_by_source_type": db_chunk_counts,
        "new_documents_this_run": sum(source_counts.values()),
        "new_chunks_this_run": len(rows),
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": OVERLAP,
        "db_sha256": digest(DB.read_bytes()),
        "chunk_manifest_sha256": digest(CHUNK_MANIFEST.read_bytes()) if CHUNK_MANIFEST.exists() else None,
        "oracle_fields_excluded": ["team", "customers", "answerable_questions", "unanswerable_questions", "ground_truth", "citations"],
        "gold_accessed": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(BUILD_REPORT, summary)
    return summary


def embedding_request(texts: list[str], timeout: int = 600) -> list[list[float]]:
    response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts}, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    values = [row["embedding"] for row in sorted(body["data"], key=lambda x: int(x["index"]))]
    if len(values) != len(texts) or any(len(v) != EMBED_DIMENSION for v in values):
        raise RuntimeError(f"embedding shape mismatch: {len(values)} rows, dimensions={sorted({len(v) for v in values})}")
    return values


def qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, timeout=600)


def probe(batch_size: int) -> dict[str, Any]:
    conn = db_connect()
    sample = conn.execute("SELECT title, text FROM chunks ORDER BY rowid LIMIT 2").fetchall()
    conn.close()
    if not sample:
        raise RuntimeError("build the corpus before probing embeddings")
    texts = [f"{title}\n{text}" for title, text in sample[:batch_size]]
    started = time.perf_counter()
    vectors = embedding_request(texts)
    result = requests.get("http://service.example.invalid/v1/models", timeout=30).json()
    out = {
        "status": "PASS", "model_requested": EMBED_MODEL, "service_identity": result,
        "batch_size": len(texts), "dimension": len(vectors[0]),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "request_sha256": digest(texts), "response_sha256": digest(vectors),
        "gold_accessed": False, "jev_calls": 0, "paid_api_calls": 0,
    }
    write_json(EMBED_PROBE, out)
    return out


def ensure_collection(client):
    from qdrant_client.models import Distance, VectorParams
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        size = info.config.params.vectors.size
        distance = str(info.config.params.vectors.distance)
        if int(size) != EMBED_DIMENSION or "COSINE" not in distance.upper():
            raise RuntimeError(f"existing H1 collection incompatible: size={size}, distance={distance}")
    else:
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=EMBED_DIMENSION, distance=Distance.COSINE))


def embed(batch_size: int, max_batches: int | None = None) -> dict[str, Any]:
    if not EMBED_PROBE.exists():
        raise RuntimeError("run --probe before --embed")
    from qdrant_client.models import PointStruct
    client = qdrant()
    ensure_collection(client)
    conn = db_connect()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    progress = json.loads(EMBED_PROGRESS.read_text()) if EMBED_PROGRESS.exists() else {}
    last_rowid = int(progress.get("last_completed_rowid", 0))
    completed_batches = 0
    started = time.perf_counter()
    while True:
        rows = conn.execute(
            "SELECT rowid, chunk_id, internal_doc_id, original_source_id, product, source_type, source_record_index, title, chunk_index, token_count, text FROM chunks WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (last_rowid, batch_size),
        ).fetchall()
        if not rows:
            break
        texts = [f"{row[7]}\n{row[10]}" for row in rows]
        vectors = embedding_request(texts)
        points = []
        for row, vector in zip(rows, vectors):
            points.append(PointStruct(
                id=int(row[0]), vector=vector,
                payload={
                    "chunk_id": row[1], "internal_doc_id": row[2], "original_source_id": row[3],
                    "product": row[4], "source_type": row[5], "source_record_index": row[6],
                    "title": row[7], "chunk_index": row[8], "token_count": row[9],
                    "text_sha256": digest(row[10]),
                },
            ))
        client.upsert(COLLECTION, points=points, wait=True)
        last_rowid = int(rows[-1][0])
        completed_batches += 1
        progress = {"status": "RUNNING", "collection": COLLECTION, "dimension": EMBED_DIMENSION, "last_completed_rowid": last_rowid, "total_chunks": total, "updated_epoch": time.time(), "embedding_service": EMBED_URL, "gold_accessed": False, "jev_calls": 0, "paid_api_calls": 0}
        write_json(EMBED_PROGRESS, progress)
        print(f"HERB embedding {last_rowid}/{total} chunks", flush=True)
        if max_batches is not None and completed_batches >= max_batches:
            conn.close()
            return progress
    conn.close()
    exact = client.count(COLLECTION, exact=True).count
    info = client.get_collection(COLLECTION)
    out = {
        "status": "PASS" if exact == total else "BLOCKED_POINT_COUNT",
        "collection": COLLECTION, "dimension": EMBED_DIMENSION, "total_chunks": total,
        "qdrant_exact_point_count": exact, "collection_status": str(info.status),
        "last_completed_rowid": last_rowid, "elapsed_seconds": time.perf_counter() - started,
        "gold_accessed": False, "jev_calls": 0, "paid_api_calls": 0,
    }
    write_json(EMBED_PROGRESS, out)
    return out


def bm25_hits(conn: sqlite3.Connection, question: str) -> list[dict[str, Any]]:
    terms = search_tokens(question)
    if not terms:
        return []
    query = " OR ".join('"' + term.replace('"', ' ') + '"' for term in terms)
    rows = conn.execute(
        "SELECT c.chunk_id, c.internal_doc_id, c.original_source_id, c.product, c.source_type, c.title, c.chunk_index, bm25(chunks_fts) AS score FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
        (query, BM25_TOP_K),
    ).fetchall()
    return [{"chunk_id": r[0], "internal_doc_id": r[1], "original_source_id": r[2], "product": r[3], "source_type": r[4], "title": r[5], "chunk_index": r[6], "rank": i + 1, "score": float(r[7])} for i, r in enumerate(rows)]


def dense_hits(client, vector: list[float]) -> list[dict[str, Any]]:
    result = client.query_points(collection_name=COLLECTION, query=vector, limit=DENSE_TOP_K, with_payload=True, with_vectors=False)
    points = result.points
    out = []
    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        out.append({"chunk_id": payload["chunk_id"], "internal_doc_id": payload["internal_doc_id"], "original_source_id": payload["original_source_id"], "product": payload["product"], "source_type": payload["source_type"], "title": payload["title"], "chunk_index": payload["chunk_index"], "rank": index, "score": float(point.score)})
    return out


def merge_hits(bm25: list[dict[str, Any]], dense: list[dict[str, Any]], conn: sqlite3.Connection) -> list[dict[str, Any]]:
    doc_scores: dict[str, dict[str, Any]] = {}
    for source, hits in (("bm25", bm25), ("dense", dense)):
        for hit in hits:
            doc = doc_scores.setdefault(hit["internal_doc_id"], {"doc_id": hit["internal_doc_id"], "original_source_id": hit["original_source_id"], "product": hit["product"], "source_type": hit["source_type"], "title": hit["title"], "rrf_score": 0.0, "bm25_rank": None, "dense_rank": None, "representative_chunk_id": hit["chunk_id"]})
            doc["rrf_score"] += 1.0 / (RRF_K + hit["rank"])
            doc[f"{source}_rank"] = hit["rank"]
            if hit["rank"] < (doc.get("representative_rank") or 10**9):
                doc["representative_chunk_id"] = hit["chunk_id"]
                doc["representative_rank"] = hit["rank"]
    ordered = sorted(doc_scores.values(), key=lambda x: (-x["rrf_score"], x["representative_rank"], x["doc_id"]))[:SLATE_K]
    for rank, doc in enumerate(ordered, start=1):
        row = conn.execute("SELECT text, text_sha256 FROM chunks WHERE chunk_id = ?", (doc["representative_chunk_id"],)).fetchone()
        if row is None:
            raise RuntimeError(f"representative chunk missing: {doc['representative_chunk_id']}")
        doc["rank"] = rank
        doc["representative_text_sha256"] = row[1]
    return ordered


def retrieve(batch_size: int) -> dict[str, Any]:
    progress = json.loads(EMBED_PROGRESS.read_text()) if EMBED_PROGRESS.exists() else {}
    if progress.get("status") != "PASS":
        raise RuntimeError("complete --embed before --retrieve")
    query_path = OUT / "HERB_H1_QUERY_MANIFEST.jsonl"
    queries = load_jsonl(query_path)
    if len(queries) != 1514 or any(not q.get("question", "").strip() for q in queries):
        raise RuntimeError("HERB query manifest must contain 1514 nonempty canonical questions")
    client = qdrant()
    conn = db_connect()
    query_vectors: list[list[float]] = []
    started = time.perf_counter()
    for begin in range(0, len(queries), batch_size):
        batch = queries[begin : begin + batch_size]
        query_vectors.extend(embedding_request([q["question"] for q in batch]))
        print(f"HERB query embeddings {min(begin + batch_size, len(queries))}/{len(queries)}", flush=True)
    rows: list[dict[str, Any]] = []
    for query, vector in zip(queries, query_vectors):
        started_query = time.perf_counter()
        bm25_started = time.perf_counter()
        bm25 = bm25_hits(conn, query["question"])
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        dense_started = time.perf_counter()
        dense = dense_hits(client, vector)
        dense_ms = (time.perf_counter() - dense_started) * 1000
        rrf_started = time.perf_counter()
        slate = merge_hits(bm25, dense, conn)
        rrf_ms = (time.perf_counter() - rrf_started) * 1000
        if len(slate) != SLATE_K or len({item["doc_id"] for item in slate}) != SLATE_K:
            raise RuntimeError(f"invalid Top-20 slate for {query['query_id']}: {len(slate)}")
        rows.append({
            "query_id": query["query_id"], "question": query["question"], "question_sha256": query["question_sha256"],
            "candidate_ids": [item["doc_id"] for item in slate], "candidates": slate,
            "bm25_top100": bm25, "dense_top100": dense,
            "timing_ms": {"bm25": bm25_ms, "dense_qdrant": dense_ms, "rrf_document_aggregation": rrf_ms, "full_retrieval": (time.perf_counter() - started_query) * 1000},
            "collection": COLLECTION, "rrf_k": RRF_K, "slate_size": SLATE_K,
            "gold_accessed": False, "status": "ok",
        })
        if len(rows) % 100 == 0 or len(rows) == len(queries):
            print(f"HERB retrieval {len(rows)}/{len(queries)}", flush=True)
    append_jsonl(RETRIEVAL, rows, "w")
    report = {"status": "PASS", "queries": len(rows), "slate_size": SLATE_K, "collection": COLLECTION, "retrieval_sha256": digest(RETRIEVAL.read_bytes()), "elapsed_seconds": time.perf_counter() - started, "gold_accessed": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(RETRIEVAL_REPORT, report)
    conn.close()
    return report


def validate() -> dict[str, Any]:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, timeout=600)
    conn = db_connect()
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    unique_docs = conn.execute("SELECT COUNT(DISTINCT internal_doc_id) FROM documents").fetchone()[0]
    unique_chunks = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunks").fetchone()[0]
    forbidden = conn.execute("SELECT COUNT(*) FROM documents WHERE text LIKE '%\"ground_truth\"%' OR text LIKE '%\"citations\"%'").fetchone()[0]
    conn.close()
    exact = client.count(COLLECTION, exact=True).count if client.collection_exists(COLLECTION) else 0
    info = client.get_collection(COLLECTION) if client.collection_exists(COLLECTION) else None
    report = {
        "status": "PASS" if chunks == exact and forbidden == 0 else "BLOCKED",
        "documents": docs, "unique_document_ids": unique_docs, "chunks": chunks, "unique_chunk_ids": unique_chunks,
        "forbidden_gold_text_hits": forbidden, "qdrant_exact_point_count": exact,
        "qdrant_dimension": getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None).size if info else None,
        "qdrant_status": str(info.status) if info else None,
        "collection": COLLECTION, "gold_accessed": False, "jev_calls": 0, "paid_api_calls": 0,
    }
    write_json(INTEGRITY_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    if not any((args.build, args.probe, args.embed, args.retrieve, args.validate)):
        parser.error("choose at least one stage")
    if args.build:
        print(json.dumps(build(), indent=2, sort_keys=True), flush=True)
    if args.probe:
        print(json.dumps(probe(min(args.batch_size, 2)), indent=2, sort_keys=True), flush=True)
    if args.embed:
        print(json.dumps(embed(args.batch_size, args.max_batches), indent=2, sort_keys=True), flush=True)
    if args.retrieve:
        print(json.dumps(retrieve(args.batch_size), indent=2, sort_keys=True), flush=True)
    if args.validate:
        print(json.dumps(validate(), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
