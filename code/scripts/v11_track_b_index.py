#!/usr/bin/env python3
"""DecisionRAG v1.1 Track B enterprise corpus, index, and retrieval.

This module intentionally keeps the enterprise track separate from Track A.
It builds a deterministic archive-derived SQLite/FTS BM25 index, a dedicated
Qdrant dense collection, and frozen document-level Top-20 slates.  It never
reads the enterprise question gold fields during corpus/index/retrieval work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import statistics
import time
import unicodedata
import urllib.parse
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0"
ARCHIVE = BASE / "all_documents.zip"
QUESTIONS = BASE / "questions.jsonl"
CONTRACT = ROOT / "results/preflight/v11_remediation/ENTERPRISE_INDEXING_CONTRACT_V11R1.json"
OUT = ROOT / "results/non_jev/v11/track_b"
DB_PATH = OUT / "V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
CHUNK_MANIFEST = OUT / "V11_TRACK_B_CHUNK_MANIFEST.jsonl"
BUILD_SUMMARY = OUT / "V11_TRACK_B_CORPUS_AND_CHUNK_MANIFEST.json"
PROBE_SUMMARY = OUT / "V11_TRACK_B_EMBEDDING_PROBE.json"
EMBED_PROGRESS = OUT / "V11_TRACK_B_EMBEDDING_PROGRESS.json"
PART1_PROGRESS = OUT / "V11_TRACK_B_EMBEDDING_PROGRESS.part1.json"
PART2_PROGRESS = OUT / "V11_TRACK_B_EMBEDDING_PROGRESS.part2.json"
RETRIEVAL = OUT / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl"
RETRIEVAL_MANIFEST = OUT / "V11_TRACK_B_RETRIEVAL_SLATES.manifest.json"
EMBED_URL = "http://service.example.invalid/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QDRANT_URL = "http://service.example.invalid"
COLLECTION = "decisionrag_v11_enterprise_qwen_embedding_r1"
CHUNK_SIZE = 384
OVERLAP = 64
BM25_TOP_K = 100
DENSE_TOP_K = 100
RRF_K = 60
SLATE_K = 20
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
WS_RE = re.compile(r"\S+", re.UNICODE)
DSID_RE = re.compile(r"(dsid_[0-9a-fA-F]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def internal_id(archive_path: str) -> str:
    encoded = urllib.parse.quote(archive_path, safe="")
    return f"enterprise:v1.0.0:path-qualified-relative-archive-member:{encoded}"


def source_type(archive_path: str) -> str:
    return archive_path.split("/", 1)[0]


def original_doc_id(archive_path: str) -> str:
    match = DSID_RE.search(Path(archive_path).name)
    return match.group(1) if match else "UNKNOWN_ORIGINAL_ID"


def normalize_document(raw: bytes) -> tuple[str, str]:
    text = raw.decode("utf-8", errors="strict")
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    title_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if title_index is None:
        return "", ""
    title = lines[title_index].strip()
    body = "\n".join(lines[title_index + 1 :]).strip()
    return title, body


def normalized_search_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def make_chunks(text: str) -> list[tuple[int, str, int]]:
    tokens = WS_RE.findall(text)
    if not tokens:
        return [(0, "", 0)]
    step = CHUNK_SIZE - OVERLAP
    unique: list[tuple[str, int]] = []
    seen: set[str] = set()
    start = 0
    while start < len(tokens):
        chunk = " ".join(tokens[start : start + CHUNK_SIZE])
        if chunk not in seen:
            unique.append((chunk, min(CHUNK_SIZE, len(tokens) - start)))
            seen.add(chunk)
        if start + CHUNK_SIZE >= len(tokens):
            break
        start += step
    return [(index, chunk, token_count) for index, (chunk, token_count) in enumerate(unique)]


def connect() -> sqlite3.Connection:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            internal_doc_id TEXT PRIMARY KEY,
            original_doc_id TEXT NOT NULL,
            archive_path TEXT NOT NULL UNIQUE,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            body_sha256 TEXT NOT NULL,
            uncompressed_bytes INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            internal_doc_id TEXT NOT NULL,
            original_doc_id TEXT NOT NULL,
            archive_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            text_sha256 TEXT NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(internal_doc_id) REFERENCES documents(internal_doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(internal_doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            search_text,
            chunk_id UNINDEXED,
            content=''
        );
        """
    )
    conn.commit()


def load_existing_documents(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT internal_doc_id FROM documents")}


def write_manifest_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_corpus() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = connect()
    init_db(conn)
    existing = load_existing_documents(conn)
    manifest_mode = "a" if CHUNK_MANIFEST.exists() and existing else "w"
    documents = chunks = skipped = 0
    source_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    started = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as archive, CHUNK_MANIFEST.open(manifest_mode, encoding="utf-8") as manifest:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".txt"))
        for ordinal, name in enumerate(members, start=1):
            doc_id = internal_id(name)
            if doc_id in existing:
                skipped += 1
                continue
            raw = archive.read(name)
            title, body = normalize_document(raw)
            original = original_doc_id(name)
            stype = source_type(name)
            chunks_for_doc = make_chunks(body)
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, original, name, stype, title, sha256_text(body), len(raw), len(chunks_for_doc)),
            )
            for chunk_index, chunk_text, token_count in chunks_for_doc:
                chunk_id = f"{doc_id}#chunk-{chunk_index}"
                row = {
                    "chunk_id": chunk_id,
                    "internal_doc_id": doc_id,
                    "original_doc_id": original,
                    "archive_path": name,
                    "source_type": stype,
                    "title": title,
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                    "text_sha256": sha256_text(chunk_text),
                }
                cursor = conn.execute(
                    "INSERT INTO chunks(chunk_id, internal_doc_id, original_doc_id, archive_path, source_type, title, chunk_index, token_count, text_sha256, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, original, name, stype, title, chunk_index, token_count, row["text_sha256"], chunk_text),
                )
                rowid = int(cursor.lastrowid)
                search_text = " ".join(normalized_search_tokens(f"{title} {chunk_text}"))
                conn.execute("INSERT INTO chunks_fts(rowid, search_text, chunk_id) VALUES (?, ?, ?)", (rowid, search_text, chunk_id))
                write_manifest_row(manifest, row)
                chunks += 1
                chunk_counts[stype] += 1
            documents += 1
            source_counts[stype] += 1
            existing.add(doc_id)
            if documents % 100 == 0:
                conn.commit()
            if ordinal % 10000 == 0 or ordinal == len(members):
                print(f"enterprise corpus {ordinal}/{len(members)} archive members; new_docs={documents} chunks={chunks}", flush=True)
    conn.commit()
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    conn.commit()
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    distinct_chunk_ids = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunks").fetchone()[0]
    # Derive final distributions from the database rather than from only the
    # documents inserted in this invocation.  This keeps resumed-build
    # manifests truthful and makes the summary a complete-corpus artifact.
    source_counts = Counter(
        dict(conn.execute("SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"))
    )
    chunk_counts = Counter(
        dict(conn.execute("SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type"))
    )
    # The database is the integrity artifact that will be handed to the
    # retrieval stage.  Consolidate WAL pages before hashing it so the hash
    # covers the complete on-disk database, not an accompanying -wal file.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    manifest = {
        "study": "DecisionRAG v1.1 Track B independent EnterpriseRAG-Bench transfer",
        "status": "CORPUS_BUILT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE),
        "archive_sha256": sha256_file(ARCHIVE),
        "questions_sha256": sha256_file(QUESTIONS),
        "contract_sha256": sha256_file(CONTRACT),
        "archive_txt_members": len(members),
        "physical_documents": doc_count,
        "new_documents_this_run": documents,
        "resumed_existing_documents": skipped,
        "chunks": chunk_count,
        "distinct_chunk_ids": distinct_chunk_ids,
        "chunk_duplicate_count": chunk_count - distinct_chunk_ids,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "source_document_counts": dict(sorted(source_counts.items())),
        "chunk_counts_by_source": dict(sorted(chunk_counts.items())),
        "database": str(DB_PATH),
        "chunk_manifest": str(CHUNK_MANIFEST),
        "database_sha256": sha256_file(DB_PATH),
        "chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST),
        "elapsed_seconds": time.perf_counter() - started,
        "gold_used": False,
        "jev_calls": 0,
    }
    BUILD_SUMMARY.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    conn.close()
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def embedding_request(texts: list[str], endpoint: str = EMBED_URL, timeout: int = 600) -> list[list[float]]:
    payload = {"model": EMBED_MODEL, "input": texts}
    last_error = None
    for attempt in range(5):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()["data"]
            data.sort(key=lambda item: int(item.get("index", 0)))
            vectors = [item["embedding"] for item in data]
            if len(vectors) != len(texts):
                raise RuntimeError(f"embedding count mismatch: {len(vectors)} != {len(texts)}")
            return vectors
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"embedding request failed after retries: {last_error}")


def probe_embeddings(batch_size: int, probe_chunks: int) -> dict:
    conn = connect()
    rows = conn.execute("SELECT rowid, title, text FROM chunks ORDER BY rowid LIMIT ?", (probe_chunks,)).fetchall()
    if len(rows) < probe_chunks:
        raise RuntimeError(f"only {len(rows)} chunks available for requested probe {probe_chunks}")
    started = time.perf_counter()
    requests_count = 0
    vectors = 0
    dimensions: set[int] = set()
    failures = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        try:
            result = embedding_request([f"{title}\n{text}" for _, title, text in batch])
            requests_count += 1
            vectors += len(result)
            dimensions.update(len(vector) for vector in result)
        except Exception:
            failures += 1
            raise
    elapsed = time.perf_counter() - started
    output = {
        "status": "PASS" if not failures and dimensions == {2048} else "BLOCKED",
        "probe_chunks": len(rows),
        "batch_size": batch_size,
        "requests": requests_count,
        "vectors": vectors,
        "dimensions": sorted(dimensions),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "chunks_per_second": vectors / elapsed if elapsed else None,
        "estimated_full_embedding_hours": (conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] / (vectors / elapsed) / 3600) if vectors and elapsed else None,
        "embedding_endpoint": EMBED_URL,
        "embedding_model": EMBED_MODEL,
        "gold_used": False,
        "jev_calls": 0,
    }
    PROBE_SUMMARY.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    conn.close()
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, timeout=600)


def ensure_collection(client, dimension: int) -> None:
    from qdrant_client import models
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        configured = info.config.params.vectors
        size = getattr(configured, "size", None)
        if size != dimension:
            raise RuntimeError(f"enterprise collection dimension={size}, expected={dimension}")
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )


def embed_full(
    batch_size: int,
    start_rowid: int = 1,
    end_rowid: int | None = None,
    endpoint: str = EMBED_URL,
    progress_path: pathlib.Path = EMBED_PROGRESS,
) -> dict:
    if not PROBE_SUMMARY.exists():
        raise RuntimeError("run --probe before --embed")
    probe = json.loads(PROBE_SUMMARY.read_text())
    dimension = int(probe["dimensions"][0])
    conn = connect()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    client = qdrant_client()
    ensure_collection(client, dimension)
    from qdrant_client import models
    range_end = min(end_rowid or total, total)
    if range_end < start_rowid:
        raise RuntimeError(f"invalid embedding range {start_rowid}..{range_end}")
    progress_start = start_rowid
    if progress_path.exists() and start_rowid == 1:
        progress_start = int(json.loads(progress_path.read_text()).get("last_completed_rowid", 0)) + 1
    started = time.perf_counter()
    processed = max(0, progress_start - 1)
    retries = 0
    while progress_start <= range_end:
        rows = conn.execute(
            "SELECT rowid, chunk_id, internal_doc_id, original_doc_id, archive_path, source_type, title, chunk_index, token_count, text_sha256, text FROM chunks WHERE rowid >= ? ORDER BY rowid LIMIT ?",
            (progress_start, min(batch_size, range_end - progress_start + 1)),
        ).fetchall()
        if not rows:
            break
        try:
            vectors = embedding_request([f"{row[6]}\n{row[10]}" for row in rows], endpoint=endpoint)
        except Exception:
            retries += 1
            raise
        points = []
        for row, vector in zip(rows, vectors):
            if len(vector) != dimension:
                raise RuntimeError(f"dimension mismatch for {row[1]}: {len(vector)}")
            points.append(
                models.PointStruct(
                    id=int(row[0]),
                    vector=vector,
                    payload={
                        "chunk_id": row[1], "internal_doc_id": row[2], "original_doc_id": row[3],
                        "archive_path": row[4], "source_type": row[5], "title": row[6],
                        "chunk_index": row[7], "token_count": row[8], "text_sha256": row[9],
                    },
                )
            )
        client.upsert(collection_name=COLLECTION, points=points, wait=False)
        last = int(rows[-1][0])
        processed = last
        progress_path.write_text(json.dumps({"status": "RUNNING", "collection": COLLECTION, "dimension": dimension, "last_completed_rowid": last, "range_start": start_rowid, "range_end": range_end, "total_chunks": total, "embedding_endpoint": endpoint, "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress_start = last + 1
        if processed % (batch_size * 20) == 0 or processed == range_end:
            print(f"enterprise dense index {processed}/{total} chunks", flush=True)
    point_count = client.count(collection_name=COLLECTION, exact=True).count
    if range_end == total:
        deadline = time.time() + 1800
        while time.time() < deadline and point_count < total:
            time.sleep(5)
            point_count = client.count(collection_name=COLLECTION, exact=True).count
    info = client.get_collection(COLLECTION)
    point_count = client.count(collection_name=COLLECTION, exact=True).count
    status = "PASS" if range_end == total and processed >= total and point_count == total else ("PASS_RANGE" if processed >= range_end else "BLOCKED_RESUMABLE")
    output = {
        "status": status, "collection": COLLECTION, "dimension": dimension, "total_chunks": total,
        "range_start": start_rowid, "range_end": range_end, "last_completed_rowid": processed,
        "qdrant_exact_point_count": point_count, "embedding_endpoint": endpoint,
        "retries": retries, "elapsed_seconds": time.perf_counter() - started,
        "collection_info": str(info), "progress_path": str(progress_path), "gold_used": False, "jev_calls": 0,
    }
    progress_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    conn.close()
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def finalize_embedding() -> dict:
    """Validate and publish the combined two-worker dense-index checkpoint."""
    if not PART1_PROGRESS.exists() or not PART2_PROGRESS.exists():
        raise RuntimeError("both range progress files are required before finalization")
    part1 = json.loads(PART1_PROGRESS.read_text())
    part2 = json.loads(PART2_PROGRESS.read_text())
    conn = connect()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    client = qdrant_client()
    exact_points = client.count(collection_name=COLLECTION, exact=True).count
    info = client.get_collection(COLLECTION)
    indexed = getattr(info, "indexed_vectors_count", None)
    if total != 1_242_073:
        raise RuntimeError(f"unexpected corpus chunk count during finalization: {total}")
    if part1.get("range_start") != 1 or part1.get("range_end") != 621_036 or part1.get("last_completed_rowid") != 621_036:
        raise RuntimeError(f"worker1 range incomplete: {part1}")
    if part2.get("range_start") != 621_037 or part2.get("range_end") != total or part2.get("last_completed_rowid") != total:
        raise RuntimeError(f"worker2 range incomplete: {part2}")
    if exact_points != total:
        raise RuntimeError(f"Qdrant exact point count {exact_points} != {total}")
    output = {
        "status": "PASS",
        "collection": COLLECTION,
        "dimension": 2048,
        "total_chunks": total,
        "qdrant_exact_point_count": exact_points,
        "indexed_vectors_count_at_finalization": indexed,
        "collection_status": str(info.status),
        "worker_ranges": [part1, part2],
        "gold_used": False,
        "jev_calls": 0,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
    }
    EMBED_PROGRESS.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def bm25_hits(conn: sqlite3.Connection, question: str) -> list[dict]:
    terms = normalized_search_tokens(question)
    if not terms:
        return []
    query = " OR ".join('"' + term.replace('"', '""') + '"' for term in dict.fromkeys(terms))
    rows = conn.execute(
        # chunks_fts is intentionally contentless, so its payload columns may
        # read back as NULL.  Use the FTS rowid to recover canonical metadata
        # from the source chunks table instead of treating FTS columns as IDs.
        "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score ASC, rowid ASC LIMIT ?",
        (query, BM25_TOP_K),
    ).fetchall()
    output = []
    for rank, (rowid, score) in enumerate(rows, start=1):
        chunk_id, internal_doc_id, original_doc_id, archive_path, source_type, title, chunk_index = conn.execute(
            "SELECT chunk_id, internal_doc_id, original_doc_id, archive_path, source_type, title, chunk_index FROM chunks WHERE rowid = ?", (rowid,)
        ).fetchone()
        output.append({"rowid": int(rowid), "chunk_id": chunk_id, "doc_id": internal_doc_id, "original_doc_id": original_doc_id, "archive_path": archive_path, "source_type": source_type, "title": title, "chunk_index": chunk_index, "score": float(-score), "rank": rank})
    return output


def dense_hits(client, vector: list[float]) -> list[dict]:
    points = client.query_points(collection_name=COLLECTION, query=vector, limit=DENSE_TOP_K, with_payload=True, with_vectors=False).points
    output = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        output.append({"rowid": int(point.id), "chunk_id": payload.get("chunk_id"), "doc_id": payload.get("internal_doc_id"), "original_doc_id": payload.get("original_doc_id"), "archive_path": payload.get("archive_path"), "source_type": payload.get("source_type"), "title": payload.get("title"), "chunk_index": payload.get("chunk_index"), "score": float(point.score), "rank": rank})
    return output


def aggregate_slate(bm25: list[dict], dense: list[dict]) -> list[dict]:
    modality = {"bm25": bm25, "dense": dense}
    by_doc: dict[str, dict] = {}
    for name, hits in modality.items():
        for hit in hits:
            doc = by_doc.setdefault(hit["doc_id"], {"doc_id": hit["doc_id"], "original_doc_id": hit["original_doc_id"], "archive_path": hit["archive_path"], "source_type": hit["source_type"], "title": hit["title"], "chunks": {}})
            doc[name + "_rank"] = min(int(hit["rank"]), int(doc.get(name + "_rank", 10**9)))
            doc[name + "_score"] = max(float(hit["score"]), float(doc.get(name + "_score", -10**30)))
            chunk = doc["chunks"].setdefault(hit["chunk_id"], {"chunk_id": hit["chunk_id"], "bm25_rank": None, "dense_rank": None})
            chunk[name + "_rank"] = int(hit["rank"])
    for doc in by_doc.values():
        doc["rrf_score"] = sum(1.0 / (RRF_K + doc[name + "_rank"]) if doc.get(name + "_rank") else 0.0 for name in modality)
        ranked_chunks = sorted(doc["chunks"].values(), key=lambda chunk: (-sum(1.0 / (RRF_K + chunk[name + "_rank"]) if chunk.get(name + "_rank") else 0.0 for name in modality), chunk["chunk_id"]))
        doc["representative_chunk_id"] = ranked_chunks[0]["chunk_id"] if ranked_chunks else None
        doc.pop("chunks", None)
    ordered = sorted(by_doc.values(), key=lambda doc: (-doc["rrf_score"], doc["doc_id"]))[:SLATE_K]
    for rank, doc in enumerate(ordered, start=1):
        doc["rank"] = rank
    return ordered


def retrieve() -> dict:
    if not BUILD_SUMMARY.exists() or not PROBE_SUMMARY.exists() or not EMBED_PROGRESS.exists():
        raise RuntimeError("corpus, embedding probe, and full dense index must be complete before retrieval")
    progress = json.loads(EMBED_PROGRESS.read_text())
    if progress.get("status") != "PASS":
        raise RuntimeError(f"dense index is not complete: {progress.get('status')}")
    questions = [{"question_id": row["question_id"], "question": row["question"]} for row in (json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip())]
    if len(questions) != 500 or len({row["question_id"] for row in questions}) != 500:
        raise RuntimeError("enterprise question manifest is not exactly 500 unique questions")
    conn = connect()
    client = qdrant_client()
    rows = []
    started = time.perf_counter()
    for ordinal, question in enumerate(questions, start=1):
        t0 = time.perf_counter_ns()
        bm25 = bm25_hits(conn, question["question"])
        bm25_ms = (time.perf_counter_ns() - t0) / 1_000_000
        t1 = time.perf_counter_ns()
        vector = embedding_request([question["question"]])[0]
        embedding_ms = (time.perf_counter_ns() - t1) / 1_000_000
        t2 = time.perf_counter_ns()
        dense = dense_hits(client, vector)
        qdrant_ms = (time.perf_counter_ns() - t2) / 1_000_000
        t3 = time.perf_counter_ns()
        slate = aggregate_slate(bm25, dense)
        rrf_ms = (time.perf_counter_ns() - t3) / 1_000_000
        if len(slate) != SLATE_K or len({item["doc_id"] for item in slate}) != SLATE_K:
            raise RuntimeError(f"invalid enterprise slate for {question['question_id']}: {len(slate)}")
        rows.append({"query_id": question["question_id"], "question": question["question"], "candidate_ids": [item["doc_id"] for item in slate], "candidates": slate, "bm25_top100": bm25, "dense_top100": dense, "timing_ms": {"bm25": bm25_ms, "embedding": embedding_ms, "qdrant": qdrant_ms, "rrf": rrf_ms}, "index_collection": COLLECTION, "gold_accessed": False, "status": "ok"})
        if ordinal % 25 == 0 or ordinal == len(questions):
            print(f"enterprise retrieval {ordinal}/{len(questions)}", flush=True)
    with RETRIEVAL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output = {"status": "PASS", "queries": len(rows), "slate_size": SLATE_K, "retrieval_sha256": sha256_file(RETRIEVAL), "collection": COLLECTION, "elapsed_seconds": time.perf_counter() - started, "gold_accessed": False, "jev_calls": 0}
    RETRIEVAL_MANIFEST.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    conn.close()
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--finalize-embed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-chunks", type=int, default=5000)
    parser.add_argument("--start-rowid", type=int, default=1)
    parser.add_argument("--end-rowid", type=int, default=None)
    parser.add_argument("--embedding-url", default=EMBED_URL)
    parser.add_argument("--progress-file", default=str(EMBED_PROGRESS))
    args = parser.parse_args()
    if not any((args.build, args.probe, args.embed, args.retrieve, args.finalize_embed)):
        parser.error("choose at least one of --build/--probe/--embed/--retrieve/--finalize-embed")
    if args.build:
        build_corpus()
    if args.probe:
        probe_embeddings(args.batch_size, args.probe_chunks)
    if args.embed:
        embed_full(args.batch_size, args.start_rowid, args.end_rowid, args.embedding_url, Path(args.progress_file))
    if args.finalize_embed:
        finalize_embedding()
    if args.retrieve:
        retrieve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
