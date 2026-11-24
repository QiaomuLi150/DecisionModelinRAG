#!/usr/bin/env python3
"""Track A retrieval: frozen BM25 + Qwen dense + RRF Top-20.

This runner is separate from the historical 750-query M2 runner.  It reads
only the gold-free v1.1 ID manifest and writes resumable, versioned artifacts.
Gold sidecars are deliberately not opened here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m2_retrieve import COLLECTION, QDRANT_URL, dense_search, rrf_slate  # noqa: E402

EMBED_URL = "http://service.example.invalid/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
CORPUS_PATH = ROOT / "data/corpus/passages.jsonl"
COHORT_PATH = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"
OUT_DIR = ROOT / "results/non_jev/v11/track_a"
FULL_PATH = OUT_DIR / "V11_TRACK_A_RETRIEVAL_SLATES.jsonl"
CHECKPOINT_PATH = OUT_DIR / "retrieval_checkpoint.json"
CONFIG = {
    "bm25_top_k": 100,
    "dense_top_k": 100,
    "rrf_k": 60,
    "slate_k": 20,
    "tokenizer": "lowercase_regex",
    "embedding_model": EMBED_MODEL,
    "embedding_revision": EMBED_REVISION,
    "qdrant_collection": COLLECTION,
    "gold_blind": True,
}
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class FastBM25:
    """Exact M2 BM25 scoring with precomputed term frequencies.

    The historical implementation calls ``list.count`` for every posting at
    query time.  This equivalent implementation preserves its tokenizer,
    constants, and tie rule while making the v1.1 run practical on the frozen
    144,650-document corpus.
    """

    def __init__(self, rows: list[dict], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.rows = rows
        self.lengths = []
        self.df: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, row in enumerate(rows):
            doc_tokens = TOKEN_RE.findall((row["title"] + " " + row["full_text"]).casefold())
            self.lengths.append(len(doc_tokens))
            frequencies = Counter(doc_tokens)
            for term, frequency in frequencies.items():
                self.df[term] += 1
                self.postings[term].append((index, frequency))
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.n = len(rows)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(TOKEN_RE.findall(query.casefold()))
        for term, qtf in query_terms.items():
            postings = self.postings.get(term, ())
            if not postings:
                continue
            idf = __import__("math").log(1.0 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for index, tf in postings:
                norm = tf + self.k1 * (1.0 - self.b + self.b * self.lengths[index] / max(self.avgdl, 1e-9))
                scores[index] += idf * ((tf * (self.k1 + 1.0)) / norm) * min(qtf, 2)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], self.rows[item[0]]["doc_id"]))
        return ordered[:top_k]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def service_identity() -> dict:
    response = requests.get("http://service.example.invalid/v1/models", timeout=20)
    response.raise_for_status()
    body = response.json()
    model = body["data"][0]
    if model.get("id") != EMBED_MODEL or EMBED_REVISION not in str(model.get("root", "")):
        raise RuntimeError(f"unexpected embedding service identity: {model}")
    return model


def request_embedding(text: str) -> tuple[list[float], float, str, str]:
    payload = {"model": EMBED_MODEL, "input": [text]}
    last_error = None
    for attempt in range(4):
        started = time.perf_counter_ns()
        try:
            response = requests.post(EMBED_URL, json=payload, timeout=300)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            response.raise_for_status()
            body = response.json()
            data = sorted(body["data"], key=lambda item: int(item.get("index", 0)))
            if len(data) != 1:
                raise RuntimeError(f"embedding count mismatch: {len(data)}")
            return data[0]["embedding"], elapsed, digest(payload), digest(body)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding request failed after retries: {last_error}")


def select_smoke(rows_: list[dict]) -> list[dict]:
    by_dataset = {}
    for row in rows_:
        by_dataset.setdefault(row["dataset"], []).append(row)
    # Fixed 7/7/6 stratification in the already frozen cohort order.
    quotas = {"hotpot": 7, "musique": 7, "wiki2": 6}
    return [row for dataset in ("hotpot", "musique", "wiki2") for row in by_dataset[dataset][:quotas[dataset]]]


def load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    state = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            state[row["query_id"]] = row
    return state


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=FULL_PATH)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cohort = rows(COHORT_PATH)
    if len(cohort) != 3767 or len({row["query_id"] for row in cohort}) != 3767:
        raise RuntimeError("frozen Track A cohort is not exactly 3,767 unique records")
    work = select_smoke(cohort) if args.mode == "smoke" else cohort
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output
    if args.mode == "smoke" and output == FULL_PATH:
        output = OUT_DIR / "V11_TRACK_A_RETRIEVAL_SMOKE.jsonl"
    existing = load_state(output)
    config_hash = digest(CONFIG)

    from qdrant_client import QdrantClient

    identity = service_identity()
    client = QdrantClient(url=QDRANT_URL, timeout=300)
    collection = client.get_collection(COLLECTION)
    exact_count = client.count(collection_name=COLLECTION, exact=True).count
    if exact_count != 144650:
        raise RuntimeError(f"Qdrant exact point count changed: {exact_count}")
    configured = collection.config.params.vectors
    dimension = getattr(configured, "size", None)
    if dimension != 2048:
        raise RuntimeError(f"Qdrant dimension changed: {dimension}")

    corpus = rows(CORPUS_PATH)
    if len(corpus) != 144650:
        raise RuntimeError(f"unexpected corpus row count: {len(corpus)}")
    by_index = {index: row for index, row in enumerate(corpus)}
    bm25 = FastBM25(corpus)
    completed = {qid: row for qid, row in existing.items() if row.get("config_sha256") == config_hash and row.get("status") == "ok"}
    failures = 0
    thread_state = threading.local()

    def worker_client():
        if not hasattr(thread_state, "client"):
            thread_state.client = QdrantClient(url=QDRANT_URL, timeout=300)
        return thread_state.client

    def run_one(query: dict) -> dict:
        qid = query["query_id"]
        started = time.perf_counter_ns()
        row = {
            "query_id": qid,
            "dataset": query["dataset"],
            "question": query["question"],
            "slate_id": f"v11-track-a:{qid}",
            "config_sha256": config_hash,
            "corpus_sha256": file_sha(CORPUS_PATH),
            "embedding_model": EMBED_MODEL,
            "embedding_revision": EMBED_REVISION,
            "qdrant_collection": COLLECTION,
            "candidate_order": "rrf_desc_then_doc_id",
            "qrels_accessed_during_generation": False,
            "status": "error",
            "real_inference": False,
        }
        try:
            embedding, embedding_ms, request_hash, response_hash = request_embedding(query["question"])
            bm25_start = time.perf_counter_ns()
            bm25_hits = bm25.search(query["question"], CONFIG["bm25_top_k"])
            bm25_ms = (time.perf_counter_ns() - bm25_start) / 1_000_000.0
            qdrant_start = time.perf_counter_ns()
            dense_hits = dense_search(worker_client(), embedding, CONFIG["dense_top_k"])
            qdrant_ms = (time.perf_counter_ns() - qdrant_start) / 1_000_000.0
            if any(index not in by_index for index, _ in dense_hits):
                raise RuntimeError("Qdrant returned point ID outside frozen corpus")
            rrf_start = time.perf_counter_ns()
            slate = rrf_slate(corpus, bm25_hits, dense_hits, CONFIG["rrf_k"], CONFIG["slate_k"])
            rrf_ms = (time.perf_counter_ns() - rrf_start) / 1_000_000.0
            candidate_ids = [item["doc_id"] for item in slate]
            if len(candidate_ids) != 20 or len(set(candidate_ids)) != 20:
                raise RuntimeError(f"invalid Top-20 slate size/uniqueness: {len(candidate_ids)}")
            row.update({
                "candidate_ids": candidate_ids,
                "candidates": slate,
                "embedding_request_hash": request_hash,
                "embedding_response_hash": response_hash,
                "embedding_ms": embedding_ms,
                "bm25_ms": bm25_ms,
                "qdrant_ms": qdrant_ms,
                "rrf_ms": rrf_ms,
                "retrieval_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                "status": "ok",
                "real_inference": True,
            })
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
            row["retrieval_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
        return row

    pending = [query for query in work if query["query_id"] not in completed]

    def checkpoint() -> None:
        ordered = [completed[query["query_id"]] for query in work if query["query_id"] in completed]
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in ordered:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, query): query for query in pending}
        for ordinal, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            completed[row["query_id"]] = row
            failures += int(row.get("status") != "ok")
            if ordinal % 10 == 0 or ordinal == len(pending):
                checkpoint()
            total_done = len(completed)
            if total_done in {100, 1000, 2000, 3000, 3767} or ordinal % 100 == 0 or ordinal == len(pending):
                print(json.dumps({"mode": args.mode, "completed": total_done, "requested": len(work), "new_completed": ordinal, "new_pending": len(pending) - ordinal, "ok": sum(item.get("status") == "ok" for item in completed.values()), "errors": failures}, sort_keys=True), flush=True)

    output_rows = [completed[query["query_id"]] for query in work]
    checkpoint()

    metadata = {
        "mode": args.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": len(work),
        "status_counts": {status: sum(item.get("status") == status for item in output_rows) for status in sorted({item.get("status") for item in output_rows})},
        "cohort_sha256": file_sha(COHORT_PATH),
        "corpus_sha256": file_sha(CORPUS_PATH),
        "config_sha256": config_hash,
        "embedding_service_identity": identity,
        "qdrant_collection": COLLECTION,
        "qdrant_exact_points": exact_count,
        "qdrant_vector_dimension": dimension,
        "gold_accessed": False,
        "jev_calls": 0,
        "output_sha256": file_sha(output),
    }
    (output.with_suffix(output.suffix + ".manifest.json")).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
