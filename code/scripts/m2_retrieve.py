#!/usr/bin/env python3
"""M2: build BM25+dense indexes and freeze deterministic Top-20 RRF slates.

The corpus is the M1 normalized passage manifest.  Gold labels are never read
by this script; they are joined only by later evaluation code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EMBED_URL = "http://service.example.invalid/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QDRANT_URL = "http://service.example.invalid"
COLLECTION = "decisionrag_m2_qwen_embedding_v1"
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25:
    def __init__(self, rows: list[dict], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.rows = rows
        self.doc_tokens = [tokens(row["title"] + " " + row["full_text"]) for row in rows]
        self.lengths = [len(value) for value in self.doc_tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.df: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, doc_tokens in enumerate(self.doc_tokens):
            for term in set(doc_tokens):
                self.df[term] += 1
                self.postings[term].append(index)
        self.n = len(rows)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokens(query))
        for term, qtf in query_terms.items():
            postings = self.postings.get(term, ())
            if not postings:
                continue
            idf = math.log(1.0 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for index in postings:
                tf = self.doc_tokens[index].count(term)
                norm = tf + self.k1 * (1.0 - self.b + self.b * self.lengths[index] / max(self.avgdl, 1e-9))
                scores[index] += idf * ((tf * (self.k1 + 1.0)) / norm) * min(qtf, 2)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], self.rows[item[0]]["doc_id"]))
        return ordered[:top_k]


def request_embeddings(inputs: list[str], timeout: int = 300) -> list[list[float]]:
    payload = {"model": EMBED_MODEL, "input": inputs}
    last_error = None
    for attempt in range(4):
        try:
            response = requests.post(EMBED_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()["data"]
            data.sort(key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in data]
            if len(vectors) != len(inputs):
                raise RuntimeError(f"embedding count mismatch: requested={len(inputs)} returned={len(vectors)}")
            return vectors
        except Exception as exc:  # service can transiently reject a large batch
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding request failed after retries: {last_error}")


def ensure_collection(client, dimension: int) -> None:
    from qdrant_client import models

    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        configured = info.config.params.vectors
        size = getattr(configured, "size", None)
        if size != dimension:
            raise RuntimeError(f"existing study collection dimension={size}, expected={dimension}")
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )


def existing_ids(client) -> set[int]:
    ids: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(collection_name=COLLECTION, limit=1000, offset=offset, with_payload=False, with_vectors=False)
        ids.update(int(point.id) for point in points)
        if offset is None:
            return ids


def index_corpus(client, rows: list[dict], batch_size: int) -> int:
    from qdrant_client import models

    present = existing_ids(client)
    missing = [index for index in range(len(rows)) if index not in present]
    if not missing:
        return 0
    completed = 0
    for start in range(0, len(missing), batch_size):
        indices = missing[start : start + batch_size]
        vectors = request_embeddings([rows[index]["title"] + "\n" + rows[index]["full_text"] for index in indices])
        points = [
            models.PointStruct(
                id=index,
                vector=vector,
                payload={"doc_id": rows[index]["doc_id"], "dataset": rows[index]["dataset"], "title": rows[index]["title"]},
            )
            for index, vector in zip(indices, vectors)
        ]
        client.upsert(collection_name=COLLECTION, points=points, wait=False)
        completed += len(points)
        if completed % (batch_size * 10) == 0 or completed == len(missing):
            print(f"dense corpus indexed {completed}/{len(missing)} missing", flush=True)
    # Ensure asynchronous upserts are visible before query generation.
    deadline = time.time() + 180
    while time.time() < deadline:
        count = client.count(collection_name=COLLECTION, exact=True).count
        if count >= len(rows):
            return completed
        time.sleep(2)
    raise RuntimeError(f"Qdrant did not reach {len(rows)} points; count={client.count(collection_name=COLLECTION, exact=True).count}")


def dense_search(client, vector: list[float], top_k: int) -> list[tuple[int, float]]:
    hits = client.query_points(collection_name=COLLECTION, query=vector, limit=top_k, with_payload=False, with_vectors=False).points
    return [(int(hit.id), float(hit.score)) for hit in hits]


def rrf_slate(rows: list[dict], bm25_hits: list[tuple[int, float]], dense_hits: list[tuple[int, float]], rrf_k: int, slate_k: int) -> list[dict]:
    bm25_rank = {index: rank for rank, (index, _) in enumerate(bm25_hits, start=1)}
    dense_rank = {index: rank for rank, (index, _) in enumerate(dense_hits, start=1)}
    bm25_score = dict(bm25_hits)
    dense_score = dict(dense_hits)
    candidates = set(bm25_rank) | set(dense_rank)
    scored = []
    for index in candidates:
        score = (1.0 / (rrf_k + bm25_rank[index]) if index in bm25_rank else 0.0) + (1.0 / (rrf_k + dense_rank[index]) if index in dense_rank else 0.0)
        scored.append((index, score))
    scored.sort(key=lambda item: (-item[1], rows[item[0]]["doc_id"]))
    return [{"doc_id": rows[index]["doc_id"], "dataset": rows[index]["dataset"], "title": rows[index]["title"], "rrf_score": score, "bm25_rank": bm25_rank.get(index), "dense_rank": dense_rank.get(index), "bm25_score": bm25_score.get(index), "dense_score": dense_score.get(index)} for index, score in scored[:slate_k]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus_path = ROOT / "data/corpus/passages.jsonl"
    questions_path = ROOT / "data/manifests/questions.jsonl"
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(questions) != 750 or any(not item["gold_mapping_valid"] for item in questions):
        raise RuntimeError("M1 manifest is not ready or contains invalid selected mappings")

    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, timeout=300)
    probe = request_embeddings(["M2 dimension probe"])[0]
    ensure_collection(client, len(probe))
    new_points = index_corpus(client, rows, args.batch_size)

    retrieval_dir = ROOT / "data/retrieval"
    frozen_dir = ROOT / "data/frozen_slates"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir.mkdir(parents=True, exist_ok=True)
    bm25 = BM25(rows)
    retrieval_rows = []
    slate_rows = []
    top_k = int(config["retrieval"]["bm25_top_k"])
    rrf_k = int(config["retrieval"]["rrf_k"])
    slate_k = int(config["retrieval"]["slate_k"])
    for count, question in enumerate(questions, start=1):
        query_vector = request_embeddings([question["question"]])[0]
        bm25_hits = bm25.search(question["question"], top_k)
        dense_hits = dense_search(client, query_vector, top_k)
        slate = rrf_slate(rows, bm25_hits, dense_hits, rrf_k, slate_k)
        retrieval_rows.append({"query_id": question["query_id"], "dataset": question["dataset"], "bm25": [{"doc_id": rows[index]["doc_id"], "score": score} for index, score in bm25_hits], "dense": [{"doc_id": rows[index]["doc_id"], "score": score} for index, score in dense_hits], "rrf_top20": slate})
        slate_rows.append({"slate_id": f"m2:{question['query_id']}", "query_id": question["query_id"], "dataset": question["dataset"], "candidate_ids": [item["doc_id"] for item in slate], "candidate_order": "rrf_desc_then_doc_id", "pool_source": "BM25+Qwen3-VL-Embedding-2B+RRF", "qrels_accessed_during_generation": False})
        if count % 25 == 0 or count == len(questions):
            print(f"queries retrieved {count}/{len(questions)}", flush=True)

    retrieval_path = retrieval_dir / "m2_retrieval_results.jsonl"
    slate_path = frozen_dir / "m2_top20_slates.jsonl"
    with retrieval_path.open("w", encoding="utf-8") as handle:
        for row in retrieval_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with slate_path.open("w", encoding="utf-8") as handle:
        for row in slate_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {"study_id": config["study_id"], "created_utc": datetime.now(timezone.utc).isoformat(), "corpus_sha256": sha256_file(corpus_path), "questions_sha256": sha256_file(questions_path), "collection": COLLECTION, "qdrant_url": QDRANT_URL, "embedding_endpoint": EMBED_URL, "embedding_model": EMBED_MODEL, "embedding_dimension": len(probe), "corpus_passages": len(rows), "new_dense_points": new_points, "qdrant_points": client.count(collection_name=COLLECTION, exact=True).count, "bm25": {"tokenizer": config["retrieval"]["tokenizer"], "k1": bm25.k1, "b": bm25.b, "top_k": top_k}, "rrf_k": rrf_k, "slate_k": slate_k, "queries": len(questions), "retrieval_results_sha256": sha256_file(retrieval_path), "slates_sha256": sha256_file(slate_path), "qrels_accessed_during_generation": False}
    (ROOT / "data/manifests/m2_retrieval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "results/audit/phase_state.json").write_text(json.dumps({"current_milestone": "M2", "status": "DONE", "jev_calls": 0, "jev_execution_allowed": False, "m2_manifest": str(ROOT / "data/manifests/m2_retrieval_manifest.json")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
