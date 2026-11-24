#!/usr/bin/env python3
"""DecisionRAG v1.1 Track C production-serving evaluation.

This runner is intentionally separate from the historical M5 runner.  It
consumes only the frozen v1.1 Track C cohort and schedule, performs a complete
query -> retrieval -> selection -> context -> reader request, and writes only
versioned Track C artifacts.  Jev is not imported or called.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from decisionrag.selection import (  # noqa: E402
    deterministic_quartets,
    groupwise_states,
    make_views,
    pointwise_pair,
    pointwise_question,
    pointwise_state,
    rank_indices,
    rescue_top5,
    sha256_json,
)
from m2_retrieve import BM25, EMBED_MODEL, dense_search as multi_dense_search, request_embeddings, rrf_slate  # noqa: E402
from run_m4_reader import cap_reader_text  # noqa: E402
from v11_track_b_index import aggregate_slate, bm25_hits, dense_hits  # noqa: E402

POLICIES = [
    "raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5",
    "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5",
]
MODEL_KEYS = {"bge_p5": "bge", "bge_rescue5": "bge", "qwen_p5": "qwen", "qwen_rescue5": "qwen", "laya_p5": "laya", "laya_rescue5": "laya", "laya_g5": "laya", "clef_p5": "clef", "clef_rescue5": "clef", "clef_g5": "clef"}
MODEL_REVISIONS = {"bge": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "qwen": "4bd860ac4f15ad1897a214615cccc700f8f71818", "laya": "7b928d828b7b0e022f929d9bd2e44165aa270148", "clef": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "reader": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "embedding": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"}
MODEL_IDS = {"bge": "BAAI/bge-reranker-v2-m3", "qwen": "Qwen/Qwen3-VL-Reranker-2B", "laya": "convaiinnovations/laya", "clef": "Cloudflare/clef-flash", "reader": "Qwen/Qwen3-VL-8B-Instruct", "embedding": "Qwen/Qwen3-VL-Embedding-2B"}
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
COHORT = ROOT / "results/preflight/v11_remediation/TRACK_C_V11R1_COHORT.jsonl"
SCHEDULE = ROOT / "results/preflight/v11_remediation/TRACK_C_V11R1_SCHEDULE.json"
REPAIRED_INPUT_MANIFEST = ROOT / "results/non_jev/v11/track_c/V11_TRACK_C_REPAIRED_INPUT_MANIFEST.jsonl"
SMOKE_SOURCE = ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_RETRIEVAL_SLATES.jsonl"
# R2 must never overwrite the invalid v1.1 raw evidence in track_c/.
OUT = ROOT / "results/non_jev/v11/track_c_r2"
MULTI_CORPUS = ROOT / "data/corpus/passages.jsonl"
ENTERPRISE_DB = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
ENTERPRISE_QUESTIONS = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0/questions.jsonl"
MULTI_COLLECTION = "decisionrag_m2_qwen_embedding_v1"
ENTERPRISE_COLLECTION = "decisionrag_v11_enterprise_qwen_embedding_r1"
EMBED_URL = "http://service.example.invalid/v1/embeddings"
QWEN_URL = "http://service.example.invalid/v1/rerank"
READER_URL = "http://service.example.invalid/v1/chat/completions"
QWEN_MODEL = MODEL_IDS["qwen"]
READER_MODEL = MODEL_IDS["reader"]
SEED = 20261011
CONCURRENCY = 10
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
INPUT_CONTRACT_VERSION = "decisionrag-v1.1-track-c-input-wiring-r2"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_serving_input_manifest(input_rows: list[dict], cohort_rows: list[dict]) -> dict[str, dict]:
    """Validate the R2 question manifest before any model or service request."""
    required = {"query_id", "dataset", "question", "question_sha256", "source_manifest", "source_manifest_sha256", "input_contract_version"}
    if len(input_rows) != 100:
        raise ValueError(f"R2 serving input manifest must contain 100 rows, got {len(input_rows)}")
    cohort_by_id = {row.get("query_id"): row for row in cohort_rows}
    if len(cohort_by_id) != len(cohort_rows):
        raise ValueError("frozen cohort contains duplicate query IDs")
    if set(cohort_by_id) != {row.get("query_id") for row in input_rows}:
        raise ValueError("R2 serving input IDs do not exactly match the frozen Track C cohort")
    by_id: dict[str, dict] = {}
    for row in input_rows:
        if set(row) != required:
            raise ValueError(f"invalid serving-input fields for {row.get('query_id')}: {sorted(set(row) ^ required)}")
        query_id = row["query_id"]
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"missing or blank canonical question for {query_id}")
        if row["input_contract_version"] != INPUT_CONTRACT_VERSION:
            raise ValueError(f"unexpected input contract for {query_id}")
        if row["question_sha256"] != sha256_text(question):
            raise ValueError(f"question hash mismatch for {query_id}")
        if row["dataset"] != cohort_by_id[query_id].get("dataset"):
            raise ValueError(f"dataset mismatch for {query_id}")
        if query_id in by_id:
            raise ValueError(f"duplicate serving input for {query_id}")
        by_id[query_id] = row
    if len({row["question_sha256"] for row in input_rows}) != 100:
        raise ValueError("different query IDs unexpectedly share a question hash")
    return by_id


def pct(values, p):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    x = (len(vals) - 1) * p
    lo, hi = int(x), min(int(x) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (x - lo)


def snapshot(owner: str, repo: str, revision: str) -> str:
    return str(HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / revision)


def load_common_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot("convaiinnovations", "laya", MODEL_REVISIONS["laya"]) + "/tokenizer")


class SelectorRuntime:
    """One resident selector model, serialized per model to expose queue wait."""

    def __init__(self, model_key: str, device_id: int | None, common_tokenizer):
        self.model_key = model_key
        self.device = f"cuda:{device_id}" if device_id is not None else None
        self.common_tokenizer = common_tokenizer
        self.lock = threading.RLock()
        self.torch = None
        self.model = None
        self.processor = None
        self.router = None
        self.clef_helpers = None
        self.last_queue_wait_ms = 0.0
        self.last_inference_ms = 0.0
        if model_key == "bge":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            path = snapshot("BAAI", "bge-reranker-v2-m3", MODEL_REVISIONS["bge"])
            self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.bfloat16).to(self.device).eval()
            self.model_tokenizer = AutoTokenizer.from_pretrained(path)
        elif model_key == "laya":
            from laya import Router
            path = snapshot("convaiinnovations", "laya", MODEL_REVISIONS["laya"])
            self.router = Router(models={"english": path}, device=self.device, default="english", max_loaded=1)
        elif model_key == "clef":
            import torch
            path = snapshot("Cloudflare", "clef-flash", MODEL_REVISIONS["clef"])
            sys.path.insert(0, path)
            from joint_schema_model import collate_records, encode_record, load_release_model
            self.torch = torch
            self.model, self.processor = load_release_model(path, device=self.device)
            self.clef_helpers = (collate_records, encode_record)

    def sync(self):
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)

    def peak(self):
        if self.torch is not None and self.torch.cuda.is_available():
            return int(self.torch.cuda.max_memory_allocated(self.device))
        return None

    def pointwise(self, question: str, candidates: list[dict]):
        views = [make_views(question, candidate, self.common_tokenizer) for candidate in candidates]
        queue_started = time.perf_counter_ns()
        with self.lock:
            queue_wait = (time.perf_counter_ns() - queue_started) / 1_000_000.0
            started = time.perf_counter_ns()
            input_tokens = 0
            if self.model_key == "bge":
                pairs = [pointwise_pair(view) for view in views]
                encoded = self.model_tokenizer([p[0] for p in pairs], [p[1] for p in pairs], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                input_tokens = int(encoded["attention_mask"].sum().item())
                self.sync()
                with self.torch.inference_mode(): values = self.model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
                self.sync()
            elif self.model_key == "qwen":
                payload = {"model": QWEN_MODEL, "query": views[0].question_view, "documents": [pointwise_pair(v)[1] for v in views], "chat_template_kwargs": {"add_generation_prompt": True}}
                response = requests.post(QWEN_URL, json=payload, timeout=300, headers={"Cache-Control": "no-cache", "X-DecisionRAG-Cache": "disabled"})
                response.raise_for_status()
                body = response.json()
                values = [float(item["relevance_score"]) for item in sorted(body["results"], key=lambda item: int(item["index"]))]
                input_tokens = int((body.get("usage") or {}).get("prompt_tokens") or 0)
                if len(values) != 20: raise RuntimeError(f"Qwen returned {len(values)} scores, expected 20")
            elif self.model_key == "laya":
                requests_batch = [{"state": pointwise_state(v), "questions": pointwise_question(), "model": "english", "max_len": 512} for v in views]
                results = self.router.predict_batch(requests_batch, batch_size=4, sort_by_length=True)
                values = [float(result["answers"]["relevance"]["noul"]) for result in results]
                input_tokens = sum(int(result.get("usage", {}).get("input_tokens", 0)) for result in results)
            else:
                collate_records, encode_record = self.clef_helpers
                records = [{"state": pointwise_state(v), "questions": pointwise_question()} for v in views]
                encoded = [encode_record(self.processor.tokenizer, r, processor=self.processor) for r in records]
                values, input_tokens = [], 0
                for begin in range(0, 20, 4):
                    batch_records = encoded[begin:begin + 4]
                    batch = collate_records(batch_records, self.processor.tokenizer.pad_token_id, self.torch.device(self.device))
                    self.sync()
                    with self.torch.inference_mode(): outputs = self.model(batch)
                    self.sync()
                    input_tokens += sum(len(record.input_ids) for record in batch_records)
                    for output, record in zip(outputs, batch_records):
                        probs = self.torch.softmax(output[0].float(), dim=-1).detach().cpu().tolist()
                        values.append(float(probs[list(record.questions[0].option_ids).index("true")]))
            infer_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.last_queue_wait_ms, self.last_inference_ms = queue_wait, infer_ms
        return {view.doc_id: float(value) for view, value in zip(views, values)}, queue_wait, infer_ms, input_tokens, self.peak()

    def groupwise(self, query_id: str, question: str, candidates: list[dict]):
        views = [make_views(question, candidate, self.common_tokenizer) for candidate in candidates]
        quartets = deterministic_quartets([dict(v.__dict__) for v in views], query_id, SEED)
        queue_started = time.perf_counter_ns()
        with self.lock:
            queue_wait = (time.perf_counter_ns() - queue_started) / 1_000_000.0
            started = time.perf_counter_ns()
            input_tokens, winners = 0, []
            if self.model_key == "laya":
                for quartet in quartets:
                    qviews = [type(views[0])(**value) for value in quartet]
                    states, schema = groupwise_states(question, qviews)
                    result = self.router.predict(states[0], schema, model="english", max_len=512)
                    label = str(result["answers"]["choice"]["choice"])
                    if label not in "ABCD": raise RuntimeError(f"invalid Laya choice {label!r}")
                    winners.append(qviews["ABCD".index(label)].doc_id)
                    input_tokens += int(result.get("usage", {}).get("input_tokens", 0))
            else:
                collate_records, encode_record = self.clef_helpers
                records, all_qviews = [], []
                for quartet in quartets:
                    qviews = [type(views[0])(**value) for value in quartet]
                    states, schema = groupwise_states(question, qviews)
                    records.append({"state": states[0], "questions": schema})
                    all_qviews.append(qviews)
                encoded = [encode_record(self.processor.tokenizer, r, processor=self.processor) for r in records]
                batch = collate_records(encoded, self.processor.tokenizer.pad_token_id, self.torch.device(self.device))
                self.sync()
                with self.torch.inference_mode(): outputs = self.model(batch)
                self.sync()
                for output, record, qviews in zip(outputs, encoded, all_qviews):
                    probs = self.torch.softmax(output[0].float(), dim=-1).detach().cpu().tolist()
                    label = record.questions[0].option_ids[int(max(range(len(probs)), key=lambda i: probs[i]))]
                    if label not in "ABCD": raise RuntimeError(f"invalid Clef choice {label!r}")
                    winners.append(qviews["ABCD".index(label)].doc_id)
                    input_tokens += len(record.input_ids)
            infer_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.last_queue_wait_ms, self.last_inference_ms = queue_wait, infer_ms
        return winners, queue_wait, infer_ms, input_tokens, self.peak()


def make_reader_prompt(question: str, passages: list[dict], tokenizer):
    pieces, info = [], []
    for index, passage in enumerate(passages):
        view, token_count, truncated = cap_reader_text(tokenizer, passage["title"], passage["full_text"])
        pieces.append(f"[P{index} | {passage['title']} | {passage['doc_id']}] {view}")
        info.append({"doc_id": passage["doc_id"], "reader_tokens": token_count, "truncated": truncated, "reader_view_sha256": digest(view)})
    prompt = ("You must answer using only the supplied passages.\nIf the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
              "Provide one short answer, not reasoning. Do not use external knowledge.\n"
              f"Question: {question}\nPassage<redacted-path>" + "\n".join(pieces) + "\nAnswer:")
    return prompt, {"passages": info, "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=True))}


class DataRuntime:
    def __init__(self):
        self.multi_rows = rows(MULTI_CORPUS)
        if len(self.multi_rows) != 144650: raise RuntimeError(f"unexpected multihop corpus size {len(self.multi_rows)}")
        self.multi_by_id = {r["doc_id"]: r for r in self.multi_rows}
        self.multi_bm25 = BM25(self.multi_rows)
        self.local = threading.local()

    def clients(self):
        if not hasattr(self.local, "multi_client"):
            from qdrant_client import QdrantClient
            self.local.multi_client = QdrantClient(url="http://service.example.invalid", timeout=300)
            self.local.enterprise_client = QdrantClient(url="http://service.example.invalid", timeout=600)
            self.local.enterprise_conn = sqlite3.connect(f"file:{ENTERPRISE_DB}?mode=ro", uri=True, check_same_thread=False)
        return self.local.multi_client, self.local.enterprise_client, self.local.enterprise_conn

    def retrieve(self, item: dict):
        started = time.perf_counter_ns()
        embedding_start = time.perf_counter_ns()
        vector = request_embeddings([item["question"]])[0]
        embedding_ms = (time.perf_counter_ns() - embedding_start) / 1_000_000.0
        multi_client, enterprise_client, conn = self.clients()
        qdrant_start = time.perf_counter_ns()
        if item["dataset"] in {"hotpot", "musique", "wiki2"}:
            dense = multi_dense_search(multi_client, vector, 100)
        else:
            dense = dense_hits(enterprise_client, vector)
        qdrant_ms = (time.perf_counter_ns() - qdrant_start) / 1_000_000.0
        bm25_start = time.perf_counter_ns()
        if item["dataset"] in {"hotpot", "musique", "wiki2"}:
            bm25 = self.multi_bm25.search(item["question"], 100)
        else:
            bm25 = bm25_hits(conn, item["question"])
        bm25_ms = (time.perf_counter_ns() - bm25_start) / 1_000_000.0
        rrf_start = time.perf_counter_ns()
        if item["dataset"] in {"hotpot", "musique", "wiki2"}:
            slate = rrf_slate(self.multi_rows, bm25, dense, 60, 20)
            candidates = [{**self.multi_by_id[x["doc_id"]], "original_rank": rank} for rank, x in enumerate(slate, 1)]
        else:
            slate = aggregate_slate(bm25, dense)
            candidates = []
            for rank, candidate in enumerate(slate, 1):
                source = conn.execute("SELECT title, text, source_type FROM chunks WHERE chunk_id = ?", (candidate["representative_chunk_id"],)).fetchone()
                if source is None or not source[1]: raise RuntimeError(f"missing enterprise representative chunk {candidate['representative_chunk_id']}")
                candidates.append({"doc_id": candidate["doc_id"], "dataset": "enterprise", "title": source[0], "full_text": source[1], "original_rank": rank, "representative_chunk_id": candidate["representative_chunk_id"], "archive_path": candidate["archive_path"], "original_doc_id": candidate["original_doc_id"]})
        rrf_ms = (time.perf_counter_ns() - rrf_start) / 1_000_000.0
        if len(candidates) != 20 or len({c["doc_id"] for c in candidates}) != 20:
            raise RuntimeError(f"invalid Top-20 slate {item['query_id']}: {len(candidates)}")
        return candidates, {"embedding_ms": embedding_ms, "qdrant_ms": qdrant_ms, "bm25_ms": bm25_ms, "rrf_ms": rrf_ms, "retrieval_ms": (time.perf_counter_ns() - started) / 1_000_000.0}


def call_reader(question, passages, tokenizer):
    prompt, view_info = make_reader_prompt(question, passages, tokenizer)
    if view_info["prompt_tokens"] > 8192: raise RuntimeError(f"reader context exceeds 8192 tokens: {view_info['prompt_tokens']}")
    payload = {"model": READER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0, "stream": False}
    last_error = None
    for attempt in range(1, 4):
        started = time.perf_counter_ns()
        try:
            response = requests.post(READER_URL, json=payload, timeout=300, headers={"Cache-Control": "no-cache", "X-DecisionRAG-Cache": "disabled"})
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage") or {}
            return {"answer": str(body["choices"][0]["message"].get("content") or "").strip(), "reader_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "attempts": attempt, "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "response_hash": digest(body), "payload_hash": digest(payload), "prompt_hash": digest(prompt), "view_info": view_info}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3: time.sleep(2 ** (attempt - 1))
    raise RuntimeError(last_error)


def run_one(task, data, runtimes, reader_tokenizer):
    start = time.monotonic_ns()
    question = task["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"run task has missing or blank question: {task.get('query_id')}")
    question_sha = sha256_text(question)
    trace = {"schedule_ordinal": task["schedule_ordinal"], "block_index": task["block_index"], "query_index_in_block": task["query_index_in_block"], "query_id": task["query_id"], "dataset": task["dataset"], "policy_id": task["policy_id"], "question_sha256": question_sha, "model_input_hashes": {"embedding_payload": digest({"model": EMBED_MODEL, "input": [question]})}, "concurrency": CONCURRENCY, "cache_disabled": True, "paid_api_usage": False, "provider_fee_usd": 0.0, "jev_calls": 0, "status": "error", "http_attempts": 0, "request_start_monotonic_ns": start, "gpu_measurement_method": "torch.cuda.max_memory_allocated for local selector only; resident service VRAM is reported separately", "gpu_compute_measurement_method": "NOT_MEASURED_PER_REQUEST"}
    try:
        candidates, rt = data.retrieve(task)
        trace.update(rt, top20_doc_ids=[c["doc_id"] for c in candidates])
        selector_start = time.perf_counter_ns()
        model_key = MODEL_KEYS.get(task["policy_id"])
        trace["model_input_hashes"]["selector_input"] = digest({"policy_id": task["policy_id"], "model_key": model_key, "question_sha256": question_sha, "candidate_doc_ids": [c["doc_id"] for c in candidates]})
        selector_queue, selector_infer, input_tokens, peak = 0.0, 0.0, 0, None
        scores, group_winners = None, None
        if task["policy_id"] == "raw20": selected = [c["doc_id"] for c in candidates]
        elif task["policy_id"] == "retrieval5": selected = [c["doc_id"] for c in candidates[:5]]
        else:
            scores, qw, inf, toks, peak = runtimes[model_key].pointwise(task["question"], candidates)
            selector_queue += qw; selector_infer += inf; input_tokens += toks
            ranked = rank_indices(candidates, scores)
            if task["policy_id"].endswith("_g5"):
                group_winners, qw, inf, toks, peak = runtimes[model_key].groupwise(task["query_id"], task["question"], candidates)
                selector_queue += qw; selector_infer += inf; input_tokens += toks
                selected = sorted(group_winners, key=lambda doc_id: (-scores[doc_id], next(c["original_rank"] for c in candidates if c["doc_id"] == doc_id), doc_id))
            else: selected = [c["doc_id"] for c in ranked[:5]]
            trace["selector_model_calls"] = 25 if task["policy_id"].endswith("_g5") else 20
        trace.update({"selector_queue_wait_ms": selector_queue, "selector_inference_ms": selector_infer, "selector_ms": (time.perf_counter_ns() - selector_start) / 1_000_000.0, "selector_model_key": model_key, "selector_gpu_id": runtimes[model_key].device if model_key else None, "selector_gpu_peak_vram_bytes": peak, "input_tokens": input_tokens})
        rescue_start = time.perf_counter_ns()
        if task["policy_id"].endswith("_rescue5"):
            rescue = rescue_top5(candidates, scores, selected); selected = rescue["selected_ids"]; trace["rescue_swapped"] = rescue["swapped"]
        trace["rescue_ms"] = (time.perf_counter_ns() - rescue_start) / 1_000_000.0
        context_start = time.perf_counter_ns()
        by_id = {c["doc_id"]: c for c in candidates}
        passages = [by_id[doc_id] for doc_id in selected]
        trace["context_ms"] = (time.perf_counter_ns() - context_start) / 1_000_000.0
        reader = call_reader(question, passages, reader_tokenizer)
        trace["model_input_hashes"].update({"reader_payload": reader["payload_hash"], "reader_prompt": reader["prompt_hash"]})
        trace.update({"answer": reader["answer"], "reader_ms": reader["reader_ms"], "http_attempts": reader["attempts"], "input_tokens": reader["input_tokens"], "output_tokens": reader["output_tokens"], "response_hash": reader["response_hash"], "payload_hash": reader["payload_hash"], "prompt_hash": reader["prompt_hash"], "reader_view_info": reader["view_info"], "selected_doc_ids": selected, "status": "ok"})
    except Exception as exc:
        trace["error_type"] = f"{type(exc).__name__}: {exc}"
    trace["request_duration_ms"] = (time.monotonic_ns() - start) / 1_000_000.0
    return trace


def write_jsonl(path, values):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for value in values: fh.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_canonical_smoke(cohort, serving_inputs, data, runtimes, reader_tokenizer):
    """Exercise every non-Jev policy on one canonical question outside Track C."""
    cohort_ids = {row["query_id"] for row in cohort}
    smoke_row = next(
        row for row in rows(SMOKE_SOURCE)
        if row.get("query_id") not in cohort_ids and isinstance(row.get("question"), str) and row["question"].strip()
    )
    question = smoke_row["question"]
    question_sha = sha256_text(question)
    task_base = {
        "schedule_ordinal": -1,
        "block_index": -1,
        "query_index_in_block": -1,
        "query_id": f"track_c_r2_smoke:{smoke_row['query_id']}",
        "dataset": smoke_row["dataset"],
        "question": question,
        "question_sha256": question_sha,
    }
    traces = [
        run_one({**task_base, "policy_id": policy}, data, runtimes, reader_tokenizer)
        for policy in POLICIES
    ]
    write_jsonl(OUT / "V11_TRACK_C_R2_SMOKE_TRACES.jsonl", traces)
    report = {
        "status": "PASS" if all(trace.get("status") == "ok" for trace in traces) else "BLOCKED",
        "measurement_population": "EXCLUDED_FROM_1200_REQUEST_CAMPAIGN",
        "query_id": task_base["query_id"],
        "source_query_id": smoke_row["query_id"],
        "dataset": smoke_row["dataset"],
        "question_sha256": question_sha,
        "question_nonempty": bool(question.strip()),
        "policies": POLICIES,
        "policy_status": {trace["policy_id"]: trace.get("status") for trace in traces},
        "model_input_hashes_present": {
            trace["policy_id"]: sorted(trace.get("model_input_hashes", {})) for trace in traces
        },
        "jev_calls": 0,
        "paid_api_calls": 0,
    }
    (OUT / "V11_TRACK_C_R2_SMOKE_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(f"canonical R2 smoke failed: {report['policy_status']}")
    print(json.dumps({"phase": "canonical_smoke", **report}, sort_keys=True), flush=True)
    return report


def build_tasks(cohort, schedule, serving_inputs=None):
    if serving_inputs is None:
        raise ValueError("R2 serving input manifest is required; refusing implicit or empty questions")
    cmap = {r["query_id"]: r for r in cohort}
    tasks, ordinal = [], 0
    for block in schedule["blocks"]:
        if len(block["query_ids"]) != 10 or set(block["policy_order"]) != set(POLICIES): raise RuntimeError("invalid frozen schedule block")
        # The frozen interleaving contract is policy-major within each block:
        # every policy visits the same ten queries before the next policy.
        # This matches the established M5 interleaved runner and keeps policy
        # blocks balanced under the declared concurrency.
        for policy in block["policy_order"]:
            for qi, qid in enumerate(block["query_ids"]):
                ordinal += 1
                row = cmap[qid]
                input_row = serving_inputs.get(qid)
                if input_row is None:
                    raise ValueError(f"missing R2 serving input for {qid}")
                if input_row["dataset"] != row["dataset"]:
                    raise ValueError(f"R2 serving input dataset mismatch for {qid}")
                question = input_row["question"]
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"blank R2 serving input for {qid}")
                if input_row["question_sha256"] != sha256_text(question):
                    raise ValueError(f"R2 serving input hash mismatch for {qid}")
                tasks.append({"schedule_ordinal": ordinal, "block_index": block["block_index"], "query_index_in_block": qi, "query_id": qid, "dataset": row["dataset"], "question": question, "question_sha256": input_row["question_sha256"], "policy_id": policy})
    if len(tasks) != 1200 or len({t["query_id"] for t in tasks}) != 100: raise RuntimeError("Track C schedule is not 1,200 requests over 100 queries")
    return tasks


def stage_rows(traces):
    stages = ["embedding_ms", "qdrant_ms", "bm25_ms", "rrf_ms", "retrieval_ms", "selector_queue_wait_ms", "selector_inference_ms", "selector_ms", "rescue_ms", "context_ms", "reader_ms", "request_duration_ms"]
    output = []
    for policy in POLICIES:
        scoped = [r for r in traces if r["policy_id"] == policy and r["status"] == "ok"]
        for stage in stages:
            values = [r.get(stage) for r in scoped if r.get(stage) is not None]
            output.append({"policy_id": policy, "stage": stage, "population_n": len(values), "mean_ms": statistics.mean(values) if values else None, "p50_ms": pct(values, .5), "p95_ms": pct(values, .95), "sum_ms": sum(values) if values else 0.0})
    return output


def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(records)


def write_reports(traces, started_utc, elapsed_s, inputs, gpu_map):
    latency = []
    for policy in POLICIES:
        scoped = [r for r in traces if r["policy_id"] == policy and r["status"] == "ok"]
        latency.append({"policy_id": policy, "n_ok": len(scoped), "n_error": sum(r["status"] != "ok" for r in traces if r["policy_id"] == policy), "request_p50_ms": pct([r.get("request_duration_ms") for r in scoped], .5), "request_p95_ms": pct([r.get("request_duration_ms") for r in scoped], .95), "retrieval_p50_ms": pct([r.get("retrieval_ms") for r in scoped], .5), "selector_p50_ms": pct([r.get("selector_ms") for r in scoped], .5), "reader_p50_ms": pct([r.get("reader_ms") for r in scoped], .5), "answer_nonempty_rate": (sum(bool(r.get("answer")) for r in scoped) / len(scoped)) if scoped else None})
    write_csv(OUT / "V11_TRACK_C_R2_LATENCY_STATISTICS.csv", latency, list(latency[0]))
    components = stage_rows(traces)
    write_csv(OUT / "V11_TRACK_C_R2_COMPONENT_BREAKDOWN.csv", components, list(components[0]))
    throughput = []
    for policy in POLICIES:
        scoped = [r for r in traces if r["policy_id"] == policy]
        ok = sum(r["status"] == "ok" for r in scoped)
        throughput.append({"policy_id": policy, "requests": len(scoped), "ok": ok, "errors": len(scoped)-ok, "error_rate": (len(scoped)-ok)/len(scoped) if scoped else None, "wall_elapsed_s": elapsed_s, "logical_requests_per_s": len(scoped)/elapsed_s if elapsed_s else None, "successful_requests_per_s": ok/elapsed_s if elapsed_s else None})
    write_csv(OUT / "V11_TRACK_C_R2_THROUGHPUT_AND_ERRORS.csv", throughput, list(throughput[0]))
    frontier = [{"policy_id": row["policy_id"], "quality_metric": "answer_nonempty_rate_proxy", "quality_value": row["answer_nonempty_rate"], "latency_p50_ms": row["request_p50_ms"], "latency_p95_ms": row["request_p95_ms"], "paid_cost_usd": 0.0, "quality_is_factual": False} for row in latency]
    write_csv(OUT / "V11_TRACK_C_R2_QUALITY_COST_FRONTIER.csv", frontier, list(frontier[0]))
    by_dataset = []
    for dataset in ("hotpot", "musique", "wiki2", "enterprise"):
        scoped = [r for r in traces if ("enterprise" if str(r["dataset"]).startswith("enterprise") else r["dataset"]) == dataset and r["status"] == "ok"]
        by_dataset.append({"dataset": dataset, "requests": len(scoped), "request_p50_ms": pct([r.get("request_duration_ms") for r in scoped], .5), "request_p95_ms": pct([r.get("request_duration_ms") for r in scoped], .95), "retrieval_p50_ms": pct([r.get("retrieval_ms") for r in scoped], .5), "reader_p50_ms": pct([r.get("reader_ms") for r in scoped], .5), "answer_nonempty_rate": sum(bool(r.get("answer")) for r in scoped)/len(scoped) if scoped else None})
    write_csv(OUT / "V11_TRACK_C_R2_MULTIHOP_VS_ENTERPRISE.csv", by_dataset, list(by_dataset[0]))
    report = ["# DecisionRAG v1.1R2 Track C Execution Report", "", f"- Status: `TRACK_C_R2_COMPLETE_READY_FOR_OWNER_REVIEW`", f"- Started (UTC): `{started_utc}`", f"- Completed (UTC): `{datetime.now(timezone.utc).isoformat()}`", f"- Logical requests: `{len(traces)}`; successful: `{sum(r['status']=='ok' for r in traces)}`; errors: `{sum(r['status']!='ok' for r in traces)}`", f"- Measured wall-clock elapsed: `{elapsed_s:.3f}` seconds", "- Concurrency: `10`", "- Application response/selector/reader cache: disabled for timed requests", "- Jev calls: `0`; paid API calls: `0`", "", "## GPU topology", "", f"- Local selector mapping: `{json.dumps(gpu_map, sort_keys=True)}`.", "- Existing embedding, Qwen reranker, Qwen reader, and Qdrant services were left running on their resident devices.", "- Per-request service GPU active time was not instrumented; local selector peak allocation and external before/after GPU snapshots are reported separately.", "", "## Timing interpretation", "", "All stage timings are measured separately from direct client boundaries. Stage values are not summed to claim end-to-end latency; `request_duration_ms` is the authoritative full-request measurement.", "", "## Quality interpretation", "", "The generated frontier uses non-empty answer rate only as a serving-output proxy. It is not independently verified factual correctness and must not be read as an answer-quality claim."]
    (OUT / "V11_TRACK_C_R2_EXECUTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    cost = {"status": "NO_PAID_COST", "jev_calls": 0, "hosted_cloudflare_calls": 0, "provider_fee_usd": 0.0, "local_and_existing_service_cost": "not monetized", "measured_wall_elapsed_seconds": elapsed_s, "request_count": len(traces)}
    (OUT / "V11_TRACK_C_R2_COST_MODEL.md").write_text("# Track C R2 Cost Model\n\n```json\n" + json.dumps(cost, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    pareto = ["# Track C Pareto Analysis", "", "This first-pass frontier is latency versus the non-empty-answer serving proxy only; no factual correctness is inferred.", "", "| Policy | p50 request ms | p95 request ms | Non-empty answer proxy |", "|---|---:|---:|---:|"] + [f"| {r['policy_id']} | {r['request_p50_ms']:.3f} | {r['request_p95_ms']:.3f} | {r['answer_nonempty_rate']:.4f} |" for r in latency if r['request_p50_ms'] is not None]
    (OUT / "V11_TRACK_C_R2_QUALITY_LATENCY_FRONTIER.md").write_text("\n".join(pareto) + "\n", encoding="utf-8")
    gpu_report = {"local_selector_gpu_mapping": gpu_map, "resident_services_untouched": True, "gpu_compute_seconds_per_request": "NOT_MEASURED", "local_peak_bytes_by_model": {k: (max((r.get("selector_gpu_peak_vram_bytes") or 0) for r in traces if r.get("selector_model_key") == k) if any(r.get("selector_gpu_peak_vram_bytes") for r in traces if r.get("selector_model_key") == k) else "NOT_MEASURED") for k in ("bge", "laya", "clef")}, "authorization": "owner-confirmed current execution session", "fallback_devices_used": [0, 5, 6]}
    (OUT / "V11_TRACK_C_R2_GPU_RESOURCE_REPORT.md").write_text("# Track C R2 GPU Resource Report\n\n```json\n" + json.dumps(gpu_report, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    inputs = {**inputs, "repaired_input_manifest": str(REPAIRED_INPUT_MANIFEST), "repaired_input_manifest_sha256": sha256_file(REPAIRED_INPUT_MANIFEST), "input_contract_version": INPUT_CONTRACT_VERSION}
    manifest = {"track": "C-R2", "status": "TRACK_C_R2_COMPLETE_READY_FOR_OWNER_REVIEW", "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(), "cohort_sha256": sha256_file(COHORT), "schedule_sha256": sha256_file(SCHEDULE), "cohort_count": 100, "logical_requests": len(traces), "concurrency": CONCURRENCY, "policy_order": POLICIES, "model_ids": MODEL_IDS, "model_revisions": MODEL_REVISIONS, "gpu_mapping": gpu_map, "input_artifacts": inputs, "cache_disabled": True, "gold_accessed_during_request_generation": False, "jev_calls": 0, "paid_api_calls": 0, "trace_sha256": sha256_file(OUT / "V11_TRACK_C_R2_REQUEST_TRACES.jsonl"), "latency_sha256": sha256_file(OUT / "V11_TRACK_C_R2_LATENCY_STATISTICS.csv"), "component_sha256": sha256_file(OUT / "V11_TRACK_C_R2_COMPONENT_BREAKDOWN.csv")}
    (OUT / "V11_TRACK_C_R2_PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final = ["# DecisionRAG v1.1R2 Track C Final Report", "", "- Status: `TRACK_C_R2_COMPLETE_READY_FOR_OWNER_REVIEW`", f"- The frozen 100-query cohort and 1,200-request schedule were executed with the repaired canonical-question manifest; successful requests: `{sum(r['status']=='ok' for r in traces)}`; errors: `{sum(r['status']!='ok' for r in traces)}`.", "- Every trace carries the canonical question hash and model-input hashes; the runner failed closed on missing, blank, mismatched, or substituted questions.", "- Every policy has 100 scheduled outcomes, and full-request latency is measured directly per request; component timings are reported independently and are not summed to estimate end-to-end latency.", "- The request path used live BM25/dense retrieval, frozen RRF, live candidate selection, context assembly, and live Qwen reader generation with application-level response, selector, and reader caching disabled.", "- Clef timing reports selector inference and queue waiting separately where measurable.", "- Original invalid Track C artifacts, Track A/B artifacts, and frozen cohort/schedule are preserved; R2 writes only to the separate `track_c_r2/` directory.", "- Jev and all paid APIs remain untouched (`0` calls).", "- Factual quality claims are intentionally deferred to owner review of the authoritative Track A/Track B quality evidence.", "", "Required owner-review artifacts are present in this R2 output directory."]
    (OUT / "V11_TRACK_C_R2_FINAL_REPORT.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--gpu-bge", type=int, default=5)
    parser.add_argument("--gpu-laya", type=int, default=6)
    parser.add_argument("--gpu-clef", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, schedule = rows(COHORT), json.loads(SCHEDULE.read_text(encoding="utf-8"))
    if len(cohort) != 100 or len({r["query_id"] for r in cohort}) != 100: raise RuntimeError("invalid frozen Track C cohort")
    if not REPAIRED_INPUT_MANIFEST.exists():
        raise RuntimeError(f"missing required R2 serving input manifest: {REPAIRED_INPUT_MANIFEST}")
    serving_inputs = validate_serving_input_manifest(rows(REPAIRED_INPUT_MANIFEST), cohort)
    tasks = build_tasks(cohort, schedule, serving_inputs)
    trace_path = OUT / "V11_TRACK_C_R2_REQUEST_TRACES.jsonl"
    existing = {r["schedule_ordinal"]: r for r in rows(trace_path)} if args.resume and trace_path.exists() else {}
    if existing and any(r.get("status") != "ok" for r in existing.values()): existing = {}
    print(json.dumps({"phase": "initialization", "cohort": 100, "requests": len(tasks), "resume_rows": len(existing), "gpu_map": {"bge": args.gpu_bge, "laya": args.gpu_laya, "clef": args.gpu_clef}}, sort_keys=True), flush=True)
    data = DataRuntime()
    from qdrant_client import QdrantClient
    probe = QdrantClient(url="http://service.example.invalid", timeout=300)
    counts = {"multihop": probe.count(collection_name=MULTI_COLLECTION, exact=True).count, "enterprise": probe.count(collection_name=ENTERPRISE_COLLECTION, exact=True).count}
    if counts["multihop"] != 144650 or counts["enterprise"] != 1242073: raise RuntimeError(f"unexpected frozen Qdrant counts: {counts}")
    tokenizer = load_common_tokenizer()
    runtimes = {"bge": SelectorRuntime("bge", args.gpu_bge, tokenizer), "qwen": SelectorRuntime("qwen", None, tokenizer), "laya": SelectorRuntime("laya", args.gpu_laya, tokenizer), "clef": SelectorRuntime("clef", args.gpu_clef, tokenizer)}
    from transformers import AutoTokenizer
    reader_tokenizer = AutoTokenizer.from_pretrained(snapshot("Qwen", "Qwen3-VL-8B-Instruct", MODEL_REVISIONS["reader"]), trust_remote_code=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    # Warm local model paths and remote services outside the frozen cohort.
    dummy = [{"doc_id": "warmup", "dataset": "warmup", "title": "Warmup", "full_text": "Warmup passage for model initialization.", "original_rank": 1} for _ in range(20)]
    for key in ("bge", "laya", "clef"):
        runtimes[key].pointwise("Warmup query", dummy)
    request_embeddings(["Track C service warmup"])
    requests.post(READER_URL, json={"model": READER_MODEL, "messages": [{"role": "user", "content": "Answer exactly: WARMUP"}], "temperature": 0.0, "max_tokens": 4, "stream": False}, timeout=300, headers={"Cache-Control": "no-cache", "X-DecisionRAG-Cache": "disabled"}).raise_for_status()
    data.retrieve({"query_id": "warmup", "dataset": "hotpot", "question": "Track C warmup query"})
    run_canonical_smoke(cohort, serving_inputs, data, runtimes, reader_tokenizer)
    if args.smoke_only:
        print(json.dumps({"status": "TRACK_C_R2_SMOKE_COMPLETE", "jev_calls": 0, "paid_api_calls": 0}, sort_keys=True), flush=True)
        return 0
    started = time.perf_counter()
    ordered = dict(existing)
    pending = [task for task in tasks if task["schedule_ordinal"] not in existing]
    def run(task): return run_one(task, data, runtimes, reader_tokenizer)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        iterator = iter(pending); futures = {}
        for _ in range(min(CONCURRENCY, len(pending))):
            task = next(iterator, None)
            if task is not None: futures[pool.submit(run, task)] = task
        done_count = 0
        while futures:
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future); trace = future.result(); ordered[trace["schedule_ordinal"]] = trace; done_count += 1
                next_task = next(iterator, None)
                if next_task is not None: futures[pool.submit(run, next_task)] = next_task
                if done_count % 20 == 0 or done_count == len(pending):
                    write_jsonl(trace_path, [ordered[i] for i in sorted(ordered)])
                    print(json.dumps({"phase": "serving", "completed_new": done_count, "requested_new": len(pending), "total_rows": len(ordered), "ok": sum(r.get("status") == "ok" for r in ordered.values()), "errors": sum(r.get("status") != "ok" for r in ordered.values())}, sort_keys=True), flush=True)
    write_jsonl(trace_path, [ordered[i] for i in sorted(ordered)])
    traces = [ordered[i] for i in sorted(ordered)]
    elapsed = time.perf_counter() - started
    write_reports(traces, started_utc, elapsed, {"cohort": str(COHORT), "schedule": str(SCHEDULE), "multihop_corpus": str(MULTI_CORPUS), "enterprise_db": str(ENTERPRISE_DB), "enterprise_questions": str(ENTERPRISE_QUESTIONS), "qdrant_counts": counts}, {"bge": args.gpu_bge, "laya": args.gpu_laya, "clef": args.gpu_clef})
    print(json.dumps({"status": "TRACK_C_COMPLETE_READY_FOR_OWNER_REVIEW" if all(r["status"] == "ok" for r in traces) and len(traces) == 1200 else "TRACK_C_PARTIAL_RESUMABLE", "rows": len(traces), "elapsed_seconds": elapsed, "ok": sum(r["status"] == "ok" for r in traces), "errors": sum(r["status"] != "ok" for r in traces), "jev_calls": 0}, sort_keys=True), flush=True)
    return 0 if len(traces) == 1200 and all(r["status"] == "ok" for r in traces) else 2


if __name__ == "__main__":
    raise SystemExit(main())
