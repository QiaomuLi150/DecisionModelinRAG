#!/usr/bin/env python3
"""M5 G2 serving measurement: one policy, 100 TEST queries, concurrency 10."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from decisionrag.selection import deterministic_quartets, groupwise_states, make_views, pointwise_pair, pointwise_question, pointwise_state, rank_indices, rescue_top5, sha256_json  # noqa: E402
from m2_retrieve import BM25, dense_search, request_embeddings, rrf_slate  # noqa: E402
from run_m4_reader import build_prompt  # noqa: E402

POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]
MODEL_REVISIONS = {"bge": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "qwen": "4bd860ac4f15ad1897a214615cccc700f8f71818", "laya": "7b928d828b7b0e022f929d9bd2e44165aa270148", "clef": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"}
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
QDRANT_URL = "http://service.example.invalid"
QWEN_URL = "http://service.example.invalid/v1/rerank"
QWEN_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
READER_URL = "http://service.example.invalid/v1/chat/completions"
READER_MODEL = "Qwen/Qwen/Qwen3-VL-8B-Instruct".replace("Qwen/Qwen/", "Qwen/")
SEED = 20261011
CONCURRENCY = 10
N_QUERIES = 100
MODEL_TOPOLOGY_HASH = "a63fa77eeba44396e980d2a2338a01be884b15eae3836c313bec70c0c6506f94"


def snapshot(owner: str, repo: str, revision: str) -> str:
    return str(HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / revision)


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    low, high = int(index), min(int(index) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


class Runtime:
    def __init__(self, model_key: str | None):
        self.model_key = model_key
        self.lock = threading.RLock()
        self.torch = None
        self.common_tokenizer = None
        self.model = None
        self.processor = None
        self.router = None
        self.clef_helpers = None
        self.last_queue_wait_ms = 0.0
        self.last_inference_ms = 0.0
        if model_key:
            from transformers import AutoTokenizer
            self.common_tokenizer = AutoTokenizer.from_pretrained(snapshot("convaiinnovations", "laya", MODEL_REVISIONS["laya"]) + "/tokenizer")
        if model_key == "bge":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            path = snapshot("BAAI", "bge-reranker-v2-m3", MODEL_REVISIONS["bge"])
            # Common Laya-tokenized views define the frozen text contract;
            # BGE itself must consume those texts through its native tokenizer.
            self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
            self.model_tokenizer = AutoTokenizer.from_pretrained(path)
        elif model_key == "laya":
            from laya import Router
            path = snapshot("convaiinnovations", "laya", MODEL_REVISIONS["laya"])
            self.router = Router(models={"english": path}, device="cuda", default="english", max_loaded=1)
        elif model_key == "clef":
            import torch
            path = snapshot("Cloudflare", "clef-flash", MODEL_REVISIONS["clef"])
            sys.path.insert(0, path)
            from joint_schema_model import collate_records, encode_record, load_release_model
            self.torch = torch
            self.model, self.processor = load_release_model(path, device="cuda")
            self.clef_helpers = (collate_records, encode_record)

    def sync(self):
        if self.torch is not None and self.torch.cuda.is_available(): self.torch.cuda.synchronize()

    def gpu_peak(self):
        if self.torch is not None and self.torch.cuda.is_available():
            return int(self.torch.cuda.max_memory_allocated())
        return None

    def pointwise(self, question: str, candidates: list[dict]) -> tuple[dict[str, float], float, int]:
        views = [make_views(question, candidate, self.common_tokenizer) for candidate in candidates]
        start = time.perf_counter_ns()
        queue_start = time.perf_counter_ns()
        with self.lock:
            self.last_queue_wait_ms = (time.perf_counter_ns() - queue_start) / 1_000_000.0
            inference_start = time.perf_counter_ns()
            if self.model_key == "bge":
                pairs = [pointwise_pair(view) for view in views]
                encoded = self.model_tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                self.sync()
                with self.torch.inference_mode(): values = self.model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
                self.sync()
                input_tokens = int(encoded["attention_mask"].sum().item())
            elif self.model_key == "qwen":
                payload = {
                    "model": QWEN_MODEL,
                    "query": views[0].question_view,
                    "documents": [pointwise_pair(view)[1] for view in views],
                    # Keep vLLM serialization identical to the official
                    # checkpoint scorer's add_generation_prompt=True path.
                    "chat_template_kwargs": {"add_generation_prompt": True},
                }
                response = requests.post(QWEN_URL, json=payload, timeout=300)
                response.raise_for_status()
                body = response.json()
                values = [float(item["relevance_score"]) for item in sorted(body["results"], key=lambda item: int(item["index"]))]
                input_tokens = int((body.get("usage") or {}).get("prompt_tokens") or 0)
            elif self.model_key == "laya":
                requests_batch = [{"state": pointwise_state(view), "questions": pointwise_question(), "model": "english", "max_len": 512} for view in views]
                results = self.router.predict_batch(requests_batch, batch_size=4, sort_by_length=True)
                values = [float(result["answers"]["relevance"]["noul"]) for result in results]
                input_tokens = sum(int(result.get("usage", {}).get("input_tokens", 0)) for result in results)
            elif self.model_key == "clef":
                collate_records, encode_record = self.clef_helpers
                records = [{"state": pointwise_state(view), "questions": pointwise_question()} for view in views]
                encoded = [encode_record(self.processor.tokenizer, record, processor=self.processor) for record in records]
                values = []
                input_tokens = 0
                for begin in range(0, 20, 4):
                    batch_records = encoded[begin : begin + 4]
                    batch = collate_records(batch_records, self.processor.tokenizer.pad_token_id, self.torch.device("cuda"))
                    self.sync()
                    with self.torch.inference_mode(): outputs = self.model(batch)
                    self.sync()
                    input_tokens += sum(len(record.input_ids) for record in batch_records)
                    for output, record in zip(outputs, encoded[begin : begin + 4]):
                        probabilities = self.torch.softmax(output[0].float(), dim=-1).detach().cpu().tolist()
                        option_ids = list(record.questions[0].option_ids)
                        values.append(float(probabilities[option_ids.index("true")]))
            else:
                raise RuntimeError(self.model_key)
            self.last_inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000.0
        return {view.doc_id: float(value) for view, value in zip(views, values)}, (time.perf_counter_ns() - start) / 1_000_000.0, input_tokens

    def groupwise(self, query_id: str, question: str, candidates: list[dict]) -> tuple[list[str], float, int]:
        if self.model_key not in ("laya", "clef"): raise RuntimeError("groupwise requires typed model")
        views = [make_views(question, candidate, self.common_tokenizer) for candidate in candidates]
        quartets = deterministic_quartets([dict(view.__dict__) for view in views], query_id, SEED)
        start = time.perf_counter_ns()
        input_tokens = 0
        queue_start = time.perf_counter_ns()
        with self.lock:
            self.last_queue_wait_ms = (time.perf_counter_ns() - queue_start) / 1_000_000.0
            inference_start = time.perf_counter_ns()
            if self.model_key == "laya":
                winners = []
                for quartet in quartets:
                    quartet_views = [type(views[0])(**value) for value in quartet]
                    states, schema = groupwise_states(question, quartet_views)
                    result = self.router.predict(states[0], schema, model="english", max_len=512)
                    label = result["answers"]["choice"]["choice"]
                    winners.append(quartet_views["ABCD".index(label)].doc_id)
                    input_tokens += int(result.get("usage", {}).get("input_tokens", 0))
            else:
                collate_records, encode_record = self.clef_helpers
                records = []
                quartet_views_all = []
                for quartet in quartets:
                    quartet_views = [type(views[0])(**value) for value in quartet]
                    states, schema = groupwise_states(question, quartet_views)
                    records.append({"state": states[0], "questions": schema})
                    quartet_views_all.append(quartet_views)
                encoded = [encode_record(self.processor.tokenizer, record, processor=self.processor) for record in records]
                batch = collate_records(encoded, self.processor.tokenizer.pad_token_id, self.torch.device("cuda"))
                self.sync()
                with self.torch.inference_mode(): outputs = self.model(batch)
                self.sync()
                winners = []
                for output, record, quartet_views in zip(outputs, encoded, quartet_views_all):
                    probabilities = self.torch.softmax(output[0].float(), dim=-1).detach().cpu().tolist()
                    label = record.questions[0].option_ids[int(max(range(len(probabilities)), key=lambda i: probabilities[i]))]
                    winners.append(quartet_views["ABCD".index(label)].doc_id)
                    input_tokens += len(record.input_ids)
            self.last_inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000.0
        return winners, (time.perf_counter_ns() - start) / 1_000_000.0, input_tokens


def load_serving_queries() -> list[dict]:
    path = ROOT / "data/manifests/m5_serving_query_ids.jsonl"
    questions = load_rows(ROOT / "data/manifests/questions.jsonl")
    by_dataset = {dataset: sorted([row for row in questions if row["dataset"] == dataset and row["split"] == "test"], key=lambda row: row["query_id"]) for dataset in ("hotpot", "musique", "wiki2")}
    if not path.exists():
        selected = by_dataset["hotpot"][:34] + by_dataset["musique"][:33] + by_dataset["wiki2"][:33]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in selected: handle.write(json.dumps({"query_id": row["query_id"], "dataset": row["dataset"], "selection": "first_sorted_test_per_dataset", "frozen_seed": SEED}) + "\n")
    ids = load_rows(path)
    qmap = {row["query_id"]: row for row in questions}
    if len(ids) != N_QUERIES or len({row["query_id"] for row in ids}) != N_QUERIES: raise RuntimeError("serving cohort must contain exactly 100 distinct IDs")
    return [qmap[row["query_id"]] for row in ids]


def run_one(item, runtime: Runtime, corpus, bm25, client, tokenizer, policy: str, schedule_meta: dict | None = None) -> dict:
    start_ns = time.monotonic_ns()
    trace = {"query_id": item["query_id"], "dataset": item["dataset"], "policy_id": policy, "concurrency": CONCURRENCY, "request_start_monotonic_ns": start_ns, "cache_disabled": True, "paid_api_usage": False, "provider_fee_usd": 0.0, "http_attempts": 0, "status": "error", "error_type": None, "gpu_peak_vram_bytes": None, "gpu_measurement_method": "torch.cuda.max_memory_allocated for local selector; service VRAM not attributed per request", "gpu_compute_seconds": None, "gpu_compute_measurement_method": "NOT_MEASURED: per-request CUDA active-time events are not instrumented in this serving runner", "model_topology_hash": MODEL_TOPOLOGY_HASH, "input_tokens": None, "output_tokens": None, "schedule": schedule_meta or {}}
    try:
        retrieval_start = time.perf_counter_ns()
        embedding_start = time.perf_counter_ns()
        query_vector = request_embeddings([item["question"]])[0]
        trace["embedding_ms"] = (time.perf_counter_ns() - embedding_start) / 1_000_000.0
        qdrant_start = time.perf_counter_ns()
        dense_hits = dense_search(client, query_vector, 100)
        trace["qdrant_ms"] = (time.perf_counter_ns() - qdrant_start) / 1_000_000.0
        bm25_start = time.perf_counter_ns()
        bm25_hits = bm25.search(item["question"], 100)
        trace["bm25_ms"] = (time.perf_counter_ns() - bm25_start) / 1_000_000.0
        rrf_start = time.perf_counter_ns()
        slate = rrf_slate(list(corpus.values()), bm25_hits, dense_hits, 60, 20)
        trace["rrf_ms"] = (time.perf_counter_ns() - rrf_start) / 1_000_000.0
        candidates = []
        for rank, scored in enumerate(slate, start=1):
            source = corpus[scored["doc_id"]]
            candidates.append({"doc_id": source["doc_id"], "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": rank})
        trace["retrieval_ms"] = (time.perf_counter_ns() - retrieval_start) / 1_000_000.0
        selector_start = time.perf_counter_ns()
        trace["selector_queue_wait_ms"] = 0.0
        trace["selector_inference_ms"] = 0.0
        model_key = {"bge_p5": "bge", "bge_rescue5": "bge", "qwen_p5": "qwen", "qwen_rescue5": "qwen", "laya_p5": "laya", "laya_rescue5": "laya", "laya_g5": "laya", "clef_p5": "clef", "clef_rescue5": "clef", "clef_g5": "clef"}.get(policy)
        scores = None
        group_winners = None
        input_tokens = 0
        if policy == "raw20": selected = [candidate["doc_id"] for candidate in candidates]
        elif policy == "retrieval5": selected = [candidate["doc_id"] for candidate in candidates[:5]]
        else:
            scores, point_ms, point_tokens = runtime.pointwise(item["question"], candidates)
            input_tokens += point_tokens
            trace["selector_queue_wait_ms"] += runtime.last_queue_wait_ms
            trace["selector_inference_ms"] += runtime.last_inference_ms
            ranked = rank_indices(candidates, scores)
            if policy.endswith("_g5"):
                group_winners, group_ms, group_tokens = runtime.groupwise(item["query_id"], item["question"], candidates)
                input_tokens += group_tokens
                trace["selector_queue_wait_ms"] += runtime.last_queue_wait_ms
                trace["selector_inference_ms"] += runtime.last_inference_ms
                selected = sorted(group_winners, key=lambda doc_id: (-scores[doc_id], next(c["original_rank"] for c in candidates if c["doc_id"] == doc_id), doc_id))
            else:
                selected = [candidate["doc_id"] for candidate in ranked[:5]]
            trace["selector_model_calls"] = 25 if policy.endswith("_g5") else 20
        trace["selector_ms"] = (time.perf_counter_ns() - selector_start) / 1_000_000.0
        rescue_start = time.perf_counter_ns()
        if policy.endswith("_rescue5"):
            rescue = rescue_top5(candidates, scores, selected)
            selected = rescue["selected_ids"]
            trace["rescue_swapped"] = rescue["swapped"]
        trace["rescue_ms"] = (time.perf_counter_ns() - rescue_start) / 1_000_000.0
        reader_start = time.perf_counter_ns()
        passages = [corpus[doc_id] for doc_id in selected]
        prompt, view_info = build_prompt(item["question"], passages, tokenizer)
        payload = {"model": READER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0, "stream": False}
        body = None
        for attempt in range(1, 4):
            trace["http_attempts"] += 1
            response = requests.post(READER_URL, json=payload, timeout=300, headers={"Cache-Control": "no-cache", "X-DecisionRAG-Cache": "disabled"})
            if response.status_code == 200:
                body = response.json(); break
            if attempt == 3: response.raise_for_status()
        trace["reader_ms"] = (time.perf_counter_ns() - reader_start) / 1_000_000.0
        trace["answer"] = str(body["choices"][0]["message"].get("content") or "").strip()
        trace["input_tokens"] = (body.get("usage") or {}).get("prompt_tokens")
        trace["output_tokens"] = (body.get("usage") or {}).get("completion_tokens")
        trace["selected_doc_ids"] = selected
        trace["reader_view_info"] = view_info
        trace["status"] = "ok"
        trace["response_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        trace["gpu_peak_vram_bytes"] = runtime.gpu_peak()
    except Exception as exc:
        trace["error_type"] = f"{type(exc).__name__}: {exc}"
    trace["request_duration_ms"] = (time.monotonic_ns() - start_ns) / 1_000_000.0
    return trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, required=True)
    args = parser.parse_args()
    queries = load_serving_queries()
    questions = {row["query_id"]: row for row in load_rows(ROOT / "data/manifests/questions.jsonl")}
    corpus_rows = load_rows(ROOT / "data/corpus/passages.jsonl")
    corpus = {row["doc_id"]: row for row in corpus_rows}
    bm25 = BM25(corpus_rows)
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, timeout=300)
    from transformers import AutoTokenizer
    reader_tokenizer = AutoTokenizer.from_pretrained(str(Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")), trust_remote_code=True)
    model_key = {"bge_p5": "bge", "bge_rescue5": "bge", "qwen_p5": "qwen", "qwen_rescue5": "qwen", "laya_p5": "laya", "laya_rescue5": "laya", "laya_g5": "laya", "clef_p5": "clef", "clef_rescue5": "clef", "clef_g5": "clef"}.get(args.policy)
    runtime = Runtime(model_key)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(run_one, item, runtime, corpus, bm25, client, reader_tokenizer, args.policy) for item in queries]
        traces = [future.result() for future in futures]
    out = ROOT / "data/results/m5" / f"{args.policy}_serving_traces.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for trace in traces: handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"policy_id": args.policy, "queries": len(traces), "concurrency": CONCURRENCY, "status_counts": {status: sum(row["status"] == status for row in traces) for status in {row["status"] for row in traces}}, "request_p50_ms": percentile([row["request_duration_ms"] for row in traces], .5), "request_p95_ms": percentile([row["request_duration_ms"] for row in traces], .95), "error_rate": sum(row["status"] != "ok" for row in traces) / len(traces), "elapsed_wall_s": time.time() - started, "cache_disabled": True, "gpu_model_key": model_key, "jev_calls": 0, "hosted_cloudflare_calls": 0}
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["error_rate"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
