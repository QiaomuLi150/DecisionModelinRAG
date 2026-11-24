#!/usr/bin/env python3
"""Run the frozen 100-query G2 workload in randomized policy blocks.

BM25, the Qdrant client/index and reader tokenizer are resident for the full
run.  Local selector models are resident for their model-family windows.  Each
policy receives the same 100 frozen TEST queries at concurrency 10; policy
blocks are shuffled independently for each of ten blocks and the schedule is
recorded with every trace.
"""
from __future__ import annotations

import concurrent.futures
import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_m5_serving import (  # noqa: E402
    CONCURRENCY,
    MODEL_TOPOLOGY_HASH,
    N_QUERIES,
    POLICIES,
    QDRANT_URL,
    SEED,
    Runtime,
    load_rows,
    load_serving_queries,
    run_one,
)
from m2_retrieve import BM25  # noqa: E402


MODEL_FOR_POLICY = {"bge_p5": "bge", "bge_rescue5": "bge", "qwen_p5": "qwen", "qwen_rescue5": "qwen", "laya_p5": "laya", "laya_rescue5": "laya", "laya_g5": "laya", "clef_p5": "clef", "clef_rescue5": "clef", "clef_g5": "clef"}
FAMILIES = [("controls", ["raw20", "retrieval5"]), ("bge", ["bge_p5", "bge_rescue5"]), ("qwen", ["qwen_p5", "qwen_rescue5"]), ("laya", ["laya_p5", "laya_g5", "laya_rescue5"]), ("clef", ["clef_p5", "clef_g5", "clef_rescue5"])]
OUT = ROOT / "data/results/audit/quality/m5_interleaved"


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    low, high = int(index), min(int(index) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = load_serving_queries()
    if len(queries) != N_QUERIES:
        raise RuntimeError("frozen serving cohort must contain 100 queries")
    corpus_rows = load_rows(ROOT / "data/corpus/passages.jsonl")
    corpus = {row["doc_id"]: row for row in corpus_rows}
    # These objects stay resident and are shared by every policy family.
    bm25 = BM25(corpus_rows)
    from qdrant_client import QdrantClient
    from transformers import AutoTokenizer
    client = QdrantClient(url=QDRANT_URL, timeout=300)
    reader_tokenizer = AutoTokenizer.from_pretrained(str(Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")), trust_remote_code=True)
    rng = random.Random(SEED)
    query_order = list(queries)
    rng.shuffle(query_order)
    blocks = [query_order[index * 10 : (index + 1) * 10] for index in range(10)]
    schedule = {"seed": SEED, "blocks": [], "concurrency": CONCURRENCY, "query_cohort": [item["query_id"] for item in queries], "resident_indexes": {"bm25": True, "qdrant_client": True, "qdrant_collection": "decisionrag_m2_qwen_embedding_v1"}, "model_topology_hash": MODEL_TOPOLOGY_HASH}
    all_traces = {policy: [] for policy in POLICIES}
    started = time.time()
    for family_name, family_policies in FAMILIES:
        model_key = MODEL_FOR_POLICY.get(family_policies[0])
        runtime = Runtime(model_key)
        try:
            for block_index, block in enumerate(blocks):
                policy_order = list(family_policies)
                rng.shuffle(policy_order)
                schedule["blocks"].append({"family": family_name, "block_index": block_index, "policy_order": policy_order, "query_ids": [item["query_id"] for item in block]})
                for policy in policy_order:
                    block_start = time.time()
                    metadata = {"family": family_name, "block_index": block_index, "policy_order_index": policy_order.index(policy), "block_query_count": len(block), "block_start_wall": block_start}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                        futures = [pool.submit(run_one, item, runtime, corpus, bm25, client, reader_tokenizer, policy, metadata) for item in block]
                        completed = [future.result() for future in futures]
                    all_traces[policy].extend(completed)
                    print(json.dumps({"family": family_name, "block": block_index, "policy": policy, "completed": len(completed), "elapsed_s": time.time() - block_start}), flush=True)
        finally:
            del runtime
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    for policy, traces in all_traces.items():
        if len(traces) != N_QUERIES or {row["query_id"] for row in traces} != {item["query_id"] for item in queries}:
            raise RuntimeError(f"invalid interleaved trace cohort for {policy}: {len(traces)}")
        path = OUT / f"{policy}_traces.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
        ok = [trace for trace in traces if trace["status"] == "ok"]
        summary = {"policy_id": policy, "queries": len(traces), "concurrency": CONCURRENCY, "status_counts": {status: sum(row["status"] == status for row in traces) for status in sorted({row["status"] for row in traces})}, "request_p50_ms": percentile([row["request_duration_ms"] for row in traces], .5), "request_p95_ms": percentile([row["request_duration_ms"] for row in traces], .95), "stage_p50_p95_ms": {stage: [percentile([row[stage] for row in ok], .5), percentile([row[stage] for row in ok], .95)] for stage in ("embedding_ms", "qdrant_ms", "bm25_ms", "rrf_ms", "retrieval_ms", "selector_queue_wait_ms", "selector_inference_ms", "selector_ms", "rescue_ms", "reader_ms")}, "error_rate": sum(row["status"] != "ok" for row in traces) / len(traces), "cache_disabled": all(row.get("cache_disabled") is True for row in traces), "jev_calls": 0, "hosted_cloudflare_calls": 0}
        (OUT / f"{policy}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    schedule["elapsed_wall_s"] = time.time() - started
    schedule["status"] = "PASS" if all(len(rows) == N_QUERIES and all(row["status"] == "ok" for row in rows) for rows in all_traces.values()) else "FAIL"
    (OUT / "INTERLEAVED_SCHEDULE.json").write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "status": schedule["status"], "elapsed_wall_s": schedule["elapsed_wall_s"], "policies": len(all_traces), "jev_calls": 0}, indent=2))
    return 0 if schedule["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
