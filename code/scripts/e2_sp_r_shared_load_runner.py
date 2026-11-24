#!/usr/bin/env python3
"""Run the R3 open-loop workload under explicit shared-GPU conditions.

This is a versioned operational follow-up to E2-S/P-R.  The frozen R3 design
requires isolation; the owner later authorized shared GPUs.  Accordingly this
runner performs the real open-loop workload but marks every result
``SHARED_CONTENTION_LIMITED`` and never presents it as an isolated-capacity
measurement.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_MODE = os.environ.get("E2_SP_EXECUTION_MODE", "shared")
GPU = int(os.environ.get("E2_N_PHYSICAL_GPU_ID", "0"))
OUT = Path(os.environ.get("E2_SP_OUTPUT_DIR", str(ROOT / "results/non_jev/v2/e2_sp_r/shared_gpu0/r3")))
ISOLATED = RUN_MODE == "isolated"
RESOURCE_MODE = "isolated_gpu1" if ISOLATED else "shared_gpu0_contention_limited"
CAPACITY_VALIDITY = "VALID_ISOLATED_CAPACITY" if ISOLATED else "NOT_VALID_FOR_ISOLATED_CAPACITY"
RUN_STATUS = "ISOLATED_CAPACITY" if ISOLATED else "SHARED_CONTENTION_LIMITED"
TRACE_PREFIX = "E2_SP_R_R3_ISOLATED_GPU1" if ISOLATED else "E2_SP_R_R3_SHARED"
SEED = 20261011
REQUESTS_PER_RATE = 40
CONCURRENCY = 8
MODEL_ORDER = ("bge_pointwise", "jina35", "nimble", "gte", "pplx_decider")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def gpu_snapshot() -> dict:
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        devices = {}
        for line in proc.stdout.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 7:
                try:
                    idx = int(fields[0])
                    devices[idx] = {"index": idx, "name": fields[1], "memory_total_mib": int(fields[2]), "memory_used_mib": int(fields[3]), "memory_free_mib": int(fields[4]), "utilization_gpu_percent": int(fields[5]), "power_draw_w": fields[6]}
                except ValueError:
                    pass
        return {"status": "PASS" if proc.returncode == 0 else "UNAVAILABLE", "command": " ".join(cmd), "devices": devices, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "command": " ".join(cmd), "devices": {}, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def compute_apps() -> list[str]:
    cmd = ["nvidia-smi", "--query-compute-apps=gpu_uuid,gpu_name,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception as exc:
        return [f"UNAVAILABLE:{type(exc).__name__}:{exc}"]


def load_cal_items() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_sp_gpu5_runner import cal_items
    rows = cal_items()
    if len(rows) != 30 or len({row["query_id"] for row in rows}) != 30:
        raise RuntimeError("CAL workload is not the frozen 30-query population")
    return rows


def make_views(item: dict, tokenizer):
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import make_views
    return [make_views(item["question"], candidate, tokenizer) for candidate in item["candidates"]]


def model_docs(views):
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import pointwise_pair
    return [pointwise_pair(view)[1] for view in views]


def tensor_fingerprint(value):
    if not hasattr(value, "detach"):
        return repr(value)
    tensor = value.detach().to("cpu").contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": sha_bytes(tensor.numpy().tobytes())}


class AdapterBase:
    model_id = ""
    revision = ""

    def __init__(self):
        self.lock = threading.Lock()

    def warmup(self, items):
        for item in items[:2]:
            self.score_item(item)

    def close(self):
        pass


class BGEAdapter(AdapterBase):
    model_id = "BAAI/bge-reranker-v2-m3"
    revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

    def __init__(self):
        super().__init__()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        sys.path.insert(0, str(ROOT / "scripts"))
        from e2_n_specialist_runner import snapshot, tensor_fingerprint as tf
        from run_m3_selectors import common_tokenizer
        self.torch = torch
        self.tf = tf
        self.model_path = snapshot({"owner": "BAAI", "repo": "bge-reranker-v2-m3", "revision": self.revision})
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.view_tokenizer = common_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
        self.forward_count = 0
        self.captured = []
        original = self.model.forward

        def counted(*args, **kwargs):
            self.forward_count += 1
            self.captured.append({"args": [tf(v) for v in args], "kwargs": {k: tf(v) for k, v in kwargs.items()}})
            return original(*args, **kwargs)

        self.model.forward = counted

    def score_item(self, item):
        import torch
        started = time.perf_counter_ns()
        views = make_views(item, self.view_tokenizer)
        pairs = model_docs(views)
        before = self.forward_count
        captured_before = len(self.captured)
        encoded = self.model_tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            scores = self.model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
        actual = self.captured[captured_before:]
        return {"scores": [float(x) for x in scores], "actual_model_input_sha256": sha_json(actual), "raw_response_sha256": sha_json(scores), "physical_forward_calls": self.forward_count - before, "input_tokens": int(encoded["attention_mask"].sum().item()), "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "candidate_ids": [view.doc_id for view in views], "selection_text_sha256": [view.selection_text_sha256 for view in views]}

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
        gc.collect()


class SpecialistAdapter(AdapterBase):
    def __init__(self, model_key: str):
        super().__init__()
        sys.path.insert(0, str(ROOT / "scripts"))
        from e2_n_specialist_runner import MODELS, NativeAdapter
        self.model_key = model_key
        self.spec = MODELS[model_key]
        self.native = NativeAdapter(self.spec)
        self.model_id = self.spec["id"]
        self.revision = self.spec["revision"]

    def score_item(self, item):
        started = time.perf_counter_ns()
        views = make_views(item, self.native.tokenizer)
        documents = model_docs(views)
        scores, telemetry = self.native.score(views[0].question_view, documents)
        telemetry["duration_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
        telemetry["scores"] = scores
        telemetry["raw_response_sha256"] = sha_json(scores)
        telemetry["candidate_ids"] = [view.doc_id for view in views]
        telemetry["selection_text_sha256"] = [view.selection_text_sha256 for view in views]
        return telemetry

    def close(self):
        self.native.close()
        gc.collect()


class NimbleAdapter(AdapterBase):
    model_id = "bespokelabs/Bespoke-Nimble-9B-v2"
    revision = "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe"

    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(ROOT / "scripts"))
        from e2_n_nimble import ADAPTER, TEMPERATURE
        sys.path.insert(0, str(ADAPTER))
        from inference import ParallelScorer
        self.scorer = ParallelScorer(ADAPTER)
        if abs(self.scorer.temperature - TEMPERATURE) > 1e-12:
            raise RuntimeError("Nimble temperature mismatch")
        self.tokenizer = self.scorer.tokenizer
        self.forward_count = 0
        self.captured = []
        original = self.scorer.model.forward

        def counted(*args, **kwargs):
            self.forward_count += 1
            self.captured.append({"args": [tensor_fingerprint(v) for v in args], "kwargs": {k: tensor_fingerprint(v) for k, v in kwargs.items()}})
            return original(*args, **kwargs)

        self.scorer.model.forward = counted

    def score_item(self, item):
        started = time.perf_counter_ns()
        views = make_views(item, self.tokenizer)
        ids = [view.doc_id for view in views]
        choices = [f"candidate_{i:02d}" for i in range(20)]
        descriptions = {choice: f"Candidate ID {doc_id}. Title and passage: {view.title}\n{view.passage_view}" for choice, doc_id, view in zip(choices, ids, views)}
        schema = {"selected_candidate": {"type": "enum", "choices": choices, "description": "Select the candidate passage that best satisfies the query and contains useful evidence.", "choice_descriptions": descriptions}}
        before = self.forward_count
        captured_before = len(self.captured)
        result = self.scorer.score(context=f"Query: {views[0].question_view}", schema=schema)
        probs = result["fields"]["selected_candidate"]["probabilities"]
        scores = [float(probs[choice]) for choice in choices]
        actual = self.captured[captured_before:]
        return {"scores": scores, "actual_model_input_sha256": sha_json(actual), "raw_response_sha256": sha_json(result), "physical_forward_calls": self.forward_count - before, "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "candidate_ids": ids, "selection_text_sha256": [view.selection_text_sha256 for view in views]}

    def close(self):
        del self.scorer
        import torch
        torch.cuda.empty_cache()
        gc.collect()


class PPLXAdapter(AdapterBase):
    model_id = "perplexity-ai/pplx-decider-v1.1-27b"
    revision = "14c9d90137962d300f3de95633f7b6535e80c5ce"

    def __init__(self):
        super().__init__()
        import torch
        checkpoint = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce")
        sys.path.insert(0, str(checkpoint / "source" / "src"))
        from autojev.model import DecisionModel
        self.torch = torch
        self.model = DecisionModel(checkpoint, device="cuda", max_image_pixels=512 * 512)
        self.tokenizer = self.model.processor.tokenizer
        self.forward_count = 0
        self.captured = []
        original = self.model.backbone.forward

        def counted(*args, **kwargs):
            self.forward_count += 1
            self.captured.append({"args": [tensor_fingerprint(v) for v in args], "kwargs": {k: tensor_fingerprint(v) for k, v in kwargs.items()}})
            return original(*args, **kwargs)

        self.model.backbone.forward = counted

    def score_item(self, item):
        started = time.perf_counter_ns()
        views = make_views(item, self.tokenizer)
        ids = [view.doc_id for view in views]
        criteria = {doc_id: f"Title: {view.title}\nPassage: {view.passage_view}" for doc_id, view in zip(ids, views)}
        request = {"state": f"Query: {views[0].question_view}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {views[0].question_view}", "criteria": criteria}}
        before = self.forward_count
        captured_before = len(self.captured)
        probabilities = self.model.predict([request], batch_size=1)[0]
        scores = [float(value) for value in probabilities]
        actual = self.captured[captured_before:]
        return {"scores": scores, "actual_model_input_sha256": sha_json(actual), "raw_response_sha256": sha_json(probabilities), "physical_forward_calls": self.forward_count - before, "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "candidate_ids": ids, "selection_text_sha256": [view.selection_text_sha256 for view in views]}

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
        gc.collect()


def make_adapter(model: str):
    if model == "bge_pointwise":
        return BGEAdapter()
    if model == "jina35":
        return SpecialistAdapter("jina35")
    if model == "gte":
        return SpecialistAdapter("gte")
    if model == "nimble":
        return NimbleAdapter()
    if model == "pplx_decider":
        return PPLXAdapter()
    raise ValueError(model)


def read_rate_grid(model: str) -> list[float]:
    if os.environ.get("E2_SP_R_RATE_SOURCE") == "frozen_protocol":
        protocol = json.loads((ROOT / "results/non_jev/v2/e2_sp_r/E2_SP_R_FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
        return [float(value) for value in protocol["open_loop"]["rate_grid_rps"][model]]
    path = ROOT / "results/non_jev/v2/e2_sp_r/E2_SP_R_SHARED_CAL_TELEMETRY.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row["model"]: row for row in csv.DictReader(handle)}
    p50 = float(rows[model]["p50_duration_ms"])
    base = 1000.0 / p50
    return [round(max(0.01, base * multiplier), 6) for multiplier in (0.25, 0.5, 1.0, 2.0)]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_rate(adapter, model: str, items: list[dict], rate: float, rate_index: int) -> tuple[dict, list[dict]]:
    rng = random.Random(SEED + rate_index * 1009 + sum(ord(c) for c in model))
    offsets = []
    elapsed = 0.0
    for _ in range(REQUESTS_PER_RATE):
        elapsed += rng.expovariate(rate)
        offsets.append(elapsed)
    before_gpu = gpu_snapshot()
    apps_before = compute_apps()
    started = time.perf_counter()
    request_rows = []
    futures = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY)

    def work(index: int, item: dict, submit_at: float, offset: float):
        worker_start = time.perf_counter()
        try:
            with adapter.lock:
                model_start = time.perf_counter()
                queue_wait_ms = (model_start - submit_at) * 1000.0
                result = adapter.score_item(item)
                done = time.perf_counter()
            return {"record_type": "request", "model": model, "rate_index": rate_index, "request_index": index, "query_id": item["query_id"], "arrival_offset_s": offset, "submit_offset_s": submit_at - started, "worker_offset_s": worker_start - started, "queue_wait_ms": queue_wait_ms, "duration_ms": (done - submit_at) * 1000.0, "model_execution_ms": result.pop("duration_ms"), "status": "ok", "error": None, "candidate_ids": result.pop("candidate_ids"), "selection_text_sha256": result.pop("selection_text_sha256"), "actual_model_input_sha256": result.pop("actual_model_input_sha256"), "raw_response_sha256": result.pop("raw_response_sha256"), "physical_forward_calls": result.pop("physical_forward_calls"), "input_tokens": result.pop("input_tokens", None), "resource_mode": RESOURCE_MODE, "isolated": ISOLATED, "gold_in_inference_input": False, "cache_reuse": False}
        except Exception as exc:
            return {"record_type": "request", "model": model, "rate_index": rate_index, "request_index": index, "query_id": item["query_id"], "arrival_offset_s": offset, "submit_offset_s": submit_at - started, "worker_offset_s": worker_start - started, "queue_wait_ms": None, "duration_ms": (time.perf_counter() - submit_at) * 1000.0, "model_execution_ms": None, "status": "error", "error": f"{type(exc).__name__}: {exc}", "physical_forward_calls": 0, "resource_mode": RESOURCE_MODE, "isolated": ISOLATED, "gold_in_inference_input": False, "cache_reuse": False}

    for index, offset in enumerate(offsets):
        target = started + offset
        while True:
            remaining = target - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.05))
        submit_at = time.perf_counter()
        futures.append(executor.submit(work, index, items[index % len(items)], submit_at, offset))
    for future in futures:
        request_rows.append(future.result())
    executor.shutdown(wait=True)
    finished = time.perf_counter()
    after_gpu = gpu_snapshot()
    latencies = [float(row["duration_ms"]) for row in request_rows if row["status"] == "ok"]
    queues = [float(row["queue_wait_ms"]) for row in request_rows if row["status"] == "ok" and row.get("queue_wait_ms") is not None]
    end_to_end_window = max(0.0, finished - started)
    first_quarter = [row["queue_wait_ms"] for row in request_rows[: max(1, len(request_rows) // 4)] if row.get("queue_wait_ms") is not None]
    last_quarter = [row["queue_wait_ms"] for row in request_rows[-max(1, len(request_rows) // 4):] if row.get("queue_wait_ms") is not None]
    ok = sum(row["status"] == "ok" for row in request_rows)
    summary = {"record_type": "rate_summary", "model": model, "rate_index": rate_index, "offered_rps": rate, "requests": len(request_rows), "completed": ok, "errors": len(request_rows) - ok, "error_rate": (len(request_rows) - ok) / len(request_rows), "achieved_qps": ok / end_to_end_window if end_to_end_window else None, "p50_ms": percentile(latencies, 0.50), "p95_ms": percentile(latencies, 0.95), "p99_ms": percentile(latencies, 0.99), "queue_p50_ms": percentile(queues, 0.50), "queue_p95_ms": percentile(queues, 0.95), "queue_first_quarter_mean_ms": mean(first_quarter), "queue_last_quarter_mean_ms": mean(last_quarter), "queue_growth_ms": (mean(last_quarter) - mean(first_quarter)) if first_quarter and last_quarter else None, "p95_order_stat_rank": math.ceil(0.95 * len(latencies)) if latencies else None, "slo_1000_pass": bool(ok == len(request_rows) and percentile(latencies, 0.95) is not None and percentile(latencies, 0.95) <= 1000.0), "slo_2000_pass": bool(ok == len(request_rows) and percentile(latencies, 0.95) is not None and percentile(latencies, 0.95) <= 2000.0), "slo_3000_pass": bool(ok == len(request_rows) and percentile(latencies, 0.95) is not None and percentile(latencies, 0.95) <= 3000.0), "stable_queue_verdict": "MEASURED_ISOLATED" if ISOLATED else "DESCRIPTIVE_SHARED_CONTENTION", "gpu_allocation_seconds_proxy": end_to_end_window, "gpu_seconds_per_completed_query_proxy": end_to_end_window / ok if ok else None, "gpu_before": before_gpu, "gpu_after": after_gpu, "compute_apps_before": apps_before, "physical_forward_calls": sum(int(row.get("physical_forward_calls", 0)) for row in request_rows), "resource_mode": RESOURCE_MODE, "isolated": ISOLATED, "capacity_validity": CAPACITY_VALIDITY, "cost_status": "MEASURED_GPU_PROCESS_WALL_PROXY" if ISOLATED else "NOT_MEASURED_SHARED_CONTENTION"}
    for row in request_rows:
        row["rate_summary_ref"] = f"{model}:{rate_index}"
    return summary, request_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    args = parser.parse_args()
    items = load_cal_items()
    rates = read_rate_grid(args.model)
    pre_gpu = gpu_snapshot()
    if GPU not in pre_gpu.get("devices", {}):
        raise RuntimeError("GPU0 unavailable")
    if pre_gpu["devices"][GPU]["memory_free_mib"] < 25_000:
        raise RuntimeError(f"GPU{GPU} does not have the minimum free memory")
    adapter = make_adapter(args.model)
    adapter.warmup(items)
    summaries = []
    traces = []
    try:
        for index, rate in enumerate(rates):
            summary, rows = run_rate(adapter, args.model, items, rate, index)
            summaries.append(summary)
            traces.extend(rows)
            write_jsonl(OUT / f"{TRACE_PREFIX}_{args.model.upper()}_REQUESTS.jsonl", traces + summaries)
            write_json(OUT / f"{TRACE_PREFIX}_{args.model.upper()}_SUMMARY.json", {"model": args.model, "model_id": adapter.model_id, "revision": adapter.revision, "seed": SEED, "requests_per_rate": REQUESTS_PER_RATE, "concurrency": CONCURRENCY, "rates": rates, "warmup_queries": 2, "summaries": summaries, "trace_sha256": sha_file(OUT / f"{TRACE_PREFIX}_{args.model.upper()}_REQUESTS.jsonl"), "jev_calls": 0, "paid_api_calls": 0, "status": RUN_STATUS, "physical_gpu": GPU, "isolated": ISOLATED})
            print(json.dumps({"model": args.model, "rate_index": index, "offered_rps": rate, "completed": summary["completed"], "p50_ms": summary["p50_ms"], "p95_ms": summary["p95_ms"], "error_rate": summary["error_rate"], "slo_1000_pass": summary["slo_1000_pass"]}, sort_keys=True), flush=True)
    finally:
        adapter.close()
    final = {"model": args.model, "model_id": adapter.model_id, "revision": adapter.revision, "seed": SEED, "requests_per_rate": REQUESTS_PER_RATE, "concurrency": CONCURRENCY, "rates": rates, "warmup_queries": 2, "summaries": summaries, "trace_sha256": sha_file(OUT / f"{TRACE_PREFIX}_{args.model.upper()}_REQUESTS.jsonl"), "jev_calls": 0, "paid_api_calls": 0, "status": RUN_STATUS, "physical_gpu": GPU, "isolated": ISOLATED}
    write_json(OUT / f"{TRACE_PREFIX}_{args.model.upper()}_SUMMARY.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
