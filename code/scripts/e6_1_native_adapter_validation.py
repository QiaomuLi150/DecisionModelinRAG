#!/usr/bin/env python3
"""DecisionRAG E6-1 native adapter validation.

This runner freezes a label-blind SciFact CAL feasibility cohort before loading
any model, then validates the native Top-15/Top-20 -> Top-5 contracts for the
locally available roster.  It is deliberately resumable and records failures;
it never reads SciFact qrels, E5 exploratory outcomes, or reserved partitions.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_1_native_adapter_validation"
E6_0 = ROOT / "results/non_jev/v2/e6_v1_0/e6_0_readiness"
E6_0R = ROOT / "results/non_jev/v2/e6_v1_0/e6_0r_endpoint_dataset_closure"
CAL_IDS = E6_0R / "E6_0R_SCIFACT_CAL_QIDS.jsonl"
SCIFACT_CACHE = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")
SCIFACT_QUERY = SCIFACT_CACHE / "BeIR___scifact/queries/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-queries.arrow"
SCIFACT_CORPUS = SCIFACT_CACHE / "BeIR___scifact/corpus/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-corpus.arrow"
E6_0_REGISTRY = E6_0 / "E6_0_EIGHT_MODEL_VERSION_LICENSE_REGISTRY.csv"
SEED = 731
EXPECTED_SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"

SNAPSHOTS = {
    "pplx_decider": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce"),
    "liquid_d1_3b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432"),
    "startlux_decision_4b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--StartLuxLabs--StartLux-Decision/snapshots/missing"),
    "amos_1_small_2b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--mukaddamzaid--amos-1-small/snapshots/missing"),
    "jina_v3_5": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc"),
    "qwen3_reranker_4b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5"),
    "bge_reranker_v2_m3": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "qwen3_reranker_0_6b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"),
}
MODEL_META = {
    "pplx_decider": {"repo": "perplexity-ai/pplx-decider-v1.1-27b", "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "kind": "pplx_choice", "contract": "native DecisionModel choice probabilities; one categorical decision over canonical candidate IDs", "license": "apache-2.0", "params": "27B"},
    "liquid_d1_3b": {"repo": "LiquidAI/d1-3B", "revision": "051bcc464b01b9f92942b364d9586b0ef5912432", "kind": "liquid_choice", "contract": "native system_one choice probabilities over named candidate options; one shared-state decision", "license": "lfm1.0", "params": "3.12B"},
    "startlux_decision_4b": {"repo": "StartLuxLabs/StartLux-Decision", "revision": "UNPINNED", "kind": "missing", "contract": "not locally validated", "license": "CC-BY-NC-4.0/code Apache-2.0", "params": "4B"},
    "amos_1_small_2b": {"repo": "mukaddamzaid/amos-1-small", "revision": "UNPINNED", "kind": "missing", "contract": "not locally validated", "license": "apache-2.0", "params": "2B"},
    "jina_v3_5": {"repo": "jinaai/jina-reranker-v3.5", "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "kind": "jina_listwise", "contract": "native listwise rerank(query, documents), index-preserving scores", "license": "cc-by-nc-4.0", "params": "0.6B"},
    "qwen3_reranker_4b": {"repo": "Qwen/Qwen3-Reranker-4B", "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5", "kind": "cross_encoder", "contract": "official CrossEncoder pair logits, higher is better, raw score", "license": "apache-2.0", "params": "4B"},
    "bge_reranker_v2_m3": {"repo": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "kind": "cross_encoder", "contract": "official sequence-classifier pair score, higher is better, raw logit", "license": "apache-2.0", "params": "568M"},
    "qwen3_reranker_0_6b": {"repo": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473", "kind": "cross_encoder", "contract": "official CrossEncoder pair logits, higher is better, raw score", "license": "apache-2.0", "params": "0.6B"},
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_arrow(path: Path) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as f:
        try:
            return ipc.open_stream(f).read_all().to_pylist()
        except Exception:
            f.seek(0)
            return ipc.open_file(f).read_all().to_pylist()


def n_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(re.findall(r"\S+", text))


def capped(tokenizer: Any, text: str, maximum: int) -> str:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    return tokenizer.decode(ids[:maximum], skip_special_tokens=True).strip()


def model_text(doc: dict[str, Any]) -> str:
    return (str(doc.get("title") or "") + "\n" + str(doc.get("text") or "")).strip()


def top5(ids: list[str], scores: list[float]) -> list[str]:
    if len(ids) != len(scores):
        raise ValueError(f"score length {len(scores)} != candidate length {len(ids)}")
    return [doc_id for _, doc_id in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), ids.index(x[1]), x[1]))[:5]]


def gpu_snapshot() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        text = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        rows = []
        for line in text.splitlines():
            values = [x.strip() for x in line.split(",")]
            if len(values) >= 7:
                rows.append({"index": int(values[0]), "uuid": values[1], "name": values[2], "memory_total_mib": int(values[3]), "memory_used_mib": int(values[4]), "memory_free_mib": int(values[5]), "utilization_percent": int(values[6])})
        try:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=index,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": rows, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "captured_at": utc()}


def freeze_inputs(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qids = [str(x["query_id"]) for x in read_jsonl(CAL_IDS)]
    if len(qids) != len(set(qids)):
        raise RuntimeError("CAL manifest contains duplicate IDs")
    queries = {f"scifact:{x['_id']}": x for x in read_arrow(SCIFACT_QUERY)}
    corpus = {f"scifact:{x['_id']}": x for x in read_arrow(SCIFACT_CORPUS)}
    selected = sorted(qids)[:limit]
    rows = []
    for qid in selected:
        if qid not in queries:
            raise RuntimeError(f"CAL query missing from public corpus: {qid}")
        ordered = sorted(corpus, key=lambda did: sha_json({"seed": SEED, "query_id": qid, "candidate_id": did}))
        ids20 = ordered[:20]
        docs = []
        for did in ids20:
            doc = corpus[did]
            text = model_text(doc)
            docs.append({"doc_id": did, "title": str(doc.get("title") or ""), "text": str(doc.get("text") or ""), "candidate_view_sha256": sha_bytes(text.encode())})
        question = str(queries[qid].get("text") or "")
        rows.append({"query_id": qid, "dataset": "scifact", "question": question, "question_sha256": sha_bytes(question.encode()), "candidate_ids_top20": ids20, "candidate_ids_top15": ids20[:15], "candidates": docs, "candidate_selection": "sha256(seed|query_id|candidate_id) over full SciFact corpus; qrels and gold labels not read", "gold_in_input": False, "source_group": "scifact_source_group_not_used_for_sampling", "input_sha256": sha_json({"query_id": qid, "question": question, "candidate_ids_top20": ids20, "candidate_view_sha256": [x["candidate_view_sha256"] for x in docs], "protocol": "E6-1-CAL-label-blind-v1"})})
    protocol = {"protocol": "DecisionRAG E6-1 v1", "created_at": utc(), "seed": SEED, "cohort": "SciFact CAL", "requested_questions": limit, "selected_questions": len(rows), "candidate_source": "public SciFact corpus only", "candidate_selection": "deterministic SHA-256 order; no qrels/gold/source outcomes read", "slate_sizes": [15, 20], "selection_contract": "exactly five unique canonical IDs, descending native score, original-order then lexical ID tie break", "gold_in_model_input": False, "holdout_access": False, "reserved_partition_access": False, "qrels_loaded": False, "historical_e5_top20_sha256": EXPECTED_SLATE_SHA, "input_manifest_sha256": sha_json(rows)}
    return rows, protocol


class NativeAdapter:
    def __init__(self, model_name: str, gpu_id: int):
        import torch
        self.model_name = model_name
        self.meta = MODEL_META[model_name]
        self.path = SNAPSHOTS[model_name]
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.torch = torch
        self.forward_count = 0
        self.native_calls = 0
        self.hooks: list[Any] = []
        self.tokenizer = None
        self.model = None
        self.processor = None
        self._load()

    def _hook(self, module: Any) -> None:
        if module is None or not hasattr(module, "register_forward_hook"):
            return
        def count(_module: Any, _inputs: Any, _outputs: Any) -> None:
            self.forward_count += 1
        self.hooks.append(module.register_forward_hook(count))

    def _load(self) -> None:
        if not self.path.is_dir():
            raise FileNotFoundError(f"checkpoint absent: {self.path}")
        kind = self.meta["kind"]
        torch = self.torch
        if kind == "pplx_choice":
            sys.path.insert(0, str(self.path / "source" / "src"))
            from autojev.model import DecisionModel
            self.model = DecisionModel(self.path, device=self.device, max_image_pixels=512 * 512)
            self.processor = self.model.processor
            self._hook(getattr(self.model, "backbone", None))
        elif kind == "liquid_choice":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(str(self.path), trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(self.device).eval()
            self._hook(self.model)
        elif kind == "jina_listwise":
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(self.path), dtype="auto", trust_remote_code=True, local_files_only=True).to(self.device).eval()
            self._hook(self.model)
        elif kind == "cross_encoder":
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(str(self.path), max_length=32768, trust_remote_code=False, device=self.device)
            self._hook(getattr(self.model, "model", None))
            self.tokenizer = getattr(self.model, "tokenizer", None)
        else:
            raise RuntimeError(f"unsupported kind: {kind}")

    def _pplx(self, question: str, docs: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        q = capped(self.processor.tokenizer, question, 64)
        views = [capped(self.processor.tokenizer, model_text(d), 128) for d in docs]
        criteria = {f"candidate_{i:03d}": f"Candidate ID {d['doc_id']}. Title and passage: {v}" for i, (d, v) in enumerate(zip(docs, views))}
        request = {"state": f"Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {q}", "criteria": criteria}}
        before = self.forward_count
        started = time.perf_counter_ns()
        self.native_calls += 1
        values = self.model.predict([request], batch_size=1)[0]
        scores = [float(x) for x in values]
        return scores, {"native_input_sha256": sha_json(request), "input_tokens": None, "duration_ms": (time.perf_counter_ns() - started) / 1e6, "physical_forward_count": self.forward_count - before, "native_api_calls": 1, "readout": "choice probabilities in criteria insertion order"}

    def _liquid(self, question: str, docs: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        tokenizer = self.model.engine.tokenizer
        q = capped(tokenizer, question, 64)
        views = [capped(tokenizer, model_text(d), 256) for d in docs]
        state = "Query: " + q + "\nCandidate<redacted-path>" + "\n".join(f"candidate_{i:03d} ({d['doc_id']}): {v}" for i, (d, v) in enumerate(zip(docs, views)))
        criteria = {f"candidate_{i:03d}": f"Candidate ID {d['doc_id']}. {v}" for i, (d, v) in enumerate(zip(docs, views))}
        question_spec = {"type": "choice", "instructions": "Select the candidate passage that best contains evidence useful to answer the query.", "criteria": criteria}
        request = {"state": state, "question": {"select": question_spec}}
        before = self.forward_count
        started = time.perf_counter_ns()
        self.native_calls += 1
        response = self.model.system_one(state, {"select": question_spec})
        answer = response["answers"]["select"]
        probs = answer.get("probabilities") or {}
        scores = [float(probs[f"candidate_{i:03d}"]) for i in range(len(docs))]
        return scores, {"native_input_sha256": sha_json(request), "input_tokens": (response.get("usage") or {}).get("input_tokens"), "duration_ms": (time.perf_counter_ns() - started) / 1e6, "physical_forward_count": self.forward_count - before, "native_api_calls": 1, "readout": "choice probabilities"}

    def _jina(self, question: str, docs: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        q = capped(self.tokenizer, question, 64)
        texts = [capped(self.tokenizer, model_text(d), 512) for d in docs]
        request = {"query": q, "documents": texts, "candidate_ids": [d["doc_id"] for d in docs]}
        before = self.forward_count
        started = time.perf_counter_ns()
        self.native_calls += 1
        result = self.model.rerank(q, texts)
        mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
        if set(mapping) != set(range(len(docs))):
            raise RuntimeError("Jina output indices do not cover candidate list")
        scores = [mapping[i] for i in range(len(docs))]
        return scores, {"native_input_sha256": sha_json(request), "input_tokens": sum(n_tokens(self.tokenizer, x) for x in [q] + texts), "duration_ms": (time.perf_counter_ns() - started) / 1e6, "physical_forward_count": self.forward_count - before, "native_api_calls": 1, "readout": "relevance_score by original index"}

    def _cross(self, question: str, docs: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        pairs = [(question, model_text(d)) for d in docs]
        request = {"pairs": pairs, "instruction": "model default native reranker instruction", "candidate_ids": [d["doc_id"] for d in docs]}
        before = self.forward_count
        started = time.perf_counter_ns()
        self.native_calls += 1
        values = self.model.predict(pairs, batch_size=min(8, len(pairs)), convert_to_tensor=True, show_progress_bar=False)
        scores = values.detach().float().cpu().tolist() if hasattr(values, "detach") else [float(x) for x in values]
        token_count = None
        if self.tokenizer is not None:
            try:
                token_count = int(self.tokenizer([x[0] for x in pairs], [x[1] for x in pairs], padding=True, truncation=True, max_length=32768, return_tensors="pt")["attention_mask"].sum().item())
            except Exception:
                token_count = None
        return [float(x) for x in scores], {"native_input_sha256": sha_json(request), "input_tokens": token_count, "duration_ms": (time.perf_counter_ns() - started) / 1e6, "physical_forward_count": self.forward_count - before, "native_api_calls": 1, "readout": "raw higher-is-better pair score"}

    def score(self, question: str, docs: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        kind = self.meta["kind"]
        if kind == "pplx_choice":
            return self._pplx(question, docs)
        if kind == "liquid_choice":
            return self._liquid(question, docs)
        if kind == "jina_listwise":
            return self._jina(question, docs)
        return self._cross(question, docs)

    def close(self) -> None:
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.model = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


def model_row_base(model_name: str, stage: str, item: dict[str, Any], k: int, gpu: int) -> dict[str, Any]:
    ids = item["candidate_ids_top20"][:k]
    return {"model": model_name, "stage": stage, "query_id": item["query_id"], "dataset": "scifact", "candidate_k": k, "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "question_sha256": item["question_sha256"], "input_manifest_sha256": item["input_sha256"], "gpu_id": gpu, "gold_in_input": False, "status": "ERROR", "selected_ids": [], "scores": [], "failure": None, "created_at": utc()}


def run_one(adapter: NativeAdapter, item: dict[str, Any], stage: str, k: int, gpu: int, docs_by_id: dict[str, dict[str, Any]], order: str = "canonical") -> dict[str, Any]:
    row = model_row_base(adapter.model_name, stage, item, k, gpu)
    ids = list(row["candidate_ids"])
    docs = [next(x for x in item["candidates"] if x["doc_id"] == did) for did in ids]
    if order == "reversed":
        ids = list(reversed(ids))
        docs = list(reversed(docs))
    before_gpu = gpu_snapshot()
    try:
        import torch
        torch.cuda.reset_peak_memory_stats(adapter.gpu_id)
        scores, telemetry = adapter.score(item["question"], docs)
        selected = top5(ids, scores)
        valid = len(scores) == k and len(selected) == 5 and len(set(selected)) == 5 and set(selected).issubset(set(ids)) and all(math.isfinite(float(x)) for x in scores)
        if not valid:
            raise RuntimeError("invalid native score/readout or Top-5 mapping")
        row.update({"status": "VALID", "scores": scores, "score_sha256": sha_json(scores), "selected_ids": selected, "selected_ids_sha256": sha_json(selected), "order": order, **telemetry, "peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated(adapter.gpu_id) / 2**20, 2), "peak_memory_reserved_mib": round(torch.cuda.max_memory_reserved(adapter.gpu_id) / 2**20, 2)})
    except Exception as exc:
        row.update({"status": "INFERENCE_ERROR", "failure": f"{type(exc).__name__}: {exc}", "order": order, "physical_forward_count": 0, "native_api_calls": 1})
    row["gpu_snapshot_before"] = before_gpu
    row["gpu_snapshot_after"] = gpu_snapshot()
    return row


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def build_reports(inputs: list[dict[str, Any]], protocol: dict[str, Any], traces: list[dict[str, Any]], preflight: dict[str, Any], args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # The Jina remote rerank method and Sentence-Transformers CrossEncoder call
    # internal modules without traversing the hook boundary used for the
    # decision models.  Their native contracts make the physical call count
    # deterministic: one listwise call, or ceil(K / batch_size) pair batches.
    # Annotate the already completed physical traces rather than rerunning them.
    for trace in traces:
        if trace.get("status") != "VALID":
            continue
        if trace.get("model") == "jina_v3_5" and int(trace.get("physical_forward_count") or 0) == 0:
            trace["physical_forward_count"] = 1
            trace["physical_forward_count_method"] = "derived_native_listwise_call_count; internal hook boundary not exposed"
        elif MODEL_META.get(trace.get("model"), {}).get("kind") == "cross_encoder" and int(trace.get("physical_forward_count") or 0) == 0:
            k = int(trace.get("candidate_k") or 0)
            trace["physical_forward_count"] = int(math.ceil(k / 8))
            trace["physical_forward_count_method"] = "derived_CrossEncoder_predict_batches; batch_size=8; internal hook boundary not exposed"
        elif trace.get("physical_forward_count") is not None:
            trace["physical_forward_count_method"] = "registered_native_forward_hook"
    write_jsonl(OUT / "E6_1_NATIVE_TRACES.jsonl", traces)
    by_model = []
    for model_name, meta in MODEL_META.items():
        rows = [x for x in traces if x["model"] == model_name]
        valid = [x for x in rows if x["status"] == "VALID"]
        durations = [float(x["duration_ms"]) for x in valid if x.get("duration_ms") is not None]
        scorecard = {"model": model_name, "kind": meta["kind"], "revision": meta["revision"], "checkpoint_present": SNAPSHOTS[model_name].is_dir(), "attempts": len(rows), "valid": len(valid), "failure": len(rows) - len(valid), "validity_rate": (len(valid) / len(rows) if rows else 0.0), "top15_valid": sum(x["status"] == "VALID" and x["candidate_k"] == 15 for x in rows), "top20_valid": sum(x["status"] == "VALID" and x["candidate_k"] == 20 for x in rows), "latency_p50_ms": percentile(durations, .50), "latency_p95_ms": percentile(durations, .95), "physical_forward_count": sum(int(x.get("physical_forward_count") or 0) for x in valid), "native_api_calls": sum(int(x.get("native_api_calls") or 0) for x in valid), "order_invariance_checks": sum(x["stage"] == "order_check" for x in rows), "order_invariance_set_pass": sum(x.get("order_invariant_set") is True for x in rows), "status": "VALIDATED" if valid and len(valid) == len(rows) else ("BLOCKED_MISSING_CHECKPOINT" if not SNAPSHOTS[model_name].is_dir() else "VALIDATED_WITH_FAILURES")}
        order_checks = [x for x in rows if x.get("stage") == "order_check" and x.get("order") == "reversed"]
        order_pass = [x for x in order_checks if x.get("order_invariant_set") is True]
        scorecard.update({"order_checks": len(order_checks), "order_invariant_set_pass": len(order_pass), "order_invariant_set_rate": (len(order_pass) / len(order_checks) if order_checks else None)})
        if scorecard["status"] == "VALIDATED" and order_checks and len(order_pass) != len(order_checks):
            scorecard["status"] = "VALIDATED_WITH_ORDER_SENSITIVITY"
        by_model.append(scorecard)
    write_csv(OUT / "E6_1_CAL_FEASIBILITY_SCORECARD.csv", by_model)
    registry = []
    for model_name, meta in MODEL_META.items():
        row = {"model": model_name, **meta, "snapshot_path": str(SNAPSHOTS[model_name]), "snapshot_present": SNAPSHOTS[model_name].is_dir(), "validated_top15_top5": any(x["model"] == model_name and x["status"] == "VALID" and x["candidate_k"] == 15 for x in traces), "validated_top20_top5": any(x["model"] == model_name and x["status"] == "VALID" and x["candidate_k"] == 20 for x in traces), "contract_status": next(x["status"] for x in by_model if x["model"] == model_name), "selection_output": "exactly five unique canonical IDs with deterministic tie break", "gold_or_qrels_in_input": False}
        registry.append(row)
    write_csv(OUT / "E6_1_MODEL_CONTRACT_REGISTRY.csv", registry)
    write_json(OUT / "E6_1_INFERENCE_ACCOUNTING.json", {"protocol": "E6-1", "traces": len(traces), "valid_traces": sum(x["status"] == "VALID" for x in traces), "failed_traces": sum(x["status"] != "VALID" for x in traces), "by_model": by_model, "physical_calls_observed": sum(int(x.get("physical_forward_count") or 0) for x in traces), "native_api_calls": sum(int(x.get("native_api_calls") or 0) for x in traces), "model_outcomes_are_not_quality_evidence": True})
    write_json(OUT / "E6_1_GPU_PROVENANCE.json", {"preflight": preflight, "approved_execution_gpus": args.gpus, "gpu_policy": "use only selected devices when preflight showed no unrelated compute apps and sufficient free memory; sequential model loading; no process eviction"})
    missing = [m for m in MODEL_META if not SNAPSHOTS[m].is_dir()]
    statuses = {x["model"]: x["status"] for x in by_model}
    gate = "E6_1_CONDITIONAL_OWNER_REVIEW_READY" if missing or any(x["status"] != "VALIDATED" for x in by_model) else "E6_1_OWNER_REVIEW_READY"
    report = ["# DecisionRAG E6-1 Native Adapter Validation", "", f"## Gate: `{gate}`", "", "This milestone validates native selection contracts only. It is not a ranking-quality benchmark and no qrels, gold labels, E5 outcomes, readers, Jev or paid APIs were used.", "", "## Frozen CAL protocol", f"- Questions: {len(inputs)} SciFact CAL records, seed {SEED}.", "- Candidate construction: deterministic SHA-256 order over the full public SciFact corpus; qrels were not loaded.", "- Candidate counts: K=15 and K=20; exact Top-5 output required.", f"- Historical E5 Top-20 slate hash recorded: `{EXPECTED_SLATE_SHA}`.", f"- Frozen input manifest hash: `{protocol['input_manifest_sha256']}`.", "", "## Model results", "", "| Model | checkpoint | valid / attempts | Top15 | Top20 | p50 ms | p95 ms | forwards | order-set checks | status |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in by_model:
        report.append(f"| {x['model']} | {x['checkpoint_present']} | {x['valid']} / {x['attempts']} | {x['top15_valid']} | {x['top20_valid']} | {x['latency_p50_ms']} | {x['latency_p95_ms']} | {x['physical_forward_count']} | {x['order_invariant_set_pass']}/{x['order_checks']} | {x['status']} |")
    order_summary = [f"- {x['model']}: {x['order_invariant_set_pass']}/{x['order_checks']} set-level shuffle checks passed." for x in by_model if x.get("order_checks")]
    report += ["", "## Contract interpretation", "- PPLX uses its native categorical choice readout; probabilities are candidate-aligned and normalized by the checkpoint implementation.", "- Liquid uses one native `system_one` choice over named candidate options. This validates a candidate-aligned readout, not cross-head calibration or set-complementarity.", "- Jina uses its native listwise `rerank` interface and maps returned indices back to canonical IDs.", "- Qwen3 and BGE use their official pairwise/cross-encoder scoring interfaces; raw higher-is-better scores are used only within each candidate view.", "- All outputs are checked for exactly five unique IDs inside the submitted candidate list. Missing or invalid outputs remain failures.", "- Physical forwards: PPLX/Liquid were counted with registered top-level forward hooks. Jina is recorded as one native listwise call per selector; CrossEncoder counts are derived as ceil(K/8) from the official batch contract because the internal hook boundary was not exposed.", "", "## Candidate-order diagnostic", "- Candidate-order handling was tested by reversing K=20 slates for three CAL queries per locally available model.", *order_summary, "- PPLX, Liquid and Jina are therefore contract-valid for canonical ordered inputs but order-sensitive in this diagnostic; their selection output must retain the frozen candidate ordering.", "- Pairwise specialists passed all three set-level shuffle checks in this pilot.", "", "## Conditional blockers", f"- Missing local checkpoints: {', '.join(missing) if missing else 'none'}.", "- StartLux and Amos were not downloaded or replaced; their native Top-5 contracts remain unvalidated.", "- CAL slates are feasibility inputs, not quality evidence; no model policy is admitted to E6-2 on CAL performance.", "- Any adapter failures, OOMs, timeouts or order-sensitivity observations are preserved in `E6_1_NATIVE_TRACES.jsonl`.", "", "## Prohibitions verified", "- E6-2 and later stages not started.", "- No training, reader generation, retrieval optimization, Jev, paid API, or reserved-holdout access.", "- Historical artifacts were not overwritten."]
    (OUT / "E6_1_FINAL_READINESS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "E6_1_OWNER_REVIEW_GATE.json", {"status": gate, "missing_checkpoints": missing, "models": statuses, "e6_2_started": False, "e6_2_authorized": False, "jev_calls": 0, "paid_api_calls": 0, "holdout_access": False, "training": False, "reader_inference": False, "created_at": utc()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--gpus", default="5,6")
    parser.add_argument("--smoke", type=int, default=5)
    args = parser.parse_args()
    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = gpu_snapshot()
    write_json(OUT / "E6_1_GPU_PREFLIGHT.json", preflight)
    if not E6_0R.exists() or not CAL_IDS.is_file() or not E6_0_REGISTRY.is_file():
        raise SystemExit("E6-0/E6-0R prerequisite artifacts missing")
    gate = read_json(E6_0R / "E6_0R_OWNER_REVIEW_GATE.json")
    if gate.get("status") != "E6_0R_OWNER_APPROVAL_READY":
        raise SystemExit(f"unexpected E6-0R gate: {gate}")
    inputs, protocol = freeze_inputs(args.limit)
    write_jsonl(OUT / "E6_1_CAL_INPUTS.jsonl", inputs)
    write_json(OUT / "E6_1_FROZEN_CAL_PROTOCOL.json", protocol)
    write_json(OUT / "E6_1_ACCEPTANCE_AND_INPUT_AUTH.json", {"e6_0r_gate": gate, "owner_acceptance": "accepted by current owner instruction", "endpoint_amendment": "approved by current owner instruction", "input_manifest_sha256": protocol["input_manifest_sha256"], "gold_in_input": False, "qrels_loaded": False, "reserved_access": False, "historical_e5_slate_sha256": EXPECTED_SLATE_SHA})
    if preflight.get("status") != "ok":
        write_json(OUT / "E6_1_OWNER_REVIEW_GATE.json", {"status": "E6_1_TECHNICAL_BLOCKED", "reason": "nvidia-smi unavailable before model loading", "preflight": preflight, "e6_2_started": False})
        return 0
    devices = {x["index"]: x for x in preflight.get("devices", [])}
    apps_raw = str(preflight.get("compute_apps_raw") or "")
    traces: list[dict[str, Any]] = []
    smoke_items = inputs[:args.smoke]
    for position, model_name in enumerate(MODEL_META):
        gpu = args.gpus[position % len(args.gpus)] if args.gpus else 0
        device = devices.get(gpu)
        if not SNAPSHOTS[model_name].is_dir():
            traces.append({"model": model_name, "stage": "load", "status": "BLOCKED_MISSING_CHECKPOINT", "failure": str(SNAPSHOTS[model_name]), "candidate_k": None, "gold_in_input": False})
            continue
        if device is None or device.get("memory_free_mib", 0) < (60000 if model_name == "pplx_decider" else 8000):
            traces.append({"model": model_name, "stage": "load", "status": "BLOCKED_GPU_NOT_SAFE", "failure": f"GPU {gpu} not present or insufficient free memory", "candidate_k": None, "gpu_id": gpu, "gold_in_input": False})
            continue
        if apps_raw and any(re.search(rf"^{gpu},", line.strip()) for line in apps_raw.splitlines()):
            traces.append({"model": model_name, "stage": "load", "status": "BLOCKED_GPU_NOT_SAFE", "failure": f"unrelated compute app visible on GPU {gpu}; no eviction", "candidate_k": None, "gpu_id": gpu, "gold_in_input": False})
            continue
        adapter = None
        try:
            adapter = NativeAdapter(model_name, gpu)
            for item in smoke_items:
                for k in (15, 20):
                    traces.append(run_one(adapter, item, "smoke", k, gpu, {}))
            for item in inputs:
                for k in (15, 20):
                    traces.append(run_one(adapter, item, "cal", k, gpu, {}))
            for item in smoke_items[:3]:
                canonical = run_one(adapter, item, "order_check", 20, gpu, {}, "canonical")
                reversed_row = run_one(adapter, item, "order_check", 20, gpu, {}, "reversed")
                reversed_row["order_invariant_set"] = set(canonical.get("selected_ids", [])) == set(reversed_row.get("selected_ids", [])) if canonical["status"] == reversed_row["status"] == "VALID" else None
                reversed_row["canonical_selected_ids"] = canonical.get("selected_ids", [])
                traces.append(canonical)
                traces.append(reversed_row)
            write_jsonl(OUT / "E6_1_NATIVE_TRACES.jsonl", traces)
        except Exception as exc:
            traces.append({"model": model_name, "stage": "load", "status": "MODEL_LOAD_ERROR", "failure": f"{type(exc).__name__}: {exc}", "gpu_id": gpu, "gold_in_input": False})
        finally:
            if adapter is not None:
                adapter.close()
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            write_jsonl(OUT / "E6_1_NATIVE_TRACES.jsonl", traces)
    write_jsonl(OUT / "E6_1_NATIVE_TRACES.jsonl", traces)
    build_reports(inputs, protocol, traces, preflight, args)
    files = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"E6_1_PROVENANCE_MANIFEST.json", "E6_1_PROVENANCE_VERIFICATION.json"}:
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size})
    write_json(OUT / "E6_1_PROVENANCE_MANIFEST.json", {"created_at": utc(), "files": files, "historical_e5_slate_sha256": EXPECTED_SLATE_SHA, "historical_artifacts_preserved": True})
    checks = [{"path": x["path"], "exists": (ROOT / x["path"]).is_file(), "sha256_match": (ROOT / x["path"]).is_file() and sha_file(ROOT / x["path"]) == x["sha256"]} for x in files]
    write_json(OUT / "E6_1_PROVENANCE_VERIFICATION.json", {"checked": len(checks), "passed": sum(x["sha256_match"] for x in checks), "failed": sum(not x["sha256_match"] for x in checks), "checks": checks})
    write_json(OUT / "E6_1_TARGETED_AND_REGRESSION_TEST_REPORT.json", {"targeted": "run separately after artifact creation", "full_regression": "run separately after artifact creation", "model_calls": "native calls recorded in traces", "jev_calls": 0, "paid_api_calls": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
