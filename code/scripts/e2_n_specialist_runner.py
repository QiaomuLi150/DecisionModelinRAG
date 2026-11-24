#!/usr/bin/env python3
"""Native adapters for the downloaded text reranker candidates.

The runner supports only the documented local interfaces of the approved
specialists.  It records one trace per frozen query, hashes the exact model
inputs, and instruments the model forward boundary.  It never contacts a
remote API and never reads gold labels during scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
SEED = 20261011
MODELS = {
    "qwen06": {"id": "Qwen/Qwen3-Reranker-0.6B", "owner": "Qwen", "repo": "Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473", "kind": "qwen_causal"},
    "qwen4": {"id": "Qwen/Qwen3-Reranker-4B", "owner": "Qwen", "repo": "Qwen3-Reranker-4B", "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5", "kind": "qwen_causal"},
    "jina35": {"id": "jinaai/jina-reranker-v3.5", "owner": "jinaai", "repo": "jina-reranker-v3.5", "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "kind": "jina_listwise"},
    "gte": {"id": "Alibaba-NLP/gte-reranker-modernbert-base", "owner": "Alibaba-NLP", "repo": "gte-reranker-modernbert-base", "revision": "f7481e6055501a30fb19d090657df9ec1f79ab2c", "kind": "sequence_classifier"},
    "minilm": {"id": "cross-encoder/ms-marco-MiniLM-L6-v2", "owner": "cross-encoder", "repo": "ms-marco-MiniLM-L6-v2", "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a", "kind": "sequence_classifier"},
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(spec: dict) -> Path:
    return HF_ROOT / f"models--{spec['owner']}--{spec['repo']}" / "snapshots" / spec["revision"]


def gpu_snapshot() -> dict:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"status": "PASS" if p.returncode == 0 else "UNAVAILABLE", "command": " ".join(command), "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "command": " ".join(command), "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def items(stage: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items, load_local_cal_items
    from e2_n_n1_smoke import load_items as load_smoke_items
    if stage == "smoke":
        return load_smoke_items()
    return load_local_cal_items() if stage == "n2" else load_full_items()


def views(item: dict, tokenizer):
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import make_views
    return [make_views(item["question"], candidate, tokenizer) for candidate in item["candidates"]]


def tensor_fingerprint(value):
    if not hasattr(value, "detach"):
        return repr(value)
    tensor = value.detach().to("cpu").contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": sha_bytes(tensor.numpy().tobytes())}


def trace_base(item: dict, spec: dict, v: list, stage: str) -> dict:
    return {
        "stage": "N1" if stage == "smoke" else "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL",
        "seed": SEED,
        "query_id": item["query_id"],
        "cohort": item.get("cohort", "cal" if stage == "n2" else "historical"),
        "dataset": item["dataset"],
        "model_id": spec["id"],
        "revision": spec["revision"],
        "adapter": f"official local native adapter:{spec['kind']}",
        "candidate_ids": [x.doc_id for x in v],
        "selection_text_sha256": [x.selection_text_sha256 for x in v],
        "score_direction": "higher_is_more_relevant",
        "cache_reuse": False,
        "cache_provenance": "not_applicable_new_direct_checkpoint_request",
        "gold_in_inference_input": False,
        "physical_gpu_id": os.environ.get("E2_N_PHYSICAL_GPU_ID"),
        "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "error",
        "real_inference": False,
    }


class NativeAdapter:
    def __init__(self, spec: dict):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        self.spec = spec
        self.path = snapshot(spec)
        if not self.path.is_dir():
            raise RuntimeError(f"missing local snapshot {self.path}")
        self.torch = torch
        self.forward_count = 0
        self.captured = []
        if spec["kind"] == "qwen_causal":
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.max_length = 8192
        elif spec["kind"] == "jina_listwise":
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.path, dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda").eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
        self.original_forward = self.model.forward

        def counted_forward(*args, **kwargs):
            self.forward_count += 1
            self.captured.append({"args": [tensor_fingerprint(value) for value in args], "kwargs": {key: tensor_fingerprint(value) for key, value in kwargs.items()}})
            return self.original_forward(*args, **kwargs)

        self.model.forward = counted_forward

    def score(self, query: str, documents: list[str]) -> tuple[list[float], dict]:
        import torch
        before = self.forward_count
        captured_before = len(self.captured)
        started = time.perf_counter_ns()
        if self.spec["kind"] == "sequence_classifier":
            encoded = self.tokenizer([query] * len(documents), documents, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.inference_mode():
                scores = self.model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
            token_count = int(encoded["attention_mask"].sum().item())
        elif self.spec["kind"] == "qwen_causal":
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            texts = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}" for document in documents]
            tokenized = self.tokenizer(texts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens))
            for i, token_ids in enumerate(tokenized["input_ids"]):
                tokenized["input_ids"][i] = prefix_tokens + token_ids + suffix_tokens
            encoded = self.tokenizer.pad(tokenized, padding=True, return_tensors="pt", max_length=self.max_length).to("cuda")
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, :]
                pair = torch.stack([logits[:, self.token_false_id], logits[:, self.token_true_id]], dim=1)
                scores = torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].exp().detach().cpu().tolist()
            token_count = int(encoded["attention_mask"].sum().item())
        else:
            result = self.model.rerank(query, documents)
            by_index = {int(row["index"]): float(row["relevance_score"]) for row in result}
            if set(by_index) != set(range(len(documents))):
                raise RuntimeError("Jina native listwise response index mapping invalid")
            scores = [by_index[i] for i in range(len(documents))]
            token_count = None
        actual = self.captured[captured_before:]
        return [float(x) for x in scores], {
            "actual_model_input_sha256": sha_json(actual),
            "physical_forward_calls": self.forward_count - before,
            "physical_forward_evidence": bool(actual) and self.forward_count > before,
            "input_forward_batches": len(actual),
            "input_tokens": token_count,
            "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        }

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def run(model_name: str, stage: str, run_items: list[dict], output: Path) -> dict:
    import torch
    spec = MODELS[model_name]
    adapter = NativeAdapter(spec)
    existing = {row["query_id"]: row for row in load_jsonl(output) if row.get("status") == "ok"}
    rows = []
    for index, item in enumerate(run_items, 1):
        if item["query_id"] in existing:
            rows.append(existing[item["query_id"]])
            continue
        v = views(item, adapter.tokenizer)
        if len(v) != 20 or len({x.doc_id for x in v}) != 20:
            raise RuntimeError(f"invalid 20-candidate slate {item['query_id']}")
        row = trace_base(item, spec, v, stage)
        documents = []
        sys.path.insert(0, str(ROOT / "src"))
        from decisionrag.selection import pointwise_pair
        for view in v:
            documents.append(pointwise_pair(view)[1])
        row["request_hash"] = sha_json({"query_id": item["query_id"], "candidate_ids": row["candidate_ids"], "selection_text_sha256": row["selection_text_sha256"], "contract": "INDEPENDENT_RELEVANCE:v2", "native_kind": spec["kind"]})
        try:
            scores, telemetry = adapter.score(v[0].question_view, documents)
            if len(scores) != 20 or any(not math.isfinite(x) for x in scores):
                raise RuntimeError("native adapter returned invalid score vector")
            row.update({"status": "ok", "real_inference": True, "scores": scores, "raw_response_sha256": sha_json(scores), "cache_source": "none", **telemetry})
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 10 == 0 or index == len(run_items):
            write_jsonl(output, rows)
            print(f"{model_name} {stage} {index}/{len(run_items)} ok={sum(r.get('status') == 'ok' for r in rows)}", flush=True)
    write_jsonl(output, rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    strict = [r for r in ok if r.get("physical_forward_evidence") is True and int(r.get("physical_forward_calls", 0)) > 0]
    summary = {"stage": "N1" if stage == "smoke" else "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL", "model": spec["id"], "revision": spec["revision"], "rows": len(rows), "expected_rows": len(run_items), "unique_queries": len({r.get("query_id") for r in rows}), "ok_rows": len(ok), "physical_forward_evidence_rows": len(strict), "total_physical_forward_calls": sum(int(r.get("physical_forward_calls", 0)) for r in strict), "input_and_response_hash_rows": sum(bool(r.get("actual_model_input_sha256") and r.get("raw_response_sha256")) for r in ok), "valid_unique_20_candidate_rows": sum(len(r.get("candidate_ids", [])) == 20 and len(set(r.get("candidate_ids", []))) == 20 for r in ok), "gold_in_inference_input_rows": sum(r.get("gold_in_inference_input") is True for r in rows), "cache_reuse_rows": sum(r.get("cache_reuse") is True for r in rows), "gpu_snapshot_after": gpu_snapshot(), "trace_sha256": sha_file(output), "jev_calls": 0, "paid_api_calls": 0, "status": "PASS" if len(ok) == len(run_items) and len(strict) == len(run_items) else "BLOCKED"}
    adapter.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--stage", choices=("smoke", "n2", "n3"), required=True)
    args = parser.parse_args()
    spec = MODELS[args.model]
    run_items = items(args.stage)
    sub = ROOT / "results/non_jev/v2/e2_n" / ("n1" if args.stage == "smoke" else "n2" if args.stage == "n2" else "n3")
    name = f"E2_N_{'N1' if args.stage == 'smoke' else 'N2' if args.stage == 'n2' else 'N3'}_{args.model.upper()}_{'SMOKE' if args.stage == 'smoke' else 'CAL' if args.stage == 'n2' else 'FULL'}.jsonl"
    output = sub / name
    summary = run(args.model, args.stage, run_items, output)
    summary_path = output.with_name(output.stem + "_SUMMARY.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
