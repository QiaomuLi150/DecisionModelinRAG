#!/usr/bin/env python3
"""M3 real local selector measurements; Jev is intentionally unavailable here."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decisionrag.selection import (  # noqa: E402
    CandidateView,
    deterministic_quartets,
    groupwise_states,
    make_views,
    pointwise_pair,
    pointwise_question,
    pointwise_state,
    sha256_json,
)

LAYER_PATHS = {
    "bge": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "qwen": ("Qwen/Qwen3-VL-Reranker-2B", "4bd860ac4f15ad1897a214615cccc700f8f71818"),
    "laya": ("convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"),
    "clef": ("Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"),
}
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
QWEN_URL = "http://service.example.invalid/v1/rerank"
QWEN_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
SEED = 20261011


def local_snapshot(owner: str, repo: str, revision: str) -> str:
    return str(HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / revision)


def common_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(local_snapshot("convaiinnovations", "laya", LAYER_PATHS["laya"][1]) + "/tokenizer")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def timing_call(fn):
    start = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - start) / 1_000_000.0


def gpu_sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def common_query_rows():
    questions = load_rows(ROOT / "data/manifests/questions.jsonl")
    slates = load_rows(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")
    corpus = {row["doc_id"]: row for row in load_rows(ROOT / "data/corpus/passages.jsonl")}
    rows = []
    for question, slate in zip(questions, slates):
        if question["query_id"] != slate["query_id"]:
            raise RuntimeError("question/slate order mismatch")
        candidates = []
        for rank, doc_id in enumerate(slate["candidate_ids"], start=1):
            source = corpus[doc_id]
            candidates.append({"doc_id": doc_id, "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": rank})
        rows.append({"query_id": question["query_id"], "dataset": question["dataset"], "question": question["question"], "candidates": candidates})
    return rows


def base_row(item: dict, model_key: str, revision: str) -> dict:
    return {"query_id": item["query_id"], "dataset": item["dataset"], "model_id": LAYER_PATHS[model_key][0], "model_key": model_key, "revision": revision, "status": "error", "pointwise": [], "real_inference": False, "error_type": None}


def run_bge(items: list[dict], revision: str) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = local_snapshot("BAAI", "bge-reranker-v2-m3", revision)
    # The frozen semantic view is measured with the pinned Laya tokenizer, but
    # the BGE checkpoint must tokenize that exact text with its own native
    # tokenizer before model inference.  These are distinct responsibilities.
    view_tokenizer = common_tokenizer()
    model_tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    out = []
    for count, item in enumerate(items, start=1):
        row = base_row(item, "bge", revision)
        try:
            views = [make_views(item["question"], c, view_tokenizer) for c in item["candidates"]]
            pairs = [pointwise_pair(view) for view in views]
            encoded = model_tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            gpu_sync(torch)
            start = time.perf_counter_ns()
            with torch.inference_mode():
                logits = model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
            gpu_sync(torch)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            row["status"] = "ok"
            row["real_inference"] = True
            row["batch_duration_ms"] = elapsed
            row["input_tokens"] = int(encoded["attention_mask"].sum().item())
            row["request_hash"] = sha256_json([view.selection_text_sha256 for view in views])
            row["response_hash"] = sha256_json(logits)
            row["pointwise"] = [{"doc_id": view.doc_id, "score": float(score), "duration_ms": elapsed / 20.0, "input_tokens": int(encoded["attention_mask"][index].sum().item()), "selection_text_sha256": view.selection_text_sha256, "real_inference": True} for index, (view, score) in enumerate(zip(views, logits))]
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
        if count % 25 == 0 or count == len(items): print(f"bge pointwise {count}/{len(items)}", flush=True)
    return out


def run_qwen(items: list[dict], revision: str) -> list[dict]:
    session = requests.Session()
    identity = session.get("http://service.example.invalid/v1/models", timeout=20).json()
    served = identity["data"][0]
    if served.get("id") != QWEN_MODEL or revision not in str(served.get("root", "")):
        raise RuntimeError(f"unexpected Qwen identity: {served}")
    tokenizer = common_tokenizer()
    out = []
    for count, item in enumerate(items, start=1):
        row = base_row(item, "qwen", revision)
        try:
            # The Laya tokenizer is the pinned common view tokenizer for all
            # models; the service receives the resulting exact text unchanged.
            laya_path = local_snapshot("convaiinnovations", "laya", LAYER_PATHS["laya"][1]) + "/tokenizer"
            views = [make_views(item["question"], c, tokenizer) for c in item["candidates"]]
            query, documents = views[0].question_view, [pointwise_pair(view)[1] for view in views]
            payload = {
                "model": QWEN_MODEL,
                "query": query,
                "documents": documents,
                # The checkpoint's official scorer appends the assistant
                # generation prompt before reading the yes/no logits.  vLLM
                # does not infer this flag for the scoring endpoint.
                "chat_template_kwargs": {"add_generation_prompt": True},
            }
            start = time.perf_counter_ns()
            response = session.post(QWEN_URL, json=payload, timeout=300)
            response.raise_for_status()
            body = response.json()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            results = sorted(body["results"], key=lambda value: int(value["index"]))
            if len(results) != 20:
                raise RuntimeError(f"expected 20 Qwen scores, got {len(results)}")
            scores = [float(value["relevance_score"]) for value in results]
            row["status"] = "ok"
            row["real_inference"] = True
            row["batch_duration_ms"] = elapsed
            row["input_tokens"] = body.get("usage", {}).get("prompt_tokens")
            row["request_hash"] = sha256_json(payload)
            row["response_hash"] = sha256_json(body)
            row["pointwise"] = [{"doc_id": view.doc_id, "score": score, "duration_ms": elapsed / 20.0, "input_tokens": row["input_tokens"], "selection_text_sha256": view.selection_text_sha256, "real_inference": True} for view, score in zip(views, scores)]
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
        if count % 25 == 0 or count == len(items): print(f"qwen pointwise {count}/{len(items)}", flush=True)
    return out


def laya_schema():
    return pointwise_question()


def parse_laya_score(result: dict) -> float:
    return float(result["answers"]["relevance"]["noul"])


def run_laya(items: list[dict], revision: str) -> tuple[list[dict], list[dict]]:
    from laya import Router
    from transformers import AutoTokenizer
    path = local_snapshot("convaiinnovations", "laya", revision)
    tokenizer = AutoTokenizer.from_pretrained(path + "/tokenizer")
    router = Router(models={"english": path}, device="cuda", default="english", max_loaded=1)
    out, groups_out = [], []
    for count, item in enumerate(items, start=1):
        row = base_row(item, "laya", revision)
        try:
            views = [make_views(item["question"], c, tokenizer) for c in item["candidates"]]
            schema = laya_schema()
            requests_batch = [{"state": pointwise_state(view), "questions": schema, "model": "english", "max_len": 512} for view in views]
            start = time.perf_counter_ns()
            results = router.predict_batch(requests_batch, batch_size=4, sort_by_length=True)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            scores = [parse_laya_score(result) for result in results]
            truncated = any(bool(result.get("usage", {}).get("truncated")) for result in results)
            row.update({"status": "ok", "real_inference": True, "batch_duration_ms": elapsed, "input_tokens": sum(result.get("usage", {}).get("input_tokens", 0) for result in results), "truncated": truncated, "request_hash": sha256_json([view.selection_text_sha256 for view in views]), "response_hash": sha256_json(results), "pointwise": [{"doc_id": view.doc_id, "score": score, "duration_ms": elapsed / 20.0, "input_tokens": result.get("usage", {}).get("input_tokens"), "selection_text_sha256": view.selection_text_sha256, "real_inference": True, "truncated": bool(result.get("usage", {}).get("truncated"))} for view, score, result in zip(views, scores, results)]})
            group_row = run_laya_groups(item, views, router, tokenizer, revision)
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
            group_row = {"query_id": item["query_id"], "dataset": item["dataset"], "model_id": LAYER_PATHS["laya"][0], "model_key": "laya", "revision": revision, "status": "error", "groups": [], "real_inference": False, "error_type": row["error_type"]}
        out.append(row)
        groups_out.append(group_row)
        if count % 25 == 0 or count == len(items): print(f"laya pointwise+groupwise {count}/{len(items)}", flush=True)
    return out, groups_out


def run_laya_groups(item, views, router, tokenizer, revision):
    quartets = deterministic_quartets([dict(view.__dict__) for view in views], item["query_id"], SEED)
    groups = []
    for group_index, quartet in enumerate(quartets):
        quartet_views = [CandidateView(**value) for value in quartet]
        states, schema = groupwise_states(item["question"], quartet_views)
        start = time.perf_counter_ns()
        result = router.predict(states[0], schema, model="english", max_len=512)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        label = str(result["answers"]["choice"]["choice"])
        if label not in "ABCD":
            raise RuntimeError(f"invalid Laya choice {label!r}")
        winner = quartet_views["ABCD".index(label)]
        groups.append({"group_index": group_index, "group_doc_ids": [view.doc_id for view in quartet_views], "winning_label": label, "winning_doc_id": winner.doc_id, "duration_ms": elapsed, "input_tokens": result.get("usage", {}).get("input_tokens"), "request_hash": sha256_json({"question": winner.question_view, "group": [view.selection_text_sha256 for view in quartet_views]}), "response_hash": sha256_json(result), "real_inference": True, "truncated": bool(result.get("usage", {}).get("truncated"))})
    return {"query_id": item["query_id"], "dataset": item["dataset"], "model_id": LAYER_PATHS["laya"][0], "model_key": "laya", "revision": revision, "status": "ok", "groups": groups, "real_inference": True}


def run_clef(items: list[dict], revision: str) -> tuple[list[dict], list[dict]]:
    import torch
    path = local_snapshot("Cloudflare", "clef-flash", revision)
    sys.path.insert(0, path)
    from joint_schema_model import collate_records, encode_record, load_release_model
    model, processor = load_release_model(path, device="cuda")
    out, groups_out = [], []
    tokenizer = common_tokenizer()
    for count, item in enumerate(items, start=1):
        row = base_row(item, "clef", revision)
        try:
            views = [make_views(item["question"], c, tokenizer) for c in item["candidates"]]
            records = [{"state": pointwise_state(view), "questions": pointwise_question()} for view in views]
            encoded = [encode_record(processor.tokenizer, record, processor=processor) for record in records]
            start = time.perf_counter_ns()
            logits = []
            input_tokens = 0
            for begin in range(0, len(encoded), 4):
                batch_records = encoded[begin : begin + 4]
                batch = collate_records(batch_records, processor.tokenizer.pad_token_id, torch.device("cuda"))
                gpu_sync(torch)
                with torch.inference_mode():
                    batch_logits = model(batch)
                gpu_sync(torch)
                logits.extend(batch_logits)
                input_tokens += sum(int(value.attention_mask.sum()) if hasattr(value, "attention_mask") else len(value.input_ids) for value in batch_records)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            scores = []
            for record_logits in logits:
                question_logits = record_logits[0]
                encoded_options = [str(option) for option in records[len(scores)]["questions"]["relevance"]["criteria"]]
                option_ids = [str(option) for option in encoded[len(scores)].questions[0].option_ids]
                probability = torch.softmax(question_logits.float(), dim=-1).detach().cpu().tolist()
                scores.append(float(probability[option_ids.index("true")]))
            row.update({"status": "ok", "real_inference": True, "batch_duration_ms": elapsed, "input_tokens": input_tokens, "request_hash": sha256_json([view.selection_text_sha256 for view in views]), "response_hash": sha256_json(scores), "pointwise": [{"doc_id": view.doc_id, "score": score, "duration_ms": elapsed / 20.0, "input_tokens": None, "selection_text_sha256": view.selection_text_sha256, "real_inference": True} for view, score in zip(views, scores)]})
            group_row = run_clef_groups(item, views, model, processor, tokenizer, revision, torch, collate_records, encode_record)
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
            group_row = {"query_id": item["query_id"], "dataset": item["dataset"], "model_id": LAYER_PATHS["clef"][0], "model_key": "clef", "revision": revision, "status": "error", "groups": [], "real_inference": False, "error_type": row["error_type"]}
        out.append(row)
        groups_out.append(group_row)
        if count % 25 == 0 or count == len(items): print(f"clef pointwise+groupwise {count}/{len(items)}", flush=True)
    return out, groups_out


def clef_choice_from_logits(record_logits, encoded_record, torch):
    probabilities = torch.softmax(record_logits[0].float(), dim=-1).detach().cpu().tolist()
    return encoded_record.questions[0].option_ids[int(max(range(len(probabilities)), key=lambda index: probabilities[index]))], probabilities


def run_clef_groups(item, views, model, processor, tokenizer, revision, torch, collate_records, encode_record):
    quartets = deterministic_quartets([dict(view.__dict__) for view in views], item["query_id"], SEED)
    records, encoded_groups = [], []
    for quartet in quartets:
        quartet_views = [CandidateView(**value) for value in quartet]
        states, schema = groupwise_states(item["question"], quartet_views)
        records.append({"state": states[0], "questions": schema})
        encoded_groups.append((quartet_views, records[-1]))
    encoded = [encode_record(processor.tokenizer, record, processor=processor) for record in records]
    batch = collate_records(encoded, processor.tokenizer.pad_token_id, torch.device("cuda"))
    gpu_sync(torch)
    start = time.perf_counter_ns()
    with torch.inference_mode():
        outputs = model(batch)
    gpu_sync(torch)
    elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
    groups = []
    for index, (quartet_views, _) in enumerate(encoded_groups):
        label, probabilities = clef_choice_from_logits(outputs[index], encoded[index], torch)
        if label not in "ABCD":
            raise RuntimeError(f"invalid Clef choice {label!r}")
        winner = quartet_views["ABCD".index(label)]
        groups.append({"group_index": index, "group_doc_ids": [view.doc_id for view in quartet_views], "winning_label": label, "winning_doc_id": winner.doc_id, "duration_ms": elapsed / 5.0, "input_tokens": len(encoded[index].input_ids), "request_hash": sha256_json({"question": winner.question_view, "group": [view.selection_text_sha256 for view in quartet_views]}), "response_hash": sha256_json(probabilities), "real_inference": True, "truncated": False})
    return {"query_id": item["query_id"], "dataset": item["dataset"], "model_id": LAYER_PATHS["clef"][0], "model_key": "clef", "revision": revision, "status": "ok", "groups": groups, "real_inference": True, "batch_duration_ms": elapsed}


def run_laya_groupwise_only(items: list[dict], revision: str) -> list[dict]:
    from laya import Router
    tokenizer = common_tokenizer()
    path = local_snapshot("convaiinnovations", "laya", revision)
    router = Router(models={"english": path}, device="cuda", default="english", max_loaded=1)
    output = []
    for count, item in enumerate(items, start=1):
        views = [make_views(item["question"], c, tokenizer) for c in item["candidates"]]
        output.append(run_laya_groups(item, views, router, tokenizer, revision))
        if count % 25 == 0 or count == len(items): print(f"laya groupwise repair {count}/{len(items)}", flush=True)
    return output


def run_clef_groupwise_only(items: list[dict], revision: str) -> list[dict]:
    import torch
    path = local_snapshot("Cloudflare", "clef-flash", revision)
    sys.path.insert(0, path)
    from joint_schema_model import collate_records, encode_record, load_release_model
    model, processor = load_release_model(path, device="cuda")
    tokenizer = common_tokenizer()
    output = []
    for count, item in enumerate(items, start=1):
        views = [make_views(item["question"], c, tokenizer) for c in item["candidates"]]
        output.append(run_clef_groups(item, views, model, processor, tokenizer, revision, torch, collate_records, encode_record))
        if count % 25 == 0 or count == len(items): print(f"clef groupwise repair {count}/{len(items)}", flush=True)
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(LAYER_PATHS))
    parser.add_argument("--split", choices=["all", "cal", "test"], default="all")
    parser.add_argument("--group-only", action="store_true", help="rerun only the affected Laya/Clef groupwise records using existing pointwise traces")
    args = parser.parse_args()
    all_items = common_query_rows()
    split_map = {row["query_id"]: row["split"] for row in load_rows(ROOT / "data/manifests/questions.jsonl")}
    items = [item for item in all_items if args.split == "all" or split_map[item["query_id"]] == args.split]
    model, revision = LAYER_PATHS[args.model]
    prefix = ROOT / "data/results/m3" / f"{args.model}_{args.split}"
    if args.group_only:
        if args.model not in ("laya", "clef"):
            raise SystemExit("--group-only is valid only for laya or clef")
        existing_pointwise = load_rows(prefix.with_name(prefix.name + "_pointwise.jsonl"))
        if len(existing_pointwise) != len(items) or any(row["status"] != "ok" for row in existing_pointwise):
            raise SystemExit("existing pointwise trace is not complete; run the normal model pass first")
        groups = run_laya_groupwise_only(items, revision) if args.model == "laya" else run_clef_groupwise_only(items, revision)
        write_rows(prefix.with_name(prefix.name + "_groupwise.jsonl"), groups)
        summary = {"model": model, "model_key": args.model, "revision": revision, "split": args.split, "group_only_repair": True, "queries": len(items), "pointwise_ok": len(existing_pointwise), "groupwise_ok": sum(row["status"] == "ok" for row in groups), "jev_calls": 0, "hosted_cloudflare_calls": 0}
        (prefix.with_name(prefix.name + "_summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["groupwise_ok"] == len(items) else 2
    if args.model == "bge": pointwise, groups = run_bge(items, revision), []
    elif args.model == "qwen": pointwise, groups = run_qwen(items, revision), []
    elif args.model == "laya": pointwise, groups = run_laya(items, revision)
    else: pointwise, groups = run_clef(items, revision)
    write_rows(prefix.with_name(prefix.name + "_pointwise.jsonl"), pointwise)
    if groups:
        write_rows(prefix.with_name(prefix.name + "_groupwise.jsonl"), groups)
    summary = {"model": model, "model_key": args.model, "revision": revision, "split": args.split, "queries": len(items), "pointwise_ok": sum(row["status"] == "ok" for row in pointwise), "groupwise_ok": sum(row["status"] == "ok" for row in groups), "jev_calls": 0, "hosted_cloudflare_calls": 0}
    (prefix.with_name(prefix.name + "_summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pointwise_ok"] == len(items) and summary["groupwise_ok"] in (0, len(items)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
