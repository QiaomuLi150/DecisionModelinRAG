#!/usr/bin/env python3
"""R4 fixed-reader window experiment on the frozen E11-V visible-text cohort.

Selection outputs and reader configuration are frozen before native reader
calls.  Gold answers are used only by the offline post-freeze scorer and never
enter the model request body.
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
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E11 = ROOT / "results/non_jev/v2/e11v_v1_0"
SEL = E11 / "02_selection/html_text_repair_1"
OUT = ROOT / "results/v1_3/r4_crag_reader_window_v2"
RAW = OUT / "raw"
EXPORT = OUT / "export"
PROTOCOL_PATH = OUT / "PROTOCOL_V3.json"
INPUT_LOCK_PATH = OUT / "INPUT_LOCK_V3.json"
MODEL_REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/{MODEL_REV}")
PROMPT_REV = "E11V_FIXED_READER_QWEN_V1"
POLICIES = {
    "retrieval_top5": SEL / "V2_R0_RETRIEVAL_TOP5_VISIBLE_TEXT.jsonl",
    "jina_full": SEL / "V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.jsonl",
    "jina_budget": SEL / "V2_S2_JINA_BUDGETED_VISIBLE_TEXT_RAW.jsonl",
    "pplx_budget": SEL / "V2_P1_PPLX_BUDGETED_VISIBLE_TEXT_RAW.jsonl",
}
WINDOWS = [512, 1024]
MAX_NEW_TOKENS = 64
BATCH_SIZE = 2


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def words(text: str) -> list[str]:
    return re.findall(r"[\w]+", str(text or "").casefold(), flags=re.UNICODE)


def f1(pred: str, ref: str) -> float:
    a, b = words(pred), words(ref)
    if not a or not b:
        return float(a == b)
    ca, cb = {x: a.count(x) for x in set(a)}, {x: b.count(x) for x in set(b)}
    overlap = sum(min(ca.get(x, 0), cb.get(x, 0)) for x in ca)
    if not overlap:
        return 0.0
    p, r = overlap / len(a), overlap / len(b)
    return 2 * p * r / (p + r)


def exact(pred: str, ref: str) -> bool:
    return " ".join(words(pred)) == " ".join(words(ref))


def snapshot() -> dict:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        devices = []
        for line in raw.splitlines():
            x = [v.strip() for v in line.split(",")]
            if len(x) >= 7:
                devices.append({"index": int(x[0]), "uuid": x[1], "name": x[2], "memory_total_mib": int(x[3]), "memory_used_mib": int(x[4]), "memory_free_mib": int(x[5]), "utilization_percent": int(x[6])})
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
        return {"status": "ok", "devices": devices, "compute_apps": apps, "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    except Exception as exc:
        return {"status": "error", "error": repr(exc)}


def build_tasks():
    cohort = json.loads((E11 / "01_admission/V1_COHORT_MANIFEST.json").read_text(encoding="utf-8"))
    cohort_rows = cohort["representative"] + cohort["oversized_diagnostic"]
    qmeta = {r["question_id"]: r for r in cohort_rows}
    content_rows = {r["question_id"]: r for r in load_jsonl(SEL / "V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl")}
    selections = {}
    for policy, path in POLICIES.items():
        rows = load_jsonl(path)
        if len(rows) != 300 or any(r.get("status") not in ("ok", "PASS") for r in rows):
            raise RuntimeError(f"{policy}: expected 300 valid frozen selection rows")
        selections[policy] = {r["question_id"]: [str(x) for x in r["selected_ids"]] for r in rows}
    tasks = []
    for policy in POLICIES:
        for window in WINDOWS:
            for qid in sorted(qmeta):
                item = content_rows[qid]
                by_id = {str(d["candidate_id"]): d for d in item["candidates"]}
                selected = selections[policy][qid]
                if len(selected) != 5 or len(set(selected)) != 5 or any(x not in by_id for x in selected):
                    raise RuntimeError(f"invalid frozen selection {policy}/{qid}")
                passages = []
                for i, sid in enumerate(selected):
                    d = by_id[sid]
                    tok_source = str(d["content"])
                    # The actual truncation is performed after loading the
                    # pinned tokenizer, so this value is only a source record.
                    passages.append({"candidate_id": sid, "page_name": str(d.get("page_name") or ""), "content": tok_source, "source_rank": d.get("archived_rank")})
                prompt = (
                    "You must answer using only the supplied passages.\n"
                    "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
                    "Provide one short answer, not reasoning. Do not use external knowledge.\n"
                    f"Question: {item['query']}\nPassage<redacted-path>"
                    + "\n".join(f"[P{i} | {p['page_name']} | {p['candidate_id']}] {{CONTENT_{i}}}" for i, p in enumerate(passages))
                    + "\nAnswer:"
                )
                tasks.append({"task_id": f"{policy}::{window}::{qid}", "policy": policy, "window": window, "question_id": qid, "cohort": qmeta[qid]["cohort"], "domain": qmeta[qid]["domain"], "query": item["query"], "candidate_ids": item["candidate_ids"], "selected_ids": selected, "passages": passages, "prompt_template": prompt, "gold_in_reader_input": False})
    return tasks, qmeta, content_rows


def prepare_prompt(task: dict, tokenizer, window: int):
    views = []
    pieces = []
    for i, p in enumerate(task["passages"]):
        ids = tokenizer(p["content"], add_special_tokens=False, truncation=True, max_length=window, return_attention_mask=False)["input_ids"]
        view = tokenizer.decode(ids[:window], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        pieces.append(f"[P{i} | {p['page_name']} | {p['candidate_id']}] {view}")
        views.append({"candidate_id": p["candidate_id"], "reader_tokens": len(ids[:window]), "source_token_count": len(ids), "truncated": len(ids) > window, "view_sha256": sha_bytes(view.encode())})
    prompt = (
        "You must answer using only the supplied passages.\n"
        "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
        "Provide one short answer, not reasoning. Do not use external knowledge.\n"
        f"Question: {task['query']}\nPassage<redacted-path>" + "\n".join(pieces) + "\nAnswer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    body = json.dumps({"messages": messages, "generation": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}}, ensure_ascii=False, separators=(",", ":")).encode()
    task = dict(task)
    task.update({"prompt": prompt, "formatted_input": text, "prompt_views": views, "actual_model_input_sha256": sha_bytes(text.encode()), "request_body_sha256": sha_bytes(body), "gold_in_reader_input": False})
    return task


def run(gpu: int, smoke: bool = False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    tasks, qmeta, content = build_tasks()
    if smoke:
        tasks = tasks[:2]
    RAW.mkdir(parents=True, exist_ok=True); EXPORT.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    # Decoder-only generation with padded batches requires left padding.  The
    # initial smoke exposed right-padding in the first implementation; this
    # corrected version is frozen as R4 protocol V2.
    tokenizer.padding_side = "left"
    freeze = {
        "protocol_id": "decisionrag-v1.3-r4-reader-window-v2.0",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": MODEL_REV,
        "model_path": str(MODEL_PATH),
        "prompt_revision": PROMPT_REV,
        "windows": WINDOWS,
        "policies": list(POLICIES),
        "cohort_questions": 300,
        "cohort_manifest_sha256": sha_file(E11 / "01_admission/V1_COHORT_MANIFEST.json"),
        "visible_content_sha256": sha_file(SEL / "V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl"),
        "selection_hashes": {p: sha_file(path) for p, path in POLICIES.items()},
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
        "batch_size": BATCH_SIZE,
        "gold_in_reader_inputs": False,
        "reference_256_outputs": str((E11 / "03_reader/html_text_repair_1/V3_READER_RAW_OUTPUTS.jsonl").relative_to(ROOT)),
        "reference_256_sha256": sha_file(E11 / "03_reader/html_text_repair_1/V3_READER_RAW_OUTPUTS.jsonl"),
        "gpu": gpu,
        "gpu_pre_model": snapshot(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    freeze["protocol_id"] = "decisionrag-v1.3-r4-reader-window-v3.0"
    freeze["supersedes"] = ["PROTOCOL.json", "PROTOCOL_V2.json / failed pre-output implementation attempts"]
    if PROTOCOL_PATH.exists() and not smoke:
        raise RuntimeError(f"refusing to overwrite frozen protocol: {PROTOCOL_PATH}")
    PROTOCOL_PATH.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    INPUT_LOCK_PATH.write_text(json.dumps({"protocol_sha256": sha_file(PROTOCOL_PATH), "tasks": len(tasks), "smoke": smoke, "labels_joined": False, "padding_side": "left"}, indent=2) + "\n", encoding="utf-8")

    # Load after the protocol is frozen.  The model is local and no external
    # download is permitted by this continuation.
    model_load_start = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True, trust_remote_code=True).eval()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - model_load_start) * 1000
    results = []
    for start in range(0, len(tasks), BATCH_SIZE):
        batch = [prepare_prompt(t, tokenizer, t["window"]) for t in tasks[start:start + BATCH_SIZE]]
        prompts = [x["formatted_input"] for x in batch]
        recs = []
        for x in batch:
            recs.append({"task_id": x["task_id"], "policy": x["policy"], "window": x["window"], "question_id": x["question_id"], "cohort": x["cohort"], "domain": x["domain"], "selected_ids": x["selected_ids"], "prompt_views": x["prompt_views"], "actual_model_input_sha256": x["actual_model_input_sha256"], "generation": freeze["generation"], "status": "error", "answer": "", "gold_in_reader_input": False, "real_inference": False, "error": None, "model_revision": MODEL_REV, "gpu": gpu, "load_ms": load_ms})
        try:
            inputs = processor(text=prompts, padding=True, return_tensors="pt")
            inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
            torch.cuda.synchronize(); t0 = time.perf_counter_ns(); torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize(); duration = (time.perf_counter_ns() - t0) / 1e6
            input_len = inputs["input_ids"].shape[1]
            continuations = output[:, input_len:]
            answers = processor.batch_decode(continuations, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            peak = torch.cuda.max_memory_allocated() / 1024**2
            for rec, ans in zip(recs, answers):
                rec.update({"status": "ok", "answer": str(ans).strip(), "real_inference": True, "duration_ms": duration / len(recs), "batch_duration_ms": duration, "peak_memory_allocated_mib": peak})
        except Exception as exc:
            for rec in recs:
                rec.update({"error": f"{type(exc).__name__}: {exc}", "duration_ms": None, "peak_memory_allocated_mib": None})
        results.extend(recs)
        if len(results) % 100 < BATCH_SIZE:
            print(json.dumps({"completed": len(results), "total": len(tasks), "gpu": gpu, "window_counts": {str(win): sum(x["window"] == win for x in results) for win in WINDOWS}}), flush=True)

    suffix = "R4_SMOKE_RAW_V2.jsonl" if smoke else "R4_READER_RAW_OUTPUTS_V2.jsonl"
    rawpath = RAW / suffix
    rawpath.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in results) + "\n", encoding="utf-8")
    runmeta = {"rows": len(results), "ok": sum(x["status"] == "ok" for x in results), "errors": sum(x["status"] != "ok" for x in results), "raw_sha256": sha_file(rawpath), "model_load_ms": load_ms, "gpu_post_model": snapshot(), "gpu": gpu, "smoke": smoke}
    (RAW / ("R4_SMOKE_RUN_V2.json" if smoke else "R4_RUN_V2.json")).write_text(json.dumps(runmeta, indent=2) + "\n", encoding="utf-8")
    del model; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(runmeta, indent=2))


def answer_offset(page_text: str, refs: list[str], tokenizer) -> int | None:
    source_ids = tokenizer(page_text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    for ref in refs:
        ref_ids = tokenizer(str(ref), add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if not ref_ids:
            continue
        for i in range(0, max(0, len(source_ids) - len(ref_ids) + 1)):
            if source_ids[i:i + len(ref_ids)] == ref_ids:
                return i
    return None


def score():
    import importlib.util
    from transformers import AutoProcessor
    raw = load_jsonl(RAW / "R4_READER_RAW_OUTPUTS_V2.jsonl")
    cohort = json.loads((E11 / "01_admission/V1_COHORT_MANIFEST.json").read_text(encoding="utf-8"))
    qmeta = {r["question_id"]: r for r in cohort["representative"] + cohort["oversized_diagnostic"]}
    content = {r["question_id"]: r for r in load_jsonl(SEL / "V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl")}
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rows = []
    for x in raw:
        refs = [str(qmeta[x["question_id"]].get("answer") or "")] + [str(y) for y in qmeta[x["question_id"]].get("alt_ans") or []]
        by = {str(d["candidate_id"]): d for d in content[x["question_id"]]["candidates"]}
        offsets = [answer_offset(by[sid]["content"], refs, tokenizer) for sid in x["selected_ids"]]
        scores = [f1(x.get("answer", ""), ref) for ref in refs if ref.strip()]
        rows.append({**x, "reference_em_proxy": any(exact(x.get("answer", ""), ref) for ref in refs if ref.strip()), "reference_f1_proxy": max(scores, default=0.0), "answer_offset_min": min((o for o in offsets if o is not None), default=None), "gold_reference_count": len([r for r in refs if r.strip()]), "answer_offsets_by_selected_page": offsets})
    scored = EXPORT / "crag_reader_window_v3.csv"
    fields = ["question_id", "cohort", "domain", "policy", "window", "status", "reference_em_proxy", "reference_f1_proxy", "answer_offset_min"]
    with scored.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for x in rows: writer.writerow({k: x.get(k) for k in fields})
    summary = []
    for policy in POLICIES:
        for window in WINDOWS:
            for cohort_name in ["representative", "oversized_diagnostic"]:
                xs = [x for x in rows if x["policy"] == policy and x["window"] == window and x["cohort"] == cohort_name]
                ok = [x for x in xs if x["status"] == "ok"]
                summary.append({"policy": policy, "window": window, "cohort": cohort_name, "n": len(xs), "reader_ok": len(ok), "reader_errors": len(xs) - len(ok), "em_all": sum(bool(x["reference_em_proxy"]) for x in xs) / len(xs) if xs else None, "f1_all": sum(x["reference_f1_proxy"] for x in xs) / len(xs) if xs else None, "answer_offset_present": sum(x["answer_offset_min"] is not None for x in xs) / len(xs) if xs else None})
    (EXPORT / "crag_reader_window_summary_v3.json").write_text(json.dumps({"summary": summary, "endpoint": "deterministic answer/reference proxies, not factual correctness", "answer_offset_definition": "first contiguous tokenizer-token match in the original selected-page text"}, indent=2) + "\n", encoding="utf-8")
    (RAW / "R4_SCORED_RAW_OUTPUTS_V3.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": summary}, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=3); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score: score()
    else: run(args.gpu, args.smoke)


if __name__ == "__main__":
    main()
