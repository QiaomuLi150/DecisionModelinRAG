#!/usr/bin/env python3
"""Blinded local adjudication of the frozen 720 reader cases.

The local judge sees only the question, selected passages, and reader answer.
References, gold IDs, policy identifiers, model scores, and expected labels
are intentionally excluded from every request.  Results are explicitly
LLM_ADJUDICATED and never represented as human gold.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/non_jev/v2/e2_sp_r/E2_SP_R_ADJUDICATION_CASES.jsonl"
OUT = ROOT / "results/non_jev/v2/e2_sp_r2"
URL = os.environ.get("E2_R2_JUDGE_URL", "http://service.example.invalid/v1/chat/completions")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SEED = 731
WORKERS = int(os.environ.get("E2_R2_JUDGE_WORKERS", "8"))
MAX_TOKENS = int(os.environ.get("E2_R2_JUDGE_MAX_TOKENS", "384"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def load_cases() -> list[dict]:
    return [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_for(case: dict) -> str:
    passages = []
    for i, passage in enumerate(case["selected_passages"], 1):
        passages.append(f"PASSAGE {i}\nTITLE: {passage.get('title', '')}\nTEX<redacted-path>{passage.get('text', '')}")
    return (
        "You are an evidence-grounded answer-quality evaluator. Evaluate the answer using only the supplied question, "
        "selected passages, and answer. Do not infer facts from outside knowledge. If the passages do not allow a "
        "reliable decision, use UNJUDGEABLE. Return one JSON object only with exactly these fields: "
        "correctness (PASS, FAIL, or UNJUDGEABLE), groundedness (PASS, FAIL, or UNJUDGEABLE), "
        "completeness (PASS, FAIL, or UNJUDGEABLE), confidence (HIGH, MEDIUM, or LOW), rationale (no more than 40 words). "
        "Do not repeat the decision or deliberate in the rationale.\n\n"
        f"QUESTIO<redacted-path>{case['question']}\n\nSELECTED PASSAGE<redacted-path>" + "\n\n".join(passages) +
        f"\n\nREADER ANSWE<redacted-path>{case['reader_answer']}"
    )


def parse_json(content: str) -> tuple[dict | None, str]:
    value = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, flags=re.S | re.I)
    if fenced:
        value = fenced.group(1)
    else:
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    try:
        obj = json.loads(value)
    except Exception as exc:
        return None, f"parse_error:{type(exc).__name__}"
    allowed = {
        "correctness": {"PASS", "FAIL", "UNJUDGEABLE"},
        "groundedness": {"PASS", "FAIL", "UNJUDGEABLE"},
        "completeness": {"PASS", "FAIL", "UNJUDGEABLE"},
        "confidence": {"HIGH", "MEDIUM", "LOW"},
    }
    for key, choices in allowed.items():
        if obj.get(key) not in choices:
            return None, f"invalid_{key}"
    if not isinstance(obj.get("rationale"), str):
        return None, "invalid_rationale"
    return {key: obj[key] for key in (*allowed, "rationale")}, "ok"


def request_one(case: dict, repeat_index: int) -> dict:
    prompt = prompt_for(case)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "stream": False,
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    started = time.perf_counter_ns()
    base = {
        "record_type": "local_adjudication",
        "case_id": case["case_id"],
        "query_id": case["query_id"],
        "repeat_index": repeat_index,
        "candidate_ids": case.get("candidate_ids", []),
        "selected_doc_ids": case.get("selected_doc_ids", []),
        "judge_model": MODEL,
        "judge_revision": REVISION,
        "assessment_provenance": "LLM_ADJUDICATED_LOCAL_BLINDED",
        "human_judgment": False,
        "gold_in_prompt": False,
        "reference_answer_in_prompt": False,
        "policy_identity_in_prompt": False,
        "model_scores_in_prompt": False,
        "cache_reuse": False,
        "cache_provenance": "none_direct_http_request",
        "request_body_sha256": sha_bytes(encoded),
        "status": "error",
    }
    try:
        req = urllib.request.Request(URL, data=encoded, headers={"Content-Type": "application/json", "User-Agent": "DecisionRAG-E2-SP-R2"}, method="POST")
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
        response_sha = sha_bytes(raw)
        payload = json.loads(raw.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed, parse_status = parse_json(content)
        base.update({
            "status": "ok" if parsed is not None else "invalid_response",
            "raw_response_sha256": response_sha,
            "response_content": content,
            "parsed": parsed,
            "parse_status": parse_status,
            "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
            "usage": payload.get("usage"),
            "service_model": payload.get("model"),
        })
    except Exception as exc:
        base.update({"error": f"{type(exc).__name__}: {exc}", "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0})
    return base


def main() -> int:
    cases = load_cases()
    if len(cases) != 720 or len({c["case_id"] for c in cases}) != 720:
        raise RuntimeError(f"expected 720 unique cases, got {len(cases)}")
    repeat_cases = {c["case_id"] for i, c in enumerate(cases) if i % 5 == 0}
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "E2_SP_R2_LOCAL_ADJUDICATION.jsonl"
    existing = []
    if output.is_file():
        existing = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if os.environ.get("E2_R2_RETRY_INVALID") == "1" and existing:
        wanted = {(row["case_id"], int(row["repeat_index"])) for row in existing if row.get("status") != "ok"}
        jobs = [(case, repeat_index) for case in cases for repeat_index in (1, 2) if (case["case_id"], repeat_index) in wanted]
        if not jobs:
            print("no invalid adjudication rows to retry")
            return 0
    else:
        jobs = [(case, 1) for case in cases] + [(case, 2) for case in cases if case["case_id"] in repeat_cases]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(request_one, case, repeat_index) for case, repeat_index in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(futures):
                print(json.dumps({"stage": "R4_LOCAL_ADJUDICATION", "completed": index, "total": len(futures), "ok": sum(r["status"] == "ok" for r in rows)}), flush=True)
    if existing and os.environ.get("E2_R2_RETRY_INVALID") == "1":
        replacements = {(row["case_id"], int(row["repeat_index"])): row for row in rows}
        rows = [replacements.get((row["case_id"], int(row["repeat_index"])), row) for row in existing]
    rows.sort(key=lambda r: (r["case_id"], r["repeat_index"]))
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "stage": "R4_LOCAL_ADJUDICATION",
        "source_cases": str(SOURCE),
        "source_case_count": len(cases),
        "primary_judgments": len(cases),
        "repeat_case_count": len(repeat_cases),
        "total_requests": len(rows),
        "workers": WORKERS,
        "endpoint": URL,
        "model": MODEL,
        "revision": REVISION,
        "temperature": 0,
        "top_p": 1,
        "seed": SEED,
        "gold_in_prompt": False,
        "paid_api_calls": 0,
        "jev_calls": 0,
        "cache_reuse": False,
        "output_sha256": sha_bytes(output.read_bytes()),
        "status": "PASS" if all(r["status"] == "ok" for r in rows) else "PARTIAL_BLOCKED",
    }
    (OUT / "E2_SP_R2_LOCAL_ADJUDICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
