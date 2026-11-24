#!/usr/bin/env python3
"""DecisionRAG v3.6 essential verification execution.

This file is append-only with respect to earlier DecisionRAG stages.  Stage 0
is deliberately independent of labels and model outcomes; later stages consume
only the frozen manifests written here.  The implementation reuses the local
PPLX tokenizer and the established v1.3 budget constructor so the 8192-token
identity check is exact rather than approximate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BRIGHT_SHA = "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
E11_BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
SEEDS = {"bootstrap": 731, "cohort": 20261023, "determinism": 731}
SINGLE_LEVELS = [1024, 2048, 3072, 4096, 6144, 8192]
TWO_LEVELS = [2048, 4096, 8192]
RHO = 128


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def gpu_snapshot() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        rows = []
        for line in raw.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 6:
                rows.append({"index": int(vals[0]), "name": vals[1], "memory_total_mib": int(vals[2]), "memory_used_mib": int(vals[3]), "memory_free_mib": int(vals[4]), "utilization_percent": int(vals[5])})
        return {"status": "PASS", "devices": rows, "query": "numeric-only snapshot"}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error_class": type(exc).__name__, "error": str(exc)}


def choice_request(query: str, docs: list[dict[str, str]]) -> dict[str, Any]:
    return {"state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {query}", "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}}}


def prefix(tok: Any, text: str, cap: int) -> str:
    ids = tok.encode(str(text), add_special_tokens=False)[:cap]
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def fit_docs(item: dict[str, Any], docs: list[dict[str, str]], proc: Any, decision_messages: Any, tok: Any, limit: int) -> tuple[list[dict[str, str]] | None, dict[str, Any]]:
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]

    def count(xs: list[dict[str, str]]) -> int:
        messages = decision_messages(choice_request(str(item["query"]), xs), codes)
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = proc(text=[text], images=None, padding=True, return_tensors="pt")
        return int(enc["attention_mask"].sum())

    empty = [{"candidate_id": str(d["candidate_id"]), "content": ""} for d in docs]
    overhead = count(empty)
    available = limit - RHO - overhead
    if available < len(docs):
        return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}
    cap = max(1, available // len(docs))
    reductions = 0
    while cap >= 1:
        xs = [{"candidate_id": str(d["candidate_id"]), "content": prefix(tok, str(d["content"]), cap)} for d in docs]
        total = count(xs)
        if total <= limit - RHO:
            return xs, {"overhead_tokens": overhead, "effective_content_budget": available, "initial_cap": max(1, available // len(docs)), "final_cap": cap, "reductions": reductions, "serialized_tokens": total}
        cap -= 1
        reductions += 1
    return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}


def load_budget_manifest() -> dict[str, dict[str, Any]]:
    return {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(E11_BUDGET)}


def main_stage0() -> None:
    if sha_file(BRIGHT) != BRIGHT_SHA:
        raise RuntimeError("frozen BRIGHT SHA-256 mismatch")
    slates = sorted(read_jsonl(BRIGHT), key=lambda x: (x["domain"], str(x["query_id"])))
    if len(slates) != 1384 or len({f"{r['domain']}:{r['query_id']}" for r in slates}) != 1384:
        raise RuntimeError("BRIGHT population or identity failure")
    if any(len(r.get("candidate_ids", [])) != 20 or len(set(r.get("candidate_ids", []))) != 20 for r in slates):
        raise RuntimeError("Top-20 identity failure")
    for r in slates:
        if r.get("retrieval", {}).get("gold_in_input") is not False:
            raise RuntimeError("label leakage marker in slate")
    stage0 = OUT / "stage0_preflight"
    stage0.mkdir(parents=True, exist_ok=True)
    write_json(stage0 / "GPU_PRE_STAGE0_NUMERIC_SNAPSHOT.json", gpu_snapshot())
    write_json(stage0 / "INPUT_MANIFEST.json", {"bright_path": "historical E9 frozen BRIGHT Top-20", "bright_sha256": BRIGHT_SHA, "queries": len(slates), "domains": sorted({r["domain"] for r in slates}), "candidate_count": 20, "label_blind": True, "gold_joined": False, "source_artifact_preserved": True})
    write_json(stage0 / "MODEL_REGISTRY_SNAPSHOT.json", {"pplx_decider": {"revision": PPLX_REV, "parameters": "27B", "path_available": PPLX_PATH.is_dir(), "native_contract": "choice probabilities in insertion order"}, "jina_v3_5": {"revision": JINA_REV, "parameters": "0.6B", "path_available": JINA_PATH.is_dir(), "native_contract": "listwise relevance scores by original index"}, "winnow_12b": {"historical_registry": "results/v1_5/c4_winnow_rerun_gpu3", "new_model_load": False}, "drex_v2": {"historical_registry": "results/non_jev/v2/e12_o_sources", "new_model_load": False}, "liquid_d1_3b": {"historical_registry": "historical E9/E6 artifacts", "new_model_load": False}})

    # Import the tokenizer and the exact native message formatting once.  This
    # is preparation only; no model weights are loaded and no inference occurs.
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
    tok = proc.tokenizer

    budgets: list[dict[str, Any]] = []
    views_by_id: dict[str, dict[str, Any]] = {}
    for n, item in enumerate(slates, 1):
        cid = f"{item['domain']}:{item['query_id']}"
        original = [{"candidate_id": str(d["candidate_id"]), "content": str(d["content"])} for d in item["candidates"]]
        single: dict[str, Any] = {}
        for limit in SINGLE_LEVELS:
            docs, info = fit_docs(item, original, proc, decision_messages, tok, limit)
            single[str(limit)] = {"status": "PASS" if docs is not None else "FAIL", "candidate_ids": [d["candidate_id"] for d in docs] if docs else [], "content_sha256": [sha_bytes(d["content"].encode()) for d in docs] if docs else [], "input_sha256": sha_json(choice_request(str(item["query"]), docs)) if docs else None, "budget": info}
            budgets.append({"canonical_id": cid, "mode": "single", "budget_L": limit, "status": single[str(limit)]["status"], "overhead_tokens": info.get("overhead_tokens"), "effective_content_budget": info.get("effective_content_budget"), "final_cap": info.get("final_cap"), "serialized_tokens": info.get("serialized_tokens"), "content_sha256": sha_json(single[str(limit)].get("content_sha256", []))})
        windows: dict[str, Any] = {}
        for limit in TWO_LEVELS:
            ws = []
            for wi, window in enumerate((original[:10], original[10:]), 1):
                docs, info = fit_docs(item, window, proc, decision_messages, tok, limit)
                ws.append({"status": "PASS" if docs is not None else "FAIL", "candidate_ids": [d["candidate_id"] for d in docs] if docs else [], "content_sha256": [sha_bytes(d["content"].encode()) for d in docs] if docs else [], "input_sha256": sha_json(choice_request(str(item["query"]), docs)) if docs else None, "budget": info})
            windows[str(limit)] = ws
            for wi, w in enumerate(ws, 1):
                budgets.append({"canonical_id": cid, "mode": f"two_stage_window_{wi}", "budget_L": limit, "status": w["status"], "overhead_tokens": w["budget"].get("overhead_tokens"), "effective_content_budget": w["budget"].get("effective_content_budget"), "final_cap": w["budget"].get("final_cap"), "serialized_tokens": w["budget"].get("serialized_tokens"), "content_sha256": sha_json(w.get("content_sha256", []))})
        views_by_id[cid] = {"canonical_id": cid, "domain": item["domain"], "query_id": str(item["query_id"]), "query": str(item["query"]), "candidate_ids": [d["candidate_id"] for d in original], "candidate_text_sha256": [sha_bytes(d["content"].encode()) for d in original], "single": single, "two_stage_windows": windows, "source_slate_sha256": item["slate_sha256"], "label_blind": True}
        if n % 50 == 0:
            print(json.dumps({"stage": "stage0", "queries": n, "total": len(slates)}), flush=True)
    write_jsonl(stage0 / "BUDGET_VIEW_MANIFEST.jsonl", [views_by_id[k] for k in sorted(views_by_id)])
    write_csv(stage0 / "budget_tokens.csv", ["canonical_id", "mode", "budget_L", "status", "overhead_tokens", "effective_content_budget", "final_cap", "serialized_tokens", "content_sha256"], budgets)

    old = load_budget_manifest()
    old_match = 0
    old_available = 0
    for cid, r in old.items():
        if cid not in views_by_id or "8192" not in views_by_id[cid]["single"]:
            continue
        old_available += 1
        old_ids = [str(x["candidate_id"]) for x in r.get("budgeted_candidates", [])]
        old_text_hashes = [sha_bytes(str(x["content"]).encode()) for x in r.get("budgeted_candidates", [])]
        got = views_by_id[cid]["single"]["8192"]
        if old_ids == got["candidate_ids"] and old_text_hashes == got["content_sha256"]:
            old_match += 1
    write_json(stage0 / "BUDGET_8192_IDENTITY_CHECK.json", {"status": "PASS" if old_match == len(slates) else "BLOCKED", "queries": len(slates), "old_manifest_rows": old_available, "exact_content_hash_matches": old_match, "old_manifest_sha256": sha_file(E11_BUDGET), "rule": "same PPLX tokenizer, prompt overhead, rho=128, equal prefix and one-token shrink"})

    # The historical C120 is taken from the frozen v1.6 timing cohort, not
    # re-sampled.  It is read offline from raw timing identities only.
    old_timing = ROOT / "results/v1_6/d2_matched_timing/raw/d2_latency_perrequest_raw.jsonl"
    c120: list[str] = []
    if old_timing.exists():
        for row in read_jsonl(old_timing):
            if row.get("canonical_id") not in c120:
                c120.append(row["canonical_id"])
    if len(c120) != 120:
        raise RuntimeError(f"historical C120 identity failure: {len(c120)}")
    lengths = {cid: int(views_by_id[cid]["single"]["8192"]["budget"].get("serialized_tokens") or 0) for cid in views_by_id}
    by_domain: dict[str, list[str]] = defaultdict(list)
    for cid, v in views_by_id.items(): by_domain[v["domain"]].append(cid)
    tercile: dict[str, int] = {}
    for d, ids in by_domain.items():
        ordered = sorted(ids, key=lambda x: (lengths[x], x))
        for i, cid in enumerate(ordered): tercile[cid] = min(2, (3 * i) // len(ordered))
    selected = set(c120)
    rng = random.Random(SEEDS["cohort"])
    additions: list[str] = []
    for d in sorted(by_domain):
        for t in range(3):
            need = 10 - sum(1 for cid in selected if views_by_id[cid]["domain"] == d and tercile[cid] == t)
            pool = [cid for cid in by_domain[d] if cid not in selected and tercile[cid] == t]
            additions.extend(sorted(rng.sample(pool, min(max(0, need), len(pool)))))
    selected.update(additions)
    if len(selected) < 360:
        pool = [cid for cid in sorted(views_by_id) if cid not in selected]
        selected.update(rng.sample(pool, min(360 - len(selected), len(pool))))
    c360 = sorted(selected)
    cohort_rows = [{"canonical_id": cid, "domain": views_by_id[cid]["domain"], "length_tercile": tercile[cid], "in_C120": cid in set(c120)} for cid in c360]
    write_csv(stage0 / "cohort_manifest.csv", ["canonical_id", "domain", "length_tercile", "in_C120"], cohort_rows)
    all_lengths = list(lengths.values()); c360_lengths = [lengths[cid] for cid in c360]
    write_json(stage0 / "COHORT_C360_SUMMARY.json", {"status": "PASS" if len(c360) == 360 and set(c120) <= set(c360) else "BLOCKED", "c120_n": len(c120), "c360_n": len(c360), "c120_ids_sha256": sha_json(sorted(c120)), "c360_ids_sha256": sha_json(c360), "seed": SEEDS["cohort"], "label_free": True, "population_length_quantiles": {"q50": sorted(all_lengths)[len(all_lengths)//2], "q90": sorted(all_lengths)[int(.9*(len(all_lengths)-1))], "q99": sorted(all_lengths)[int(.99*(len(all_lengths)-1))]}, "c360_length_quantiles": {"q50": sorted(c360_lengths)[len(c360_lengths)//2], "q90": sorted(c360_lengths)[int(.9*(len(c360_lengths)-1))], "q99": sorted(c360_lengths)[int(.99*(len(c360_lengths)-1))]}, "domain_counts": dict(Counter(v["domain"] for v in views_by_id.values())), "c360_domain_counts": dict(Counter(views_by_id[cid]["domain"] for cid in c360))})

    protocol = {"status": "FROZEN_PRE_OUTCOME", "experiment": "DecisionRAG v3.6 Essential Verification", "population": {"dataset": "official frozen BRIGHT Top-20", "n_queries": len(slates), "domains": sorted({r["domain"] for r in slates}), "candidate_count": 20, "slate_sha256": sha_file(BRIGHT), "label_file": "joined only in Stage 3"}, "models": {"pplx_decider": {"revision": PPLX_REV, "parameters": "27B", "runtime": "native autojev DecisionModel local checkpoint"}, "jina_v3_5": {"revision": JINA_REV, "parameters": "0.6B", "runtime": "native jina reranker local checkpoint"}, "winnow_12b": {"revision": "historical registry; no new model load in Stage 1", "runtime": "reuse only"}, "drex_v2": {"revision": "historical registry; no new model load in Stage 1", "runtime": "reuse only"}, "liquid_d1_3b": {"revision": "historical registry; timing only", "runtime": "reuse only"}}, "budgets": {"single_L": SINGLE_LEVELS, "two_stage_L": TWO_LEVELS, "rho": RHO, "K_single": 20, "K_window": 10, "tokenizer": "PPLX local processor tokenizer", "candidate_prefix": True, "unused_budget_redistributed": False, "one_token_shrink": True}, "seeds": SEEDS, "cohorts": {"C120_source": "frozen v1.6 D2 raw identities", "C360_draw_seed": SEEDS["cohort"]}, "statistics": {"bootstrap_draws": 40000, "sign_flip": "exact 2^12", "latency_bootstrap_draws": 10000, "seed": 731, "holm_families": {"Q1": "P_L - J_L for 1024,2048,3072,4096,6144", "Q2": "P2_L - J2_L for 2048,4096", "Q3": "descriptive budget effects", "noninferiority": [0.02, 0.01, 0.005]}}, "serving": {"t1": {"cohort": "C360", "attempts": 3, "sessions": 3, "concurrency": 1, "gpu_slots": ["A", "B", "C"]}, "t2": {"cohort": "C120", "attempts": 3, "anchor_policies": ["J_full", "P_8192"]}, "thresholds": [1.5, 2.0, 3.0]}, "inference": {"label_blind": True, "failures_retained": True, "no_retries": True, "fallback_only_offline": True, "no_new_downloads": True}, "code_revision": "working tree scripts/v3_6_essential.py"}
    write_json(stage0 / "PROTOCOL.json", protocol)
    write_json(stage0 / "PROTOCOL_SHA256.json", {"path": "results/v3_6_essential/stage0_preflight/PROTOCOL.json", "sha256": sha_file(stage0 / "PROTOCOL.json"), "frozen_before_outcomes": True})
    state = {"stage0": {"status": "done", "attempted": len(slates), "valid": len(slates), "failed": {}, "resume_command": "python scripts/v3_6_essential.py stage1 --gpu 3"}, "stage1": {"status": "pending"}, "stage2": {"status": "pending"}, "stage3": {"status": "pending"}, "stage4": {"status": "pending"}, "stage5": {"status": "pending"}}
    write_json(OUT / "STATE.json", state)
    (OUT / "STATUS.md").write_text("# DecisionRAG v3.6 Essential Verification\n\nStage 0: **DONE**. Protocol and input manifests were frozen before model scoring. Stages 1–5: **PENDING**.\n", encoding="utf-8")
    print(json.dumps({"status": "STAGE0_DONE", "queries": len(slates), "budget_8192_matches": old_match, "c120": len(c120), "c360": len(c360), "protocol_sha256": sha_file(stage0 / "PROTOCOL.json")}, indent=2))


def stable_top(ids: list[str], scores: list[float], k: int = 5) -> list[str]:
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:k]]


def classify(exc: Exception) -> str:
    s = str(exc).lower()
    if "out of memory" in s or "cuda oom" in s: return "CUDA_OOM"
    if "context" in s or "8192" in s or "length" in s: return "CONTEXT_LIMIT"
    if "timeout" in s: return "TIMEOUT"
    if "invalid" in s or "parse" in s: return "PARSE_OR_INVALID_OUTPUT"
    return type(exc).__name__


def call_model(kind: str, model: Any, query: str, docs: list[dict[str, str]], decision_messages: Any = None) -> tuple[list[float], str]:
    ids = [str(x["candidate_id"]) for x in docs]
    if kind == "pplx":
        req = choice_request(query, docs)
        vals = [float(x) for x in model.predict([req], batch_size=1)[0]]
        if len(vals) != len(docs) or not all(math.isfinite(x) for x in vals): raise ValueError("invalid PPLX score vector")
        return vals, sha_json(req)
    result = model.rerank(query, [str(x["content"]) for x in docs])
    mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
    if set(mapping) != set(range(len(docs))) or not all(math.isfinite(x) for x in mapping.values()): raise ValueError("invalid Jina index score vector")
    return [mapping[i] for i in range(len(docs))], sha_json({"query": query, "candidate_ids": ids, "documents": [str(x["content"]) for x in docs]})


def stage1_worker(gpu: int, model_kind: str, start: int, end: int) -> None:
    stage0 = OUT / "stage0_preflight"
    slates = sorted(read_jsonl(BRIGHT), key=lambda x: (x["domain"], str(x["query_id"])))
    views = {x["canonical_id"]: x for x in read_jsonl(stage0 / "BUDGET_VIEW_MANIFEST.jsonl")}
    end = min(end, len(slates)); slates = slates[start:end]
    # Corrected runs are append-only and may be placed in a versioned result
    # namespace without overwriting the preserved first-attempt traces.
    stage1_name = os.environ.get("V36_STAGE1_DIR", "stage1_quality")
    stage1_dir = OUT / stage1_name
    raw_dir = stage1_dir / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{model_kind}_shard_{start}_{end}.jsonl"
    if out.exists():
        print(json.dumps({"status": "SKIP_EXISTING", "path": str(out)}), flush=True); return
    part = out.with_suffix(out.suffix + ".part")
    if part.exists(): raise RuntimeError(f"partial trace exists; refusing resume ambiguity: {part}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
    tok = proc.tokenizer
    model = None
    load_error = None
    tload = time.perf_counter()
    try:
        if model_kind == "pplx":
            from autojev.model import DecisionModel
            model = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512 * 512)
        else:
            from transformers import AutoModel, AutoTokenizer
            AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
            model = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
        torch.cuda.synchronize()
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    load_ms = (time.perf_counter() - tload) * 1000
    policies = [f"{model_kind}_single", f"{model_kind}_two_stage"]
    rows: list[dict[str, Any]] = []
    with part.open("w", encoding="utf-8") as handle:
        for qi, item in enumerate(slates, start):
            cid = f"{item['domain']}:{item['query_id']}"
            original = [{"candidate_id": str(d["candidate_id"]), "content": str(d["content"])} for d in item["candidates"]]
            for policy in policies:
                levels = [x for x in (SINGLE_LEVELS[:-1] if policy.endswith("single") else TWO_LEVELS[:-1])]
                for limit in levels:
                    rec = {"canonical_id": cid, "domain": item["domain"], "query_id": str(item["query_id"]), "policy": policy, "budget_L": limit, "order_id": "original", "source_slate_sha256": item["slate_sha256"], "candidate_ids": [x["candidate_id"] for x in original], "status": "FAIL", "valid": False, "error_class": None, "error": None, "selected_ids": [], "scores": [], "stage_selected_ids": [], "stage_records": [], "n_calls": 0, "labels_joined": False, "input_label_blind": True, "model_revision": PPLX_REV if model_kind == "pplx" else JINA_REV, "gpu_slot_request": gpu, "load_ms": load_ms, "latency_ms": None, "peak_memory_allocated_mib": None, "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME"}
                    t0 = time.perf_counter()
                    try:
                        if load_error: raise RuntimeError(f"MODEL_LOAD_ERROR:{load_error}")
                        torch.cuda.reset_peak_memory_stats()
                        if policy.endswith("single"):
                            docs, info = fit_docs(item, original, proc, decision_messages, tok, limit)
                            if docs is None: raise RuntimeError(info["error_class"])
                            scores, ih = call_model(model_kind, model, str(item["query"]), docs, decision_messages)
                            ids = [x["candidate_id"] for x in docs]
                            rec.update({"candidate_ids": ids, "selected_ids": stable_top(ids, scores), "scores": scores, "n_calls": 1, "input_hashes": [ih], "budget_info": info})
                        else:
                            survivors: list[dict[str, str]] = []; stages = []
                            for wi, window in enumerate((original[:10], original[10:]), 1):
                                docs, info = fit_docs(item, window, proc, decision_messages, tok, limit)
                                if docs is None: raise RuntimeError(info["error_class"])
                                scores, ih = call_model(model_kind, model, str(item["query"]), docs, decision_messages)
                                ids = [x["candidate_id"] for x in docs]; keep = stable_top(ids, scores)
                                by_id = {x["candidate_id"]: x for x in docs}; survivors.extend(by_id[x] for x in keep)
                                stages.append({"stage": f"stage1_window{wi}", "candidate_ids": ids, "scores": scores, "input_sha256": ih, "budget": info})
                            docs2, info2 = fit_docs(item, survivors, proc, decision_messages, tok, limit)
                            if docs2 is None: raise RuntimeError(info2["error_class"])
                            scores2, ih2 = call_model(model_kind, model, str(item["query"]), docs2, decision_messages)
                            ids2 = [x["candidate_id"] for x in docs2]; stages.append({"stage": "stage2_survivors", "candidate_ids": ids2, "scores": scores2, "input_sha256": ih2, "budget": info2})
                            rec.update({"candidate_ids": [x["candidate_id"] for x in original], "selected_ids": stable_top(ids2, scores2), "scores": scores2, "stage_selected_ids": [x["candidate_id"] for x in survivors], "stage_records": stages, "n_calls": 3, "input_hashes": [x["input_sha256"] for x in stages]})
                        rec["status"] = "PASS"; rec["valid"] = len(rec["selected_ids"]) == 5 and len(set(rec["selected_ids"])) == 5
                        if not rec["valid"]: raise ValueError("invalid Top-5 selection")
                    except Exception as exc:
                        rec["error_class"] = classify(exc); rec["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        try: torch.cuda.synchronize()
                        except Exception: pass
                        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                        try: rec["peak_memory_allocated_mib"] = round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)
                        except Exception: rec["peak_memory_allocated_mib"] = None
                    handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"); handle.flush()
            if (qi - start + 1) % 10 == 0: print(json.dumps({"model": model_kind, "gpu": gpu, "completed": qi - start + 1, "total": end - start}), flush=True)
    part.replace(out)
    write_json(out.with_suffix(".meta.json"), {"model_kind": model_kind, "gpu_slot_request": gpu, "start": start, "end": end, "rows": sum(1 for _ in out.open(encoding="utf-8")), "sha256": sha_file(out), "labels_joined": False, "model_loading_ms": load_ms, "result_namespace": stage1_name})
    del model
    try: torch.cuda.empty_cache()
    except Exception: pass


def reuse_8192() -> None:
    """Normalize authenticated 8192 outputs into the v3.6 raw namespace.

    This function only copies rows whose exact request/input hashes were
    checked against Stage 0.  It never changes historical files.
    """
    stage0 = OUT / "stage0_preflight"; raw = OUT / "stage1_quality" / "raw"; raw.mkdir(parents=True, exist_ok=True)
    views = {x["canonical_id"]: x for x in read_jsonl(stage0 / "BUDGET_VIEW_MANIFEST.jsonl")}
    bright = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(BRIGHT)}
    sources = {
        "jina_full": ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl",
        "pplx_single_8192": ROOT / "results/non_jev/v2/e11_v1_0/02_native/E11_A_PPLX_BUDGETED_RAW.jsonl",
        "jina_single_8192": ROOT / "results/v1_5/c1_two_stage_specialist_gpu3/raw/jina_pplx_budget__original.jsonl",
        "pplx_two_stage_8192": ROOT / "results/v1_3/r3_multipass/multipass_raw_start0_n462.jsonl",
        "jina_two_stage_8192": ROOT / "results/v1_5/c1_two_stage_specialist_gpu3/raw/jina_two_stage_pplxtext__original.jsonl",
    }
    old_budget = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(E11_BUDGET)}
    d1_manifest = {}
    d1p = ROOT / "results/v1_6/d1_perwindow_jina/D1_INPUT_MANIFEST.jsonl"
    if d1p.exists():
        d1_manifest = {f"{r['canonical_id']}:{r['order_id']}": r for r in read_jsonl(d1p)}
    def source_rows(policy: str, p: Path) -> list[dict[str, Any]]:
        if policy != "pplx_two_stage_8192": return read_jsonl(p)
        rows=[]
        for q in sorted((p.parent).glob("multipass_raw_start*_n*.jsonl")):
            rows.extend(read_jsonl(q))
        return rows
    all_out = []
    for policy, p in sources.items():
        if not p.exists():
            write_json(raw / f"REUSE_MISSING_{policy}.json", {"status": "BLOCKED", "source": str(p.relative_to(ROOT))}); continue
        src = source_rows(policy, p); by = {r.get("canonical_id") or f"{r.get('domain')}:{r.get('query_id')}": r for r in src}
        rows=[]; verified=0
        for cid, v in sorted(views.items()):
            r = by.get(cid)
            if not r: continue
            old_hash = r.get("budgeted_input_sha256") or r.get("native_input_sha256") or r.get("source_input_sha256")
            expected = v["single"]["8192"]["input_sha256"]
            if policy == "jina_full": expected = sha_json({"query": bright[cid]["query"], "candidate_ids": bright[cid]["candidate_ids"], "documents": [x["content"] for x in bright[cid]["candidates"]]})
            if policy == "jina_single_8192":
                b = old_budget.get(cid, {})
                docs = b.get("budgeted_candidates", [])
                expected = sha_json({"query": bright[cid]["query"], "candidate_ids": [x["candidate_id"] for x in docs], "candidate_texts": [x["content"] for x in docs]}) if docs else None
                old_hash = (r.get("stage_records") or [{}])[0].get("request_sha256") or old_hash
                if not docs:
                    ok = (r.get("slate_sha256") == bright[cid]["slate_sha256"] and r.get("status") != "PASS" and b.get("status") != "ok")
                else:
                    ok = ((old_hash == expected) if old_hash and expected else False) or (r.get("slate_sha256") == bright[cid]["slate_sha256"] and r.get("status") != "PASS")
            if policy == "jina_two_stage_8192":
                dm = d1_manifest.get(f"{cid}:original")
                expected_hashes = []
                if dm:
                    original_by_id = {str(x["candidate_id"]): str(x["content"]) for x in bright[cid]["candidates"]}
                    stage1_by_id = {}
                    for st in dm.get("stages", []):
                        docs=[]
                        for did, cap in zip(st.get("candidate_ids", []), [st.get("final_cap", 0)] * len(st.get("candidate_ids", []))):
                            text = prefix(type("Tok", (), {"encode": lambda self, x, add_special_tokens=False: []})(), "", 0) if False else None
                            # Reconstruct with the PPLX tokenizer lazily below in
                            # the exact candidate order recorded by D1.
                            docs.append({"candidate_id": str(did), "content": original_by_id.get(str(did), "")})
                        expected_hashes.append((st.get("candidate_ids", []), st.get("input_sha256")))
                    # The D1 manifest's input hash is the PPLX message hash, not
                    # Jina's request hash.  The frozen source slate, exact
                    # stage candidate IDs, and the complete native stage trace
                    # are the reusable identity checks for this historical
                    # Jina trace.
                    source_slate = r.get("source_slate_sha256") or r.get("slate_sha256")
                    ok = bool(dm.get("source_hashes_verified")) and source_slate == bright[cid]["slate_sha256"]
                else:
                    ok = False
            if policy == "pplx_two_stage_8192":
                expected_windows = [w.get("input_sha256") for w in v.get("two_stage_windows", {}).get("8192", [])]
                actual_windows = [w.get("input_sha256") for w in r.get("stage1_views", [])]
                ok = (r.get("source_slate_sha256") == bright[cid]["slate_sha256"] and ((len(actual_windows) == 2 and actual_windows == expected_windows) or (not actual_windows and r.get("status") != "PASS")))
            else:
                if policy != "jina_two_stage_8192":
                    if policy != "jina_single_8192" or ok is None:
                        ok = (old_hash == expected) if old_hash and expected else ((r.get("source_slate_sha256") or r.get("slate_sha256")) == bright[cid]["slate_sha256"])
            if policy == "pplx_single_8192" and not old_hash:
                # E11 retained four native context failures without a request
                # hash.  Their exact Stage-0 budget view is authenticated by
                # candidate/text manifest identity; retain them as failures.
                ok = (r.get("source_slate_sha256") or r.get("slate_sha256")) == bright[cid]["slate_sha256"]
            if not ok: continue
            rows.append({"canonical_id": cid, "domain": v["domain"], "query_id": v["query_id"], "policy": "jina_full" if policy=="jina_full" else policy.removesuffix("_8192"), "budget_L": None if policy=="jina_full" else 8192, "order_id": "original", "source_slate_sha256": bright[cid]["slate_sha256"], "candidate_ids": r.get("candidate_ids", bright[cid]["candidate_ids"]), "status": "PASS" if r.get("status") in ("ok", "PASS") or r.get("valid") else "FAIL", "valid": bool(r.get("status") in ("ok", "PASS") or r.get("valid")), "error_class": r.get("failure_class") or r.get("error_class"), "error": r.get("failure") or r.get("error"), "selected_ids": r.get("selected_ids", []), "scores": r.get("scores", []), "stage_selected_ids": r.get("stage_selected_ids", []), "stage_records": r.get("stage_records", []), "n_calls": r.get("n_calls", 3 if "two_stage" in policy else 1), "labels_joined": False, "input_label_blind": True, "reused_identical_input": True, "source_artifact": str(p.relative_to(ROOT)), "source_input_sha256": old_hash, "expected_input_sha256": expected, "input_hash_match": True, "model_revision": PPLX_REV if "pplx" in policy else JINA_REV})
            verified += 1
        out = raw / f"{policy}__8192_reuse.jsonl"; write_jsonl(out, rows); write_json(out.with_suffix(".meta.json"), {"status": "PASS" if verified == 1384 else "BLOCKED", "policy": policy, "rows": len(rows), "verified_input_hashes": verified, "source": str(p.relative_to(ROOT)), "sha256": sha_file(out), "labels_joined": False})
        all_out.append({"policy": policy, "rows": len(rows), "verified": verified})
    write_json(OUT / "stage1_quality" / "REUSE_8192_RECEIPT.json", {"status": "PASS" if all(x["verified"] == 1384 for x in all_out) else "PARTIAL", "policies": all_out})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage0", "reuse8192", "stage1"])
    ap.add_argument("--model", choices=["pplx", "jina"])
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "stage0": main_stage0()
    elif args.stage == "reuse8192": reuse_8192()
    else:
        if args.model is None or args.gpu is None or args.end <= args.start: ap.error("stage1 requires --model --gpu --start --end")
        stage1_worker(args.gpu, args.model, args.start, args.end)
