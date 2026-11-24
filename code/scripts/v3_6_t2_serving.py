#!/usr/bin/env python3
"""DecisionRAG v3.6 B6.3 T2 frontier timing.

This is an append-only companion to the completed v3.6 T1 runner.  It uses
the frozen BRIGHT slates and the Stage-0 PPLX-tokenized views, performs no
label access, never retries a request, and records failures with elapsed time.
Each T2 session contains the required J_full/P_8192 anchors plus a fixed
subset of the single- and two-stage budget sweep.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/stage2_t2"
STAGE0 = ROOT / "results/v3_6_essential/stage0_preflight"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
BRIGHT_SHA = "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
SINGLE = [1024, 2048, 3072, 4096, 6144, 8192]
TWO = [2048, 4096, 8192]
SESSIONS = {
    "A": {
        "pplx": ["pplx_budget", "pplx_single_1024", "pplx_single_2048", "pplx_two_stage_2048"],
        "jina": ["jina_full", "jina_single_1024", "jina_single_2048", "jina_two_stage_2048"],
    },
    "B": {
        "pplx": ["pplx_budget", "pplx_single_3072", "pplx_single_4096", "pplx_two_stage_4096"],
        "jina": ["jina_full", "jina_single_3072", "jina_single_4096", "jina_two_stage_4096"],
    },
    "C": {
        "pplx": ["pplx_budget", "pplx_single_6144", "pplx_two_stage_8192"],
        "jina": ["jina_full", "jina_single_6144", "jina_single_8192", "jina_two_stage_8192"],
    },
}


def import_v36():
    spec = importlib.util.spec_from_file_location("v36_serving", ROOT / "scripts/v3_6_serving.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("V36_SERVING_IMPORT_FAILED")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v36 = import_v36()
base = v36.base


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def prefix(tok, text: str, cap: int) -> str:
    ids = tok.encode(str(text), add_special_tokens=False)[:cap]
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def fit_docs(query: str, docs: list[dict[str, str]], proc, decision_messages, tok, limit: int):
    """Exact B1.4 builder, parameterized by the frozen T2 budget."""
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]

    def count(xs):
        messages = decision_messages(base.choice_request(query, xs), codes)
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = proc(text=[text], images=None, padding=True, return_tensors="pt")
        return int(enc["attention_mask"].sum())

    empty = [{"candidate_id": str(d["candidate_id"]), "content": ""} for d in docs]
    overhead = count(empty)
    available = limit - 128 - overhead
    if available < len(docs):
        return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead,
                      "effective_content_budget": available}
    cap = max(1, available // len(docs))
    reductions = 0
    while cap >= 1:
        out = [{"candidate_id": str(d["candidate_id"]), "content": prefix(tok, d["content"], cap)} for d in docs]
        total = count(out)
        if total <= limit - 128:
            return out, {"overhead_tokens": overhead, "initial_cap": max(1, available // len(docs)),
                         "final_cap": cap, "reductions": reductions, "serialized_tokens": total}
        cap -= 1
        reductions += 1
    return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead,
                  "effective_content_budget": available}


def build_views(proc, decision_messages, tok):
    slates = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BRIGHT)}
    if len(slates) != 1384 or sha_file(BRIGHT) != BRIGHT_SHA:
        raise RuntimeError("T2_FROZEN_BRIGHT_FAILURE")
    cohort = [r["canonical_id"] for r in csv.DictReader((STAGE0 / "cohort_manifest.csv").open(encoding="utf-8")) if r["in_C120"].lower() == "true"]
    if len(cohort) != 120 or len(set(cohort)) != 120:
        raise RuntimeError(f"T2_C120_FAILURE:{len(cohort)}")
    views = []
    manifest = []
    for cid in cohort:
        s = slates[cid]
        original = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in s["candidates"]]
        single = {}
        windows = {}
        for limit in SINGLE:
            docs, info = fit_docs(str(s["query"]), original, proc, decision_messages, tok, limit)
            single[str(limit)] = docs
            manifest.append({"canonical_id": cid, "mode": "single", "budget_L": limit,
                             "status": "PASS" if docs is not None else "FAIL",
                             "candidate_ids": [x["candidate_id"] for x in docs] if docs else [],
                             "content_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in docs] if docs else [],
                             "input_sha256": base.sha(base.choice_request(str(s["query"]), docs)) if docs else None, "builder": info})
        for limit in TWO:
            ws = []
            for wi, part in enumerate((original[:10], original[10:]), 1):
                docs, info = fit_docs(str(s["query"]), part, proc, decision_messages, tok, limit)
                ws.append(docs)
                manifest.append({"canonical_id": cid, "mode": f"two_stage_window_{wi}", "budget_L": limit,
                                 "status": "PASS" if docs is not None else "FAIL",
                                 "candidate_ids": [x["candidate_id"] for x in docs] if docs else [],
                                 "content_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in docs] if docs else [],
                                 "input_sha256": base.sha(base.choice_request(str(s["query"]), docs)) if docs else None, "builder": info})
            windows[str(limit)] = ws
        views.append({"canonical_id": cid, "domain": str(s["domain"]), "query_id": str(s["query_id"]),
                      "query": str(s["query"]), "original": original, "single": single, "windows": windows,
                      "original_ids": [x["candidate_id"] for x in original], "slate_sha256": s["slate_sha256"]})
    manifest.sort(key=lambda x: (x["canonical_id"], x["mode"], int(x["budget_L"])))
    return views, manifest


def policy_kind(policy: str):
    return "pplx" if policy.startswith("pplx") else "jina"


def policy_budget(policy: str):
    if policy == "pplx_budget":
        return 8192, "single"
    if policy == "jina_full":
        return None, "full"
    if "two_stage" in policy:
        return int(policy.rsplit("_", 1)[-1]), "two_stage"
    return int(policy.rsplit("_", 1)[-1]), "single"


def stage_hashes(query, docs):
    return {"query": query, "candidate_ids": [x["candidate_id"] for x in docs],
            "candidate_text_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in docs]}


def score(model_kind, model, query, docs):
    if model_kind == "pplx":
        return base.pplx_score(model, base.choice_request(query, docs), len(docs))
    return base.jina_score(model, query, docs)


def request(view, policy, attempt, gpu, torch, model, proc, tok, decision_messages, pplx_survivors):
    row = base.empty_row(view, policy, attempt, gpu)
    row["policy_contract"] = "v3.6_B6.3_T2"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        kind = policy_kind(policy)
        limit, mode = policy_budget(policy)
        if mode == "full":
            docs = view["original"]
            scores = score(kind, model, view["query"], docs)
            selected = base.stable_top([x["candidate_id"] for x in docs], scores)
            row.update({"candidate_ids": view["original_ids"], "scores": scores, "selected_ids": selected,
                        "n_calls": 1, "input_view": "BRIGHT frozen original candidates",
                        "native_request_sha256": base.sha(stage_hashes(view["query"], docs))})
        elif mode == "single":
            docs = view["single"][str(limit)]
            if docs is None:
                raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
            scores = score(kind, model, view["query"], docs)
            selected = base.stable_top([x["candidate_id"] for x in docs], scores)
            row.update({"candidate_ids": [x["candidate_id"] for x in docs], "scores": scores, "selected_ids": selected,
                        "n_calls": 1, "budget_L": limit, "input_view": f"B1.4 frozen matched text L={limit}",
                        "native_request_sha256": base.sha(stage_hashes(view["query"], docs))})
        else:
            windows = view["windows"][str(limit)]
            if any(part is None for part in windows):
                raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
            by_stage = {x["candidate_id"]: x for part in windows for x in part}
            records, survivors = [], []
            for wi, part in enumerate(windows, 1):
                sc = score(kind, model, view["query"], part)
                ids = [x["candidate_id"] for x in part]
                keep = base.stable_top(ids, sc)
                survivors.extend(keep)
                records.append({"stage": f"stage1_window{wi}", "candidate_ids": ids, "scores": sc,
                                "input_sha256": base.sha(stage_hashes(view["query"], part))})
            original_by_id = {x["candidate_id"]: x for x in view["original"]}
            if kind == "pplx":
                final_original = [original_by_id[x] for x in survivors]
                final_docs, info = fit_docs(view["query"], final_original, proc, decision_messages, tok, limit)
                same = None
            else:
                pids = pplx_survivors.get((view["canonical_id"], limit))
                same = bool(pids is not None and set(pids) == set(survivors) and len(survivors) == 10)
                if same:
                    final_original = [original_by_id[x] for x in pids]
                    final_docs, info = fit_docs(view["query"], final_original, proc, decision_messages, tok, limit)
                else:
                    final_docs = [by_stage[x] for x in survivors]
                    info = {"stage2_text_rule": "survivor_stage1_strings", "pplx_survivor_match": same}
            sc2 = score(kind, model, view["query"], final_docs)
            final_ids = [x["candidate_id"] for x in final_docs]
            records.append({"stage": "stage2_survivors", "candidate_ids": final_ids, "scores": sc2,
                            "budget": info, "input_sha256": base.sha(stage_hashes(view["query"], final_docs))})
            selected = base.stable_top(final_ids, sc2)
            row.update({"candidate_ids": view["original_ids"], "stage_selected_ids": survivors, "scores": sc2,
                        "selected_ids": selected, "stage_records": records, "n_calls": 3, "budget_L": limit,
                        "survivor_set_matches_pplx": same, "input_view": f"B1.5 matched two-stage text L={limit}"})
        if len(row["selected_ids"]) != 5 or len(set(row["selected_ids"])) != 5:
            raise ValueError("INVALID_TOP5_SELECTION")
        row["status"] = "PASS"; row["valid"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"; row["error_class"] = type(exc).__name__
    finally:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        try:
            row.update(base.cuda_stats(torch))
        except Exception:
            row["alloc_peak_mib"] = None
    return row


def run_phase(session, phase, gpu, policies, views, proc, tok, decision_messages, pplx_survivors, resume, gate):
    out = OUT / f"T2_{session}" / "raw" / phase
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "serving_perrequest.checkpoint.jsonl"
    rows = read_jsonl(checkpoint) if resume and checkpoint.exists() else []
    done = {(x["policy"], x["canonical_id"], int(x["attempt"])) for x in rows}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    torch.cuda.set_device(0)
    load_t = time.perf_counter()
    model = None
    if phase == "pplx":
        model, decision_messages_model = base.load_pplx(torch)
        # The frozen builder's formatter is the native formatter used by the model.
        decision_messages = decision_messages_model
    else:
        model, _tok = base.load_jina(torch)
    load_ms = round((time.perf_counter() - load_t) * 1000, 3)
    for policy in policies:
        for attempt in range(1, 4):
            for view in views:
                key = (policy, view["canonical_id"], attempt)
                if key in done:
                    continue
                row = request(view, policy, attempt, gpu, torch, model, proc, tok, decision_messages, pplx_survivors)
                row["gpu_uuid"] = gate.get("gpu_uuid")
                rows.append(row); done.add(key)
                checkpoint.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
            print(json.dumps({"session": session, "phase": phase, "policy": policy, "attempt": attempt,
                              "rows": len(rows), "valid": sum(x.get("valid") is True for x in rows)}), flush=True)
    rows.sort(key=lambda x: (policies.index(x["policy"]), int(x["attempt"]), x["canonical_id"]))
    raw = out / "serving_perrequest_raw.jsonl"
    write_jsonl(raw, rows)
    write_json(out / "TRACE_META.json", {"status": "PASS" if len(rows) == len(policies) * 120 * 3 else "INCOMPLETE",
                                         "records": len(rows), "expected_records": len(policies) * 120 * 3,
                                         "valid": sum(x.get("valid") is True for x in rows), "failed": sum(x.get("valid") is not True for x in rows),
                                         "policies": policies, "sha256": sha_file(raw), "labels_joined": False,
                                         "model_load_ms": load_ms, "gpu_index": gpu})
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    del model, torch
    gc.collect()
    return rows, gate, load_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=sorted(SESSIONS), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.session not in SESSIONS:
        raise RuntimeError("T2_SESSION_INVALID")
    # This must precede processor/native-module imports: autojev imports
    # torch transitively, and the native loaders address the selected device
    # as logical cuda:0 after masking to the requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Dedicated-device gate is taken before processor/native imports can
    # expose this process to nvidia-smi as a compute client.
    gate = base.gpu_gate(args.gpu)
    OUT.mkdir(parents=True, exist_ok=True)
    # Processor/model formatter are CPU-side preparation; no model weights are
    # loaded until after the immutable input manifest and protocol are written.
    proc = v36.base.load_pplx_processor()
    sys.path.insert(0, str(base.PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    tok = proc.tokenizer
    views, manifest_rows = build_views(proc, decision_messages, tok)
    manifest_path = OUT / "T2_INPUT_MANIFEST.jsonl"
    if manifest_path.exists():
        old = sha_file(manifest_path)
        current = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in manifest_rows)
        if hashlib.sha256(current.encode()).hexdigest() != old:
            raise RuntimeError("T2_INPUT_MANIFEST_CHANGED")
    else:
        write_jsonl(manifest_path, manifest_rows)
    protocol_path = OUT / f"T2_{args.session}" / "PROTOCOL.json"
    protocol = {"status": "FROZEN_PRE_OUTCOME", "experiment": "DecisionRAG v3.6 Essential Verification",
                "stage": "B6.3_T2_frontier_timing", "session": args.session, "gpu_index": args.gpu,
                "cohort": "C120 from frozen v3.6 cohort manifest", "cohort_size": len(views), "attempts": 3,
                "concurrency": 1, "policy_order": SESSIONS[args.session],
                "models": {"jina_revision": JINA_REV, "pplx_revision": PPLX_REV},
                "input_hashes": {"bright": BRIGHT_SHA, "stage0_protocol": sha_file(STAGE0 / "PROTOCOL.json"),
                                 "t2_input_manifest": sha_file(manifest_path)},
                "timed_boundary": "view preparation + all native calls + score parsing + ID mapping + Top-5 packaging",
                "labels_joined": False, "protected_holdouts_accessed": False, "paid_api_calls": 0,
                "failure_policy": "no retries; failure elapsed time retained; no fallback in raw timing",
                "reduced_hardware_schedule": "Sessions run sequentially on GPUs 3/6 because only two free dedicated GPUs were available at preflight; no foreign process was evicted.",
                "t2_anchor_policy": ["jina_full", "pplx_budget"], "bootstrap": {"draws": 10000, "seed": 731, "cluster": "canonical_id"}}
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old.get("input_hashes") != protocol["input_hashes"] or old.get("session") != args.session:
            raise RuntimeError("T2_PROTOCOL_INPUT_OR_SESSION_MISMATCH")
        protocol = old
    else:
        write_json(protocol_path, protocol)
    # PPLX phase is first so its stage-1 survivor IDs can be used only for the
    # predeclared D1-compatible Jina stage-2 text rule.
    pplx_rows, _, _ = run_phase(args.session, "pplx", args.gpu, SESSIONS[args.session]["pplx"], views, proc, tok, decision_messages, {}, args.resume, gate)
    helper = {}
    for row in pplx_rows:
        if "two_stage" in row["policy"] and row.get("valid") is True:
            L = int(row["policy"].rsplit("_", 1)[-1])
            helper[(row["canonical_id"], L)] = list(row.get("stage_selected_ids", []))
    jina_rows, _, _ = run_phase(args.session, "jina", args.gpu, SESSIONS[args.session]["jina"], views, proc, tok, decision_messages, helper, args.resume, gate)
    all_rows = pplx_rows + jina_rows
    raw_hashes = {}
    for phase in ("pplx", "jina"):
        p = OUT / f"T2_{args.session}/raw/{phase}/serving_perrequest_raw.jsonl"
        raw_hashes[phase] = {"sha256": sha_file(p), "records": len(read_jsonl(p))}
    write_json(OUT / f"T2_{args.session}/SESSION.json", {"status": "PASS", "session": args.session, "gpu": args.gpu,
                                                           "gpu_uuid": gate.get("gpu_uuid"), "records": len(all_rows),
                                                           "phase_hashes": raw_hashes, "labels_joined": False,
                                                           "protected_holdouts_accessed": False})
    print(json.dumps({"status": "T2_SESSION_DONE", "session": args.session, "records": len(all_rows), "phase_hashes": raw_hashes}, indent=2), flush=True)


if __name__ == "__main__":
    main()
