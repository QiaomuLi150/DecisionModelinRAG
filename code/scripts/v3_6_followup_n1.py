#!/usr/bin/env python3
"""Full-CALL v3.6 follow-up serving timing (N1).

This runner is deliberately label-blind and append-only.  It reuses the
authenticated v3.6 native adapters, but expands the frozen serving workload
from C360 to all 1,384 BRIGHT IDs and records failures rather than retrying.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v1/stage2_n1"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
T2 = ROOT / "scripts/v3_6_t2_serving.py"
QUALITY = ROOT / "results/v3_6_essential/stage1_quality_v2/raw"
STAGE0 = ROOT / "results/v3_6_essential/stage0_preflight"
BRIGHT_SHA = "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
POLICIES = [
    "jina_full", "jina_single_8192", "jina_single_6144", "jina_two_stage_8192",
    "pplx_single_2048", "pplx_single_3072", "pplx_single_4096", "pplx_single_6144",
    "pplx_budget", "pplx_two_stage_4096",
]


def import_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t2 = import_mod("v36_t2_n1", T2)
base = t2.base


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def load_quality_metadata():
    """Load frozen label-free builder metadata from the completed quality run.

    The quality traces already contain the exact final token cap and input
    hashes for every requested budget.  Reusing that metadata avoids repeating
    tens of thousands of CPU tokenizer calls before timing; request-time
    construction still performs the declared dynamic stage-2 fit for PPLX.
    """
    out = {}
    for path in sorted(QUALITY.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["canonical_id"], row["policy"], int(row.get("budget_L") or 0))
            out[key] = row
    return out


_TOKEN_CACHE = {}


def prefix_cached(tok, text, cap):
    key = str(text)
    ids_all = _TOKEN_CACHE.get(key)
    if ids_all is None:
        ids_all = tok.encode(key, add_special_tokens=False)
        _TOKEN_CACHE[key] = ids_all
    ids = ids_all[:int(cap)]
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def docs_from_ids(original, ids, cap, tok):
    byid = {x["candidate_id"]: x for x in original}
    return [{"candidate_id": str(cid), "content": prefix_cached(tok, byid[str(cid)]["content"], cap)} for cid in ids]


def load_all_views(proc, decision_messages, tok):
    slates = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BRIGHT)}
    if len(slates) != 1384 or sha_file(BRIGHT) != BRIGHT_SHA:
        raise RuntimeError("N1_FROZEN_BRIGHT_FAILURE")
    quality = load_quality_metadata()
    stage0 = {r["canonical_id"]: r for r in read_jsonl(STAGE0 / "BUDGET_VIEW_MANIFEST.jsonl")}
    if len(stage0) != 1384:
        raise RuntimeError(f"N1_STAGE0_METADATA_COVERAGE_FAILURE:{len(stage0)}")
    # Tokenize each source passage once in a Rust-tokenizer batch.  The
    # previous implementation called tok.encode separately for every budget,
    # which was correct but needlessly repeated ~200k CPU tokenizations.
    all_texts = []
    for s in slates.values():
        all_texts.extend(str(x["content"]) for x in s["candidates"])
    unique_texts = list(dict.fromkeys(all_texts))
    encoded = tok(unique_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    _TOKEN_CACHE.update(zip(unique_texts, encoded))
    views = []
    manifest = []
    for cid, s in sorted(slates.items()):
        original = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in s["candidates"]]
        single = {}
        for L in (2048, 3072, 4096, 6144, 8192):
            qrow = quality.get((cid, "pplx_single", L)) or quality.get((cid, "jina_single", L))
            if qrow is not None:
                info = qrow.get("budget_info") or {}
            elif L == 8192:
                info = (stage0[cid].get("single") or {}).get("8192", {}).get("budget") or {}
            else:
                raise RuntimeError(f"N1_SINGLE_METADATA_MISSING:{cid}:{L}")
            cap = info.get("final_cap")
            docs = docs_from_ids(original, [x["candidate_id"] for x in original], cap, tok) if cap is not None else None
            single[str(L)] = docs
            manifest.append({"canonical_id": cid, "mode": "single", "budget_L": L,
                             "status": "PASS" if docs is not None else "FAIL",
                             "candidate_ids": [x["candidate_id"] for x in docs] if docs else [],
                             "content_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in docs] if docs else [],
                             "input_sha256": base.sha(base.choice_request(str(s["query"]), docs)) if docs else None,
                             "builder": info, "source_quality_trace": qrow.get("policy") if qrow else "stage0"})
        windows = {}
        for L in (4096, 8192):
            ws = []
            qrow = quality.get((cid, "pplx_two_stage", L)) or quality.get((cid, "jina_two_stage", L))
            if qrow is not None:
                stage_meta = [x for x in qrow.get("stage_records", []) if str(x.get("stage", "")).startswith("stage1_")]
            elif L == 8192:
                stage_meta = [(x, "stage0") for x in ((stage0[cid].get("two_stage_windows") or {}).get("8192") or [])]
            else:
                raise RuntimeError(f"N1_TWO_STAGE_METADATA_MISSING:{cid}:{L}")
            for wi, part in enumerate((original[:10], original[10:]), 1):
                if len(stage_meta) != 2:
                    docs = None
                    info = {"error_class": qrow.get("error_class", "BUILDER_INFEASIBLE") if qrow else "BUILDER_INFEASIBLE",
                            "error": qrow.get("error") if qrow else "stage0 metadata unavailable"}
                else:
                    sm = stage_meta[wi - 1][0] if isinstance(stage_meta[wi - 1], tuple) else stage_meta[wi - 1]
                    cap = ((sm.get("budget") or {}).get("final_cap") if isinstance(sm, dict) and "budget" in sm else sm.get("final_cap"))
                    docs = [{"candidate_id": str(x["candidate_id"]), "content": prefix_cached(tok, x["content"], cap)} for x in part] if cap is not None else None
                    info = (sm.get("budget") or {}) if isinstance(sm, dict) else {}
                ws.append(docs)
                manifest.append({"canonical_id": cid, "mode": f"two_stage_window_{wi}", "budget_L": L,
                                 "status": "PASS" if docs is not None else "FAIL",
                                 "candidate_ids": [x["candidate_id"] for x in docs] if docs else [],
                                 "content_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in docs] if docs else [],
                                 "input_sha256": base.sha(base.choice_request(str(s["query"]), docs)) if docs else None,
                                 "builder": info, "source_quality_trace": qrow.get("policy") if qrow else "stage0"})
            windows[str(L)] = ws
        views.append({"canonical_id": cid, "domain": str(s["domain"]), "query_id": str(s["query_id"]),
                      "query": str(s["query"]), "original": original, "single": single, "windows": windows,
                      "original_ids": [x["candidate_id"] for x in original], "slate_sha256": s["slate_sha256"]})
    manifest.sort(key=lambda x: (x["canonical_id"], x["mode"], int(x["budget_L"])))
    return views, manifest


def fallback_cost(view):
    started = time.perf_counter()
    # The deployed fallback packages the frozen BM25-prefix IDs only.  This
    # operation is measured per request and never enters model input.
    ids = [str(x["candidate_id"]) for x in view["original"][:5]]
    package = json.dumps({"selected_ids": ids, "status": "BM25_FALLBACK"}, sort_keys=True, separators=(",", ":"))
    _ = hashlib.sha256(package.encode()).hexdigest()
    return round((time.perf_counter() - started) * 1000, 6)


def base_row(view, policy, attempt, gpu):
    return {"canonical_id": view["canonical_id"], "domain": view["domain"], "query_id": view["query_id"],
            "policy": policy, "attempt": attempt, "status": "FAIL", "valid": False,
            "error_class": None, "error": None, "latency_ms": None, "latency_with_fallback_ms": None,
            "fallback_packaging_ms": None, "alloc_peak_mib": None, "candidate_ids": [], "selected_ids": [],
            "scores": [], "stage_records": [], "n_calls": 0, "gpu_index": gpu, "gpu_uuid": None,
            "model_revision": PPLX_REV if policy.startswith("pplx") else JINA_REV,
            "labels_joined": False, "input_label_blind": True,
            "timed_boundary": "view preparation + native scoring + score parsing + ID mapping + Top-5 packaging"}


def score(kind, model, query, docs):
    if kind == "jina":
        return base.jina_score(model, query, docs)
    return base.pplx_score(model, base.choice_request(query, docs), len(docs))


def request(view, policy, attempt, gpu, torch, model, proc, tok, decision_messages):
    row = base_row(view, policy, attempt, gpu)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        kind = "jina" if policy.startswith("jina") else "pplx"
        if policy == "jina_full":
            docs = view["original"]
            sc = score(kind, model, view["query"], docs)
            ids = [x["candidate_id"] for x in docs]
            row.update({"candidate_ids": ids, "scores": sc, "selected_ids": base.stable_top(ids, sc), "n_calls": 1,
                        "native_request_sha256": base.sha({"query": view["query"], "candidate_ids": ids,
                                                            "candidate_texts": [x["content"] for x in docs]}),
                        "input_view": "BRIGHT frozen original candidates"})
        elif policy.startswith("jina_single_"):
            L = int(policy.rsplit("_", 1)[-1]); docs = view["single"][str(L)]
            if docs is None: raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
            sc = score(kind, model, view["query"], docs); ids = [x["candidate_id"] for x in docs]
            row.update({"candidate_ids": ids, "scores": sc, "selected_ids": base.stable_top(ids, sc), "n_calls": 1,
                        "budget_L": L, "input_view": f"frozen matched text L={L}",
                        "native_request_sha256": base.sha({"query": view["query"], "candidate_ids": ids,
                                                            "candidate_texts": [x["content"] for x in docs]})})
        elif policy == "jina_two_stage_8192":
            windows = view["windows"]["8192"]
            if any(x is None for x in windows): raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
            records, survivors = [], []
            for wi, docs in enumerate(windows, 1):
                ids = [x["candidate_id"] for x in docs]; sc = score(kind, model, view["query"], docs)
                keep = base.stable_top(ids, sc); survivors.extend(keep)
                records.append({"stage": f"stage1_window{wi}", "candidate_ids": ids, "scores": sc,
                                "input_sha256": base.sha({"query": view["query"], "candidate_ids": ids,
                                                           "candidate_texts": [x["content"] for x in docs]})})
            byid = {x["candidate_id"]: x for part in windows for x in part}; final_docs = [byid[x] for x in survivors]
            ids = [x["candidate_id"] for x in final_docs]; sc = score(kind, model, view["query"], final_docs)
            records.append({"stage": "stage2_survivors", "candidate_ids": ids, "scores": sc,
                            "input_sha256": base.sha({"query": view["query"], "candidate_ids": ids,
                                                       "candidate_texts": [x["content"] for x in final_docs]})})
            row.update({"candidate_ids": view["original_ids"], "stage_selected_ids": survivors, "scores": sc,
                        "selected_ids": base.stable_top(ids, sc), "stage_records": records, "n_calls": 3,
                        "budget_L": 8192, "input_view": "frozen two-stage window text"})
        elif policy.startswith("pplx_single_") or policy == "pplx_budget":
            L = 8192 if policy == "pplx_budget" else int(policy.rsplit("_", 1)[-1])
            docs = view["single"][str(L)]
            if docs is None: raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
            sc = score(kind, model, view["query"], docs); ids = [x["candidate_id"] for x in docs]
            row.update({"candidate_ids": ids, "scores": sc, "selected_ids": base.stable_top(ids, sc), "n_calls": 1,
                        "budget_L": L, "input_view": f"frozen matched text L={L}",
                        "native_request_sha256": base.sha(base.choice_request(view["query"], docs))})
        elif policy == "pplx_two_stage_4096":
            windows = view["windows"]["4096"]
            if any(x is None for x in windows): raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
            records, survivors = [], []
            for wi, docs in enumerate(windows, 1):
                ids = [x["candidate_id"] for x in docs]; sc = score(kind, model, view["query"], docs)
                keep = base.stable_top(ids, sc); survivors.extend(keep)
                records.append({"stage": f"stage1_window{wi}", "candidate_ids": ids, "scores": sc,
                                "input_sha256": base.sha(base.choice_request(view["query"], docs))})
            byid = {x["candidate_id"]: x for x in view["original"]}; original_survivors = [byid[x] for x in survivors]
            docs2, info = t2.fit_docs(view["query"], original_survivors, proc, decision_messages, tok, 4096)
            if docs2 is None: raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
            ids = [x["candidate_id"] for x in docs2]; req = base.choice_request(view["query"], docs2); sc = base.pplx_score(model, req, len(docs2))
            records.append({"stage": "stage2_survivors", "candidate_ids": ids, "scores": sc, "budget": info,
                            "input_sha256": base.sha(req)})
            row.update({"candidate_ids": view["original_ids"], "stage_selected_ids": survivors, "scores": sc,
                        "selected_ids": base.stable_top(ids, sc), "stage_records": records, "n_calls": 3,
                        "budget_L": 4096, "input_view": "frozen two-stage text L=4096"})
        else:
            raise ValueError(f"UNKNOWN_POLICY:{policy}")
        if len(row["selected_ids"]) != 5 or len(set(row["selected_ids"])) != 5:
            raise ValueError("INVALID_TOP5_SELECTION")
        row["status"] = "PASS"; row["valid"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"; row["error_class"] = type(exc).__name__
    finally:
        try: torch.cuda.synchronize()
        except Exception: pass
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        row["fallback_packaging_ms"] = fallback_cost(view)
        row["latency_with_fallback_ms"] = round(row["latency_ms"] + (row["fallback_packaging_ms"] if not row["valid"] else 0), 3)
        try: row.update(base.cuda_stats(torch))
        except Exception: row["alloc_peak_mib"] = None
    return row


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gate = base.gpu_gate(args.gpu)
    import torch
    torch.cuda.set_device(0)
    proc = t2.v36.base.load_pplx_processor(); tok = proc.tokenizer
    sys.path.insert(0, str(base.PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    views, manifest = load_all_views(proc, decision_messages, tok)
    manifest_path = OUT / "N1_INPUT_MANIFEST.jsonl"
    serialized = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in manifest)
    if manifest_path.exists():
        if sha_file(manifest_path) != hashlib.sha256(serialized.encode()).hexdigest(): raise RuntimeError("N1_MANIFEST_CHANGED")
    else: manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(serialized, encoding="utf-8")
    protocol = {"status": "FROZEN_PRE_OUTCOME", "stage": "N1_full_CALL_timing", "session": args.session,
                "gpu_slot": "A" if args.session == "A" else "B", "cohort": "CALL", "cohort_size": len(views), "attempts": 2,
                "concurrency": 1, "policy_order": POLICIES, "models": {"jina_revision": JINA_REV, "pplx_revision": PPLX_REV},
                "input_hashes": {"bright": BRIGHT_SHA, "n1_manifest": sha_file(manifest_path)},
                "timed_boundary": "view preparation + all native calls + score parsing + ID mapping + Top-5 packaging",
                "failure_policy": "no retries; failed elapsed time retained; all-request basis adds measured fallback packaging",
                "labels_joined": False, "protected_holdouts_accessed": False, "bootstrap": {"draws": 10000, "seed": 731, "cluster": "canonical_id"}}
    pp = OUT / f"PROTOCOL_N1_{args.session}.json"
    if pp.exists(): protocol = json.loads(pp.read_text())
    else: write_json(pp, protocol)
    outdir = OUT / f"session_{args.session}"; rawdir = outdir / "raw"; rawdir.mkdir(parents=True, exist_ok=True)
    checkpoint = rawdir / "serving_perrequest.checkpoint.jsonl"
    rows = read_jsonl(checkpoint) if args.resume and checkpoint.exists() else []
    done = {(r["policy"], r["canonical_id"], int(r["attempt"])) for r in rows}
    checkpoint_handle = checkpoint.open("a", encoding="utf-8")
    load_times = {}
    for phase, policies in (("jina", POLICIES[:4]), ("pplx", POLICIES[4:])):
        t0 = time.perf_counter()
        if phase == "jina": model, _ = base.load_jina(torch)
        else: model, decision_messages = base.load_pplx(torch)
        torch.cuda.synchronize(); load_times[f"{phase}_load_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        for policy in policies:
            for attempt in (1, 2):
                for view in views:
                    key = (policy, view["canonical_id"], attempt)
                    if key in done: continue
                    row = request(view, policy, attempt, args.gpu, torch, model, proc, tok, decision_messages if phase == "pplx" else None)
                    row["gpu_uuid"] = gate.get("gpu_uuid"); row["gpu_slot"] = "A" if args.session == "A" else "B"; row["session"] = args.session
                    rows.append(row); done.add(key)
                    checkpoint_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    checkpoint_handle.flush()
                print(json.dumps({"session": args.session, "policy": policy, "attempt": attempt, "rows": len(rows), "valid": sum(x.get("valid") is True for x in rows)}), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    checkpoint_handle.close()
    rows.sort(key=lambda x: (POLICIES.index(x["policy"]), int(x["attempt"]), x["canonical_id"]))
    raw = rawdir / "serving_perrequest_raw.jsonl"; write_jsonl(raw, rows)
    meta = {"status": "PASS" if len(rows) == len(POLICIES) * len(views) * 2 else "INCOMPLETE", "session": args.session,
            "records": len(rows), "expected_records": len(POLICIES) * len(views) * 2,
            "valid": sum(x.get("valid") is True for x in rows), "failed": sum(x.get("valid") is not True for x in rows),
            "sha256": sha_file(raw), "gpu_slot": "A" if args.session == "A" else "B", "gpu_uuid": gate.get("gpu_uuid"),
            "load_times": load_times, "labels_joined": False}
    write_json(outdir / "TRACE_META.json", meta); write_json(outdir / "GPU_GATE.json", gate)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--session", choices=["A", "B"], required=True); ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--resume", action="store_true")
    run(ap.parse_args())
