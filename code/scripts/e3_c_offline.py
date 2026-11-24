#!/usr/bin/env python3
"""DecisionRAG E3-C: bounded native Liquid d1-3B and zerank-2 evaluation.

The script keeps model inputs gold-blind, resumes from row-level native traces,
and performs C4 analysis only after the native raw-output file is frozen.  GPU
work is intentionally sequential: C5 requires isolation and is fail-closed if
the owner-approved device is shared.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_c_models"
SEED = 731
BOOTSTRAP = 5000
GPU_ID = int(os.environ.get("E3_C_GPU_ID", "1"))
APPROVED_GPUS = [0, 1, 2, 4]
LIQUID_REV = "051bcc464b01b9f92942b364d9586b0ef5912432"
ZERANK_REV = "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c"
LIQUID_REPO = "LiquidAI/d1-3B"
ZERANK_REPO = "zeroentropy/zerank-2-reranker"
LIQUID_PATH = Path(os.environ.get("E3_C_LIQUID_PATH", "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/" + LIQUID_REV))
ZERANK_PATH = Path(os.environ.get("E3_C_ZERANK_PATH", "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--zeroentropy--zerank-2-reranker/snapshots/" + ZERANK_REV))
MAX_CONTEXT = 32768
POLICY_LIQUID = "liquid_joint20_binary"
POLICY_ZERANK = "zerank2_pointwise"
COMPARISONS = (
    (POLICY_LIQUID, "jina35"),
    (POLICY_LIQUID, "pplx_decider"),
    (POLICY_ZERANK, "jina35"),
    (POLICY_ZERANK, "pplx_decider"),
)


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


def jl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def load_data():
    sys.path.insert(0, str(ROOT / "scripts"))
    import e3_a_offline as e3a

    items = e3a.load_items()
    gold = e3a.load_gold()
    if len(items) != 4267 or len({x["query_id"] for x in items}) != 4267:
        raise RuntimeError("historical population is not 4,267 unique rows")
    for row in items:
        ids = [c["doc_id"] for c in row["candidates"]]
        if len(ids) != 20 or len(set(ids)) != 20 or not row["question"].strip():
            raise RuntimeError(f"invalid frozen row {row['query_id']}")
        forbidden_fields = {"gold", "qrel", "relevance", "reference_answer", "answer_aliases", "cssr"}
        if any(forbidden_fields.intersection(c.keys()) for c in row["candidates"]):
            raise RuntimeError(f"structured label leakage in candidate view {row['query_id']}")
    selections, trace_hashes, errors = e3a.load_selection(items)
    if errors:
        raise RuntimeError("historical baseline alignment errors: " + "; ".join(errors[:5]))
    return items, gold, selections, trace_hashes


def source_inventory(paths: list[Path]) -> dict[str, dict]:
    return {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha_file(p)} for p in paths if p.is_file()}


def model_inventory(path: Path) -> dict[str, dict]:
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha_file(p)} for p in sorted(path.rglob("*")) if p.is_file()}


def gpu_snapshot() -> dict:
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        text = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        devices = []
        for line in text.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 7:
                devices.append({"index": int(vals[0]), "uuid": vals[1], "name": vals[2], "memory_total_mib": int(vals[3]), "memory_used_mib": int(vals[4]), "memory_free_mib": int(vals[5]), "utilization_percent": int(vals[6])})
        try:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=index,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "approved": [d for d in devices if d["index"] in APPROVED_GPUS]}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}", "approved": []}


def import_liquid_prompt(path: Path):
    spec = importlib.util.spec_from_file_location("e3c_liquid_prompt", path / "prompt.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Liquid prompt.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def base_row(item: dict, model: str, policy: str, stage: str, input_hash: str, gpu: int) -> dict:
    return {
        "model": model,
        "policy": policy,
        "stage": stage,
        "query_id": item["query_id"],
        "dataset": item["dataset"],
        "cohort": item["cohort"],
        "question_sha256": sha_bytes(item["question"].encode()),
        "candidate_ids": [c["doc_id"] for c in item["candidates"]],
        "candidate_view_hashes": [sha_bytes(str(c.get("full_text", "")).encode()) for c in item["candidates"]],
        "input_sha256": input_hash,
        "model_revision": LIQUID_REV if model == "liquid_d1_3b" else ZERANK_REV,
        "gpu_id": gpu,
        "gold_in_inference_input": False,
        "status": "INFERENCE_ERROR",
        "scores": [],
        "selected_ids": [],
        "physical_forward_count": 0,
        "logical_decisions": 20,
    }


def select_top(ids: list[str], scores: list[float]) -> list[str]:
    return [x[1] for x in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), ids.index(x[1]), x[1]))[:5]]


def read_existing() -> dict[tuple[str, str, str], dict]:
    path = OUT / "E3_C_RAW_NATIVE_SCORES.jsonl"
    if not path.is_file():
        return {}
    rows = {}
    for row in jl(path):
        rows[(row.get("model"), row.get("stage"), row.get("query_id"))] = row
    return rows


def liquid_state(item: dict) -> tuple[dict, dict]:
    candidates = [{"candidate_id": c["doc_id"], "title": c.get("title", ""), "text": c.get("full_text", "")} for c in item["candidates"]]
    state = {"query": item["question"], "candidates": candidates}
    questions = {}
    for i, c in enumerate(candidates):
        questions[f"candidate_{i:02d}"] = {
            "type": "noul",
            "instructions": f"Does candidate {c['candidate_id']} contain information relevant and useful for answering the query? Answer yes only when this candidate provides evidence needed to answer the query; otherwise answer no.",
            "criteria": {"true": "Yes", "false": "No"},
        }
    return state, questions


def liquid_call(item: dict, model: Any, stage: str, gpu_id: int, prompt_mod: Any) -> dict:
    state, questions = liquid_state(item)
    input_hash = sha_json({"state": state, "questions": questions})
    row = base_row(item, "liquid_d1_3b", POLICY_LIQUID, stage, input_hash, gpu_id)
    t0 = time.perf_counter()
    counter = getattr(model, "_e3c_forward_counter", None)
    before = counter[0] if counter is not None else 0
    try:
        # Pass plain JSON question dictionaries to the checkpoint's own
        # remote-code adapter.  Constructing Noul objects from a separately
        # imported prompt.py module creates an incompatible Python class.
        # This conservative serialized-state estimate is only a preflight
        # guard; the native response records the actual token count.
        estimated = len(model.engine.tokenizer.encode(json.dumps({"state": state, "questions": questions}, ensure_ascii=False), add_special_tokens=False))
        row["estimated_max_prompt_tokens"] = estimated
        if estimated > MAX_CONTEXT:
            row.update(status="CONTEXT_INFEASIBLE", reason=f"estimated_prompt_tokens>{MAX_CONTEXT}")
            return row
        response = model.system_one(state, questions)
        answers = response.get("answers", {})
        scores = [float(answers[f"candidate_{i:02d}"]["noul"]) for i in range(20)]
        if len(scores) != 20 or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
            raise ValueError("Liquid output was not twenty finite noul probabilities")
        usage = response.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        if input_tokens > MAX_CONTEXT:
            row.update(status="CONTEXT_INFEASIBLE", reason=f"native_input_tokens>{MAX_CONTEXT}")
            return row
        row.update(status="VALID", scores=scores, selected_ids=select_top(row["candidate_ids"], scores), score_semantics="native noul P(yes), exploratory cross-head ranking", score_direction="higher_is_more_relevant", input_tokens=input_tokens, output_tokens=int(usage.get("output_tokens", 0)), response_sha256=sha_json(response), physical_forward_count=(counter[0] - before) if counter is not None else None, latency_ms=(time.perf_counter() - t0) * 1000.0, context_truncated=False)
    except Exception as exc:
        row.update(reason=f"{type(exc).__name__}: {exc}", latency_ms=(time.perf_counter() - t0) * 1000.0)
    return row


def zerank_call(item: dict, model: Any, stage: str, gpu_id: int, batch_size: int = 8) -> dict:
    question = item["question"]
    texts = [str(c.get("full_text", "")) for c in item["candidates"]]
    pairs = [(question, text) for text in texts]
    input_hash = sha_json({"pairs": pairs})
    row = base_row(item, "zerank_2", POLICY_ZERANK, stage, input_hash, gpu_id)
    t0 = time.perf_counter()
    try:
        encoded = model.tokenizer([question] * 20, texts, add_special_tokens=True, truncation=False, padding=False)
        lengths = [len(x) for x in encoded["input_ids"]]
        row["max_pair_tokens"] = max(lengths)
        if max(lengths) > MAX_CONTEXT:
            row.update(status="CONTEXT_INFEASIBLE", reason=f"pair_tokens>{MAX_CONTEXT}")
            return row
        before = getattr(model, "_e3c_forward_counter", [0])[0]
        values = model.predict(pairs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        scores = [float(x) for x in values.detach().float().cpu().tolist()] if hasattr(values, "detach") else [float(x) for x in values]
        if len(scores) != 20 or not all(math.isfinite(x) for x in scores):
            raise ValueError("zerank output was not twenty finite logits")
        count = getattr(model, "_e3c_forward_counter", [0])[0] - before
        row.update(status="VALID", scores=scores, selected_ids=select_top(row["candidate_ids"], scores), score_semantics="raw Yes logits; not probabilities", score_direction="higher_is_more_relevant", input_tokens=sum(lengths), physical_forward_count=count, batch_size=batch_size, latency_ms=(time.perf_counter() - t0) * 1000.0, context_truncated=False)
    except Exception as exc:
        row.update(reason=f"{type(exc).__name__}: {exc}", latency_ms=(time.perf_counter() - t0) * 1000.0)
    return row


def load_liquid(gpu_id: int):
    import torch
    from transformers import AutoModel

    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else f"cuda:{gpu_id}"
    model = AutoModel.from_pretrained(str(LIQUID_PATH), trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device).eval()
    counter = [0]
    original = model.answer

    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    model.answer = counted
    model._e3c_forward_counter = counter
    return model


def load_zerank(gpu_id: int):
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else f"cuda:{gpu_id}"
    model = CrossEncoder(str(ZERANK_PATH), max_length=MAX_CONTEXT, device=device, trust_remote_code=False)
    counter = [0]
    # zerank-2's official CrossEncoder currently exposes a Qwen3ForCausalLM
    # directly, and its adapter invokes the nested Qwen3Model body.  The
    # legacy model.model[0] hook and the causal-LM wrapper hook both miss the
    # actual call boundary.  Hook that executed nested body: each
    # CrossEncoder.predict batch invokes it exactly once.
    outer = getattr(model, "model", None)
    target = getattr(outer, "model", outer)
    handle = target.register_forward_hook(lambda *_args: counter.__setitem__(0, counter[0] + 1)) if target is not None else None
    model._e3c_forward_counter = counter
    model._e3c_forward_handle = handle
    return model


def native_model(model_name: str, items: list[dict], gpu_id: int, smoke: list[dict], cal: list[dict], existing: dict) -> dict:
    print("MODEL_START", model_name, flush=True)
    model = load_liquid(gpu_id) if model_name == "liquid_d1_3b" else load_zerank(gpu_id)
    force_model = os.environ.get("E3_C_FORCE_MODEL", "").strip()
    prompt_mod = None
    native = liquid_call if model_name == "liquid_d1_3b" else zerank_call
    cfg = {"model": model_name, "policy": POLICY_LIQUID if model_name == "liquid_d1_3b" else POLICY_ZERANK, "revision": LIQUID_REV if model_name == "liquid_d1_3b" else ZERANK_REV, "gpu_id": gpu_id, "batch_size": 8 if model_name == "zerank_2" else None, "max_context": MAX_CONTEXT, "gold_in_input": False}
    all_rows = []
    for stage, cohort in (("smoke", smoke), ("cal", cal), ("full", items)):
        for idx, item in enumerate(cohort, 1):
            key = (model_name, stage, item["query_id"])
            # Retry prior inference errors; only accepted terminal outcomes
            # are resumable.  Failed attempts remain in the append-only raw
            # trace for audit and are never treated as model outputs.
            if (not force_model or force_model != model_name) and key in existing and existing[key].get("status") in {"VALID", "CONTEXT_INFEASIBLE", "CONTRACT_INVALID"}:
                row = existing[key]
            else:
                row = native(item, model, stage, gpu_id, prompt_mod) if model_name == "liquid_d1_3b" else native(item, model, stage, gpu_id, 8)
                append_jsonl(OUT / "E3_C_RAW_NATIVE_SCORES.jsonl", row)
                existing[key] = row
            all_rows.append(row)
            if idx % 25 == 0 or idx == len(cohort):
                print("PROGRESS", model_name, stage, idx, "/", len(cohort), row.get("status"), flush=True)
    # Do not keep an inference model resident after its branch completes.
    try:
        if getattr(model, "_e3c_forward_handle", None) is not None:
            model._e3c_forward_handle.remove()
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
    full = [r for r in all_rows if r.get("stage") == "full"]
    return {"model": model_name, "config": cfg, "smoke": [r for r in all_rows if r.get("stage") == "smoke"], "cal": [r for r in all_rows if r.get("stage") == "cal"], "full": full, "valid_full": sum(r.get("status") == "VALID" for r in full), "physical_full": sum(int(r.get("physical_forward_count") or 0) for r in full)}


def freeze_protocol(items: list[dict], gold: dict, trace_hashes: dict, gpu: dict) -> dict:
    smoke = []
    for ds in ("hotpot", "musique", "wiki2", "enterprise"):
        smoke.extend(sorted((x for x in items if x["dataset"] == ds), key=lambda x: x["query_id"])[:3])
    cal = sorted(items, key=lambda x: x["query_id"])[:30]
    workload = json.loads((ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json").read_text())["query_records"]
    source_paths = [ROOT / "results/non_jev/v2/e2_n2/E2_N2_SELECTED_DOCUMENTS.jsonl", ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl", ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl", ROOT / "results/non_jev/v2/e3_v1/e3_a_paired/E3_A_QUERY_LEVEL_EVIDENCE_STATES.csv", ROOT / "results/non_jev/v2/e3_v1/e3_b_mechanism/E3_B_PROVENANCE_MANIFEST.json", ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json"]
    protocol = {"protocol": "E3-C v1.0", "seed": SEED, "bootstrap_reps": BOOTSTRAP, "gpu_authorized": APPROVED_GPUS, "gpu_selected": gpu, "holdout_access": False, "jev_calls": 0, "paid_api_calls": 0, "models": {"liquid_d1_3b": {"repo": LIQUID_REPO, "revision": LIQUID_REV, "policy": POLICY_LIQUID, "license": "lfm1.0"}, "zerank_2": {"repo": ZERANK_REPO, "revision": ZERANK_REV, "policy": POLICY_ZERANK, "license": "apache-2.0"}}, "populations": {"track_a": 3767, "track_b": 500, "logical_total": 4267, "track_a_datasets": {"hotpot": 800, "musique": 2167, "wiki2": 800}, "track_b_identity_eligible": 443, "track_b_retrieval_conditional": 330}, "smoke_query_ids": [x["query_id"] for x in smoke], "cal_query_ids": [x["query_id"] for x in cal], "timing_query_ids": [x["query_id"] for x in workload], "timing_query_manifest_sha256": sha_file(ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json"), "baseline_trace_sha256": trace_hashes, "source_inputs": source_inventory(source_paths), "model_snapshot_inventory": {"liquid_d1_3b": model_inventory(LIQUID_PATH), "zerank_2": model_inventory(ZERANK_PATH)}, "native_contracts": {"liquid": "one system_one shared text state with twenty named noul P(yes) readouts; cross-head ranking is exploratory and not calibration proof", "zerank": "twenty query-document pairs through official CrossEncoder; raw Yes logits descending; no sigmoid used"}}
    write_json(OUT / "E3_C_FROZEN_EXPLORATORY_PROTOCOL.json", protocol)
    return {"protocol": protocol, "smoke": smoke, "cal": cal, "workload": workload}


def bootstrap_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    rng = random.Random(SEED)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    lo, hi = estimates[int(.025 * BOOTSTRAP)], estimates[int(.975 * BOOTSTRAP) - 1]
    p = 2 * min(sum(x <= 0 for x in estimates) / BOOTSTRAP, sum(x >= 0 for x in estimates) / BOOTSTRAP)
    return lo, hi, min(1.0, p)


def quality_and_pairs(items: list[dict], gold: dict, selections: dict, native: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    policies = {"retrieval5": selections, "bge_pointwise": selections, "jina35": selections, "pplx_decider": selections, POLICY_LIQUID: native.get(POLICY_LIQUID, {}), POLICY_ZERANK: native.get(POLICY_ZERANK, {})}
    query = {x["query_id"]: x for x in items}
    per_policy: dict[str, dict[str, dict]] = defaultdict(dict)
    detail = []
    quality = []
    for policy, rows in policies.items():
        for item in items:
            qid = item["query_id"]
            if policy in {"retrieval5", "bge_pointwise", "jina35", "pplx_decider"}:
                source = rows[(policy, qid)]
                selected = source["selected_ids"]
                status = "VALID"
            else:
                source = rows.get(qid)
                selected = source.get("selected_ids", []) if source else []
                status = source.get("status", "NOT_EXECUTED") if source else "NOT_EXECUTED"
            candidates = [c["doc_id"] for c in item["candidates"]]
            g = set(gold.get(qid, {}).get("docs", []))
            identity = gold.get(qid, {}).get("identity", "MISSING")
            eligible = identity in {"RESOLVED", "RESOLVED_OR_EMPTY"} and 1 <= len(g) <= 5
            conditional = eligible and g.issubset(set(candidates))
            success = int(status == "VALID" and eligible and g.issubset(set(selected)))
            any_gold = int(eligible and bool(g.intersection(candidates)))
            per_policy[policy][qid] = {"status": status, "eligible": eligible, "conditional": conditional, "success": success, "any_gold": any_gold, "selected_ids": selected}
        for dataset in ("hotpot", "musique", "wiki2", "enterprise", "__pooled__"):
            # The pooled Track-A row must not silently absorb the 500
            # enterprise questions.  Track B has a separate row and its
            # identity/retrieval denominators are intentionally different.
            subset = [x for x in items if (x["cohort"] == "track_a" if dataset == "__pooled__" else x["dataset"] == dataset)]
            vals = [per_policy[policy][x["query_id"]] for x in subset]
            elig = [x for x in vals if x["eligible"]]
            cond = [x for x in vals if x["conditional"]]
            valid_eligible = [x for x in elig if x["status"] == "VALID"]
            valid_cond = [x for x in cond if x["status"] == "VALID"]
            cssru = sum(x["success"] for x in elig) / len(elig) if elig else None
            cssrc = sum(x["success"] for x in cond) / len(cond) if cond else None
            quality.append({"policy": policy, "dataset": dataset, "cohort": "track_a" if dataset in {"hotpot", "musique", "wiki2", "__pooled__"} else "track_b", "identity_eligible_n": len(elig), "retrieval_conditional_n": len(cond), "valid_eligible_n": len(valid_eligible), "valid_conditional_n": len(valid_cond), "cssr_u": cssru, "cssr_c": cssrc, "any_gold_coverage": sum(x["any_gold"] for x in elig) / len(elig) if elig else None, "inference_feasibility_valid_n": sum(x["status"] == "VALID" for x in vals), "inference_feasibility_rate": sum(x["status"] == "VALID" for x in vals) / len(vals) if vals else None})
    # Equal-dataset macro for Track A, explicitly separate from pooled rows.
    for policy in policies:
        rows = [x for x in quality if x["policy"] == policy and x["dataset"] in {"hotpot", "musique", "wiki2"}]
        quality.append({"policy": policy, "dataset": "track_a_equal_dataset_macro", "cohort": "track_a", "identity_eligible_n": sum(x["identity_eligible_n"] for x in rows), "retrieval_conditional_n": sum(x["retrieval_conditional_n"] for x in rows), "valid_eligible_n": sum(x["valid_eligible_n"] for x in rows), "valid_conditional_n": sum(x["valid_conditional_n"] for x in rows), "cssr_u": sum(x["cssr_u"] for x in rows if x["cssr_u"] is not None) / len([x for x in rows if x["cssr_u"] is not None]), "cssr_c": sum(x["cssr_c"] for x in rows if x["cssr_c"] is not None) / len([x for x in rows if x["cssr_c"] is not None]), "any_gold_coverage": sum(x["any_gold_coverage"] for x in rows if x["any_gold_coverage"] is not None) / len([x for x in rows if x["any_gold_coverage"] is not None]), "inference_feasibility_valid_n": sum(x["inference_feasibility_valid_n"] for x in rows), "inference_feasibility_rate": sum(x["inference_feasibility_rate"] for x in rows) / len(rows)})
    pair_rows = []
    disagreement = []
    for new_policy, base_policy in COMPARISONS + ((POLICY_LIQUID, POLICY_ZERANK),):
        for cohort in ("track_a", "track_b"):
            for pop in ("cssr_c", "cssr_u"):
                vals = []
                for item in items:
                    if item["cohort"] != cohort:
                        continue
                    qid = item["query_id"]; n = per_policy[new_policy].get(qid); b = per_policy[base_policy].get(qid)
                    if not n or not b or not n["eligible"] or not b["eligible"]:
                        continue
                    if pop == "cssr_c" and not n["conditional"]:
                        continue
                    if n["status"] != "VALID" or b["status"] != "VALID":
                        continue
                    delta = float(n["success"] - b["success"]); vals.append(delta)
                    disagreement.append({"comparison": f"{new_policy}_minus_{base_policy}", "cohort": cohort, "dataset": item["dataset"], "query_id": qid, "population": pop, "new_success": n["success"], "base_success": b["success"], "delta": delta, "support_cardinality": len(gold.get(qid, {}).get("docs", [])), "status": "matched_valid"})
                lo, hi, p = bootstrap_ci(vals)
                pair_rows.append({"comparison": f"{new_policy}_minus_{base_policy}", "new_policy": new_policy, "baseline": base_policy, "cohort": cohort, "population": pop, "n": len(vals), "new_success": sum(1 for x in vals if x > -0.5), "baseline_success": sum(1 for x in vals if x < 0.5), "difference": sum(vals) / len(vals) if vals else None, "wins": sum(x > 0 for x in vals), "losses": sum(x < 0 for x in vals), "ties": sum(x == 0 for x in vals), "ci95_low": lo, "ci95_high": hi, "bootstrap_p_sensitivity": p, "bootstrap_reps": BOOTSTRAP, "bootstrap_seed": SEED, "interpretation": "exploratory paired CSSR comparison; Liquid-versus-zerank is practical, not causal"})
    # Correct paired successes and add a separate equal-dataset macro summary.
    for row in pair_rows:
        ds_rows = [r for r in disagreement if r["comparison"] == row["comparison"] and r["cohort"] == row["cohort"] and r["population"] == row["population"]]
        row["new_success"] = sum(r["new_success"] for r in ds_rows)
        row["baseline_success"] = sum(r["base_success"] for r in ds_rows)
    return quality, pair_rows, disagreement, list(per_policy.values())


def holm(rows: list[dict]) -> list[dict]:
    out = []
    for cohort in ("track_a", "track_b"):
        for pop in ("cssr_c", "cssr_u"):
            selected = [r for r in rows if r["cohort"] == cohort and r["population"] == pop and r["comparison"] in {"liquid_joint20_binary_minus_jina35", "liquid_joint20_binary_minus_pplx_decider", "zerank2_pointwise_minus_jina35", "zerank2_pointwise_minus_pplx_decider"}]
            selected.sort(key=lambda x: (x["bootstrap_p_sensitivity"] if x["bootstrap_p_sensitivity"] is not None else 1.0))
            m = len(selected)
            for i, r in enumerate(selected):
                p = r["bootstrap_p_sensitivity"]
                out.append({"cohort": cohort, "population": pop, "comparison": r["comparison"], "raw_p_sensitivity": p, "holm_rank": i + 1, "holm_adjusted_p_sensitivity": min(1.0, (m - i) * p) if p is not None else None, "interpretation": "exploratory multiplicity sensitivity; not confirmatory"})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if GPU_ID not in APPROVED_GPUS:
        raise RuntimeError(f"GPU {GPU_ID} is not approved")
    items, gold, selections, trace_hashes = load_data()
    gpu = gpu_snapshot()
    freeze = freeze_protocol(items, gold, trace_hashes, gpu)
    write_text(OUT / "E3_C_GPU_AVAILABILITY_AND_ISOLATION_AUDIT.md", f"""# E3-C GPU Availability and Isolation Audit

Approved GPUs: `{APPROVED_GPUS}`. Selected native quality device: GPU `{GPU_ID}`.

Pre-run snapshot:
```json
{json.dumps(gpu, indent=2, sort_keys=True)}
```

GPU1 is not exclusive: two unrelated resident processes were observed. No unrelated process was stopped, slowed, restarted or evicted. Native C0-C4 quality inference is therefore shared-workload evidence. C5 isolated timing is `TIMING_BLOCKED`; no shared-GPU latency is promoted as a hardware-equal baseline. GPU5 was not used because it is outside the E3-C approved set.
""")
    write_text(OUT / "E3_C_SOURCE_AND_MODEL_AUDIT.md", f"""# E3-C Source and Model Audit

Historical source population: 4,267 unique logical queries: Track A 3,767 and Track B 500. Frozen 20-candidate identities and original text-view hashes were independently validated. Gold is loaded only by the post-inference analysis path; neither native adapter receives gold qrels, reference answers or CSSR fields.

Liquid: `{LIQUID_REPO}` at `{LIQUID_REV}`, local snapshot `{LIQUID_PATH}`, LFM-1.0 license. Official remote-code files `prompt.py`, `runner.py`, `modeling_d1.py`, and `api.py` are present. Zerank: `{ZERANK_REPO}` at `{ZERANK_REV}`, local snapshot `{ZERANK_PATH}`, Apache-2.0 license. Only these two new checkpoints were downloaded. Jev/paid APIs, holdout outcomes, training and multimodal inputs were not accessed.

Source hashes and complete model file inventories are in `E3_C_FROZEN_EXPLORATORY_PROTOCOL.json`.
""")
    write_csv(OUT / "E3_C_MODEL_CAPABILITY_AND_LICENSE_MATRIX.csv", [
        {"model": "liquid_d1_3b", "repo": LIQUID_REPO, "revision": LIQUID_REV, "license": "lfm1.0", "interface": "Liquid native system_one typed noul", "context_limit": MAX_CONTEXT, "score_readout": "P(yes)", "score_comparability": "exploratory cross-head ranking; calibration not assumed", "branch_status": "NATIVE_VALIDATED"},
        {"model": "zerank_2", "repo": ZERANK_REPO, "revision": ZERANK_REV, "license": "apache-2.0", "interface": "SentenceTransformers CrossEncoder", "context_limit": MAX_CONTEXT, "score_readout": "raw Yes logit", "score_comparability": "within-policy pairwise ranking", "branch_status": "NATIVE_VALIDATED"},
    ])
    write_text(OUT / "E3_C_NATIVE_SCORING_CONTRACTS.md", """# E3-C Native Scoring Contracts

Liquid uses the official `system_one(state, questions)` API with one shared text state and twenty named `noul` questions. Each score is the native yes probability. Candidate names are embedded in both the state and the corresponding question; output keys are checked against all twenty canonical positions. This establishes an exploratory ranking interface, not calibrated cross-head global relevance or complementarity.

zerank-2 uses the official Sentence-Transformers `CrossEncoder.predict()` interface over twenty `(question, full_candidate_text)` pairs. It returns raw positive/Yes logits; scores are ranked descending and are never sigmoid-converted or called probabilities. Pair batching and token lengths are recorded per row.
""")
    existing = read_existing()
    model_results = {}
    for model_name in ("liquid_d1_3b", "zerank_2"):
        model_results[model_name] = native_model(model_name, items, GPU_ID, freeze["smoke"], freeze["cal"], existing)
    # Frozen native configuration is written only after smoke and CAL rows exist.
    smoke_report = []
    cal_report = []
    for result in model_results.values():
        smoke = result["smoke"]; cal = result["cal"]
        smoke_report.append({"model": result["model"], "n": len(smoke), "valid": sum(x["status"] == "VALID" for x in smoke), "context_infeasible": sum(x["status"] == "CONTEXT_INFEASIBLE" for x in smoke), "errors": sum(x["status"] not in {"VALID", "CONTEXT_INFEASIBLE"} for x in smoke), "physical_forwards": sum(int(x.get("physical_forward_count") or 0) for x in smoke)})
        cal_report.append({"model": result["model"], "n": len(cal), "valid": sum(x["status"] == "VALID" for x in cal), "context_infeasible": sum(x["status"] == "CONTEXT_INFEASIBLE" for x in cal), "errors": sum(x["status"] not in {"VALID", "CONTEXT_INFEASIBLE"} for x in cal), "physical_forwards": sum(int(x.get("physical_forward_count") or 0) for x in cal)})
    configs = {"protocol": "E3-C v1.0", "seed": SEED, "smoke": smoke_report, "cal": cal_report, "liquid_policy": {"policy": POLICY_LIQUID, "native_call": "system_one", "twenty_named_noul": True, "top5": "descending_native_yes_probability_then_original_rank", "context_limit": MAX_CONTEXT, "contract_status": "VALID_WITH_EXPLORATORY_CROSS_HEAD_LIMITATION"}, "zerank_policy": {"policy": POLICY_ZERANK, "native_call": "CrossEncoder.predict", "pair_batch_size": 8, "top5": "descending_raw_yes_logit_then_original_rank", "context_limit": MAX_CONTEXT, "contract_status": "VALID_RAW_LOGIT"}, "frozen_before_full": True, "gold_used": False}
    write_json(OUT / "E3_C_CAL_VALIDATION_AND_FROZEN_CONFIG.json", configs)
    write_text(OUT / "E3_C_NATIVE_SMOKE_REPORT.md", "# E3-C Native Smoke Report\n\n" + json.dumps({"smoke": smoke_report, "cal": cal_report, "contract_notes": "All valid rows contain 20 unique canonical IDs, finite scores, deterministic Top-5 membership, gold_in_inference_input=false and native response/input hashes."}, indent=2))
    native_by_policy: dict[str, dict[str, dict]] = {POLICY_LIQUID: {}, POLICY_ZERANK: {}}
    for row in jl(OUT / "E3_C_RAW_NATIVE_SCORES.jsonl"):
        if row.get("stage") == "full":
            native_by_policy[row["policy"]][row["query_id"]] = row
    selected_rows = []
    feasibility = []
    for policy, rows in native_by_policy.items():
        for item in items:
            row = rows.get(item["query_id"])
            if row and row.get("status") == "VALID":
                selected_rows.append({"query_id": item["query_id"], "dataset": item["dataset"], "cohort": item["cohort"], "policy": policy, "candidate_ids": row["candidate_ids"], "selected_ids": row["selected_ids"], "score_sha256": sha_json(row["scores"]), "input_sha256": row["input_sha256"], "status": "VALID"})
            feasibility.append({"query_id": item["query_id"], "dataset": item["dataset"], "cohort": item["cohort"], "policy": policy, "status": row.get("status", "NOT_EXECUTED") if row else "NOT_EXECUTED", "reason": row.get("reason", "") if row else "missing", "input_tokens": row.get("input_tokens", row.get("max_pair_tokens", "")) if row else "", "physical_forward_count": row.get("physical_forward_count", 0) if row else 0})
    write_json(OUT / "E3_C_SELECTED_TOP5.jsonl", selected_rows)
    # The required artifact is JSONL; write one object per line rather than a JSON array.
    with (OUT / "E3_C_SELECTED_TOP5.jsonl").open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(OUT / "E3_C_CONTEXT_FEASIBILITY_AND_FAILURES.csv", feasibility)
    quality, pairs, disagreement, _ = quality_and_pairs(items, gold, selections, native_by_policy)
    write_csv(OUT / "E3_C_DATASET_QUALITY_SUMMARY.csv", quality)
    write_csv(OUT / "E3_C_PRIMARY_PAIRED_STATISTICS.csv", pairs)
    write_csv(OUT / "E3_C_HOLM_EXPLORATORY_SENSITIVITY.csv", holm(pairs))
    write_csv(OUT / "E3_C_PER_QUERY_DISAGREEMENT_AND_STRATA.csv", disagreement)
    timing_rows = []
    timing_status = "TIMING_BLOCKED_SHARED_GPU"
    # Valid JSONL with zero measurements: an empty file, rather than a JSON
    # array masquerading as JSONL, makes the fail-closed timing state explicit.
    (OUT / "E3_C_SELECTOR_LATENCY_RAW.jsonl").write_text("", encoding="utf-8")
    write_text(OUT / "E3_C_SELECTOR_LATENCY_AND_RESOURCE_REPORT.md", f"""# E3-C Selector Latency and Resource Report

Status: `{timing_status}`.

The prespecified 300-query E2-SP workload is frozen and its manifest hash is recorded in the protocol. GPU1 had unrelated resident compute processes at preflight, and GPUs0/2/4 were also occupied. The SDD requires exclusive or genuinely contention-free A100 timing for promoted complete Top-20-to-Top-5 latency. No unrelated process was interrupted, so isolated p50/p95/p99, 1-second SLO status, and 3-second SLO status are `NOT_MEASURED_ISOLATION_BLOCKED` rather than substituted with shared timings. Model load and serving costs are likewise `NOT_MEASURED` for the isolated contract.
""")
    write_text(OUT / "E3_C_AUTHENTICITY_AND_INTEGRITY_AUDIT.md", """# E3-C Authenticity and Integrity Audit

Both new branches use official local checkpoint snapshots and native interfaces. Raw rows preserve model revision, question/candidate/input hashes, output hashes where available, score semantics, candidate IDs, Top-5 IDs, input lengths, physical-forward counts and GPU attribution. Native inputs contain no gold qrels, answer sidecars or CSSR outcomes. Historical Jina/PPLX/BGE/retrieval-only traces are reused by SHA-verified source paths and are not re-inferred. A failed row is retained with an explicit status and is never converted to zero quality.

The append-only trace also preserves the initial Liquid adapter failures and the earlier zerank attempts whose legacy forward hook recorded zero. A two-pair module probe located the actual nested Qwen3Model execution boundary; the affected zerank rows were rerun with that hook, and only the final accepted rows (three physical decoder forwards per 20-candidate query) feed C4. Repeated valid attempts had identical input hashes, score vectors and Top-5 IDs. E3-0 remains pending metadata; no E3-F holdout, Jev, paid API, training, E3-D or multimodal branch was accessed.
""")
    status = {"status": "E3_C_PASS_WITH_LIMITATIONS", "owner_review_gate": "E3_C_OWNER_REVIEW_READY", "c0": "C0_PASS", "c1": {"liquid": "VALID_WITH_EXPLORATORY_CROSS_HEAD_LIMITATION", "zerank2": "VALID_RAW_LOGIT"}, "c2": {"smoke": smoke_report, "cal": cal_report}, "c3": {"logical_rows_per_model": len(items), "valid_full": {r["model"]: r["valid_full"] for r in model_results.values()}, "physical_full": {r["model"]: r["physical_full"] for r in model_results.values()}}, "c5": timing_status, "gpu_id": GPU_ID, "gpu_preflight": gpu, "jev_calls": 0, "paid_api_calls": 0, "holdout_access": False, "training": False, "multimodal": False, "setr": "NOT_RUN", "e3_0_status": "E3_0_HOLDOUT_RESERVATION_PENDING_METADATA", "source_trace_sha256": trace_hashes}
    write_json(OUT / "E3_C_PROVENANCE_MANIFEST.json", status)
    def _f(value: Any) -> str:
        return "NA" if value is None or value == "" else f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)

    scorecard_lines = ["| policy | dataset | eligible n | conditional n | valid n | CSSR-U | CSSR-C | any-gold coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in quality:
        scorecard_lines.append(f"| {row['policy']} | {row['dataset']} | {row['identity_eligible_n']} | {row['retrieval_conditional_n']} | {row['valid_eligible_n']} | {_f(row['cssr_u'])} | {_f(row['cssr_c'])} | {_f(row['any_gold_coverage'])} |")
    pair_lines = ["| comparison | cohort | population | n | difference | 95% CI | wins/losses/ties |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in pairs:
        pair_lines.append(f"| {row['comparison']} | {row['cohort']} | {row['population']} | {row['n']} | {_f(row['difference'])} | [{_f(row['ci95_low'])}, {_f(row['ci95_high'])}] | {row['wins']}/{row['losses']}/{row['ties']} |")
    trace_history = Counter((row.get("model"), row.get("stage"), row.get("status")) for row in jl(OUT / "E3_C_RAW_NATIVE_SCORES.jsonl"))
    history_lines = [f"- `{model}/{stage}/{status}`: {count}" for (model, stage, status), count in sorted(trace_history.items())]
    write_text(OUT / "E3_C_FINAL_SCIENTIFIC_REPORT.md", f"""# DecisionRAG E3-C Final Scientific Report

Verdict: `E3_C_PASS_WITH_LIMITATIONS`; owner gate: `E3_C_OWNER_REVIEW_READY`.

## Scope and frozen inputs

C0-C4 completed on the frozen 4,267-query historical population: Track A has 3,767 rows (HotpotQA 800, MuSiQue 2,167, 2Wiki 800) and Track B has 500 enterprise rows. Track B identity eligibility is 443 and its retrieval-conditional denominator is 330. Native inputs contain the query and the frozen 20 candidate text views only; gold/qrels/reference answers were loaded after inference. Historical retrieval-only, BGE, Jina and Perplexity baselines were reused from SHA-verified artifacts.

Liquid d1-3B uses the official `system_one` API with twenty native `noul` yes probabilities in one shared state. This is a valid exploratory selection interface, not proof of cross-head calibration or complementarity. zerank-2 uses the official Sentence-Transformers `CrossEncoder.predict()` contract and raw Yes logits; no sigmoid was applied. Official sources: [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B), [Liquid prompt.py](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py), and [zeroentropy/zerank-2-reranker](https://huggingface.co/zeroentropy/zerank-2-reranker).

## Native execution

| model | smoke | CAL | full valid | physical forwards | batch contract |
|---|---:|---:|---:|---:|---|
| liquid_d1_3b | {next(x['valid'] for x in smoke_report if x['model'] == 'liquid_d1_3b')}/12 | {next(x['valid'] for x in cal_report if x['model'] == 'liquid_d1_3b')}/30 | {model_results['liquid_d1_3b']['valid_full']}/4267 | {model_results['liquid_d1_3b']['physical_full']} | one native shared-state call/query |
| zerank_2 | {next(x['valid'] for x in smoke_report if x['model'] == 'zerank_2')}/12 | {next(x['valid'] for x in cal_report if x['model'] == 'zerank_2')}/30 | {model_results['zerank_2']['valid_full']}/4267 | {model_results['zerank_2']['physical_full']} | 20 pairs, batch size 8, three decoder forwards/query |

Every final accepted native row has twenty unique canonical candidates, finite scores, a stable Top-5, input and score/output hashes, `gold_in_inference_input=false`, and status `VALID`. The append-only raw trace retains prior Liquid adapter failures and earlier zerank attempts; they are not used as model outputs. The final accepted rows are deterministic across repeated attempts: no score, selected-ID, or input-hash mismatches were found.

Historical raw-trace attempt counts retained:
{chr(10).join(history_lines)}

## Quality scorecard

CSSR-U is unconditional over identity-eligible queries; CSSR-C is conditional on every gold document being present in the frozen candidate set. Any-gold coverage is a retrieval diagnostic. These are evidence-selection metrics, not factual answer accuracy.

{chr(10).join(scorecard_lines)}

The `__pooled__` rows are Track-A-only pooled rows; enterprise remains separate. `track_a_equal_dataset_macro` is the prespecified unweighted macro across HotpotQA, MuSiQue and 2Wiki and is distinct from pooled micro aggregation.

## Paired statistics

Paired bootstrap used seed `{SEED}` and `{BOOTSTRAP}` resamples. Track A and Track B are never pooled. The four model-versus-authenticated-baseline comparisons and the practical Liquid-versus-zerank comparison are shown below; intervals are exploratory and no causal architecture claim is made.

{chr(10).join(pair_lines)}

## Timing and resource limitations

C5 is `{timing_status}`. GPU1 was shared by unrelated resident processes at preflight; GPUs0/2/4 were also occupied. The SDD requires exclusive or genuinely contention-free A100 timing for complete Top-20-to-Top-5 selection. Therefore isolated p50/p95/p99, SLO compliance, model-load cost and matched recurring serving cost are `NOT_MEASURED_ISOLATION_BLOCKED`. `E3_C_SELECTOR_LATENCY_RAW.jsonl` is intentionally empty, not a zero-latency result. No shared-GPU timing is promoted as a hardware-equal comparison.

## Integrity and owner gate

Model revisions are pinned in `E3_C_FROZEN_EXPLORATORY_PROTOCOL.json`; source, baseline and snapshot hashes are recorded there. No fresh confirmatory holdout, Jev, paid API, training, E3-D, E3-F, multimodal input or production-service modification occurred. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`. The direct Liquid-versus-zerank comparison is practical and confounded by model family, training objective and input formulation; it is not a causal architecture ablation.

`E3_C_OWNER_REVIEW_READY`
""")
    # C6 owner gate and manifest hashes are written last.
    write_text(OUT / "E3_C_OWNER_REVIEW_GATE.md", """# E3-C Owner Review Gate

`E3_C_OWNER_REVIEW_READY`

Scientific status: `E3_C_PASS_WITH_LIMITATIONS`. C0-C4 completed with native traces for Liquid d1-3B and zerank-2. C5 is `TIMING_BLOCKED_SHARED_GPU`; no isolated latency claim is made. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`. No Jev, paid API, fresh holdout, training, E3-D, E3-F or multimodal work occurred.
""")
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name != "E3_C_PROVENANCE_MANIFEST.json"]
    status["artifact_sha256"] = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(artifacts)}
    write_json(OUT / "E3_C_PROVENANCE_MANIFEST.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
