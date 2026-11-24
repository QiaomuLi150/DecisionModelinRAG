#!/usr/bin/env python3
"""Assemble the frozen twelve selector policies for HERB H1."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/herb_h1"
RETRIEVAL = OUT / "HERB_H1_RETRIEVAL_SLATES.jsonl"
MODELS = ("bge", "qwen", "laya", "clef")
POLICIES = ("raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5")
MODEL_IDS = {"bge": "BAAI/bge-reranker-v2-m3", "qwen": "Qwen/Qwen3-VL-Reranker-2B", "laya": "convaiinnovations/laya", "clef": "Cloudflare/clef-flash"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def helpers():
    path = ROOT / "scripts/assemble_m3.py"
    spec = importlib.util.spec_from_file_location("herb_h1_assembly_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen assembly helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from herb_h1_selectors import build_items

    items = build_items()
    retrieval = load(RETRIEVAL)
    if len(retrieval) != len(items) or len({r["query_id"] for r in retrieval}) != len(retrieval):
        raise RuntimeError("invalid H1 retrieval population")
    traces = {model: {r["query_id"]: r for r in load(OUT / f"HERB_H1_{model.upper()}_POINTWISE_FULL.jsonl")} for model in MODELS}
    groups = {model: {r["query_id"]: r for r in load(OUT / f"HERB_H1_{model.upper()}_GROUPWISE_FULL.jsonl")} for model in ("laya", "clef")}
    if any(len(traces[model]) != len(items) for model in MODELS) or any(len(groups[model]) != len(items) for model in ("laya", "clef")):
        raise RuntimeError("incomplete H1 selector traces")
    if any(r.get("status") != "ok" or len(r.get("pointwise", [])) != 20 for model in MODELS for r in traces[model].values()):
        raise RuntimeError("H1 pointwise selector trace contains errors")
    if any(r.get("status") != "ok" or len(r.get("groups", [])) != 5 for model in ("laya", "clef") for r in groups[model].values()):
        raise RuntimeError("H1 groupwise selector trace contains errors")
    frozen = helpers()
    by_id = {item["query_id"]: item for item in items}
    results = []
    diagnostics = {"view_hash_mismatches": 0, "rescue_swaps": {model: 0 for model in MODELS}, "policy_counts": {p: 0 for p in POLICIES}}
    for slate in retrieval:
        qid = slate["query_id"]
        item = by_id[qid]
        candidates = item["candidates"]
        reference = [r["selection_text_sha256"] for r in traces["bge"][qid]["pointwise"]]
        for model in MODELS[1:]:
            current = [r["selection_text_sha256"] for r in traces[model][qid]["pointwise"]]
            if current != reference:
                diagnostics["view_hash_mismatches"] += 1
                raise RuntimeError(f"common selection-view mismatch {qid}:{model}")
        top20 = [c["doc_id"] for c in candidates]
        policies = [("raw20", top20, None, None, 0.0), ("retrieval5", top20[:5], None, None, 0.0)]
        selections = {}
        for model in MODELS:
            trace = traces[model][qid]
            score_map = {r["doc_id"]: float(r["score"]) for r in trace["pointwise"]}
            ranked = frozen.rank_indices(candidates, score_map)
            top5 = [c["doc_id"] for c in ranked[:5]]
            selections[model] = (score_map, top5)
            policies.append((f"{model}_p5", top5, model, None, float(trace.get("batch_duration_ms") or 0.0)))
        for model in ("laya", "clef"):
            score_map, _ = selections[model]
            group = groups[model][qid]
            winners = [g["winning_doc_id"] for g in group["groups"]]
            if len(set(winners)) != 5 or not set(winners).issubset(set(top20)):
                raise RuntimeError(f"invalid groupwise winners: {model}:{qid}")
            original = {c["doc_id"]: c["original_rank"] for c in candidates}
            ordered = sorted(winners, key=lambda doc: (-score_map[doc], original[doc], doc))
            policies.append((f"{model}_g5", ordered, model, group, sum(float(g.get("duration_ms") or 0.0) for g in group["groups"])))
        for model in MODELS:
            score_map, top5 = selections[model]
            start = time.perf_counter_ns()
            rescue = frozen.rescue_top5(candidates, score_map, top5)
            rescue_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            diagnostics["rescue_swaps"][model] += int(rescue["swapped"])
            policies.append((f"{model}_rescue5", rescue["selected_ids"], model, rescue, float(traces[model][qid].get("batch_duration_ms") or 0.0) + rescue_ms))
        for policy, selected, model, detail, wall_ms in policies:
            if len(selected) not in (5, 20) or len(set(selected)) != len(selected) or not set(selected).issubset(set(top20)):
                raise RuntimeError(f"invalid H1 policy output: {qid}:{policy}")
            results.append({
                "query_id": qid, "dataset": "herb", "policy_id": policy, "model_id": MODEL_IDS.get(model), "model_key": model,
                "top20_doc_ids": top20, "selected_doc_ids": selected, "status": "ok", "real_inference": policy not in ("raw20", "retrieval5"),
                "selection_wall_ms": wall_ms, "groupwise": detail if policy.endswith("_g5") else None, "rescue": detail if policy.endswith("_rescue5") else None,
                "pointwise_trace_ref": f"HERB_H1_{model.upper()}_POINTWISE_FULL.jsonl" if model else None,
                "qrels_accessed_by_selection": False, "gold_accessed": False,
            })
            diagnostics["policy_counts"][policy] += 1
    output = OUT / "HERB_H1_SELECTOR_RESULTS.jsonl"
    write(output, results)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "benchmark": "HERB", "track": "H1", "query_count": len(items), "policy_count": len(POLICIES), "rows": len(results),
        "policy_row_counts": diagnostics["policy_counts"], "view_hash_mismatches": diagnostics["view_hash_mismatches"], "rescue_swaps": diagnostics["rescue_swaps"],
        "retrieval_sha256": digest(RETRIEVAL), "selector_results_sha256": digest(output), "gold_accessed": False, "jev_calls": 0, "hosted_cloudflare_calls": 0,
    }
    (OUT / "HERB_H1_SELECTOR_RESULTS.manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
