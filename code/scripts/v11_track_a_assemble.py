#!/usr/bin/env python3
"""Assemble the frozen 12 Track A policies from selector traces."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/non_jev/v11/track_a"
COHORT_PATH = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"
CORPUS_PATH = ROOT / "data/corpus/passages.jsonl"
MODELS = ("bge", "qwen", "laya", "clef")
POLICIES = ("raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5")
MODEL_IDS = {"bge": "BAAI/bge-reranker-v2-m3", "qwen": "Qwen/Qwen3-VL-Reranker-2B", "laya": "convaiinnovations/laya", "clef": "Cloudflare/clef-flash"}
SEED = 20261011


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_selection_module():
    spec = importlib.util.spec_from_file_location("decisionrag_v11_assemble_legacy", ROOT / "scripts/assemble_m3.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen selection assembly helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    suffix = args.mode.upper()
    retrieval_path = OUT_DIR / ("V11_TRACK_A_RETRIEVAL_SMOKE.jsonl" if args.mode == "smoke" else "V11_TRACK_A_RETRIEVAL_SLATES.jsonl")
    cohort = load_rows(COHORT_PATH)
    retrieval = load_rows(retrieval_path)
    queries = {row["query_id"]: row for row in cohort}
    corpus = {row["doc_id"]: row for row in load_rows(CORPUS_PATH)}
    pointwise = {model: {row["query_id"]: row for row in load_rows(OUT_DIR / f"V11_TRACK_A_{model.upper()}_POINTWISE_{suffix}.jsonl")} for model in MODELS}
    groups = {model: {row["query_id"]: row for row in load_rows(OUT_DIR / f"V11_TRACK_A_{model.upper()}_GROUPWISE_{suffix}.jsonl")} for model in ("laya", "clef")}
    if not retrieval:
        raise RuntimeError("retrieval trace is empty")
    for model in MODELS:
        if len(pointwise[model]) != len(retrieval) or any(row.get("status") != "ok" or len(row.get("pointwise", [])) != 20 for row in pointwise[model].values()):
            raise RuntimeError(f"incomplete selector trace for {model}")
    for model in ("laya", "clef"):
        if len(groups[model]) != len(retrieval) or any(row.get("status") != "ok" or len(row.get("groups", [])) != 5 for row in groups[model].values()):
            raise RuntimeError(f"incomplete groupwise trace for {model}")
    helpers = legacy_selection_module()
    selection_rows = []
    diagnostics = {"view_hash_mismatches": 0, "policy_counts": {}, "rescue_swaps": {model: 0 for model in MODELS}}
    for slate in retrieval:
        qid = slate["query_id"]
        question = queries[qid]
        candidates = []
        for rank, doc_id in enumerate(slate["candidate_ids"], start=1):
            source = corpus[doc_id]
            candidates.append({"doc_id": doc_id, "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": rank})
        reference = [point["selection_text_sha256"] for point in pointwise["bge"][qid]["pointwise"]]
        for model in MODELS[1:]:
            current = [point["selection_text_sha256"] for point in pointwise[model][qid]["pointwise"]]
            if current != reference:
                diagnostics["view_hash_mismatches"] += 1
                raise RuntimeError(f"common selection view mismatch {qid}:{model}")
        common = {"query_id": qid, "dataset": question["dataset"], "top20_doc_ids": [candidate["doc_id"] for candidate in candidates], "gold_accessed": False}
        policies = [("raw20", list(common["top20_doc_ids"]), None, None, 0.0), ("retrieval5", list(common["top20_doc_ids"][:5]), None, None, 0.0)]
        model_selections = {}
        for model in MODELS:
            trace = pointwise[model][qid]
            score_map = {point["doc_id"]: float(point["score"]) for point in trace["pointwise"]}
            ranked = helpers.rank_indices(candidates, score_map)
            top5 = [candidate["doc_id"] for candidate in ranked[:5]]
            model_selections[model] = (score_map, ranked, top5)
            policies.append((f"{model}_p5", top5, model, None, float(trace.get("batch_duration_ms") or 0.0)))
        for model in ("laya", "clef"):
            score_map, _, _ = model_selections[model]
            group = groups[model][qid]
            winners = [entry["winning_doc_id"] for entry in group["groups"]]
            if len(set(winners)) != 5 or not set(winners).issubset(set(common["top20_doc_ids"])):
                raise RuntimeError(f"invalid groupwise winners {model}:{qid}")
            ordered = sorted(winners, key=lambda doc_id: (-score_map[doc_id], next(c["original_rank"] for c in candidates if c["doc_id"] == doc_id), doc_id))
            policies.append((f"{model}_g5", ordered, model, group, sum(float(entry.get("duration_ms") or 0.0) for entry in group["groups"])))
        for model in MODELS:
            score_map, _, top5 = model_selections[model]
            start = time.perf_counter_ns()
            rescue = helpers.rescue_top5(candidates, score_map, top5)
            rescue_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            diagnostics["rescue_swaps"][model] += int(rescue["swapped"])
            policies.append((f"{model}_rescue5", rescue["selected_ids"], model, rescue, float(pointwise[model][qid].get("batch_duration_ms") or 0.0) + rescue_ms))
        for policy_id, selected, model, detail, wall_ms in policies:
            if len(selected) not in (5, 20) or len(set(selected)) != len(selected) or not set(selected).issubset(set(common["top20_doc_ids"])):
                raise RuntimeError(f"invalid selection {qid}:{policy_id}")
            row = dict(common)
            row.update({
                "policy_id": policy_id,
                "model_id": MODEL_IDS.get(model),
                "model_key": model,
                "selected_doc_ids": selected,
                "status": "ok",
                "real_inference": policy_id not in ("raw20", "retrieval5"),
                "selection_wall_ms": wall_ms,
                "groupwise": detail if policy_id.endswith("_g5") else None,
                "rescue": detail if policy_id.endswith("_rescue5") else None,
                "pointwise_trace_ref": f"V11_TRACK_A_{model.upper()}_POINTWISE_{suffix}.jsonl" if model else None,
                "qrels_accessed_by_selection": False,
            })
            selection_rows.append(row)
            diagnostics["policy_counts"][policy_id] = diagnostics["policy_counts"].get(policy_id, 0) + 1
    output = OUT_DIR / f"V11_TRACK_A_SELECTOR_RESULTS_{suffix}.jsonl"
    write_rows(output, selection_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "query_count": len(retrieval),
        "policy_count": len(POLICIES),
        "rows": len(selection_rows),
        "policy_row_counts": diagnostics["policy_counts"],
        "view_hash_mismatches": diagnostics["view_hash_mismatches"],
        "rescue_swaps": diagnostics["rescue_swaps"],
        "retrieval_sha256": sha256(retrieval_path),
        "selector_results_sha256": sha256(output),
        "gold_accessed": False,
        "jev_calls": 0,
        "hosted_cloudflare_calls": 0,
    }
    (OUT_DIR / f"V11_TRACK_A_SELECTOR_RESULTS_{suffix}.manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
