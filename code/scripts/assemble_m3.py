#!/usr/bin/env python3
"""Assemble the twelve non-Jev M3 selection policies from real selector traces."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decisionrag.selection import deterministic_quartets, rank_indices, rescue_top5  # noqa: E402

MODELS = ("bge", "qwen", "laya", "clef")
MODEL_IDS = {"bge": "BAAI/bge-reranker-v2-m3", "qwen": "Qwen/Qwen3-VL-Reranker-2B", "laya": "convaiinnovations/laya", "clef": "Cloudflare/clef-flash"}
SEED = 20261011


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    questions = rows(ROOT / "data/manifests/questions.jsonl")
    slates = rows(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")
    corpus = {row["doc_id"]: row for row in rows(ROOT / "data/corpus/passages.jsonl")}
    traces = {model: {row["query_id"]: row for row in rows(ROOT / "data/results/m3" / f"{model}_all_pointwise.jsonl")} for model in MODELS}
    groups = {model: {row["query_id"]: row for row in rows(ROOT / "data/results/m3" / f"{model}_all_groupwise.jsonl")} for model in ("laya", "clef")}
    if any(len(value) != 750 for value in traces.values()) or any(len(value) != 750 for value in groups.values()):
        raise RuntimeError("M3 selector trace count is incomplete")
    for model in MODELS:
        if any(row["status"] != "ok" or len(row["pointwise"]) != 20 for row in traces[model].values()):
            raise RuntimeError(f"M3 {model} trace contains an error or non-20 result")
    # Every model must consume the same frozen semantic view hash for every pair.
    for question in questions:
        qid = question["query_id"]
        reference = [point["selection_text_sha256"] for point in traces["bge"][qid]["pointwise"]]
        for model in MODELS[1:]:
            current = [point["selection_text_sha256"] for point in traces[model][qid]["pointwise"]]
            if current != reference:
                raise RuntimeError(f"common selection view mismatch for {qid}: {model}")
    selection_rows = []
    diagnostics = {"view_hash_mismatches": 0, "policy_counts": {}, "rescue_swaps": {}}
    for question, slate in zip(questions, slates):
        qid = question["query_id"]
        candidates = []
        for rank, doc_id in enumerate(slate["candidate_ids"], start=1):
            source = corpus[doc_id]
            candidates.append({"doc_id": doc_id, "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": rank})
        common = {"query_id": qid, "dataset": question["dataset"], "top20_doc_ids": [c["doc_id"] for c in candidates]}
        policies = [("raw20", list(common["top20_doc_ids"]), None, None, 0.0), ("retrieval5", list(common["top20_doc_ids"][:5]), None, None, 0.0)]
        model_selections = {}
        for model in MODELS:
            trace = traces[model][qid]
            score_map = {point["doc_id"]: float(point["score"]) for point in trace["pointwise"]}
            ranked = rank_indices(candidates, score_map)
            top5 = [candidate["doc_id"] for candidate in ranked[:5]]
            model_selections[model] = (score_map, ranked, top5)
            policies.append((f"{model}_p5", top5, model, None, float(trace["batch_duration_ms"])))
        for model in ("laya", "clef"):
            score_map, ranked, _ = model_selections[model]
            group = groups[model][qid]
            if group["status"] != "ok" or len(group["groups"]) != 5:
                raise RuntimeError(f"invalid groupwise trace {model}:{qid}")
            winners = [entry["winning_doc_id"] for entry in group["groups"]]
            if len(set(winners)) != 5 or not set(winners).issubset(set(common["top20_doc_ids"])):
                raise RuntimeError(f"invalid groupwise winners {model}:{qid}")
            ordered = sorted(winners, key=lambda doc_id: (-score_map[doc_id], next(c["original_rank"] for c in candidates if c["doc_id"] == doc_id), doc_id))
            policies.append((f"{model}_g5", ordered, model, group, sum(float(entry["duration_ms"]) for entry in group["groups"])))
        for model in MODELS:
            score_map, ranked, top5 = model_selections[model]
            rescue_start = time.perf_counter_ns()
            rescue = rescue_top5(candidates, score_map, top5)
            rescue_ms = (time.perf_counter_ns() - rescue_start) / 1_000_000.0
            diagnostics["rescue_swaps"].setdefault(model, 0)
            diagnostics["rescue_swaps"][model] += int(rescue["swapped"])
            policies.append((f"{model}_rescue5", rescue["selected_ids"], model, rescue, float(traces[model][qid]["batch_duration_ms"]) + rescue_ms))
        for policy_id, selected, model, detail, wall_ms in policies:
            if len(selected) not in (5, 20) or len(set(selected)) != len(selected) or not set(selected).issubset(set(common["top20_doc_ids"])):
                raise RuntimeError(f"invalid selected set {qid}:{policy_id}")
            row = dict(common)
            row.update({"policy_id": policy_id, "model_id": MODEL_IDS.get(model), "model_key": model, "selected_doc_ids": selected, "status": "ok", "real_inference": policy_id not in ("raw20", "retrieval5"), "selection_wall_ms": wall_ms, "groupwise": detail if policy_id.endswith("_g5") else None, "rescue": detail if policy_id.endswith("_rescue5") else None, "pointwise_trace_ref": f"data/results/m3/{model}_all_pointwise.jsonl" if model else None, "qrels_accessed_by_selection": False})
            selection_rows.append(row)
            diagnostics["policy_counts"][policy_id] = diagnostics["policy_counts"].get(policy_id, 0) + 1
    out = ROOT / "data/results/m3/selection_results.jsonl"
    write_rows(out, selection_rows)
    manifest = {"study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2", "created_utc": datetime.now(timezone.utc).isoformat(), "queries": 750, "policies": sorted(diagnostics["policy_counts"]), "rows": len(selection_rows), "selection_results_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "diagnostics": diagnostics, "jev_calls": 0, "hosted_cloudflare_calls": 0}
    (ROOT / "data/manifests/m3_selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "results/audit/phase_state.json").write_text(json.dumps({"current_milestone": "M3", "status": "DONE", "jev_calls": 0, "jev_execution_allowed": False, "m3_manifest": str(ROOT / "data/manifests/m3_selection_manifest.json")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
