#!/usr/bin/env python3
"""Independent offline E2-N selection and quality analysis.

Consumes only frozen Track A/B slates, read-only gold sidecars, and N3 score
traces.  It does not load a model, call a service, or generate reader output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("E2_N_N4_OUTPUT_DIR", str(ROOT / "results/non_jev/v2/e2_n/n4"))).resolve()
BOOTSTRAP_SEED = 731
BOOTSTRAP_REPS = 5000


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_items() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items

    rows = load_full_items()
    if len(rows) != 4267:
        raise RuntimeError(f"unexpected frozen population {len(rows)}")
    return rows


def load_gold() -> dict[str, set[str]]:
    """Resolve only one-to-one enterprise identities; preserve qst_0413 exclusion."""
    enterprise_db = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
    import sqlite3

    mapping: dict[str, list[str]] = defaultdict(list)
    conn = sqlite3.connect(f"file:{enterprise_db}?mode=ro", uri=True)
    try:
        for internal_id, original_id in conn.execute("SELECT internal_doc_id, original_doc_id FROM documents"):
            mapping[original_id].append(internal_id)
    finally:
        conn.close()
    gold: dict[str, set[str]] = {}
    for path in (ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl", ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl"):
        for row in load_jsonl(path):
            qid = row.get("query_id") or row.get("question_id")
            original = row.get("gold_doc_ids")
            if original is None:
                original = row.get("expected_doc_ids")
            if qid is None:
                raise RuntimeError(f"gold row missing ID in {path}")
            if path.name == "ENTERPRISE_GOLD_SIDECAR.jsonl":
                resolved = []
                if any(len(mapping[x]) != 1 for x in dict.fromkeys(original or [])):
                    continue
                for x in dict.fromkeys(original or []):
                    resolved.append(mapping[x][0])
                gold[qid] = set(resolved)
            else:
                gold[qid] = set(original or [])
    return gold


def read_traces() -> dict[str, dict[str, dict]]:
    base = ROOT / "results/non_jev/v2/e2_n/n3"
    paths = {
        "bge_pointwise": base / "E2_N_N3_BGE_FULL.jsonl",
        "laya_pointwise": base / "E2_N_N3_LAYA_FULL.jsonl",
        "clef_pointwise": base / "E2_N_N3_CLEF_FULL.jsonl",
    }
    if os.environ.get("E2_N_N4_INCLUDE_QWEN") == "1":
        paths["qwen_direct"] = base / "E2_N_N3_QWEN_DIRECT_FULL.jsonl"
    extra_paths = {
        "qwen06": base / "E2_N_N3_QWEN06_FULL.jsonl",
        "qwen4": base / "E2_N_N3_QWEN4_FULL.jsonl",
        "gte": base / "E2_N_N3_GTE_FULL.jsonl",
        "minilm": base / "E2_N_N3_MINILM_FULL.jsonl",
        "jina35": base / "E2_N_N3_JINA35_FULL.jsonl",
        "nimble": base / "E2_N_N3_NIMBLE_FULL.jsonl",
        "intern_decision": base / "E2_N_N3_INTERN_DECISION_FULL.jsonl",
        "pplx_decider": base / "E2_N_N3_PPLX_DECIDER_FULL.jsonl",
    }
    for policy in filter(None, os.environ.get("E2_N_N4_EXTRA_POLICIES", "").split(",")):
        if policy not in extra_paths:
            raise RuntimeError(f"unknown extra E2-N policy {policy}")
        paths[policy] = extra_paths[policy]
    traces: dict[str, dict[str, dict]] = {}
    for policy, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing N3 trace {path}")
        rows = load_jsonl(path)
        if policy == "clef_pointwise":
            traces["clef_pointwise"] = {}
            traces["clef_joint"] = {}
        for row in rows:
            if row.get("status") != "ok":
                if policy == "intern_decision" and "above 8192" in row.get("error_type", ""):
                    # The native Intern-Decision contract rejects these views;
                    # retain the explicit context-blocked row in N3 and let N4
                    # analyze the valid Track-A subset without relabeling it.
                    continue
                raise RuntimeError(f"N3 error row in {path}: {row.get('query_id')}")
            if row.get("physical_forward_evidence") is not True:
                raise RuntimeError(f"N3 physical evidence missing in {path}: {row.get('query_id')}")
            if not row.get("actual_model_input_sha256") or not row.get("raw_response_sha256"):
                raise RuntimeError(f"N3 input/output hash missing in {path}: {row.get('query_id')}")
            ids = row.get("candidate_ids", [])
            scores = row.get("scores", [])
            if len(ids) != 20 or len(set(ids)) != 20 or len(scores) != 20 or any(not math.isfinite(float(x)) for x in scores):
                raise RuntimeError(f"N3 score alignment invalid in {path}: {row.get('query_id')}")
            if policy == "clef_pointwise":
                target = "clef_joint" if row.get("formulation") == "native_joint_binary" else "clef_pointwise"
                traces[target][row["query_id"]] = row
            else:
                traces.setdefault(policy, {})[row["query_id"]] = row
    return traces


def top5(ids: list[str], scores: list[float]) -> list[str]:
    return [doc_id for _, doc_id in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), ids.index(x[1]), x[1]))[:5]]


def metric(gold: set[str], candidates: list[str], selected: list[str]) -> dict:
    eligible = 1 <= len(gold) <= 5
    conditional = eligible and gold.issubset(set(candidates))
    success = eligible and gold.issubset(set(selected))
    return {"eligible": eligible, "conditional": conditional, "success": success, "any_gold_present": bool(gold.intersection(candidates))}


def bootstrap_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_REPS))
    return estimates[int(0.025 * BOOTSTRAP_REPS)], estimates[int(0.975 * BOOTSTRAP_REPS) - 1]


def safe_rate(n: int, d: int) -> float | None:
    return n / d if d else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    gold = load_gold()
    traces = read_traces()
    qids = {x["query_id"] for x in items}
    for policy, rows in traces.items():
        if set(rows) != qids:
            if policy == "intern_decision" and set(rows).issubset(qids):
                # N3 preserves the 500 enterprise context-blocked rows as
                # explicit errors; N4 evaluates only the valid native rows.
                continue
            raise RuntimeError(f"{policy} trace population mismatch: {len(rows)} vs {len(qids)}")

    policy_names = ["retrieval5", "bge_pointwise", "laya_pointwise", "clef_pointwise", "clef_joint"]
    if os.environ.get("E2_N_N4_INCLUDE_QWEN") == "1":
        policy_names.append("qwen_direct")
    policy_names.extend(filter(None, os.environ.get("E2_N_N4_EXTRA_POLICIES", "").split(",")))
    score_policies = tuple(policy for policy in policy_names if policy != "retrieval5")
    selections: dict[str, dict[str, list[str]]] = {p: {} for p in policy_names}
    selection_rows = []
    for item in items:
        qid = item["query_id"]
        ids = [c["doc_id"] for c in item["candidates"]]
        selections["retrieval5"][qid] = ids[:5]
        selection_rows.append({"query_id": qid, "cohort": item["cohort"], "dataset": item["dataset"], "policy": "retrieval5", "candidate_ids": ids, "selected_ids": ids[:5], "score_source": "frozen_retriever_order"})
        for policy in score_policies:
            if qid not in traces[policy]:
                selections[policy][qid] = None
                selection_rows.append({"query_id": qid, "cohort": item["cohort"], "dataset": item["dataset"], "policy": policy, "candidate_ids": ids, "selected_ids": [], "status": "context_blocked_native_contract"})
                continue
            row = traces[policy][qid]
            selected = top5(ids, row["scores"])
            selections[policy][qid] = selected
            selection_rows.append({"query_id": qid, "cohort": item["cohort"], "dataset": item["dataset"], "policy": policy, "candidate_ids": ids, "selected_ids": selected, "score_source": row.get("trace_origin", "e2_n_n3_trace"), "score_sha256": row.get("raw_response_sha256"), "trace_input_sha256": row.get("actual_model_input_sha256")})
    with (OUT / "E2_N_TOP5_SELECTIONS.jsonl").open("w", encoding="utf-8") as f:
        for row in selection_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # Per-policy quality summaries, kept separate for Track A and Track B.
    quality = []
    for cohort in ("track_a", "track_b"):
        cohort_items = [x for x in items if x["cohort"] == cohort]
        for dataset in sorted({x["dataset"] for x in cohort_items}):
            dataset_items = [x for x in cohort_items if x["dataset"] == dataset]
            scopes = [(dataset, dataset_items)]
            for policy in policy_names:
                rows = [metric(gold.get(x["query_id"], set()), [c["doc_id"] for c in x["candidates"]], selections[policy][x["query_id"]]) for x in dataset_items if selections[policy].get(x["query_id"]) is not None]
                eligible = [r for r in rows if r["eligible"]]
                conditional = [r for r in eligible if r["conditional"]]
                quality.append({"cohort": cohort, "dataset": dataset, "policy": policy, "n_total": len(rows), "n_eligible": len(eligible), "n_conditional": len(conditional), "n_any_gold": sum(r["any_gold_present"] for r in rows), "success_unconditional": sum(r["success"] for r in eligible), "success_conditional": sum(r["success"] for r in conditional), "cssr_unconditional": safe_rate(sum(r["success"] for r in eligible), len(eligible)), "cssr_conditional": safe_rate(sum(r["success"] for r in conditional), len(conditional)), "any_gold_coverage": safe_rate(sum(r["any_gold_present"] for r in rows), len(rows)), "interpretation": "complete-support selection proxy; not reader factual accuracy"})
        # Equal-dataset macro is a secondary descriptive estimate, not pooled.
        for policy in policy_names:
            parts = [r for r in quality if r["cohort"] == cohort and r["policy"] == policy]
            eligible_parts = [r["cssr_unconditional"] for r in parts if r["cssr_unconditional"] is not None]
            conditional_parts = [r["cssr_conditional"] for r in parts if r["cssr_conditional"] is not None]
            coverage_parts = [r["any_gold_coverage"] for r in parts if r["any_gold_coverage"] is not None]
            quality.append({"cohort": cohort, "dataset": "__equal_dataset_macro__", "policy": policy, "n_total": sum(r["n_total"] for r in parts), "n_eligible": sum(r["n_eligible"] for r in parts), "n_conditional": sum(r["n_conditional"] for r in parts), "n_any_gold": sum(r["n_any_gold"] for r in parts), "success_unconditional": sum(r["success_unconditional"] for r in parts), "success_conditional": sum(r["success_conditional"] for r in parts), "cssr_unconditional": statistics.mean(eligible_parts) if eligible_parts else None, "cssr_conditional": statistics.mean(conditional_parts) if conditional_parts else None, "any_gold_coverage": statistics.mean(coverage_parts) if coverage_parts else None, "interpretation": "equal-dataset macro of dataset-level complete-support proxies; not a pooled query estimate"})
    quality_fields = list(quality[0])
    with (OUT / "E2_N_MODEL_DATASET_QUALITY_SUMMARY.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=quality_fields)
        writer.writeheader(); writer.writerows(quality)

    comparison_specs = [
        ("bge_minus_retrieval5", "bge_pointwise", "retrieval5"),
        ("laya_minus_retrieval5", "laya_pointwise", "retrieval5"),
        ("clef_pointwise_minus_retrieval5", "clef_pointwise", "retrieval5"),
        ("clef_joint_minus_retrieval5", "clef_joint", "retrieval5"),
        ("laya_minus_bge", "laya_pointwise", "bge_pointwise"),
        ("clef_pointwise_minus_bge", "clef_pointwise", "bge_pointwise"),
        ("clef_joint_minus_bge", "clef_joint", "bge_pointwise"),
        ("clef_joint_minus_clef_pointwise", "clef_joint", "clef_pointwise"),
    ]
    if os.environ.get("E2_N_N4_INCLUDE_QWEN") == "1":
        comparison_specs.extend([
            ("qwen_direct_minus_retrieval5", "qwen_direct", "retrieval5"),
            ("qwen_direct_minus_bge", "qwen_direct", "bge_pointwise"),
        ])
    for policy in filter(None, os.environ.get("E2_N_N4_EXTRA_POLICIES", "").split(",")):
        comparison_specs.extend([(f"{policy}_minus_retrieval5", policy, "retrieval5"), (f"{policy}_minus_bge", policy, "bge_pointwise")])
    paired = []
    for cohort in ("track_a", "track_b"):
        cohort_items = [x for x in items if x["cohort"] == cohort]
        for population in ("unconditional_eligible", "retrieval_conditional"):
            for name, left, right in comparison_specs:
                deltas = []
                for item in cohort_items:
                    g = gold.get(item["query_id"], set())
                    if not (1 <= len(g) <= 5):
                        continue
                    ids = [c["doc_id"] for c in item["candidates"]]
                    if population == "retrieval_conditional" and not g.issubset(set(ids)):
                        continue
                    left_selected = selections[left].get(item["query_id"])
                    right_selected = selections[right].get(item["query_id"])
                    if left_selected is None or right_selected is None:
                        continue
                    left_success = g.issubset(set(left_selected))
                    right_success = g.issubset(set(right_selected))
                    deltas.append(float(left_success) - float(right_success))
                low, high = bootstrap_ci(deltas)
                paired.append({"cohort": cohort, "population": population, "comparison": name, "left_policy": left, "right_policy": right, "n": len(deltas), "difference": safe_rate(sum(deltas), len(deltas)), "ci95_low": low, "ci95_high": high, "paired_wins": sum(x > 0 for x in deltas), "paired_losses": sum(x < 0 for x in deltas), "paired_ties": sum(x == 0 for x in deltas), "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED, "interpretation": "exploratory follow-up; CI is query-paired and not a confirmatory significance claim"})
    paired_fields = list(paired[0])
    with (OUT / "E2_N_PRIMARY_EXPLORATORY_PAIRED_STATISTICS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paired_fields)
        writer.writeheader(); writer.writerows(paired)

    # Runtime/context diagnostics from raw N3 traces only.
    trace_paths = {"bge_pointwise": ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_BGE_FULL.jsonl", "laya_pointwise": ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_LAYA_FULL.jsonl", "clef": ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_CLEF_FULL.jsonl"}
    if os.environ.get("E2_N_N4_INCLUDE_QWEN") == "1":
        trace_paths["qwen_direct"] = ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_QWEN_DIRECT_FULL.jsonl"
    for policy in filter(None, os.environ.get("E2_N_N4_EXTRA_POLICIES", "").split(",")):
        trace_paths[policy] = ROOT / "results/non_jev/v2/e2_n/n3" / {
            "qwen06": "E2_N_N3_QWEN06_FULL.jsonl",
            "qwen4": "E2_N_N3_QWEN4_FULL.jsonl",
            "gte": "E2_N_N3_GTE_FULL.jsonl",
            "minilm": "E2_N_N3_MINILM_FULL.jsonl",
            "jina35": "E2_N_N3_JINA35_FULL.jsonl",
            "nimble": "E2_N_N3_NIMBLE_FULL.jsonl",
            "intern_decision": "E2_N_N3_INTERN_DECISION_FULL.jsonl",
            "pplx_decider": "E2_N_N3_PPLX_DECIDER_FULL.jsonl",
        }[policy]
    runtime = []
    for policy, path in trace_paths.items():
        rows = load_jsonl(path)
        durations = []
        tokens = []
        forwards = 0
        for row in rows:
            duration = row.get("duration_ms") or row.get("telemetry", {}).get("duration_ms")
            if duration is not None:
                durations.append(float(duration))
            value = row.get("input_tokens")
            if isinstance(value, list):
                tokens.append(sum(int(x) for x in value))
            elif isinstance(value, int):
                tokens.append(value)
            forwards += int(row.get("physical_forward_calls", 0)) if isinstance(row.get("physical_forward_calls"), int) else 0
        durations.sort(); tokens.sort()
        runtime.append({"policy": policy, "n": len(rows), "duration_p50_ms": durations[len(durations)//2] if durations else None, "duration_p95_ms": durations[max(0, math.ceil(.95 * len(durations)) - 1)] if durations else None, "duration_mean_ms": statistics.mean(durations) if durations else None, "measured_selector_seconds": sum(durations) / 1000 if durations else None, "input_token_total_p50": tokens[len(tokens)//2] if tokens else "NOT_AVAILABLE", "input_token_total_p95": tokens[max(0, math.ceil(.95 * len(tokens)) - 1)] if tokens else "NOT_AVAILABLE", "physical_forward_calls": forwards, "cost_dollars": "NOT_MEASURED"})
    with (OUT / "E2_N_RUNTIME_GPU_AND_COST_DIAGNOSTICS.md").open("w", encoding="utf-8") as f:
        f.write("# E2-N Runtime, GPU and Cost Diagnostics\n\n")
        f.write("These are selector-scoring measurements, not end-to-end production latency or a monetary invoice. Clef full traces are compatible E2-M1 traces reused with source hashes; BGE and Laya are fresh E2-N traces. GPU active-seconds and dollar rates were not measured consistently, so cost is `NOT_MEASURED`.\n\n")
        f.write("| Policy | n | p50 ms | p95 ms | Mean ms | Measured selector seconds | Input-token p50 | Input-token p95 | Physical forwards | Cost |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in runtime:
            f.write(f"| {r['policy']} | {r['n']} | {r['duration_p50_ms']} | {r['duration_p95_ms']} | {r['duration_mean_ms']} | {r['measured_selector_seconds']} | {r['input_token_total_p50']} | {r['input_token_total_p95']} | {r['physical_forward_calls']} | {r['cost_dollars']} |\n")
        f.write("\nNo Jev or paid API calls occurred. Existing host services and unrelated GPU workloads were not restarted or interrupted.\n")

    with (OUT / "E2_N_RAW_SCORES_AND_TRACE_INDEX.jsonl").open("w", encoding="utf-8") as f:
        for policy, path in trace_paths.items():
            f.write(json.dumps({"policy": policy, "path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "rows": len(load_jsonl(path)), "gold_in_inference_input": False}, sort_keys=True) + "\n")

    # Concise analysis report.
    lines = [
        "# E2-N Context and Mechanism Analysis",
        "",
        "The evaluator applied one deterministic Top-5 rule to all valid 20-candidate traces: descending score, then original candidate order, then candidate ID. Retrieval5 is the frozen first-five prefix.",
        "",
        "Track A and Track B are reported separately. Complete-support CSSR means every identity-resolved gold document is in the selected Top-5; it is not a reader answer-accuracy measure.",
        "",
        "- BGE and Laya support independent pointwise relevance only in this run.",
        "- Clef supports both independent pointwise and native joint K=20 binary relevance; its full traces were reused from compatible E2-M1 execution with source hashes.",
        "- Qwen service scores remain a separate limited branch; direct checkpoint Qwen traces are included only when E2_N_N4_INCLUDE_QWEN=1 and are never mixed with service scores.",
        "- Laya’s checkpoint temperature warning is retained: score ordering was evaluated, but confidence calibration is not certified.",
        "- No new reader answers were generated; these results assess selection quality proxies only.",
        "",
        f"Bootstrap protocol: {BOOTSTRAP_REPS} query-level resamples, seed {BOOTSTRAP_SEED}; intervals are exploratory and not confirmatory significance claims.",
    ]
    (OUT / "E2_N_CONTEXT_AND_MECHANISM_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_index = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "E2_N_N4_OWNER_REVIEW_GATE.md":
            artifact_index.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size})
    gate = {
        "stage": "N4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_LIMITATIONS",
        "completed": ["independent offline selection", "Track A quality summaries", "Track B quality summaries", "paired bootstrap intervals", "runtime/context diagnostics", "trace hash index"],
        "blocked_or_limited": ["resident Qwen service telemetry remains non-instrumented", "additional official checkpoints not yet adapter-validated", "monetary GPU cost", "reader factual accuracy"],
        "jev_calls": 0,
        "paid_api_calls": 0,
        "fresh_confirmatory_holdout_accessed": False,
        "artifact_index": artifact_index,
    }
    (OUT / "E2_N4_OWNER_REVIEW_GATE.md").write_text("# E2-N4 Owner Review Gate\n\nStatus: `PASS_WITH_LIMITATIONS`\n\nN4 offline analysis is complete for the declared trace roster. Resident Qwen service telemetry remains limited; direct checkpoint Qwen, when enabled, is a separate physical trace contract. Additional downloaded checkpoints require their own native-adapter validation before promotion. Track A and Track B remain separate, and complete-support CSSR is not factual reader accuracy. Jev calls: `0`; paid calls: `0`; fresh confirmatory holdout accessed: `false`.\n", encoding="utf-8")
    gate["artifact_index"] = [{"path": str((OUT / "E2_N4_OWNER_REVIEW_GATE.md").relative_to(ROOT)), "sha256": sha_file(OUT / "E2_N4_OWNER_REVIEW_GATE.md"), "bytes": (OUT / "E2_N4_OWNER_REVIEW_GATE.md").stat().st_size}] + artifact_index
    (OUT / "E2_N4_PROVENANCE_MANIFEST.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "selection_rows": len(selection_rows), "quality_rows": len(quality), "paired_rows": len(paired)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
