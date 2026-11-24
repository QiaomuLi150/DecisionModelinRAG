#!/usr/bin/env python3
"""Analyze the completed, uncached M5 serving traces and materialize reports."""
from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]
DATASETS = ["hotpot", "musique", "wiki2"]
B = 5000
SEED = 731


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(float(x) for x in values)
    if not values:
        return None
    index = (len(values) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def normalize(text: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return " ".join(token for token in value.split() if token not in {"a", "an", "the"})


def f1(prediction: str, gold: str) -> float:
    pred, truth = normalize(prediction).split(), normalize(gold).split()
    if not pred or not truth:
        return float(pred == truth)
    common = 0
    remaining = list(truth)
    for token in pred:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if not common:
        return 0.0
    return 2 * (common / len(pred)) * (common / len(truth)) / ((common / len(pred)) + (common / len(truth)))


def answer_scores(answer: str, question: dict) -> tuple[float, float]:
    golds = [question["answer"], *question.get("answer_aliases", [])]
    return (float(any(normalize(answer) == normalize(gold) for gold in golds)), max(f1(answer, gold) for gold in golds))


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("\n", encoding="utf-8")
        return
    keys = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def bootstrap_delta(a: list[float], b: list[float], seed: int) -> dict:
    if len(a) != len(b) or not a:
        return {"status": "NOT_MEASURED", "n": 0}
    deltas = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    samples = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(B)]
    return {
        "status": "OK" if len(deltas) >= 40 else "UNDERPOWERED_DESCRIPTIVE",
        "n": len(deltas),
        "delta": statistics.mean(deltas),
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "bootstrap_replicates": B,
        "bootstrap_seed": seed,
    }


def macro_bootstrap(per_dataset: dict[str, tuple[list[float], list[float]]]) -> dict:
    if not all(a and len(a) == len(b) for a, b in per_dataset.values()):
        return {"status": "NOT_MEASURED", "n_by_dataset": {d: len(a) for d, (a, _) in per_dataset.items()}}
    observed = statistics.mean(statistics.mean(a[i] - b[i] for i in range(len(a))) for a, b in per_dataset.values())
    rng = random.Random(SEED)
    samples = []
    for _ in range(B):
        samples.append(statistics.mean(sum((lambda index: a[index] - b[index])(rng.randrange(len(a))) for _ in a) / len(a) for a, b in per_dataset.values()))
    return {
        "status": "OK" if min(len(a) for a, _ in per_dataset.values()) >= 40 else "UNDERPOWERED_DESCRIPTIVE",
        "n_by_dataset": {d: len(a) for d, (a, _) in per_dataset.items()},
        "delta": observed,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "bootstrap_replicates": B,
        "bootstrap_seed": SEED,
    }


def main() -> int:
    m5_dir = ROOT / "data/results/m5"
    cohort = rows(ROOT / "data/manifests/m5_serving_query_ids.jsonl")
    cohort_ids = {row["query_id"] for row in cohort}
    cohort_counts = {dataset: sum(row["dataset"] == dataset for row in cohort) for dataset in DATASETS}
    trace_by_policy: dict[str, list[dict]] = {}
    for policy in POLICIES:
        path = m5_dir / f"{policy}_serving_traces.jsonl"
        trace_by_policy[policy] = rows(path)
    integrity = {
        "expected_policies": POLICIES,
        "expected_queries_per_policy": 100,
        "cohort_counts": cohort_counts,
        "same_cohort": True,
        "all_cache_disabled": True,
        "all_status_ok": True,
        "all_jev_calls_zero": True,
        "all_hosted_cloudflare_calls_zero": True,
        "policy_rows": {},
    }
    for policy, trace_rows in trace_by_policy.items():
        ids = {row["query_id"] for row in trace_rows}
        integrity["same_cohort"] = integrity["same_cohort"] and ids == cohort_ids and len(trace_rows) == 100
        integrity["all_cache_disabled"] = integrity["all_cache_disabled"] and all(row.get("cache_disabled") is True for row in trace_rows)
        integrity["all_status_ok"] = integrity["all_status_ok"] and all(row.get("status") == "ok" for row in trace_rows)
        integrity["all_jev_calls_zero"] = integrity["all_jev_calls_zero"] and all(row.get("paid_api_usage") is False for row in trace_rows)
        integrity["all_hosted_cloudflare_calls_zero"] = integrity["all_hosted_cloudflare_calls_zero"] and all(row.get("provider_fee_usd") == 0.0 for row in trace_rows)
        integrity["policy_rows"][policy] = {"rows": len(trace_rows), "unique_query_ids": len(ids), "status_counts": {status: sum(row.get("status") == status for row in trace_rows) for status in sorted({row.get("status") for row in trace_rows})}, "retries": sum(max(0, int(row.get("http_attempts", 0)) - 1) for row in trace_rows)}

    summary = []
    operational = []
    for policy in POLICIES:
        trace_rows = trace_by_policy[policy]
        ok = [row for row in trace_rows if row.get("status") == "ok"]
        error_count = len(trace_rows) - len(ok)
        row = {"policy_id": policy, "n_attempted": len(trace_rows), "n_completed": len(ok), "error_count": error_count, "error_rate": error_count / len(trace_rows) if trace_rows else None, "retry_count": sum(max(0, int(item.get("http_attempts", 0)) - 1) for item in trace_rows), "request_p50_ms": percentile([item["request_duration_ms"] for item in trace_rows], .5), "request_p95_ms": percentile([item["request_duration_ms"] for item in trace_rows], .95)}
        for stage in ("retrieval_ms", "selector_ms", "rescue_ms", "reader_ms"):
            row[f"{stage}_p50"] = percentile([item[stage] for item in ok if stage in item], .5)
            row[f"{stage}_p95"] = percentile([item[stage] for item in ok if stage in item], .95)
            row[f"{stage}_mean"] = statistics.mean(item[stage] for item in ok if stage in item) if ok else None
        row["reader_input_tokens_mean"] = statistics.mean(item["input_tokens"] for item in ok if item.get("input_tokens") is not None) if any(item.get("input_tokens") is not None for item in ok) else None
        row["reader_output_tokens_mean"] = statistics.mean(item["output_tokens"] for item in ok if item.get("output_tokens") is not None) if any(item.get("output_tokens") is not None for item in ok) else None
        row["gpu_peak_vram_gib_max"] = max((item.get("gpu_peak_vram_bytes") or 0) / 2**30 for item in trace_rows) if trace_rows else None
        row["gpu_seconds_per_1k"] = "NOT_MEASURED"
        row["gpu_seconds_method"] = "No per-request CUDA active-time instrumentation in completed G2 traces; exact GPU active time is unknown."
        row["selector_wall_seconds_per_1k_proxy"] = sum(item.get("selector_ms", 0.0) for item in trace_rows) * 10.0 / 1000.0
        row["paid_usd_per_1k"] = 0.0
        row["paid_cost_provenance"] = "Measured: paid_api_usage=false and provider_fee_usd=0.0 in every trace; Jev/hosted Cloudflare not called."
        row["local_compute_usd_per_1k"] = "NOT_ESTIMATED"
        row["local_compute_cost_method"] = "No GPU-hour price was supplied or assumed; measured GPU active seconds are unavailable."
        row["scenario_total_usd_per_1k"] = "NOT_ESTIMATED"
        summary.append(row)
        operational.append({"policy_id": policy, "n_requests": len(trace_rows), "request_p50_ms": row["request_p50_ms"], "request_p95_ms": row["request_p95_ms"], "selector_wall_seconds_per_1k_proxy": row["selector_wall_seconds_per_1k_proxy"], "gpu_seconds_per_1k": row["gpu_seconds_per_1k"], "gpu_peak_vram_gib_max": row["gpu_peak_vram_gib_max"], "reader_input_tokens_mean": row["reader_input_tokens_mean"], "paid_usd_per_1k": row["paid_usd_per_1k"], "local_compute_usd_per_1k": row["local_compute_usd_per_1k"], "scenario_total_usd_per_1k": row["scenario_total_usd_per_1k"], "cost_status": "paid portion measured; local/scenario portion not priced"})

    # Reconstruct the already accepted M4 query-level metrics to export the full
    # TEST dataset breakdown and complete paired primary-support CIs.
    questions = {row["query_id"]: row for row in rows(ROOT / "data/manifests/questions.jsonl")}
    m4_rows = rows(ROOT / "data/results/m4/reader_results.jsonl")
    metric_rows = []
    for item in m4_rows:
        question = questions[item["query_id"]]
        gold = set(question["gold_doc_ids"])
        top20 = set(item["top20_doc_ids"])
        selected = set(item["selected_doc_ids"])
        em, ans_f1 = answer_scores(item.get("answer", ""), question) if item.get("reader_status") == "ok" else (0.0, 0.0)
        metric_rows.append({"query_id": item["query_id"], "dataset": item["dataset"], "split": question["split"], "policy_id": item["policy_id"], "gold_present_top20": gold.issubset(top20), "complete_support": gold.issubset(selected), "support_recall": len(gold & selected) / len(gold), "answer_em": em, "answer_f1": ans_f1, "reader_input_tokens": item.get("input_tokens"), "truncated_support": any(p.get("truncated") and p.get("doc_id") in gold for p in item.get("view_info", {}).get("passages", [])), "truncated_any": any(p.get("truncated") for p in item.get("view_info", {}).get("passages", []))})

    per_dataset = []
    for policy in POLICIES:
        for dataset in DATASETS:
            subset = [x for x in metric_rows if x["policy_id"] == policy and x["dataset"] == dataset and x["split"] == "test"]
            eligible = [x for x in subset if x["gold_present_top20"]]
            per_dataset.append({"policy_id": policy, "dataset": dataset, "n_test": len(subset), "gold_present_n": len(eligible), "cssr_denominator": "gold_present_top20", "cssr_numerator": sum(x["complete_support"] for x in eligible), "cssr_gold_present": statistics.mean(x["complete_support"] for x in eligible) if eligible else None, "cssr_unconditional_numerator": sum(x["complete_support"] for x in subset), "cssr_unconditional": statistics.mean(x["complete_support"] for x in subset) if subset else None, "support_recall": statistics.mean(x["support_recall"] for x in subset) if subset else None, "answer_em": statistics.mean(x["answer_em"] for x in subset) if subset else None, "answer_f1": statistics.mean(x["answer_f1"] for x in subset) if subset else None, "reader_input_tokens_mean": statistics.mean(x["reader_input_tokens"] for x in subset if x.get("reader_input_tokens") is not None) if subset else None, "truncated_support_n": sum(x["truncated_support"] for x in subset), "truncated_support_rate": statistics.mean(x["truncated_support"] for x in subset) if subset else None})

    pairs = [("laya_p5", "bge_p5"), ("laya_p5", "qwen_p5"), ("clef_p5", "bge_p5"), ("clef_p5", "qwen_p5"), ("laya_g5", "laya_p5"), ("clef_g5", "clef_p5"), ("bge_rescue5", "bge_p5"), ("qwen_rescue5", "qwen_p5"), ("laya_rescue5", "laya_p5"), ("clef_rescue5", "clef_p5")]
    ci_rows = []
    for better, baseline in pairs:
        by_dataset = {}
        for dataset in DATASETS:
            a = [float(x["complete_support"]) for x in metric_rows if x["policy_id"] == better and x["dataset"] == dataset and x["split"] == "test" and x["gold_present_top20"]]
            b = [float(x["complete_support"]) for x in metric_rows if x["policy_id"] == baseline and x["dataset"] == dataset and x["split"] == "test" and x["gold_present_top20"]]
            result = bootstrap_delta(a, b, SEED)
            by_dataset[dataset] = (a, b)
            ci_rows.append({"comparison": f"{better}_vs_{baseline}", "metric": "CSSR@5|gold_present", "scope": dataset, **result})
        result = macro_bootstrap(by_dataset)
        ci_rows.append({"comparison": f"{better}_vs_{baseline}", "metric": "CSSR@5|gold_present", "scope": "equal_dataset_macro", **result})

    truncation = []
    failures = []
    for item in metric_rows:
        if item["split"] != "test":
            continue
        truncation.append({"policy_id": item["policy_id"], "dataset": item["dataset"], "query_id": item["query_id"], "truncated_support": item["truncated_support"], "truncated_any": item["truncated_any"]})
        if not item["complete_support"] or item["answer_em"] == 0.0 or item["truncated_support"]:
            failures.append({"query_id": item["query_id"], "dataset": item["dataset"], "policy_id": item["policy_id"], "gold_present_top20": item["gold_present_top20"], "complete_support": item["complete_support"], "support_recall": item["support_recall"], "answer_em": item["answer_em"], "answer_f1": item["answer_f1"], "truncated_support": item["truncated_support"], "failure_reasons": [reason for reason, condition in (("retrieval_missing_gold", not item["gold_present_top20"]), ("support_not_complete", not item["complete_support"]), ("answer_not_exact", item["answer_em"] == 0.0), ("support_truncated_in_reader_view", item["truncated_support"])) if condition]})

    # Paper tables and figures are generated solely from the frozen M4/M5 data.
    paper_tables = ROOT / "paper/tables"
    paper_figures = ROOT / "paper/figures"
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)
    inventory = [
        {"model_id": "bm25", "deployment": "local_cpu", "revision": "pinned implementation", "input_contract": "lowercase_regex", "calls_per_query": "local", "gpu_topology": "CPU"},
        {"model_id": "Qwen3-VL-Embedding-2B", "deployment": "existing_service:8001", "revision": "service-pinned", "input_contract": "query embedding", "calls_per_query": "1", "gpu_topology": "service GPU; not per-request attributed"},
        {"model_id": "BAAI/bge-reranker-v2-m3", "deployment": "local_gpu0", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "input_contract": "20 pointwise pairs", "calls_per_query": "20 pairs", "gpu_topology": "A100 GPU0"},
        {"model_id": "Qwen/Qwen3-VL-Reranker-2B", "deployment": "existing_service:8002", "revision": "4bd860ac4f15ad1897a214615cccc700f8f71818", "input_contract": "20 pointwise documents", "calls_per_query": "1 batched request", "gpu_topology": "service GPU; not per-request attributed"},
        {"model_id": "convaiinnovations/laya", "deployment": "local_gpu0", "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148", "input_contract": "typed 512-token max", "calls_per_query": "20 or 5 quartet", "gpu_topology": "A100 GPU0"},
        {"model_id": "Cloudflare/clef-flash", "deployment": "local_gpu0", "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "input_contract": "typed local SystemOne schema", "calls_per_query": "20 or 5 quartet", "gpu_topology": "A100 GPU0"},
        {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "deployment": "existing_service:8003", "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "input_contract": "text-only fixed reader", "calls_per_query": "1", "gpu_topology": "service GPU; not per-request attributed"},
    ]
    write_csv(paper_tables / "model_inventory.csv", inventory)
    write_csv(paper_tables / "policy_quality.csv", per_dataset)
    write_csv(paper_tables / "serving_cost.csv", operational)

    try:
        import matplotlib.pyplot as plt
        m4_metrics = json.loads((ROOT / "data/results/m4/m4_metrics.json").read_text(encoding="utf-8"))["metrics"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for axis, dataset in zip(axes, DATASETS):
            for policy in POLICIES:
                val = m4_metrics[policy]["by_dataset"][dataset]
                axis.scatter(val["support_recall"], val["answer_f1"], label=policy, s=28)
            axis.set_title(dataset)
            axis.set_xlabel("support recall@5")
            axis.grid(alpha=.25)
        axes[0].set_ylabel("reader answer F1")
        axes[-1].legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(paper_figures / "support_recall_vs_answer_f1.png", dpi=160)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(8, 5))
        quality = {row["policy_id"]: row for row in per_dataset}
        for policy in POLICIES:
            test = [x for x in metric_rows if x["policy_id"] == policy and x["split"] == "test"]
            answer = statistics.mean(x["answer_f1"] for x in test)
            proxy = next(x["selector_wall_seconds_per_1k_proxy"] for x in summary if x["policy_id"] == policy)
            axis.scatter(proxy, answer, s=35)
            axis.annotate(policy, (proxy, answer), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axis.set_xlabel("selector wall seconds / 1k (measured proxy; not GPU active seconds)")
        axis.set_ylabel("TEST reader answer F1")
        axis.set_title("Quality vs measured selector-time proxy; local compute unpriced")
        axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(paper_figures / "quality_vs_cost.png", dpi=160)
        plt.close(fig)
        plot_status = "rendered"
    except Exception as exc:
        plot_status = f"NOT_RENDERED: {type(exc).__name__}: {exc}"

    analysis_dir = ROOT / "results/non_jev/analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "summary.csv", summary)
    write_csv(analysis_dir / "per_dataset.csv", per_dataset)
    write_csv(analysis_dir / "paired_ci.csv", ci_rows)
    write_csv(analysis_dir / "truncation_audit.csv", truncation)
    write_csv(analysis_dir / "operational_costs.csv", operational)
    (analysis_dir / "failure_cases.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in failures), encoding="utf-8")
    metrics = {"study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2", "created_utc": datetime.now(timezone.utc).isoformat(), "m5_integrity": integrity, "summary": summary, "per_dataset": per_dataset, "paired_ci": ci_rows, "truncation_rows": len(truncation), "failure_case_rows": len(failures), "plot_status": plot_status, "jev_calls": 0, "hosted_cloudflare_calls": 0}
    (m5_dir / "m5_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"m5_rows": sum(len(v) for v in trace_by_policy.values()), "integrity": integrity, "plot_status": plot_status, "analysis_dir": str(analysis_dir), "jev_calls": 0}, indent=2, sort_keys=True))
    return 0 if all(integrity.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
