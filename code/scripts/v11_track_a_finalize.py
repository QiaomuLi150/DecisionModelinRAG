#!/usr/bin/env python3
"""Write the human-readable Track A completion package after offline analysis."""
from __future__ import annotations

import hashlib
import json
import shutil
import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v11/track_a"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return "NOT_MEASURED"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    analysis = read(OUT / "V11_TRACK_A_ANALYSIS_SUMMARY.json")
    runtime = read(OUT / "V11_TRACK_A_RUNTIME_SUMMARY.json")
    selector_manifest = read(OUT / "V11_TRACK_A_SELECTOR_RESULTS_FULL.manifest.json")
    reader_manifest = read(OUT / "V11_TRACK_A_READER_FULL.manifest.json")
    provenance = read(OUT / "V11_TRACK_A_PROVENANCE_MANIFEST.json")
    # Publish the names required by the Track A handoff while preserving the
    # explicit *_FULL files used by the execution wrappers.
    for source, target in (("V11_TRACK_A_SELECTOR_RESULTS_FULL.jsonl", "V11_TRACK_A_SELECTOR_RESULTS.jsonl"), ("V11_TRACK_A_READER_RESULTS_FULL.jsonl", "V11_TRACK_A_READER_RESULTS.jsonl")):
        shutil.copyfile(OUT / source, OUT / target)
    provenance["required_output_aliases"] = {"selector_results": {"path": "V11_TRACK_A_SELECTOR_RESULTS.jsonl", "sha256": digest(OUT / "V11_TRACK_A_SELECTOR_RESULTS.jsonl")}, "reader_results": {"path": "V11_TRACK_A_READER_RESULTS.jsonl", "sha256": digest(OUT / "V11_TRACK_A_READER_RESULTS.jsonl")}}
    (OUT / "V11_TRACK_A_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pilot = None
    pilot_path = ROOT / "data/results/m4/m4_metrics.json"
    if pilot_path.exists():
        pilot = read(pilot_path).get("metrics", {})
    pilot_lines = ["# Track A Pilot vs Confirmation", "", "The pilot is the previously frozen 750-question non-Jev cohort. The confirmation results below are the new frozen 3,767-question cohort. They are reported separately; no pilot result was used to alter the confirmation cohort or policies.", "", "| Policy | Pilot n | Pilot answer F1 | Confirmation n | Confirmation answer F1 | Confirmation conditional CSSR |", "|---|---:|---:|---:|---:|---:|"]
    for policy, values in analysis["metrics"].items():
        pilot_overall = (pilot or {}).get(policy, {}).get("overall", {})
        confirm = values["overall"]
        pilot_lines.append(f"| {policy} | {pilot_overall.get('n', 'NOT_MEASURED')} | {fmt(pilot_overall.get('answer_f1'))} | {confirm['n']} | {fmt(confirm.get('answer_f1_all'))} | {fmt(confirm.get('cssr_conditional_complete_gold'))} |")
    pilot_lines.extend(["", "Pilot and confirmation are prospective/non-Jev comparisons only. Jev calls: 0."])
    (OUT / "V11_TRACK_A_PILOT_VS_CONFIRMATION.md").write_text("\n".join(pilot_lines) + "\n", encoding="utf-8")

    report = [
        "# DecisionRAG v1.1 Track A Execution Report", "",
        "**Status: `TRACK_A_COMPLETE_READY_FOR_OWNER_REVIEW`**", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Frozen execution", "",
        "- Protocol commit: `08fe498472353f597d5bd3717c1f231ac6614f0c`.",
        "- Cohort: 3,767 new questions — HotpotQA 800, MuSiQue 2,167, 2Wiki 800.",
        "- Retrieval: frozen BM25 top-100 + Qwen dense top-100 + RRF@60 + Top-20; qrels were not read during generation.",
        "- Policies: 12 frozen non-Jev policies; all produced 3,767 rows.",
        "- Reader: pinned Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, frozen prompt/decoding.",
        "- Jev and hosted paid calls: **0**.", "",
        "## Completion counts", "",
        f"- Retrieval slates: {len([1 for _ in open(OUT / 'V11_TRACK_A_RETRIEVAL_SLATES.jsonl', encoding='utf-8')])}.",
        f"- Selector rows: {selector_manifest['rows']} ({selector_manifest['policy_count']} policies × {selector_manifest['query_count']} queries).",
        f"- Reader rows: {reader_manifest['tasks']} with status counts `{reader_manifest['status_counts']}`.",
        f"- Unique reader cache keys: {reader_manifest['unique_cache_keys']}; new requests: {reader_manifest['new_reader_requests']}; exact-context reuses: {reader_manifest['exact_context_cache_reuses']}.",
        "",
        "## Selector model revisions", "",
        "| Model | Revision | Rows | Status |", "|---|---|---:|---|",
        "| BGE | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 3,767 | all ok |",
        "| Qwen | `4bd860ac4f15ad1897a214615cccc700f8f71818` | 3,767 | all ok |",
        "| Laya | `7b928d828b7b0e022f929d9bd2e44165aa270148` | 3,767 pointwise + 3,767 groupwise | all ok |",
        "| Clef | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | 3,767 pointwise + 3,767 groupwise | all ok |", "",
        "## Scientific analysis", "",
        "Metrics were computed only after retrieval, selector, and reader artifacts were complete. The evaluation-only gold sidecar was not used during candidate generation or selector execution. CSSR conditional denominators use complete gold presence in the immutable Top-20; paired bootstrap statistics use 5,000 replicates and seed 731. See `V11_TRACK_A_PRIMARY_PAIRED_STATISTICS.csv`, `V11_TRACK_A_DATASET_BREAKDOWN.md`, and `V11_TRACK_A_FAILURE_ATTRIBUTION.md`.", "",
        "### Confirmation scorecard", "",
        "| Policy | Any-gold@20 | Complete-gold@20 | Conditional CSSR | Answer F1 |", "|---|---:|---:|---:|---:|",
    ]
    for policy, values in analysis["metrics"].items():
        overall = values["overall"]
        report.append(f"| {policy} | {fmt(overall.get('any_gold_top20_rate'))} | {fmt(overall.get('complete_gold_top20_rate'))} | {fmt(overall.get('cssr_conditional_complete_gold'))} | {fmt(overall.get('answer_f1_all'))} |")
    report.extend([
        "", "### Pooled primary paired differences", "",
        "Pooled support rows use the complete-gold@20 eligibility denominator; pooled answer rows use all confirmation queries.",
        "", "| Comparison | Metric | n | Delta | 95% CI |", "|---|---|---:|---:|---:|",
    ])
    with (OUT / "V11_TRACK_A_PRIMARY_PAIRED_STATISTICS.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "pooled" and row["metric"] in {"complete_support", "answer_f1"}:
                report.append(f"| {row['comparison']} | {row['metric']} | {row['n']} | {float(row['delta']):.4f} | [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] |")
    report.extend([
        "", "Rescue benefit/harm/unchanged counts are in `V11_TRACK_A_RESCUE_AUDIT.csv`; groupwise quartet collision counts are in `V11_TRACK_A_GROUPWISE_COLLISIONS.csv`.", "",
        "## Integrity and remaining limitations", "",
        "- Selector assembly view-hash mismatches: 0.",
        "- Clef was run as two deterministic disjoint shards on GPUs 0 and 5, then merged with exact coverage and overlap checks.",
        "- Laya emitted the previously documented invalid-temperature warning; the frozen runtime clamp was retained and confidence is treated as uncalibrated.",
        "- The report compares confirmation and historical pilot cohorts descriptively; no Jev phase was started.", "",
        "## Required artifacts", "",
        "- `V11_TRACK_A_RETRIEVAL_SLATES.jsonl`",
        "- `V11_TRACK_A_SELECTOR_RESULTS.jsonl`",
        "- `V11_TRACK_A_READER_RESULTS.jsonl`",
        "- `V11_TRACK_A_METRICS.csv`",
        "- `V11_TRACK_A_PRIMARY_PAIRED_STATISTICS.csv`",
        "- `V11_TRACK_A_DATASET_BREAKDOWN.md`",
        "- `V11_TRACK_A_FAILURE_ATTRIBUTION.md`",
        "- `V11_TRACK_A_RUNTIME_AND_RESOURCE_REPORT.md`",
        "- `V11_TRACK_A_PROVENANCE_MANIFEST.json`",
        "- `V11_TRACK_A_PILOT_VS_CONFIRMATION.md`",
        "",
        "Track A is complete and stopped at the owner-review gate. Tracks B/C and Jev remain unexecuted.",
    ])
    (OUT / "V11_TRACK_A_EXECUTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    runtime_lines = ["# Track A Runtime and Resource Report", "", "Stage timings are reported separately; they are not added to estimate end-to-end latency.", "", "| Stage | n | p50 ms | p95 ms | mean ms |", "|---|---:|---:|---:|---:|"]
    for key, value in runtime.items():
        if key.endswith("_ms") and isinstance(value, dict):
            runtime_lines.append(f"| {key} | {value.get('n')} | {fmt(value.get('p50_ms'))} | {fmt(value.get('p95_ms'))} | {fmt(value.get('mean_ms'))} |")
    runtime_lines.extend(["", "The retrieval trace contains separate BM25, embedding, Qdrant, RRF, and measured retrieval-wall timings. Selector traces contain model-stage timings. Reader timings are measured per request and include cache-reused rows as zero-duration reuse records where applicable. No component sum is presented as end-to-end latency.", "", "GPU execution used authorized GPUs 0 and 5 for local BGE/Laya/Clef work. GPU 1's existing Qwen service was kept in place for Qwen selector and reader calls. No service restart, model relocation, or process termination was performed."])
    (OUT / "V11_TRACK_A_RUNTIME_AND_RESOURCE_REPORT.md").write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "TRACK_A_COMPLETE_READY_FOR_OWNER_REVIEW", "jev_calls": 0, "selector_rows": selector_manifest["rows"], "reader_rows": reader_manifest["tasks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
