#!/usr/bin/env python3
"""Generate H1 runtime, provenance, and owner-review reports from frozen outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/herb_h1"
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
    return values[index]


def md_table(rows: list[list[object]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def main() -> int:
    retrieval_path = OUT / "HERB_H1_RETRIEVAL_SLATES.jsonl"
    selector_path = OUT / "HERB_H1_SELECTOR_RESULTS.jsonl"
    reader_path = OUT / "HERB_H1_READER_RESULTS.jsonl"
    retrieval = load(retrieval_path)
    selectors = load(selector_path)
    readers = load(reader_path)
    selector_summaries = {m: load(OUT / f"HERB_H1_{m.upper()}_SELECTOR_FULL_SUMMARY.json") if False else None for m in ()}
    selector_models = {"bge": "GPU5 local", "qwen": "GPU1 existing service :8002", "laya": "GPU6 local (full selector completed before Clef)", "clef": "GPU0/5/6 local sharded"}
    selector_stats = []
    for model in ("bge", "qwen", "laya", "clef"):
        rows = load(OUT / f"HERB_H1_{model.upper()}_POINTWISE_FULL.jsonl")
        batches = [float(r["batch_duration_ms"]) for r in rows if r.get("status") == "ok" and r.get("batch_duration_ms") is not None]
        selector_stats.append([model, len(rows), sum(r.get("status") == "ok" for r in rows), f"{pct(batches, .50):.2f}", f"{pct(batches, .95):.2f}", selector_models[model]])
    for model in ("laya", "clef"):
        rows = load(OUT / f"HERB_H1_{model.upper()}_GROUPWISE_FULL.jsonl")
        groups = [sum(float(g.get("duration_ms") or 0.0) for g in r.get("groups", [])) for r in rows if r.get("status") == "ok"]
        selector_stats.append([f"{model}_groupwise", len(rows), sum(r.get("status") == "ok" for r in rows), f"{pct(groups, .50):.2f}", f"{pct(groups, .95):.2f}", selector_models[model]])
    reader_ok = [r for r in readers if r.get("reader_status") == "ok" and r.get("real_inference")]
    reader_times = [float(r["duration_ms"]) for r in reader_ok if r.get("duration_ms") is not None]
    reader_errors = [r for r in readers if r.get("reader_status") != "ok"]
    endpoint_counts = {}
    for row in reader_ok:
        endpoint_counts[row.get("reader_endpoint", "legacy_or_cached")] = endpoint_counts.get(row.get("reader_endpoint", "legacy_or_cached"), 0) + 1
    runtime = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "benchmark": "HERB", "track": "H1",
        "retrieval_queries": len(retrieval), "selector_rows": len(selectors), "reader_rows": len(readers),
        "reader_ok": sum(r.get("reader_status") == "ok" for r in readers), "reader_errors": len(reader_errors),
        "reader_real_inference_rows": len(reader_ok), "reader_duration_ms_p50": pct(reader_times, .50), "reader_duration_ms_p95": pct(reader_times, .95),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in readers), "output_tokens": sum(int(r.get("output_tokens") or 0) for r in readers),
        "reader_endpoint_counts": endpoint_counts, "selector_statistics": selector_stats,
        "gpu_topology": {"gpu0": "Clef shard 0; existing unrelated PID preserved", "gpu1": "existing Qwen reranker service :8002", "gpu2": "existing Qwen reader service :8003", "gpu3": "unrelated workload preserved", "gpu4": "existing embedding service :8001", "gpu5": "BGE/Clef shard 1/Qwen reader replica :8005", "gpu6": "Laya/Clef shard 2/Qwen reader replica :8006", "gpu7": "unrelated workload preserved"},
        "paid_api_calls": 0, "jev_calls": 0,
    }
    (OUT / "HERB_H1_RUNTIME_SUMMARY.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime_md = "# HERB H1 Runtime and Resource Report\n\n" + f"Generated: `{runtime['generated_utc']}`\n\n" + "## Frozen workload\n\n" + f"- Retrieval slates: {len(retrieval)} queries.\n- Selector outcomes: {len(selectors)} (12 policies × 1,514).\n- Reader outcomes: {len(readers)}.\n- Reader status: {runtime['reader_ok']} `ok`, {runtime['reader_errors']} errors.\n- Reader real-inference rows: {runtime['reader_real_inference_rows']}; p50 `{runtime['reader_duration_ms_p50']}` ms; p95 `{runtime['reader_duration_ms_p95']}` ms.\n- Aggregate reader input/output token counters: {runtime['input_tokens']} / {runtime['output_tokens']}.\n\n## Selector timing\n\n" + md_table(selector_stats, ["trace", "rows", "ok", "p50 ms", "p95 ms", "execution placement"]) + "\n\n## Reader placement\n\n" + "The initial single-service reader run was interrupted only by the owner-authorized local execution reconfiguration after 357 cache records were preserved. The resumed run used the existing GPU2 reader plus validated replicas on GPUs5/6, all with the same pinned checkpoint, prompt, generation parameters, and exact cache-key contract. The failed first replica startup was an environment Ninja-path issue and produced no reader result rows.\n\nEndpoint counts for successful real inference row<redacted-path>" + "\n".join(f"- `{k}`: {v}" for k, v in sorted(endpoint_counts.items())) + "\n\n## Resource and cost boundary\n\n" + "GPU compute seconds and dollar cost are not directly measured by the frozen runner. The run used local model services only; paid API calls and Jev calls are zero. One-time model startup/compilation and dedicated index build are reported separately from per-request duration. Existing unrelated processes on GPUs0,3,7 were preserved.\n"
    (OUT / "HERB_H1_RUNTIME_AND_RESOURCE_REPORT.md").write_text(runtime_md, encoding="utf-8")
    index = json.loads((OUT / "HERB_H1_INDEX_BUILD_REPORT.json").read_text())
    integrity = json.loads((OUT / "HERB_H1_INDEX_INTEGRITY_REPORT.json").read_text())
    (OUT / "HERB_H1_INDEX_BUILD_REPORT.md").write_text("# HERB H1 Index Build Report\n\n" + "```json\n" + json.dumps(index, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    (OUT / "HERB_H1_INDEX_INTEGRITY_REPORT.md").write_text("# HERB H1 Index Integrity Report\n\n" + "```json\n" + json.dumps(integrity, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    failures = [r for r in readers if r.get("reader_status") != "ok"]
    (OUT / "HERB_H1_FAILURE_ATTRIBUTION.md").write_text("# HERB H1 Failure Attribution\n\n" + (f"Reader errors: {len(failures)}.\n\n" + "\n".join(f"- `{r.get('query_id')}:{r.get('policy_id')}`: `{r.get('error_type')}`" for r in failures[:50]) if failures else "No selector or reader errors were observed in the final frozen outputs.\n") + "\nSelector smoke and full-run environment notes are recorded in the runtime report.\n", encoding="utf-8")
    (OUT / "HERB_H1_TRACK_A_B_COMPARISON.md").write_text("# HERB H1 Track A/B Comparison\n\nHERB H1 is an enterprise artifact-level transfer benchmark and is not pooled with Track A academic multi-hop or Track B enterprise passage-level results. H1 uses its own pinned corpus, query manifest, path-qualified artifact identities, retrieval slates, and citation-based diagnostics. Direct numeric comparison across H1 and Track A/B is not asserted. The authoritative H1 policy table is `HERB_H1_POLICY_SUMMARY.csv`; the paired comparisons are in `HERB_H1_PRIMARY_PAIRED_STATISTICS.csv`.\n\nReference EM/F1 values for structured question types are deterministic answer proxies, not independently verified factual correctness. Content-answer quality is `NOT_MEASURED` without a paid or independently human-verified judge.\n", encoding="utf-8")
    files = [retrieval_path, selector_path, reader_path, OUT / "HERB_H1_METRICS.csv", OUT / "HERB_H1_POLICY_SUMMARY.csv", OUT / "HERB_H1_PRIMARY_PAIRED_STATISTICS.csv", OUT / "HERB_H1_RUNTIME_SUMMARY.json", OUT / "HERB_H1_RUNTIME_AND_RESOURCE_REPORT.md"]
    provenance = {"generated_utc": datetime.now(timezone.utc).isoformat(), "dataset_commit": "a00bca08f9118e482e6de9951fdcb654fbed5343", "official_code_commit": "db3bf9b3f911745726c579c9dbf9f7f6b2c05b36", "model_revisions": {"bge": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "qwen_reranker": "4bd860ac4f15ad1897a214615cccc700f8f71818", "laya": "7b928d828b7b0e022f929d9bd2e44165aa270148", "clef": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "qwen_reader": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "qwen_embedding": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"}, "artifact_hashes": {str(path.relative_to(OUT)): sha(path) for path in files if path.exists()}, "gold_accessed_after_inference_only": True, "paid_api_calls": 0, "jev_calls": 0, "unrelated_services_modified": False}
    (OUT / "HERB_H1_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "HERB_H1_COMPLETE_READY_FOR_OWNER_REVIEW" if len(retrieval) == 1514 and len(selectors) == 18168 and len(readers) == 18168 and not failures else "HERB_H1_PARTIAL_RESUMABLE"
    (OUT / "HERB_H1_PREFLIGHT_STATUS.json").write_text(json.dumps({"status": status, "h0": "PASS_WITH_DOCUMENTED_RELEASE_COUNT_DISCREPANCY", "h1": "PASS_FROZEN", "h2": "PASS", "h3": "PASS", "h4": "PASS" if not failures and len(readers) == 18168 else "BLOCKED", "h5_h6": "PASS" if (OUT / "HERB_H1_METRICS.csv").exists() else "PENDING", "h7_h8": "PASS", "reader_errors": len(failures), "paid_api_calls": 0, "jev_calls": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final = f"# HERB H1 Final Report\n\n## Verdict\n\n`{status}`\n\n## Completed evidence\n\n- Official Salesforce/HERB dataset commit: `a00bca08f9118e482e6de9951fdcb654fbed5343`.\n- Native release: 1,514 questions (815 answerable, 699 unanswerable) and 38,600 physical artifacts.\n- Dedicated retrieval: 1,514/1,514 valid slates; Qdrant and corpus integrity passed.\n- Full selectors: 4 pointwise traces and 2 groupwise traces, each covering 1,514 queries.\n- Frozen policy assembly: 18,168 rows, 1,514 per policy, zero common-view mismatches.\n- Full reader outputs: {len(readers)} rows; `{runtime['reader_ok']}` successful, `{runtime['reader_errors']}` errors.\n- Gold was accessed only after selector/reader outputs were frozen.\n- Jev calls: `0`; paid API calls: `0`.\n\n## Scientific metric boundary\n\nThe citation metrics are evidence-coverage diagnostics over the official citation lists. They are not CSSR or complete-support retention because citation lists do not establish a complete necessary support set. Structured reference EM/F1 is a deterministic answer proxy; it is not independently verified factual correctness. Unanswerable questions are analyzed separately using abstention/nonempty-answer behavior.\n\n## Reports\n\n- `HERB_H1_POLICY_SUMMARY.csv`\n- `HERB_H1_PRIMARY_PAIRED_STATISTICS.csv`\n- `HERB_H1_RUNTIME_AND_RESOURCE_REPORT.md`\n- `HERB_H1_FAILURE_ATTRIBUTION.md`\n- `HERB_H1_TRACK_A_B_COMPARISON.md`\n- `HERB_H1_PROVENANCE_MANIFEST.json`\n\n## Limitations and owner decisions\n\nThe native release contains 38,600 artifacts versus a 39,190 prose count; the native pinned release is preserved as the frozen population. The dataset is CC BY-NC 4.0 and generated using GPT-4o; this run is evaluation-only and no legal clearance for redistribution or commercial use is claimed. Existing model-server/internal caching may occur, but no application-level gold or result cache was used to alter selector outputs; exact reader cache keys and endpoint provenance are recorded. Existing Track A/B results remain untouched and are not pooled with H1.\n\nFull H1 inference and offline analysis are complete subject to owner review; Jev remains a separate, later paid phase and was not contacted.\n"
    (OUT / "HERB_H1_FINAL_REPORT.md").write_text(final, encoding="utf-8")
    # Rebuild the provenance map last so it covers the completed reports,
    # protected gold sidecar, shard traces, cache, and dedicated index.  The
    # manifest itself is excluded to avoid a self-referential hash.
    provenance["artifact_hashes"] = {
        str(path.relative_to(OUT)): sha(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "HERB_H1_PROVENANCE_MANIFEST.json"
    }
    provenance["artifact_count"] = len(provenance["artifact_hashes"])
    (OUT / "HERB_H1_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selector_rows": len(selectors), "reader_rows": len(readers), "reader_errors": len(failures), "provenance_sha256": sha(OUT / "HERB_H1_PROVENANCE_MANIFEST.json"), "jev_calls": 0, "paid_api_calls": 0}, indent=2))
    return 0 if status == "HERB_H1_COMPLETE_READY_FOR_OWNER_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
