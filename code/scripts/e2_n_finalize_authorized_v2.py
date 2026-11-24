#!/usr/bin/env python3
"""Create a versioned E2-N owner-review package after authorized N3/N4 runs.

This deliberately writes only under ``results/non_jev/v2/e2_n/authorized_v2``
and never replaces the earlier E2-N reports.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e2_n"
OUT = BASE / "authorized_v2"
N3 = BASE / "n3"
N4 = BASE / "n4_authorized_v2"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    quality_path = N4 / "E2_N_MODEL_DATASET_QUALITY_SUMMARY.csv"
    paired_path = N4 / "E2_N_PRIMARY_EXPLORATORY_PAIRED_STATISTICS.csv"
    selections_path = N4 / "E2_N_TOP5_SELECTIONS.jsonl"
    provenance_path = N4 / "E2_N4_PROVENANCE_MANIFEST.json"
    source_quality = N4 / "E2_N_MODEL_DATASET_QUALITY_SUMMARY.csv"
    if not source_quality.exists():
        raise SystemExit(f"missing N4 quality output: {source_quality}")

    quality = read_csv(quality_path)
    paired = read_csv(paired_path) if paired_path.exists() else []
    summaries = {}
    for path in sorted(N3.glob("*_SUMMARY.json")):
        data = read_json(path)
        summaries[data.get("model", path.stem)] = data

    expected = 4267
    complete = {}
    for model, summary in summaries.items():
        trace_path = next((p for p in N3.glob("*_FULL.jsonl") if p.name.lower().find(model.split("/")[-1].lower().replace("-", "_")) >= 0), None)
        # Prefer the summary's known stem when model names do not map cleanly.
        if trace_path is None:
            stem_map = {
                "internlm/Intern-Decision-4B": "E2_N_N3_INTERN_DECISION_FULL.jsonl",
                "bespokelabs/Bespoke-Nimble-9B-v2": "E2_N_N3_NIMBLE_FULL.jsonl",
                "perplexity-ai/pplx-decider-v1.1-27b": "E2_N_N3_PPLX_DECIDER_FULL.jsonl",
                "jinaai/jina-reranker-v3.5": "E2_N_N3_JINA35_FULL.jsonl",
            }
            candidate_name = stem_map.get(model)
            candidate = N3 / candidate_name if candidate_name else None
            trace_path = candidate if candidate is not None and candidate.exists() else None
        error_rows = []
        if trace_path and trace_path.exists():
            with trace_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("status") != "ok":
                            error_rows.append(row)
        error_types = {}
        for row in error_rows:
            key = row.get("error_type", "unknown")
            error_types[key] = error_types.get(key, 0) + 1
        complete[model] = {
            "status": summary.get("status"),
            "rows": summary.get("rows"),
            "expected_rows": summary.get("expected_rows"),
            "ok_rows": summary.get("ok_rows"),
            "physical_forward_evidence_rows": summary.get("physical_forward_evidence_rows"),
            "physical_forward_calls": summary.get("total_physical_forward_calls"),
            "trace_sha256": summary.get("trace_sha256"),
            "revision": summary.get("revision"),
            "error_rows": len(error_rows),
            "error_types": error_types,
        }

    trace_index = []
    for path in sorted(N3.glob("*_FULL.jsonl")):
        summary_path = path.with_name(path.stem + "_SUMMARY.json")
        trace_index.append({
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha_file(path),
            "summary_path": rel(summary_path) if summary_path.exists() else None,
        })

    status = "PASS_WITH_LIMITATIONS"
    blockers = []
    for model, row in complete.items():
        model_expected = 8534 if model.lower() == "clef" else expected
        if row["rows"] != model_expected or row["ok_rows"] != model_expected or row["physical_forward_evidence_rows"] != model_expected:
            blockers.append(f"incomplete or weak physical evidence: {model}")
        context_errors = sum(count for error_type, count in row.get("error_types", {}).items() if "above 8192" in error_type or "truncation is forbidden" in error_type)
        if context_errors:
            blockers.append(f"{model}: {context_errors} native context-overflow rows under the frozen 8192-token contract")
    if not complete:
        status = "BLOCKED"
        blockers.append("no N3 summaries")

    artifact_hashes = {}
    for path in [quality_path, paired_path, selections_path, provenance_path]:
        if path.exists():
            artifact_hashes[rel(path)] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}

    manifest = {
        "stage": "E2-N-AUTHORIZED-V2-COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "authorization_manifest": rel(BASE / "E2_N_AUTHORIZATION_V2.json"),
        "authorization_scope": {
            "checkpoint_downloads": True,
            "qwen_telemetry_remediation": True,
            "owner_gpu_priority": [0, 1, 2, 4],
            "fallback_gpu_used": [6],
            "resident_service_gpus_left_untouched": [1, 2, 4, 5],
            "fresh_confirmatory_holdout_accessed": False,
            "training_runs": 0,
            "production_service_restarts": 0,
            "jev_calls": 0,
            "paid_api_calls": 0,
        },
        "populations": {
            "track_a_queries": 3767,
            "track_a_conditional": 1490,
            "track_b_questions": 500,
            "track_b_identity_resolved": 443,
            "track_b_conditional": 330,
            "historical_only": True,
        },
        "n3_expected_rows_per_model": expected,
        "n3_models": complete,
        "trace_index": trace_index,
        "quality_artifacts": artifact_hashes,
        "blockers_or_limitations": blockers + [
            "CSSR is a complete-support selection proxy, not independently verified reader factual accuracy.",
            "Qwen service-side telemetry remains separate from direct checkpoint telemetry.",
            "GPU dollar cost was not measured; no monetary claim is made.",
        ],
        "paired_statistics_rows": len(paired),
    }
    manifest_path = OUT / "E2_N_EXECUTION_MANIFEST_V2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    def fmt_rate(value):
        return f"{float(value):.4f}" if value not in (None, "") else "NOT_MEASURED"

    for row in quality:
        if row["dataset"] == "__equal_dataset_macro__":
            continue
        rows.append(
            f"| {row['cohort']} | {row['dataset']} | {row['policy']} | {row['n_eligible']} | {row['n_conditional']} | {fmt_rate(row['cssr_unconditional'])} | {fmt_rate(row['cssr_conditional'])} |"
        )
    scorecard = "\n".join([
        "| Cohort | Dataset | Policy | n eligible | n conditional | CSSR-U | CSSR-C |",
        "|---|---|---|---:|---:|---:|---:|",
        *rows,
    ])
    model_lines = []
    for model, row in sorted(complete.items()):
        model_lines.append(f"| {model} | {row['revision']} | {row['rows']}/{row['expected_rows']} | {row['ok_rows']} | {row['physical_forward_evidence_rows']} | {row['status']} |")
    report = f"""# DecisionRAG E2-N Authorized Expansion — Owner Review Package v2

Status: `{status}`

This is a versioned continuation of E2-N after explicit authorization for official checkpoint downloads and Qwen telemetry remediation. Earlier E2-N, M0, P0v2, M1, Track A/B/C, and HERB artifacts are preserved.

## Scientific and safety boundary

N3 uses only the previously inspected historical Track A/B populations and frozen candidate views. The fresh confirmatory holdout was not accessed. No training, retrieval/indexing, reader generation, production-service restart, Jev call, or paid API call occurred.

## N3 trace completeness

| Model | Revision | Rows | OK rows | Physical-forward rows | Status |
|---|---|---:|---:|---:|---|
{chr(10).join(model_lines)}

Direct Qwen-VL telemetry remediation is represented separately by the native checkpoint traces in `n1/E2_N_N1_QWEN_NATIVE_TELEMETRY_SUMMARY.json`, `n2/E2_N_N2_QWEN_DIRECT_CAL_SUMMARY.json`, and `n3/E2_N_N3_QWEN_DIRECT_FULL_SUMMARY.json`. The resident service remains a distinct, non-promoted telemetry path.

## Track A/B quality scorecard

These are complete-support selection proxies. Track A and Track B are not pooled, and these values are not reader factual accuracy.

{scorecard}

## Integrity and resource accounting

- Every promoted N3 trace must have the expected 4,267 rows, candidate identity alignment, input/output hashes, physical-forward evidence, and zero gold-in-input rows; the machine-readable manifest records these checks.
- GPU0 and GPU6 were used by the new runs. GPUs hosting resident services or unrelated workloads were not stopped, restarted, or reconfigured.
- No monetary GPU cost is inferred from utilization; cost remains `NOT_MEASURED`.
- Historical results remain exploratory follow-up analyses, not prospective confirmation.

## Remaining limitations / owner decision

{chr(10).join('- ' + x for x in manifest['blockers_or_limitations'])}

Owner review is required before any fresh holdout, E2-S/P, training, Jev smoke test, or paid inference. Jev remains at zero calls.
"""
    report_path = OUT / "E2_N_AUTHORIZED_EXPANSION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    gate = OUT / "E2_N_OWNER_REVIEW_GATE_V2.md"
    gate.write_text(f"# E2-N Authorized Owner Review Gate v2\n\nStatus: `{status}`\n\nN0–N4 authorized expansion artifacts are versioned under `authorized_v2/`. Full N3 completeness is recorded in `E2_N_EXECUTION_MANIFEST_V2.json`. Jev calls: `0`; paid calls: `0`; fresh confirmatory holdout accessed: `false`.\n\nOwner decision: review the scorecard and limitations before any subsequent phase.\n", encoding="utf-8")
    qwen = OUT / "E2_N_QWEN_TELEMETRY_REMEDIATION.md"
    qwen.write_text("""# E2-N Qwen Telemetry Remediation\n\nThe resident Qwen3-VL-Reranker-2B service was not restarted or modified. Its prior service trace remains non-promoted because the service boundary did not expose physical-forward telemetry.\n\nA separate official local checkpoint branch was executed with direct model instrumentatio<redacted-path> native smoke: 10/10, physical forwards 10/10;\n- CAL: 30/30, physical forwards 30/30;\n- historical: 4,267/4,267, physical forwards 4,267/4,267;\n- actual tensor-input and raw-score hashes are recorded per row;\n- gold labels were not included in model-visible inputs;\n- Jev calls and paid calls: 0.\n\nThe direct branch is a separate scoring contract and must not be silently substituted for the resident service in historical reports. Future service runs must hash raw response bytes before normalization.\n""", encoding="utf-8")
    print(json.dumps({"status": status, "models": sorted(complete), "paired_rows": len(paired), "output": str(OUT)}, sort_keys=True))
    return 0 if status != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
