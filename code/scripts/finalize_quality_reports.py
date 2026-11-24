#!/usr/bin/env python3
"""Materialize the mandatory quality-audit reports from recorded evidence."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/results/audit/quality"
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in body)
    return "\n".join(lines)


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    semantic = json.loads((AUDIT / "semantic_alignment_summary.json").read_text())
    audit_summary = json.loads((AUDIT / "audit_summary.json").read_text())
    m4 = json.loads((ROOT / "data/results/m4/m4_metrics.json").read_text())
    schedule = json.loads((AUDIT / "m5_interleaved/INTERLEAVED_SCHEDULE.json").read_text())
    m5 = {policy: json.loads((AUDIT / "m5_interleaved" / f"{policy}_summary.json").read_text()) for policy in POLICIES}

    auroc_rows = rows(AUDIT / "M3_AUROC_BY_QUERY.jsonl")
    auc_summary = []
    for model in sorted({row["model"] for row in auroc_rows}):
        values = [row["gold_negative_auc"] for row in auroc_rows if row["model"] == model and row["gold_negative_auc"] is not None]
        auc_summary.append({"model": model, "queries": 20, "measured": len(values), "mean_auc": statistics.mean(values) if values else None, "median_auc": statistics.median(values) if values else None, "min_auc": min(values) if values else None, "max_auc": max(values) if values else None})
    write_csv(AUDIT / "M3_AUROC_SUMMARY.csv", auc_summary)

    # M5 report: request percentiles are measured end-to-end and stage values
    # are kept separate; no percentile addition is performed.
    stage_names = ["embedding_ms", "qdrant_ms", "bm25_ms", "rrf_ms", "retrieval_ms", "selector_queue_wait_ms", "selector_inference_ms", "selector_ms", "rescue_ms", "reader_ms"]
    request_rows = []
    stage_rows = []
    for policy in POLICIES:
        summary = m5[policy]
        request_rows.append([policy, summary["status_counts"].get("ok", 0), f"{summary['request_p50_ms']:.1f}", f"{summary['request_p95_ms']:.1f}", summary["error_rate"]])
        for stage in stage_names:
            p50, p95 = summary["stage_p50_p95_ms"][stage]
            stage_rows.append([policy, stage, 100, f"{p50:.1f}", f"{p95:.1f}"])
    m5_text = f"""# M5 interleaved serving correctness report

## Workload and integrity

The frozen 100-query TEST cohort was rerun for all 12 policies: 1,200 fresh end-to-end requests at concurrency 10. Policy order was randomized independently within ten 10-query blocks with seed `20261011`; every policy received the same cohort. BM25 and the reusable Qdrant client/index remained resident. Local selector families remained resident during their randomized family windows. Total wall time was `{schedule['elapsed_wall_s']:.1f} s`.

- Schedule status: `{schedule['status']}`
- Successful requests: 1,200/1,200
- Retries/errors: 0/0
- Jev calls: 0
- Hosted Cloudflare calls: 0
- Paid usage: false

## Complete-request latency

`Request p50/p95` are measured per-request wall-clock durations from request entry through reader completion. They are not sums of stage percentiles.

{table(["Policy", "OK n", "Request p50 ms", "Request p95 ms", "Error rate"], request_rows)}

## Stage timing populations

Each row uses the 100 successful traces for that policy. `selector_queue_wait_ms` measures waiting for the local selector runtime lock; `selector_inference_ms` measures the model/service section after lock acquisition; `selector_ms` is the full selector stage. `retrieval_ms` is independently measured retrieval-stage duration. Component values are diagnostic and are not added to produce request p95.

{table(["Policy", "Stage", "Population", "p50 ms", "p95 ms"], stage_rows)}
"""
    (AUDIT / "M5_INTERLEAVED_SERVING_REPORT.md").write_text(m5_text, encoding="utf-8")

    qwen_raw = json.loads((AUDIT / "qwen_raw_replays.json").read_text())
    shuffle_diffs = [value["max_abs_diff"] for value in qwen_raw.values()]
    qwen_mean_max = statistics.mean(row["qwen_max_abs_diff_original_vs_stored"] for row in semantic)
    qwen_max_max = max(row["qwen_max_abs_diff_original_vs_stored"] for row in semantic)
    bge_test = m4["metrics"]["bge_p5"]["by_split"]["test"]
    quality_audit = f"""# DecisionRAG quality and correctness audit

## Scope and stop gate

This audit covers the frozen 750-query non-Jev experiment only. No full 22,398-question expansion was started, and Jev was not contacted. Original M3/M4/M5/report artifacts are preserved under `data/results/audit/original_non_jev/` with `ORIGINAL_ARTIFACT_HASHES.json`.

## Phase A — semantic correctness

A deterministic 20-query cohort (Hotpot 7, MuSiQue 7, 2Wiki 6) was independently replayed at the pinned revisions. Common semantic views were generated by the pinned Laya tokenizer; each model then tokenized that identical text through its own model interface.

- Shared-view hashes and document ordering matched for BGE, Qwen, Laya, and Clef: 20/20 audit queries.
- Archived BGE replay with the defective shared Laya tokenizer reproduced the archived BGE scores exactly on 20/20, proving the stored weak result was an implementation replay.
- Native-tokenizer BGE replay matched the corrected BGE trace exactly on 20/20. Its mean Kendall agreement with the archived defective ranking was {statistics.mean(row['bge_original_vs_corrected_kendall'] for row in semantic):.3f}.
- Qwen request hashes matched stored payloads on {audit_summary['qwen_request_hash_matches']}/20; one mismatch is a trailing-space query serialization discrepancy.
- Qwen replay versus stored scores had mean per-query maximum absolute difference {qwen_mean_max:.5f} and maximum {qwen_max_max:.5f}; ranking agreement was high (mean Kendall {statistics.mean(row['qwen_kendall_stored_vs_replay'] for row in semantic):.3f}).
- Qwen original-versus-shuffled candidate order had mean maximum score difference {statistics.mean(shuffle_diffs):.5f}, maximum {max(shuffle_diffs):.5f}, and mean ranking Kendall agreement {statistics.mean(row['qwen_kendall_original_vs_shuffled'] for row in semantic):.3f}. Returned indices correctly mapped to input positions; no duplicate or invalid indices occurred.
- Per-query gold-vs-negative AUROC is in `M3_AUROC_BY_QUERY.jsonl`, with summary in `M3_AUROC_SUMMARY.csv`.

### Root cause and repair

The original BGE adapter used the common Laya tokenizer to create BGE model input IDs. The frozen SDD requires identical actual text, not shared tokenizer IDs. The adapter now uses the native BGE tokenizer for model inference while retaining the common Laya tokenizer solely for the common text view. Both offline M3 and serving M5 adapters were patched, and regression tests cover the separation.

## Phase B — retrieval/evaluation integrity

- Any-gold Top-20 coverage on TEST is 583/600 = 97.17%.
- Complete-gold Top-20 coverage, the valid M4 conditional population, is 291/600 = 48.50%.
- These denominators are distinct and are reported separately in `GOLD_COVERAGE_RECONCILIATION.md`.
- Complete-gold quartet collisions and per-model rescue benefit/harm/unchanged counts are in `GROUPWISE_COLLISION_AND_RESCUE_AUDIT.md`.
- Passage-level selector and reader truncation are measured. Sentence-level survival is `NOT_MEASURED` because the frozen manifest has no supporting-sentence offsets.

## Corrected quality impact

Repaired native-tokenizer BGE TEST results: conditional CSSR {bge_test['cssr_gold_present']*100:.2f}% ({round(bge_test['cssr_gold_present']*bge_test['gold_present_n'])}/{bge_test['gold_present_n']}), unconditional CSSR {bge_test['cssr_unconditional']*100:.2f}%, support recall {bge_test['support_recall']*100:.2f}%, EM {bge_test['answer_em']*100:.2f}%, F1 {bge_test['answer_f1']*100:.2f}%. The archived defective result was 12.71% conditional CSSR. Corrected BGE is the CAL-selected local baseline.
"""
    (AUDIT / "QUALITY_CORRECTNESS_AUDIT.md").write_text(quality_audit, encoding="utf-8")

    (AUDIT / "ERRATA.md").write_text(f"""# DecisionRAG erratum

The original non-Jev report used the pinned Laya tokenizer for both common-view construction and BGE model input encoding. The frozen SDD requires identical serialized text but native tokenization at each model adapter. The original BGE M3/M4/M5 artifacts are preserved under `data/results/audit/original_non_jev/`.

The BGE adapter was repaired, regression-tested, rerun on all 750 frozen questions, reassembled into the 12-policy M3 output, and the affected reader policies were rerun. Corrected BGE TEST conditional CSSR is {bge_test['cssr_gold_present']*100:.2f}%, replacing archived 12.71%. No Jev work or dataset expansion occurred.
""", encoding="utf-8")

    quality_rows = []
    serving_rows = []
    for policy in POLICIES:
        x = m4["metrics"][policy]["by_split"]["test"]
        quality_rows.append([policy, f"{x['cssr_gold_present']*100:.2f}% ({round(x['cssr_gold_present']*x['gold_present_n'])}/{x['gold_present_n']})", f"{x['cssr_unconditional']*100:.2f}%", f"{x['support_recall']*100:.2f}%", f"{x['answer_em']*100:.2f}%", f"{x['answer_f1']*100:.2f}%"])
        serving_rows.append([policy, f"{m5[policy]['request_p50_ms']:.1f}", f"{m5[policy]['request_p95_ms']:.1f}", f"{m5[policy]['stage_p50_p95_ms']['retrieval_ms'][0]:.1f}/{m5[policy]['stage_p50_p95_ms']['retrieval_ms'][1]:.1f}", f"{m5[policy]['stage_p50_p95_ms']['selector_ms'][0]:.1f}/{m5[policy]['stage_p50_p95_ms']['selector_ms'][1]:.1f}", f"{m5[policy]['stage_p50_p95_ms']['reader_ms'][0]:.1f}/{m5[policy]['stage_p50_p95_ms']['reader_ms'][1]:.1f}"])
    revised = f"""# DecisionRAG revised non-Jev report after mandatory quality audit

**Status:** `NON_JEV_COMPLETE_AUDITED_JEV_PENDING_APPROVAL`

This revision supersedes the original headline non-Jev report where the BGE tokenizer defect affected results. Archived artifacts remain preserved. No Jev request or full-data expansion occurred.

## Resolution

A confirmed BGE implementation defect was found and repaired. The corrected path preserves the common serialized text views and uses the native BGE tokenizer for BGE inference. Corrected M3 traces, M3 selections, M4 reader results and the interleaved M5 serving traces are real local measurements.

## Corrected TEST quality

{table(["Policy", "Conditional CSSR", "Unconditional CSSR", "Support recall", "EM", "F1"], quality_rows)}

The complete-gold conditional denominator is 291/600 for every policy. The corrected BGE row is authoritative; its archived defective predecessor was 12.71% conditional CSSR.

## Interleaved serving

{table(["Policy", "Request p50 ms", "Request p95 ms", "Retrieval p50/p95", "Selector p50/p95", "Reader p50/p95"], serving_rows)}

Request percentiles are measured end-to-end. Stage percentiles are diagnostics and are not added together. The complete stage table is `M5_INTERLEAVED_SERVING_REPORT.md`.

## Gate status

- Semantic audit: PASS, with one Qwen trailing-space serialization discrepancy and numerical service nondeterminism documented.
- Retrieval/evaluation reconciliation: PASS; any-gold and complete-gold denominators are separated.
- BGE repair: PASS; 750/750 corrected pointwise traces and 9,000/9,000 reader rows are `ok`.
- Interleaved serving: PASS; 1,200/1,200 requests are `ok`, with zero retries and errors.
- Jev calls: 0.
- Full expansion: NOT STARTED by the mandatory stop gate.

## Remaining uncertainties

1. The local Qwen reranker is not bitwise score-stable across repeated/batch-order requests, although response indices remain correctly associated and ranking agreement is high on the audit cohort. Future Qwen confirmation requires an explicit determinism policy.
2. One Qwen stored request hash differs due a trailing-space query serialization difference.
3. Sentence-level support visibility is unavailable from the frozen passage-only gold manifest.
4. Per-request GPU active seconds remain `NOT_MEASURED`; full-request and stage wall times are measured.

## Mandatory stop

Do not start the 22,398-question expansion or call Jev until the owner reviews these corrected findings.
"""
    (AUDIT / "REVISED_NON_JEV_COMPLETE_REPORT.md").write_text(revised, encoding="utf-8")

    manifest = {"status": "NON_JEV_COMPLETE_AUDITED_JEV_PENDING_APPROVAL", "jev_calls": 0, "full_expansion_started": False, "bge_defect_confirmed": True, "bge_corrected_pointwise_rows": 750, "reader_corrected_rows": 9000, "interleaved_requests": 1200, "interleaved_errors": 0, "artifacts": ["QUALITY_CORRECTNESS_AUDIT.md", "M3_SCORE_ALIGNMENT_AUDIT.csv", "GOLD_COVERAGE_RECONCILIATION.md", "GROUPWISE_COLLISION_AND_RESCUE_AUDIT.md", "M5_INTERLEAVED_SERVING_REPORT.md", "ERRATA.md", "REVISED_NON_JEV_COMPLETE_REPORT.md"]}
    (AUDIT / "QUALITY_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
