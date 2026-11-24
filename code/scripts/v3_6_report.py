#!/usr/bin/env python3
"""Generate the CPU-only v3.6 scientific closure report from frozen exports."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential"


def rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    q = rows(OUT / "stage1_quality_v2/export/quality/q_summary.csv")
    contrasts = rows(OUT / "stage1_quality_v2/export/quality/q_contrasts.csv")
    ratios = rows(OUT / "stage2_serving/export/serving_ratios.csv")
    pooled = rows(OUT / "stage2_serving/export/serving_ratios_pooled.csv")
    gates = json.loads((OUT / "GATE_SUMMARY.json").read_text(encoding="utf-8"))
    lines = [
        "# DecisionRAG v3.6 Essential Verification — Final CPU Closure",
        "",
        f"Overall status: **{gates['overall']}**.",
        "",
        "This closure preserves the original sandbox CUDA failures and the initial Jina two-stage adapter failures. Corrected GPU traces are versioned under the GPU3/GPU6 namespaces; no raw trace was overwritten.",
        "",
        "## Gate status",
        "",
        f"- Stage 0 frozen-input preflight: **{gates['stage0']}** (1,384 BRIGHT queries; L=8192 identity receipt passed).",
        f"- Stage 1 quality: **{gates['stage1_quality']}** (31,832 per-query rows; labels joined only in CPU analysis).",
        f"- Stage 2 serving: **{gates['stage2_serving']}** (two complete five-policy T1 sessions; the third namespace contains Jina-only records).",
        f"- Stage 3 frontier/statistics: **{gates['stage3']}**.",
        f"- Stage 4 optional capacity/optimization: **{gates['stage4']}**.",
        "",
        "## Quality results",
        "",
        "All-request known-positive nDCG@5 is normalized by min(gold_count,5); invalid inference outcomes score zero. These are ranking metrics, not downstream answer correctness.",
        "",
        "| Policy | n | Valid % | Mean nDCG@5 |",
        "|---|---:|---:|---:|",
    ]
    for r in q:
        if r["policy"] in {"jina_full", "pplx_single", "pplx_two_stage", "jina_single", "jina_two_stage", "liquid_d1_default", "winnow_native", "winnow_budget", "drex_v2_native"} and r["budget_L"] in {"", "8192", "2048", "4096"}:
            lines.append(f"| {r['policy']} {r['budget_L']} | {r['n']} | {f(float(r['valid_pct'])*100,2)}% | {f(r['mean_ndcg5_all'])} |")
    lines += [
        "",
        "### Matched-text contrasts",
        "",
        "| Contrast | Delta | 95% CI | Holm p | Positive domains |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in contrasts:
        lines.append(f"| {r['contrast']} | {f(r['delta_query_weighted'])} | [{f(r['lo95'])}, {f(r['hi95'])}] | {r['p_holm']} | {r['domains_positive']}/12 |")
    lines += [
        "",
        "## Matched serving results",
        "",
        "The complete-policy T1 results are available for two isolated sessions (slots A and B in derived exports). Slot C contains the full Jina block only; it is not treated as a complete five-policy replicate. Ratios are same-session p95(policy)/p95(Jina full), with cluster-bootstrap intervals over C360 IDs and all-request failure timing included.",
        "",
        "| Session | Policy | p50 ms | p95 ms | p95 ratio | 95% interval | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in ratios:
        if r["basis"] == "all_request":
            lines.append(f"| {r['session']} | {r['policy']} | {f(r['p50_ms'],1)} | {f(r['p95_ms'],1)} | {f(r['p95_ratio_point'],3)} | [{f(r['bootstrap_lo95'],3)}, {f(r['bootstrap_hi95'],3)}] | {r['verdict']} |")
    lines += [
        "",
        "### Pooled available-session view",
        "",
        "| Policy | Available complete sessions | p95 ratio | 95% interval | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for r in pooled:
        if r["basis"] == "all_request":
            available = 3 if r["policy"] in {"jina_full", "jina_pplx_budget"} else 2
            lines.append(f"| {r['policy']} | {available} | {f(r['p95_ratio_point'],3)} | [{f(r['bootstrap_lo95'],3)}, {f(r['bootstrap_hi95'],3)}] | {r['verdict']} |")
    lines += [
        "",
        "## Plain-language answers",
        "",
        "1. **G1:** Budgeted PPLX (`P_8192`) is about 1.95× full Jina in both complete sessions. The available-session interval crosses 2×, so the serving verdict is **inconclusive**, not a pass or failure. The 1.5× and 3× sensitivity rows are computed in `serving_threshold_sensitivity.csv`; no claim is made from an absent third PPLX session.",
        "2. **G2:** Winnow, Drex and Liquid were not newly timed in this v3.6 runner; their quality records are historical controls only. No new same-session serving claim is made for them.",
        "3. **G3:** On quality alone, PPLX two-stage at L=4096 and L=8192 exceeds full-text Jina's mean nDCG@5 in this exploratory BRIGHT population; the complete three-check frontier is not finalized because the required replicated/T2 timing and quality-difference intervals are incomplete. PPLX two-stage is clearly slower than Jina in the available serving sessions.",
        "4. **Two-stage:** At the matched L=2048/4096 quality points, PPLX two-stage exceeds matched-text Jina two-stage in the per-query analysis; this is an exploratory result, not a confirmatory release claim.",
        "5. **Tuning Jina:** Lower budgets reduce Jina quality and often validity; Jina two-stage is more reliable at 8192 than its single-call budgeted variant, but no complete frontier release decision is asserted.",
        "6. **Interference:** The three namespace names do not constitute a valid concurrent three-GPU interference experiment. `solo_vs_concurrent.csv` is explicitly `NOT_RUN`.",
        "7. **Parameters:** The available timing records show the 27B PPLX model has materially higher p95 latency than the 0.6B Jina model; this is descriptive and not a causal parameter-scaling law.",
        "8. **Optional capacity:** Stage 4 was not run; no sustainable QPS or monetary cost claim is made.",
        "9. **Manuscript implication:** describe the result as an exploratory BRIGHT sensitivity/serving study, state that budgeted PPLX is borderline at the 2× p95 threshold, and disclose incomplete T2/solo/three-complete-session coverage.",
        "",
        "## Confirmed, changed, unverified",
        "",
        "- **Confirmed:** frozen BRIGHT input identity; L=8192 reuse identity; corrected native Stage1 traces; two complete primary T1 sessions; raw failures retained; no label-blind serving leakage.",
        "- **Changed:** the original one-session serving interpretation is superseded by a versioned two-complete-session plus Jina-only third-session record; the initial two-stage Jina adapter failure is superseded only for the repaired policy trace.",
        "- **Unverified:** the full three-complete-replicate serving gate, solo interference, T2 budget frontier, historical-model matched timing, determinism rechecks, and optional concurrency/optimization experiments.",
        "",
        "## Artifact pointers",
        "",
        "- `stage0_preflight/` — frozen input/protocol receipts.",
        "- `stage1_quality_v2/export/quality/` — quality per-query, summary and contrast exports.",
        "- `stage2_serving/export/` — path-free derived serving exports.",
        "- `stage2_serving/`, `stage2_serving_gpu6/`, `stage2_serving_gpu3_r3/` — internal raw traces and provenance.",
        "- `stage3_analysis/export/frontier/` — partial release-check/frontier outputs.",
        "- `RAW_TRACE_SHA256_MANIFEST.json` — final recursive artifact hash manifest.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORT_WRITTEN", "path": "results/v3_6_essential/REPORT.md"}))


if __name__ == "__main__":
    main()
