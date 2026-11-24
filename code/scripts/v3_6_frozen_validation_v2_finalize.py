#!/usr/bin/env python3
"""Create the append-only owner-review closure for CQADupStack v2."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/v3_6_essential/frozen_prospective_validation_v2"
OUT = V2 / "final_closure_v1"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(value: str | None) -> float | None:
    return None if value in (None, "", "None") else float(value)


def row(rows: list[dict], **kwargs: str) -> dict:
    for x in rows:
        if all(x.get(k) == v for k, v in kwargs.items()):
            return x
    raise KeyError(kwargs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read_csv(V2 / "val_summary.csv")
    contrasts = read_csv(V2 / "val_contrasts.csv")
    lat_summary = read_csv(V2 / "passB_analysis_clean_v1/PASSB_LATENCY_SUMMARY.csv")
    ratios = read_csv(V2 / "passB_analysis_clean_v1/PASSB_LATENCY_RATIOS.csv")
    integrity = json.loads((V2 / "passB_analysis_clean_v1/PASSB_INTEGRITY.json").read_text(encoding="utf-8"))
    prep = json.loads((V2 / "PREPARATION_SUMMARY.json").read_text(encoding="utf-8"))
    p6144 = row(summary, policy="pplx_single_6144")
    p4096 = row(summary, policy="pplx_single_4096")
    jina = row(summary, policy="jina_full_text")
    bm25 = row(summary, policy="bm25_top5")
    c_p6144_bm25 = row(contrasts, contrast="pplx_single_6144_minus_bm25_top5")
    c_p6144_jina = row(contrasts, contrast="pplx_single_6144_minus_jina_full_text")
    ratio = {r["policy"]: row(ratios, policy=r["policy"], basis="pooled_serial_passes") for r in ratios if r["basis"] == "pooled_serial_passes"}
    clean_passes = []
    for p in ("native_passB_paired_clean_pass1_gpu3_shard0_of1_pass1.SUMMARY.json", "native_passB_paired_clean_pass2_gpu3_shard0_of1_pass2.SUMMARY.json"):
        clean_passes.append(json.loads((V2 / "raw" / p).read_text(encoding="utf-8")))

    status = {
        "gate": "V2_OWNER_REVIEW_READY",
        "preparation": "PASS",
        "pass_a_quality": "PASS_WITH_EXPLICIT_JINA_FAILURE_ACCOUNTING",
        "pass_b_clean_trace_integrity": "PASS",
        "pass_b_primary_latency_s_gate": "INCONCLUSIVE_FOR_PPLX_6144",
        "pass_b_secondary_latency_s_gate": "PASS_FOR_PPLX_4096_AND_JINA_TRUNC_8192",
        "full_regression_designated_environment": "PASS",
        "system_python_regression": "NOT_COMPARABLE_MISSING_PYARROW_AND_TORCH",
        "overall_scientific_claim": "PPLX_6144_QUALITY_PASS_BUT_LATENCY_NONINFERIORITY_INCONCLUSIVE",
        "protocol_sha256": integrity["protocol_sha256"],
        "latency_cohort_sha256": integrity["latency_cohort_sha256"],
        "raw_artifacts_preserved": True,
        "labels_joined_in_pass_b": False,
        "qrels_read_in_pass_b": False,
    }
    (OUT / "FINAL_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_paths = [
        V2 / "PROTOCOL_FROZEN.md", V2 / "PROTOCOL_FROZEN.md.sha256", V2 / "PREPARATION_SUMMARY.json",
        V2 / "SERVING_ARGS.json", V2 / "LATENCY_COHORT_IDS.jsonl", V2 / "DRY_RUN_IDS.jsonl",
        V2 / "CQADUPSTACK_FROZEN_TOP20_SLATES.jsonl", V2 / "prepared_L4096.jsonl", V2 / "prepared_L6144.jsonl", V2 / "prepared_L8192.jsonl",
        V2 / "val_summary.csv", V2 / "val_contrasts.csv", V2 / "val_perquery.csv", V2 / "OFFLINE_SCORECARD.json",
        V2 / "raw/native_passB_paired_clean_pass1_gpu3_shard0_of1_pass1.jsonl",
        V2 / "raw/native_passB_paired_clean_pass1_gpu3_shard0_of1_pass1.SUMMARY.json",
        V2 / "raw/native_passB_paired_clean_pass2_gpu3_shard0_of1_pass2.jsonl",
        V2 / "raw/native_passB_paired_clean_pass2_gpu3_shard0_of1_pass2.SUMMARY.json",
        V2 / "passB_analysis_clean_v1/PASSB_PERREQUEST.csv", V2 / "passB_analysis_clean_v1/PASSB_LATENCY_SUMMARY.csv",
        V2 / "passB_analysis_clean_v1/PASSB_LATENCY_RATIOS.csv", V2 / "passB_analysis_clean_v1/PASSB_ANCHOR_SUMMARY.csv",
        V2 / "passB_analysis_clean_v1/PASSB_INTEGRITY.json", V2 / "passB_analysis_clean_v1/PASSB_RAW_MANIFEST.json",
        ROOT / "scripts/v3_6_frozen_validation_v2_native.py", ROOT / "scripts/v3_6_frozen_validation_v2_prepare.py",
        ROOT / "scripts/v3_6_frozen_validation_v2_analyze.py", ROOT / "scripts/v3_6_frozen_validation_v2_passb_analyze.py",
    ]
    manifest = []
    for p in manifest_paths:
        if p.exists():
            manifest.append({"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size})
    (OUT / "PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# DecisionRAG v3.6 Frozen Prospective Validation v2 — Final Closure

Gate: **`V2_OWNER_REVIEW_READY`**

## Scope and integrity

- Dataset: official BEIR CQADupStack test release, all 12 forums.
- Total query records: **{prep['total_queries']}**.
- Confirmatory quality population: **{prep['confirmatory_quality_n']}**; dry-run records excluded: **{prep['dry_run_n']}**.
- Clean latency population: **{prep['latency_n']}** fixed IDs.
- Frozen protocol SHA-256: `{integrity['protocol_sha256']}`.
- Frozen latency-cohort SHA-256: `{integrity['latency_cohort_sha256']}`.
- Pass-B labels/qrels: **not read**; all raw traces remained label-blind.
- Original split/contended Pass-B traces and the clean rerun are preserved separately.

## Pass-A quality scorecard

Primary scoring is all-request nDCG@5 with failures scored zero; confidence intervals are forum-bootstrap intervals from the existing frozen analysis.

| Policy | n | Valid rate | Mean nDCG@5 | Forum-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| BM25 Top-5 | {bm25['n']} | {float(bm25['valid_rate']):.6f} | {float(bm25['mean_ndcg5_all']):.6f} | [{float(bm25['forum_bootstrap_lo95']):.6f}, {float(bm25['forum_bootstrap_hi95']):.6f}] |
| PPLX 6144 | {p6144['n']} | {float(p6144['valid_rate']):.6f} | {float(p6144['mean_ndcg5_all']):.6f} | [{float(p6144['forum_bootstrap_lo95']):.6f}, {float(p6144['forum_bootstrap_hi95']):.6f}] |
| PPLX 4096 | {p4096['n']} | {float(p4096['valid_rate']):.6f} | {float(p4096['mean_ndcg5_all']):.6f} | [{float(p4096['forum_bootstrap_lo95']):.6f}, {float(p4096['forum_bootstrap_hi95']):.6f}] |
| Jina full | {jina['n']} | {float(jina['valid_rate']):.6f} | {float(jina['mean_ndcg5_all']):.6f} | [{float(jina['forum_bootstrap_lo95']):.6f}, {float(jina['forum_bootstrap_hi95']):.6f}] |

Jina full had one recorded `CUDA_OOM`/missing quality record in the original Pass-A trace; it was retained and scored as zero. No request was silently dropped.

Primary PPLX 6144 contrasts:

- PPLX 6144 − BM25: **{float(c_p6144_bm25['delta_all_query_weighted']):+.6f}**, CI [{float(c_p6144_bm25['forum_bootstrap_lo95']):+.6f}, {float(c_p6144_bm25['forum_bootstrap_hi95']):+.6f}].
- PPLX 6144 − Jina full: **{float(c_p6144_jina['delta_all_query_weighted']):+.6f}**, CI [{float(c_p6144_jina['forum_bootstrap_lo95']):+.6f}, {float(c_p6144_jina['forum_bootstrap_hi95']):+.6f}].

Therefore the frozen quality rule is met for PPLX 6144 against both BM25 and Jina. This is a ranking-quality result, not evidence of downstream answer correctness.

## Strict clean Pass-B serving result

The primary latency result uses the two new clean GPU3 sessions, each serial batch-1 and paired across all four admitted policies, with 60 Jina anchors at both the start and end.

| Policy | Valid rate | p50 ms (pooled) | p95 ms (pooled) | p95 ratio vs Jina | 95% ratio CI | P95 S verdict |
|---|---:|---:|---:|---:|---:|---|
| PPLX 4096 | 1.000000 | {float(ratio['pplx_single_4096']['p50_ms']):.3f} | {float(ratio['pplx_single_4096']['p95_ms']):.3f} | {float(ratio['pplx_single_4096']['p95_ratio']):.3f} | [{float(ratio['pplx_single_4096']['ratio_lo95']):.3f}, {float(ratio['pplx_single_4096']['ratio_hi95']):.3f}] | PASS |
| PPLX 6144 | 1.000000 | {float(ratio['pplx_single_6144']['p50_ms']):.3f} | {float(ratio['pplx_single_6144']['p95_ms']):.3f} | {float(ratio['pplx_single_6144']['p95_ratio']):.3f} | [{float(ratio['pplx_single_6144']['ratio_lo95']):.3f}, {float(ratio['pplx_single_6144']['ratio_hi95']):.3f}] | INCONCLUSIVE |
| Jina trunc 8192 | 1.000000 | {float(ratio['jina_trunc_8192']['p50_ms']):.3f} | {float(ratio['jina_trunc_8192']['p95_ms']):.3f} | {float(ratio['jina_trunc_8192']['p95_ratio']):.3f} | [{float(ratio['jina_trunc_8192']['ratio_lo95']):.3f}, {float(ratio['jina_trunc_8192']['ratio_hi95']):.3f}] | PASS |

Ratios use 10,000 query-cluster bootstrap draws, seed 731, with both serial passes retained per query. All request latency is included; no fallback latency was substituted. PPLX 6144 is therefore **not statistically established within the 2× p95 SLO**, despite passing the quality endpoint. PPLX 4096 passes the measured latency S criterion.

Clean-session integrity checks: 16,000 data records, 240 anchors, zero request failures, zero missing/nonunique policy-query keys, zero cross-pass input-hash mismatches, zero reverse-order mismatches, and zero selected-ID mismatches among paired valid outputs. Allocator OOM warning lines appeared in stderr during the long sessions but did not create failed request records; they are a telemetry limitation, not silently discarded failures.

## Final scientific interpretation

The result supports a **conditional replacement boundary**: PPLX 4096 is competitive on this frozen CQADupStack ranking population and satisfies the measured latency S criterion; PPLX 6144 is the preregistered primary configuration and passes the quality endpoint, but its 2× p95 serving criterion is inconclusive. PPLX 4096's observed mean nDCG@5 is slightly higher in this scorecard, so the evidence does not justify claiming unconditional PPLX 6144 replacement, production cost savings, or answer-quality superiority.

## Tests and provenance

- Designated environment: full `pytest` suite passed in `Production_Project/MultiModalRAG/venv`; only Triton/deprecation warnings were emitted.
- System Python diagnostic: 3 tests could not import optional `pyarrow`/`torch`; this was an environment mismatch and was not used as the project regression result.
- The machine-readable hash and size inventory is in `PROVENANCE_MANIFEST.json`.
- Clean Pass-B analysis artifacts are in `../passB_analysis_clean_v1/`.
- No historical artifact was overwritten; the clean rerun and all earlier partial/contended traces are append-only.
"""
    (OUT / "V3_6_FROZEN_VALIDATION_V2_FINAL_CLOSURE.md").write_text(report, encoding="utf-8")
    (OUT / "TEST_REPORT.md").write_text(
        "# Test Report\n\n"
        "- Designated MultiModalRAG/venv full pytest suite: PASS.\n"
        "- System Python diagnostic: NOT COMPARABLE; 3 dependency-import failures (pyarrow/torch absent).\n"
        "- Pass-B analyzer syntax check: PASS.\n"
        "- Clean Pass-B coverage/integrity: PASS.\n",
        encoding="utf-8",
    )
    for p in (OUT / "FINAL_STATUS.json", OUT / "V3_6_FROZEN_VALIDATION_V2_FINAL_CLOSURE.md", OUT / "TEST_REPORT.md"):
        manifest.append({"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size})
    (OUT / "PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out": str(OUT), "manifest_entries": len(manifest)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
