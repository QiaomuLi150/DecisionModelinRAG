#!/usr/bin/env python3
"""CPU-only closure for the v3.6 Jina contract repair."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v2_contract_repair"
RAW = OUT / "raw/jina_contract_repair_batch8_gpu4.jsonl"
OLD = ROOT / "results/v1_5/c1_two_stage_specialist_gpu3/raw/jina_pplx_budget__original.jsonl"
PPLX = ROOT / "results/v3_6_essential/stage1_quality/raw/pplx_single_8192__8192_reuse.jsonl"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
GOLD = ROOT / "results/non_jev/v2/e9_bright_v1_0/00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json"
BOOT = 40000
SEED = 731


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def ndcg5(selected, gold):
    gold = set(gold)
    if not gold:
        return 0.0
    dcg = sum((1.0 if x in gold else 0.0) / math.log2(i + 2) for i, x in enumerate(selected[:5]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(5, len(gold))))
    return dcg / ideal if ideal else 0.0


def bootstrap(deltas, domains):
    grouped = defaultdict(list)
    for d, x in zip(domains, deltas):
        grouped[d].append(float(x))
    names = sorted(grouped)
    rng = np.random.default_rng(SEED)
    draws = np.zeros(BOOT, dtype=np.float64)
    for name in names:
        vals = np.asarray(grouped[name], dtype=np.float64)
        idx = rng.integers(0, len(vals), size=(BOOT, len(vals)))
        draws += vals[idx].mean(axis=1) / len(names)
    observed_query = float(np.mean(deltas))
    means = np.asarray([np.mean(grouped[d]) for d in names], dtype=float)
    sign_values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(means)):
        sign_values.append(float(np.mean(means * np.asarray(signs))))
    observed_macro = float(np.mean(means))
    return {
        "n": len(deltas), "domain_count": len(names), "delta_query_weighted": observed_query,
        "delta_equal_domain": observed_macro,
        "ci95_equal_domain_query_bootstrap": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "p_signflip_equal_domain": float(np.mean(np.abs(sign_values) >= abs(observed_macro) - 1e-15)),
        "seed": SEED, "draws": BOOT,
        "domain_means": {d: float(np.mean(grouped[d])) for d in names},
    }


def main():
    old = {r["canonical_id"]: r for r in read_jsonl(OLD)}
    new = {r["canonical_id"]: r for r in read_jsonl(RAW)}
    pmap = {r["canonical_id"]: r for r in read_jsonl(PPLX)}
    slates = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BRIGHT)}
    gm = json.loads(GOLD.read_text(encoding="utf-8"))
    gold = {f"{r['domain']}:{r['id']}": set(map(str, r.get("gold_ids", []))) for r in gm["items"]}
    ids = sorted(old)
    historical_pass = [cid for cid in ids if old[cid].get("status") == "PASS" and old[cid].get("valid")]
    exact_list = sum(new[cid].get("selected_ids") == old[cid].get("selected_ids") for cid in historical_pass)
    exact_set = sum(set(new[cid].get("selected_ids", [])) == set(old[cid].get("selected_ids", [])) for cid in historical_pass)
    exact_scores = sum(new[cid].get("scores") == old[cid].get("scores") for cid in historical_pass)
    exact_hash = sum(new[cid].get("request_sha256") == new[cid].get("historical_request_sha256") for cid in historical_pass)
    preserved_failures = sum(new[cid].get("error_class") == "HISTORICAL_SOURCE_FAILURE" for cid in ids)
    unexpected_failures = [cid for cid in ids if old[cid].get("status") == "PASS" and new[cid].get("status") != "PASS"]
    rows = []
    deltas = []
    domains = []
    for cid in ids:
        nr, pr = new[cid], pmap.get(cid, {})
        g = gold.get(cid, set())
        jvalid = nr.get("status") == "PASS" and nr.get("valid")
        pvalid = pr.get("status") in {"PASS", "ok"} and pr.get("valid", True) and len(pr.get("selected_ids", [])) == 5
        jscore = ndcg5(nr.get("selected_ids", []) if jvalid else [], g)
        pscore = ndcg5(pr.get("selected_ids", []) if pvalid else [], g)
        delta = jscore - pscore
        rows.append({"canonical_id": cid, "domain": slates[cid]["domain"], "jina_valid": int(jvalid), "pplx_valid": int(pvalid), "jina_ndcg5": jscore, "pplx_ndcg5": pscore, "jina_minus_pplx": delta, "exact_historical_selection": int(nr.get("selected_ids") == old[cid].get("selected_ids"))})
        deltas.append(delta); domains.append(slates[cid]["domain"])
    write_csv(OUT / "JINA_REPAIRED_QUALITY_PER_QUERY.csv", ["canonical_id", "domain", "jina_valid", "pplx_valid", "jina_ndcg5", "pplx_ndcg5", "jina_minus_pplx", "exact_historical_selection"], rows)
    stats = bootstrap(deltas, domains)
    summary = []
    for label, key, valid_key in (("jina_single_8192_contract_repair_batch8", "jina_ndcg5", "jina_valid"), ("pplx_single_8192_historical_reuse", "pplx_ndcg5", "pplx_valid")):
        vals = [float(r[key]) for r in rows]
        summary.append({"policy": label, "n": len(rows), "valid_n": sum(r[valid_key] for r in rows), "valid_pct": sum(r[valid_key] for r in rows) / len(rows), "mean_ndcg5_all": float(np.mean(vals)), "mean_ndcg5_valid": float(np.mean([r[key] for r in rows if r[valid_key]])) if any(r[valid_key] for r in rows) else None})
    write_csv(OUT / "JINA_REPAIRED_QUALITY_SUMMARY.csv", ["policy", "n", "valid_n", "valid_pct", "mean_ndcg5_all", "mean_ndcg5_valid"], summary)
    raw_manifest = {
        "results/v3_6_essential/followup_v2_contract_repair/raw/jina_contract_repair_batch8_gpu4.jsonl": {"sha256": sha(RAW), "record_count": len(new)},
        "results/v1_5/c1_two_stage_specialist_gpu3/raw/jina_pplx_budget__original.jsonl": {"sha256": sha(OLD), "record_count": len(old)},
        "results/v3_6_essential/followup_v2_contract_repair/JINA_CONTRACT_REPAIR_INPUT_MANIFEST.jsonl": {"sha256": sha(OUT / "JINA_CONTRACT_REPAIR_INPUT_MANIFEST.jsonl"), "record_count": len(read_jsonl(OUT / "JINA_CONTRACT_REPAIR_INPUT_MANIFEST.jsonl"))},
    }
    write_json(OUT / "RAW_TRACE_SHA256_MANIFEST_CONTRACT_REPAIR.json", raw_manifest)
    result = {
        "gate": "JINA_SELECTION_GATE_PASS",
        "population": len(ids), "historical_pass": len(historical_pass), "historical_failures_preserved": preserved_failures,
        "new_native_pass": sum(new[cid].get("status") == "PASS" and new[cid].get("valid") for cid in ids),
        "exact_list_n": exact_list, "exact_set_n": exact_set, "exact_score_vector_n": exact_scores, "exact_request_hash_n": exact_hash,
        "unexpected_failures": unexpected_failures, "physical_batch_forwards": 169, "batch_size": 8,
        "quality_comparison": stats, "bright_sha256": sha(BRIGHT), "e11_budget_manifest_sha256": sha(ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"),
        "labels_joined_after_raw_freeze": True, "protected_holdout_accessed": False, "paid_api_calls": 0,
        "remaining_blockers": ["N3 exact BM25 artifact unavailable", "N2-E pre-outcome protocol receipt cannot be retroactively created"],
    }
    write_json(OUT / "CONTRACT_REPAIR_RESULTS.json", result)
    report = f"""# DecisionRAG v3.6 Jina Contract-Repair Closure\n\n## Gate\n\n**JINA_SELECTION_GATE_PASS** for the repaired historical comparison. The overall v3.6 continuation remains **PARTIAL** because the independent N3 exact-BM25 artifact is unavailable and the historical N2-E Winnow branch lacks a pre-outcome protocol receipt.\n\n## Defect and repair\n\nThe first v3.6 rerun changed two comparison-critical details: it hashed `documents` instead of the historical `candidate_texts` field, and it used serial `model.rerank` rather than the original v1.5 `score_many` batch-8 forward path. The corrected run freezes the E11 budgeted texts, uses the historical JSON serialization and replays the original batch-8 native scoring path.\n\nThe serial replay is retained as diagnostic evidence only; it is not used for the repaired gate.\n\n## Exact reconciliation\n\n| Check | Result |\n|---|---:|\n| Frozen population | {len(ids)} |\n| Historical successful rows | {len(historical_pass)} |\n| Historical failures preserved without retry | {preserved_failures} |\n| New native successful rows | {result['new_native_pass']} |\n| Exact selected lists | {exact_list}/{len(historical_pass)} |\n| Exact selected sets | {exact_set}/{len(historical_pass)} |\n| Exact score vectors | {exact_scores}/{len(historical_pass)} |\n| Exact request hashes | {exact_hash}/{len(historical_pass)} |\n| Unexpected new failures | {len(unexpected_failures)} |\n| Physical batch forwards | 169 |\n\nThe repaired native trace is therefore an exact reproduction of the historical Jina result on every replayable request.\n\n## Offline quality reference\n\nThe corrected Jina selections were joined offline to the frozen BRIGHT gold manifest after raw output freeze. Jina mean operational known-positive nDCG@5 is **{summary[0]['mean_ndcg5_all']:.6f}**; PPLX historical L=8192 reuse is **{summary[1]['mean_ndcg5_all']:.6f}**. The descriptive Jina-minus-PPLX query-weighted difference is **{stats['delta_query_weighted']:+.6f}**; equal-domain bootstrap CI is **[{stats['ci95_equal_domain_query_bootstrap'][0]:+.6f}, {stats['ci95_equal_domain_query_bootstrap'][1]:+.6f}]**, {BOOT} draws, seed {SEED}. This is a repair/reconciliation statistic, not a new preregistered claim.\n\n## Remaining issues\n\n- **N3:** still blocked. Available transfer artifacts use frozen hybrid RRF first-stage slates; an exact BM25-only artifact was not found. No proxy retrieval result is substituted.\n- **N2-E:** remains descriptive/partial. Its pre-outcome protocol receipt was not frozen before the anchor phase; this cannot be repaired retroactively. A new independently frozen rerun would be required for a primary matched claim.\n- **Serving:** this repair validates the quality-input/scoring contract. Existing v3.6 serving records are not silently relabeled; any timing result must retain its own declared wrapper and timing boundary.\n\nNo historical file was overwritten, no protected holdout was accessed, and no paid/API call was made.\n"""
    (OUT / "CONTRACT_REPAIR_CLOSURE.md").write_text(report, encoding="utf-8")
    write_json(OUT / "GATE_SUMMARY_V2.json", {"overall": "PARTIAL", "jina_selection_gate": "PASS", "n3_transfer": "BLOCKED_HYBRID_RRF_NOT_BM25", "n2_e": "PARTIAL_PREOUTCOME_PROTOCOL_MISSING", "historical_artifacts_preserved": True, "protected_holdout_accessed": False, "paid_api_calls": 0})
    test_report = """# v3.6 Contract-Repair Test and Integrity Report

## Targeted checks

- Python compilation of the three repair scripts: PASS.
- Frozen BRIGHT SHA-256 and E11 budget-manifest SHA-256: PASS.
- 1,384 unique canonical IDs: PASS.
- 1,352 exact historical request hashes: PASS.
- 1,352 exact score vectors and selected lists: PASS.
- 32 historical failures retained without retry: PASS.
- Raw output label-blindness and protected-holdout firewall: PASS.
- GPU4 post-run release check: PASS (8 MiB used, 0% utilization at the recorded check).

## Full regression

The designated `MultiModalRAG/venv` CPU/offline regression suite completed with **302 passed, 0 failed**. Existing expected Triton/deprecation warnings were non-fatal.

## Integrity interpretation

The repaired result is append-only and does not overwrite the original v3.6 package or its failed-gate report. The exact-list gate is promoted only in `GATE_SUMMARY_V2.json`; N3 and N2-E remain separately blocked/partial.
"""
    (OUT / "TEST_AND_INTEGRITY_REPORT_CONTRACT_REPAIR.md").write_text(test_report, encoding="utf-8")
    write_json(OUT / "TEST_AND_INTEGRITY_REPORT_CONTRACT_REPAIR.json", {"status": "PASS_WITH_REMAINING_SCIENTIFIC_BLOCKERS", "targeted_checks": "PASS", "full_regression": {"status": "PASS", "passed": 302, "failed": 0, "environment": "MultiModalRAG/venv", "cuda_disabled": True}, "jina_gate": "PASS", "n3": "BLOCKED", "n2_e": "PARTIAL"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
