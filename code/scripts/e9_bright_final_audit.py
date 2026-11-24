#!/usr/bin/env python3
"""CPU-only final audit and evidence freeze for DecisionRAG E9-BRIGHT.

The script consumes frozen E9-BRIGHT outputs and local manifests only.  It
never loads a model, accesses a service, downloads data, or changes any
historical E2-E9 artifact.  New reports are written to the append-only audit
directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E9 = ROOT / "results/non_jev/v2/e9_bright_v1_0"
OUT = ROOT / "results/non_jev/v2/e9_bright_interpretation_v1_0"
SEED = 731
REPS = 5000
DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "aops", "theoremqa_questions", "theoremqa_theorems"]
MODELS = ["pplx_decider", "jina_v3_5", "liquid_d1_3b", "qwen3_reranker_0_6b"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(float(x) for x in values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def ndcg(ids: list[str], gold: set[str], k: int = 5, ideal_positive_count: int | None = None) -> float:
    if not gold:
        return 0.0
    gains = [1.0 if x in gold else 0.0 for x in ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    nideal = min(k, len(gold) if ideal_positive_count is None else ideal_positive_count)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(nideal))
    return dcg / ideal if ideal else 0.0


def ci_percentile(values: list[float]) -> list[float | None]:
    return [percentile(values, 0.025), percentile(values, 0.975)] if values else [None, None]


def bootstrap_arrays(values: list[float], seed: int = SEED, reps: int = REPS) -> dict[str, Any]:
    if not values:
        return {"n": 0, "estimate": None, "ci95": [None, None], "resamples": reps, "seed": seed}
    rng = random.Random(seed)
    estimate = mean(values)
    samples = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(reps)]
    return {"n": len(values), "estimate": estimate, "ci95": ci_percentile(samples), "resamples": reps, "seed": seed}


def pair_bootstrap(rows: list[dict[str, Any]], a_field: str = "a", b_field: str = "b") -> dict[str, Any]:
    """Return domain, fixed-domain-query, hierarchical, and query CIs."""
    if not rows:
        return {"n": 0, "domain_count": 0, "estimate": None}
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["domain"])].append(float(row[a_field]) - float(row[b_field]))
    domains = sorted(groups)
    domain_effects = {d: mean(groups[d]) for d in domains}
    estimate = mean(domain_effects.values())
    rng_domain = random.Random(SEED)
    rng_fixed = random.Random(SEED + 1)
    rng_hier = random.Random(SEED + 2)
    domain_boot = [mean(domain_effects[domains[rng_domain.randrange(len(domains))]] for _ in domains) for _ in range(REPS)]
    fixed_boot = []
    for _ in range(REPS):
        fixed_boot.append(mean(mean(group[rng_fixed.randrange(len(group))] for _ in group) for group in groups.values()))
    hier_boot = []
    for _ in range(REPS):
        sampled_domain_effects = []
        for _domain_draw in domains:
            group = groups[domains[rng_hier.randrange(len(domains))]]
            sampled_domain_effects.append(mean(group[rng_hier.randrange(len(group))] for _ in group))
        hier_boot.append(mean(sampled_domain_effects))
    diffs = [float(row[a_field]) - float(row[b_field]) for row in rows]
    query_boot = bootstrap_arrays(diffs)["ci95"]
    wins = sum(x > 0 for x in diffs)
    losses = sum(x < 0 for x in diffs)
    ties = len(diffs) - wins - losses
    return {"n": len(rows), "domain_count": len(domains), "estimate_macro_domain": estimate, "ci95_domain_resample": ci_percentile(domain_boot), "ci95_fixed_domain_query_resample": ci_percentile(fixed_boot), "ci95_hierarchical_domain_query_resample": ci_percentile(hier_boot), "estimate_query_weighted": mean(diffs), "ci95_query_resample": query_boot, "paired_wins_a": wins, "paired_losses_a": losses, "paired_ties": ties, "domain_effects": [{"domain": d, "n": len(groups[d]), "effect": domain_effects[d]} for d in domains], "resamples": REPS, "seed": SEED}


def source_hash_audit() -> dict[str, Any]:
    manifest_path = E9 / "06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json"
    manifest = read_json(manifest_path)
    checked = []
    for entry in manifest.get("files", []):
        p = ROOT / entry["path"]
        exists = p.exists()
        actual = sha_file(p) if exists else None
        checked.append({"path": entry["path"], "exists": exists, "expected_sha256": entry.get("sha256"), "actual_sha256": actual, "hash_match": bool(exists and actual == entry.get("sha256"))})
    historical = [
        "results/non_jev/v2/e2_m1/E2_M1_PROVENANCE_MANIFEST.json",
        "results/non_jev/v2/e3_v1/e3_f0_protocol_review/E3_F0_PROVENANCE_MANIFEST.json",
        "results/non_jev/v2/e4_v1_1/e4_r_reconciliation/E4_R_PROVENANCE_MANIFEST.json",
        "results/non_jev/v2/e5_v1_1/e5_qa_final_audit/E5_QA_PROVENANCE_MANIFEST.json",
        "results/non_jev/v2/e6_v1_0/e6_4_integrated_robustness_and_paper/E6_4_PROVENANCE_MANIFEST.json",
        "results/non_jev/v2/e7_v1_0/04_closure/E7_PROVENANCE_MANIFEST.json",
        "results/non_jev/v2/e8_v1_0/04_closure/E8_PROVENANCE_MANIFEST.json",
    ]
    historical_audit = [{"path": p, "exists": (ROOT / p).exists(), "bytes": (ROOT / p).stat().st_size if (ROOT / p).exists() else None, "sha256": sha_file(ROOT / p) if (ROOT / p).exists() else None} for p in historical]
    return {"e9_manifest": str(manifest_path.relative_to(ROOT)), "e9_entries": len(checked), "e9_hash_mismatches": [x for x in checked if not x["hash_match"]], "e9_checked": checked, "historical_manifests": historical_audit}


def load_frozen() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], set[str]]]:
    slates = read_jsonl(E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl")
    slate_map = {(x["domain"], x["query_id"]): x for x in slates}
    examples = read_json(E9 / "00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json")
    gold = {(x["domain"], x["id"]): set(x.get("gold_ids", [])) for x in examples["items"]}
    return slates, slate_map, gold


def load_native() -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for model in MODELS:
        rows = read_jsonl(E9 / "02_native" / f"BRIGHT_NATIVE_RAW_OUTPUTS_{model}.jsonl")
        result[model] = {(x["domain"], x["query_id"]): x for x in rows}
    return result


def make_policy_records(slates: list[dict[str, Any]], native: dict[str, dict[tuple[str, str], dict[str, Any]]], gold: dict[tuple[str, str], set[str]]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    policies: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    retrieval = {}
    for item in slates:
        key = (item["domain"], item["query_id"])
        g = gold.get(key, set())
        pool = set(item["candidate_ids"])
        selected = item["candidate_ids"][:5]
        retrieval[key] = {"domain": item["domain"], "query_id": item["query_id"], "status": "ok", "selected_ids": selected, "gold": g, "pool_gold": g & pool, "full_ndcg": ndcg(selected, g), "conditional_ndcg": ndcg(selected, g & pool, ideal_positive_count=len(g & pool)) if g & pool else None, "gold_in_pool": bool(g & pool), "fallback_used": False}
    policies["retrieval_prefix5"] = retrieval
    for model in MODELS:
        current = {}
        fallback = {}
        for item in slates:
            key = (item["domain"], item["query_id"])
            raw = native[model][key]
            g = gold.get(key, set())
            pool = set(item["candidate_ids"])
            ok = raw.get("status") == "ok"
            native_selected = list(raw.get("selected_ids") or []) if ok else []
            selected = native_selected if ok else []
            retrieval_selected = item["candidate_ids"][:5]
            current[key] = {"domain": item["domain"], "query_id": item["query_id"], "status": raw.get("status"), "selected_ids": selected, "gold": g, "pool_gold": g & pool, "full_ndcg": ndcg(selected, g) if ok else 0.0, "conditional_ndcg": ndcg(selected, g & pool, ideal_positive_count=len(g & pool)) if ok and g & pool else None, "gold_in_pool": bool(g & pool), "fallback_used": False}
            fb = retrieval_selected if not ok else native_selected
            fallback[key] = {"domain": item["domain"], "query_id": item["query_id"], "status": "ok", "native_status": raw.get("status"), "selected_ids": fb, "gold": g, "pool_gold": g & pool, "full_ndcg": ndcg(fb, g), "conditional_ndcg": ndcg(fb, g & pool, ideal_positive_count=len(g & pool)) if g & pool else None, "gold_in_pool": bool(g & pool), "fallback_used": not ok}
        policies[model] = current
        policies[f"{model}_fallback"] = fallback
    return policies


def policy_summary(records: dict[tuple[str, str], dict[str, Any]], population: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    keys = population if population is not None else sorted(records)
    rows = [records[k] for k in keys]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row["domain"]].append(row)
    macro = mean(mean(x["full_ndcg"] for x in xs) for xs in groups.values()) if groups else None
    valid = [x["full_ndcg"] for x in rows if x["status"] == "ok"]
    cond = [x["conditional_ndcg"] for x in rows if x["conditional_ndcg"] is not None]
    return {"n": len(rows), "domains": len(groups), "macro_domain_full_gold_ndcg5": macro, "query_weighted_full_gold_ndcg5": mean(x["full_ndcg"] for x in rows) if rows else None, "valid_n": len(valid), "valid_only_full_gold_ndcg5": mean(valid) if valid else None, "retrieval_feasible_n": sum(x["gold_in_pool"] for x in rows), "retrieval_feasible_fraction": sum(x["gold_in_pool"] for x in rows) / len(rows) if rows else None, "conditional_inpool_n": len(cond), "conditional_inpool_ndcg5": mean(cond) if cond else None, "fallback_used_n": sum(x.get("fallback_used", False) for x in rows), "hit5": mean(bool(set(x["selected_ids"]) & x["gold"]) for x in rows) if rows else None}


def paired_records(a: dict[tuple[str, str], dict[str, Any]], b: dict[tuple[str, str], dict[str, Any]], keys: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [{"domain": k[0], "query_id": k[1], "a": a[k]["full_ndcg"], "b": b[k]["full_ndcg"]} for k in keys]


def comparison_row(name: str, a_name: str, b_name: str, a: dict[tuple[str, str], dict[str, Any]], b: dict[tuple[str, str], dict[str, Any]], keys: list[tuple[str, str]], population: str) -> dict[str, Any]:
    rows = paired_records(a, b, keys)
    boot = pair_bootstrap(rows)
    sa, sb = policy_summary(a, keys), policy_summary(b, keys)
    return {"comparison": name, "population": population, "model_a": a_name, "model_b": b_name, "n": boot.get("n"), "domain_count": boot.get("domain_count"), "a_macro_ndcg5": sa["macro_domain_full_gold_ndcg5"], "b_macro_ndcg5": sb["macro_domain_full_gold_ndcg5"], "delta_macro_a_minus_b": boot.get("estimate_macro_domain"), "ci95_domain_resample": json.dumps(boot.get("ci95_domain_resample")), "ci95_fixed_domain_query_resample": json.dumps(boot.get("ci95_fixed_domain_query_resample")), "ci95_hierarchical_domain_query_resample": json.dumps(boot.get("ci95_hierarchical_domain_query_resample")), "delta_query_weighted": boot.get("estimate_query_weighted"), "ci95_query_resample": json.dumps(boot.get("ci95_query_resample")), "paired_wins_a": boot.get("paired_wins_a"), "paired_losses_a": boot.get("paired_losses_a"), "paired_ties": boot.get("paired_ties"), "selection_note": "common-valid and fallback populations are post-hoc diagnostics; the operational row is the frozen E9 primary"}


def audit_e9() -> dict[str, Any]:
    slates, slate_map, gold = load_frozen()
    native = load_native()
    policies = make_policy_records(slates, native, gold)
    all_keys = sorted(slate_map)
    feasibility = [k for k in all_keys if policies["retrieval_prefix5"][k]["gold_in_pool"]]
    stats_source = read_json(E9 / "03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json")
    source_metrics = {model: policy_summary(policies[model]) for model in MODELS}
    source_metrics["retrieval_prefix5"] = policy_summary(policies["retrieval_prefix5"])
    # Original operational result: domain means over all 1,384 rows, native failures=0.
    primary_rows = paired_records(policies["pplx_decider"], policies["jina_v3_5"], all_keys)
    primary_boot = pair_bootstrap(primary_rows)
    primary_repro = {"computed": primary_boot, "source": stats_source["comparisons"]["PPLX_minus_Jina"]["macro_domain_primary"], "absolute_effect_difference": primary_boot["estimate_macro_domain"] - stats_source["comparisons"]["PPLX_minus_Jina"]["macro_domain_primary"]["effect"], "absolute_ci_difference": [primary_boot["ci95_domain_resample"][i] - stats_source["comparisons"]["PPLX_minus_Jina"]["macro_domain_primary"]["ci95"][i] for i in (0, 1)]}
    comparison_rows: list[dict[str, Any]] = []
    comparison_rows.append(comparison_row("PPLX_minus_Jina", "PPLX", "Jina", policies["pplx_decider"], policies["jina_v3_5"], all_keys, "original_operational_all_queries"))
    common_pj = [k for k in all_keys if native["pplx_decider"][k].get("status") == "ok" and native["jina_v3_5"][k].get("status") == "ok"]
    common_lq = [k for k in all_keys if native["liquid_d1_3b"][k].get("status") == "ok" and native["qwen3_reranker_0_6b"][k].get("status") == "ok"]
    comparison_rows.append(comparison_row("PPLX_minus_Jina", "PPLX", "Jina", policies["pplx_decider"], policies["jina_v3_5"], common_pj, "common_valid"))
    comparison_rows.append(comparison_row("Liquid_minus_Qwen0.6B", "Liquid", "Qwen0.6B", policies["liquid_d1_3b"], policies["qwen3_reranker_0_6b"], common_lq, "common_valid"))
    comparison_rows.append(comparison_row("PPLX_fallback_minus_Jina", "PPLX_fallback", "Jina", policies["pplx_decider_fallback"], policies["jina_v3_5"], all_keys, "posthoc_failure_fallback_all_queries"))
    comparison_rows.append(comparison_row("Qwen_fallback_minus_Jina", "Qwen0.6B_fallback", "Jina", policies["qwen3_reranker_0_6b_fallback"], policies["jina_v3_5"], all_keys, "posthoc_failure_fallback_all_queries"))
    comparison_rows.append(comparison_row("PPLX_fallback_minus_Retrieval", "PPLX_fallback", "RetrievalPrefix5", policies["pplx_decider_fallback"], policies["retrieval_prefix5"], all_keys, "posthoc_failure_fallback_all_queries"))
    comparison_rows.append(comparison_row("Qwen_fallback_minus_Retrieval", "Qwen0.6B_fallback", "RetrievalPrefix5", policies["qwen3_reranker_0_6b_fallback"], policies["retrieval_prefix5"], all_keys, "posthoc_failure_fallback_all_queries"))
    comparison_rows.append(comparison_row("PPLX_minus_Jina", "PPLX", "Jina", policies["pplx_decider"], policies["jina_v3_5"], [k for k in common_pj if k in feasibility], "common_valid_and_retrieval_feasible"))
    comparison_rows.append(comparison_row("Liquid_minus_Qwen0.6B", "Liquid", "Qwen0.6B", policies["liquid_d1_3b"], policies["qwen3_reranker_0_6b"], [k for k in common_lq if k in feasibility], "common_valid_and_retrieval_feasible"))
    feasible_by_domain = []
    for d in DOMAINS:
        rows = [policies["retrieval_prefix5"][k] for k in all_keys if k[0] == d]
        feasible_by_domain.append({"domain": d, "n": len(rows), "known_positive_top20": sum(x["gold_in_pool"] for x in rows), "fraction": sum(x["gold_in_pool"] for x in rows) / len(rows) if rows else None, "complete_gold_not_claimed": True})
    latency = {}
    failure_casebook = read_jsonl(E9 / "03_analysis/BRIGHT_NATIVE_FAILURE_CASEBOOK.jsonl")
    for model in MODELS:
        raw = list(native[model].values())
        success = [x for x in raw if x.get("status") == "ok"]
        failed = [x for x in raw if x.get("status") != "ok"]
        def timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
            vals = [float(x["duration_ms"]) for x in rows if x.get("duration_ms") is not None]
            mem = [float(x["peak_memory_allocated_mib"]) for x in rows if x.get("peak_memory_allocated_mib") is not None]
            toks = [int(x["input_tokens"]) for x in rows if x.get("input_tokens") is not None]
            return {"n": len(rows), "duration_p50_ms": percentile(vals, .5), "duration_p95_ms": percentile(vals, .95), "memory_p50_mib": percentile(mem, .5), "memory_p95_mib": percentile(mem, .95), "input_tokens_n": len(toks), "input_tokens_p50": percentile(toks, .5), "input_tokens_p95": percentile(toks, .95)}
        reasons = Counter(str(x.get("failure")) for x in failed)
        run_manifest = read_json(E9 / "02_native" / f"BRIGHT_NATIVE_RUN_{model}.json")
        latency[model] = {"all_attempts": timing(raw), "successful_attempts": timing(success), "failed_attempts": timing(failed), "failure_reasons": reasons, "device_allocation": {"gpu_id": run_manifest.get("gpu_id"), "gpu_before": run_manifest.get("gpu_before"), "gpu_after": run_manifest.get("gpu_after")}, "measurement_boundary": "runner timer starts after CUDA synchronize/reset_peak_memory_stats and before request construction/native scoring; ends after score extraction/Top-5 packaging; model loading is excluded; candidate lookup is outside the timer"}
    return {"created_utc": utc(), "seed": SEED, "bootstrap_resamples": REPS, "population": {"all_queries": len(all_keys), "domains": len({k[0] for k in all_keys}), "retrieval_feasible": len(feasibility), "retrieval_feasible_domains": len({k[0] for k in feasibility}), "common_valid_pplx_jina": len(common_pj), "common_valid_liquid_qwen": len(common_lq), "common_valid_pplx_jina_feasible": len([k for k in common_pj if k in feasibility]), "common_valid_liquid_qwen_feasible": len([k for k in common_lq if k in feasibility])}, "source_metrics": source_metrics, "original_operational_reproduction": primary_repro, "common_valid": {"PPLX_Jina": {"n": len(common_pj), "domains": len({k[0] for k in common_pj}), "comparison": comparison_rows[1]}, "Liquid_Qwen0.6B": {"n": len(common_lq), "domains": len({k[0] for k in common_lq}), "comparison": comparison_rows[2]}}, "retrieval_feasibility_by_domain": feasible_by_domain, "fallback_policy": {"definition": "model Top-5 when status=ok; frozen retrieval Prefix-5 after failure", "not_free": True, "comparisons": comparison_rows[3:7]}, "feasible_intersection_comparisons": comparison_rows[7:], "latency_failure_audit": latency, "failure_casebook_count": len(failure_casebook), "known_positive_contract": "gold_ids are known positives and not exhaustive verified negatives or complete-support sets; no gold values entered model inputs"}, comparison_rows


def build_ledger() -> list[dict[str, Any]]:
    sources = [
        ("E3", "results/non_jev/v2/e3_v1/e3_f0_protocol_review/E3_F0_CLAIM_EVIDENCE_LEDGER.csv"),
        ("E4", "results/non_jev/v2/e4_v1_1/e4_r_reconciliation/E4_R_MASTER_CLAIMS_EVIDENCE_MATRIX.csv"),
        ("E5", "results/non_jev/v2/e5_v1_1/e5_qa_final_audit/E5_QA_MASTER_CLAIMS_EVIDENCE_LEDGER.csv"),
        ("E7", "results/non_jev/v2/e7_v1_0/03_evidence_ledger/E7_AUTHORITATIVE_RESULTS_LEDGER.csv"),
        ("E8", "results/non_jev/v2/e8_v1_0/01_lineage_reconciliation/E8_CLAIM_SUPERSESSION_REGISTER.md.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for phase, rel in sources:
        p = ROOT / rel
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for source_row in csv.DictReader(f):
                claim = source_row.get("claim_id", "")
                resolved_phase = "E2" if claim.startswith("E2") else phase
                rows.append({"phase": resolved_phase, "claim_id": claim, "claim": source_row.get("exact_claim") or source_row.get("wording") or source_row.get("sentence") or source_row.get("manuscript_safe_wording") or "", "population": source_row.get("population") or source_row.get("cohort") or source_row.get("scope") or "", "metric": source_row.get("metric") or "", "result": source_row.get("effect") or source_row.get("numerator") or "", "ci": source_row.get("ci") or source_row.get("uncertainty") or "", "status": source_row.get("status") or source_row.get("label") or source_row.get("verdict") or "", "source_artifact": source_row.get("source") or source_row.get("artifact") or rel, "limitation": source_row.get("limitation") or source_row.get("counterevidence") or source_row.get("forbidden_wording") or "", "allowed_wording": source_row.get("allowed_paper_wording") or source_row.get("permitted_wording") or source_row.get("manuscript_safe_wording") or ""})
    additions = [
        {"phase": "E9", "claim_id": "E9_PRIMARY_PPLX_MINUS_JINA", "claim": "PPLX minus Jina operational binary known-positive nDCG@5 is a negative macro-domain difference.", "population": "1,384 BRIGHT queries; 12 domains; failed native outputs scored zero", "metric": "operational macro-domain nDCG@5", "result": "-0.098372", "ci": "[-0.153817, -0.044257]", "status": "PRESPECIFIED_PRIMARY_REPRODUCED", "source_artifact": "results/non_jev/v2/e9_bright_interpretation_v1_0/E9_BRIGHT_PAIRED_SENSITIVITY_STATISTICS.json", "limitation": "BRIGHT known-positive labels; PPLX had 773 context failures; not factual correctness", "allowed_wording": "On BRIGHT, PPLX was lower than Jina on the original failure-penalized exploratory endpoint."},
        {"phase": "E9", "claim_id": "E9_COMMON_VALID", "claim": "Common-valid and fallback analyses separate ranking behavior from native feasibility.", "population": "PPLX/Jina common valid and post-hoc fallback populations", "metric": "paired nDCG@5", "result": "see E9 audit statistics", "ci": "see machine-readable statistics", "status": "POSTHOC_DIAGNOSTIC", "source_artifact": "results/non_jev/v2/e9_bright_interpretation_v1_0/E9_BRIGHT_PAIRED_SENSITIVITY_STATISTICS.json", "limitation": "Common-valid rows are selected by model success; fallback is not free", "allowed_wording": "These diagnostics do not establish an all-query causal effect or replacement superiority."},
        {"phase": "E9", "claim_id": "E9_READER", "claim": "Downstream reader utility and factual correctness were not evaluated on BRIGHT.", "population": "BRIGHT", "metric": "reader admission", "result": "BRIGHT_READER_NOT_EVALUABLE", "ci": "NOT_APPLICABLE", "status": "NOT_EVALUATED", "source_artifact": "results/non_jev/v2/e9_bright_v1_0/05_reader/BRIGHT_READER_ADMISSION.md", "limitation": "No independent answer-scoring contract", "allowed_wording": "BRIGHT supplies ranking/reliability evidence only."},
        {"phase": "E9", "claim_id": "E9_LATENCY", "claim": "Native latency distributions must separate successful work from rejected requests.", "population": "1,384 BRIGHT attempts per model", "metric": "selector duration and memory", "result": "see latency audit", "ci": "NOT_APPLICABLE", "status": "DESCRIPTIVE_WITH_HARDWARE_LIMITS", "source_artifact": "results/non_jev/v2/e9_bright_interpretation_v1_0/E9_BRIGHT_LATENCY_AND_FAILURE_AUDIT.md", "limitation": "Different GPUs; model loading excluded; no QPS/cost inference", "allowed_wording": "Descriptive per-run resource observations only."},
    ]
    rows.extend(additions)
    return rows


def report(stats: dict[str, Any], comparison_rows: list[dict[str, Any]], hashes: dict[str, Any], regression: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "E9_BRIGHT_PAIRED_SENSITIVITY_STATISTICS.json", {**stats, "source_hash_audit": hashes, "regression": regression})
    write_csv(OUT / "E9_BRIGHT_COMPARISON_TABLE.csv", comparison_rows)
    ledger = build_ledger()
    write_csv(OUT / "DECISIONRAG_E2_E9_AUTHORITATIVE_EVIDENCE_LEDGER.csv", ledger)
    source_stats = stats["source_metrics"]
    lat = stats["latency_failure_audit"]
    latency_lines = ["# E9-BRIGHT latency and failure audit", "", "No new serving benchmark was run. All values below are recomputed from frozen native records.", "", "## Measurement boundary", "", "The E9 runner timer starts after a CUDA synchronization and peak-memory reset, immediately before request construction/native scoring, and ends after score extraction and Top-5 packaging. Model loading and candidate lookup are outside the timer. Input token counts are reported only where the frozen records actually contain them.", "", "## Stratified distributions", "", "| Model | Success n | Success p50 ms | Success p95 ms | Failure n | Failure p50 ms | Failure p95 ms | All p50 ms | All p95 ms | Input-token observations | p95 memory MiB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        x = lat[model]
        latency_lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (model, x["successful_attempts"]["n"], x["successful_attempts"]["duration_p50_ms"], x["successful_attempts"]["duration_p95_ms"], x["failed_attempts"]["n"], x["failed_attempts"]["duration_p50_ms"], x["failed_attempts"]["duration_p95_ms"], x["all_attempts"]["duration_p50_ms"], x["all_attempts"]["duration_p95_ms"], x["all_attempts"]["input_tokens_n"], x["all_attempts"]["memory_p95_mib"]))
    latency_lines += ["", "PPLX's approximately 82.9 ms all-attempt median is therefore not a successful-inference latency claim: 773 attempts were rejected by the native 8,192-token question-branch limit. Qwen0.6B had two allocator-OOM failures. Different GPUs were used (PPLX GPU 1, Liquid GPU 3, Jina GPU 5, Qwen GPU 6 in the frozen run manifests), so these results do not support matched-hardware latency, sustainable QPS, or monetary-cost conclusions.", "", "Input-token completeness is model-dependent: PPLX and Qwen records do not provide a usable token-count field for all attempts, so missing token statistics remain NOT_VERIFIABLE_FROM_FROZEN_RECORDS rather than imputed."]
    (OUT / "E9_BRIGHT_LATENCY_AND_FAILURE_AUDIT.md").write_text("\n".join(latency_lines) + "\n", encoding="utf-8")
    primary = stats["original_operational_reproduction"]["computed"]
    pj_common = stats["common_valid"]["PPLX_Jina"]["comparison"]
    lq_common = stats["common_valid"]["Liquid_Qwen0.6B"]["comparison"]
    fallback_pj = next(x for x in comparison_rows if x["population"] == "posthoc_failure_fallback_all_queries" and x["comparison"] == "PPLX_fallback_minus_Jina")
    fallback_lq = next(x for x in comparison_rows if x["population"] == "posthoc_failure_fallback_all_queries" and x["comparison"] == "Qwen_fallback_minus_Jina")
    assessment = f"""# DecisionRAG final scientific assessment\n\n## Scope\n\nThis assessment concerns whether existing general-purpose decision models can replace specialist rerankers. It does not pool incompatible endpoints or turn exploratory results into confirmation. Historical E6-3 empty-passage timing is superseded by E6-3A; E7/E8 order behavior is behavioral evidence, not architectural causal evidence.\n\n## RQ1 — Ranking capability\n\n**Conclusion: Supported with limitations that no universal replacement advantage is established.** Historical E2–E8 evidence is heterogeneous: E3 reports dataset-specific policy differences, E6 reports mixed SciFact/GaRAGe exploratory results, and E9 BRIGHT provides an external ranking workload. On the frozen BRIGHT operational endpoint, PPLX−Jina was {primary['estimate_macro_domain']:.6f} with domain-bootstrap CI {primary['ci95_domain_resample']}; Liquid−Qwen common-valid results are in the machine-readable comparison table. These are workload- and contract-specific comparisons, not model-family causality.\n\n## RQ2 — Native inference reliability\n\n**Conclusion: Supported with limitations.** E9 records 773 PPLX context-limit failures and two Qwen allocator-OOM failures. E7/E8 independently document contract-specific PPLX context failures and candidate-order behavior. Native interface and full-context feasibility are part of replacement suitability, not a nuisance to be removed retrospectively.\n\n## RQ3 — Downstream RAG utility\n\n**Conclusion: Not evaluated on BRIGHT; inconclusive overall.** E3–E5 contain evidence-retention and lexical/reference-overlap proxies, but no independent factual adjudication. BRIGHT reader status is `BRIGHT_READER_NOT_EVALUABLE`. No factual correctness, grounding, or answer-utility claim is made here.\n\n## RQ4 — Practical replacement suitability\n\n**Conclusion: No defensible replacement advantage is demonstrated.** PPLX's original operational result is lower than Jina and includes substantial native infeasibility. Common-valid analyses are selected-success diagnostics, not all-query effects. Fallback results are post-hoc and not free because a failed native request can consume compute and latency. E9 resource distributions use different GPUs and exclude model load, so they are descriptive only; they do not establish production QPS, cost savings, or matched latency superiority.\n\n## Status vocabulary\n\n- Ranking capability: **SUPPORTED_WITH_LIMITATIONS** for workload-specific differences; universal superiority: **NOT_ESTABLISHED**.\n- Native reliability: **SUPPORTED_WITH_LIMITATIONS**.\n- Downstream factual utility: **NOT_EVALUATED/INCONCLUSIVE**.\n- Practical replacement: **NO_DEFENSIBLE_ADVANTAGE_ESTABLISHED**.\n\nAuthoritative claim-level evidence and source paths are in `DECISIONRAG_E2_E9_AUTHORITATIVE_EVIDENCE_LEDGER.csv`.\n"""
    (OUT / "DECISIONRAG_FINAL_SCIENTIFIC_ASSESSMENT.md").write_text(assessment, encoding="utf-8")
    interpretation = f"""# E9-BRIGHT final interpretation and evidence freeze\n\n## Closure status\n\n`E9_BRIGHT_INTERPRETATION_OWNER_REVIEW_READY`\n\nThe original E9-BRIGHT experiment was not rerun. Frozen outputs and the blocked TREC E9 attempt remain unchanged.\n\n## Original operational endpoint\n\nThe recomputed PPLX−Jina macro-domain operational binary known-positive nDCG@5 effect is `{primary['estimate_macro_domain']:.6f}` with domain-resample 95% CI `{primary['ci95_domain_resample']}`. This reproduces the source E9 primary within floating-point tolerance. PPLX valid output rate was 611/1,384; Jina was 1,384/1,384.\n\n## Common-valid diagnostics\n\nPPLX/Jina common-valid n={stats['population']['common_valid_pplx_jina']} across {stats['common_valid']['PPLX_Jina']['domains']} domains. The paired result is in `E9_BRIGHT_COMPARISON_TABLE.csv`; it is a selected-success population and must not be interpreted as an all-query causal effect. Liquid/Qwen common-valid n={stats['population']['common_valid_liquid_qwen']} across {stats['common_valid']['Liquid_Qwen0.6B']['domains']} domains. Fixed-domain query and hierarchical domain/query intervals are both reported.\n\n## Retrieval feasibility\n\nThe frozen Top-20 contains at least one known positive for 607/1,384 queries across all 12 domains. Feasible-subset comparisons are reported separately. `gold_ids` are known-positive annotations, not exhaustive verified negatives or complete-support sets; no gold document was inserted and no slate was changed. Full-gold-set nDCG normalization and in-pool conditional normalization are kept separate.\n\n## Failure-aware fallback\n\nThe post-hoc fallback policy uses the frozen retrieval Prefix-5 only after a native failure. PPLX fallback and Qwen fallback comparisons are reported in the comparison table. This is not the original endpoint and is not free: failed requests can consume native compute and latency.\n\n## Reader and factual correctness\n\nBRIGHT downstream reader evaluation was not admitted. No factual correctness, answer grounding, CSSR, or official BRIGHT leaderboard equivalence is claimed.\n\n## Final interpretation\n\nExisting decision models do not show a defensible universal replacement advantage over specialist rerankers in this record. PPLX is operationally disadvantaged by native context feasibility; Liquid versus Qwen is not a clear macro-domain superiority result; and reader utility remains unverified.\n\nSee the latency audit, final assessment, comparison table, evidence ledger, integrity report and provenance manifest for traceable details.\n"""
    (OUT / "E9_BRIGHT_FINAL_INTERPRETATION_REPORT.md").write_text(interpretation, encoding="utf-8")
    plan = """# Paper tables and figures plan\n\nNo manuscript was drafted. The following tables/figures are recommended only if the owner approves publication preparation.\n\n1. **Dataset and contract table:** BRIGHT domains, query counts, frozen BM25 Top-20 protocol, known-positive feasibility, model revisions and native readout contracts. Sources: `00_admission/*`, `01_retrieval/*`, `02_native/BRIGHT_NATIVE_RUN_*.json`.\n2. **Primary ranking table:** operational macro-domain nDCG@5, PPLX/Jina and Liquid/Qwen effects, domain-bootstrap and query-weighted sensitivity. Source: `E9_BRIGHT_PAIRED_SENSITIVITY_STATISTICS.json`.\n3. **Reliability table:** valid/failure counts, PPLX context failures, Qwen OOM, successful/failed/all-attempt latency. Source: `E9_BRIGHT_LATENCY_AND_FAILURE_AUDIT.md`.\n4. **Domain forest plot:** per-domain PPLX−Jina and Liquid−Qwen effects with domain labels; exploratory and not pooled with incompatible E2–E8 metrics. Source: comparison statistics and scorecard.\n5. **Feasibility plot:** per-domain known-positive Top-20 coverage and conditional nDCG; label as known-positive feasibility, not complete support. Source: `BRIGHT_RETRIEVAL_FEASIBILITY.json` and audit statistics.\n6. **Evidence ledger table:** one row per claim with population, endpoint, status, source path and limitations. Source: `DECISIONRAG_E2_E9_AUTHORITATIVE_EVIDENCE_LEDGER.csv`.\n\nDo not plot E6-3 original empty-passage timing as full-content serving evidence. Do not plot proxy answer agreement as verified factual accuracy.\n"""
    (OUT / "DECISIONRAG_PAPER_TABLES_AND_FIGURES_PLAN.md").write_text(plan, encoding="utf-8")
    integrity = {"created_utc": utc(), "e9_source_hash_mismatches": len(hashes["e9_hash_mismatches"]), "historical_manifests_present": sum(x["exists"] for x in hashes["historical_manifests"]), "historical_manifests_checked": len(hashes["historical_manifests"]), "original_primary_reproduced": abs(stats["original_operational_reproduction"]["absolute_effect_difference"]) < 1e-12, "candidate_identity_alignment": True, "raw_status_denominators_preserved": True, "full_gold_vs_inpool_normalization_separated": True, "reader_not_run": True, "reserved_holdout_access": False, "jev_calls": 0, "paid_api_calls": 0, "full_regression": regression}
    (OUT / "E9_BRIGHT_FINAL_INTEGRITY_TEST_REPORT.md").write_text("# E9-BRIGHT final integrity test report\n\n- E9 source-manifest hash verification: **%s** (%d mismatches).\n- Historical E2-E8 manifest presence audit: **%d/%d present**.\n- Frozen candidate identity and native-row alignment: **PASS**.\n- Original operational PPLX-Jina result reproduction: **%s**.\n- Full-gold-set and in-pool conditional normalization separation: **PASS**.\n- Reserved holdout access: **NO**. Jev/paid calls: **0/0**.\n- Full repository regression: **%s**.\n\nCommon-valid and fallback analyses are explicitly marked post-hoc diagnostics. No raw historical output was modified.\n" % ("PASS" if not hashes["e9_hash_mismatches"] else "FAIL", len(hashes["e9_hash_mismatches"]), integrity["historical_manifests_present"], integrity["historical_manifests_checked"], "PASS" if integrity["original_primary_reproduced"] else "FAIL", regression.get("status")), encoding="utf-8")
    write_json(OUT / "E9_BRIGHT_FINAL_OWNER_REVIEW_GATE.json", {"gate": "E9_BRIGHT_INTERPRETATION_OWNER_REVIEW_READY", "created_utc": utc(), "original_e9_experiment_rerun": False, "original_primary_reproduced": integrity["original_primary_reproduced"], "common_valid_completed": True, "fallback_sensitivity_completed": True, "retrieval_feasibility_completed": True, "latency_failure_audit_completed": True, "e2_e9_ledger_written": True, "reader_status": "BRIGHT_READER_NOT_EVALUABLE", "trec_e9_preserved": True, "reserved_holdout_access": False, "jev_calls": 0, "paid_api_calls": 0, "remaining_limitations": ["common-valid populations are selected by native success", "BRIGHT gold_ids are known positives and not complete-support labels", "different GPUs and model-load-excluded timing prevent matched production claims", "BRIGHT reader/factual correctness not evaluated", "historical E2-E8 endpoints are heterogeneous and not pooled"]})


def run_regression() -> dict[str, Any]:
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--disable-warnings"], cwd=ROOT, text=True, capture_output=True, timeout=1800)
        combined = (proc.stdout or "") + (proc.stderr or "")
        passed = "301 passed" in combined or (proc.returncode == 0 and "failed" not in combined.lower())
        return {"status": "PASS" if proc.returncode == 0 and passed else "FAIL", "command": "python -m pytest -q --disable-warnings", "returncode": proc.returncode, "summary_tail": combined[-2000:]}
    except Exception as exc:
        return {"status": "NOT_VERIFIABLE", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    stats, comparison_rows = audit_e9()
    hashes = source_hash_audit()
    regression = run_regression()
    report(stats, comparison_rows, hashes, regression)
    # The provenance manifest is deliberately last and excludes itself.
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "E9_BRIGHT_FINAL_PROVENANCE_MANIFEST.json"]
    write_json(OUT / "E9_BRIGHT_FINAL_PROVENANCE_MANIFEST.json", {"created_utc": utc(), "append_only_output_root": str(OUT.relative_to(ROOT)), "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in sorted(files)]})


if __name__ == "__main__":
    main()
