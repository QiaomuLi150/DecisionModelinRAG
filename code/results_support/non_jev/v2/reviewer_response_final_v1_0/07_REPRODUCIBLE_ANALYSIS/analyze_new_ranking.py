#!/usr/bin/env python3
"""Offline analysis of the reviewer-response ranking traces."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/non_jev/v2/reviewer_response_final_v1_0"
RANK = OUT / "ranking"
CPU = OUT / "cpu"


def read_jsonl(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ndcg(ids, gold):
    if not gold:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, x in enumerate(ids[:5]) if x in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(5, len(gold))))
    return dcg / ideal if ideal else 0.0


def key(r):
    return r.get("canonical_id") or f"{r.get('domain')}:{r.get('query_id')}"


def load_map(p, dataset=None):
    rows = read_jsonl(p)
    if dataset is not None:
        rows = [r for r in rows if r.get("dataset", "BRIGHT") == dataset]
    return {key(r): r for r in rows}


def load_inputs():
    slates = read_jsonl(ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl")
    # Use the cached official examples Arrow files for the full frozen gold_ids.
    # The post-freeze sidecar is retained for feasibility reporting, but scoring
    # with only its in-slate intersection would inflate nDCG for selected IDs.
    from datasets import Dataset
    examples_root = ROOT / "data/external/e9_bright_v1_0/hf_cache/xlangai___bright/examples/0.0.0/3066d29c9651a576c8aba4832d249807b181ecae"
    c = {}
    for p in sorted(examples_root.glob("bright-*.arrow")):
        domain_name = p.stem[len("bright-"):]
        ds = Dataset.from_file(str(p))
        for r in ds:
            c[f"{domain_name}:{r['id']}"] = set(str(x) for x in (r.get("gold_ids") or []))
    domain = {f"{r['domain']}:{r['query_id']}": r["domain"] for r in slates}
    return slates, c, domain


def outcome(row, gold):
    selected = row.get("selected_ids") or []
    valid = row.get("status") in {"PASS", "ok", "pass"} and len(selected) == 5 and len(set(selected)) == 5
    return ndcg(selected, gold) if valid else 0.0, valid


def bootstrap(diffs, domains, reps=5000, seed=731):
    if not diffs:
        return {"n": 0, "estimate": None, "ci95": [None, None], "reps": reps, "seed": seed}
    obs = statistics.mean(diffs)
    rng = random.Random(seed)
    q = [statistics.mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(reps)]
    q.sort()
    by = defaultdict(list)
    for x, d in zip(diffs, domains):
        by[d].append(x)
    dkeys = sorted(by)
    dm = [statistics.mean(by[d]) for d in dkeys]
    rng = random.Random(seed)
    dg = [statistics.mean(dm[rng.randrange(len(dm))] for _ in dm) for _ in range(reps)]
    dg.sort()
    rng = random.Random(seed)
    fd = []
    for _ in range(reps):
        vals = []
        for d in dkeys:
            xs = by[d]
            vals.extend(xs[rng.randrange(len(xs))] for _ in xs)
        fd.append(statistics.mean(vals))
    fd.sort()
    q2 = lambda xs: [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]]
    return {"n": len(diffs), "estimate": obs, "query_weighted_ci95": q2(q), "equal_domain_ci95": q2(dg), "fixed_domain_query_ci95": q2(fd), "domain_count": len(dkeys), "domain_estimates": {d: statistics.mean(by[d]) for d in dkeys}, "reps": reps, "seed": seed}


def load_policy_rows():
    return {
        "jina_full": load_map(ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl"),
        "pplx_native": load_map(ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_pplx_decider.jsonl"),
        "pplx_budget": load_map(ROOT / "results/non_jev/v2/e11_v1_0/02_native/E11_A_PPLX_BUDGETED_RAW.jsonl"),
        "jina_pplx_budget": load_map(ROOT / "results/non_jev/v2/e11_v1_0/02_native/E11_A_JINA_BUDGETED_RAW.jsonl"),
        "kev_native": load_map(ROOT / "results/non_jev/v2/e12_o_final_v1_0/02_ranking/e12o_o2_attempt_01/kev9_NATIVE_RAW_V2.jsonl", "BRIGHT"),
        "drex_native": load_map(ROOT / "results/non_jev/v2/e12_o_final_v1_0/02_ranking/e12o_o2_attempt_01/drex_dlm8_NATIVE_RAW_V2.jsonl", "BRIGHT"),
        "kev_budget": load_map(RANK / "04_RR_kev9_BUDGET_RAW.jsonl"),
        "drex_budget": load_map(RANK / "04_RR_drex_dlm8_BUDGET_RAW.jsonl"),
        "jina_kev_budget": load_map(RANK / "04_RR_jina_D_budget_kev9_RAW.jsonl"),
        "jina_drex_budget": load_map(RANK / "04_RR_jina_D_budget_drex_dlm8_RAW.jsonl"),
    }


def score_rows(rows, gold):
    out = {}
    for name, mp in rows.items():
        vals = {}
        valid = 0
        for k, r in mp.items():
            x, ok = outcome(r, gold.get(k, set()))
            vals[k] = x
            valid += int(ok)
        out[name] = {"values": vals, "valid": valid, "n": len(vals), "missing": 1384 - len(vals), "mean": statistics.mean(vals.values()) if vals else None}
    return out


def main():
    _slates, gold, domains = load_inputs()
    rows = load_policy_rows()
    scored = score_rows(rows, gold)
    score_rows_out = []
    for name in ["jina_full", "pplx_native", "pplx_budget", "jina_pplx_budget", "kev_native", "kev_budget", "jina_kev_budget", "drex_native", "drex_budget", "jina_drex_budget"]:
        x = scored[name]
        score_rows_out.append({"policy": name, "n": x["n"], "valid_top5": x["valid"], "invalid_or_missing": 1384 - x["valid"], "operational_ndcg5_all_attempts": x["mean"], "endpoint": "BRIGHT known-positive binary nDCG@5; failed output=0", "status": "MEASURED" if x["n"] == 1384 else "PARTIAL"})
    with (RANK / "04_RR_MODEL_SCORECARD.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=score_rows_out[0].keys())
        w.writeheader()
        w.writerows(score_rows_out)

    pairs = [
        ("pplx_budget_minus_jina_full", "pplx_budget", "jina_full"),
        ("pplx_budget_minus_jina_pplx_budget", "pplx_budget", "jina_pplx_budget"),
        ("pplx_budget_minus_pplx_native", "pplx_budget", "pplx_native"),
        ("jina_pplx_budget_minus_jina_full", "jina_pplx_budget", "jina_full"),
        ("kev_budget_minus_jina_full", "kev_budget", "jina_full"),
        ("kev_budget_minus_jina_kev_budget", "kev_budget", "jina_kev_budget"),
        ("kev_budget_minus_kev_native", "kev_budget", "kev_native"),
        ("jina_kev_budget_minus_jina_full", "jina_kev_budget", "jina_full"),
        ("drex_budget_minus_jina_full", "drex_budget", "jina_full"),
        ("drex_budget_minus_jina_drex_budget", "drex_budget", "jina_drex_budget"),
        ("drex_budget_minus_drex_native", "drex_budget", "drex_native"),
        ("jina_drex_budget_minus_jina_full", "jina_drex_budget", "jina_full"),
    ]
    comparison_rows = []
    effects = {}
    for name, a, b in pairs:
        keys = sorted(set(scored[a]["values"]) & set(scored[b]["values"]))
        diffs = [scored[a]["values"][k] - scored[b]["values"][k] for k in keys]
        bs = bootstrap(diffs, [domains[k] for k in keys])
        effects[name] = {"a": a, "b": b, "keys": keys, "diffs": diffs, "bootstrap": bs}
        comparison_rows.append({"comparison": name, "a": a, "b": b, "n_paired": len(keys), "a_mean_all_attempts": statistics.mean([scored[a]["values"][k] for k in keys]) if keys else None, "b_mean_all_attempts": statistics.mean([scored[b]["values"][k] for k in keys]) if keys else None, "effect": bs["estimate"], "query_weighted_ci95": json.dumps(bs.get("query_weighted_ci95")), "fixed_domain_query_ci95": json.dumps(bs.get("fixed_domain_query_ci95")), "equal_domain_ci95": json.dumps(bs.get("equal_domain_ci95")), "seed": 731, "reps": 5000, "status": "MEASURED" if len(keys) == 1384 else "PARTIAL"})
    with (RANK / "04_RR_OPERATIONAL_EFFECTS_AND_CIS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
        w.writeheader()
        w.writerows(comparison_rows)
    (RANK / "04_RR_OPERATIONAL_EFFECTS_AND_CIS.json").write_text(json.dumps(effects, indent=2) + "\n", encoding="utf-8")

    loo = []
    for cname in ["pplx_budget_minus_jina_full", "kev_budget_minus_jina_full", "drex_budget_minus_jina_full"]:
        e = effects[cname]
        by = defaultdict(list)
        for k, d in zip(e["keys"], e["diffs"]):
            by[domains[k]].append(d)
        allv = statistics.mean(e["diffs"])
        for d in sorted(by):
            rest = [x for dd, xs in by.items() if dd != d for x in xs]
            loo.append({"comparison": cname, "left_out_domain": d, "n_remaining": len(rest), "effect": statistics.mean(rest), "full_effect": allv})
    with (CPU / "02_LEAVE_ONE_DOMAIN_OUT.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=loo[0].keys())
        w.writeheader()
        w.writerows(loo)

    audit = []
    with (RANK / "04_RR_BUDGET_VIEW_HASHES.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            audit.append({"policy": r["policy_id"], "budget_cap": r["budget_cap"], "original_tokens_total": r["original_tokens_total"], "retained_tokens_total": r["retained_tokens_total"], "serialized_tokens": r["serialized_tokens"], "status": r["status"]})
    byp = defaultdict(list)
    for r in audit:
        byp[r["policy"]].append(r)
    with (CPU / "01_MODEL_CONTEXT_TOKENIZER_TRUNCATION_AUDIT.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["policy", "n", "budget_cap_p50", "budget_cap_p95", "original_tokens_total_p50", "retained_tokens_total_p50", "retention_fraction_mean", "serialized_tokens_p95", "pass", "fail_or_unbudgetable", "status"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p, rs in sorted(byp.items()):
            nums = lambda k: [int(x[k]) for x in rs if str(x.get(k, "")).strip() not in {"", "None", "null"}]
            ot, rt, caps, ser = nums("original_tokens_total"), nums("retained_tokens_total"), nums("budget_cap"), nums("serialized_tokens")
            w.writerow({"policy": p, "n": len(rs), "budget_cap_p50": statistics.median(caps) if caps else None, "budget_cap_p95": sorted(caps)[max(0, math.ceil(.95 * len(caps)) - 1)] if caps else None, "original_tokens_total_p50": statistics.median(ot) if ot else None, "retained_tokens_total_p50": statistics.median(rt) if rt else None, "retention_fraction_mean": statistics.mean([y / x for x, y in zip(ot, rt)]) if ot else None, "serialized_tokens_p95": sorted(ser)[max(0, math.ceil(.95 * len(ser)) - 1)] if ser else None, "pass": sum(x["status"] == "PASS" for x in rs), "fail_or_unbudgetable": sum(x["status"] != "PASS" for x in rs), "status": "MEASURED"})
    (CPU / "01_CAL_OVERLAP_AND_TECHNICAL_SETUP.md").write_text("""# CAL overlap and technical setup

- No new CAL sample was used for the reviewer-response ranking population. The new ranking protocol was frozen on the complete E9 BRIGHT cohort before reading new outcomes.
- Historical E12 native traces are reused only as D_native controls after query, slate and policy identity checks.
- The new budgeted views are label-blind; the known-positive sidecar is joined only by this offline analysis.
- New native calls used the local Kev and Drex-DLM adapters on owner-authorized GPU5/GPU6. Winnow's prior native HTTP runtime was not reachable in the current execution environment and is not represented as successful output.
""", encoding="utf-8")
    report = ["# Reviewer-response ranking analysis", "", "The authoritative operational endpoint is known-positive binary nDCG@5 on the full 1,384-query BRIGHT denominator; invalid or missing native selections score zero. The sidecar contains known positives intersecting the frozen Top-20, not exhaustive negatives or complete-support labels.", "", "## Scorecard", "", (RANK / "04_RR_MODEL_SCORECARD.csv").read_text(encoding="utf-8"), "", "## Comparisons", "", (RANK / "04_RR_OPERATIONAL_EFFECTS_AND_CIS.csv").read_text(encoding="utf-8"), "", "## Interpretation", "", "The PPLX historical budgeted row is the E11-compatible anchor. Kev and Drex-DLM budgeted outputs were physically executed under the frozen protocol; their matched Jina controls use the exact Unicode budget strings. Winnow is blocked by unavailable HTTP serving, and Liquid is not locally available, so the mandatory full gate cannot be claimed. These are exploratory workload-specific results, not architecture-causal or noninferiority evidence."]
    (OUT / "04_RR_RANKING_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "RANKING_ANALYSIS_COMPLETE", "scorecard": len(score_rows_out), "comparisons": len(comparison_rows)}, indent=2))


if __name__ == "__main__":
    main()
