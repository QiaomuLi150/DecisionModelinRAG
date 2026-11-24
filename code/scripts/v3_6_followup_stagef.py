#!/usr/bin/env python3
"""DecisionRAG v3.6 follow-up Stage F: protocol freeze and CPU corrections."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential"
FOLLOW = OUT / "followup_v1"
QOLD = OUT / "stage1_quality_v2/export/quality"
COHORT = OUT / "stage0_preflight/cohort_manifest.csv"
PROTOCOL_OLD = OUT / "stage0_preflight/PROTOCOL.json"
SEED = 731
DETERMINISM_SEED = 1731
REPS = 40_000
LAT_REPS = 10_000
DOMAINS = ["aops", "biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "theoremqa_questions", "theoremqa_theorems"]


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def protocol():
    cohorts = read_csv(COHORT)
    c120 = [r["canonical_id"] for r in cohorts if str(r["in_C120"]).lower() == "true"]
    c360 = [r["canonical_id"] for r in cohorts]
    input_manifest = {
        "bright": {"source": "historical E9 frozen BRIGHT Top-20", "sha256": "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2", "queries": 1384, "candidate_count": 20},
        "cohort_manifest": {"path": "results/v3_6_essential/stage0_preflight/cohort_manifest.csv", "sha256": sha(COHORT), "rows": len(cohorts)},
        "c120": {"ids": c120, "count": len(c120)},
        "c360": {"ids": c360, "count": len(c360)},
        "call": {"ids": "all 1,384 BRIGHT canonical IDs from frozen slate", "count": 1384, "source_hash": "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2"},
    }
    write_json(FOLLOW / "INPUT_MANIFEST_FOLLOWUP.json", input_manifest)
    p = {
        "experiment": "DecisionRAG v3.6 follow-up corrections, full-population timing and frozen confirmation",
        "status": "FROZEN_PRE_OUTCOME",
        "protocol_version": "v3.6-followup-v1",
        "rules": {"label_blind": True, "no_retries": True, "failures_retained": True, "no_bm25_fill_in_raw": True, "no_new_downloads": True, "no_new_retrieval": True, "append_only": True, "paper_edited": False},
        "definitions": {"model_valid": "valid five distinct original candidate IDs without fallback", "builder_infeasible": "budget builder error containing UNBUDGETABLE or BudgetUnavailable:UNBUDGETABLE", "fallback_used": "not model_valid", "fallback_basis": "failed requests score ndcg5_bm25_fallback; R_fallback=100% by construction", "no_fallback_basis": "failed requests score ndcg5; R_strict=model_valid_pct"},
        "release_checks": {"Q": "whole-domain bootstrap lower 95% bound of policy-minus-jina_full > -0.02", "S": "same-session p95 ratio; upper 95% bound <=2 pass, lower >2 fail, otherwise inconclusive", "thresholds": [1.5, 2.0, 3.0], "quality_bootstrap": {"method": "resample 12 domains with replacement, concatenate selected-domain query differences, query-weighted", "draws": REPS, "seed": SEED}, "sign_flip": "exact 2^12 domain-mean sign flip; Holm within Q1/Q2 families", "latency_bootstrap": {"cluster": "canonical_id", "draws": LAT_REPS, "seed": SEED}},
        "models": {"jina_v3_5": {"revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "parameters_b": 0.6, "runtime": "native local Jina reranker"}, "pplx_decider": {"revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "parameters_b": 27, "runtime": "native local decision model"}, "historical_models": ["winnow_native", "winnow_budget", "drex_v2_native", "liquid_d1_default", "qwen3_reranker_0_6b", "mxbai_rerank_large_v2", "qwen3_reranker_4b"]},
        "policies": {"n1_call": ["jina_full", "jina_single_8192", "jina_single_6144", "jina_two_stage_8192", "pplx_single_2048", "pplx_single_3072", "pplx_single_4096", "pplx_single_6144", "pplx_budget", "pplx_two_stage_4096"], "n2": ["winnow_native", "winnow_budget", "drex_v2_native", "liquid_d1_default", "qwen3_reranker_0_6b", "mxbai_rerank_large_v2", "qwen3_reranker_4b"]},
        "schedules": {"n1": {"cohort": "CALL", "queries": 1384, "attempts": 2, "concurrency": 1, "replicate_sessions": 2, "anchors": ["jina_full", "pplx_budget"], "model_order": ["Jina policies then release Jina then PPLX policies"]}, "n2": {"cohort": "C360", "attempts": 3, "concurrency": 1, "anchors": ["jina_full", "pplx_budget"], "hardware_reduction_if_needed": True}},
        "confirmation": {"pick": "single-call L=4096 for PPLX and Jina; unchanged L=8192 reference", "fixed_before_transfer_outcomes": True, "transfer_pipeline": "v2.1 replication pipeline unchanged", "equivalence_rule": "if L4096 strings hash-identical to L8192 for >=99% of requests, skip identical requests and run only differing requests"},
        "serving_contract": {"timed_boundary": "prompt construction + native calls + parsing + original-ID mapping + Top-5 packaging", "model_loading_excluded": True, "failure_time_retained": True, "fallback_latency_formula": "raw latency + measured fallback packaging ms on failed rows only", "dtype_backend_cache_batch_threads": "inherit pinned model registry/runtime; record actual session receipts"},
        "gpu_policy": {"slots": ["A", "B", "C"], "foreign_processes_untouched": True, "initial_free_slots": ["A", "B"], "device_indices_excluded_from_exports": True},
        "hashes": {"old_protocol_sha256": sha(PROTOCOL_OLD), "cohort_manifest_sha256": sha(COHORT)},
        "freeze_statement": "All thresholds, cohort choices, policy lists, confirmation pick, seeds, bootstrap definitions and session schedules were frozen before any new outcome or transfer inference.",
    }
    raw = json.dumps(p, indent=2, sort_keys=True) + "\n"
    (FOLLOW / "PROTOCOL_FOLLOWUP.json").write_text(raw, encoding="utf-8")
    (FOLLOW / "PROTOCOL_FOLLOWUP.sha256").write_text(hashlib.sha256(raw.encode()).hexdigest() + "  PROTOCOL_FOLLOWUP.json\n", encoding="utf-8")
    return input_manifest, p


def boolish(v):
    return str(v).lower() in {"1", "true", "pass", "ok", "success"}


def builder_flag(r):
    e = f"{r.get('error_class','')} {r.get('error','')}".upper()
    return int("UNBUDGETABLE" in e or "BUDGETUNAVAILABLE" in e)


def quality_rows_v2():
    rows = read_csv(QOLD / "q_perquery.csv")
    out = []
    for r in rows:
        valid = boolish(r["valid"])
        b = builder_flag(r)
        out.append({"canonical_id": r["canonical_id"], "domain": r["domain"], "policy": r["policy"], "model": r["model"], "mode": r["mode"], "budget_L": r["budget_L"], "order_id": r["order_id"], "model_valid": int(valid), "builder_infeasible": b, "fallback_used": int(not valid), "error_class": r["error_class"], "ndcg5": float(r["ndcg5"]), "ndcg5_bm25_fallback": float(r["ndcg5_bm25_fallback"]), "n_calls": r["n_calls"]})
    fields = list(out[0])
    write_csv(FOLLOW / "export/quality_v2/q_perquery_v2.csv", fields, out)
    return out


def groups_for(rows, policy, budget, basis):
    return {r["canonical_id"]: (r["ndcg5_bm25_fallback"] if basis == "fallback" else r["ndcg5"]) for r in rows if r["policy"] == policy and str(r["budget_L"]) == str(budget)}


def domain_bootstrap(diffs, dom):
    grouped = defaultdict(list)
    for cid, d in zip(diffs.keys(), diffs.values()): grouped[dom[cid]].append(float(d))
    keys = sorted(grouped)
    sums = np.asarray([sum(grouped[k]) for k in keys], dtype=np.float64)
    counts = np.asarray([len(grouped[k]) for k in keys], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(keys), size=(REPS, len(keys)))
    vals = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    # Original within-domain bootstrap retained for side-by-side audit.
    within = np.zeros(REPS, dtype=np.float64)
    for k in keys:
        a = np.asarray(grouped[k], dtype=np.float64)
        ids = rng.integers(0, len(a), size=(REPS, len(a)))
        within += a[ids].mean(axis=1) / len(keys)
    effects = np.asarray([np.mean(grouped[k]) for k in keys], dtype=np.float64)
    signs = np.asarray([(-1 if (mask >> i) & 1 else 1) for mask in range(1 << len(effects)) for i in range(len(effects))], dtype=np.float64).reshape(1 << len(effects), len(effects))
    sf = np.mean(np.abs((signs * effects).mean(axis=1)) >= abs(effects.mean()) - 1e-15)
    return {"lo95": float(np.quantile(vals, .025)), "hi95": float(np.quantile(vals, .975)), "lo90": float(np.quantile(vals, .05)), "hi90": float(np.quantile(vals, .95)), "withindomain_lo95": float(np.quantile(within, .025)), "withindomain_hi95": float(np.quantile(within, .975)), "p_signflip": float(sf), "domains_positive": int(sum(effects > 0)), "domains": len(keys)}


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values); last = 0.0
    for rank, idx in enumerate(order):
        last = max(last, min(1.0, (len(values) - rank) * values[idx])); out[idx] = last
    return out


def quality_summary(rows):
    out = []
    for key in sorted({(r["policy"], r["model"], r["mode"], r["budget_L"]) for r in rows}):
        rs = [r for r in rows if (r["policy"], r["model"], r["mode"], r["budget_L"]) == key]
        out.append({"policy": key[0], "model": key[1], "mode": key[2], "budget_L": key[3], "n": len(rs), "model_valid_pct": sum(r["model_valid"] for r in rs) / len(rs), "fallback_share": sum(r["fallback_used"] for r in rs) / len(rs), "builder_infeasible_share": sum(r["builder_infeasible"] for r in rs) / len(rs), "mean_ndcg5_nofallback": np.mean([r["ndcg5"] for r in rs]), "mean_ndcg5_fallback": np.mean([r["ndcg5_bm25_fallback"] for r in rs]), "mean_ndcg5_modelvalid_only": np.mean([r["ndcg5"] for r in rs if r["model_valid"]]) if any(r["model_valid"] for r in rs) else None})
    write_csv(FOLLOW / "export/quality_v2/q_summary_v2.csv", list(out[0]), out)


def contrasts(rows):
    baseline = {(r["canonical_id"]): r for r in rows if r["policy"] == "jina_full"}
    dom = {cid: r["domain"] for cid, r in baseline.items()}
    specs = [("P_L-J_L", "pplx_single", "jina_single", ["1024", "2048", "3072", "4096", "6144"]), ("P2_L-J2_L", "pplx_two_stage", "jina_two_stage", ["2048", "4096"])]
    out = []
    for family, pa, pb, budgets in specs:
        for L in budgets:
            for basis in ("fallback", "strict"):
                a = groups_for(rows, pa, L, basis); b = groups_for(rows, pb, L, basis); ids = sorted(set(a) & set(b) & set(baseline))
                diffs = {cid: a[cid] - b[cid] for cid in ids}; st = domain_bootstrap(diffs, dom)
                out.append({"contrast": f"{pa}_{L}-{pb}_{L}", "family": family, "basis": basis, "delta_query_weighted": float(np.mean(list(diffs.values()))), "lo95": st["lo95"], "hi95": st["hi95"], "lo90": st["lo90"], "hi90": st["hi90"], "p_signflip": st["p_signflip"], "p_holm": "PENDING", "domains_positive": st["domains_positive"], "bootstrap_method": "whole_domain_resample_query_weighted", "withindomain_lo95": st["withindomain_lo95"], "withindomain_hi95": st["withindomain_hi95"]})
    for family in ("P_L-J_L", "P2_L-J2_L"):
        for basis in ("fallback", "strict"):
            idx = [i for i, r in enumerate(out) if r["family"] == family and r["basis"] == basis]
            vals = holm([out[i]["p_signflip"] for i in idx])
            for i, v in zip(idx, vals): out[i]["p_holm"] = v
    fields = ["contrast", "family", "basis", "delta_query_weighted", "lo95", "hi95", "lo90", "hi90", "p_signflip", "p_holm", "domains_positive", "bootstrap_method", "withindomain_lo95", "withindomain_hi95"]
    write_csv(FOLLOW / "export/quality_v2/q_contrasts_v2.csv", fields, out)
    return out


def noninferiority(rows):
    baseline = {r["canonical_id"]: r for r in rows if r["policy"] == "jina_full"}
    dom = {cid: r["domain"] for cid, r in baseline.items()}
    policies = [("pplx_single", str(L)) for L in [1024, 2048, 3072, 4096, 6144, 8192]] + [("pplx_two_stage", str(L)) for L in [2048, 4096, 8192]] + [("jina_single", str(L)) for L in [1024, 2048, 3072, 4096, 6144, 8192]] + [("jina_two_stage", str(L)) for L in [2048, 4096, 8192]]
    out = []
    for p, L in policies:
        for basis in ("fallback", "strict"):
            a = groups_for(rows, p, L, basis); b = groups_for(rows, "jina_full", "", basis); ids = sorted(set(a) & set(b)); diffs = {cid: a[cid] - b[cid] for cid in ids}; st = domain_bootstrap(diffs, dom)
            out.append({"policy": p if L == "" else f"{p}_{L}", "basis": basis, "delta_vs_jina_full": np.mean(list(diffs.values())), "lo95": st["lo95"], "hi95": st["hi95"], "pass_0.02": st["lo95"] > -.02, "pass_0.01": st["lo95"] > -.01, "pass_0.005": st["lo95"] > -.005})
    write_csv(FOLLOW / "export/quality_v2/noninferiority_v2.csv", list(out[0]), out)


def robustness(rows):
    base = {r["canonical_id"]: r for r in rows if r["policy"] == "jina_full"}
    dom = {cid: r["domain"] for cid, r in base.items()}
    specs = [("pplx_single", str(L)) for L in [2048, 3072, 4096, 6144, 8192]] + [("pplx_two_stage", str(L)) for L in [4096, 8192]] + [("jina_single", str(L)) for L in [2048, 3072, 4096, 6144, 8192]] + [("jina_two_stage", str(L)) for L in [2048, 4096, 8192]]
    per = []; loo = []
    for p, L in specs:
        a = groups_for(rows, p, L, "fallback"); b = groups_for(rows, "jina_full", "", "fallback"); ids = sorted(set(a)&set(b)); gd=defaultdict(list)
        for cid in ids: gd[dom[cid]].append(a[cid]-b[cid])
        effects={d:float(np.mean(v)) for d,v in gd.items()}; per.append({"policy":f"{p}_{L}","domains_positive":sum(v>0 for v in effects.values()),"domains_total":len(effects),**{d:effects.get(d) for d in DOMAINS}})
        for excluded in [None] + sorted(effects):
            vals=[a[cid]-b[cid] for cid in ids if dom[cid] != excluded]; loo.append({"policy":f"{p}_{L}","excluded_domain":excluded or "NONE","delta_query_weighted":float(np.mean(vals))})
    write_csv(FOLLOW / "export/quality_v2/per_domain_deltas.csv", list(per[0]), per)
    ranges=[]
    for p in sorted({x["policy"] for x in loo}):
        rs=[x for x in loo if x["policy"]==p and x["excluded_domain"]!="NONE"]; ranges.append({"policy":p,"min_delta_excluding_one_domain":min(x["delta_query_weighted"] for x in rs),"max_delta_excluding_one_domain":max(x["delta_query_weighted"] for x in rs),"full_delta":next(x["delta_query_weighted"] for x in loo if x["policy"]==p and x["excluded_domain"]=="NONE")})
    write_csv(FOLLOW / "export/quality_v2/leave_one_domain_out.csv", list(ranges[0]), ranges)


def crash_log():
    # The historical placeholder is not copied.  Check raw traces for an
    # explicit server-crash/restart marker; ordinary request failures are not
    # server crashes and remain in per-request traces.
    events=[]
    for p in sorted(OUT.glob("stage2_serving*/raw/*/serving_perrequest_raw.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                r=json.loads(line); e=f"{r.get('error_class','')} {r.get('error','')}".lower()
                if any(x in e for x in ["server_crash", "restart", "cuda out of memory"]): events.append({"source":str(p.relative_to(ROOT)),"canonical_id":r.get("canonical_id"),"policy":r.get("policy"),"error_class":r.get("error_class") or r.get("error")})
    write_csv(FOLLOW / "export/serving_full/crash_log.csv", ["source", "canonical_id", "policy", "error_class"], events)
    (FOLLOW / "export/serving_full/crash_log_note.md").write_text(f"# Crash log note\n\nChecked all historical v3.6 T1 raw serving traces. Explicit server-crash/restart/OOM markers found: **{len(events)}**. Ordinary request failures remain in the per-request exports and are not relabeled as server crashes.\n", encoding="utf-8")


def reports(rows, contrasts_rows):
    expected = {"pplx_single_2048": (-0.0022, -0.0195, 0.0138), "pplx_single_4096": (0.0036, -0.0139, 0.0203), "pplx_single_8192": (0.0088, -0.0121, 0.0295), "jina_single_4096": (-0.0202, -0.0401, -0.0048)}
    # Compare the requested independent figures to the recomputation.  The
    # result is intentionally recorded rather than altered to match them.
    qrows=read_csv(FOLLOW / "export/quality_v2/noninferiority_v2.csv")
    checks=[]
    for name, target in expected.items():
        base=next((r for r in qrows if r["policy"]==name and r["basis"]=="fallback"),None)
        if base: checks.append({"policy":name,"expected":target,"observed":(float(base["delta_vs_jina_full"]),float(base["lo95"]),float(base["hi95"])),"within_0.0005":all(abs(float(a)-float(b))<=.0005 for a,b in zip((base["delta_vs_jina_full"],base["lo95"],base["hi95"]),target))})
    write_json(FOLLOW / "FIGURE_REPRODUCTION_CHECK.json", {"checks":checks,"all_within_0.0005":all(x["within_0.0005"] for x in checks),"note":"The declared whole-domain query-weighted bootstrap is authoritative; any mismatch is reported, not tuned."})
    (FOLLOW / "REPORT_INTERIM.md").write_text("""# DecisionRAG v3.6 follow-up interim report — Stage F

Stage F CPU corrections are complete. The new quality schema separates
`model_valid`, `builder_infeasible`, and `fallback_used`; fallback-basis and
strict no-fallback scores are both exported. Quality intervals use the frozen
40,000-draw whole-domain query-weighted bootstrap (seed 731); the old
within-domain intervals are retained in `q_contrasts_v2.csv`. Exact 2^12
domain sign flips and Holm correction are included. The independent figure
reproduction result is in `FIGURE_REPRODUCTION_CHECK.json`.

Jina wrapper rerun, full-population timing, N2 timing, transfer confirmation,
and final Stage-4/5 reports are not yet executed in this interim artifact.
""", encoding="utf-8")
    (FOLLOW / "DEVIATIONS.md").write_text("""# v3.6 follow-up deviations

- Earlier v3.6 outputs are preserved unchanged. Follow-up outputs are append-only.
- The old `crash_log.csv` placeholder is not reused; the follow-up crash log is
  regenerated from raw traces and ordinary request failures are not treated as
  server crashes.
- Device indices and UUIDs are retained only in raw provenance where present;
  derived exports use slot letters A/B/C and are sanitized before packaging.
""", encoding="utf-8")


def main():
    FOLLOW.mkdir(parents=True, exist_ok=True)
    protocol()
    rows=quality_rows_v2(); quality_summary(rows); c=contrasts(rows); noninferiority(rows); robustness(rows); crash_log(); reports(rows,c)
    write_json(FOLLOW / "STATE.json", {"stageF":{"status":"done","counts":{"quality_rows":len(rows)},"resume_command":"python scripts/v3_6_followup_stagef.py"},"stage1":{"status":"pending"},"stage2":{"status":"pending"},"stage3":{"status":"pending"},"stage4":{"status":"pending"},"stage5":{"status":"pending"}})
    write_json(FOLLOW / "STATUS.json", {"stageF":"PASS","stage1":"PENDING","stage2":"PENDING","stage3":"PENDING","stage4":"PENDING","stage5":"PENDING"})
    print(json.dumps({"status":"STAGE_F_DONE","quality_rows":len(rows),"followup":str(FOLLOW.relative_to(ROOT))},indent=2))


if __name__ == "__main__": main()
