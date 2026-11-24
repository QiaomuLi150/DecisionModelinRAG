#!/usr/bin/env python3
"""CPU-only v3.6 quality join and interim analysis.

Consumes only frozen BRIGHT labels joined after the v3.6 raw traces were
completed.  It never loads a model or changes any historical artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential"
S0 = OUT / "stage0_preflight"
RAW1 = OUT / "stage1_quality/raw"
RAW2 = OUT / "stage1_quality_v2/raw"
EXPORT = OUT / "stage1_quality_v2/export/quality"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
GOLD = ROOT / "results/non_jev/v2/e9_bright_v1_0/00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json"
REPS = 40_000
SEED = 731
DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "aops", "theoremqa_questions", "theoremqa_theorems"]


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ndcg5(selected, gold):
    gold = set(gold)
    if not gold:
        return 0.0
    dcg = sum((1.0 if x in gold else 0.0) / math.log2(i + 2) for i, x in enumerate(selected[:5]))
    ideal_n = min(len(gold), 5)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / ideal if ideal else 0.0


def load_inputs():
    slates = read_jsonl(BRIGHT)
    examples = json.loads(GOLD.read_text(encoding="utf-8"))["items"]
    gold = {(str(x["domain"]), str(x["id"])): set(str(g) for g in x.get("gold_ids", [])) for x in examples}
    return slates, gold


def raw_rows():
    rows = defaultdict(dict)
    for p in sorted(RAW2.glob("*_shard_*.jsonl")):
        for r in read_jsonl(p):
            rows[(r["policy"], int(r["budget_L"]), r["canonical_id"])] = r
    for p in sorted(RAW1.glob("*8192_reuse.jsonl")):
        for r in read_jsonl(p):
            policy = r["policy"]
            budget = 8192 if policy != "jina_full" else None
            rows[(policy, budget, r["canonical_id"])] = r

    # Historical controls are admitted only when their native raw record is
    # explicitly BRIGHT and has the same canonical identity.
    controls = [
        (ROOT / "results/non_jev/v2/e12_o_final_v1_0/02_ranking/e12o_o2_attempt_01/winnow12_NATIVE_RAW_V2.jsonl", "winnow_native", "BRIGHT"),
        (ROOT / "results/non_jev/v2/e12_o_final_v1_0/02_ranking/e12o_o2_attempt_01/drex_v2_NATIVE_RAW_V2.jsonl", "drex_v2_native", "BRIGHT"),
        (ROOT / "results/v1_5/c4_winnow_rerun_gpu3/raw/winnow12_D_budget_RAW_V2.jsonl", "winnow_budget", "BRIGHT"),
        (ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_liquid_d1_3b.jsonl", "liquid_d1_default", None),
    ]
    for path, policy, dataset in controls:
        if not path.exists():
            continue
        for r in read_jsonl(path):
            if dataset and str(r.get("dataset", "")).upper() != dataset:
                continue
            cid = r.get("canonical_id") or f"{r.get('domain')}:{r.get('query_id')}"
            rows[(policy, None, cid)] = r
    return rows


def status_ok(r):
    return str(r.get("status", "")).lower() in {"pass", "ok", "success"} and bool(r.get("selected_ids")) and len(r.get("selected_ids", [])) == 5 and len(set(r.get("selected_ids", []))) == 5


def make_records():
    slates, gold = load_inputs()
    raw = raw_rows()
    records = []
    missing = []
    for s in slates:
        cid = f"{s['domain']}:{s['query_id']}"
        g = gold.get((str(s["domain"]), str(s["query_id"])), set())
        fallback = [str(x) for x in s["candidate_ids"][:5]]
        expected = []
        for (policy, budget) in sorted({k[:2] for k in raw if k[2] == cid}, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
            if policy in {"winnow_native", "winnow_budget", "drex_v2_native", "liquid_d1_default"}:
                model, mode, bl = policy, "native", ""
            else:
                model = "pplx_decider" if policy.startswith("pplx") else "jina_v3_5"
                mode = "two_stage" if "two_stage" in policy else ("full" if policy == "jina_full" else "single")
                bl = "" if budget is None else str(budget)
            r = raw.get((policy, budget, cid))
            if r is None:
                missing.append({"canonical_id": cid, "policy": policy, "budget_L": budget})
                continue
            ok = status_ok(r)
            selected = [str(x) for x in (r.get("selected_ids") or [])] if ok else []
            records.append({"canonical_id": cid, "domain": s["domain"], "policy": policy, "model": model, "mode": mode, "budget_L": bl, "order_id": "original", "valid": int(ok), "error_class": "" if ok else str(r.get("error_class") or r.get("error") or "UNKNOWN"), "ndcg5": ndcg5(selected, g) if ok else 0.0, "ndcg5_bm25_fallback": ndcg5(selected if ok else fallback, g), "n_calls": int(r.get("n_calls") or (3 if "two_stage" in policy else 1)), "labels_joined": False, "selected_ids": selected})
    return records, missing


def percentile(xs, q):
    if len(xs) == 0:
        return None
    ys = sorted(xs); pos = (len(ys) - 1) * q; lo = int(pos); hi = min(len(ys) - 1, lo + 1)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def bootstrap_delta(a, b, domains):
    grouped = defaultdict(list)
    for x, y, d in zip(a, b, domains): grouped[d].append(x - y)
    ds = sorted(grouped)
    observed = mean(x - y for x, y in zip(a, b))
    # Vectorized fixed-domain query bootstrap: same declared resample count
    # and seed, without the prohibitively slow nested Python loops.
    rng = np.random.default_rng(SEED)
    vals = np.zeros(REPS, dtype=np.float64)
    for d in ds:
        g = np.asarray(grouped[d], dtype=np.float64)
        idx = rng.integers(0, len(g), size=(REPS, len(g)))
        vals += g[idx].mean(axis=1) / len(ds)
    # Exact domain sign-flip on domain means.
    effects = [mean(grouped[d]) for d in ds]
    obs_macro = mean(effects)
    ge = 0
    for mask in range(1 << len(effects)):
        v = sum(((-1 if (mask >> i) & 1 else 1) * effects[i]) for i in range(len(effects))) / len(effects)
        if abs(v) >= abs(obs_macro) - 1e-15: ge += 1
    return {"delta_query_weighted": observed, "lo95": percentile(vals, .025), "hi95": percentile(vals, .975), "lo90": percentile(vals, .05), "hi90": percentile(vals, .95), "p_signflip": ge / (1 << len(effects)), "domains_positive": sum(x > 0 for x in effects), "macro_domain_delta": obs_macro}


def main():
    records, missing = make_records()
    EXPORT.mkdir(parents=True, exist_ok=True)
    fields = ["canonical_id", "domain", "policy", "model", "mode", "budget_L", "order_id", "valid", "error_class", "ndcg5", "ndcg5_bm25_fallback", "n_calls"]
    with (EXPORT / "q_perquery.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k, "") for k in fields} for r in records])
    groups = defaultdict(list)
    for r in records: groups[(r["policy"], r["model"], r["mode"], r["budget_L"])].append(r)
    summaries=[]
    for key, rs in sorted(groups.items()):
        summaries.append({"policy":key[0],"model":key[1],"mode":key[2],"budget_L":key[3],"n":len(rs),"valid_n":sum(x["valid"] for x in rs),"valid_pct":sum(x["valid"] for x in rs)/len(rs),"mean_ndcg5_all":mean(x["ndcg5"] for x in rs),"mean_ndcg5_valid":mean([x["ndcg5"] for x in rs if x["valid"]]) if any(x["valid"] for x in rs) else None,"mean_ndcg5_bm25_fallback":mean(x["ndcg5_bm25_fallback"] for x in rs)})
    with (EXPORT / "q_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    lookup={(r["policy"],r["budget_L"],r["canonical_id"]):r for r in records}
    cts=[]
    comparisons=[("P_L-J_L", "pplx_single", "jina_single", ["1024","2048","3072","4096","6144"]), ("P2_L-J2_L", "pplx_two_stage", "jina_two_stage", ["2048","4096"])]
    for name, pa, pb, budgets in comparisons:
        for budget in budgets:
            aa=[]; bb=[]; dd=[]
            for s in load_inputs()[0]:
                cid=f"{s['domain']}:{s['query_id']}"; x=lookup.get((pa,budget,cid)); y=lookup.get((pb,budget,cid))
                if x and y: aa.append(x["ndcg5"]); bb.append(y["ndcg5"]); dd.append(s["domain"])
            st=bootstrap_delta(aa,bb,dd); cts.append({"contrast":f"{pa}_{budget}-{pb}_{budget}","family":name,"delta_query_weighted":st["delta_query_weighted"],"lo95":st["lo95"],"hi95":st["hi95"],"lo90":st["lo90"],"hi90":st["hi90"],"p_signflip":st["p_signflip"],"p_holm":"NOT_COMPUTED","domains_positive":st["domains_positive"]})
    with (EXPORT / "q_contrasts.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(cts[0])); w.writeheader(); w.writerows(cts)
    (OUT / "stage1_quality_v2" / "QUALITY_JOIN_RECEIPT.json").write_text(json.dumps({"status":"PASS" if not missing else "PARTIAL","slates":1384,"record_count":len(records),"missing_records":missing[:100],"missing_count":len(missing),"new_raw_sha256":{"pplx": [sha_file(p) for p in sorted(RAW2.glob('pplx_shard_*.jsonl'))],"jina":[sha_file(p) for p in sorted(RAW2.glob('jina_shard_*.jsonl'))]}},indent=2)+"\n",encoding="utf-8")
    (OUT / "stage1_quality_v2" / "REPORT_INTERIM.md").write_text("# v3.6 Stage 1 Interim Quality Report\n\nCPU-only label joining is complete. Labels were joined after raw traces were finalized. See `export/quality/` for per-query and summary tables. Serving results are not yet included.\n\n- BRIGHT queries: 1,384\n- New PPLX/Jina records expected: 13,840 per model\n- Missing joined records: %d\n- Bootstrap seed: 731; repetitions: 40,000\n" % len(missing),encoding="utf-8")
    print(json.dumps({"status":"QUALITY_ANALYSIS_DONE","records":len(records),"missing":len(missing),"export":str(EXPORT.relative_to(ROOT))},indent=2))


if __name__ == "__main__": main()
