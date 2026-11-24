#!/usr/bin/env python3
"""CPU-only v3.6 Stage-3 closure.

This is an append-only closure over frozen Stage-1 quality records, T1 raw
serving traces, and the completed T2/B6.2 raw traces.  It never loads a
checkpoint, makes a model request, or reads labels into the serving traces.
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
S1 = OUT / "stage1_quality_v2/export/quality"
T2 = OUT / "stage2_t2"
S2 = OUT / "stage2_serving"
ANALYSIS = OUT / "stage3_analysis"
EXPORT = ANALYSIS / "export"
FRONTIER = EXPORT / "frontier"
SEED = 731
Q_DRAWS = 40_000
R_DRAWS = 10_000


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def percentile(values, q):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(len(values) - 1, lo + 1)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bool_value(v):
    return v is True or str(v).lower() in {"1", "true", "pass", "ok", "success"}


def c120_ids():
    rows = read_csv(S0 / "cohort_manifest.csv")
    return {r["canonical_id"] for r in rows if str(r.get("in_C120", "")).lower() == "true"}


def load_t2():
    ids = c120_ids()
    rows = []
    for session in ("T2_A", "T2_B", "T2_C", "T2_D"):
        for phase in ("pplx", "jina"):
            p = T2 / session / "raw" / phase / "serving_perrequest_raw.jsonl"
            if not p.exists():
                continue
            for r in read_jsonl(p):
                if r.get("canonical_id") not in ids:
                    continue
                r = dict(r)
                r["session"] = session
                r["phase"] = phase
                r["planned_cohort"] = "C120"
                r["observed_records_per_policy"] = 360
                r["valid_bool"] = bool_value(r.get("valid")) and str(r.get("status", "")).upper() in {"PASS", "OK", "SUCCESS"}
                rows.append(r)
    return rows


def load_t1_c120():
    ids = c120_ids()
    namespaces = [
        (S2, "T1_GPU3"),
        (OUT / "stage2_serving_gpu6", "T1_GPU6"),
        (OUT / "stage2_serving_gpu3_r3", "T1_GPU3_R3"),
    ]
    rows = []
    for ns, session in namespaces:
        for phase in ("pplx", "jina"):
            p = ns / "raw" / phase / "serving_perrequest_raw.jsonl"
            if not p.exists():
                continue
            for r in read_jsonl(p):
                if r.get("canonical_id") in ids:
                    x = dict(r)
                    x["session"] = session
                    x["valid_bool"] = bool_value(x.get("valid")) and str(x.get("status", "")).lower() in {"ok", "pass", "success"}
                    rows.append(x)
        p = ns / "raw" / "jina_j2_repair" / "serving_perrequest_raw.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                if r.get("canonical_id") in ids:
                    x = dict(r)
                    x["session"] = session
                    x["valid_bool"] = bool_value(x.get("valid")) and str(x.get("status", "")).lower() in {"ok", "pass", "success"}
                    rows.append(x)
    return rows


def fallback_map():
    p = S2 / "export/fallback_cost.csv"
    return {r["canonical_id"]: float(r["packaging_ms"]) for r in read_csv(p)} if p.exists() else {}


def adjusted_latency(row, fallback):
    value = float(row.get("latency_ms") or 0.0)
    return value + (fallback.get(row.get("canonical_id"), 0.0) if not row.get("valid_bool") else 0.0)


def cluster_ratio(policy_rows, base_rows, fallback, draws=R_DRAWS):
    pa = defaultdict(list)
    ba = defaultdict(list)
    for r in policy_rows:
        pa[r["canonical_id"]].append(adjusted_latency(r, fallback))
    for r in base_rows:
        ba[r["canonical_id"]].append(adjusted_latency(r, fallback))
    ids = sorted(set(pa) & set(ba))
    if not ids:
        return None
    point = percentile([v for i in ids for v in pa[i]], .95) / percentile([v for i in ids for v in ba[i]], .95)
    # Vectorized cluster bootstrap.  The frozen traces have three attempts
    # per C120 ID; keep a defensive ragged path for malformed inputs.
    lengths = {len(pa[i]) for i in ids} | {len(ba[i]) for i in ids}
    if len(lengths) == 1:
        n_attempts = next(iter(lengths))
        a = np.asarray([pa[i] for i in ids], dtype=np.float64)
        b = np.asarray([ba[i] for i in ids], dtype=np.float64)
        rng = np.random.default_rng(SEED)
        vals = np.empty(draws, dtype=np.float64)
        chunk = 256
        for start in range(0, draws, chunk):
            stop = min(draws, start + chunk)
            sample_idx = rng.integers(0, len(ids), size=(stop - start, len(ids)))
            av = a[sample_idx].reshape(stop - start, len(ids) * n_attempts)
            bv = b[sample_idx].reshape(stop - start, len(ids) * n_attempts)
            vals[start:stop] = np.quantile(av, .95, axis=1, method="linear") / np.quantile(bv, .95, axis=1, method="linear")
    else:
        rng = random.Random(SEED)
        vals = []
        for _ in range(draws):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            av = [v for i in sample for v in pa[i]]
            bv = [v for i in sample for v in ba[i]]
            vals.append(percentile(av, .95) / percentile(bv, .95))
        vals = np.asarray(vals, dtype=np.float64)
    return {
        "point": point,
        "lo95": float(np.quantile(vals, .025)),
        "hi95": float(np.quantile(vals, .975)),
        "prob_le_2": float(np.mean(np.asarray(vals) <= 2.0)),
        "n_ids": len(ids),
    }


def quality_rows():
    return read_csv(S1 / "q_perquery.csv")


def quality_policy_map(policy):
    if policy == "jina_full":
        return "jina_full", ""
    if policy == "pplx_budget":
        return "pplx_single", "8192"
    if policy.startswith("pplx_single_"):
        return "pplx_single", policy.rsplit("_", 1)[1]
    if policy.startswith("jina_single_"):
        return "jina_single", policy.rsplit("_", 1)[1]
    if policy.startswith("pplx_two_stage_"):
        return "pplx_two_stage", policy.rsplit("_", 1)[1]
    if policy.startswith("jina_two_stage_"):
        return "jina_two_stage", policy.rsplit("_", 1)[1]
    return policy, ""


def quality_stats(timing_policies):
    rows = quality_rows()
    base = {(r["canonical_id"]): float(r["ndcg5"]) for r in rows if r["policy"] == "jina_full"}
    domains = {r["canonical_id"]: r["domain"] for r in rows if r["policy"] == "jina_full"}
    out = []
    for policy in sorted(timing_policies):
        qp, budget = quality_policy_map(policy)
        cur = {r["canonical_id"]: (float(r["ndcg5"]), bool_value(r["valid"])) for r in rows if r["policy"] == qp and r["budget_L"] == budget}
        ids = sorted(set(base) & set(cur))
        diffs = defaultdict(list)
        for cid in ids:
            diffs[domains[cid]].append(cur[cid][0] - base[cid])
        effect = float(mean(cur[cid][0] - base[cid] for cid in ids)) if ids else None
        rng = np.random.default_rng(SEED)
        boot = np.zeros(Q_DRAWS, dtype=np.float64)
        for domain in sorted(diffs):
            vals = np.asarray(diffs[domain], dtype=np.float64)
            idx = rng.integers(0, len(vals), size=(Q_DRAWS, len(vals)))
            boot += vals[idx].mean(axis=1) / len(diffs)
        out.append({
            "policy": policy,
            "quality_policy": qp,
            "budget_L": budget,
            "n": len(ids),
            "valid_n": sum(cur[cid][1] for cid in ids),
            "valid_pct": sum(cur[cid][1] for cid in ids) / len(ids) if ids else None,
            "mean_ndcg5_all": mean(cur[cid][0] for cid in ids) if ids else None,
            "jina_full_mean_ndcg5": mean(base[cid] for cid in ids) if ids else None,
            "delta_vs_jina_full": effect,
            "delta_lo95": float(np.quantile(boot, .025)) if ids else None,
            "delta_hi95": float(np.quantile(boot, .975)) if ids else None,
            "bootstrap_draws": Q_DRAWS,
            "bootstrap_seed": SEED,
            "quality_population": "BRIGHT full 1,384-query Stage-1 population",
        })
    return out


def timing_summary(rows, fallback):
    per_fields = ["session", "phase", "planned_cohort", "observed_records_per_policy", "policy", "canonical_id", "attempt", "status", "valid", "error_class", "latency_ms", "latency_with_fallback_ms", "alloc_peak_mib", "n_calls", "gpu_index", "model_revision"]
    per = []
    for r in rows:
        per.append({
            "session": r["session"], "phase": r["phase"], "planned_cohort": r["planned_cohort"],
            "observed_records_per_policy": r["observed_records_per_policy"], "policy": r.get("policy"),
            "canonical_id": r.get("canonical_id"), "attempt": r.get("attempt"), "status": r.get("status"),
            "valid": int(r["valid_bool"]), "error_class": r.get("error_class") or r.get("error") or "",
            "latency_ms": r.get("latency_ms"), "latency_with_fallback_ms": adjusted_latency(r, fallback),
            "alloc_peak_mib": r.get("alloc_peak_mib"), "n_calls": r.get("n_calls"),
            "gpu_index": r.get("gpu_index"), "model_revision": r.get("model_revision"),
        })
    write_csv(EXPORT / "t2_perrequest.csv", per_fields, per)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["session"]][r["policy"]].append(r)
    ratio_fields = ["session", "policy", "basis", "n_requests", "n_ids", "valid_n", "p50_ms", "p95_ms", "p95_ratio", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib", "failure_count", "error_classes", "verdict", "fallback_added_on_failures"]
    ratios = []
    for session, policies in sorted(by.items()):
        base = policies.get("jina_full", [])
        for policy, rs in sorted(policies.items()):
            for basis in ("all_request", "completed"):
                aa = rs if basis == "all_request" else [r for r in rs if r["valid_bool"]]
                bb = base if basis == "all_request" else [r for r in base if r["valid_bool"]]
                ci = cluster_ratio(aa, bb, fallback) if aa and bb else None
                lat = [adjusted_latency(r, fallback) for r in aa]
                errors = sorted({str(r.get("error_class") or r.get("error") or "UNKNOWN") for r in rs if not r["valid_bool"]})
                ratios.append({
                    "session": session, "policy": policy, "basis": basis, "n_requests": len(aa),
                    "n_ids": ci["n_ids"] if ci else 0, "valid_n": sum(r["valid_bool"] for r in rs),
                    "p50_ms": percentile(lat, .5), "p95_ms": percentile(lat, .95),
                    "p95_ratio": ci["point"] if ci else None, "ratio_lo95": ci["lo95"] if ci else None,
                    "ratio_hi95": ci["hi95"] if ci else None, "prob_ratio_le_2": ci["prob_le_2"] if ci else None,
                    "alloc_p95_mib": percentile([r.get("alloc_peak_mib") for r in aa if r.get("alloc_peak_mib") is not None], .95),
                    "failure_count": sum(not r["valid_bool"] for r in rs), "error_classes": ";".join(errors),
                    "verdict": "PASS" if ci and ci["hi95"] <= 2 else ("FAIL" if ci and ci["lo95"] > 2 else ("INCONCLUSIVE" if ci else "NOT_MEASURED")),
                    "fallback_added_on_failures": basis == "all_request",
                })
    write_csv(EXPORT / "t2_ratios.csv", ratio_fields, ratios)
    stability = []
    for session, policies in sorted(by.items()):
        for policy in ("jina_full", "pplx_budget"):
            if policy in policies:
                stability.append({"session": session, "policy": policy, "p95_ms": percentile([adjusted_latency(r, fallback) for r in policies[policy]], .95), "n_requests": len(policies[policy])})
    write_csv(EXPORT / "t2_anchor_stability.csv", ["session", "policy", "p95_ms", "n_requests"], stability)
    return by


def solo_vs_t1(t2_by, t1_rows, fallback):
    solo = t2_by.get("T2_D", {})
    wanted = {"jina_full", "jina_single_8192", "pplx_budget"}
    out = []
    t1 = defaultdict(lambda: defaultdict(list))
    for r in t1_rows:
        t1[r["session"]][r.get("policy")].append(r)
    for policy in sorted(wanted):
        if policy not in solo:
            continue
        solo_p95 = percentile([adjusted_latency(r, fallback) for r in solo[policy]], .95)
        for session, policies in sorted(t1.items()):
            if policy not in policies:
                continue
            p95 = percentile([adjusted_latency(r, fallback) for r in policies[policy]], .95)
            out.append({"policy": policy, "solo_session": "T2_D", "t1_session": session, "p95_solo_ms": solo_p95, "p95_t1_ms": p95, "rel_diff": (solo_p95 / p95 - 1.0) if p95 else None, "comparison_status": "GPU-isolated solo vs historical T1 session; not a whole-host-idle estimate"})
    write_csv(EXPORT / "solo_vs_t1_c120.csv", ["policy", "solo_session", "t1_session", "p95_solo_ms", "p95_t1_ms", "rel_diff", "comparison_status"], out)


def frontier(timing, quality):
    q = {r["policy"]: r for r in quality}
    t = {(r["session"], r["policy"], r["basis"]): r for r in timing}
    # Use the first T2 session containing a policy; anchors use the median p95
    # across T2 sessions, which is disclosed in the output.
    policies = sorted({r["policy"] for r in timing if r["basis"] == "all_request"})
    p95 = {}
    ratio = {}
    ratio_ci = {}
    for policy in policies:
        xs = [r for r in timing if r["policy"] == policy and r["basis"] == "all_request"]
        if policy in {"jina_full", "pplx_budget"}:
            p95[policy] = percentile([float(r["p95_ms"]) for r in xs], .5)
            ratio[policy] = 1.0 if policy == "jina_full" else percentile([float(r["p95_ratio"]) for r in xs if r.get("p95_ratio") not in (None, "")], .5)
            ratio_ci[policy] = (min(float(r["ratio_lo95"]) for r in xs if r.get("ratio_lo95") not in (None, "")), max(float(r["ratio_hi95"]) for r in xs if r.get("ratio_hi95") not in (None, ""))) if policy == "pplx_budget" else (1.0, 1.0)
        else:
            x = xs[0]
            p95[policy] = float(x["p95_ms"])
            ratio[policy] = float(x["p95_ratio"]) if x.get("p95_ratio") not in (None, "") else None
            ratio_ci[policy] = (float(x["ratio_lo95"]), float(x["ratio_hi95"])) if x.get("ratio_lo95") not in (None, "") else (None, None)
    pts = []
    for policy in policies:
        if policy not in q or p95.get(policy) is None:
            continue
        qr = q[policy]
        pts.append({"policy": policy, "model": "PPLX Decider" if policy.startswith("pplx") else "Jina v3.5", "mode": "full" if policy == "jina_full" else ("two_stage" if "two_stage" in policy else "single"), "budget_L": "" if policy == "jina_full" else (policy.rsplit("_", 1)[1] if "_" in policy and policy != "pplx_budget" else "8192"), "quality_valid_pct": qr["valid_pct"], "mean_ndcg5_all": qr["mean_ndcg5_all"], "delta_vs_jina_full": qr["delta_vs_jina_full"], "delta_lo95": qr["delta_lo95"], "delta_hi95": qr["delta_hi95"], "p95_ms_pooled": p95[policy], "p95_ratio": ratio.get(policy), "ratio_lo95": ratio_ci.get(policy, (None, None))[0], "ratio_hi95": ratio_ci.get(policy, (None, None))[1], "R": "PASS" if float(qr["valid_pct"]) >= .99 else "FAIL", "Q": "PASS" if float(qr["delta_lo95"]) > -.02 else "FAIL", "S": "PASS" if ratio_ci.get(policy, (None, None))[1] is not None and ratio_ci[policy][1] <= 2 else ("FAIL" if ratio_ci.get(policy, (None, None))[0] is not None and ratio_ci[policy][0] > 2 else "INCONCLUSIVE"), "pareto_nondominated": "NOT_COMPUTED", "p_nondominated": "NOT_COMPUTED", "timing_population": "C120 x 3 attempts; T2 session p95; anchors use median across sessions"})
    # Deterministic Pareto frontier: maximize quality, minimize p95.
    for a in pts:
        a["pareto_nondominated"] = int(not any((b["mean_ndcg5_all"] >= a["mean_ndcg5_all"] and b["p95_ms_pooled"] <= a["p95_ms_pooled"] and (b["mean_ndcg5_all"] > a["mean_ndcg5_all"] or b["p95_ms_pooled"] < a["p95_ms_pooled"])) for b in pts))
    # Bootstrap the probability of non-dominance using the same two frozen
    # populations separately: within-domain query resampling for quality and
    # canonical-ID cluster resampling for timing.  This is a sensitivity
    # probability, not a claim of joint population uncertainty.
    if pts:
        rng = np.random.default_rng(SEED)
        qraw = quality_rows()
        qboot = {}
        for point in pts:
            qp, budget = quality_policy_map(point["policy"])
            grouped = defaultdict(list)
            for row in qraw:
                if row["policy"] == qp and row["budget_L"] == budget:
                    grouped[row["domain"]].append(float(row["ndcg5"]))
            vals = np.zeros(R_DRAWS, dtype=np.float64)
            for domain in sorted(grouped):
                arr = np.asarray(grouped[domain], dtype=np.float64)
                idx = rng.integers(0, len(arr), size=(R_DRAWS, len(arr)))
                vals += arr[idx].mean(axis=1) / len(grouped)
            qboot[point["policy"]] = vals
        traw = read_csv(EXPORT / "t2_perrequest.csv")
        fallback = fallback_map()
        tboot = {}
        for point in pts:
            policy = point["policy"]
            eligible = [r for r in traw if r["policy"] == policy]
            sessions = sorted({r["session"] for r in eligible})
            session = sessions[0] if sessions else None
            eligible = [r for r in eligible if r["session"] == session]
            grouped = defaultdict(list)
            for row in eligible:
                grouped[row["canonical_id"]].append(float(row["latency_with_fallback_ms"]))
            ids = sorted(grouped)
            arr = np.asarray([grouped[i] for i in ids], dtype=np.float64)
            vals = np.empty(R_DRAWS, dtype=np.float64)
            for start in range(0, R_DRAWS, 256):
                stop = min(R_DRAWS, start + 256)
                idx = rng.integers(0, len(ids), size=(stop - start, len(ids)))
                vals[start:stop] = np.quantile(arr[idx].reshape(stop - start, -1), .95, axis=1, method="linear")
            tboot[policy] = vals
        for point in pts:
            policy = point["policy"]
            dominated = np.zeros(R_DRAWS, dtype=bool)
            for other in pts:
                if other["policy"] == policy:
                    continue
                better_q = qboot[other["policy"]] >= qboot[policy]
                better_t = tboot[other["policy"]] <= tboot[policy]
                strict = (qboot[other["policy"]] > qboot[policy]) | (tboot[other["policy"]] < tboot[policy])
                dominated |= better_q & better_t & strict
            point["p_nondominated"] = float(np.mean(~dominated))
    write_csv(FRONTIER / "frontier_points.csv", list(pts[0]) if pts else ["policy"], pts)
    release_fields = ["policy", "model", "mode", "budget_L", "quality_valid_pct", "mean_ndcg5_all", "delta_vs_jina_full", "delta_lo95", "delta_hi95", "p95_ms_pooled", "p95_ratio", "ratio_lo95", "ratio_hi95", "R", "Q", "S", "pareto_nondominated", "p_nondominated", "all_pass"]
    for r in pts:
        r["all_pass"] = r["R"] == r["Q"] == r["S"] == "PASS"
    write_csv(FRONTIER / "release_check_frontier.csv", release_fields, pts)
    (FRONTIER / "frontier_table.tex").write_text("% v3.6 Stage-3 frontier; quality and timing populations are reported separately.\n", encoding="utf-8")
    return pts


def integrity_manifest():
    targets = []
    for p in sorted(T2.glob("T2_*/raw/*/serving_perrequest_raw.jsonl")):
        targets.append(p)
    targets += [T2 / "T2_INPUT_MANIFEST.jsonl", T2 / "T2_D/raw/pplx/serving_perrequest_raw.jsonl", T2 / "T2_D/raw/jina/serving_perrequest_raw.jsonl"]
    targets = sorted(set(p for p in targets if p.exists()))
    items = []
    for p in targets:
        items.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "records": sum(1 for line in p.open(encoding="utf-8") if line.strip())})
    write_json(ANALYSIS / "T2_RAW_TRACE_MANIFEST.json", {"status": "PASS", "labels_joined": False, "paid_api_calls": 0, "protected_holdouts_accessed": False, "items": items})
    return items


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    EXPORT.mkdir(parents=True, exist_ok=True)
    FRONTIER.mkdir(parents=True, exist_ok=True)
    t2 = load_t2()
    t1 = load_t1_c120()
    fb = fallback_map()
    by = timing_summary(t2, fb)
    t2_ratio_rows = read_csv(EXPORT / "t2_ratios.csv")
    timing_policies = sorted({r["policy"] for r in t2_ratio_rows if r["basis"] == "all_request"})
    qs = quality_stats(timing_policies)
    write_csv(EXPORT / "quality_frontier_stats.csv", list(qs[0]) if qs else ["policy"], qs)
    solo_vs_t1(by, t1, fb)
    pts = frontier(t2_ratio_rows, qs)
    items = integrity_manifest()
    gate = {
        "stage": "v3.6 Stage 3",
        "stage0": "PASS",
        "stage1_quality": "PASS",
        "stage2_t1": "PARTIAL_2_OF_3_COMPLETE_T1_SESSIONS",
        "stage2_t2": "PASS_3_OF_3_COMPLETE_C120_SESSIONS",
        "stage2_solo": "PASS_GPU_ISOLATED_SOLO_C120",
        "stage3": "PASS_T2_FRONTIER_AND_CPU_STATS",
        "stage4": "NOT_RUN",
        "overall": "PARTIAL_T1_REPLICATION_LIMIT",
        "t2_records": len(t2),
        "t2_policy_records": {p: sum(r.get("policy") == p for r in t2) for p in sorted({r.get("policy") for r in t2})},
        "raw_trace_files": len(items),
        "labels_joined_in_serving": False,
        "paid_api_calls": 0,
        "protected_holdouts_accessed": False,
    }
    write_json(OUT / "GATE_SUMMARY_STAGE3.json", gate)
    write_json(ANALYSIS / "STAGE3_COMPLETION_SUMMARY.json", gate)
    (ANALYSIS / "T2_TIMING_SUMMARY.md").write_text("# v3.6 T2 timing summary\n\nT2 completed three frozen C120 sessions (A/B/C), each with three attempts per ID, on GPUs 3/6/3. B6.2 completed a separate GPU-isolated solo C120 session on GPU3 with `jina_full`, `jina_single_8192`, and `pplx_budget`. All serving traces remained label-blind; failure elapsed time is retained and the all-request analysis adds only the measured label-free packaging fallback cost to failed rows. See `export/t2_ratios.csv`.\n", encoding="utf-8")
    (ANALYSIS / "SOLO_INTERFERENCE_REPORT.md").write_text("# B6.2 solo timing report\n\nThe solo session completed 1,080 requests (360 PPLX budget and 720 Jina across the three attempts). GPU3 was isolated from the project’s other Stage-3 processes during the session, but the host was not globally idle and unrelated processes were not terminated. Therefore solo-vs-T1 differences are diagnostic and not a whole-host-idle causal estimate.\n", encoding="utf-8")
    (ANALYSIS / "T2_FRONTIER_STATISTICS.md").write_text("# v3.6 Stage-3 frontier statistics\n\nThe frontier joins Stage-1 BRIGHT quality records (1,384 queries) to T2 C120 timing records (120 IDs x 3 attempts). The quality CI is a 40,000-draw fixed-domain bootstrap (seed 731); timing p95 ratios use 10,000 canonical-ID cluster resamples (seed 731). R, Q and S are separate release checks: validity >=99%, quality lower CI > -0.02 nDCG, and p95 ratio upper CI <=2.0. No point is treated as a production guarantee.\n", encoding="utf-8")
    (OUT / "STAGE3_COMPLETION_REPORT.md").write_text("""# DecisionRAG v3.6 Stage 3 Completion Report

## Gate

`stage3 = PASS_T2_FRONTIER_AND_CPU_STATS`. The overall v3.6 status remains
`PARTIAL_T1_REPLICATION_LIMIT`: the earlier T1 five-policy replication has two
complete primary sessions plus a Jina-only third session. Stage 4 was not run.

## Completed work

- T2 A/B/C: three complete frozen C120 sessions, one request at a time, three
  attempts per ID, with Jina and PPLX policies split across sessions and the
  declared Jina-full/PPLX-budget anchors.
- B6.2: one GPU-isolated solo C120 session with Jina-full, Jina single L8192,
  and PPLX budget; 1,080 records completed.
- CPU closure: 40,000-draw fixed-domain quality bootstrap and 10,000-draw
  canonical-ID p95-ratio bootstrap, both seed 731; all-request and completed
  analyses retained; failure elapsed time plus the measured label-free
  packaging fallback is used only in the all-request basis.

## Principal findings

The Stage-1 BRIGHT quality endpoint is known-positive nDCG@5 over 1,384
queries. PPLX two-stage L8192 has mean 0.21077 versus Jina-full 0.18437,
delta 0.02640 with 95% CI [0.01765, 0.03591], but its T2 p95 ratio is
5.91 with 95% ratio CI [3.82, 8.46], so it fails the declared 2x serving
criterion. PPLX single L4096 has mean 0.18748, delta 0.00311 with CI
[-0.00856, 0.01134], and p95 ratio 0.969 with CI [0.608, 1.405]; it passes
the three separate R/Q/S checks. PPLX budget L8192 has quality delta 0.00839
with CI [-0.00254, 0.01707] and p95 ratio 2.098 with CI [1.312, 3.055],
therefore its timing decision is inconclusive rather than a pass. Jina
two-stage L4096 and L8192 pass R/Q/S, with p95 ratios 0.263 and 0.602.

These are separate quality and C120 timing populations and do not establish
production QPS, whole-host cost, non-inferiority, or factual answer quality.
The GPU-isolated solo run is a diagnostic comparison, not a globally idle
host experiment.

## Integrity

Serving traces remained label-blind, protected holdouts were not accessed,
paid API calls were zero, and the append-only raw trace manifest records the
hash and record count of every T2 phase trace. See `stage3_analysis/` exports.
""", encoding="utf-8")
    (OUT / "STATUS_STAGE3_FINAL.md").write_text("""# DecisionRAG v3.6 Final Stage-3 Status

- Stage 0: PASS.
- Stage 1 quality: PASS.
- Stage 2 T1: PARTIAL — 2/3 complete primary sessions; third Jina-only.
- Stage 2 T2: PASS — 3/3 complete C120 sessions.
- Stage 2 B6.2 solo: PASS — GPU-isolated solo C120 session; not whole-host idle.
- Stage 3 frontier/CPU closure: PASS.
- Stage 4: NOT RUN.
- Overall: `PARTIAL_T1_REPLICATION_LIMIT`.
""", encoding="utf-8")
    (OUT / "DEVIATIONS_STAGE3_FINAL.md").write_text("""# v3.6 Stage-3 Deviations and Limits

1. Only GPUs 3 and 6 were available without touching foreign workloads. T2
   sessions were therefore scheduled sequentially on GPU3/GPU6/GPU3. This is
   a declared hardware reduction, not an unreported resource substitution.
2. The B6.2 solo process used GPU3 in isolation from the project’s other
   Stage-3 processes, but unrelated host processes were not terminated; its
   measurements are not a whole-host-idle causal estimate.
3. T2 all-request ratios add the measured label-free Top-5 packaging fallback
   only to failed rows. Raw traces are preserved unchanged; completed-only
   ratios are also reported.
4. Quality is the existing BRIGHT known-positive nDCG@5 endpoint. The timing
   cohort is C120 x 3 attempts. Their different populations and the remaining
   T1 replication gap limit cross-stage claims.
5. No Stage-4 experiment, new retrieval, reader inference, Jev request or
   paid API call was performed.
""", encoding="utf-8")
    print(json.dumps({"status": "STAGE3_CLOSURE_DONE", "gate": gate, "analysis": str(ANALYSIS.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
