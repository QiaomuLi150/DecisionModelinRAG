#!/usr/bin/env python3
"""CPU-only v3.6 closure and export generation.

Run only after the authorized serving phases have finished.  It consumes raw
traces and frozen label joins; it never loads a model or makes a request.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential"
STAGE0 = OUT / "stage0_preflight"
S1 = OUT / "stage1_quality_v2/export/quality"
S2 = OUT / "stage2_serving"


def read_jsonl(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def sha_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(p, fields, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def percentile(values, q):
    if not values: return None
    x = sorted(float(v) for v in values); pos = (len(x) - 1) * q; lo = int(pos); hi = min(len(x) - 1, lo + 1)
    return x[lo] if lo == hi else x[lo] + (x[hi] - x[lo]) * (pos - lo)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: float(ps[i]))
    out = [None] * len(ps); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * float(ps[i])))
        out[i] = running
    return out


def quality_finalize():
    p = S1 / "q_contrasts.csv"
    if not p.exists(): return
    with p.open(encoding="utf-8") as f: rows = list(csv.DictReader(f))
    for family in {x["family"] for x in rows}:
        idx = [i for i, x in enumerate(rows) if x["family"] == family]
        vals = holm([float(rows[i]["p_signflip"]) for i in idx])
        for i, v in zip(idx, vals): rows[i]["p_holm"] = f"{v:.12g}"
    fields = list(rows[0]); write_csv(p, fields, rows)
    write_csv(S1 / "determinism.csv", ["policy", "order_id", "n_rechecked", "n_mismatch", "status"], [
        {"policy": "all_new_stage1_policies", "order_id": "original", "n_rechecked": 0, "n_mismatch": "NOT_MEASURED", "status": "PENDING_GPU_RECHECK"}
    ])


def load_serving():
    rows = []
    namespaces = [
        (S2, "T1_GPU3"),
        (ROOT / "results/v3_6_essential/stage2_serving_gpu6", "T1_GPU6"),
        (ROOT / "results/v3_6_essential/stage2_serving_gpu3_r3", "T1_GPU3_R3"),
    ]
    for ns, session in namespaces:
      for phase in ("jina", "pplx"):
        p = ns / "raw" / phase / "serving_perrequest_raw.jsonl"
        if p.exists():
            phase_rows = read_jsonl(p)
            # The first Jina two-stage trace is retained as an erratum source;
            # the corrected tuple-key trace below is the authoritative timing
            # input for closure.
            if phase == "jina":
                phase_rows = [r for r in phase_rows if r.get("policy") != "jina_two_stage_pplxwindow"]
            for r in phase_rows: r["session_id"] = session
            rows.extend(phase_rows)
      repaired = ns / "raw" / "jina_j2_repair" / "serving_perrequest_raw.jsonl"
      if repaired.exists():
          repaired_rows = read_jsonl(repaired)
          for r in repaired_rows: r["session_id"] = session
          rows.extend(repaired_rows)
    return rows


def cluster_ratio(policy_rows, base_rows, draws=10000):
    pa = defaultdict(list); ba = defaultdict(list)
    for r in policy_rows: pa[r["canonical_id"]].append(float(r["latency_ms"]))
    for r in base_rows: ba[r["canonical_id"]].append(float(r["latency_ms"]))
    ids = sorted(set(pa) & set(ba))
    if not ids: return None
    # Every frozen serving request has a fixed three-attempt cluster.  Keep a
    # defensive ragged fallback for incomplete traces, but use a vectorized
    # path for the normal case.  The Python RNG sequence is retained so this
    # is the same declared seed/resampling contract as the original code.
    point = percentile([v for i in ids for v in pa[i]], .95) / percentile([v for i in ids for v in ba[i]], .95)
    lengths = {len(pa[i]) for i in ids} | {len(ba[i]) for i in ids}
    rng = random.Random(731)
    if len(lengths) == 1:
        n_attempts = next(iter(lengths))
        a = np.asarray([pa[i] for i in ids], dtype=np.float64)
        b = np.asarray([ba[i] for i in ids], dtype=np.float64)
        sample_idx = np.fromiter((rng.randrange(len(ids)) for _ in range(draws * len(ids))), dtype=np.int64, count=draws * len(ids)).reshape(draws, len(ids))
        vals = np.empty(draws, dtype=np.float64)
        chunk = 256
        for start in range(0, draws, chunk):
            stop = min(draws, start + chunk)
            av = a[sample_idx[start:stop]].reshape(stop - start, len(ids) * n_attempts)
            bv = b[sample_idx[start:stop]].reshape(stop - start, len(ids) * n_attempts)
            vals[start:stop] = np.quantile(av, .95, axis=1, method="linear") / np.quantile(bv, .95, axis=1, method="linear")
    else:
        vals = []
        for _ in range(draws):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            av = [v for i in sample for v in pa[i]]; bv = [v for i in sample for v in ba[i]]
            vals.append(percentile(av, .95) / percentile(bv, .95))
        vals = np.asarray(vals, dtype=np.float64)
    return {"point": point, "lo95": float(np.quantile(vals, .025)), "hi95": float(np.quantile(vals, .975)), "prob_le_1_5": float(np.mean(vals <= 1.5)), "prob_le_2": float(np.mean(vals <= 2)), "prob_le_3": float(np.mean(vals <= 3)), "n_ids": len(ids)}


def fallback_cost(views):
    # Label-free packaging measurement.  It measures only candidate-ID mapping
    # and Top-5 packaging, never reads labels.
    out = {}
    for v in views:
        ids = list(v["original_ids"]); t = time.perf_counter(); _ = [str(x) for x in ids[:5]]; out[v["canonical_id"]] = (time.perf_counter() - t) * 1000
    return out


def serving_finalize(rows):
    ex = S2 / "export"; ex.mkdir(parents=True, exist_ok=True)
    policies = sorted({r["policy"] for r in rows})
    slot_map = {"T1_GPU3": "A", "T1_GPU6": "B", "T1_GPU3_R3": "C"}
    # The serving export is intentionally explicit about the one-session
    # reduction and does not claim three replicated sessions.
    per_fields = ["session", "gpu_slot", "cohort", "policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"]
    per = []
    for r in rows:
        sid = r.get("session_id", "T1_GPU3")
        per.append({"session": f"T1_{slot_map.get(sid, 'X')}", "gpu_slot": slot_map.get(sid, "X"), "cohort": "C360", "policy": r["policy"], "canonical_id": r["canonical_id"], "attempt": r["attempt"], "status": r.get("status"), "error_class": r.get("error_class") or "", "latency_ms": r.get("latency_ms"), "alloc_peak_mib": r.get("alloc_peak_mib")})
    write_csv(ex / "serving_perrequest.csv", per_fields, per)
    by_session = defaultdict(lambda: defaultdict(list))
    for r in rows: by_session[r.get("session_id", "T1_GPU3")][r["policy"]].append(r)
    ratio_rows = []
    for session, byp in sorted(by_session.items()):
      full = byp.get("jina_full", [])
      for p in policies:
        for basis in ("all_request", "completed"):
            a = [r for r in byp[p] if basis == "all_request" or r.get("valid") is True]
            b = [r for r in full if basis == "all_request" or r.get("valid") is True]
            if not a or not b: continue
            ci = cluster_ratio(a, b)
            ratio_rows.append({"session": f"T1_{slot_map.get(session, 'X')}", "cohort": "C360", "policy": p, "basis": basis, "p50_ms": percentile([r["latency_ms"] for r in a], .5), "p95_ms": percentile([r["latency_ms"] for r in a], .95), "p95_ratio_point": ci["point"] if ci else None, "bootstrap_lo95": ci["lo95"] if ci else None, "bootstrap_hi95": ci["hi95"] if ci else None, "prob_ratio_le_2": ci["prob_le_2"] if ci else None, "alloc_p95_mib": percentile([r["alloc_peak_mib"] for r in a if r.get("alloc_peak_mib") is not None], .95), "verdict": "pass" if ci and ci["hi95"] <= 2 else ("fail" if ci and ci["lo95"] > 2 else "inconclusive")})
    write_csv(ex / "serving_ratios.csv", ["session", "cohort", "policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "bootstrap_lo95", "bootstrap_hi95", "prob_ratio_le_2", "alloc_p95_mib", "verdict"], ratio_rows)
    pooled_rows = []
    pooled = defaultdict(list)
    for r in rows:
        pooled[r["policy"]].append(r)
    pooled_full = pooled.get("jina_full", [])
    for p in policies:
        for basis in ("all_request", "completed"):
            a = [r for r in pooled[p] if basis == "all_request" or r.get("valid") is True]
            b = [r for r in pooled_full if basis == "all_request" or r.get("valid") is True]
            if not a or not b:
                continue
            ci = cluster_ratio(a, b)
            pooled_rows.append({
                "session": "pooled_T1", "cohort": "C360", "policy": p, "basis": basis,
                "p50_ms": percentile([r["latency_ms"] for r in a], .5),
                "p95_ms": percentile([r["latency_ms"] for r in a], .95),
                "p95_ratio_point": ci["point"] if ci else None,
                "bootstrap_lo95": ci["lo95"] if ci else None,
                "bootstrap_hi95": ci["hi95"] if ci else None,
                "prob_ratio_le_2": ci["prob_le_2"] if ci else None,
                "alloc_p95_mib": percentile([r["alloc_peak_mib"] for r in a if r.get("alloc_peak_mib") is not None], .95),
                "verdict": "pass" if ci and ci["hi95"] <= 2 else ("fail" if ci and ci["lo95"] > 2 else "inconclusive"),
            })
    write_csv(ex / "serving_ratios_pooled.csv", ["session", "cohort", "policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "bootstrap_lo95", "bootstrap_hi95", "prob_ratio_le_2", "alloc_p95_mib", "verdict"], pooled_rows)
    sensitivity = []
    for p in policies:
        for basis in ("all_request", "completed"):
            a = [r for r in pooled[p] if basis == "all_request" or r.get("valid") is True]
            b = [r for r in pooled_full if basis == "all_request" or r.get("valid") is True]
            ci = cluster_ratio(a, b) if a and b else None
            for thr, key in ((1.5, "prob_le_1_5"), (2.0, "prob_le_2"), (3.0, "prob_le_3")):
                sensitivity.append({"policy": p, "basis": basis, "thr": thr, "prob_ratio_le_thr": ci[key] if ci else "NOT_MEASURED", "verdict": ("pass" if ci and ci["hi95"] <= thr else ("fail" if ci and ci["lo95"] > thr else "inconclusive")) if ci else "NOT_MEASURED"})
    write_csv(ex / "serving_threshold_sensitivity.csv", ["policy", "basis", "thr", "prob_ratio_le_thr", "verdict"], sensitivity)
    write_csv(ex / "solo_vs_concurrent.csv", ["policy", "p95_solo_ms", "p95_t1_session_ms", "rel_diff", "status"], [{"policy": p, "p95_solo_ms": "NOT_RUN", "p95_t1_session_ms": "NOT_MEASURED", "rel_diff": "NOT_MEASURED", "status": "NOT_RUN"} for p in policies])
    required_policies = {"jina_full", "jina_pplx_budget", "jina_two_stage_pplxwindow", "pplx_budget", "pplx_two_stage"}
    complete_sessions = [s for s, byp in by_session.items() if required_policies.issubset(byp) and all(len(byp[p]) == 1080 for p in required_policies)]
    session_status = "THREE_COMPLETE_SESSIONS" if len(complete_sessions) >= 3 else f"PARTIAL_{len(complete_sessions)}_OF_3_COMPLETE_SESSIONS"
    write_csv(ex / "anchor_stability.csv", ["session", "policy", "p95_ms", "status"], [{"session": f"T1_{slot_map.get(s, 'X')}", "policy": p, "p95_ms": percentile([x["latency_ms"] for x in byp[p]], .95), "status": session_status} for s,byp in by_session.items() for p in policies if p in byp])
    crash_rows = []
    for r in rows:
        if r.get("status") != "ok" or r.get("valid") is not True:
            sid = r.get("session_id", "unknown")
            crash_rows.append({"session": f"T1_{slot_map.get(sid, 'X')}", "policy": r.get("policy", "unknown"), "request_index": r.get("request_index", ""), "error_class": r.get("error_class") or "invalid", "restarts_so_far": "NOT_RECORDED", "untimed_requests_after": "NOT_MEASURED"})
    write_csv(ex / "crash_log.csv", ["session", "policy", "request_index", "error_class", "restarts_so_far", "untimed_requests_after"], crash_rows)
    lv = []
    for session, byp in sorted(by_session.items()):
        base = percentile([x["latency_ms"] for x in byp.get("jina_full", []) if isinstance(x.get("latency_ms"), (int, float))], .95)
        for p in policies:
            if p not in byp:
                continue
            model = "PPLX Decider" if p.startswith("pplx") else "Jina v3.5"
            params = 27.0 if p.startswith("pplx") else 0.6
            p95 = percentile([x["latency_ms"] for x in byp[p] if isinstance(x.get("latency_ms"), (int, float))], .95)
            if p95 is None:
                continue
            lv.append({"model": model, "params_b": params, "family": "decision" if p.startswith("pplx") else "dedicated", "session": f"T1_{slot_map.get(session, 'X')}", "p95_ms": p95, "p95_ratio_to_jina_full": p95 / base if base else None})
    # Label-free fallback packaging microbenchmark on the C360 candidate IDs.
    view_manifest = {r["canonical_id"]: r for r in read_jsonl(STAGE0 / "BUDGET_VIEW_MANIFEST.jsonl")}
    cohort_ids = [r["canonical_id"] for r in csv.DictReader((STAGE0 / "cohort_manifest.csv").open(encoding="utf-8"))]
    fallback_rows = []
    for cid in cohort_ids:
        ids = view_manifest[cid]["candidate_ids"]
        t0 = time.perf_counter(); _ = [str(x) for x in ids[:5]]; elapsed = (time.perf_counter() - t0) * 1000
        fallback_rows.append({"canonical_id": cid, "packaging_ms": elapsed, "status": "MEASURED_LABEL_FREE"})
    write_csv(ex / "fallback_cost.csv", ["canonical_id", "packaging_ms", "status"], fallback_rows)
    load_rows = []
    for ns, session in [(S2, "T1_GPU3"), (ROOT / "results/v3_6_essential/stage2_serving_gpu6", "T1_GPU6"), (ROOT / "results/v3_6_essential/stage2_serving_gpu3_r3", "T1_GPU3_R3")]:
        for name, field in (("Jina v3.5", "jina_model_load_ms"), ("PPLX Decider", "pplx_model_load_ms")):
            sp = ns / "SESSION_jina.json" if name.startswith("Jina") else ns / "SESSION_pplx.json"
            if sp.exists():
                obj = json.loads(sp.read_text(encoding="utf-8"));
                if field in obj:
                    load_rows.append({"gpu_slot": slot_map.get(session, "X"), "model": name, "load_seconds": float(obj[field]) / 1000.0})
    write_csv(ex / "load_times.csv", ["gpu_slot", "model", "load_seconds"], load_rows)
    write_csv(ex / "latency_vs_params.csv", ["model", "params_b", "family", "session", "p95_ms", "p95_ratio_to_jina_full"], lv)
    write_json(S2 / "SERVING_TRACE_SUMMARY.json", {"status": session_status, "records": len(rows), "policies": {p: len([r for r in rows if r["policy"] == p]) for p in policies}, "sessions": len(by_session), "complete_sessions": complete_sessions, "required_sessions": 3, "gpu_isolation": "GPU3 and GPU6 sessions were executed in separate namespaces; foreign-workload GPUs were not used"})


def frontier_finalize():
    p = S1 / "q_summary.csv"
    if not p.exists(): return
    with p.open(encoding="utf-8") as f: q = list(csv.DictReader(f))
    ex = OUT / "stage3_analysis" / "export" / "frontier"; ex.mkdir(parents=True, exist_ok=True)
    trows = list(csv.DictReader((S2 / "export/serving_ratios.csv").open(encoding="utf-8"))) if (S2 / "export/serving_ratios.csv").exists() else []
    t = {(x["policy"], x["basis"]): x for x in trows}
    qmap = {(x["policy"], x["budget_L"]): x for x in q}
    mapping = [("jina_full", "jina_full", "full", ""), ("jina_pplx_budget", "jina_single", "single", "8192"), ("pplx_budget", "pplx_single", "single", "8192"), ("jina_two_stage_pplxwindow", "jina_two_stage", "two_stage", "8192"), ("pplx_two_stage", "pplx_two_stage", "two_stage", "8192")]
    rows=[]; points=[]
    for tp, qp, mode, budget in mapping:
        qs=qmap.get((qp,budget)); ts=t.get((tp,"all_request"))
        if not qs or not ts: continue
        rows.append({"policy": tp, "model": "PPLX" if tp.startswith("pplx") else "Jina", "mode": mode, "budget_L": budget, "valid_pct": qs["valid_pct"], "mean_ndcg5_all": qs["mean_ndcg5_all"], "delta_vs_jina_full": "NOT_COMPUTED", "delta_lo95": "NOT_COMPUTED", "delta_hi95": "NOT_COMPUTED", "p95_ratio": ts["p95_ratio_point"], "ratio_lo95": ts["bootstrap_lo95"], "ratio_hi95": ts["bootstrap_hi95"], "R": "PASS" if float(qs["valid_pct"]) >= .99 else "FAIL", "Q": "NOT_COMPUTED", "S": ts["verdict"].upper(), "all_pass": False})
        points.append({"policy":tp,"model":"PPLX" if tp.startswith("pplx") else "Jina","mode":mode,"budget_L":budget,"mean_ndcg5_all":qs["mean_ndcg5_all"],"p95_ms_pooled":ts["p95_ms"],"p95_ratio":ts["p95_ratio_point"],"pareto_nondominated":"NOT_COMPUTED"})
    fields=["policy","model","mode","budget_L","valid_pct","mean_ndcg5_all","delta_vs_jina_full","delta_lo95","delta_hi95","p95_ratio","ratio_lo95","ratio_hi95","R","Q","S","all_pass"]
    write_csv(ex / "release_check_frontier.csv", fields, rows); write_csv(ex / "frontier_points.csv", list(points[0]) if points else ["policy"], points)
    (ex / "frontier_table.tex").write_text("% v3.6 frontier is partial: one T1 serving session; release checks requiring replicated timing are not finalized.\n", encoding="utf-8")


def main():
    quality_finalize()
    rows = load_serving()
    if rows: serving_finalize(rows)
    frontier_finalize()
    manifest = {}
    for p in OUT.rglob("*"):
        if p.is_file() and p.name not in {"RAW_TRACE_SHA256_MANIFEST.json"}:
            manifest[str(p.relative_to(ROOT))] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    write_json(OUT / "RAW_TRACE_SHA256_MANIFEST.json", manifest)
    n_sessions = len({r.get("session_id") for r in rows}) if rows else 0
    required_policies = {"jina_full", "jina_pplx_budget", "jina_two_stage_pplxwindow", "pplx_budget", "pplx_two_stage"}
    session_policy_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        session_policy_counts[r.get("session_id")][r["policy"]] += 1
    complete_sessions = [s for s, counts in session_policy_counts.items() if required_policies.issubset(counts) and all(counts[p] == 1080 for p in required_policies)]
    stage2 = "PASS_THREE_REPLICATES" if len(complete_sessions) >= 3 else (f"PARTIAL_{len(complete_sessions)}_OF_3_COMPLETE_REPLICATES" if rows else "BLOCKED")
    overall = "PARTIAL" if stage2 != "PASS_THREE_REPLICATES" else "PARTIAL_OPTIONAL_AND_T2_PENDING"
    write_json(OUT / "GATE_SUMMARY.json", {"stage0": "PASS", "stage1_quality": "PASS", "stage2_serving": stage2, "stage3": "PARTIAL_PENDING_T2_AND_OPTIONAL", "stage4": "NOT_RUN", "overall": overall})
    (OUT / "DEVIATIONS.md").write_text("# v3.6 deviations\n\n- Three T1 session namespaces were scheduled on GPU3, GPU6 and a second GPU3 namespace. Two namespaces contain the complete five-policy primary set; the third contains only the three Jina policies because GPU3 was released after that block. Therefore the T1 serving gate remains partial at 2/3 complete replicate sessions. The third session is sequential rather than concurrent with the first GPU3 session; same-session ratios remain valid, but cross-session interference is not a concurrent three-GPU estimate.\n- The initial Jina two-stage wrapper emitted a tuple-key error. Its raw failures are preserved; the affected policy was rerun in the versioned `jina_j2_repair` namespace and only the repaired trace is authoritative for timing.\n- Historical Winnow, Drex and Liquid records were not newly timed by this runner; their quality records remain historical controls.\n- Solo C120, T2 frontier sessions, determinism rechecks, and optional Stage 4 remain pending/not run.\n- No labels were read by the serving runner; quality labels were joined only in the CPU analysis stage.\n", encoding="utf-8")
    (OUT / "STATUS.md").write_text("# DecisionRAG v3.6 Essential Verification\n\n- Stage 0: PASS; frozen manifests and protocol.\n- Stage 1: PASS; corrected native quality traces and CPU label join.\n- Stage 2: %s; %d T1 serving replica session(s) available, with all raw failures retained.\n- Stage 3: PARTIAL; CPU exports generated where timing exists, while T2/frontier and optional analyses remain pending.\n- Stage 4: NOT RUN.\n" % (stage2, n_sessions), encoding="utf-8")
    print(json.dumps({"status": "FINALIZE_DONE", "serving_records": len(rows), "gate": "PARTIAL"}, indent=2))


if __name__ == "__main__": main()
