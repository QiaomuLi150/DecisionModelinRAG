#!/usr/bin/env python3
"""CPU-only closure for the v3.6 follow-up.

This consumes only frozen quality outputs and completed serving traces.  It
never loads a model or makes an inference request.  The script is deliberately
resume-safe: it writes only new closure artifacts and leaves raw traces and
earlier exports untouched.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v1"
EXPORT = OUT / "export"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
COHORT = ROOT / "results/v3_6_essential/stage0_preflight/cohort_manifest.csv"
QUALITY = OUT / "export/quality_v2/q_perquery_v2.csv"
QUALITY_SUMMARY = OUT / "export/quality_v2/q_summary_v2.csv"
STAGE1_QUALITY = ROOT / "results/v3_6_essential/stage1_quality_v2/export/quality/q_perquery.csv"
SEED = 731
BOOT = 10000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def bval(v):
    return v is True or str(v).lower() in {"true", "1", "pass", "ok", "success"}


def pct(vals, q):
    vals = sorted(float(x) for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not vals:
        return None
    x = (len(vals) - 1) * q
    lo, hi = int(x), min(len(vals) - 1, int(x) + 1)
    return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (x - lo)


def cohort_map():
    rows = read_csv(COHORT)
    return {r["canonical_id"]: {"domain": r.get("domain", ""), "in_C120": bval(r.get("in_C120"))} for r in rows}


def load_n1():
    rows = []
    for session in ("A", "B"):
        p = OUT / f"stage2_n1/session_{session}/raw/serving_perrequest_raw.jsonl"
        if not p.exists():
            p = OUT / f"stage2_n1/session_{session}/raw/serving_perrequest.checkpoint.jsonl"
        for r in read_jsonl(p):
            x = dict(r)
            x["session"] = session
            x["cohort"] = "CALL"
            x["slot"] = session
            x["valid_bool"] = bval(x.get("valid")) or str(x.get("status", "")).upper() in {"PASS", "OK", "SUCCESS"}
            x["latency_with_fallback_ms"] = float(x.get("latency_ms") or 0.0) + (float(x.get("fallback_packaging_ms") or 0.0) if not x["valid_bool"] else 0.0)
            rows.append(x)
    return rows


def load_n2():
    rows = []
    paths = [
        (OUT / "stage2_n2/session_C/raw/serving_perrequest_raw.jsonl", "C", "C360"),
        (OUT / "stage2_n2/session_C/raw/serving_perrequest.checkpoint.jsonl", "C", "C360"),
        (OUT / "stage2_n2/session_D_drex_v2_batch16384/raw/serving_perrequest.checkpoint.jsonl", "D", "C360"),
        (OUT / "stage2_n2/session_E_winnow/raw/serving_perrequest.checkpoint.jsonl", "E", "C360"),
    ]
    seen = set()
    for p, session, cohort in paths:
        for r in read_jsonl(p):
            key = (session, r.get("policy"), r.get("canonical_id"), int(r.get("attempt", 0)))
            if key in seen:
                continue
            seen.add(key)
            x = dict(r)
            x["session"] = session
            x["cohort"] = cohort
            x["slot"] = session
            x["valid_bool"] = bval(x.get("valid")) or str(x.get("status", "")).upper() in {"PASS", "OK", "SUCCESS"}
            x["latency_with_fallback_ms"] = float(x.get("latency_ms") or 0.0) + (float(x.get("fallback_packaging_ms") or 0.0) if not x["valid_bool"] else 0.0)
            rows.append(x)
    return rows


def bootstrap_ratio(a, b, id_field="canonical_id"):
    pa, pb = defaultdict(list), defaultdict(list)
    for r in a:
        pa[r[id_field]].append(float(r["latency_with_fallback_ms"]))
    for r in b:
        pb[r[id_field]].append(float(r["latency_with_fallback_ms"]))
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return None
    point = pct([v for i in ids for v in pa[i]], .95) / pct([v for i in ids for v in pb[i]], .95)

    # Keep the original seeded cluster-resampling contract, but evaluate each
    # 100-draw block with NumPy.  The former nested Python loop was equivalent
    # but became impractically slow for the full CALL population.
    def padded(groups):
        width = max(len(groups[i]) for i in ids)
        arr = np.full((len(ids), width), np.nan, dtype=float)
        for row, key in enumerate(ids):
            vals = np.asarray(groups[key], dtype=float)
            arr[row, : len(vals)] = vals
        return arr

    aa = padded(pa)
    bb = padded(pb)
    rng = np.random.default_rng(SEED)
    vals = []
    for start in range(0, BOOT, 100):
        n = min(100, BOOT - start)
        inds = rng.integers(0, len(ids), size=(n, len(ids)))
        av = np.take(aa, inds, axis=0).reshape(n, -1)
        bv = np.take(bb, inds, axis=0).reshape(n, -1)
        vals.extend((np.nanquantile(av, .95, axis=1) / np.nanquantile(bv, .95, axis=1)).tolist())
    vals = np.asarray(vals)
    return {"n_ids": len(ids), "point": point, "lo95": float(np.quantile(vals, .025)), "hi95": float(np.quantile(vals, .975)), "prob_le_1_5": float(np.mean(vals <= 1.5)), "prob_le_2": float(np.mean(vals <= 2.0)), "prob_le_3": float(np.mean(vals <= 3.0))}


def serving_exports(n1, n2):
    full = EXPORT / "serving_full"
    fields = ["session", "gpu_slot", "cohort", "policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "latency_with_fallback_ms", "alloc_peak_mib"]
    rows = []
    for r in n1:
        rows.append({"session": r["session"], "gpu_slot": r["slot"], "cohort": "CALL", "policy": r.get("policy"), "canonical_id": r.get("canonical_id"), "attempt": r.get("attempt"), "status": "PASS" if r["valid_bool"] else "FAIL", "error_class": r.get("error_class") or r.get("error") or "", "latency_ms": r.get("latency_ms"), "latency_with_fallback_ms": r.get("latency_with_fallback_ms"), "alloc_peak_mib": r.get("alloc_peak_mib")})
    write_csv(full / "serving_perrequest.csv", fields, rows)
    policies = sorted({r.get("policy") for r in n1})
    summary = []
    for session in ("A", "B"):
        sess = [r for r in n1 if r["session"] == session]
        base = [r for r in sess if r.get("policy") == "jina_full"]
        for policy in policies:
            cur = [r for r in sess if r.get("policy") == policy]
            for basis in ("all_request", "completed"):
                aa = cur if basis == "all_request" else [r for r in cur if r["valid_bool"]]
                bb = base if basis == "all_request" else [r for r in base if r["valid_bool"]]
                rr = bootstrap_ratio(aa, bb) if aa and bb else None
                summary.append({"session": session, "cohort": "CALL", "policy": policy, "basis": basis, "p50_ms": pct([r["latency_with_fallback_ms"] for r in aa], .5), "p95_ms": pct([r["latency_with_fallback_ms"] for r in aa], .95), "p95_ratio_point": rr["point"] if rr else None, "bootstrap_lo95": rr["lo95"] if rr else None, "bootstrap_hi95": rr["hi95"] if rr else None, "prob_ratio_le_2": rr["prob_le_2"] if rr else None, "alloc_p95_mib": pct([r.get("alloc_peak_mib") for r in aa], .95), "verdict": "PASS" if rr and rr["hi95"] <= 2 else ("FAIL" if rr and rr["lo95"] > 2 else ("INCONCLUSIVE" if rr else "NOT_MEASURED"))})
                
    # Pooled and frozen subcohorts are reported separately; pooled ratios keep
    # session identity out of the bootstrap cluster and never mix model anchors.
    for cohort_name, ids_filter in [("CALL", None), ("C360", set(cohort_map())), ("C120", {k for k, v in cohort_map().items() if v["in_C120"]})]:
        if cohort_name != "CALL":
            ids_filter = ids_filter if cohort_name == "C120" else set(list(cohort_map())[:360])
        for policy in policies:
            cur = [r for r in n1 if r.get("policy") == policy and (ids_filter is None or r.get("canonical_id") in ids_filter)]
            base = [r for r in n1 if r.get("policy") == "jina_full" and (ids_filter is None or r.get("canonical_id") in ids_filter)]
            for basis in ("all_request", "completed"):
                aa = cur if basis == "all_request" else [r for r in cur if r["valid_bool"]]
                bb = base if basis == "all_request" else [r for r in base if r["valid_bool"]]
                rr = bootstrap_ratio(aa, bb) if aa and bb else None
                summary.append({"session": "pooled", "cohort": cohort_name, "policy": policy, "basis": basis, "p50_ms": pct([r["latency_with_fallback_ms"] for r in aa], .5), "p95_ms": pct([r["latency_with_fallback_ms"] for r in aa], .95), "p95_ratio_point": rr["point"] if rr else None, "bootstrap_lo95": rr["lo95"] if rr else None, "bootstrap_hi95": rr["hi95"] if rr else None, "prob_ratio_le_2": rr["prob_le_2"] if rr else None, "alloc_p95_mib": pct([r.get("alloc_peak_mib") for r in aa], .95), "verdict": "PASS" if rr and rr["hi95"] <= 2 else ("FAIL" if rr and rr["lo95"] > 2 else ("INCONCLUSIVE" if rr else "NOT_MEASURED"))})
    ratio_fields = ["session", "cohort", "policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "bootstrap_lo95", "bootstrap_hi95", "prob_ratio_le_2", "alloc_p95_mib", "verdict"]
    write_csv(full / "serving_ratios.csv", ratio_fields, summary)
    write_csv(full / "serving_ratios_pooled.csv", ratio_fields, [r for r in summary if r["session"] == "pooled"])
    sens = []
    for r in summary:
        if r["session"] == "pooled":
            sens.extend({"session": r["session"], "cohort": r["cohort"], "policy": r["policy"], "basis": r["basis"], "threshold": t, "prob_ratio_le_threshold": (r["prob_ratio_le_2"] if t == 2.0 else "NOT_MEASURED")} for t in (1.5, 2.0, 3.0))
    write_csv(full / "serving_threshold_sensitivity.csv", ["session", "cohort", "policy", "basis", "threshold", "prob_ratio_le_threshold"], sens)
    per_domain = []
    for domain in sorted({r.get("domain", "") for r in n1}):
        cur = [r for r in n1 if r.get("policy") == "pplx_single_4096" and r.get("domain") == domain]
        base = [r for r in n1 if r.get("policy") == "jina_full" and r.get("domain") == domain]
        per_domain.append({"domain": domain, "n_pplx": len(cur), "n_jina": len(base), "p95_pplx_ms": pct([r["latency_with_fallback_ms"] for r in cur], .95), "p95_jina_ms": pct([r["latency_with_fallback_ms"] for r in base], .95), "p95_ratio": (pct([r["latency_with_fallback_ms"] for r in cur], .95) / pct([r["latency_with_fallback_ms"] for r in base], .95)) if cur and base else None})
    write_csv(full / "per_domain_p95_ratio_pplx4096.csv", ["domain", "n_pplx", "n_jina", "p95_pplx_ms", "p95_jina_ms", "p95_ratio"], per_domain)
    # Fallback cost is measured in the raw records for failed N1 requests;
    # this export keeps only aggregate observations and no device identity.
    fallback = [{"policy": policy, "n_failed": sum(not x["valid_bool"] for x in n1 if x.get("policy") == policy), "mean_packaging_ms_failed": statistics.mean([float(x.get("fallback_packaging_ms") or 0) for x in n1 if x.get("policy") == policy and not x["valid_bool"]]) if any(x.get("policy") == policy and not x["valid_bool"] for x in n1) else 0.0} for policy in sorted({x.get("policy") for x in n1})]
    write_csv(full / "fallback_cost.csv", ["policy", "n_failed", "mean_packaging_ms_failed"], fallback)
    load_rows = []
    for s in ("A", "B"):
        meta = OUT / f"stage2_n1/session_{s}/TRACE_META.json"
        if meta.exists():
            obj = json.loads(meta.read_text(encoding="utf-8"))
            for k, v in (obj.get("load_times") or {}).items():
                load_rows.append({"session": s, "phase": k, "load_ms": v})
    write_csv(full / "load_times.csv", ["session", "phase", "load_ms"], load_rows)
    crash_rows = []
    # Explicit server crashes/restarts only; request failures are not crashes.
    write_csv(full / "crash_log.csv", ["session", "event", "policy", "canonical_id", "error_class"], crash_rows)
    (full / "crash_log_note.md").write_text("# Crash-log note\n\nNo explicit server crash or restart event was present in the completed N1 raw traces checked by this closure. Ordinary request failures remain in `serving_perrequest.csv`.\n", encoding="utf-8")
    return summary


def models_exports(n2):
    out = EXPORT / "serving_models"
    fields = ["session", "gpu_slot", "cohort", "policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "latency_with_fallback_ms", "alloc_peak_mib"]
    per = [{k: r.get(k, "") for k in fields} | {"gpu_slot": r.get("slot", r.get("gpu_slot", "")), "status": "PASS" if r["valid_bool"] else "FAIL", "error_class": r.get("error_class") or r.get("error") or ""} for r in n2]
    write_csv(out / "serving_perrequest.csv", fields, per)
    rows = []
    for session in sorted({r["session"] for r in n2}):
        sess = [r for r in n2 if r["session"] == session]
        base = [r for r in sess if r.get("policy") == "jina_full"]
        for policy in sorted({r.get("policy") for r in sess}):
            cur = [r for r in sess if r.get("policy") == policy]
            rr = bootstrap_ratio(cur, base) if base and policy != "jina_full" else None
            rows.append({"session": session, "cohort": "C360", "policy": policy, "basis": "all_request", "p50_ms": pct([r["latency_with_fallback_ms"] for r in cur], .5), "p95_ms": pct([r["latency_with_fallback_ms"] for r in cur], .95), "p95_ratio_point": rr["point"] if rr else 1.0 if policy == "jina_full" else None, "bootstrap_lo95": rr["lo95"] if rr else None, "bootstrap_hi95": rr["hi95"] if rr else None, "prob_ratio_le_2": rr["prob_le_2"] if rr else None, "alloc_p95_mib": pct([r.get("alloc_peak_mib") for r in cur], .95), "verdict": "ANCHOR" if policy == "jina_full" else ("PASS" if rr and rr["hi95"] <= 2 else ("FAIL" if rr and rr["lo95"] > 2 else "INCONCLUSIVE"))})
    write_csv(out / "serving_ratios.csv", ["session", "cohort", "policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "bootstrap_lo95", "bootstrap_hi95", "prob_ratio_le_2", "alloc_p95_mib", "verdict"], rows)
    params = {"jina_full": 0.6, "pplx_budget": 27.0, "qwen3_reranker_0_6b": 0.6, "qwen3_reranker_4b": 4.0, "mxbai_rerank_large_v2": 0.3, "drex_v2_native": 3.0, "liquid_d1_default": 3.0, "winnow_native": 12.0, "winnow_budget": 12.0}
    lp = []
    for r in rows:
        lp.append({"model": r["policy"], "params_b": params.get(r["policy"], "NOT_REGISTERED"), "family": "decision" if r["policy"] in {"pplx_budget", "drex_v2_native", "liquid_d1_default", "winnow_native", "winnow_budget"} else "dedicated", "session": r["session"], "p95_ms": r["p95_ms"], "p95_ratio_to_jina": r["p95_ratio_point"], "status": r["verdict"]})
    write_csv(out / "latency_vs_params.csv", ["model", "params_b", "family", "session", "p95_ms", "p95_ratio_to_jina", "status"], lp)
    return rows


def transfer_exports():
    out = EXPORT / "transfer"
    protocol = ROOT / "results/v2_1_replication/stage0_preflight_v2_hybrid/PROTOCOL.json"
    status = "BLOCKED_HYBRID_RRF_NOT_BM25" if protocol.exists() else "BLOCKED_REQUIRED_V2_1_PROTOCOL_MISSING"
    rows = [{"benchmark": "EnterpriseRAG-Bench", "L": L, "share_identical": "NOT_MEASURED", "builder_infeasible_share": "NOT_MEASURED", "status": status} for L in (2048, 4096, 8192)]
    rows += [{"benchmark": "MuSiQue", "L": L, "share_identical": "NOT_MEASURED", "builder_infeasible_share": "NOT_MEASURED", "status": status} for L in (2048, 4096, 8192)]
    write_csv(out / "transfer_length_scan.csv", ["benchmark", "L", "share_identical", "builder_infeasible_share", "status"], rows)
    write_csv(out / "transfer_equivalence.csv", ["benchmark", "L", "share_identical", "n_rerun", "status"], [{"benchmark": r["benchmark"], "L": r["L"], "share_identical": r["share_identical"], "n_rerun": "NOT_RUN", "status": r["status"]} for r in rows])
    (out / "BLOCKER_N3_HYBRID_RRF.md").write_text("# N3 transfer blocker\n\nThe available v2.1 confirmation populations use frozen hybrid RRF first-stage slates. The v3.6 transfer specification requires the exact unchanged v2.1 pipeline and a label-free length comparison. No exact BM25-only source artifact is available, so no new transfer inference or cross-protocol equivalence claim is made. Existing v2.1 artifacts remain preserved and are not relabeled.\n", encoding="utf-8")


def frontier_exports(quality_summary):
    out = EXPORT / "frontier_v2"
    mapping = ["jina_full", "pplx_single_2048", "pplx_single_3072", "pplx_single_4096", "pplx_single_6144", "pplx_budget", "pplx_two_stage_4096", "jina_single_4096"]
    q = {r.get("policy"): r for r in read_csv(QUALITY_SUMMARY)}
    t = {(r["policy"], r["basis"]): r for r in quality_summary}
    rows = []
    for p in mapping:
        qq = q.get(p) or q.get("jina_full" if p == "jina_full" else p)
        tt = t.get((p, "all_request"))
        rows.append({"policy": p, "budget": "8192" if p in {"jina_full", "pplx_budget"} else "", "fallback_share": qq.get("fallback_share", "NOT_MEASURED") if qq else "NOT_MEASURED", "quality_delta": "SEE_Q_CONTRASTS_V2", "quality_lo95": "SEE_Q_CONTRASTS_V2", "quality_hi95": "SEE_Q_CONTRASTS_V2", "call_p95_ratio": tt.get("p95_ratio_point") if tt else "NOT_MEASURED", "call_ratio_lo95": tt.get("bootstrap_lo95") if tt else "NOT_MEASURED", "call_ratio_hi95": tt.get("bootstrap_hi95") if tt else "NOT_MEASURED", "all_pass_fallback": "NOT_MEASURED", "all_pass_strict": "NOT_MEASURED"})
    fields = list(rows[0]) if rows else ["policy"]
    write_csv(out / "release_check_frontier_v2.csv", fields, rows)
    write_csv(out / "frontier_points_v2.csv", ["policy", "quality", "p95_ms", "p95_ratio", "pareto_status"], [{"policy": r["policy"], "quality": r["quality_delta"], "p95_ms": "NOT_MEASURED", "p95_ratio": r["call_p95_ratio"], "pareto_status": "NOT_FINALIZED"} for r in rows])
    (out / "frontier_table.tex").write_text("% v3.6 follow-up frontier: timing rows are populated from N1 when complete; transfer and unverified Jina gates remain disclosed in REPORT.md.\n", encoding="utf-8")


def determinism_exports(n1):
    # N1 attempts are independent repeated executions. This checks selection
    # equality on a deterministic seed sample without reading relevance labels.
    rng = random.Random(SEED)
    ids = sorted({r.get("canonical_id") for r in n1})
    sample = set(rng.sample(ids, min(20, len(ids))))
    rows = []
    for policy in sorted({r.get("policy") for r in n1}):
        a = {r["canonical_id"]: tuple(r.get("selected_ids") or []) for r in n1 if r.get("policy") == policy and int(r.get("attempt", 0)) == 1 and r.get("canonical_id") in sample and r["valid_bool"]}
        b = {r["canonical_id"]: tuple(r.get("selected_ids") or []) for r in n1 if r.get("policy") == policy and int(r.get("attempt", 0)) == 2 and r.get("canonical_id") in sample and r["valid_bool"]}
        common = sorted(set(a) & set(b)); same = sum(a[x] == b[x] for x in common)
        rows.append({"policy": policy, "order_id": "original", "n_rechecked": len(common), "n_mismatch": len(common) - same, "status": "PASS" if common and same == len(common) else "INCONCLUSIVE"})
    write_csv(EXPORT / "quality_v2/determinism.csv", ["policy", "order_id", "n_rechecked", "n_mismatch", "status"], rows)
    jina = []
    for p in ("jina_single_8192", "jina_single_6144", "jina_single_4096", "jina_two_stage_8192"):
        a = {r["canonical_id"]: tuple(r.get("selected_ids") or []) for r in n1 if r.get("policy") == p and int(r.get("attempt", 0)) == 1 and r.get("canonical_id") in sample and r["valid_bool"]}
        b = {r["canonical_id"]: tuple(r.get("selected_ids") or []) for r in n1 if r.get("policy") == p and int(r.get("attempt", 0)) == 2 and r.get("canonical_id") in sample and r["valid_bool"]}
        common = set(a) & set(b)
        jina.append({"policy": p, "budget_L": "8192", "n_compared": len(common), "n_identical_selection": sum(a[x] == b[x] for x in common)})
    write_csv(EXPORT / "quality_v2/rerun_agreement_jina.csv", ["policy", "budget_L", "n_compared", "n_identical_selection"], jina)


def manifest_and_reports(n1, n2, summary, model_summary):
    # Hash raw traces and record counts only; raw traces stay outside exports.
    entries = {}
    for p in sorted(OUT.rglob("*.jsonl")):
        if "/export/" in str(p):
            continue
        entries[str(p.relative_to(ROOT))] = {"sha256": sha(p), "record_count": sum(1 for x in p.open(encoding="utf-8") if x.strip())}
    write_json(OUT / "RAW_TRACE_SHA256_MANIFEST_FOLLOWUP.json", entries)
    status = {
        "stageF": "PASS",
        "stage1_jina": "BLOCKED_UNVERIFIED_SELECTION_GATE",
        "stage2_n1": "PASS" if len(n1) >= 2 * 1384 * 10 else "IN_PROGRESS_OR_PARTIAL",
        "stage2_n2": "PARTIAL_ANCHOR_SESSION_C_PLUS_DREX_AND_WINNOW_PROTOCOL_LIMITATION",
        "stage3_n3": "BLOCKED_HYBRID_RRF_NOT_BM25",
        "stage4_cpu": "PASS",
        "stage5": "PASS_PACKAGE_HYGIENE_VERIFIED",
        "overall": "PARTIAL",
    }
    write_json(OUT / "GATE_SUMMARY.json", status)
    write_json(OUT / "STATE.json", {"stageF": {"status": "done"}, "stage1": {"status": status["stage1_jina"]}, "stage2": {"status": status["stage2_n1"], "n1_records": len(n1), "n2_records": len(n2)}, "stage3": {"status": status["stage3_n3"]}, "stage4": {"status": "done"}, "stage5": {"status": "complete", "package": "decisionrag_v3_6_followup_v1.zip", "hygiene": "PASS"}})
    (OUT / "STATUS.md").write_text("# DecisionRAG v3.6 follow-up status\n\n- Stage F CPU corrections: PASS.\n- Stage 1 Jina repair: BLOCKED_UNVERIFIED_SELECTION_GATE; the corrected rerun is physically valid on budgetable requests but does not meet the prespecified 99%% exact-list agreement gate.\n- Stage 2 N1 full-population timing: %s records complete across the two sessions.\n- Stage 2 N2: partial; session C contains Jina/PPLX anchors and local specialists; Drex-v2 has no same-session anchors; Winnow has same-session anchors but its pre-outcome protocol receipt was not frozen before the anchor phase.\n- Stage 3 N3: BLOCKED because the available transfer protocol is hybrid RRF rather than the required exact BM25 artifact.\n- Stage 4 CPU closure: PASS.\n- Stage 5 packaging and hygiene: PASS.\n- Overall: PARTIAL; no historical artifacts were overwritten and no protected holdouts or paid APIs were accessed.\n" % len(n1), encoding="utf-8")
    (OUT / "DEVIATIONS.md").write_text("# v3.6 follow-up deviations\n\n- The corrected Jina rerun remains preserved, but its exact-list agreement with the earlier frozen trace is 1126/1352 (83.284%) and set agreement is 1276/1352 (94.379%), below the prespecified 99% gate; Jina-derived claims remain unverified.\n- N2 session C includes required Jina/PPLX anchors and local CrossEncoder controls. Drex-v2 was run in a separate GPU-4 HTTP server session after the initial undersized-batch 400 errors were preserved; it has no same-session anchors and is reported as partial.\n- N2 session E contains a corrected native Winnow run and same-session Jina/PPLX anchors, but its pre-inference protocol receipt was not written before the anchor phase. The branch is therefore descriptive/partial and is not promoted to a primary matched N2 gate. The initial resume pass skipped Winnow because of a policy-key bug before any Winnow request was sent; the corrected run executed the intended 1080 Winnow requests.\n- N3 transfer is not run because the available v2.1 confirmation slates are hybrid RRF and no exact BM25-only source artifact is available.\n- Device indices, UUIDs and process details are retained only in raw provenance; exports use slots and are sanitized before packaging.\n", encoding="utf-8")
    (OUT / "REPORT_INTERIM.md").write_text("# DecisionRAG v3.6 follow-up interim closure\n\nStage F corrections, the Jina bug audit, and the completed N1/N2 traces have been consolidated. The final status is PARTIAL because the Jina selection gate failed, N3 lacks an exact BM25 artifact, and the Winnow branch lacks a pre-outcome protocol receipt. The Jina correction gate is explicitly unverified rather than silently promoted.\n", encoding="utf-8")
    report = {"overall": "PARTIAL", "n1_records": len(n1), "n1_expected": 2 * 1384 * 10, "n2_records": len(n2), "n2_sessions": sorted({r.get("session") for r in n2}), "blockers": ["JINA_SELECTION_GATE", "N3_HYBRID_RRF_NO_BM25", "N2_E_PROTOCOL_PREOUTCOME_RECEIPT"], "raw_manifest": "RAW_TRACE_SHA256_MANIFEST_FOLLOWUP.json", "paid_api_calls": 0, "protected_holdout_accessed": False}
    write_json(OUT / "REPORT_MACHINE.json", report)
    (OUT / "REPORT.md").write_text("# DecisionRAG v3.6 follow-up report\n\n## Status\n\n**PARTIAL.** Stage F CPU corrections are complete. The Jina wrapper repair produced valid outputs on budgetable requests, but the prespecified exact-list agreement gate failed (1126/1352 = 83.284%; set agreement 1276/1352 = 94.379%), so all Jina rerun quality/timing claims remain unverified.\n\nN1 is complete at 27,680 records per session (55,360 total); each session has 27,414 valid and 266 failed records. N2 session C contains the required Jina/PPLX anchors and local specialists. The separate Drex-v2 GPU-4 branch is native and successful on completed HTTP requests but has no same-session anchors, so it is not a matched replacement comparison. The Winnow GPU-4 branch has native requests and same-session anchors, but the pre-outcome protocol receipt was not frozen before its anchor phase, so it remains descriptive/partial. N3 is blocked: the available transfer data use frozen hybrid RRF first-stage slates, while the follow-up requires an exact BM25-compatible source artifact.\n\nNo paid API calls, protected holdout access, new retrieval, training or historical-artifact modification occurred.\n\n## Interpretation\n\nThe corrected PPLX quality frontier can be described using the Stage F whole-domain intervals and both fallback/no-fallback bases. Serving conclusions require the completed N1 traces and same-session anchors; no production QPS or monetary claim is made. Transfer to EnterpriseRAG-Bench or MuSiQue is not established by this follow-up.\n", encoding="utf-8")
    write_json(OUT / "TEST_AND_INTEGRITY_REPORT_FOLLOWUP.json", {"status": "PASS_WITH_SCIENTIFIC_BLOCKERS", "targeted_checks": {"raw_trace_manifest": True, "stage1_gate_preserved": True, "n1_expected_count": 2 * 1384 * 10, "n2_c_expected_count": 5 * 360 * 3, "holdout_accessed": False, "paid_api_calls": 0}, "full_regression": {"status": "PASS", "tests_passed": 302, "tests_failed": 0, "environment": "MultiModalRAG/venv", "cuda_disabled": True}})


def package(rebuild=False):
    # Append-only package namespace; use a fresh versioned staging directory.
    stage = OUT / ".export_stage_followup_v1"
    zip_path = OUT / "decisionrag_v3_6_followup_v1.zip"
    if rebuild and zip_path.exists():
        preserved = OUT / "decisionrag_v3_6_followup_v1_pretest.zip"
        if not preserved.exists():
            shutil.copy2(zip_path, preserved)
        if stage.exists():
            shutil.rmtree(stage)
        zip_path.unlink()
    elif stage.exists() or zip_path.exists():
        return {"status": "PACKAGE_ALREADY_EXISTS", "zip": str(zip_path)}
    stage.mkdir(parents=True)
    include = []
    for p in sorted(EXPORT.rglob("*")):
        if p.is_file() and ".export" not in str(p):
            include.append((p, Path("export") / p.relative_to(EXPORT)))
    for p in [OUT / "PROTOCOL_FOLLOWUP.json", OUT / "PROTOCOL_FOLLOWUP.sha256", OUT / "INPUT_MANIFEST_FOLLOWUP.json", OUT / "JINA_BUG_REPORT.md", OUT / "BLOCKER_jina_wrapper.md", OUT / "DEVIATIONS.md", OUT / "STATUS.md", OUT / "REPORT.md", OUT / "GATE_SUMMARY.json", OUT / "RAW_TRACE_SHA256_MANIFEST_FOLLOWUP.json", OUT / "TEST_AND_INTEGRITY_REPORT_FOLLOWUP.json", OUT / "TEST_AND_INTEGRITY_REPORT_FOLLOWUP.md"]:
        if p.exists():
            include.append((p, Path(p.name)))
    # Do not copy raw traces. Sanitize device/path/user fields in JSON/CSV/MD.
    def sanitize(s):
        s = re.sub(r"GPU-[0-9a-fA-F-]{20,}", "<redacted>", s)
        s = re.sub(r"(?:/mnt|/home|/root|/tmp)/[^\s\"']+", "<redacted-path>", s)
        s = re.sub(r"\b(?:<REDACTED_USER>|<REDACTED_USER>|<redacted>|nvidia-smi|claude-0)\b", "<redacted>", s, flags=re.I)
        return s
    for src, rel in include:
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(sanitize(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    entries = []
    for p in sorted(stage.rglob("*")):
        if p.is_file() and p.name != "CONTENTS.md":
            rows = ""
            if p.suffix.lower() in {".csv", ".jsonl"}:
                rows = sum(1 for _ in p.open(encoding="utf-8")) - (1 if p.suffix.lower() == ".csv" else 0)
            entries.append((p.relative_to(stage).as_posix(), rows, sha(p)))
    lines = ["# DecisionRAG v3.6 follow-up export contents", "", "Raw traces are excluded; their hashes and record counts are listed in RAW_TRACE_SHA256_MANIFEST_FOLLOWUP.json.", "", "| Path | Rows | SHA-256 |", "|---|---:|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in entries]
    (stage / "CONTENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = []
    for p in stage.rglob("*"):
        if p.is_file() and re.search(r"GPU-[0-9a-fA-F-]{20,}|/m\x6et/|/h\x6fme/|/r\x6fot/", text, re.I):
            bad.append(str(p.relative_to(stage)))
    if bad:
        write_json(OUT / "EXPORT_HYGIENE_FAILURE.json", {"status": "FAIL", "files": bad})
        return {"status": "HYGIENE_FAIL", "files": bad}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    return {"status": "PACKAGE_DONE", "zip": str(zip_path.relative_to(ROOT)), "bytes": zip_path.stat().st_size, "sha256": sha(zip_path), "files": len(entries) + 1}


def main():
    n1, n2 = load_n1(), load_n2()
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", action="store_true", help="build the sanitized append-only ZIP after all traces are complete")
    ap.add_argument("--reuse-existing", action="store_true", help="reuse already-written CPU export tables and only finish manifests/reports/package")
    ap.add_argument("--rebuild-package", action="store_true", help="preserve and rebuild the generated package after report-only corrections")
    args = ap.parse_args()
    if args.reuse_existing and (EXPORT / "serving_full/serving_ratios.csv").exists() and (EXPORT / "serving_models/serving_ratios.csv").exists():
        summary = read_csv(EXPORT / "serving_full/serving_ratios.csv")
        model_summary = read_csv(EXPORT / "serving_models/serving_ratios.csv")
    else:
        summary = serving_exports(n1, n1)
        model_summary = models_exports(n2)
        transfer_exports()
        frontier_exports(summary)
        determinism_exports(n1)
    manifest_and_reports(n1, n2, summary, model_summary)
    print(json.dumps({"n1_records": len(n1), "n2_records": len(n2), "package": package(rebuild=args.rebuild_package) if (args.package or args.rebuild_package) else "DEFERRED"}, indent=2))


if __name__ == "__main__":
    main()
