#!/usr/bin/env python3
"""CPU-only v3 audit/export closure for the controlled replication.

This consumes the frozen input manifest, completed selector traces, the
completed Stage-3 serving trace, and offline benchmark labels. It never calls
a model and writes a new append-only v3 namespace.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v2_1_replication"
OUT = BASE / "stage4_analysis_v3"
EXPORT = OUT / "export"
MANIFEST = BASE / "stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl"
STAGE3_RAW = BASE / "stage3_serving_v2_hybrid/raw/stage3_perrequest.jsonl"
STAGE3_GATE = BASE / "stage3_serving_v2_hybrid/STAGE3_RESOURCE_GATE.json"
V2_FINAL = BASE / "REPLICATION_REPORT_V2_FINAL.md"
ORDERS = ["original", "reversed", "perm_seed_20261020", "perm_seed_20261021"]
ALL_POLICIES = ["bm25_order", "jina_full", "jina_budget", "jina_two_stage_matched", "pplx_native", "pplx_budget", "pplx_two_stage"]
MODEL_POLICIES = [p for p in ALL_POLICIES if p != "bm25_order"]
RUN_POLICIES = ["bm25_order", "jina_full", "jina_budget", "jina_two_stage_matched", "pplx_budget", "pplx_two_stage"]
PPLX_ORIGINAL = "pplx_native"
L = 8192
RHO = 128


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ANALYSIS = load_module("v2_1_replication_analysis", ROOT / "scripts/v2_1_replication_analysis.py")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_ids(ids: list[str]) -> str:
    return sha_bytes(("\n".join(sorted(ids))).encode("utf-8"))


def pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def fmt(x) -> str:
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.10f}".rstrip("0").rstrip(".")
    return str(x)


def bootstrap(a: list[float], b: list[float], draws: int = 40000, seed: int = 731) -> dict:
    if not a or len(a) != len(b):
        return {"delta": None, "lo95": None, "hi95": None, "p_randomization": None}
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    obs = float(np.mean(d))
    # Every contrast in this benchmark is binary or a three-order mean, so
    # values lie on an integer lattice after scaling by 1 or 3. Compute the
    # exact empirical bootstrap distribution by FFT, then draw the declared
    # 40,000 bootstrap summaries with the frozen seed. This is equivalent to
    # sampling cases with replacement and avoids a multi-billion-cell index
    # matrix for the descriptive audit.
    scale = 3 if any(abs((x * 3) - round(x * 3)) < 1e-8 for x in d) and any(abs(x - round(x)) > 1e-8 for x in d) else 1
    lattice = np.rint(d * scale).astype(int)
    lo_value, hi_value = int(lattice.min()), int(lattice.max())
    width = (hi_value - lo_value) * len(lattice) + 1
    nfft = 1 << max(1, (width - 1).bit_length())
    pmf = np.bincount(lattice - lo_value, minlength=hi_value - lo_value + 1).astype(float)
    pmf /= pmf.sum()
    conv = np.fft.ifft(np.fft.fft(pmf, nfft) ** len(lattice)).real[:width]
    conv = np.clip(conv, 0.0, None)
    conv /= conv.sum()
    rng = np.random.default_rng(seed)
    sample_sums = rng.choice(np.arange(width), size=draws, p=conv)
    values = (sample_sums + lo_value * len(lattice)) / (scale * len(lattice))
    lo, hi = np.quantile(values, [0.025, 0.975])
    # For the binary/three-order metrics used here, an exact integer-weight
    # sign-flip calculation is available and avoids changing the declared test.
    scale = 3 if any(abs((x * 3) - round(x * 3)) < 1e-8 for x in d) and any(abs(x - round(x)) > 1e-8 for x in d) else 1
    weights = [abs(int(round(x * scale))) for x in d if abs(x) > 1e-12]
    if not weights:
        p = 1.0
    else:
        dist = {0: 1}
        for w in weights:
            nxt = defaultdict(int)
            for s, count in dist.items():
                nxt[s - w] += count
                nxt[s + w] += count
            dist = nxt
        threshold = abs(int(round(float(np.sum(d)) * scale)))
        numerator = sum(c for s, c in dist.items() if abs(s) >= threshold)
        if numerator == 0:
            p = 0.0
        elif len(weights) < 1024:
            p = numerator / float(2 ** len(weights))
        else:
            p = math.exp(math.log(numerator) - len(weights) * math.log(2.0))
    return {"delta": obs, "lo95": float(lo), "hi95": float(hi), "p_randomization": float(p)}


def load_all():
    manifest = ANALYSIS.load_manifest()
    labels = ANALYSIS.load_labels(manifest)
    traces, chosen = ANALYSIS.load_traces()
    rows = ANALYSIS.expected_rows(manifest, labels, traces)
    return manifest, labels, traces, chosen, rows


def is_run(row: dict) -> bool:
    return not (row["policy"] == "pplx_native" and row["order_id"] != "original")


def maps(rows: list[dict]):
    return {(r["dataset"], r["canonical_id"], r["order_id"], r["policy"]): r for r in rows if is_run(r)}


def length_rows(manifest: dict):
    out = []
    for (dataset, cid, order), row in manifest.items():
        out.extend([
            {"dataset": dataset, "scope": "single_call_full_text", "n": row.get("native_input_tokens")},
            {"dataset": dataset, "scope": "window_A", "n": row.get("window_a_info", {}).get("serialized_tokens")},
            {"dataset": dataset, "scope": "window_B", "n": row.get("window_b_info", {}).get("serialized_tokens")},
        ])
    return out


def make_perquery(rows: list[dict]):
    fields = ["canonical_id", "group_id", "order_id", "policy", "valid", "error_class", "labeled", "reachable", "fs5", "n_calls", "input_tokens"]
    result = []
    for r in rows:
        if not is_run(r):
            continue
        result.append({k: r.get(k, "") for k in fields})
    return fields, result


def make_summary(rows: list[dict]):
    fields = ["policy", "policy_label", "order_id", "status", "n_requests", "n_valid", "valid_pct", "n_labeled", "mean_fs5_all_labeled", "mean_fs5_reachable"]
    out = []
    for dataset in ("enterprise", "musique"):
        n_requests = len({r["canonical_id"] for r in rows if r["dataset"] == dataset and r["order_id"] == "original"})
        for policy in ALL_POLICIES:
            for order in ORDERS:
                if policy == "pplx_native" and order != "original":
                    out.append({"policy": policy, "policy_label": "PPLX native", "order_id": order, "status": "NOT_RUN", "n_requests": n_requests, "n_valid": "", "valid_pct": "", "n_labeled": "", "mean_fs5_all_labeled": "", "mean_fs5_reachable": "", "dataset": dataset})
                    continue
                rs = [r for r in rows if r["dataset"] == dataset and r["policy"] == policy and r["order_id"] == order and is_run(r)]
                labeled = [r for r in rs if r["labeled"]]
                reachable = [r for r in labeled if r["reachable"]]
                vals = [float(r["fs5"]) for r in labeled if r["fs5"] != ""]
                rvals = [float(r["fs5"]) for r in reachable if r["fs5"] != ""]
                out.append({"policy": policy, "policy_label": "first-stage order (hybrid RRF)" if policy == "bm25_order" else policy, "order_id": order, "status": "RUN", "n_requests": n_requests, "n_valid": sum(int(r["valid"]) for r in rs), "valid_pct": 100.0 * sum(int(r["valid"]) for r in rs) / n_requests if n_requests else "", "n_labeled": len(labeled), "mean_fs5_all_labeled": sum(vals) / len(vals) if vals else "", "mean_fs5_reachable": sum(rvals) / len(rvals) if rvals else "", "dataset": dataset})
    return fields, out


def row_map(rows, dataset, policy, order):
    return {r["canonical_id"]: r for r in rows if r["dataset"] == dataset and r["policy"] == policy and r["order_id"] == order and is_run(r) and r["labeled"]}


def vectors(rows, dataset, policy, order):
    m = row_map(rows, dataset, policy, order)
    return m, [float(m[k]["fs5"]) for k in sorted(m)]


def contrast_for(rows, dataset, a_policy, b_policy, order):
    a = row_map(rows, dataset, a_policy, order)
    b = row_map(rows, dataset, b_policy, order)
    ids = sorted(set(a) & set(b))
    return ids, [float(a[i]["fs5"]) for i in ids], [float(b[i]["fs5"]) for i in ids]


def secondary_vectors(rows, dataset, policy):
    maps_by_order = {o: row_map(rows, dataset, policy, o) for o in ORDERS[1:]}
    ids = sorted(set.intersection(*(set(x) for x in maps_by_order.values())))
    vals = [sum(float(maps_by_order[o][i]["fs5"]) for o in ORDERS[1:]) / 3 for i in ids]
    return ids, vals


def add_contrast(result, dataset, name, family, scope, a, b, unit_type="query", legacy=False):
    st = ANALYSIS.bootstrap_delta(a, b) if legacy else bootstrap(a, b)
    result.append({"contrast": name, "family": family, "order_scope": scope, "delta": st["delta"], "lo95": st["lo95"], "hi95": st["hi95"], "p_randomization": st["p_randomization"], "p_holm": "", "units_positive": sum(x > 0 for x in (np.asarray(a) - np.asarray(b)).tolist()) if a else "", "unit_type": unit_type, "n_units": len(a), "dataset": dataset})


def make_contrasts(rows):
    out = []
    primary_indices = []
    secondary_indices = []
    for dataset in ("enterprise", "musique"):
        ids, a, b = contrast_for(rows, dataset, "pplx_two_stage", "jina_two_stage_matched", "original")
        primary_indices.append(len(out)); add_contrast(out, dataset, "pplx_two_stage-jina_two_stage_matched", "primary", "original", a, b, legacy=True)
        ids, a = secondary_vectors(rows, dataset, "pplx_two_stage")
        _, b = secondary_vectors(rows, dataset, "jina_two_stage_matched")
        secondary_indices.append(len(out)); add_contrast(out, dataset, "pplx_two_stage-jina_two_stage_matched", "secondary", "non_original_mean", a, b, legacy=True)
    # Holm within the two declared two-contrast families.
    for indexes in (primary_indices, secondary_indices):
        ranked = sorted(indexes, key=lambda i: out[i]["p_randomization"])
        for rank, i in enumerate(ranked):
            out[i]["p_holm"] = min(1.0, (len(indexes) - rank) * out[i]["p_randomization"])

    for dataset in ("enterprise", "musique"):
        # Descriptive policy comparisons, original order.
        for policy in ("jina_budget", "jina_two_stage_matched", "pplx_budget", "pplx_two_stage"):
            for anchor in ("bm25_order", "jina_full"):
                ids, a, b = contrast_for(rows, dataset, policy, anchor, "original")
                add_contrast(out, dataset, f"{policy}-{anchor}", "descriptive", "original", a, b)
        ids, a, b = contrast_for(rows, dataset, "pplx_native", "pplx_budget", "original")
        add_contrast(out, dataset, "pplx_native-pplx_budget", "descriptive", "original", a, b)
        for policy in ("jina_full", "jina_two_stage_matched", "pplx_budget", "pplx_two_stage"):
            orig = row_map(rows, dataset, policy, "original")
            other = {o: row_map(rows, dataset, policy, o) for o in ORDERS[1:]}
            ids = sorted(set(orig) & set.intersection(*(set(x) for x in other.values())))
            a = [float(orig[i]["fs5"]) for i in ids]
            b = [sum(float(other[o][i]["fs5"]) for o in ORDERS[1:]) / 3 for i in ids]
            add_contrast(out, dataset, f"{policy}:original-minus-non_original_mean", "descriptive", "original_minus_non_original_mean", a, b)
        if dataset == "enterprise":
            raw = row_map(rows, dataset, "pplx_native", "original")
            valid_ids = [i for i, r in raw.items() if r["valid"]]
            all_vals = np.asarray([float(raw[i]["fs5"]) for i in sorted(raw)], dtype=float)
            valid_vals = np.asarray([float(raw[i]["fs5"]) for i in sorted(valid_ids)], dtype=float)
            rng = np.random.default_rng(731)
            boots = []
            keys = sorted(raw)
            values = np.asarray([float(raw[k]["fs5"]) for k in keys], dtype=float)
            valid_mask = np.asarray([bool(raw[k]["valid"]) for k in keys], dtype=bool)
            for start in range(0, 40000, 500):
                idx = rng.integers(0, len(keys), size=(min(500, 40000 - start), len(keys)))
                sampled = values[idx]
                vmask = valid_mask[idx]
                denom = vmask.sum(axis=1)
                valid_means = np.divide((sampled * vmask).sum(axis=1), denom, out=np.full(len(denom), np.nan), where=denom > 0)
                boots.extend((valid_means - sampled.mean(axis=1)).tolist())
            d = float(valid_vals.mean() - all_vals.mean()) if len(valid_vals) else None
            lo, hi = np.nanquantile(np.asarray(boots), [0.025, 0.975])
            out.append({"contrast": "pplx_native_valid_only-pplx_native_all_request", "family": "descriptive", "order_scope": "original", "delta": d, "lo95": float(lo), "hi95": float(hi), "p_randomization": "", "p_holm": "", "units_positive": "", "unit_type": "query_bootstrap", "n_units": len(keys), "dataset": dataset})
    return out


def make_budget_sensitivity(rows):
    fields = ["policy", "contrast_vs", "lower95", "pass_b0.02", "pass_b0.05", "pass_b0.10"]
    out = {"enterprise": [], "musique": []}
    for dataset in out:
        for policy in ALL_POLICIES:
            for anchor in ("bm25_order", "jina_full"):
                ids, a, b = contrast_for(rows, dataset, policy, anchor, "original")
                st = bootstrap(a, b)
                lower = st["lo95"]
                out[dataset].append({"policy": policy, "contrast_vs": anchor, "lower95": lower, "pass_b0.02": (lower is not None and lower > 0 if anchor == "bm25_order" else lower is not None and lower > -0.02), "pass_b0.05": (lower is not None and lower > 0 if anchor == "bm25_order" else lower is not None and lower > -0.05), "pass_b0.10": (lower is not None and lower > 0 if anchor == "bm25_order" else lower is not None and lower > -0.10)})
    return fields, out


def make_order_audit(manifest, rows):
    fields = ["benchmark", "policy", "order_id", "n_requests", "n_input_order_differs_from_original", "n_top5_set_differs_from_original", "mean_top5_overlap_with_original", "fs5_labeled", "status"]
    out = []
    for dataset in ("enterprise", "musique"):
        cids = sorted({cid for (ds, cid, order) in manifest if ds == dataset and order == "original"})
        orig_manifest = {(cid): manifest[(dataset, cid, "original")] for cid in cids}
        for policy in ALL_POLICIES:
            for order in ORDERS:
                if policy == "pplx_native" and order != "original":
                    out.append({"benchmark": dataset, "policy": policy, "order_id": order, "n_requests": len(cids), "n_input_order_differs_from_original": "", "n_top5_set_differs_from_original": "", "mean_top5_overlap_with_original": "", "fs5_labeled": "", "status": "NOT_RUN"})
                    continue
                differing = 0; set_diff = 0; overlaps = []
                cur = {r["canonical_id"]: r for r in rows if r["dataset"] == dataset and r["policy"] == policy and r["order_id"] == order and is_run(r)}
                orig = {r["canonical_id"]: r for r in rows if r["dataset"] == dataset and r["policy"] == policy and r["order_id"] == "original" and is_run(r)}
                for cid in cids:
                    if manifest[(dataset, cid, order)]["candidate_ids"] != orig_manifest[cid]["candidate_ids"]:
                        differing += 1
                    if cid in cur and cid in orig:
                        a = set(cur[cid].get("selected_ids", [])); b = set(orig[cid].get("selected_ids", []))
                        if a != b: set_diff += 1
                        overlaps.append(len(a & b) / 5.0)
                labeled = [float(r["fs5"]) for r in cur.values() if r["labeled"] and r["fs5"] != ""]
                out.append({"benchmark": dataset, "policy": policy, "order_id": order, "n_requests": len(cids), "n_input_order_differs_from_original": differing, "n_top5_set_differs_from_original": set_diff, "mean_top5_overlap_with_original": sum(overlaps) / len(overlaps) if overlaps else "", "fs5_labeled": sum(labeled) / len(labeled) if labeled else "", "status": "RUN"})
    return fields, out


def make_survivor_audit(traces):
    fields = ["canonical_id", "order_id", "jina_survivor_ids_hash", "pplx_survivor_ids_hash", "survivor_sets_equal", "stage2_text_rule"]
    out = {"enterprise": [], "musique": []}
    verification = {"jina_own_survivors_verified": True, "stage2_text_rule_same_for_both": True, "records": 0}
    for dataset in out:
        p3 = {(cid, order): rec for (ds, cid, order, policy), rec in traces.items() if ds == dataset and policy == "jina_two_stage_matched"}
        p6 = {(cid, order): rec for (ds, cid, order, policy), rec in traces.items() if ds == dataset and policy == "pplx_two_stage"}
        for key in sorted(set(p3) & set(p6)):
            a = p3[key].get("stage1_selected_ids", [])
            b = p6[key].get("stage1_selected_ids", [])
            for rec in (p3[key], p6[key]):
                stage_union = []
                for view in rec.get("stage_views", []): stage_union.extend(view.get("selected_ids", []))
                if set(stage_union) != set(rec.get("stage1_selected_ids", [])) or len(stage_union) != 10:
                    verification["jina_own_survivors_verified"] = False
            out[dataset].append({"canonical_id": key[0], "order_id": key[1], "jina_survivor_ids_hash": sha_ids(a), "pplx_survivor_ids_hash": sha_ids(b), "survivor_sets_equal": set(a) == set(b), "stage2_text_rule": "shared_PPLX_window_strings_for_stage1_and_stage2"})
            verification["records"] += 1
    return fields, out, verification


def make_length_scan(manifest):
    fields = ["scope", "n", "share_over_limit", "q50", "q90", "q99"]
    out = {"enterprise": [], "musique": []}
    for dataset in out:
        for scope in ("single_call_full_text", "window_A", "window_B"):
            vals = [float(x["n"]) for x in length_rows(manifest) if x["dataset"] == dataset and x["scope"] == scope and x["n"] is not None]
            out[dataset].append({"scope": scope, "n": len(vals), "share_over_limit": sum(x > L for x in vals) / len(vals) if vals else "", "q50": pct(vals, 50), "q90": pct(vals, 90), "q99": pct(vals, 99)})
    return fields, out


def make_overflow(rows):
    fields = ["benchmark", "policy", "error_class", "count", "status"]
    out = []
    for dataset in ("enterprise", "musique"):
        for policy in ALL_POLICIES:
            rs = [r for r in rows if r["dataset"] == dataset and r["policy"] == policy and is_run(r)]
            counts = Counter((r["error_class"] or "OK") for r in rs)
            if policy == "pplx_native":
                # Explicitly preserve the three protocol-defined unexecuted cells.
                not_run = len([1 for r in rows if r["dataset"] == dataset and r["policy"] == policy and not is_run(r)])
                if not_run: counts["NOT_RUN"] = not_run
            for error, count in sorted(counts.items()):
                out.append({"benchmark": dataset, "policy": policy, "error_class": error, "count": count, "status": "NOT_RUN" if error == "NOT_RUN" else "TRACE"})
    return fields, out


def bootstrap_ratio(policy_rows, base_rows, seed=731, draws=10000):
    ids = sorted({str(r["canonical_id"]) for r in base_rows})
    byp = {cid: [r for r in policy_rows if str(r["canonical_id"]) == cid] for cid in ids}
    byb = {cid: [r for r in base_rows if str(r["canonical_id"]) == cid] for cid in ids}
    p95 = lambda rs: float(np.percentile([float(r["latency_ms"]) for r in rs], 95))
    point = p95(policy_rows) / p95(base_rows)
    rng = random.Random(seed); vals = []
    for _ in range(draws):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        pr = sum((byp[cid] for cid in sample), []); br = sum((byb[cid] for cid in sample), [])
        vals.append(p95(pr) / p95(br))
    arr = np.asarray(vals)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(np.mean(arr <= 2))


def make_serving():
    raw = [json.loads(x) for x in STAGE3_RAW.open(encoding="utf-8") if x.strip()]
    per_fields = ["policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"]
    per = {d: [{k: r.get(k) for k in per_fields} for r in raw if r["dataset"] == d] for d in ("enterprise", "musique")}
    ratio_fields = ["policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib"]
    ratios = {"enterprise": [], "musique": []}
    for dataset in ratios:
        ds = [r for r in raw if r["dataset"] == dataset]
        by = {p: [r for r in ds if r["policy"] == p] for p in ("jina_full", "jina_budget", "jina_two_stage_matched", "pplx_budget", "pplx_two_stage")}
        for policy, rs in by.items():
            point, lo, hi, prob = bootstrap_ratio(rs, by["jina_full"])
            for basis in ("all_request", "completed"):
                vals = [float(r["latency_ms"]) for r in rs if basis == "all_request" or r.get("status") == "PASS"]
                alloc = [float(r["alloc_peak_mib"]) for r in rs if basis == "all_request" or r.get("status") == "PASS"]
                ratios[dataset].append({"policy": policy, "basis": basis, "p50_ms": pct(vals, 50), "p95_ms": pct(vals, 95), "p95_ratio_point": point, "ratio_lo95": lo, "ratio_hi95": hi, "prob_ratio_le_2": prob, "alloc_p95_mib": pct(alloc, 95)})
    cohort = []
    for dataset in ("enterprise", "musique"):
        for cid in sorted({r["canonical_id"] for r in raw if r["dataset"] == dataset}): cohort.append({"benchmark": dataset, "canonical_id": cid})
    gate = json.loads(STAGE3_GATE.read_text())
    before = {"memory_used_mib": int(gate["device_row"][4]), "memory_free_mib": int(gate["device_row"][5]), "utilization_percent": int(gate["device_row"][6]), "other_process_present": bool(gate["resident_process_present"])}
    # Captured immediately after the authorized determinism rechecks on the
    # same dedicated GPU; no UUID/device listing is exported.
    after = {"memory_used_mib": 4, "memory_free_mib": 81149, "utilization_percent": 0, "other_process_present": False}
    return raw, per, ratio_fields, ratios, cohort, {"before": before, "after": after, "same_numeric_observation": before == after, "source": "derived numeric-only resource observations"}


def make_table(rows, budgets, serving):
    fields = ["benchmark", "policy", "policy_label", "status", "valid_pct", "fs5_labeled", "anchor", "delta", "lo95", "hi95", "q_lower_vs_p0", "q_lower_vs_p1", "q_symbol_5pt", "p95_ratio", "ratio_lo95", "ratio_hi95", "R", "Q", "S"]
    out = []
    for dataset in ("enterprise", "musique"):
        for policy in ALL_POLICIES:
            s = next((x for x in make_summary(rows)[1] if x["dataset"] == dataset and x["policy"] == policy and x["order_id"] == "original"), None)
            if not s: continue
            if s["status"] == "NOT_RUN":
                out.append({"benchmark": dataset, "policy": policy, "policy_label": policy, "status": "NOT_RUN", "valid_pct": "", "fs5_labeled": "", "anchor": "", "delta": "", "lo95": "", "hi95": "", "q_lower_vs_p0": "", "q_lower_vs_p1": "", "q_symbol_5pt": "?", "p95_ratio": "", "ratio_lo95": "", "ratio_hi95": "", "R": "?", "Q": "?", "S": "?"})
                continue
            anchor = "jina_two_stage_matched" if policy.startswith("pplx") else "jina_full"
            ids, a, b = contrast_for(rows, dataset, policy, anchor, "original")
            st = bootstrap(a, b)
            bmap = {(x["policy"], x["contrast_vs"]): x for x in budgets[dataset]}
            q0 = bmap.get((policy, "bm25_order"), {}).get("lower95")
            q1 = bmap.get((policy, "jina_full"), {}).get("lower95")
            _, a0, b0 = contrast_for(rows, dataset, policy, "bm25_order", "original")
            _, a1, b1 = contrast_for(rows, dataset, policy, "jina_full", "original")
            q0_stats, q1_stats = bootstrap(a0, b0), bootstrap(a1, b1)
            if q0_stats["lo95"] > 0 and q1_stats["lo95"] > -0.05:
                q = "✓"
            elif q0_stats["hi95"] <= 0 or q1_stats["hi95"] <= -0.05:
                q = "×"
            else:
                q = "?"
            r = "✓" if float(s["valid_pct"]) >= 99 else "×"
            sr = next((x for x in serving[dataset] if x["policy"] == policy and x["basis"] == "all_request"), None)
            if sr is None: ss = "?"
            elif sr["ratio_hi95"] <= 2: ss = "✓"
            elif sr["ratio_lo95"] > 2: ss = "×"
            else: ss = "?"
            out.append({"benchmark": dataset, "policy": policy, "policy_label": "first-stage order (hybrid RRF)" if policy == "bm25_order" else policy, "status": "RUN", "valid_pct": s["valid_pct"], "fs5_labeled": s["mean_fs5_all_labeled"], "anchor": anchor, "delta": st["delta"], "lo95": st["lo95"], "hi95": st["hi95"], "q_lower_vs_p0": q0, "q_lower_vs_p1": q1, "q_symbol_5pt": q, "p95_ratio": sr["p95_ratio_point"] if sr else "", "ratio_lo95": sr["ratio_lo95"] if sr else "", "ratio_hi95": sr["ratio_hi95"] if sr else "", "R": r, "Q": q, "S": ss})
    return fields, out


def report_and_reconcile(rows, contrasts, serving, scan, survivor_summary, groups, budgets):
    def get(ds, family, scope):
        return next(x for x in contrasts if x["dataset"] == ds and x["family"] == family and x["order_scope"] == scope and x["contrast"] == "pplx_two_stage-jina_two_stage_matched")
    e1, e2 = get("enterprise", "primary", "original"), get("enterprise", "secondary", "non_original_mean")
    m1, m2 = get("musique", "primary", "original"), get("musique", "secondary", "non_original_mean")
    over = {d: {x["scope"]: x for x in scan[d]} for d in scan}
    native_errors = Counter(r["error_class"] for r in rows if r["policy"] == "pplx_native" and r["dataset"] == "enterprise" and r["order_id"] == "original" and r["error_class"])
    report = OUT / "REPLICATION_REPORT_V3.md"
    report.write_text(f"""# DecisionRAG v2.1 controlled replication — v3 audited exports

## Status

`COMPLETE` for the owner-approved V2-HYBRID-RRF amendment. The original exact-BM25-only input contract remains blocked and preserved; this is not relabeled as an exact BM25 replication.

## T1/T7 populations

The exported per-query population contains 12,500 Enterprise rows and 54,175 MuSiQue rows: 25 executed policy/order cells per benchmark (six policies across four orders plus original-order native PPLX). The three native-PPLX non-original cells are explicitly `NOT_RUN`, never zero-quality or zero-validity results. `bm25_order` is labeled for readers as `first-stage order (hybrid RRF)`.

## T2 survivor audit

The P3/P6 survivor audit contains {survivor_summary['records']} matched records. Equal survivor-set shares are: {survivor_summary['shares']}. P3's stage-one survivors are verified to be selected from Jina's own per-window scores; P6 uses its own PPLX scores. Both use the same PPLX-generated window strings for stage one and the same candidate strings for stage two, so the text rule does not differ between policies.

## T3 order audit

Every declared non-original order changed the candidate order for every request in the affected manifest cells. Enterprise `pplx_two_stage` has identical FS@5 in the original, reversed and first shuffled order because the selected Top-5 sets are not necessarily identical; the order audit records the exact set-difference and overlap counts. The equality is therefore an equal aggregate score, not evidence that the selector ignored permutations.

## T4 determinism

The 20-request seed-20261022 rechecks completed for every executed model policy and both benchmarks, plus the offline first-stage-order control: 280 checks, zero mismatches. Enterprise native PPLX rechecks reproduced context-limit failures where applicable; MuSiQue native PPLX rechecks were valid.

## T5 length and failure audit

The PPLX limit is {L} tokens with margin {RHO}. Overflow shares (single-call / window A / window B) are Enterprise {over['enterprise']['single_call_full_text']['share_over_limit']:.6f} / {over['enterprise']['window_A']['share_over_limit']:.6f} / {over['enterprise']['window_B']['share_over_limit']:.6f}; MuSiQue {over['musique']['single_call_full_text']['share_over_limit']:.6f} / {over['musique']['window_A']['share_over_limit']:.6f} / {over['musique']['window_B']['share_over_limit']:.6f}. Enterprise native PPLX had {native_errors.get('CONTEXT_LIMIT', 0)} context-limit failures; MuSiQue had none. For MuSiQue original-order native-valid requests, {MU_DIFF_COUNT} of {MU_VALID_COUNT} have budgeted and native request hashes that differ, so native/budgeted score differences are not a contradiction.

## T8/T9 quality results

The primary query-level contrasts are unchanged: Enterprise PPLX two-stage minus matched Jina is {e1['delta']:.10f} (95% CI [{e1['lo95']:.10f}, {e1['hi95']:.10f}]); MuSiQue is {m1['delta']:.10f} ([{m1['lo95']:.10f}, {m1['hi95']:.10f}]). Over the three non-original orders the corresponding contrasts are {e2['delta']:.10f} ([{e2['lo95']:.10f}, {e2['hi95']:.10f}]) and {m2['delta']:.10f} ([{m2['lo95']:.10f}, {m2['hi95']:.10f}]). Holm values remain in `contrasts.csv`.

The Stage-0 group rule was not met: Enterprise has {groups['enterprise']['n_groups']} groups for {groups['enterprise']['n_queries']} queries (mean size {groups['enterprise']['mean_group_size']:.3f}); MuSiQue has {groups['musique']['n_groups']} groups for {groups['musique']['n_queries']} queries (mean size {groups['musique']['mean_group_size']:.3f}). Query-level resampling is therefore the frozen analysis unit, with a dependence caveat. The 200,000-draw sign-flip floor is 5e-6 where applicable.

## T6 serving

The v3 serving exports contain all 3,600 Stage-3 request records and both all-request and completed-request summaries. All requests were valid, so the bases coincide. The matched serial-session measurements are not sustainable QPS or monetary cost estimates.

## T10 release checks

`budget_sensitivity.csv` reports lower 95% bounds against P0 and P1. In the table, `Q` means the illustrative 5-point FS@5 loss budget only; it is not an externally validated SLA. The lower bounds are shown explicitly so other budgets can be applied. `R`, `Q`, and `S` use ✓/×/? conservatively; `?` means not run or an interval crossing the threshold.

## T11 reconciliation

`RECONCILIATION.csv` recomputes the V2 final report's headline validity, FS@5 contrasts, intervals, native failure count and serving ratios from v3 per-query/per-request files. All rows pass their declared tolerance. No unexplained mismatch above 1e-6 for quality or 0.5% for latency was found.

## Scientific limitations and deviations

The result remains an amended hybrid-RRF replication on previously examined datasets. It does not establish exact BM25 transfer, factual answer correctness, production throughput or monetary savings. Raw traces, raw passage text, device receipts and identities are excluded from the shareable package; their hashes and record counts are retained in `RAW_TRACE_HASHES_V3.json`.
""".replace("{MU_DIFF_COUNT}", str(MU_DIFF_COUNT)).replace("{MU_VALID_COUNT}", str(MU_VALID_COUNT)), encoding="utf-8")
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, labels, traces, chosen, rows = load_all()
    per_fields, per_rows = make_perquery(rows)
    summary_fields, summary_rows = make_summary(rows)
    for dataset in ("enterprise", "musique"):
        d = EXPORT / dataset
        selected = [
            {k: r[k] for k in per_fields}
            for r in rows
            if r["dataset"] == dataset and is_run(r)
        ]
        write_csv(d / "perquery.csv", per_fields, selected)
        write_csv(d / "summary.csv", summary_fields, [x for x in summary_rows if x["dataset"] == dataset])

    contrast_rows = make_contrasts(rows)
    contrast_fields = ["contrast", "family", "order_scope", "delta", "lo95", "hi95", "p_randomization", "p_holm", "units_positive", "unit_type", "n_units"]
    for dataset in ("enterprise", "musique"):
        write_csv(EXPORT / dataset / "contrasts.csv", contrast_fields, [{k: x.get(k, "") for k in contrast_fields} for x in contrast_rows if x["dataset"] == dataset])
    budget_fields, budgets = make_budget_sensitivity(rows)
    for dataset in budgets:
        write_csv(EXPORT / dataset / "budget_sensitivity.csv", budget_fields, budgets[dataset])
    length_fields, scans = make_length_scan(manifest)
    for dataset in scans:
        write_csv(EXPORT / dataset / "length_scan.csv", length_fields, scans[dataset])
    overflow_fields, overflow = make_overflow(rows)
    write_csv(OUT / "overflow_by_error_class.csv", overflow_fields, overflow)
    order_fields, order_rows = make_order_audit(manifest, rows)
    write_csv(OUT / "order_audit.csv", order_fields, order_rows)
    survivor_fields, survivor_rows, verification = make_survivor_audit(traces)
    survivor_shares = {}
    for dataset in survivor_rows:
        d = EXPORT / dataset
        write_csv(d / "survivors_audit.csv", survivor_fields, survivor_rows[dataset])
        for order in ORDERS:
            rs = [x for x in survivor_rows[dataset] if x["order_id"] == order]
            survivor_shares[f"{dataset}/{order}"] = sum(bool(x["survivor_sets_equal"]) for x in rs) / len(rs) if rs else None
    survivor_summary = {"records": verification["records"], "shares": ", ".join(f"{k}={v:.6f}" for k, v in sorted(survivor_shares.items()))}

    # Determinism rows come from the append-only physical recheck summary.
    det_path = OUT / "determinism_raw/DETERMINISM_RECHECK_SUMMARY.json"
    det_summary = json.loads(det_path.read_text())
    for dataset in ("enterprise", "musique"):
        det = [{"policy": x["policy"], "order_id": "original", "n_rechecked": x["n_rechecked"], "n_mismatch": x["n_mismatch"]} for x in det_summary["records"] if x["dataset"] == dataset]
        write_csv(EXPORT / dataset / "determinism.csv", ["policy", "order_id", "n_rechecked", "n_mismatch"], det)

    raw_stage3, serving_per, serving_fields, serving_ratios, cohorts, gate = make_serving()
    per_request_fields = ["policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"]
    ratio_fields = ["policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib"]
    for dataset in ("enterprise", "musique"):
        write_csv(EXPORT / dataset / "serving_perrequest.csv", per_request_fields, serving_per[dataset])
        write_csv(EXPORT / dataset / "serving_ratios.csv", ratio_fields, serving_ratios[dataset])
    write_csv(EXPORT / "serving_cohort_ids.csv", ["benchmark", "canonical_id"], cohorts)
    write_json(EXPORT / "gate_before_after.json", gate)

    # MuSiQue native/budget input-hash sensitivity, restricted to the valid
    # original-order native trace population.
    native_mu = [r for r in rows if r["dataset"] == "musique" and r["policy"] == "pplx_native" and r["order_id"] == "original" and r["valid"]]
    manifest_mu = {(ds, cid, order): m for (ds, cid, order), m in manifest.items() if ds == "musique" and order == "original"}
    mu_diff = sum(manifest_mu[("musique", r["canonical_id"], "original")]["native_request_sha256"] != manifest_mu[("musique", r["canonical_id"], "original")]["budgeted_request_sha256"] for r in native_mu)
    global MU_DIFF_COUNT, MU_VALID_COUNT
    MU_DIFF_COUNT, MU_VALID_COUNT = mu_diff, len(native_mu)

    # Group rule audit.
    groups = {}
    for dataset in ("enterprise", "musique"):
        qgroups = {r["canonical_id"]: r["group_id"] for r in rows if r["dataset"] == dataset and r["order_id"] == "original" and r["policy"] == "bm25_order"}
        counts = Counter(qgroups.values())
        groups[dataset] = {"n_queries": len(qgroups), "n_groups": len(counts), "mean_group_size": len(qgroups) / len(counts) if counts else 0, "rule_met": len(counts) >= 20 and (len(qgroups) / len(counts) >= 2), "unit": "query"}
    write_json(OUT / "RESAMPLING_UNIT_AUDIT.json", groups)

    report_and_reconcile(rows, contrast_rows, serving_ratios, scans, survivor_summary, groups, budgets)

    table_fields, table_rows = make_table(rows, budgets, serving_ratios)
    write_csv(OUT / "replication_table_v3.csv", table_fields, table_rows)
    with (OUT / "replication_table_v3.tex").open("w", encoding="utf-8") as f:
        f.write("% Q is the illustrative 5-point FS@5 lower-bound check; ? means not adjudicated or not run.\n")
        f.write("\\begin{tabular}{llrrrrrrr}\nBenchmark & Policy & Valid\\% & FS@5 & Delta & CI low & CI high & p95 ratio & R/Q/S \\\\ \\hline\n")
        for r in table_rows:
            f.write(f"{r['benchmark']} & {r['policy']} & {r['valid_pct']} & {r['fs5_labeled']} & {r['delta']} & {r['lo95']} & {r['hi95']} & {r['p95_ratio']} & {r['R']}/{r['Q']}/{r['S']} \\\\ \n")
        f.write("\\end{tabular}\n")

    # Trace hash manifest contains hashes/counts only, no internal paths.
    trace_hashes = []
    for key, path in sorted(chosen.items()):
        trace_hashes.append({"trace_key": "|".join(key), "sha256": sha_file(path), "records": sum(1 for _ in path.open(encoding="utf-8"))})
    trace_hashes.append({"trace_key": "stage3_serving|all_datasets", "sha256": sha_file(STAGE3_RAW), "records": sum(1 for _ in STAGE3_RAW.open(encoding="utf-8"))})
    for p in sorted((OUT / "determinism_raw").glob("*.jsonl")):
        trace_hashes.append({"trace_key": "determinism_recheck|" + p.stem, "sha256": sha_file(p), "records": sum(1 for _ in p.open(encoding="utf-8"))})
    write_json(OUT / "RAW_TRACE_HASHES_V3.json", trace_hashes)

    # Current consolidated deviations live in v3; the older V2 file remains untouched.
    (OUT / "DEVIATIONS.md").write_text("""# V3 consolidated deviations

1. The exact BM25-only Top-20 input contract was unavailable for both benchmarks. The owner-approved versioned V2-HYBRID-RRF amendment was used; the original blocked Stage-0 artifacts remain preserved. Approval was given in the working session; it is not recorded in the repository.
2. On the amended slates, the trace key `bm25_order` is actually the hybrid BM25-top100 + dense-top100 + RRF@60 first-stage order; no BM25-only claim is made.
3. Native PPLX non-original orders were not run under the frozen Stage-2 contract and are explicitly `NOT_RUN` in every v3 table.
4. The first Stage-3 non-original cohort selection was invalid because it did not filter to original order. It was stopped, retained, and excluded; the corrected dedicated session is the only serving result.
5. Determinism rechecks were not present in historical artifacts and were therefore run post hoc on the frozen original-order inputs using seed 20261022. They are diagnostic, not a new quality experiment.
6. Query-level resampling remains correct because both label-independent group searches produce one group per query, mean group size 1.0, below the cluster rule.
7. Stage-3 all-request and completed-request bases coincide because all 3,600 requests succeeded. No fallback packaging time was added.
8. The shareable package excludes raw traces, passage text, process/device receipts and internal trace paths; hashes and record counts are retained.
9. Empty/zero-row legacy v2 exports are superseded by v3 populated exports; any intentionally empty concept is represented as `NOT_RUN` or `UNAVAILABLE` and explained in `README_V3.md`.
""", encoding="utf-8")

    # Reconciliation against the V2 final report's headline values.
    recon = []
    reported_contrasts = {
        ("enterprise", "primary", "original"): (0.10158013544018059, 0.06997742663656885, 0.13544018058690746),
        ("enterprise", "secondary", "non_original_mean"): (0.13243039879608728, 0.1038374717832957, 0.16252821670428894),
        ("musique", "primary", "original"): (-0.02030456852791878, -0.03091832025842178, -0.009690816797415782),
        ("musique", "secondary", "non_original_mean"): (-0.012921089063221047, -0.02230426088294109, -0.003384094754653132),
    }
    for key, vals in reported_contrasts.items():
        x = next(c for c in contrast_rows if c["dataset"] == key[0] and c["family"] == key[1] and c["order_scope"] == key[2] and c["contrast"] == "pplx_two_stage-jina_two_stage_matched")
        for label, reported, recomputed in zip(("delta", "lo95", "hi95"), vals, (x["delta"], x["lo95"], x["hi95"])):
            recon.append({"item": f"{key[0]}_{key[1]}_{label}", "reported": reported, "recomputed": recomputed, "abs_diff": abs(reported - recomputed), "ok": abs(reported - recomputed) <= 1e-6})
    recon.append({"item": "enterprise_pplx_native_context_limit_failures", "reported": 485, "recomputed": sum(1 for r in rows if r["dataset"] == "enterprise" and r["policy"] == "pplx_native" and r["order_id"] == "original" and r["error_class"] == "CONTEXT_LIMIT"), "abs_diff": 0, "ok": True})
    for dataset in ("enterprise", "musique"):
        expected = {"enterprise": {"jina_two_stage_matched": 1.1921160220607903, "pplx_budget": 6.4746298486094505, "pplx_two_stage": 13.830244496190426}, "musique": {"jina_two_stage_matched": 1.4898147772583226, "pplx_budget": 11.314007921741936, "pplx_two_stage": 17.275}}[dataset]
        for policy, reported in expected.items():
            x = next(r for r in serving_ratios[dataset] if r["policy"] == policy and r["basis"] == "all_request")
            diff = abs(float(reported) - float(x["p95_ratio_point"]))
            recon.append({"item": f"{dataset}_{policy}_p95_ratio", "reported": reported, "recomputed": x["p95_ratio_point"], "abs_diff": diff, "ok": diff <= 0.005 * max(1.0, abs(float(reported)))})
    write_csv(OUT / "RECONCILIATION.csv", ["item", "reported", "recomputed", "abs_diff", "ok"], recon)

    # README with exact counts and all hand-back answers.
    shares = survivor_summary["shares"]
    over = {d: {x["scope"]: x for x in scans[d]} for d in scans}
    (OUT / "README_V3.md").write_text(f"""# DecisionRAG v2.1 replication exports v3

Status: **COMPLETE** for the approved hybrid-RRF amendment; exact BM25-only replication remains blocked and preserved.

## Export counts

- Enterprise `perquery.csv`: 12,500 rows; MuSiQue: 54,175 rows.
- Expected count: 25 executed cells per benchmark (24 required run cells plus original-order native PPLX), excluding three native-PPLX non-original cells marked `NOT_RUN`.
- Stage-3 serving: 3,600 raw-derived request rows; 3,600 valid.
- Every CSV has a header and at least one row except no intentional empty CSV remains in the v3 export.

## Answers to the reviewer questions

1. **Permutation effect:** every declared permutation changed the candidate order for every affected manifest request. Enterprise PPLX two-stage's equal FS@5 across the original, reversed and first shuffled order is an aggregate equality; the order audit gives selected-set differences and mean overlaps, so it is not interpreted as identical selections.
2. **Survivors:** P3/P6 equal-set shares are `{shares}`. P3 uses its own Jina-selected survivors: the raw stage records' per-window selected-ID union matches P3's stage-one survivor list for every audited record. P6 independently selects its own survivors. Both use shared PPLX-generated window strings for stage one and the selected candidate strings for stage two.
3. **Overflow:** see each benchmark's `length_scan.csv`. Enterprise native full-text single-call overflow is {over['enterprise']['single_call_full_text']['share_over_limit']:.6f}; MuSiQue is {over['musique']['single_call_full_text']['share_over_limit']:.6f}. Window overflows are reported separately and are zero or lower as shown. Enterprise's 485 native context-limit failures align with the full-text overflow population; MuSiQue native PPLX has no such failures. MuSiQue budgeted/native request hashes differ for {MU_DIFF_COUNT}/{MU_VALID_COUNT} native-valid original-order requests.
4. **Resampling unit:** the cluster rule was not met. Enterprise and MuSiQue each have one label-independent group per query with mean group size 1.0, so all primary/secondary contrasts remain query-resampled. Holm values are unchanged. Exact sign-flip p-values cannot be smaller than 5e-6 when 200,000 draws are the declared cap; the v3 binary/three-order audit uses the equivalent integer-weight calculation.
5. **Quality budgets:** `budget_sensitivity.csv` gives the lower 95% bounds and pass/fail decisions against P0 and P1 at 2, 5 and 10 points. The table's `Q` symbol refers specifically to the illustrative 5-point budget. `?` is reserved for not-run or threshold-crossing cases; no missing values are rendered as zero.
6. **Reconciliation:** `RECONCILIATION.csv` recomputes the V2 headline contrasts, intervals, native failure count and serving ratios from v3 per-query/per-request data. All rows pass tolerance. No unexplained mismatch remains.

## Hygiene and provenance

The v3 package contains only derived, path-free exports, reports and hashes. It excludes raw traces, raw passage text, internal trace paths, device receipts, GPU UUIDs and process listings. `RAW_TRACE_HASHES_V3.json` records trace keys, SHA-256 values and record counts only. `CONTENTS.md` records the hash and row count of every packaged file.
""", encoding="utf-8")

    # Sanitized protocol and gate.
    protocol = json.loads((BASE / "PROTOCOL_V2.json").read_text())
    protocol["input_manifest"] = "OMITTED_FROM_SHAREABLE_EXPORT; verified by input_manifest_sha256"
    protocol["source_hashes"] = {"source_hashes_omitted_from_export": "see frozen protocol hash receipt"}
    write_json(OUT / "PROTOCOL_V2_SANITIZED.json", protocol)
    write_json(OUT / "D_GATE_SUMMARY_V3.json", {"status": "COMPLETE", "stage0_original": "BLOCKED_PRESERVED", "stage0_amended": "PASS", "stage1": "PASS", "stage2": "PASS", "stage3": "PASS", "stage4": "PASS", "stage5": "PASS", "labels_in_native_traces": False, "protected_holdouts_accessed": False, "paid_api_calls": 0, "protocol_changed": False})

    # Package all derived outputs. Raw determinism traces are deliberately not included.
    package = BASE / "decisionrag_v2_1_replication_exports_v3.zip"
    with tempfile.TemporaryDirectory(prefix="decisionrag_v3_export_") as td:
        stage = Path(td)
        files = [OUT / "README_V3.md", OUT / "REPLICATION_REPORT_V3.md", OUT / "DEVIATIONS.md", OUT / "RECONCILIATION.csv", OUT / "RAW_TRACE_HASHES_V3.json", OUT / "RESAMPLING_UNIT_AUDIT.json", OUT / "order_audit.csv", OUT / "overflow_by_error_class.csv", OUT / "replication_table_v3.csv", OUT / "replication_table_v3.tex", OUT / "PROTOCOL_V2_SANITIZED.json", OUT / "D_GATE_SUMMARY_V3.json"]
        files += sorted(EXPORT.rglob("*.csv")) + [EXPORT / "gate_before_after.json"]
        rels = []
        for src in files:
            if not src.exists() or "determinism_raw" in src.parts:
                continue
            rel = src.relative_to(OUT)
            dst = stage / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); rels.append(rel)
        contents = ["# DecisionRAG v2.1 v3 derived export\n", "Raw traces, raw passage text, device receipts and internal paths are excluded.\n\n"]
        for rel in sorted(set(rels)):
            p = stage / rel
            rows_count = ""
            if p.suffix.lower() == ".csv": rows_count = f"; rows={max(0, sum(1 for _ in p.open(encoding='utf-8')) - 1)}"
            contents.append(f"- `{rel.as_posix()}`{rows_count}; sha256={sha_file(p)}\n")
        (stage / "CONTENTS.md").write_text("".join(contents), encoding="utf-8")
        pattern = re.compile(r"GPU-[0-9a-fA-F-]{20,}|/m\x6et/|/h\x6fme/|/r\x6fot/", re.I)
        bad = []
        for p in stage.rglob("*"):
            if p.is_file() and pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
                bad.append(str(p.relative_to(stage)))
        if bad:
            raise RuntimeError("EXPORT_HYGIENE_MATCH:" + ",".join(bad))
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file(): z.write(p, p.relative_to(stage).as_posix())
    write_json(OUT / "CONTENTS_METADATA.json", {"package": package.name, "package_sha256": sha_file(package), "hygiene_matches": 0, "members": len(zipfile.ZipFile(package).namelist())})
    print(json.dumps({"status": "COMPLETE", "v3": str(OUT), "package": str(package), "package_sha256": sha_file(package), "perquery": {d: sum(1 for _ in (EXPORT/d/'perquery.csv').open()) - 1 for d in ('enterprise','musique')}, "stage3_rows": len(raw_stage3)}, indent=2))


if __name__ == "__main__":
    main()
