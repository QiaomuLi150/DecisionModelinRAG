#!/usr/bin/env python3
"""Offline C1 analysis for the GPU3 continuation.

All native traces are required to be complete and label-blind before this
script opens the frozen BRIGHT gold manifest.  Serial and batched traces are
kept distinguishable; batched elapsed time is never presented as isolated
request latency.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_5/c1_two_stage_specialist_gpu3"
RAW = OUT / "raw"
EXPORT = OUT / "export"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
GOLD = ROOT / "results/non_jev/v2/e9_bright_v1_0/00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json"
ORDERS = ["original", "reversed", "perm_seed_1", "perm_seed_2", "perm_seed_3", "perm_seed_4", "perm_seed_5"]
NEW_POLICIES = ["jina_two_stage_full", "jina_two_stage_pplxtext", "jina_pplx_budget"]
SEED = 731
DRAWS = 40_000


def fsha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def ndcg5(selected: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    dcg = sum((1.0 if x in gold else 0.0) / math.log2(i + 2) for i, x in enumerate(selected[:5]))
    ideal_n = min(5, len(gold))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / ideal if ideal else 0.0


def valid(row: dict[str, Any]) -> bool:
    ids = [str(x) for x in row.get("selected_ids", [])]
    return row.get("status") in {"PASS", "ok"} and len(ids) == 5 and len(set(ids)) == 5


def bootstrap(groups: dict[str, list[float]]) -> dict[str, Any]:
    domains = sorted(groups)
    if not domains:
        return {"n": 0, "n_domains": 0, "query_mean": None, "domain_mean": None}
    vals = np.concatenate([np.asarray(groups[d], dtype=float) for d in domains])
    dm = np.asarray([np.mean(groups[d]) for d in domains], dtype=float)
    rng = np.random.Generator(np.random.PCG64(SEED))
    idx = rng.integers(0, len(domains), size=(DRAWS, len(domains)))
    draws = dm[idx].mean(axis=1)
    return {
        "n": int(vals.size), "n_domains": len(domains),
        "query_mean": float(vals.mean()), "domain_mean": float(dm.mean()),
        "domain_bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "seed": SEED, "draws": DRAWS,
        "domains": {d: {"n": len(groups[d]), "mean": float(np.mean(groups[d]))} for d in domains},
    }


def paired(groups: dict[str, list[float]]) -> dict[str, Any]:
    out = bootstrap(groups)
    if not groups:
        out["sign_flip_p_two_sided"] = None
        return out
    dm = np.asarray([np.mean(groups[d]) for d in sorted(groups)], dtype=float)
    if len(dm) <= 20:
        signs = np.asarray([np.mean(dm * np.asarray(s)) for s in itertools.product((-1.0, 1.0), repeat=len(dm))])
        out["sign_flip_p_two_sided"] = float(np.mean(np.abs(signs) >= abs(float(dm.mean())) - 1e-15))
    else:
        out["sign_flip_p_two_sided"] = None
    out["observed_domain_mean"] = float(dm.mean())
    return out


def read_map(paths: list[Path], expected: int, label: str) -> dict[str, dict[str, Any]]:
    rows = [r for p in paths for r in load_jsonl(p)]
    m = {str(r["canonical_id"]): r for r in rows if r.get("canonical_id") is not None}
    if len(rows) != expected or len(m) != expected:
        raise RuntimeError(f"{label}: expected {expected} rows/IDs, got {len(rows)}/{len(m)}")
    return m


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    slates = load_jsonl(BRIGHT)
    slate = {f"{x['domain']}:{x['query_id']}": x for x in slates}
    if len(slate) != 1384 or fsha(BRIGHT) != "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2":
        raise RuntimeError("BRIGHT Top20 manifest integrity failure")
    gm = json.loads(GOLD.read_text(encoding="utf-8"))
    gold = {f"{x['domain']}:{x['id']}": set(map(str, x.get("gold_ids", []))) for x in gm["items"]}
    if set(gold) != set(slate):
        raise RuntimeError("gold/slate population mismatch")

    # Validate every new raw shard before any label access is used for scoring.
    raw_rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    raw_validation = []
    for policy in NEW_POLICIES:
        for order in ORDERS:
            path = RAW / f"{policy}__{order}.jsonl"
            meta = path.with_suffix(".meta.json")
            if not path.exists() or not meta.exists():
                raise RuntimeError(f"missing complete shard: {path}")
            rows = load_jsonl(path)
            if len(rows) != 1384:
                raise RuntimeError(f"{path.name}: {len(rows)} rows")
            mmap = {}
            for r in rows:
                cid = str(r["canonical_id"])
                if cid in mmap or cid not in slate:
                    raise RuntimeError(f"duplicate/missing C1 ID {cid}")
                if r.get("labels_joined") is not False:
                    raise RuntimeError(f"label marker not false in {path.name}/{cid}")
                if r.get("policy") != policy or r.get("order_id") != order:
                    raise RuntimeError(f"policy/order mismatch in {path.name}/{cid}")
                candidates = {str(x["candidate_id"]) for x in slate[cid]["candidates"]}
                got = [str(x) for x in r.get("candidate_ids", [])]
                # A deterministic pre-selector failure may legitimately have
                # no candidate view (for example an unbudgetable frozen E11
                # query).  It remains an explicit failure/zero, not a
                # candidate-filtered success.  Successful rows must carry the
                # complete canonical Top-20 view.
                if valid(r) and (len(got) != 20 or set(got) != candidates):
                    raise RuntimeError(f"candidate ID mismatch in {path.name}/{cid}")
                if r.get("slate_sha256") != slate[cid]["slate_sha256"]:
                    raise RuntimeError(f"slate hash mismatch in {path.name}/{cid}")
                mmap[cid] = r
            if set(mmap) != set(slate):
                raise RuntimeError(f"coverage mismatch in {path.name}")
            raw_rows[(policy, order)] = mmap
            raw_validation.append({
                "path": str(path.relative_to(ROOT)), "sha256": fsha(path), "records": len(rows),
                "valid": sum(valid(r) for r in rows), "failed": sum(not valid(r) for r in rows),
                "batch_sizes": sorted({r.get("batch_size", 1) for r in rows}),
                "timing_valid_rows": sum(r.get("timing_valid") is True or (r.get("batch_size") in (None, 1) and r.get("elapsed_ms") is not None) for r in rows),
            })

    # Authenticated historical anchors, also label-blind at inference time.
    pplx_orig = read_map(sorted((ROOT / "results/v1_3/r3_multipass").glob("multipass_raw_start*_n*.jsonl")), 1384, "PPLX original")
    pplx_shuf_paths = sorted((ROOT / "results/v1_3/r3_multipass_ordered/raw").glob("*.jsonl"))
    pplx_shuf_rows = [r for p in pplx_shuf_paths for r in load_jsonl(p)]
    pplx_by_order: dict[str, dict[str, dict[str, Any]]] = {}
    for r in pplx_shuf_rows:
        order = str(r.get("order_id"))
        pplx_by_order.setdefault(order, {})[str(r["canonical_id"])] = r
    if set(pplx_by_order) != set(ORDERS[1:]) or any(len(x) != 1384 for x in pplx_by_order.values()):
        raise RuntimeError("PPLX shuffled anchor coverage failure")
    jina_full_rows = load_jsonl(ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl")
    jina_full = {f"{r['domain']}:{r['query_id']}": r for r in jina_full_rows}
    if len(jina_full) != 1384:
        raise RuntimeError("Jina full anchor coverage failure")

    def score_row(r: dict[str, Any]) -> float:
        return ndcg5([str(x) for x in r.get("selected_ids", [])], gold[str(r["canonical_id"])] if r.get("canonical_id") else set()) if valid(r) else 0.0

    def score_cid(cid: str, r: dict[str, Any]) -> float:
        return ndcg5([str(x) for x in r.get("selected_ids", [])], gold[cid]) if valid(r) else 0.0

    # Unified per-query table.
    perquery = []
    for (policy, order), mmap in sorted(raw_rows.items()):
        for cid, r in sorted(mmap.items()):
            perquery.append({"canonical_id": cid, "domain": r["domain"], "policy": policy, "order_id": order, "valid": str(valid(r)).lower(), "metric": score_cid(cid, r), "n_calls": r.get("n_calls"), "elapsed_ms": r.get("elapsed_ms"), "batch_elapsed_ms": r.get("batch_elapsed_ms"), "batch_size": r.get("batch_size", 1), "timing_valid": str(r.get("timing_valid", r.get("batch_size") in (None, 1))).lower()})
    for cid, r in sorted(pplx_orig.items()):
        perquery.append({"canonical_id": cid, "domain": r["domain"], "policy": "pplx_two_stage", "order_id": "original", "valid": str(valid(r)).lower(), "metric": score_cid(cid, r), "n_calls": r.get("n_calls"), "elapsed_ms": r.get("total_ms"), "batch_elapsed_ms": None, "batch_size": 1, "timing_valid": "historical"})
    for order, mmap in sorted(pplx_by_order.items()):
        for cid, r in sorted(mmap.items()):
            r = dict(r); r["canonical_id"] = cid
            perquery.append({"canonical_id": cid, "domain": r["domain"], "policy": "pplx_two_stage", "order_id": order, "valid": str(valid(r)).lower(), "metric": score_cid(cid, r), "n_calls": r.get("n_calls"), "elapsed_ms": r.get("total_ms"), "batch_elapsed_ms": None, "batch_size": 1, "timing_valid": "historical"})
    ppath = EXPORT / "c1_perquery.csv"
    with ppath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perquery[0])); w.writeheader(); w.writerows(perquery)

    # Policy/order summary, with timing separated by measurement validity.
    summary = []
    all_views: dict[tuple[str, str], dict[str, float]] = {}
    for policy in NEW_POLICIES:
        for order in ORDERS:
            mmap = raw_rows[(policy, order)]
            vals = {cid: score_cid(cid, r) for cid, r in mmap.items()}
            all_views[(policy, order)] = vals
            serial_ms = [float(r["elapsed_ms"]) for r in mmap.values() if r.get("elapsed_ms") is not None and r.get("timing_valid") is not False]
            batch_ms = [float(r["batch_elapsed_ms"]) for r in mmap.values() if r.get("batch_elapsed_ms") is not None]
            summary.append({"policy": policy, "order_id": order, "n": 1384, "valid": sum(valid(r) for r in mmap.values()), "failed": sum(not valid(r) for r in mmap.values()), "all_attempt_mean_ndcg5": float(np.mean(list(vals.values()))), "serial_p50_ms": float(np.quantile(serial_ms, .5)) if serial_ms else None, "serial_p95_ms": float(np.quantile(serial_ms, .95)) if serial_ms else None, "batch_elapsed_p50_ms": float(np.quantile(batch_ms, .5)) if batch_ms else None, "batch_elapsed_p95_ms": float(np.quantile(batch_ms, .95)) if batch_ms else None, "timing_population": "serial_rows_only" if serial_ms else "NOT_ISOLATED_BATCH_ROWS"})
    for order, mmap in [("original", pplx_orig), *sorted(pplx_by_order.items())]:
        vals = {cid: score_cid(cid, r) for cid, r in mmap.items()}
        summary.append({"policy": "pplx_two_stage", "order_id": order, "n": 1384, "valid": sum(valid(r) for r in mmap.values()), "failed": sum(not valid(r) for r in mmap.values()), "all_attempt_mean_ndcg5": float(np.mean(list(vals.values())),), "serial_p50_ms": None, "serial_p95_ms": None, "batch_elapsed_p50_ms": None, "batch_elapsed_p95_ms": None, "timing_population": "historical_R3"})
    with (EXPORT / "c1_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # Per-domain scorecard for all raw policy/order views.
    domain_rows = []
    for x in summary:
        policy, order = x["policy"], x["order_id"]
        mmap = raw_rows[(policy, order)] if policy in NEW_POLICIES else (pplx_orig if order == "original" else pplx_by_order[order])
        by = defaultdict(list)
        for cid, r in mmap.items(): by[r["domain"]].append(score_cid(cid, r))
        for d, vals in sorted(by.items()): domain_rows.append({"policy": policy, "order_id": order, "domain": d, "n": len(vals), "valid_or_anchor": sum(vals_i > -1 for vals_i in vals), "mean_ndcg5": float(np.mean(vals))})
    with (EXPORT / "c1_domain_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(domain_rows[0])); w.writeheader(); w.writerows(domain_rows)

    # Primary contrasts. Values are paired by canonical query ID.
    contrasts = []
    def add_contrast(name: str, left: dict[str, float], right: dict[str, float], left_label: str, right_label: str):
        deltas = {cid: left[cid] - right[cid] for cid in sorted(left)}
        groups = defaultdict(list)
        for cid, d in deltas.items(): groups[slate[cid]["domain"]].append(d)
        st = paired(groups)
        contrasts.append({"contrast": name, "left": left_label, "right": right_label, "n": len(deltas), "query_weighted_effect": float(np.mean(list(deltas.values()))), "domain_weighted_effect": float(np.mean([np.mean(v) for v in groups.values()])), "ci95_low": st.get("domain_bootstrap_ci95", [None, None])[0], "ci95_high": st.get("domain_bootstrap_ci95", [None, None])[1], "p_two_sided_sign_flip": st.get("sign_flip_p_two_sided"), "bootstrap_seed": SEED, "bootstrap_draws": DRAWS, "domain_n": len(groups), "domain_effects": st.get("domains")})

    pplx_orig_scores = {cid: score_cid(cid, r) for cid, r in pplx_orig.items()}
    jina_full_scores = {cid: score_cid(cid, r) for cid, r in jina_full.items()}
    add_contrast("pplx_two_stage - jina_two_stage_full (original)", pplx_orig_scores, all_views[("jina_two_stage_full", "original")], "pplx_two_stage/original", "jina_two_stage_full/original")
    pmean = {cid: float(np.mean([score_cid(cid, pplx_by_order[o][cid]) for o in ORDERS[1:]])) for cid in sorted(slate)}
    jmean = {cid: float(np.mean([all_views[("jina_two_stage_full", o)][cid] for o in ORDERS[1:]])) for cid in sorted(slate)}
    add_contrast("pplx_two_stage - jina_two_stage_full (mean six shuffled)", pmean, jmean, "pplx_two_stage/shuffled_mean", "jina_two_stage_full/shuffled_mean")
    add_contrast("pplx_two_stage - jina_two_stage_pplxtext (original)", pplx_orig_scores, all_views[("jina_two_stage_pplxtext", "original")], "pplx_two_stage/original", "jina_two_stage_pplxtext/original")
    add_contrast("jina_two_stage_full - jina_full (original)", all_views[("jina_two_stage_full", "original")], jina_full_scores, "jina_two_stage_full/original", "jina_full/original")
    raw_p = [x["p_two_sided_sign_flip"] for x in contrasts]
    order_idx = sorted(range(len(raw_p)), key=lambda i: (float("inf") if raw_p[i] is None else raw_p[i]))
    holm = [None] * len(raw_p)
    running = 0.0
    for rank, i in enumerate(order_idx):
        p = raw_p[i]
        if p is None: continue
        running = max(running, min(1.0, p * (len(raw_p) - rank)))
        holm[i] = running
    for i, x in enumerate(contrasts): x["holm_adjusted_p_two_sided"] = holm[i]
    (EXPORT / "c1_contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n", encoding="utf-8")
    with (EXPORT / "c1_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [k for k, v in contrasts[0].items() if k != "domain_effects"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: v for k, v in x.items() if k in fields} for x in contrasts])

    # Order stability, descriptive only.
    stability = []
    for policy in ["pplx_two_stage"] + NEW_POLICIES:
        views = {"original": pplx_orig_scores if policy == "pplx_two_stage" else all_views[(policy, "original")]}
        for o in ORDERS[1:]: views[o] = ({cid: score_cid(cid, pplx_by_order[o][cid]) for cid in slate} if policy == "pplx_two_stage" else all_views[(policy, o)])
        # Compare selected sets rather than scores where available.
        for cid in sorted(slate):
            rows = []
            for o in ORDERS:
                if policy == "pplx_two_stage": r = pplx_orig[cid] if o == "original" else pplx_by_order[o][cid]
                else: r = raw_rows[(policy, o)][cid]
                rows.append(r)
            sets = [tuple(sorted(map(str, r.get("selected_ids", [])))) for r in rows]
            stability.append({"policy": policy, "canonical_id": cid, "domain": slate[cid]["domain"], "valid_order_count": sum(valid(r) for r in rows), "unique_selected_sets": len(set(sets)), "mean_ndcg5": float(np.mean([score_cid(cid, r) for r in rows]))})
    with (EXPORT / "c1_order_stability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stability[0])); w.writeheader(); w.writerows(stability)

    receipt_paths = sorted(OUT.glob("EXECUTION_RECEIPT*.json"))
    provenance = {
        "status": "COMPLETE_OFFLINE_ANALYSIS",
        "labels_joined_after_raw_freeze": True,
        "bright_path": str(BRIGHT.relative_to(ROOT)), "bright_sha256": fsha(BRIGHT),
        "gold_path": str(GOLD.relative_to(ROOT)), "gold_sha256": fsha(GOLD),
        "raw_validation": raw_validation,
        "receipt_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": fsha(p)} for p in receipt_paths],
        "historical_anchor_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": fsha(p), "records": len(load_jsonl(p))} for p in sorted((ROOT / "results/v1_3/r3_multipass").glob("multipass_raw_start*_n*.jsonl"))],
        "pplx_shuffled_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": fsha(p), "records": len(load_jsonl(p))} for p in pplx_shuf_paths],
        "jina_full_sha256": fsha(ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl"),
        "protocol_sha256": fsha(OUT / "PROTOCOL_V2.json"), "manifest_sha256": fsha(OUT / "EXECUTION_MANIFEST_V2.json"),
        "seed": SEED, "bootstrap_draws": DRAWS, "protected_holdout_accessed": False, "paid_api_calls": 0,
    }
    (EXPORT / "C1_GOLD_JOIN_RECEIPT.json").write_text(json.dumps({"status": "JOINED_AFTER_RAW_FREEZE", "records": len(perquery), "gold_records": len(gold), "labels_in_native_inputs": False, "gold_manifest_sha256": fsha(GOLD)}, indent=2) + "\n", encoding="utf-8")
    (EXPORT / "C1_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    report = [
        "# V1.5 C1 GPU3 Offline Analysis", "",
        "Status: COMPLETE_WITH_SERIAL_AND_BATCHED_QUALITY_POPULATIONS", "",
        "The frozen BRIGHT Top-20 manifest was validated before the gold manifest was opened. All 21 new Jina policy/order shards contain 1,384 records and preserve 20 canonical candidate IDs per query. Batched traces are valid for selection-quality analysis but their elapsed time is not isolated-request latency.", "",
        "## Primary contrasts", "",
        "| contrast | n | query effect | domain effect | 95% domain-bootstrap CI | raw sign-flip p | Holm p |", "|---|---:|---:|---:|---|---:|---:|",
    ]
    for x in contrasts:
        report.append(f"| {x['contrast']} | {x['n']} | {x['query_weighted_effect']:.6f} | {x['domain_weighted_effect']:.6f} | [{x['ci95_low']:.6f}, {x['ci95_high']:.6f}] | {x['p_two_sided_sign_flip']:.6f} | {x['holm_adjusted_p_two_sided']:.6f} |")
    report += ["", "## Measurement boundary", "", "Serial rows report complete selector elapsed milliseconds. Batched rows report batch elapsed milliseconds and `timing_valid=false`; they are not used as isolated p50/p95 latency claims. GPU3 was released after inference.", "", "## Artifacts", "", "- `export/c1_perquery.csv`", "- `export/c1_summary.csv`", "- `export/c1_domain_summary.csv`", "- `export/c1_contrasts.csv` and `c1_contrasts.json`", "- `export/c1_order_stability.csv`", "- `export/C1_GOLD_JOIN_RECEIPT.json`", "- `export/C1_PROVENANCE_MANIFEST.json`", ""]
    (OUT / "C1_OFFLINE_ANALYSIS_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "new_raw_shards": len(raw_validation), "perquery_rows": len(perquery), "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
