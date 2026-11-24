#!/usr/bin/env python3
"""Offline evaluator for the V2 hybrid-RRF controlled replication.

This script never loads a model and never reads labels while native traces are
being produced. It consumes only completed raw JSONL traces and the frozen
input manifest, then joins benchmark support annotations offline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "v2_1_replication"
MANIFEST = BASE / "stage0_preflight_v2_hybrid" / "INPUT_MANIFEST_V2_HYBRID.jsonl"
OUT = BASE / "stage4_analysis_v2_hybrid"
ORDERS = ["original", "reversed", "perm_seed_20261020", "perm_seed_20261021"]
POLICIES = ["bm25_order", "jina_full", "jina_budget", "jina_two_stage_matched", "pplx_native", "pplx_budget", "pplx_two_stage"]
MODEL_POLICIES = [p for p in POLICIES if p != "bm25_order"]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_manifest():
    rows = load_jsonl(MANIFEST)
    return {(x["dataset"], x["canonical_id"], x["order_id"]): x for x in rows}


def load_labels(manifest):
    labels = {}
    ent_path = ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"
    ent = {x["question_id"]: x for x in load_jsonl(ent_path)}
    for (dataset, cid, order), row in manifest.items():
        if dataset == "enterprise":
            x = ent.get(cid, {})
            gold = set(x.get("expected_internal_doc_ids", [])) if x.get("cssr_denominator_status") == "APPLICABLE" else set()
            labels[(dataset, cid)] = {
                "labeled": bool(gold),
                "gold": gold,
                "status": x.get("cssr_denominator_status", "MISSING"),
                "group_id": row.get("group_id", ""),
            }
    mu_path = ROOT / "data/raw/vendor/musique/musique_ans_v1.0_dev.jsonl"
    mu = {x["id"]: x for x in load_jsonl(mu_path)}
    corpus = {x["doc_id"]: x for x in load_jsonl(ROOT / "data/corpus/passages.jsonl")}
    source_to_docs = defaultdict(set)
    for did, doc in corpus.items():
        for source_id in doc.get("original_source_ids", []):
            source_to_docs[source_id].add(did)
    for (dataset, cid, order), row in manifest.items():
        if dataset != "musique":
            continue
        rid = cid.split(":", 1)[1] if ":" in cid else cid
        rec = mu.get(rid, {})
        source_tokens = {
            f"musique:{rid}:context:{p.get('idx')}"
            for p in rec.get("paragraphs", []) if p.get("is_supporting")
        }
        gold = set()
        for did in row["candidate_ids"]:
            doc = corpus.get(did, {})
            if source_tokens.intersection(set(doc.get("original_source_ids", []))):
                gold.add(did)
        # Keep the benchmark-native support set, including supports not in Top-20.
        # A candidate is reachable only when all support IDs are in the frozen slate.
        all_support = set().union(*(source_to_docs.get(token, set()) for token in source_tokens)) if source_tokens else set()
        labels[(dataset, cid)] = {
            "labeled": bool(all_support),
            "gold": all_support,
            "status": "APPLICABLE" if all_support else "MISSING",
            "group_id": row.get("group_id", ""),
        }
    return labels


def choose_trace_files():
    by_key = defaultdict(list)
    for directory in [BASE / "stage1_native_v2_hybrid", BASE / "stage2_orders_v2_hybrid"]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            if path.name.endswith(".part"):
                continue
            try:
                records = load_jsonl(path)
                first = records[0]
                # Preserve smoke traces as evidence, but never treat them as
                # full evaluation runs. Valid combined or dataset-split runs
                # contain exactly 2,667, 500, or 2,167 rows.
                if len(records) not in {500, 2167, 2667}:
                    continue
                key = (first.get("policy"), first.get("order_id"))
                by_key[key].append(path)
            except (IndexError, json.JSONDecodeError):
                continue
    chosen = {}
    for key, paths in by_key.items():
        # Prefer a declared full run over the preserved 2-row smoke trace.
        paths.sort(key=lambda p: ("__full" not in p.stem, -p.stat().st_size, str(p)))
        chosen[key] = paths[0]
    return chosen


def load_traces():
    traces = {}
    chosen = choose_trace_files()
    for key, path in chosen.items():
        for rec in load_jsonl(path):
            k = (rec.get("dataset"), rec.get("canonical_id"), rec.get("order_id"), rec.get("policy"))
            # The largest completed trace is selected above; do not silently overwrite duplicates.
            traces[k] = rec
    return traces, chosen


def fs5(selected, gold):
    return 1 if gold and gold.issubset(set(selected[:5])) else 0


def expected_rows(manifest, labels, traces):
    rows = []
    for key, m in sorted(manifest.items()):
        dataset, cid, order = key
        lab = labels.get((dataset, cid), {"labeled": False, "gold": set(), "status": "MISSING", "group_id": m.get("group_id", "")})
        for policy in POLICIES:
            if policy == "bm25_order":
                selected = m["candidate_ids"][:5]
                valid = True
                err = ""
                n_calls = 0
                input_tokens = ""
            else:
                rec = traces.get((dataset, cid, order, policy))
                if rec is None:
                    selected = []
                    valid = False
                    err = "NOT_RUN"
                    n_calls = ""
                    input_tokens = ""
                else:
                    selected = rec.get("selected_ids", []) or []
                    valid = rec.get("status") == "PASS" and len(selected) == 5 and len(set(selected)) == 5 and set(selected).issubset(set(m["candidate_ids"]))
                    err = "" if valid else (rec.get("error_class") or "INVALID_OUTPUT")
                    n_calls = rec.get("n_calls", "")
                    input_tokens = rec.get("input_tokens", "")
            reachable = bool(lab["labeled"] and lab["gold"].issubset(set(m["candidate_ids"])))
            score = fs5(selected, lab["gold"]) if lab["labeled"] and valid else (0 if lab["labeled"] and not valid else "")
            rows.append({
                "dataset": dataset, "canonical_id": cid, "group_id": lab["group_id"], "order_id": order,
                "policy": policy, "valid": int(valid), "error_class": err, "labeled": int(lab["labeled"]),
                "reachable": int(reachable), "fs5": score, "n_calls": n_calls, "input_tokens": input_tokens,
                "selected_ids": selected, "candidate_ids": m["candidate_ids"],
            })
    return rows


def percentile(values, q):
    if not values:
        return None
    a = sorted(values)
    if len(a) == 1:
        return float(a[0])
    x = (len(a) - 1) * q
    lo, hi = int(math.floor(x)), int(math.ceil(x))
    return float(a[lo] + (a[hi] - a[lo]) * (x - lo))


def bootstrap_delta(a, b, draws=40000, seed=731):
    import numpy as np
    if not a:
        return {"delta": None, "lo95": None, "hi95": None, "p_randomization": None}
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = av - bv
    rng = np.random.default_rng(seed)
    vals = []
    batch = 1000
    for start in range(0, draws, batch):
        n = min(batch, draws - start)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        vals.extend(np.asarray(d[idx].mean(axis=1)).tolist())
    obs = float(d.mean())
    lo, hi = np.quantile(vals, [0.025, 0.975])
    if len(d) <= 20:
        count = 0
        total = 1 << len(d)
        for mask in range(total):
            signs = np.array([1 if mask & (1 << i) else -1 for i in range(len(d))])
            if abs(float((d * signs).mean())) >= abs(obs) - 1e-15:
                count += 1
        p = count / total
    else:
        exceed = 0
        total = 200000
        for start in range(0, total, batch):
            n = min(batch, total - start)
            signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(d)))
            null = (signs * d).mean(axis=1)
            exceed += int(np.sum(np.abs(null) >= abs(obs)))
        p = float((1 + exceed) / (total + 1))
    return {"delta": obs, "lo95": float(lo), "hi95": float(hi), "p_randomization": float(p)}


def summarize(rows):
    out = []
    for dataset in ("enterprise", "musique"):
        n_requests = len({r["canonical_id"] for r in rows if r["dataset"] == dataset and r["order_id"] == "original"})
        for policy in POLICIES:
            for order in ORDERS:
                rs = [r for r in rows if r["dataset"] == dataset and r["policy"] == policy and r["order_id"] == order]
                labeled = [r for r in rs if r["labeled"]]
                reachable = [r for r in labeled if r["reachable"]]
                vals = [float(r["fs5"]) for r in labeled if r["fs5"] != ""]
                rvals = [float(r["fs5"]) for r in reachable if r["fs5"] != ""]
                out.append({
                    "dataset": dataset, "policy": policy, "order_id": order,
                    "n_requests": n_requests, "n_valid": sum(r["valid"] for r in rs),
                    "valid_pct": round(100 * sum(r["valid"] for r in rs) / n_requests, 6) if n_requests else "",
                    "n_labeled": len(labeled), "n_reachable": len(reachable),
                    "mean_fs5_all_labeled": round(sum(vals) / len(vals), 8) if vals else "",
                    "mean_fs5_reachable": round(sum(rvals) / len(rvals), 8) if rvals else "",
                    "failure_classes": dict(sorted({e: sum(1 for r in rs if r["error_class"] == e) for e in {r["error_class"] for r in rs if r["error_class"]}}.items())),
                })
    return out


def make_contrasts(rows):
    result = []
    primary = []
    secondary = []
    for dataset in ("enterprise", "musique"):
        for order_scope, orders, family in [("original", ["original"], "primary"), ("non_original_mean", ORDERS[1:], "secondary")]:
            if family == "secondary":
                # Average each policy's per-query scores across the three frozen orders.
                by = defaultdict(dict)
                for r in rows:
                    if r["dataset"] == dataset and r["order_id"] in orders and r["labeled"]:
                        by[(r["canonical_id"], r["policy"])][r["order_id"]] = float(r["fs5"])
                a, b = [], []
                for cid in sorted({k[0] for k in by}):
                    p3 = by.get((cid, "jina_two_stage_matched"), {})
                    p6 = by.get((cid, "pplx_two_stage"), {})
                    if len(p3) == 3 and len(p6) == 3:
                        a.append(sum(p6.values()) / 3); b.append(sum(p3.values()) / 3)
            else:
                rs = [r for r in rows if r["dataset"] == dataset and r["order_id"] == "original" and r["labeled"]]
                p3 = {r["canonical_id"]: r for r in rs if r["policy"] == "jina_two_stage_matched"}
                p6 = {r["canonical_id"]: r for r in rs if r["policy"] == "pplx_two_stage"}
                ids = sorted(set(p3) & set(p6))
                a = [float(p6[x]["fs5"]) for x in ids if p6[x]["fs5"] != ""]
                b = [float(p3[x]["fs5"]) for x in ids if p3[x]["fs5"] != ""]
            stats = bootstrap_delta(a, b)
            entry = {"dataset": dataset, "contrast": "pplx_two_stage-jina_two_stage_matched", "family": family, "order_scope": order_scope, **stats, "n_units": len(a), "unit_type": "query"}
            result.append(entry)
            (primary if family == "primary" else secondary).append(entry)
    # Holm correction within each declared two-contrast family.
    for family_entries in (primary, secondary):
        ranked = sorted(family_entries, key=lambda x: (x["p_randomization"] is None, x["p_randomization"] or 1))
        m = len(ranked)
        for i, x in enumerate(ranked):
            x["p_holm"] = min(1.0, (m - i) * (x["p_randomization"] or 1.0)) if x["p_randomization"] is not None else None
    for x in result:
        x.setdefault("p_holm", None)
        x["units_positive"] = ""
    return result


def write_exports(rows, summaries, contrasts, chosen):
    for dataset in ("enterprise", "musique"):
        d = OUT / "export" / dataset
        pr = [r for r in rows if r["dataset"] == dataset]
        fields = ["canonical_id", "group_id", "order_id", "policy", "valid", "error_class", "labeled", "reachable", "fs5", "n_calls", "input_tokens"]
        write_csv(d / "perquery.csv", [{k: r[k] for k in fields} for r in pr], fields)
        sf = ["policy", "order_id", "n_requests", "n_valid", "valid_pct", "n_labeled", "mean_fs5_all_labeled", "mean_fs5_reachable"]
        write_csv(d / "summary.csv", [{k: x[k] for k in sf} for x in summaries if x["dataset"] == dataset], sf)
        cf = ["contrast", "family", "order_scope", "delta", "lo95", "hi95", "p_randomization", "p_holm", "units_positive", "unit_type", "n_units"]
        write_csv(d / "contrasts.csv", [{k: x.get(k, "") for k in cf} for x in contrasts if x["dataset"] == dataset], cf)
        write_csv(d / "budget_sensitivity.csv", [], ["policy", "contrast_vs", "lower95", "pass_b0.02", "pass_b0.05", "pass_b0.10"])
        write_csv(d / "length_scan.csv", [], ["scope", "n", "share_over_limit", "q50", "q90", "q99"])
        # Serving is intentionally empty until a valid same-session Stage 3 run exists.
        write_csv(d / "serving_perrequest.csv", [], ["policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"])
        write_csv(d / "serving_ratios.csv", [], ["policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib"])
        write_csv(d / "determinism.csv", [], ["policy", "order_id", "n_rechecked", "n_mismatch"])
        audit = []
        p3 = {(r["canonical_id"], r["order_id"]): r for r in pr if r["policy"] == "jina_two_stage_matched"}
        p6 = {(r["canonical_id"], r["order_id"]): r for r in pr if r["policy"] == "pplx_two_stage"}
        for key in sorted(set(p3) | set(p6)):
            audit.append({"canonical_id": key[0], "order_id": key[1], "survivor_sets_equal": "NOT_AVAILABLE_FROM_MISSING_OR_REDACTED_TRACE" if key not in p3 or key not in p6 else "NOT_EMITTED_BY_EXPORT_CONTRACT"})
        write_csv(d / "survivors_audit.csv", audit, ["canonical_id", "order_id", "survivor_sets_equal"])


def main():
    print("analysis: loading frozen manifest", flush=True)
    manifest = load_manifest()
    print(f"analysis: manifest={len(manifest)}", flush=True)
    labels = load_labels(manifest)
    print(f"analysis: labels={len(labels)}", flush=True)
    traces, chosen = load_traces()
    print(f"analysis: trace_files={len(chosen)} keys={sorted(chosen)}", flush=True)
    rows = expected_rows(manifest, labels, traces)
    print(f"analysis: perquery_rows={len(rows)}", flush=True)
    summaries = summarize(rows)
    print("analysis: summaries complete", flush=True)
    contrasts = make_contrasts(rows)
    print("analysis: contrasts complete", flush=True)
    write_exports(rows, summaries, contrasts, chosen)
    raw_manifest = {}
    for key, path in sorted(chosen.items()):
        raw_manifest[str(key)] = {"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "records": sum(1 for _ in path.open(encoding="utf-8"))}
    write_json(OUT / "RAW_TRACE_SELECTION.json", raw_manifest)
    write_json(OUT / "CONTRASTS.json", contrasts)
    write_json(OUT / "SUMMARY.json", summaries)
    write_json(OUT / "STATE.json", {
        "status": "PARTIAL_UNTIL_PPLX_AND_STAGE3_COMPLETE",
        "label_joined_after_raw_trace_freeze": True,
        "manifest_rows": len(manifest), "perquery_rows": len(rows),
        "raw_trace_keys": [list(k) for k in sorted(chosen)],
        "missing_pplx": [p for p in ("pplx_native", "pplx_budget", "pplx_two_stage") if not any(k[0] == p for k in chosen)],
        "missing_stage3": True,
    })
    print(json.dumps({"manifest_rows": len(manifest), "perquery_rows": len(rows), "trace_files": len(chosen), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
