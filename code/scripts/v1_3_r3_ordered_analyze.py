#!/usr/bin/env python3
"""Offline analysis for the frozen shuffled-order PPLX two-stage run.

The native traces are label-blind.  This script is run only after all six
ordered shards have been sealed and joins the frozen BRIGHT gold manifest
offline.  It never changes or rewrites the raw traces.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E9 = ROOT / "results/non_jev/v2/e9_bright_v1_0"
V = ROOT / "results/v1_3/r3_multipass_ordered"
RAW = V / "raw"
EXPORT = V / "export"
SEED = 731
BOOT_DRAWS = 40_000
ORDERS = ["reversed", "perm_seed_1", "perm_seed_2", "perm_seed_3", "perm_seed_4", "perm_seed_5"]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def ndcg5(selected: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    dcg = sum((1.0 if doc in gold else 0.0) / math.log2(i + 2) for i, doc in enumerate(selected[:5]))
    ideal_n = min(5, len(gold))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / ideal if ideal else 0.0


def is_valid(row: dict) -> bool:
    selected = [str(x) for x in row.get("selected_ids", [])]
    return row.get("status") in {"PASS", "ok"} and len(selected) == 5 and len(set(selected)) == 5


def bootstrap(values_by_domain: dict[str, list[float]]) -> dict:
    domains = sorted(values_by_domain)
    values = [np.asarray(values_by_domain[d], dtype=float) for d in domains]
    if not values:
        return {"n": 0, "query_weighted": None, "domain_weighted": None}
    query_values = np.concatenate(values)
    domain_means = np.asarray([v.mean() for v in values], dtype=float)
    rng = np.random.Generator(np.random.PCG64(SEED))
    idx = rng.integers(0, len(domains), size=(BOOT_DRAWS, len(domains)))
    draws = domain_means[idx].mean(axis=1)
    return {
        "n": int(query_values.size),
        "n_domains": len(domains),
        "query_weighted": float(query_values.mean()),
        "domain_weighted": float(domain_means.mean()),
        "domain_bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "seed": SEED,
        "draws": BOOT_DRAWS,
        "domains": {d: {"n": len(values_by_domain[d]), "mean": float(np.mean(values_by_domain[d]))} for d in domains},
    }


def paired_bootstrap(deltas_by_domain: dict[str, list[float]]) -> dict:
    out = bootstrap(deltas_by_domain)
    if not deltas_by_domain:
        out["sign_flip_p_two_sided"] = None
        return out
    domain_means = np.asarray([np.mean(deltas_by_domain[d]) for d in sorted(deltas_by_domain)], dtype=float)
    # Exact sign flip is feasible for the 12 BRIGHT domains and is the same
    # paired-domain sensitivity convention used in the prior R3 report.
    if len(domain_means) <= 20:
        signs = np.asarray(
            [np.mean(domain_means * np.asarray(s)) for s in itertools.product((-1.0, 1.0), repeat=len(domain_means))]
        )
        observed = float(domain_means.mean())
        out["sign_flip_p_two_sided"] = float(np.mean(np.abs(signs) >= abs(observed) - 1e-15))
    else:
        out["sign_flip_p_two_sided"] = None
    out["observed_domain_mean"] = float(domain_means.mean())
    return out


def read_original() -> dict[str, dict]:
    paths = sorted((ROOT / "results/v1_3/r3_multipass").glob("multipass_raw_start*_n*.jsonl"))
    rows = [row for path in paths for row in load_jsonl(path)]
    if len(rows) != 1384 or len({row["canonical_id"] for row in rows}) != 1384:
        raise RuntimeError(f"authenticated original R3 rows invalid: {len(rows)}")
    return {row["canonical_id"]: row for row in rows}


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((V / "PROTOCOL.json").read_text(encoding="utf-8"))
    slates = load_jsonl(E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl")
    manifest = json.loads((E9 / "00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json").read_text(encoding="utf-8"))
    gold = {f"{x['domain']}:{x['id']}": set(map(str, x.get("gold_ids", []))) for x in manifest["items"]}
    slate_by_id = {f"{x['domain']}:{x['query_id']}": x for x in slates}
    original = read_original()

    shard_paths = sorted(RAW.glob("multipass_gpu*_start0_n1384_orders_*.jsonl"))
    if len(shard_paths) != 3:
        raise RuntimeError(f"expected 3 ordered shards, found {len(shard_paths)}")
    ordered_rows: list[dict] = []
    validation = []
    for path in shard_paths:
        rows = load_jsonl(path)
        if len(rows) != 2768:
            raise RuntimeError(f"{path.name}: expected 2768 rows, found {len(rows)}")
        orders = sorted({row.get("order_id") for row in rows})
        if len(orders) != 2 or any(order not in ORDERS for order in orders):
            raise RuntimeError(f"{path.name}: invalid orders {orders}")
        by_order = {order: [row for row in rows if row.get("order_id") == order] for order in orders}
        for order, order_rows in by_order.items():
            ids = [row["canonical_id"] for row in order_rows]
            if len(ids) != 1384 or len(set(ids)) != 1384:
                raise RuntimeError(f"{path.name}/{order}: invalid canonical-id coverage")
            for row in order_rows:
                cid = row["canonical_id"]
                slate = slate_by_id.get(cid)
                if slate is None:
                    raise RuntimeError(f"missing slate for {cid}")
                if row.get("labels_joined") is not False:
                    raise RuntimeError(f"label leakage marker in {path.name}/{cid}")
                candidates = [str(x["candidate_id"]) for x in slate["candidates"]]
                if set(row.get("candidate_ids", [])) != set(candidates) or len(row.get("candidate_ids", [])) != 20:
                    raise RuntimeError(f"candidate-set mismatch for {path.name}/{cid}")
                if row.get("source_slate_sha256") != slate["slate_sha256"]:
                    raise RuntimeError(f"slate hash mismatch for {path.name}/{cid}")
                ordered_rows.append(row)
        validation.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha(path),
            "rows": len(rows),
            "orders": orders,
            "valid": sum(is_valid(row) for row in rows),
            "failed": sum(not is_valid(row) for row in rows),
        })

    by_order = defaultdict(list)
    for row in ordered_rows:
        by_order[row["order_id"]].append(row)
    for order in ORDERS:
        if len(by_order[order]) != 1384:
            raise RuntimeError(f"order {order}: expected 1384 rows, found {len(by_order[order])}")

    detailed = []
    summary = []
    for order in ORDERS:
        order_map = {row["canonical_id"]: row for row in by_order[order]}
        values = []
        deltas = defaultdict(list)
        selection_changes = 0
        order_changes = 0
        valid = 0
        for cid in sorted(order_map):
            row = order_map[cid]
            base = original[cid]
            g = gold.get(cid, set())
            row_valid = is_valid(row)
            base_valid = is_valid(base)
            score = ndcg5([str(x) for x in row.get("selected_ids", [])], g) if row_valid else 0.0
            base_score = ndcg5([str(x) for x in base.get("selected_ids", [])], g) if base_valid else 0.0
            selected = [str(x) for x in row.get("selected_ids", [])]
            base_selected = [str(x) for x in base.get("selected_ids", [])]
            if row_valid:
                valid += 1
            if set(selected) != set(base_selected):
                selection_changes += 1
            if selected != base_selected:
                order_changes += 1
            values.append(score)
            deltas[cid.split(":", 1)[0]].append(score - base_score)
            detailed.append({
                "canonical_id": cid,
                "domain": row["domain"],
                "order_id": order,
                "status": row.get("status"),
                "valid": str(row_valid).lower(),
                "ndcg5": score,
                "original_ndcg5": base_score,
                "delta_vs_original": score - base_score,
                "selected_ids": json.dumps(selected, ensure_ascii=False),
                "original_selected_ids": json.dumps(base_selected, ensure_ascii=False),
                "selection_set_changed": str(set(selected) != set(base_selected)).lower(),
                "selection_order_changed": str(selected != base_selected).lower(),
                "n_calls": row.get("n_calls"),
                "total_ms": row.get("total_ms"),
            })
        summary.append({
            "policy": "pplx_two_stage",
            "order_id": order,
            "rows": 1384,
            "valid": valid,
            "failed": 1384 - valid,
            "all_attempt_mean_ndcg5": float(np.mean(values)),
            "valid_mean_ndcg5": float(np.mean([x for x, row in zip(values, by_order[order]) if is_valid(row)])) if valid else None,
            "selection_set_changed_vs_original": selection_changes,
            "selection_set_changed_rate": selection_changes / 1384.0,
            "selection_order_changed_vs_original": order_changes,
            "selection_order_changed_rate": order_changes / 1384.0,
            "paired_delta_query_weighted": float(np.mean([x for domain in deltas.values() for x in domain])),
            "paired_delta_domain_weighted": float(np.mean([np.mean(x) for x in deltas.values()])),
        })

    # Add confidence intervals and domain sensitivity to summary rows.
    for row in summary:
        order = row["order_id"]
        delta_groups = defaultdict(list)
        value_groups = defaultdict(list)
        order_map = {x["canonical_id"]: x for x in by_order[order]}
        for cid, x in order_map.items():
            domain = x["domain"]
            g = gold.get(cid, set())
            score = ndcg5([str(i) for i in x.get("selected_ids", [])], g) if is_valid(x) else 0.0
            base = original[cid]
            bscore = ndcg5([str(i) for i in base.get("selected_ids", [])], g) if is_valid(base) else 0.0
            value_groups[domain].append(score)
            delta_groups[domain].append(score - bscore)
        row["ndcg5_domain_bootstrap"] = bootstrap(value_groups)
        row["delta_vs_original_domain_bootstrap"] = paired_bootstrap(delta_groups)

    # Cross-order stability is descriptive, not a new quality endpoint.
    stability = []
    for cid in sorted(original):
        rows = [next(x for x in by_order[order] if x["canonical_id"] == cid) for order in ORDERS]
        sets = [set(map(str, x.get("selected_ids", []))) for x in rows]
        scores = [ndcg5(list(map(str, x.get("selected_ids", []))), gold.get(cid, set())) if is_valid(x) else 0.0 for x in rows]
        stability.append({
            "canonical_id": cid,
            "domain": original[cid]["domain"],
            "valid_order_count": sum(is_valid(x) for x in rows),
            "unique_selected_sets": len({tuple(sorted(s)) for s in sets}),
            "mean_ndcg5": float(np.mean(scores)),
            "std_ndcg5": float(np.std(scores)),
        })

    with (EXPORT / "multipass_ordered_perquery.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed[0]))
        writer.writeheader(); writer.writerows(detailed)
    with (EXPORT / "multipass_ordered_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (EXPORT / "multipass_ordered_bootstrap.json").write_text(json.dumps({"orders": summary, "seed": SEED, "draws": BOOT_DRAWS}, indent=2) + "\n", encoding="utf-8")
    with (EXPORT / "multipass_ordered_stability.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stability[0])); writer.writeheader(); writer.writerows(stability)

    provenance = {
        "protocol": str((V / "PROTOCOL.json").relative_to(ROOT)),
        "protocol_sha256": file_sha(V / "PROTOCOL.json"),
        "input_lock": str((V / "INPUT_LOCK.json").relative_to(ROOT)),
        "input_lock_sha256": file_sha(V / "INPUT_LOCK.json"),
        "labels_joined_after_raw_freeze": True,
        "raw_shards": validation,
        "original_r3_shards": [{"path": str(p.relative_to(ROOT)), "sha256": file_sha(p), "rows": len(load_jsonl(p))} for p in sorted((ROOT / "results/v1_3/r3_multipass").glob("multipass_raw_start*_n*.jsonl"))],
        "bright_sha256": file_sha(E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"),
        "gold_manifest_sha256": file_sha(E9 / "00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json"),
        "orders": ORDERS,
        "raw_labels_joined_field_required_false": True,
    }
    (EXPORT / "multipass_ordered_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    report = [
        "# R3 Ordered PPLX Two-Stage Order-Robustness Report",
        "",
        "Status: COMPLETE — six frozen non-original candidate orders were executed and analyzed offline after raw-output freeze.",
        "",
        f"- Population: 1,384 BRIGHT queries × 6 orders = {len(ordered_rows):,} selector attempts.",
        f"- Native model revision: `14c9d90137962d300f3de95633f7b6535e80c5ce`.",
        "- Candidate IDs were permuted only; the frozen Top-20 slate identity and source hash were preserved.",
        "- Gold labels were not read by inference workers; all scoring below was joined offline after all shards were sealed.",
        "",
        "## Order-level summary",
        "",
        "| order | valid | failed | all-attempt nDCG@5 | valid nDCG@5 | set changed vs original | paired Δ query | paired Δ domain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary:
        report.append(f"| {x['order_id']} | {x['valid']} | {x['failed']} | {x['all_attempt_mean_ndcg5']:.6f} | {x['valid_mean_ndcg5']:.6f} | {x['selection_set_changed_rate']:.3f} | {x['paired_delta_query_weighted']:.6f} | {x['paired_delta_domain_weighted']:.6f} |")
    report += [
        "",
        "## Interpretation",
        "",
        "The order rows measure sensitivity of the same two-stage policy to candidate ordering. They are not independent model comparisons. The original-order R3 result is retained as the authenticated historical anchor; all six ordered runs use identical query slates, model revision, budget rule and native three-call contract.",
        "",
        "See `multipass_ordered_bootstrap.json` for paired query/domain uncertainty, `multipass_ordered_stability.csv` for cross-order dispersion, and `multipass_ordered_provenance.json` for raw hashes.",
        "",
    ]
    (V / "MULTIPASS_ORDERED_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"rows": len(ordered_rows), "summary": summary, "shards": validation}, indent=2))


if __name__ == "__main__":
    main()
