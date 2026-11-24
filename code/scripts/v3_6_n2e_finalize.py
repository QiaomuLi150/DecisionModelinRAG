#!/usr/bin/env python3
"""CPU-only closure for the corrected v3.6 N2-E repair shards."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/v3_6_essential/followup_v2_n2e_repair_attempt2"
GPU5 = ROOT / "results/v3_6_essential/followup_v2_n2e_repair_attempt5_gpu5"
OUT = ROOT / "results/v3_6_essential/followup_v2_n2e_repair_final"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
GOLD = ROOT / "results/non_jev/v2/e9_bright_v1_0/00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json"
SEED = 731
BOOT = 40_000
POLICIES = ("jina_full", "pplx_budget", "winnow_native")


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


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


def load_rows():
    paths = [SRC / "shards" / f"gpu{g}" / "raw.jsonl" for g in (3, 4, 6)]
    paths.append(GPU5 / "shards/gpu5/raw.jsonl")
    rows = []
    for p in paths:
        if not p.exists():
            raise RuntimeError(f"MISSING_RAW_SHARD:{p}")
        shard = read_jsonl(p)
        if len(shard) != 810:
            raise RuntimeError(f"SHARD_RECORD_COUNT:{p}:{len(shard)}")
        rows.extend(shard)
    return rows, paths


def validate(rows):
    expected = 360 * 3 * 3
    if len(rows) != expected:
        raise RuntimeError(f"FINAL_RECORD_COUNT:{len(rows)}:{expected}")
    keys = [(r.get("policy"), r.get("canonical_id"), int(r.get("attempt", 0))) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("DUPLICATE_POLICY_QUERY_ATTEMPT")
    ids = {r.get("canonical_id") for r in rows}
    if len(ids) != 360:
        raise RuntimeError(f"QUERY_COVERAGE:{len(ids)}")
    counts = {p: sum(r.get("policy") == p for r in rows) for p in POLICIES}
    if counts != {p: 1080 for p in POLICIES}:
        raise RuntimeError(f"POLICY_COUNTS:{counts}")
    bad = [r for r in rows if r.get("labels_joined") is not False or r.get("input_label_blind") is not True]
    if bad:
        raise RuntimeError(f"LABEL_LEAKAGE_OR_SCHEMA:{len(bad)}")
    return counts


def query_metrics(rows, slates, gold):
    slate_by_cid = {f"{s['domain']}:{s['query_id']}": s for s in slates}
    row_ids = {r["canonical_id"] for r in rows}
    selected_slates = [s for cid, s in slate_by_cid.items() if cid in row_ids]
    by = defaultdict(list)
    for r in rows:
        cid = r["canonical_id"]
        selected = r.get("selected_ids") or []
        valid = r.get("status") == "PASS" and len(selected) == 5 and len(set(selected)) == 5
        slate = slate_by_cid[cid]
        g = gold.get((str(slate["domain"]), str(slate["query_id"])), set())
        metric = ndcg5(selected, g) if valid else 0.0
        by[(r["policy"], cid)].append({"metric": metric, "valid": int(valid), "latency_ms": float(r.get("latency_ms") or 0.0), "alloc_peak_mib": r.get("alloc_peak_mib")})
    records = []
    for (policy, cid), vals in sorted(by.items()):
        records.append({"canonical_id": cid, "domain": str(slate_by_cid[cid]["domain"]), "policy": policy, "attempts": len(vals), "valid_attempts": sum(v["valid"] for v in vals), "mean_ndcg5": float(np.mean([v["metric"] for v in vals])), "mean_latency_ms": float(np.mean([v["latency_ms"] for v in vals])), "p95_latency_ms": float(np.quantile([v["latency_ms"] for v in vals], .95))})
    return records, selected_slates


def bootstrap(a, b, domains):
    groups = defaultdict(list)
    for x, y, d in zip(a, b, domains):
        groups[d].append(float(x - y))
    names = sorted(groups)
    rng = np.random.default_rng(SEED)
    vals = np.zeros(BOOT, dtype=np.float64)
    for d in names:
        g = np.asarray(groups[d], dtype=np.float64)
        idx = rng.integers(0, len(g), size=(BOOT, len(g)))
        vals += g[idx].mean(axis=1) / len(names)
    effects = np.asarray([np.mean(groups[d]) for d in names])
    sign = []
    for mask in range(1 << len(effects)):
        sign.append(float(np.mean([(-1 if (mask >> i) & 1 else 1) * effects[i] for i in range(len(effects))])))
    observed = float(np.mean([x - y for x, y in zip(a, b)]))
    return {"delta_query_weighted": observed, "lo95": float(np.quantile(vals, .025)), "hi95": float(np.quantile(vals, .975),), "lo90": float(np.quantile(vals, .05)), "hi90": float(np.quantile(vals, .95)), "p_signflip": float(np.mean(np.abs(sign) >= abs(float(np.mean(effects))) - 1e-15)), "domains_positive": int(sum(x > 0 for x in effects)), "n_queries": len(a), "n_domains": len(names)}


def main():
    rows, source_paths = load_rows()
    slates, gold = load_inputs()
    counts = validate(rows)
    records, selected_slates = query_metrics(rows, slates, gold)
    lookup = {(r["policy"], r["canonical_id"]): r for r in records}
    contrasts = []
    for policy in ("pplx_budget", "winnow_native"):
        a = [lookup[(policy, f"{s['domain']}:{s['query_id']}")]["mean_ndcg5"] for s in selected_slates]
        b = [lookup[("jina_full", f"{s['domain']}:{s['query_id']}")]["mean_ndcg5"] for s in selected_slates]
        d = [str(s["domain"]) for s in selected_slates]
        st = bootstrap(a, b, d)
        contrasts.append({"contrast": f"{policy}-jina_full", **st})
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "RAW_N2E_REPAIR_FINAL.jsonl", sorted(rows, key=lambda x: (x["policy"], x["canonical_id"], int(x["attempt"]))))
    with (OUT / "N2E_PERQUERY.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(records[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    with (OUT / "N2E_CONTRASTS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(contrasts[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(contrasts)
    summary = []
    for policy in POLICIES:
        rs = [r for r in records if r["policy"] == policy]
        summary.append({"policy": policy, "queries": len(rs), "attempt_records": counts[policy], "valid_attempts": sum(r["valid_attempts"] for r in rs), "valid_attempt_rate": sum(r["valid_attempts"] for r in rs) / counts[policy], "mean_ndcg5_all_attempts": float(np.mean([r["mean_ndcg5"] for r in rs])), "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in rs])), "p95_query_mean_latency_ms": float(np.quantile([r["mean_latency_ms"] for r in rs], .95))})
    write_json(OUT / "N2E_SUMMARY.json", {"status": "PASS_WITH_RESOURCE_CAVEAT", "counts": summary, "contrasts": contrasts, "labels_joined_after_raw_freeze": True, "source_paths": [str(p.relative_to(ROOT)) for p in source_paths], "raw_sha256": sha_file(OUT / "RAW_N2E_REPAIR_FINAL.jsonl")})
    manifest = {"source_hashes": {str(p.relative_to(ROOT)): sha_file(p) for p in source_paths}, "output_hashes": {"RAW_N2E_REPAIR_FINAL.jsonl": sha_file(OUT / "RAW_N2E_REPAIR_FINAL.jsonl"), "N2E_PERQUERY.csv": sha_file(OUT / "N2E_PERQUERY.csv"), "N2E_CONTRASTS.csv": sha_file(OUT / "N2E_CONTRASTS.csv")}, "validation": {"counts": counts, "duplicate_keys": False, "query_coverage": 360, "protected_holdouts_accessed": False}}
    write_json(OUT / "PROVENANCE_MANIFEST.json", manifest)
    lines = ["# v3.6 N2-E repaired closure", "", "Status: `PASS_WITH_RESOURCE_CAVEAT`", "", f"The primary repaired N2-E record set contains {len(rows)} records: 360 queries × 3 policies × 3 attempts. GPU 3, 4 and 6 came from the clean corrected-run shards; GPU 5 came from the corrected no-fill-in rerun after its first implementation attempt was invalidated.", "", "All raw serving records were label-blind. Frozen BRIGHT labels were joined only in this CPU closure.", "", "The first GPU-5 attempt is preserved separately and excluded because it contained 327 PPLX rows rather than the required 270, caused by synthetic fatal-error fill-in. The corrected GPU-5 shard contains exactly 810 records.", "", "Resource limitation: the current worker process retained a model allocation while the Winnow server was active on some shards. Quality/validity results are retained; those latency and memory observations are descriptive and are not promoted as isolated matched-serving evidence.", "", "N3 remains blocked independently because the exact BM25-only transfer artifact is unavailable; the available v2.1 transfer population is hybrid RRF."]
    (OUT / "N2E_REPAIR_CLOSURE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_RESOURCE_CAVEAT", "records": len(rows), "summary": summary, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
