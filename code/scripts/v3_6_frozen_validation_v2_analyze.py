#!/usr/bin/env python3
"""Offline label join and scorecard for the frozen CQADupStack v2 run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/frozen_prospective_validation_v2"
DATA = ROOT / "data/external/v3_6_cqadupstack/release/cqadupstack"
SEED = 731
BOOT = 40_000
POLICIES = ["bm25_top5", "pplx_single_6144", "pplx_single_4096", "jina_full_text", "jina_trunc_8192"]


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def qrels():
    out = {}
    for forum in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        vals = defaultdict(set)
        with (DATA / forum / "qrels/test.tsv").open(encoding="utf-8") as f:
            next(f)
            for line in f:
                q, d, rel = line.rstrip("\n").split("\t")
                if int(rel) > 0:
                    vals[q].add(f"{forum}:{d}")
        out[forum] = dict(vals)
    return out


def ndcg5(selected, gold):
    if not gold:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, x in enumerate(selected[:5]) if x in gold)
    ideal = sum((1.0 / math.log2(i + 2)) for i in range(min(5, len(gold))))
    return dcg / ideal if ideal else 0.0


def cp(successes, n, alpha=0.05):
    if n == 0:
        return None, None
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lo, hi


def forum_bootstrap(values_by_forum, draws=BOOT, seed=SEED):
    forums = sorted(values_by_forum)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        sample = rng.choice(forums, size=len(forums), replace=True)
        vals.append(float(np.mean([np.mean(values_by_forum[x]) for x in sample if values_by_forum[x]])))
    return float(np.mean([np.mean(v) for v in values_by_forum.values() if v])), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def main():
    slates = {x["canonical_id"]: x for x in (json.loads(l) for l in (OUT / "CQADUPSTACK_FROZEN_TOP20_SLATES.jsonl").open(encoding="utf-8")) if x}
    lengths = {x["canonical_id"]: x for x in (json.loads(l) for l in (OUT / "QUERY_LENGTHS.jsonl").open(encoding="utf-8")) if x}
    dry = {x["canonical_id"] for x in (json.loads(l) for l in (OUT / "DRY_RUN_IDS.jsonl").open(encoding="utf-8")) if x}
    gold = qrels()
    by_policy = {p: {} for p in POLICIES if p != "bm25_top5"}
    raw_files = sorted((OUT / "raw").glob("native_passA_jina_*.jsonl*")) + sorted((OUT / "raw").glob("native_passA_pplx_*.jsonl"))
    records = []
    for p in raw_files:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                records.append(json.loads(line))
    # A resumed Jina shard is complementary to the interrupted GPU-2 partial.
    # Duplicate (query, policy) records are retained in the audit but the first
    # record is the deterministic primary record for the scorecard.
    duplicates = []
    for r in records:
        key = (r["canonical_id"], r["policy"])
        if r["policy"] in by_policy:
            if key in by_policy[r["policy"]]:
                duplicates.append(key)
            else:
                by_policy[r["policy"]][key[0]] = r
    rows = []
    score_ids = sorted(set(slates) - dry)
    for cid in score_ids:
        s = slates[cid]
        forum = s["forum"]
        g = gold[forum].get(s["query_id"], set())
        base = {"canonical_id": cid, "forum": forum, "length_tercile": lengths[cid]["length_tercile"], "gold_count": len(g), "top20_has_gold": int(bool(set(s["candidate_ids"]) & g))}
        bm = list(s["candidate_ids"][:5])
        rows.append({**base, "policy": "bm25_top5", "valid": 1, "ndcg5": ndcg5(bm, g), "conditional_ndcg5": ndcg5(bm, g), "selected_ids_sha256": sha(bm), "failure_class": ""})
        for policy in POLICIES:
            if policy == "bm25_top5":
                continue
            r = by_policy[policy].get(cid)
            valid = int(bool(r and r.get("valid")))
            selected = r.get("selected_ids", []) if r else []
            rows.append({**base, "policy": policy, "valid": valid, "ndcg5": ndcg5(selected if valid else [], g), "conditional_ndcg5": ndcg5(selected, g) if valid else "", "selected_ids_sha256": sha(selected) if valid else "", "failure_class": (r or {}).get("error_class", "MISSING") if not valid else ""})
    fields = ["canonical_id", "forum", "length_tercile", "policy", "valid", "gold_count", "top20_has_gold", "ndcg5", "conditional_ndcg5", "selected_ids_sha256", "failure_class"]
    write_csv(OUT / "val_perquery.csv", fields, rows)
    summaries = []
    for policy in POLICIES:
        rr = [x for x in rows if x["policy"] == policy]
        vals = [float(x["ndcg5"]) for x in rr]
        valid = sum(int(x["valid"]) for x in rr)
        failures = defaultdict(int)
        for x in rr:
            if not x["valid"]:
                failures[x["failure_class"] or "UNKNOWN"] += 1
        grouped = defaultdict(list)
        for x in rr:
            grouped[x["forum"]].append(float(x["ndcg5"]))
        mean, lo, hi = forum_bootstrap(grouped)
        summaries.append({"policy": policy, "n": len(rr), "valid_n": valid, "valid_rate": valid / len(rr), "valid_cp_lo95": cp(valid, len(rr))[0], "valid_cp_hi95": cp(valid, len(rr))[1], "mean_ndcg5_all": float(np.mean(vals)), "mean_ndcg5_valid": float(np.mean([float(x["ndcg5"]) for x in rr if x["valid"]])) if valid else None, "forum_bootstrap_mean": mean, "forum_bootstrap_lo95": lo, "forum_bootstrap_hi95": hi, "top20_gold_share": float(np.mean([x["top20_has_gold"] for x in rr])), "failures": dict(failures)})
    write_csv(OUT / "val_summary.csv", list(summaries[0]), summaries)
    by_forum = []
    for forum in sorted(slates):
        pass
    for forum in sorted({x["forum"] for x in rows}):
        for policy in POLICIES:
            rr = [x for x in rows if x["forum"] == forum and x["policy"] == policy]
            by_forum.append({"forum": forum, "policy": policy, "n": len(rr), "valid_n": sum(int(x["valid"]) for x in rr), "valid_rate": np.mean([x["valid"] for x in rr]), "mean_ndcg5": np.mean([float(x["ndcg5"]) for x in rr]), "top20_gold_share": np.mean([x["top20_has_gold"] for x in rr])})
    write_csv(OUT / "val_by_forum.csv", ["forum", "policy", "n", "valid_n", "valid_rate", "mean_ndcg5", "top20_gold_share"], by_forum)
    terc = []
    for t in ("short", "medium", "long"):
        for policy in POLICIES:
            rr = [x for x in rows if x["length_tercile"] == t and x["policy"] == policy]
            terc.append({"length_tercile": t, "policy": policy, "n": len(rr), "valid_n": sum(int(x["valid"]) for x in rr), "valid_rate": np.mean([x["valid"] for x in rr]), "mean_ndcg5": np.mean([float(x["ndcg5"]) for x in rr])})
    write_csv(OUT / "val_by_length_tercile.csv", ["length_tercile", "policy", "n", "valid_n", "valid_rate", "mean_ndcg5"], terc)
    judged = []
    for policy in POLICIES:
        rr = [x for x in rows if x["policy"] == policy]
        j = [x for x in rr if x["top20_has_gold"]]
        judged.append({"policy": policy, "all_n": len(rr), "top20_judged_n": len(j), "top20_judged_share": len(j) / len(rr), "all_mean_ndcg5": np.mean([float(x["ndcg5"]) for x in rr]), "judged_only_mean_ndcg5": np.mean([float(x["ndcg5"]) for x in j]) if j else None})
    write_csv(OUT / "val_judged_share.csv", list(judged[0]), judged)
    lookup = {(x["canonical_id"], x["policy"]): x for x in rows}
    contrasts = []
    for left in ("pplx_single_6144", "pplx_single_4096"):
        for right in ("bm25_top5", "jina_full_text"):
            diffs = []
            grouped = defaultdict(list)
            for cid in score_ids:
                a = float(lookup[(cid, left)]["ndcg5"])
                b = float(lookup[(cid, right)]["ndcg5"])
                d = a - b
                diffs.append(d)
                grouped[slates[cid]["forum"]].append(d)
            mean, lo, hi = forum_bootstrap(grouped)
            contrasts.append({"contrast": f"{left}_minus_{right}", "left_policy": left, "right_policy": right, "n": len(diffs), "delta_all_query_weighted": float(np.mean(diffs)), "forum_bootstrap_mean": mean, "forum_bootstrap_lo95": lo, "forum_bootstrap_hi95": hi, "bootstrap_draws": BOOT, "bootstrap_seed": SEED, "bootstrap_unit": "forum"})
    write_csv(OUT / "val_contrasts.csv", list(contrasts[0]), contrasts)
    primary = {x["policy"]: x for x in summaries}
    ledger = {"status": "OFFLINE_LABEL_JOIN_COMPLETE", "protocol_sha256": hashlib.sha256((OUT / "PROTOCOL_FROZEN.md").read_bytes()).hexdigest(), "raw_files": [str(x.relative_to(ROOT)) for x in raw_files], "raw_record_n": len(records), "duplicate_policy_query_records": len(duplicates), "dry_run_n_excluded": len(dry), "scorecard_query_n": len(score_ids), "gold_queries": sum(bool(gold[slates[c]["forum"]].get(slates[c]["query_id"], set())) for c in score_ids), "policies": primary, "contrasts": contrasts, "labels_joined_after_raw_read": True}
    (OUT / "OFFLINE_SCORECARD.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
