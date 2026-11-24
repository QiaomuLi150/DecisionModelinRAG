#!/usr/bin/env python3
"""Offline label join and analysis for the v1.5 GPU3 C4 trace."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_5/c4_winnow_rerun_gpu3"
RAW = OUT / "raw/winnow12_D_budget_RAW_V2.jsonl"
SLATES = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
EXAMPLES = ROOT / "data/external/e9_bright_v1_0/hf_cache/xlangai___bright/examples/0.0.0/3066d29c9651a576c8aba4832d249807b181ecae"
JINA = ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl"


def read_jsonl(path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ndcg(selected, gold):
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(selected[:5]) if x in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    return dcg / ideal if ideal else 0.0


def ci(values):
    xs = sorted(values)
    return [xs[int(.025 * (len(xs) - 1))], xs[int(.975 * (len(xs) - 1))]]


def bootstrap(diffs, domains, reps=5000, seed=731):
    rng = random.Random(seed)
    q = []
    for _ in range(reps):
        q.append(statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs))
    groups = defaultdict(list)
    for d, g in zip(diffs, domains): groups[g].append(d)
    names = sorted(groups)
    rng = random.Random(seed)
    eq = []
    for _ in range(reps):
        eq.append(statistics.mean(statistics.mean(groups[names[rng.randrange(len(names))]]) for _ in names))
    return {"query_weighted": {"estimate": statistics.mean(diffs), "lo95": ci(q)[0], "hi95": ci(q)[1]}, "equal_domain": {"estimate": statistics.mean(diffs), "lo95": ci(eq)[0], "hi95": ci(eq)[1]}, "reps": reps, "seed": seed, "domain_count": len(names)}


def main():
    (OUT / "export").mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(RAW)
    if len(rows) != 1384:
        raise SystemExit(f"expected 1384 raw rows, got {len(rows)}")
    slates = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(SLATES)}
    gold = {}
    from datasets import Dataset
    for path in sorted(EXAMPLES.glob("bright-*.arrow")):
        domain = path.stem[len("bright-"):]
        for item in Dataset.from_file(str(path)):
            gold[f"{domain}:{item['id']}"] = set(str(x) for x in (item.get("gold_ids") or []))
    jina_rows = {f"{x.get('domain')}:{x.get('query_id')}" : x for x in read_jsonl(JINA)}
    out_rows = []
    diffs, domains = [], []
    for row in rows:
        key = row["canonical_id"]
        valid = row.get("status") == "PASS" and len(row.get("selected_ids", [])) == 5 and len(set(row.get("selected_ids", []))) == 5
        wm = ndcg(row.get("selected_ids", []) if valid else [], gold.get(key, set()))
        jr = jina_rows.get(key, {})
        jvalid = jr.get("status") in {"PASS", "ok"}
        jm = ndcg(jr.get("selected_ids", []) if jvalid else [], gold.get(key, set()))
        diffs.append(wm - jm); domains.append(slates[key]["domain"])
        out_rows.append({"canonical_id": key, "domain": slates[key]["domain"], "policy": "winnow12_D_budget", "valid": valid, "metric": wm, "jina_metric": jm, "elapsed_ms": row.get("elapsed_ms"), "error": row.get("error")})
    with (OUT / "export/c4_perquery.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    valid_n = sum(x["valid"] for x in out_rows)
    summary = {"policy": "winnow12_D_budget", "n": len(out_rows), "valid": valid_n, "failed": len(out_rows) - valid_n, "operational_ndcg5_all_attempts": statistics.mean(x["metric"] for x in out_rows), "jina_full_ndcg5_all_attempts": statistics.mean(x["jina_metric"] for x in out_rows), "winnow_minus_jina_full": bootstrap(diffs, domains), "frozen_slate_sha256": file_sha(SLATES), "raw_sha256": file_sha(RAW), "labels_joined_after_raw_freeze": True}
    (OUT / "export/c4_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (OUT / "export/c4_domain_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "n", "valid", "winnow_ndcg5", "jina_ndcg5", "difference"]); w.writeheader()
        by = defaultdict(list)
        for x in out_rows: by[x["domain"]].append(x)
        for domain in sorted(by):
            xs = by[domain]; w.writerow({"domain": domain, "n": len(xs), "valid": sum(x["valid"] for x in xs), "winnow_ndcg5": statistics.mean(x["metric"] for x in xs), "jina_ndcg5": statistics.mean(x["jina_metric"] for x in xs), "difference": statistics.mean(x["metric"] - x["jina_metric"] for x in xs)})
    (OUT / "C4_OFFLINE_ANALYSIS_REPORT.md").write_text(f"# v1.5 C4 GPU3 Offline Analysis\n\n- Raw rows: `{len(rows)}`; valid native selections: `{valid_n}`; failures retained: `{len(rows)-valid_n}`.\n- Operational all-attempt Winnow known-positive nDCG@5: `{summary['operational_ndcg5_all_attempts']:.6f}`.\n- Jina full reference all-attempt nDCG@5: `{summary['jina_full_ndcg5_all_attempts']:.6f}`.\n- Winnow minus Jina query-weighted estimate: `{summary['winnow_minus_jina_full']['query_weighted']['estimate']:.6f}`, 95% CI `{summary['winnow_minus_jina_full']['query_weighted']['lo95']}` to `{summary['winnow_minus_jina_full']['query_weighted']['hi95']}`.\n- Equal-domain sensitivity is in `export/c4_summary.json`.\n\nThe raw trace was label-blind; labels were joined only by this offline command after raw-output completion. Native failures score zero in the operational all-attempt metric and remain separately reported. This is a v1.5 control result, not a production replacement claim.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
