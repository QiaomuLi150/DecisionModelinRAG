#!/usr/bin/env python3
"""Regenerate available paper data tables from released CSV/JSONL exports.

This script deliberately does not synthesize omitted queries, labels, plots, or
model-screen rows. Outputs are derived only from files shipped in this release.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/regenerated"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: list[float]) -> float | str:
    return sum(xs) / len(xs) if xs else ""


bright = load_csv(ROOT / "data/scores/bright_perquery.csv")
groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
for row in bright:
    groups[(row["policy"], row.get("budget_L", ""), row.get("order_id", "original"))].append(row)
bright_rows = []
for (policy, budget, order_id), rows in sorted(groups.items()):
    valid = [r for r in rows if r.get("valid", "").lower() in {"1", "true", "yes"}]
    bright_rows.append({
        "policy": policy,
        "budget_L": budget,
        "order_id": order_id,
        "n": len(rows),
        "valid_n": len(valid),
        "valid_rate": len(valid) / len(rows) if rows else "",
        "mean_ndcg5_all": mean([float(r["ndcg5"]) for r in rows]),
        "mean_ndcg5_valid": mean([float(r["ndcg5"]) for r in valid]),
        "source": "data/scores/bright_perquery.csv",
    })
# The baseline is from the separately frozen E9 metrics file, whose identities
# are retained as an explicit model label rather than joined by positional row.
e9: dict[str, list[float]] = defaultdict(list)
with gzip.open(ROOT / "data/scores/bright_metrics.jsonl.gz", "rt", encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            e9[row["model"]].append(float(row["operational_ndcg5"]))
write_csv("bright_summary.csv", bright_rows, ["policy", "budget_L", "order_id", "n", "valid_n", "valid_rate", "mean_ndcg5_all", "mean_ndcg5_valid", "source"])
write_csv("bright_e9_model_summary.csv", [
    {"model": model, "n": len(values), "mean_operational_ndcg5": mean(values), "source": "data/scores/bright_metrics.jsonl.gz"}
    for model, values in sorted(e9.items())
], ["model", "n", "mean_operational_ndcg5", "source"])

cqa = load_csv(ROOT / "data/scores/cqadupstack_perquery.csv")
cqa_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in cqa:
    cqa_groups[row["policy"]].append(row)
cqa_rows = []
for policy, rows in sorted(cqa_groups.items()):
    valid = [r for r in rows if r.get("valid", "").lower() in {"1", "true", "yes"}]
    cqa_rows.append({"policy": policy, "n": len({r["canonical_id"] for r in rows}), "valid_n": len(valid),
                     "valid_rate": len(valid) / len(rows) if rows else "",
                     "mean_ndcg5_all": mean([float(r["ndcg5"]) for r in rows]),
                     "source": "data/scores/cqadupstack_perquery.csv"})
write_csv("cqadupstack_summary.csv", cqa_rows, ["policy", "n", "valid_n", "valid_rate", "mean_ndcg5_all", "source"])

transfer = load_csv(ROOT / "data/scores/transfer_perquery.csv")
transfer_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
for row in transfer:
    transfer_groups[(row["dataset"], row["L"], row.get("policy", row.get("model", "")))].append(row)
transfer_rows = []
for (dataset, budget, policy), rows in sorted(transfer_groups.items()):
    labeled = [r for r in rows if r.get("ndcg5", "") != "" or r.get("fs5", "") != ""]
    transfer_rows.append({"dataset": dataset, "L": budget, "policy": policy, "n": len({r.get("canonical_id", "") for r in rows}),
                          "labeled_n": len(labeled), "mean_ndcg5": mean([float(r["ndcg5"]) for r in labeled if r.get("ndcg5", "") != ""]),
                          "mean_fs5": mean([float(r["fs5"]) for r in labeled if r.get("fs5", "") != ""]),
                          "source": "data/scores/transfer_perquery.csv"})
write_csv("transfer_summary.csv", transfer_rows, ["dataset", "L", "policy", "n", "labeled_n", "mean_ndcg5", "mean_fs5", "source"])

frontier = load_csv(ROOT / "results/05_latency_and_memory/frontier_points.csv")
write_csv("frontier_points.csv", frontier, list(frontier[0]) if frontier else [])
for name, source in [
    ("cqadupstack_forum_contrasts.csv", "results/07_frozen_cqadupstack/cqadupstack_contrasts.csv"),
    ("transfer_paired_contrasts.csv", "results/06_transfer_enterprise_musique/transfer_contrasts.csv"),
    ("order_multistage_results.csv", "results/03_order_and_multistage/two_stage_specialist_perquery.csv"),
]:
    rows = load_csv(ROOT / source)
    write_csv(name, rows, list(rows[0]) if rows else [])

summary = {
    "status": "PARTIAL_REGENERATION",
    "generated": sorted(p.name for p in OUT.glob("*.csv")),
    "not_generated": ["paper source TeX tables", "Figures 1-3 images", "full 21-model/11-benchmark screen"],
    "reason": "required original source inputs/plot program and complete screen matrix are not present in the release",
}
(OUT / "generation_status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
