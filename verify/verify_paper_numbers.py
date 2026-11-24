#!/usr/bin/env python3
"""Check printed paper assertions against the released, sanitized evidence."""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def rounded(value: float, digits: int) -> str:
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def main() -> int:
    expected = json.loads((ROOT / "verify/expected_numbers.json").read_text(encoding="utf-8"))
    bright = records(ROOT / "data/scores/bright_perquery.csv")
    cqa = records(ROOT / "data/scores/cqadupstack_perquery.csv")
    transfer = records(ROOT / "data/scores/transfer_perquery.csv")
    cqa_summary = records(ROOT / "results/07_frozen_cqadupstack/cqadupstack_summary.csv")
    cqa_contrasts = records(ROOT / "results/07_frozen_cqadupstack/cqadupstack_contrasts.csv")
    cqa_latency = records(ROOT / "results/07_frozen_cqadupstack/clean_latency_ratios.csv")
    cqa_positive = records(ROOT / "results/07_frozen_cqadupstack/judged_share.csv")
    cqa_manifest = json.loads((ROOT / "protocols/cqadupstack_download_manifest.json").read_text(encoding="utf-8"))

    bright_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in bright:
        bright_groups[(row.get("policy", ""), row.get("budget_L", ""), row.get("order_id", "original"))].append(row)
    cqa_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cqa:
        cqa_groups[row["policy"]].append(row)
    cqa_summary_map = {row["policy"]: row for row in cqa_summary}
    transfer_ids: dict[str, set[str]] = defaultdict(set)
    transfer_labeled_ids: dict[str, set[str]] = defaultdict(set)
    for row in transfer:
        transfer_ids[row["dataset"]].add(row.get("canonical_id", ""))
        if truth(row.get("labeled", "")):
            transfer_labeled_ids[row["dataset"]].add(row.get("canonical_id", ""))

    def metric_for(selector: dict[str, str]) -> float | None:
        if selector.get("source") == "bright_metrics":
            model = selector["model"]
            values = []
            with gzip.open(ROOT / "data/scores/bright_metrics.jsonl.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("model") == model:
                        values.append(float(row["operational_ndcg5"]))
            return sum(values) / len(values) if values else None
        key = (selector.get("policy", ""), selector.get("budget_L", ""), selector.get("order_id", "original"))
        values = bright_groups.get(key, [])
        if not values:
            return None
        return sum(float(row["ndcg5"]) for row in values) / len(values)

    def observed(assertion: dict) -> float | None:
        check = assertion.get("check")
        if check == "bright_candidate_query_count":
            with gzip.open(ROOT / "data/candidates/bright_top20_ids.jsonl.gz", "rt", encoding="utf-8") as stream:
                return float(len({(json.loads(line).get("domain", ""), json.loads(line)["query_id"]) for line in stream if line.strip()}))
        if check == "bright_positive_top20_count":
            count = 0
            seen: set[tuple[str, str]] = set()
            with gzip.open(ROOT / "data/scores/bright_metrics.jsonl.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row.get("domain", ""), row["query_id"])
                    if row.get("model") == "jina_v3_5" and key not in seen:
                        seen.add(key)
                        count += truth(row.get("gold_in_top20", False))
            return float(count)
        if check == "bright_positive_top20_percent":
            count = observed({"check": "bright_positive_top20_count"})
            n = observed({"check": "bright_candidate_query_count"})
            return 100.0 * count / n if n else None
        if check == "bright_mean":
            return metric_for(assertion["selector"])
        if check == "bright_valid_percent":
            sel = assertion["selector"]
            values = bright_groups.get((sel["policy"], sel.get("budget_L", ""), sel.get("order_id", "original")), [])
            return 100.0 * sum(truth(row["valid"]) for row in values) / len(values) if values else None
        if check == "cqadupstack_scored_count":
            return float(len({row["canonical_id"] for row in cqa}))
        if check == "cqadupstack_total_queries":
            return float(cqa_manifest["total_queries"])
        if check == "cqadupstack_dry_run_exclusions":
            return float(cqa_manifest["total_queries"] - len({row["canonical_id"] for row in cqa}))
        if check == "cqadupstack_mean":
            policy = assertion["selector"]["policy"]
            values = cqa_groups.get(policy, [])
            return sum(float(row["ndcg5"]) for row in values) / len(values) if values else None
        if check == "cqadupstack_valid_percent":
            policy = assertion["selector"]["policy"]
            row = cqa_summary_map.get(policy)
            return 100.0 * float(row["valid_rate"]) if row else None
        if check == "cqadupstack_jina_oom_count":
            raw = cqa_summary_map.get("jina_full_text", {}).get("failures", "{}")
            try:
                parsed = json.loads(raw.replace("'", '"'))
            except Exception:
                parsed = {"CUDA_OOM": 1} if "CUDA_OOM" in raw else {}
            return float(parsed.get("CUDA_OOM", 0))
        if check == "transfer_musique_count":
            return float(len(transfer_ids.get("musique", set())))
        if check == "transfer_total_count":
            return float(sum(len(ids) for ids in transfer_ids.values()))
        if check == "transfer_enterprise_count":
            return float(len(transfer_ids.get("enterprise", set())))
        if check == "transfer_enterprise_labeled_count":
            return float(len(transfer_labeled_ids.get("enterprise", set())))
        if check == "cqadupstack_top20_share_percent":
            return 100.0 * float(cqa_positive[0]["top20_judged_share"]) if cqa_positive else None
        if check == "cqadupstack_top20_gold_count":
            return float(cqa_positive[0]["top20_judged_n"]) if cqa_positive else None
        if check == "cqadupstack_latency_ratio":
            policy = assertion["selector"]["policy"]
            basis = assertion["selector"].get("basis", "pooled_serial_passes")
            values = [r for r in cqa_latency if r["policy"] == policy and r["basis"] == basis]
            return float(values[0]["p95_ratio"]) if values else None
        if check == "cqadupstack_latency_ci":
            policy = assertion["selector"]["policy"]
            basis = assertion["selector"].get("basis", "pooled_serial_passes")
            values = [r for r in cqa_latency if r["policy"] == policy and r["basis"] == basis]
            if not values:
                return None
            key = assertion["selector"]["bound"]
            return float(values[0][key])
        if check == "cqadupstack_forum_contrast":
            contrast = assertion["selector"]["contrast"]
            values = [r for r in cqa_contrasts if r["contrast"] == contrast]
            if not values:
                return None
            return float(values[0][assertion["selector"]["column"]])
        return None

    passed = failed = not_verifiable = 0
    for item in expected["assertions"]:
        if item.get("verification") == "NOT_VERIFIABLE":
            not_verifiable += 1
            print(f"NOT_VERIFIABLE {item['id']}: {item.get('reason', 'source input absent')}")
            continue
        value = observed(item)
        if value is None or not math.isfinite(value):
            not_verifiable += 1
            print(f"NOT_VERIFIABLE {item['id']}: configured source check could not be evaluated")
            continue
        if "digits" in item:
            ok = rounded(value, int(item["digits"])) == rounded(float(item["expected"]), int(item["digits"]))
            observed_text = rounded(value, int(item["digits"]))
        else:
            tolerance = float(item.get("tolerance", 0))
            ok = abs(value - float(item["expected"])) <= tolerance
            observed_text = f"{value:.8g}"
        if ok:
            passed += 1
            print(f"PASS {item['id']}: observed={observed_text} expected={item['expected']}")
        else:
            failed += 1
            print(f"FAIL {item['id']}: observed={observed_text} expected={item['expected']}")
    print(f"SUMMARY PASS={passed} FAIL={failed} NOT_VERIFIABLE={not_verifiable}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
