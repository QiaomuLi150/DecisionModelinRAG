#!/usr/bin/env python3
"""Independent CPU cross-check for pivotal E4-3 estimates.

Intentionally does not import the E4-3 runner or E4-R reconciliation module.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e4_v1_1/e4_r_reconciliation"
SEED = 731
REPS = 5000


def rows(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def boot(values):
    if not values:
        return None, None, None
    rng = random.Random(SEED)
    n = len(values)
    vals = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(REPS)]
    vals.sort()
    return sum(values) / n, vals[int(.025 * (REPS - 1))], vals[int(.975 * (REPS - 1))]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
    cohort = rows(ROOT / "results/non_jev/v2/e4_v1_1/e4_3_operating_regimes/E4_3_COHORT_AND_STRATA_MANIFEST.jsonl")
    qmap = {x["query_id"]: x for x in cohort}
    raw = rows(ROOT / "results/non_jev/v2/e4_v1_1/e4_3_operating_regimes/E4_3_RAW_NATIVE_INFERENCE.jsonl")
    valid = {}
    for x in raw:
        if x.get("status") == "VALID" and x.get("query_id") in qmap and len(x.get("selected_ids", [])) == 5 and len(set(x.get("selected_ids", []))) == 5:
            valid[(x["e4_policy"], int(x["k"]), x["query_id"])] = x
    def ok(policy, k, qid):
        return int(set(qmap[qid].get("gold_support_ids_for_analysis_only") or []).issubset(set(valid[(policy, k, qid)]["selected_ids"])))
    rep = [x["query_id"] for x in cohort if x["cohort"] == "representative"]
    k10 = [q for q in rep if all((p, 10, q) in valid for p in ("clef_pointwise_binary", "clef_joint_binary"))]
    k10_feasible = [q for q in k10 if set(qmap[q].get("gold_support_ids_for_analysis_only") or []).issubset(set(qmap[q]["k10_candidate_ids"]))]
    k20 = [q for q in rep if all((p, 20, q) in valid for p in ("clef_pointwise_binary", "clef_joint_binary"))]
    d10 = [ok("clef_joint_binary", 10, q) - ok("clef_pointwise_binary", 10, q) for q in k10_feasible]
    d20 = [ok("clef_joint_binary", 20, q) - ok("clef_pointwise_binary", 20, q) for q in k20]
    inter = [((ok("clef_joint_binary", 20, q) - ok("clef_pointwise_binary", 20, q)) - (ok("clef_joint_binary", 10, q) - ok("clef_pointwise_binary", 10, q))) for q in k10_feasible]
    result = {"status": "PASS", "cohort_rep_n": len(rep), "k10_valid_n": len(k10), "k10_feasible_n": len(k10_feasible), "k20_valid_n": len(k20), "k10_joint_success": sum(ok("clef_joint_binary", 10, q) for q in k10_feasible), "k10_pointwise_success": sum(ok("clef_pointwise_binary", 10, q) for q in k10_feasible), "k10_conditional": boot(d10), "k20_joint_success": sum(ok("clef_joint_binary", 20, q) for q in k20), "k20_pointwise_success": sum(ok("clef_pointwise_binary", 20, q) for q in k20), "k20_unconditional": boot(d20), "interaction": boot(inter), "valid_rows_used": len(valid), "seed": SEED, "reps": REPS}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
