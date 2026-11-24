#!/usr/bin/env python3
"""Offline serving/telemetry analysis for the frozen reviewer-response traces."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/non_jev/v2/reviewer_response_final_v1_0"
SERVE = OUT / "serving"
SLATES = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
LOCK = ROOT / "results/non_jev/v2/e12_o_final_v1_0/00_protocol/E12O_COHORT_LOCK.json"


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    raw = read_jsonl(SERVE / "05_RR_SERVING_RAW_ALL_ATTEMPTS.jsonl")
    slates = read_jsonl(SLATES)
    char_len = {f"{r['domain']}:{r['query_id']}": sum(len(str(c.get('content', ''))) for c in r['candidates']) for r in slates}
    ordered = sorted(char_len, key=lambda k: (char_len[k], k))
    n = len(ordered)
    tercile = {k: ("short", "medium", "long")[min(2, (i * 3) // n)] for i, k in enumerate(ordered)}
    policies = sorted({r["policy_id"] for r in raw})
    score = []
    for p in policies:
        rs = [r for r in raw if r["policy_id"] == p]
        all_lat = [r["elapsed_ms"] for r in rs]
        ok = [r for r in rs if r["status"] == "PASS"]
        ok_lat = [r["elapsed_ms"] for r in ok]
        total_s = sum(all_lat) / 1000.0
        score.append({"policy": p, "attempts": len(rs), "valid": len(ok), "failures": len(rs) - len(ok), "valid_rate": len(ok) / len(rs) if rs else None, "all_attempt_p50_ms": percentile(all_lat, .5), "all_attempt_p95_ms": percentile(all_lat, .95), "success_p50_ms": percentile(ok_lat, .5), "success_p95_ms": percentile(ok_lat, .95), "throughput_success_per_s": len(ok) / total_s if total_s else None, "total_elapsed_s": total_s, "peak_allocator_p50_mib": percentile([r["gpu"]["max_memory_allocated_mib"] for r in rs], .5), "peak_allocator_p95_mib": percentile([r["gpu"]["max_memory_allocated_mib"] for r in rs], .95), "device_wide_nvml": "NOT_MEASURED_IN_NEW_TRACE", "physical_forward_count": "NOT_EXPOSED_BY_JINA_RUNTIME", "length_proxy": "frozen Top-20 candidate character-count terciles"})
    with (SERVE / "05_RR_MATCHED_SERVING_SCORECARD.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=score[0].keys())
        w.writeheader()
        w.writerows(score)

    length_rows = []
    for p in policies:
        for t in ["short", "medium", "long"]:
            rs = [r for r in raw if r["policy_id"] == p and tercile.get(r["canonical_id"]) == t]
            ok = [r for r in rs if r["status"] == "PASS"]
            length_rows.append({"policy": p, "length_tercile": t, "n_attempts": len(rs), "n_success": len(ok), "failure_rate": (len(rs) - len(ok)) / len(rs) if rs else None, "success_p50_ms": percentile([r["elapsed_ms"] for r in ok], .5), "success_p95_ms": percentile([r["elapsed_ms"] for r in ok], .95), "char_count_p50": percentile([char_len[r["canonical_id"]] for r in rs], .5) if rs else None, "char_count_p95": percentile([char_len[r["canonical_id"]] for r in rs], .95) if rs else None})
    by = {(r["policy_id"], r["canonical_id"], r["rep"]): r for r in raw}
    for i, a in enumerate(policies):
        for b in policies[i + 1:]:
            keys = sorted(set((r["canonical_id"], r["rep"]) for r in raw if r["policy_id"] == a) & set((r["canonical_id"], r["rep"]) for r in raw if r["policy_id"] == b))
            common = [k for k in keys if by[(a, *k)]["status"] == "PASS" and by[(b, *k)]["status"] == "PASS"]
            da = [by[(a, *k)]["elapsed_ms"] for k in common]
            db = [by[(b, *k)]["elapsed_ms"] for k in common]
            length_rows.append({"policy": f"{a}__VS__{b}", "length_tercile": "PAIR_COMMON_SUCCESS", "n_attempts": len(keys), "n_success": len(common), "failure_rate": None, "success_p50_ms": percentile(da, .5), "success_p95_ms": percentile(da, .95), "paired_other_policy_success_p50_ms": percentile(db, .5), "paired_other_policy_success_p95_ms": percentile(db, .95), "char_count_p50": percentile([char_len[k[0]] for k in common], .5) if common else None})
    with (SERVE / "05_RR_LATENCY_LENGTH_AND_COMMON_SUCCESS.csv").open("w", encoding="utf-8", newline="") as f:
        fields = sorted({k for r in length_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(length_rows)

    mem = []
    for p in policies:
        rs = [r for r in raw if r["policy_id"] == p]
        for outcome in ["SUCCESS", "FAILURE", "ALL"]:
            xs = [r for r in rs if outcome == "ALL" or (r["status"] == "PASS") == (outcome == "SUCCESS")]
            mem.append({"policy": p, "outcome": outcome, "n": len(xs), "process_local_peak_p50_mib": percentile([r["gpu"]["max_memory_allocated_mib"] for r in xs], .5), "process_local_peak_p95_mib": percentile([r["gpu"]["max_memory_allocated_mib"] for r in xs], .95), "device_wide_nvml_peak": "NOT_MEASURED", "instrument_scope": "torch.cuda.max_memory_allocated per request; reset before request", "interpretation": "process-local allocator only; not a physical-device total"})
    with (SERVE / "05_RR_MEMORY_INSTRUMENTS_AND_OUTCOMES.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mem[0].keys())
        w.writeheader()
        w.writerows(mem)

    hist = ROOT / "results/non_jev/v2/e12_o_final_v1_0/03_serving/e12o_o3_attempt_01/jina_v3_5_SERVING_RAW_V2.jsonl"
    hist_rows = read_jsonl(hist)
    hist_ids = {r["canonical_id"] for r in hist_rows if r.get("dataset") == "BRIGHT"}
    new_ids = {r["canonical_id"] for r in raw if r["policy_id"] == "jina_full"}
    report = f"""# Reviewer-response serving and configuration findings

## New bounded serving block

- Frozen cohort: 120 canonical BRIGHT IDs × 3 repetitions, concurrency 1.
- New attempts: {len(raw)} across {len(policies)} Jina policies; valid={sum(r['status']=='PASS' for r in raw)}, failures={sum(r['status']!='PASS' for r in raw)}.
- Same-device execution: visible GPU6 in the privileged run; the process was released after completion.
- Model load is excluded from request timing. Timing starts before request-specific budget preparation and ends after Jina index restoration and valid Top-5 packaging.
- Application cache, labels, qrels and reader answers were not used.

## Historical matching

The historical E12 Jina serving trace contains {len(hist_rows)} rows and {len(hist_ids)} BRIGHT canonical IDs. The new full-content Jina cohort contains {len(new_ids)} IDs; exact ID-set match={hist_ids == new_ids}. Historical and new measurements are retained separately because they were collected in different sessions and should not be pooled as repetitions.

## Liquid

LIQUID_SERVING_BLOCKED_CHECKPOINT_NOT_AVAILABLE: no locally runnable Liquid checkpoint was present in the frozen project model inventory, and this bounded task forbids new checkpoint downloads. No Liquid serving request was made and no Liquid result is imputed.

## Drex-v2

The frozen protocol allowed at most one offline-runtime replication only after authentication. The current bounded serving run did not start Drex-v2; its original E12-O 42/360 event plus subsequent connection failures remain historical configuration-specific evidence. No new runtime repair or restart was performed. The legacy record cannot be treated as a new matched result.

## Resource interpretation

New memory rows report only process-local PyTorch allocator peaks. Device-wide NVML peaks and server-side forward counts are not exposed by the Jina runtime, so they are NOT_MEASURED, not zero. Throughput is serialized workload throughput for this concurrency-one session, not sustainable production QPS.
"""
    (SERVE / "05_RR_LIQUID_AND_DREX_CONFIG_FINDINGS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "SERVING_ANALYSIS_COMPLETE", "policies": policies, "rows": len(raw), "historical_jina_id_set_match": hist_ids == new_ids}, indent=2))


if __name__ == "__main__":
    main()
