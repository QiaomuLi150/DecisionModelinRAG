#!/usr/bin/env python3
"""Freeze the additive two-stage PPLX order-sensitivity continuation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_3/r3_multipass_ordered"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
PERM = ROOT / "results/v1_3/r2_order_sensitivity/permutation_manifest.jsonl"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"

def fsha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def main():
    rows = [json.loads(x) for x in BRIGHT.open(encoding="utf-8") if x.strip()]
    perms = [json.loads(x) for x in PERM.open(encoding="utf-8") if x.strip()]
    assert len(rows) == 1384 and len(perms) == 9688
    assert {r["order_id"] for r in perms} == {"original", "reversed", "perm_seed_1", "perm_seed_2", "perm_seed_3", "perm_seed_4", "perm_seed_5"}
    out = {
        "protocol_id": "decisionrag-v1.3-r3_multipass_ordered-v1.0",
        "experiment_id": "r3_multipass_ordered",
        "status": "FROZEN_PRE_OUTCOME",
        "description": "PPLX two-stage selection on the six non-original candidate orders; original order is reused from authenticated R3.",
        "population": {"bright_records": 1384, "orders": ["reversed", "perm_seed_1", "perm_seed_2", "perm_seed_3", "perm_seed_4", "perm_seed_5"], "attempts_per_order": 1384, "native_calls_per_success": 3, "concurrency_per_worker": 1, "shards": {"gpu3": ["reversed", "perm_seed_1"], "gpu5": ["perm_seed_2", "perm_seed_3"], "gpu6": ["perm_seed_4", "perm_seed_5"]}},
        "policy": {"name": "pplx_two_stage", "revision": PPLX_REV, "stage_1": "two windows of five?", "contract": "two windows of ten, native Top-5 per window, native Top-5 over resulting ten", "budget": {"L": 8192, "rho": 128, "question_shortening": False, "redistribution": False}, "failure_rule": "any failed native call fails the request"},
        "inputs": {"bright_path": str(BRIGHT.relative_to(ROOT)), "bright_sha256": fsha(BRIGHT), "permutation_manifest": str(PERM.relative_to(ROOT)), "permutation_manifest_sha256": fsha(PERM), "labels_joined": False, "gold_or_outcome_fields_read": False},
        "analysis": {"original_order": "authenticated R3 result only", "paired_orders": "same canonical query and source slate; candidate order only changed", "metrics": "offline known-positive nDCG@5 after raw-output freeze", "intervals": "follow R3/R2 predeclared bootstrap conventions; no outcome-dependent exclusions"},
        "deviations": [],
    }
    # Correct the human-readable accidental placeholder before writing the lock.
    out["policy"]["stage_1"] = "two windows of ten"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROTOCOL.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "INPUT_LOCK.json").write_text(json.dumps({"bright_sha256": fsha(BRIGHT), "permutation_sha256": fsha(PERM), "bright_records": 1384, "orders": out["population"]["orders"], "labels_joined": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "protocol": str((OUT/"PROTOCOL.json").relative_to(ROOT)), "bright_sha256": fsha(BRIGHT), "permutation_sha256": fsha(PERM)}, indent=2))

if __name__ == "__main__": main()
