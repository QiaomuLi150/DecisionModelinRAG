#!/usr/bin/env python3
"""Capture read-only host GPU state and v1.3 protocol hashes."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_3/preflight"

def run(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).splitlines()
    except Exception as exc:
        return [f"ERROR:{type(exc).__name__}:{exc}"]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"])
    apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    snap = {"capture_type": "read_only", "gpu_query": gpu, "compute_apps_query": apps, "dedicated_gpu_gate": "BLOCKED_IF_ANY_RESIDENT_PROCESS"}
    (OUT / "hardware_snapshot.json").write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    hashes = {}
    for p in sorted((ROOT / "results/v1_3").glob("*/PROTOCOL.json")):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    lock_files = {}
    for p in sorted((ROOT / "results/v1_3").rglob("*.lock.json")):
        lock_files[str(p.relative_to(ROOT))] = sha(p)
    receipt = {
        "receipt_id": "decisionrag-v1.3-preflight-receipt-v1.0",
        "status": "PRE_OUTCOME_PROTOCOLS_HASHED",
        "protocol_sha256": hashes,
        "lock_file_sha256": lock_files,
        "labels_joined": False,
        "model_calls": 0,
        "paid_api_calls": 0,
        "protected_holdouts_accessed": False,
    }
    (OUT / "PROTOCOL_HASH_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocols": len(hashes), "locks": len(lock_files), "gpu_lines": len(gpu), "app_lines": len(apps), "receipt": str((OUT / "PROTOCOL_HASH_RECEIPT.json").relative_to(ROOT))}, indent=2))

if __name__ == "__main__":
    main()
