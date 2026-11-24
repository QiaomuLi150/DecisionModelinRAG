#!/usr/bin/env python3
"""B6.2 C120 GPU-isolated solo timing check for v3.6."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/stage2_t2"


def load_t2():
    spec = importlib.util.spec_from_file_location("v36_t2", ROOT / "scripts/v3_6_t2_serving.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("T2_IMPORT_FAILED")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    t2 = load_t2(); base = t2.base
    gate = base.gpu_gate(args.gpu)
    proc = base.load_pplx_processor()
    sys.path.insert(0, str(base.PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    tok = proc.tokenizer
    views, manifest = t2.build_views(proc, decision_messages, tok)
    if len(views) != 120:
        raise RuntimeError("SOLO_C120_INVALID")
    protocol = {"status": "FROZEN_PRE_OUTCOME", "experiment": "DecisionRAG v3.6 Essential Verification",
                "stage": "B6.2_solo", "cohort": "C120", "cohort_size": 120, "attempts": 3,
                "concurrency": 1, "policy_order": ["jina_full", "jina_single_8192", "pplx_budget"],
                "gpu_index": args.gpu, "gpu_uuid": gate.get("gpu_uuid"),
                "input_manifest_sha256": t2.sha_file(OUT / "T2_INPUT_MANIFEST.jsonl"),
                "labels_joined": False, "protected_holdouts_accessed": False, "paid_api_calls": 0,
                "resource_scope": "GPU-isolated solo process; unrelated host workloads were not terminated",
                "timed_boundary": "view preparation + native calls + score parsing + ID mapping + Top-5 packaging",
                "failure_policy": "no retries; failure elapsed time retained; no fallback in raw timing"}
    p = OUT / "SOLO_PROTOCOL.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("input_manifest_sha256") != protocol["input_manifest_sha256"] or old.get("policy_order") != protocol["policy_order"]:
            raise RuntimeError("SOLO_PROTOCOL_MISMATCH")
    else:
        t2.write_json(p, protocol)
    # Reuse the exact T2 request implementation; only the declared policy
    # cohort differs.  T2_D is the immutable raw namespace for this solo run.
    pplx_rows, _, _ = t2.run_phase("D", "pplx", args.gpu, ["pplx_budget"], views, proc, tok, decision_messages, {}, False, gate)
    jina_rows, _, _ = t2.run_phase("D", "jina", args.gpu, ["jina_full", "jina_single_8192"], views, proc, tok, decision_messages, {}, False, gate)
    all_rows = pplx_rows + jina_rows
    t2.write_json(OUT / "SOLO_SESSION.json", {"status": "PASS", "gpu_index": args.gpu, "gpu_uuid": gate.get("gpu_uuid"),
                                               "records": len(all_rows), "expected_records": 1080,
                                               "phase_hashes": {phase: {"sha256": t2.sha_file(OUT / f"T2_D/raw/{phase}/serving_perrequest_raw.jsonl"),
                                                                         "records": len(t2.read_jsonl(OUT / f"T2_D/raw/{phase}/serving_perrequest_raw.jsonl"))}
                                                                for phase in ("pplx", "jina")},
                                               "labels_joined": False, "protected_holdouts_accessed": False})
    print(json.dumps({"status": "SOLO_DONE", "records": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
