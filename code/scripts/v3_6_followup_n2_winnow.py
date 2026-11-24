#!/usr/bin/env python3
"""Label-blind C360 Winnow native timing client for the v3.6 N2 branch.

The server is launched separately with the pinned local Winnow runtime. This
client preserves every HTTP failure and never retries or joins labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v1/stage2_n2/session_E_winnow"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
COHORT = ROOT / "results/v3_6_essential/stage0_preflight/cohort_manifest.csv"


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def read_jsonl(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://service.example.invalid/v1/systemone"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--gpu", type=int, default=4)
    a = ap.parse_args()
    wanted = {r["canonical_id"] for r in csv.DictReader((COHORT).open(encoding="utf-8"))}
    slates = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BRIGHT)}
    views = [r for k, r in sorted(slates.items()) if k in wanted]
    if len(views) != 360:
        raise RuntimeError(f"C360_INVALID:{len(views)}")
    raw = OUT / "raw" / "serving_perrequest.checkpoint.jsonl"; raw.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(raw) if a.resume and raw.exists() else []
    done = {(x.get("policy"), x.get("canonical_id"), int(x.get("attempt", 0))) for x in existing}
    fh = raw.open("a", encoding="utf-8")
    for attempt in (1, 2, 3):
        for v in views:
            key = (f"{v['domain']}:{v['query_id']}", attempt)
            if ("winnow_native", key[0], key[1]) in done:
                continue
            ids = [str(x["candidate_id"]) for x in v["candidates"]]
            req = {"model": "Winnow-12B", "state": {"query": str(v["query"])}, "questions": {"candidate": {"type": "choice", "instructions": "Which candidate passage is most relevant to the query?", "criteria": {str(x["candidate_id"]): str(x["content"]) for x in v["candidates"]}}}}
            started = time.perf_counter(); status = "FAIL"; error = None; scores = []; selected = []; response_bytes = b""
            try:
                request = urllib.request.Request(a.url, data=canonical(req).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=300) as resp:
                    response_bytes = resp.read(); obj = json.loads(response_bytes)
                probs = (obj.get("answers") or {}).get("candidate", {}).get("probabilities", {})
                scores = [float(probs[x]) for x in ids]
                order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
                selected = [ids[i] for i in order[:5]]
                if len(selected) != 5 or len(set(selected)) != 5:
                    raise ValueError("INVALID_TOP5_SELECTION")
                status = "PASS"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            row = {"session": "E", "gpu_slot": "D", "cohort": "C360", "policy": "winnow_native", "canonical_id": key[0], "domain": str(v["domain"]), "attempt": attempt, "status": status, "error_class": error, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "alloc_peak_mib": None, "model_revision": "b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea", "candidate_ids": ids, "selected_ids": selected, "scores": scores, "native_request_sha256": sha(req), "raw_response_sha256": hashlib.sha256(response_bytes).hexdigest() if response_bytes else None, "labels_joined": False, "input_label_blind": True, "anchor_status": "ANCHORS_RUN_SEPARATELY_SAME_GPU_SESSION_E_PENDING", "timed_boundary": "native HTTP transaction through response parsing and Top-5 packaging"}
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"); fh.flush(); done.add(("winnow_native", key[0], key[1]))
        print(json.dumps({"session": "E", "policy": "winnow_native", "attempt": attempt, "rows": sum(1 for x in existing if x.get("policy") == "winnow_native") + sum(1 for x in done if x[0] == "winnow_native"), "valid": sum(x.get("status") == "PASS" for x in existing if x.get("policy") == "winnow_native")}), flush=True)
    fh.close()
    rows = read_jsonl(raw)
    (OUT / "TRACE_META.json").write_text(json.dumps({"status": "PASS" if len(rows) == 1080 else "INCOMPLETE", "records": len(rows), "expected_records": 1080, "valid": sum(x.get("status") == "PASS" for x in rows), "failed": sum(x.get("status") != "PASS" for x in rows), "labels_joined": False, "gpu_slot": "D"}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
