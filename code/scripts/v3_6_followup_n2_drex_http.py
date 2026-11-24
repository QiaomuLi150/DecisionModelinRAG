#!/usr/bin/env python3
"""Label-blind Drex-v2 C360 native HTTP timing branch.

The official local server is already loaded on one physical GPU.  This branch
records the model's native System-One transactions, retains failures and is
explicitly marked partial until same-session Jina/PPLX anchors are available.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
COHORT = ROOT / "results/v3_6_essential/stage0_preflight/cohort_manifest.csv"
OUT = ROOT / "results/v3_6_essential/followup_v1/stage2_n2/session_D_drex_v2"
URL = "http://service.example.invalid/v1/systemone"


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canon(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_views():
    wanted = {r["canonical_id"] for r in csv.DictReader(COHORT.open(encoding="utf-8"))}
    rows = []
    for line in BRIGHT.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        cid = f"{r['domain']}:{r['query_id']}"
        if cid not in wanted:
            continue
        ids = [str(x["candidate_id"]) for x in r["candidates"]]
        criteria = {i: str(x["content"]) for i, x in zip(ids, r["candidates"])}
        request = {
            "state": {"query": str(r["query"])},
            "model": "drex-v2.0",
            "questions": {"candidate": {"type": "choice", "instructions": "Which candidate passage is most relevant to the query?", "criteria": criteria}},
        }
        rows.append({"canonical_id": cid, "domain": r["domain"], "query_id": str(r["query_id"]), "query": str(r["query"]), "ids": ids, "request": request, "slate_sha256": r["slate_sha256"]})
    rows.sort(key=lambda x: x["canonical_id"])
    if len(rows) != 360 or len({x["canonical_id"] for x in rows}) != 360:
        raise RuntimeError(f"C360_INVALID:{len(rows)}")
    return rows


def choose(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def main(args):
    global OUT, URL
    OUT = ROOT / args.out_dir
    URL = args.url
    views = load_views()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "FROZEN_PRE_OUTCOME_PARTIAL", "stage": "N2_Drex_v2_native_C360", "session": "D", "gpu_slot": "D",
        "cohort": "C360", "cohort_size": 360, "attempts": 3, "concurrency": 1,
        "native_interface": "official /v1/systemone typed choice probabilities", "server_url": "loopback-only",
        "model_revision": "3e6d164cb5c8cf264eec167905cc2448d89ea97d", "labels_joined": False,
        "anchor_status": "NOT_IN_SESSION_D_SERVER_OCCUPANCY; same-session Jina/PPLX anchors pending",
        "frozen_bright_sha256": file_sha(BRIGHT), "cohort_sha256": sha(sorted(x["canonical_id"] for x in views)),
        "failure_policy": "no retries; failures retain elapsed time and error class",
    }
    write_json(OUT / "PROTOCOL_N2_D.json", protocol)
    checkpoint = OUT / "raw" / "serving_perrequest.checkpoint.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(x) for x in checkpoint.open(encoding="utf-8")] if args.resume and checkpoint.exists() else []
    done = {(x["canonical_id"], int(x["attempt"])) for x in existing}
    rows = existing
    with checkpoint.open("a", encoding="utf-8") as out:
        for attempt in (1, 2, 3):
            for n, view in enumerate(views, 1):
                if (view["canonical_id"], attempt) in done:
                    continue
                started = time.perf_counter(); status = "FAIL"; error = None; scores = []; selected = []; raw = b""; http_status = None
                body = dict(view["request"]); body["model"] = "drex-v2.0"
                try:
                    req = urllib.request.Request(URL, data=canon(body).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=900) as response:
                        http_status = response.status; raw = response.read()
                    payload = json.loads(raw)
                    probs = payload.get("answers", {}).get("candidate", {}).get("probabilities", {})
                    scores = [float(probs[i]) for i in view["ids"]]
                    if len(scores) != 20 or not all(math.isfinite(x) for x in scores):
                        raise ValueError("INVALID_NATIVE_PROBABILITY_VECTOR")
                    selected = choose(view["ids"], scores)
                    if len(set(selected)) != 5:
                        raise ValueError("INVALID_TOP5_SELECTION")
                    status = "PASS"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                row = {"session": "D", "gpu_slot": "D", "cohort": "C360", "policy": "drex_v2_native", "canonical_id": view["canonical_id"], "domain": view["domain"], "attempt": attempt, "status": status, "error_class": error, "http_status": http_status, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "candidate_ids": view["ids"], "scores": scores, "selected_ids": selected, "slate_sha256": view["slate_sha256"], "native_request_sha256": sha(body), "raw_response_sha256": hashlib.sha256(raw).hexdigest() if raw else None, "model_revision": "3e6d164cb5c8cf264eec167905cc2448d89ea97d", "labels_joined": False, "anchor_status": "NOT_IN_SESSION_D", "timed_boundary": "native HTTP transaction through response parsing and Top-5 packaging"}
                rows.append(row); done.add((view["canonical_id"], attempt)); out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"); out.flush()
            print(json.dumps({"session": "D", "attempt": attempt, "rows": len(rows), "valid": sum(x["status"] == "PASS" for x in rows)}), flush=True)
    raw_path = OUT / "raw" / "serving_perrequest_raw.jsonl"
    raw_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    meta = {"status": "PARTIAL_ANCHORS_MISSING", "records": len(rows), "expected_records": 1080, "valid": sum(x["status"] == "PASS" for x in rows), "failed": sum(x["status"] != "PASS" for x in rows), "sha256": file_sha(raw_path), "gpu_slot": "D", "labels_joined": False}
    write_json(OUT / "TRACE_META.json", meta)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out-dir", default="results/v3_6_essential/followup_v1/stage2_n2/session_D_drex_v2")
    main(ap.parse_args())
