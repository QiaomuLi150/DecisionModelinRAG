#!/usr/bin/env python3
"""Contract-corrected, append-only Jina repair for v3.6."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v2_contract_repair"
RAW = OUT / "raw"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
E11_BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
HIST = ROOT / "results/v1_5/c1_two_stage_specialist_gpu3/raw/jina_pplx_budget__original.jsonl"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots") / JINA_REV


def canonical(x: Any) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_json(x: Any) -> str:
    return hashlib.sha256(canonical(x)).hexdigest()


def sha_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def gpu_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=20,
        )
        rows = []
        for line in out.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) == 5:
                rows.append({"index": vals[0], "uuid": vals[1], "memory_used_mib": vals[2], "memory_free_mib": vals[3], "utilization_percent": vals[4]})
        return {"status": "PASS", "gpus": rows}
    except Exception as exc:
        return {"status": "UNOBSERVABLE", "error_class": type(exc).__name__, "error": str(exc)}


def load_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bright = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BRIGHT)}
    budget = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(E11_BUDGET)}
    hist = {str(r["canonical_id"]): r for r in read_jsonl(HIST)}
    if (len(bright), len(budget), len(hist)) != (1384, 1384, 1384):
        raise RuntimeError(f"unexpected population sizes: {len(bright)},{len(budget)},{len(hist)}")
    return bright, budget, hist


def make_manifest() -> list[dict[str, Any]]:
    bright, budget, hist = load_maps()
    rows = []
    for cid in sorted(bright):
        b, h, q = budget[cid], hist[cid], bright[cid]
        docs = b.get("budgeted_candidates") or []
        ids = [str(x["candidate_id"]) for x in docs]
        texts = [str(x["content"]) for x in docs]
        request_sha = sha_json({"query": str(q["query"]), "candidate_ids": ids, "candidate_texts": texts}) if docs else None
        old_stage = (h.get("stage_records") or [{}])[0]
        old_sha = old_stage.get("request_sha256")
        old_pass = h.get("status") == "PASS" and bool(h.get("valid"))
        ids_match = ids == [str(x) for x in h.get("candidate_ids", [])]
        hash_match = bool(old_sha and request_sha and old_sha == request_sha)
        ready = bool(old_pass and ids_match and hash_match and len(ids) == 20)
        rows.append({
            "canonical_id": cid, "domain": str(q["domain"]), "query_id": str(q["query_id"]), "query": str(q["query"]),
            "candidate_ids": ids, "candidate_texts": texts,
            "candidate_text_sha256": [sha_bytes(t.encode("utf-8")) for t in texts],
            "request_sha256": request_sha, "historical_request_sha256": old_sha,
            "historical_status": h.get("status"), "historical_valid": bool(h.get("valid")),
            "historical_selected_ids": h.get("selected_ids", []), "historical_scores_sha256": old_stage.get("scores_sha256"),
            "historical_candidate_ids_sha256": old_stage.get("candidate_ids_sha256"),
            "ids_match": ids_match, "request_hash_match": hash_match, "eligible_for_native_replay": ready,
            "failure_preserved_without_retry": not ready,
            "source_contract": "v1.5 Jina: query + candidate_ids + candidate_texts",
            "source_budget_manifest": str(E11_BUDGET.relative_to(ROOT)), "source_historical_trace": str(HIST.relative_to(ROOT)),
        })
    return rows


def preflight() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_manifest()
    manifest = OUT / "JINA_CONTRACT_REPAIR_INPUT_MANIFEST.jsonl"
    write_jsonl(manifest, rows)
    counts = {
        "population": len(rows),
        "historical_pass": sum(r["historical_status"] == "PASS" and r["historical_valid"] for r in rows),
        "historical_fail": sum(not (r["historical_status"] == "PASS" and r["historical_valid"]) for r in rows),
        "candidate_id_exact": sum(r["ids_match"] for r in rows),
        "historical_request_hash_exact": sum(r["request_hash_match"] for r in rows),
        "native_replay_eligible": sum(r["eligible_for_native_replay"] for r in rows),
    }
    receipt = {
        "status": "PASS" if counts["native_replay_eligible"] == 1352 and counts["historical_fail"] == 32 else "BLOCKED",
        "repair_reason": "v3.6 used documents; historical Jina uses candidate_texts",
        "population_counts": counts, "jina_revision": JINA_REV,
        "bright_sha256": sha_file(BRIGHT), "e11_budget_manifest_sha256": sha_file(E11_BUDGET),
        "historical_jina_trace_sha256": sha_file(HIST), "input_manifest_sha256": sha_file(manifest),
        "labels_joined": False, "protected_holdouts_accessed": False, "native_inference_executed": False,
        "gpu_snapshot": gpu_snapshot(),
    }
    write_json(OUT / "CONTRACT_REPAIR_PREFLIGHT.json", receipt)
    report = "\n".join([
        "# Jina Contract Repair Preflight", "", f"Status: **{receipt['status']}**", "",
        "The historical request hash is over `query + candidate_ids + candidate_texts`. The v3.6 rerun used `documents`, so its exact-input gate was invalid.", "",
        f"- Population: {counts['population']}", f"- Historical PASS: {counts['historical_pass']}",
        f"- Historical FAIL retained without retry: {counts['historical_fail']}", f"- Candidate-ID exact: {counts['candidate_id_exact']}",
        f"- Historical request-hash exact: {counts['historical_request_hash_exact']}", f"- Native replay eligible: {counts['native_replay_eligible']}",
        "- Labels joined: NO", "- Protected holdouts accessed: NO", "- Native inference in preflight: NO", "",
        "The manifest freezes exact E11 budgeted texts. Historical failures are preserved and not retried.", "",
        f"GPU snapshot status: `{receipt['gpu_snapshot']['status']}`.", "",
    ])
    (OUT / "CONTRACT_REPAIR_PREFLIGHT.md").write_text(report, encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


def stable_top(ids: list[str], scores: list[float]) -> list[str]:
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def run_native(gpu: int) -> int:
    receipt = json.loads((OUT / "CONTRACT_REPAIR_PREFLIGHT.json").read_text())
    if receipt.get("status") != "PASS":
        raise RuntimeError("contract repair preflight is not PASS")
    rows = read_jsonl(OUT / "JINA_CONTRACT_REPAIR_INPUT_MANIFEST.jsonl")
    protocol = {
        "protocol_id": "v3.6_jina_contract_repair_v1", "jina_revision": JINA_REV,
        "input_contract": {"fields": ["query", "candidate_ids", "candidate_texts"], "serialization": "json sort_keys=true separators=(',', ':') ensure_ascii=false"},
        "input_manifest_sha256": sha_file(OUT / "JINA_CONTRACT_REPAIR_INPUT_MANIFEST.jsonl"),
        "historical_trace_sha256": receipt["historical_jina_trace_sha256"], "no_retries": True,
        "labels_joined": False, "protected_holdouts_accessed": False,
        "selection": "stable descending score, then original index, then candidate_id; k=5",
        "gpu_requested": gpu, "frozen_before_inference": True,
    }
    write_json(OUT / "PROTOCOL.json", protocol)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    out_rows = []
    for item in rows:
        rec = {"canonical_id": item["canonical_id"], "domain": item["domain"], "query_id": item["query_id"], "policy": "jina_single_8192_contract_repair", "order_id": "original", "candidate_ids": item["candidate_ids"], "request_sha256": item["request_sha256"], "historical_request_sha256": item["historical_request_sha256"], "input_contract": "candidate_texts", "model_revision": JINA_REV, "labels_joined": False, "protected_holdouts_accessed": False, "native_forward_count": 0, "status": "FAIL", "valid": False, "selected_ids": [], "scores": [], "error_class": None, "error": None}
        if not item["eligible_for_native_replay"]:
            rec.update({"error_class": "HISTORICAL_SOURCE_FAILURE", "error": "preserved historical failure; no retry", "replayed_historical_failure": True})
            out_rows.append(rec)
            continue
        start = time.perf_counter()
        try:
            result = model.rerank(item["query"], item["candidate_texts"])
            if result is None or not isinstance(result, list):
                raise RuntimeError("JINA_RERANK_NONE_OR_NONLIST_OUTPUT")
            mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
            if set(mapping) != set(range(len(item["candidate_texts"]))):
                raise ValueError("JINA_INDEX_COVERAGE_INVALID")
            scores = [mapping[i] for i in range(len(item["candidate_texts"]))]
            selected = stable_top(item["candidate_ids"], scores)
            rec.update({"status": "PASS", "valid": len(selected) == 5 and len(set(selected)) == 5, "selected_ids": selected, "scores": scores, "native_forward_count": 1, "latency_ms": round((time.perf_counter() - start) * 1000, 3), "selected_ids_exact_match": selected == item["historical_selected_ids"], "selected_set_exact_match": set(selected) == set(item["historical_selected_ids"])})
        except Exception as exc:
            rec.update({"error_class": type(exc).__name__, "error": str(exc), "latency_ms": round((time.perf_counter() - start) * 1000, 3)})
        out_rows.append(rec)
    out = RAW / f"jina_contract_repair_gpu{gpu}.jsonl"
    write_jsonl(out, out_rows)
    write_json(out.with_suffix(".meta.json"), {"status": "PASS", "rows": len(out_rows), "physical_native_rows": sum(r["native_forward_count"] for r in out_rows), "sha256": sha_file(out), "protocol_sha256": sha_file(OUT / "PROTOCOL.json")})
    print(json.dumps({"status": "PASS", "output": str(out.relative_to(ROOT)), "rows": len(out_rows)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["preflight", "run"])
    ap.add_argument("--gpu", type=int)
    args = ap.parse_args()
    if args.action == "preflight":
        return preflight()
    if args.gpu is None:
        raise SystemExit("--gpu is required for run")
    return run_native(args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
