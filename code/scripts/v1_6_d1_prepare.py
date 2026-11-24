#!/usr/bin/env python3
"""Recover and authenticate the exact R3 PPLX per-window inputs for v1.6 D1.

This is CPU-only preparation.  It never opens labels and never invokes a model.
The R3 records retained request hashes and per-stage caps, while stage-2 used
the already-truncated stage-1 survivor strings.  This script reconstructs that
rule from the frozen BRIGHT corpus, verifies every retained hash, and writes a
path-free request manifest for the D1 runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
R3 = ROOT / "results/v1_3/r3_multipass"
R3O = ROOT / "results/v1_3/r3_multipass_ordered/raw"
OUT = ROOT / "results/v1_6/d1_perwindow_jina"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
BRIGHT_SHA = "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2"
ORDERS = ["original", "reversed", "perm_seed_1", "perm_seed_2", "perm_seed_3", "perm_seed_4", "perm_seed_5"]
PPLX_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots") / PPLX_REV


def canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x: Any) -> str:
    return hashlib.sha256(canon(x).encode()).hexdigest()


def fsha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def load_r3() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for p in sorted(R3.glob("multipass_raw_start*_n*.jsonl")):
        for row in read_jsonl(p):
            rows[(row["canonical_id"], "original")] = row
    ordered_files = sorted(R3O.glob("multipass_*.jsonl"))
    for p in ordered_files:
        for row in read_jsonl(p):
            rows[(row["canonical_id"], row["order_id"])] = row
    return rows


def prefix(tok, text: str, cap: int) -> str:
    return tok.decode(tok.encode(str(text), add_special_tokens=False)[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def make_row(item: dict[str, Any], docs: list[dict[str, str]], codes: list[str]) -> dict[str, Any]:
    return {
        "state": f"Query: {item['query']}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {item['query']}",
            "criteria": {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT / "D1_INPUT_MANIFEST.jsonl"))
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    import v1_3_r3_multipass as base
    sys.path.insert(0, str(PPLX_PATH / "source/src"))
    from transformers import AutoProcessor

    if fsha(BRIGHT) != BRIGHT_SHA:
        raise RuntimeError("frozen BRIGHT hash mismatch")
    slates = read_jsonl(BRIGHT)
    slates = sorted(slates, key=lambda x: (x["domain"], str(x["query_id"])))
    by_cid = {f"{x['domain']}:{x['query_id']}": x for x in slates}
    r3 = load_r3()
    if len(by_cid) != 1384 or len(r3) != 1384 * 7:
        raise RuntimeError(f"R3 coverage mismatch slates={len(by_cid)} request_records={len(r3)}")

    proc = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
    tok = proc.tokenizer
    codes = [f"candidate_{i:03d}" for i in range(10)]
    out = []
    mismatches = []
    recoverable = 0
    for n, item in enumerate(slates, 1):
        cid = f"{item['domain']}:{item['query_id']}"
        original_by_id = {str(d["candidate_id"]): str(d["content"]) for d in item["candidates"]}
        for order_id in ORDERS:
            row = r3[(cid, order_id)]
            stages = row.get("stage_views") if order_id != "original" else row.get("stage1_views")
            manifest = {
                "canonical_id": cid,
                "domain": item["domain"],
                "query_id": str(item["query_id"]),
                "order_id": order_id,
                "pplx_status": row.get("status"),
                "pplx_error_classes": [str(e.get("error_class", "UNKNOWN")) for e in row.get("errors", [])],
                "pplx_selected_ids": row.get("selected_ids", []),
                "pplx_stage_selected_ids": row.get("stage_selected_ids", []),
                "source_slate_sha256": item["slate_sha256"],
                "pplx_model_revision": row.get("model_revision", PPLX_REV),
                "stages": [],
                "source_reconstruction": "frozen BRIGHT content + stored R3 final_cap + stage1-survivor carry-forward",
                "labels_joined": False,
                "source_hashes_verified": False,
            }
            stage1_map: dict[str, dict[str, str]] = {}
            local_ok = True
            for stage_no, sv in enumerate(stages or [], 1):
                cap = int(sv["budget"]["final_cap"])
                docs = [{"candidate_id": str(i), "content": prefix(tok, original_by_id[str(i)], cap)} for i in sv["candidate_ids"]]
                got = sha(make_row(item, docs, codes))
                expected = sv["input_sha256"]
                if got != expected:
                    local_ok = False
                    mismatches.append({"canonical_id": cid, "order_id": order_id, "stage": f"stage1_window{stage_no}", "expected": expected, "actual": got})
                for d in docs:
                    stage1_map[d["candidate_id"]] = d
                manifest["stages"].append({
                    "stage": f"stage1_window{stage_no}",
                    "candidate_ids": [d["candidate_id"] for d in docs],
                    "candidate_content_sha256": [hashlib.sha256(d["content"].encode()).hexdigest() for d in docs],
                    "final_cap": cap,
                    "input_sha256": expected,
                    "reconstructed_input_sha256": got,
                    "hash_match": got == expected,
                })
            s2 = row.get("stage2_view")
            if row.get("status") == "PASS" and s2:
                try:
                    survivors = [stage1_map[str(i)] for i in row["stage_selected_ids"]]
                except KeyError as exc:
                    local_ok = False
                    mismatches.append({"canonical_id": cid, "order_id": order_id, "stage": "stage2", "error": f"missing_stage1_survivor:{exc}"})
                    survivors = []
                cap = int(s2["budget"]["final_cap"])
                docs = [{"candidate_id": d["candidate_id"], "content": prefix(tok, d["content"], cap)} for d in survivors]
                got = sha(make_row(item, docs, codes)) if docs else None
                expected = s2["input_sha256"]
                if got != expected:
                    local_ok = False
                    mismatches.append({"canonical_id": cid, "order_id": order_id, "stage": "stage2", "expected": expected, "actual": got})
                manifest["stages"].append({
                    "stage": "stage2_survivors",
                    "candidate_ids": [d["candidate_id"] for d in docs],
                    "candidate_content_sha256": [hashlib.sha256(d["content"].encode()).hexdigest() for d in docs],
                    "final_cap": cap,
                    "input_sha256": expected,
                    "reconstructed_input_sha256": got,
                    "hash_match": got == expected,
                    "pplx_survivor_set": row["stage_selected_ids"],
                })
            manifest["source_hashes_verified"] = local_ok
            recoverable += int(local_ok)
            if not local_ok:
                manifest["pplx_error_classes"].append("R3_INPUT_HASH_MISMATCH")
            out.append(manifest)
        if n % 50 == 0:
            print(json.dumps({"queries": n, "records": len(out), "recoverable": recoverable, "mismatches": len(mismatches)}), flush=True)

    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in out), encoding="utf-8")
    report = {
        "status": "PASS" if not mismatches else "BLOCKED_INPUT_HASH_MISMATCH",
        "records": len(out),
        "queries": len(by_cid),
        "orders": ORDERS,
        "recoverable_records": recoverable,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "bright_sha256": BRIGHT_SHA,
        "r3_model_revision": PPLX_REV,
        "labels_joined": False,
        "manifest_sha256": fsha(op),
    }
    (op.parent / "D1_INPUT_MANIFEST_LOCK.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
