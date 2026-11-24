#!/usr/bin/env python3
"""Prepare the v3.6 corrective C1 length scan and label-blind inputs.

This is CPU-only.  It consumes the already-frozen v2.1 hybrid-RRF original
order and never reads qrels or labels.  The b(q) rule is intentionally kept
literal: max(0, floor((L-h(q)-128)/20)).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v2_correction/C1_transfer_confirmation"
MANIFEST = ROOT / "results/v2_1_replication/stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl"
PPLX = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce")
LS = (2048, 4096, 6144, 8192)


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x):
    return hashlib.sha256(canon(x).encode("utf-8")).hexdigest()


def request(query, docs):
    return {"state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
            "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {query}",
                         "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}}}


def truncate(tok, text, cap):
    ids = tok.encode(str(text), add_special_tokens=False)
    return tok.decode(ids[:max(0, cap)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def main():
    sys.path.insert(0, str(PPLX / "source" / "src"))
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(PPLX), local_files_only=True)
    tok = proc.tokenizer
    rows = [json.loads(x) for x in MANIFEST.open(encoding="utf-8") if x.strip()]
    rows = [r for r in rows if str(r.get("order_id")) == "original"]
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["canonical_id"])))
    if len(rows) != 2667 or len({r["canonical_id"] for r in rows}) != 2667:
        raise RuntimeError(f"C1_ORIGINAL_COVERAGE:{len(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    scan_path = OUT / "length_scan.csv"
    scan_fields = ["canonical_id", "dataset", "group_id", "L", "h_q", "b_q", "passage_lengths", "full_input_sha256", "truncated_input_sha256", "hash_identical", "candidate_ids_sha256"]
    scan = []
    prepared = {L: [] for L in LS}
    for r in rows:
        docs = [{"candidate_id": str(d["candidate_id"]), "content": str(d["content"])} for d in r["candidates"]]
        hq = len(tok.encode(str(r["query"]), add_special_tokens=False))
        full_hash = sha(request(str(r["query"]), docs))
        for L in LS:
            b = max(0, math.floor((L - hq - 128) / 20))
            trunc_docs = [{"candidate_id": d["candidate_id"], "content": truncate(tok, d["content"], b)} for d in docs]
            trunc_hash = sha(request(str(r["query"]), trunc_docs))
            scan.append({"canonical_id": r["canonical_id"], "dataset": r["dataset"], "group_id": r.get("group_id", ""), "L": L, "h_q": hq, "b_q": b,
                         "passage_lengths": json.dumps([len(tok.encode(d["content"], add_special_tokens=False)) for d in docs], separators=(",", ":")),
                         "full_input_sha256": full_hash, "truncated_input_sha256": trunc_hash, "hash_identical": int(full_hash == trunc_hash),
                         "candidate_ids_sha256": sha([d["candidate_id"] for d in docs])})
            prepared[L].append({"canonical_id": r["canonical_id"], "dataset": r["dataset"], "query_id": r["query_id"], "group_id": r.get("group_id", ""),
                                "query": r["query"], "order_id": "original", "L": L, "b_q": b, "candidate_ids": [d["candidate_id"] for d in docs],
                                "candidates": trunc_docs, "source_slate_sha256": r["source_slate_sha256"], "full_input_sha256": full_hash,
                                "input_sha256": trunc_hash, "labels_joined": False, "input_label_blind": True})
    with scan_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scan_fields); w.writeheader(); w.writerows(scan)
    for L, vals in prepared.items():
        p = OUT / f"prepared_L{L}.jsonl"
        p.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in vals), encoding="utf-8")
    summary = []
    for L in LS:
        ss = [x for x in scan if int(x["L"]) == L]
        for ds in ["enterprise", "musique", "ALL"]:
            z = ss if ds == "ALL" else [x for x in ss if x["dataset"] == ds]
            summary.append({"L": L, "dataset": ds, "n": len(z), "hash_identical_n": sum(int(x["hash_identical"]) for x in z),
                            "share_hash_identical": sum(int(x["hash_identical"]) for x in z) / len(z), "skip_99pct": int(sum(int(x["hash_identical"]) for x in z) / len(z) >= .99)})
    (OUT / "length_scan_summary.csv").write_text("L,dataset,n,hash_identical_n,share_hash_identical,skip_99pct\n" + "\n".join(",".join(str(x[k]) for k in ["L","dataset","n","hash_identical_n","share_hash_identical","skip_99pct"]) for x in summary) + "\n", encoding="utf-8")
    receipt = {"status": "PREPARED", "manifest": str(MANIFEST.relative_to(ROOT)), "original_rows": len(rows), "datasets": {"enterprise": 500, "musique": 2167}, "Ls": list(LS), "formula": "max(0,floor((L-h(q)-128)/20))", "label_blind": True, "scan_rows": len(scan), "prepared_files": {str(L): str((OUT/f'prepared_L{L}.jsonl').relative_to(ROOT)) for L in LS}}
    (OUT / "C1_PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREPARED", "rows": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
