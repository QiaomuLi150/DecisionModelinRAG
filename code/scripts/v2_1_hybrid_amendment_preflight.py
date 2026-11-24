#!/usr/bin/env python3
"""Freeze the explicitly approved hybrid-RRF amendment before selector calls.

The original Stage-0 gate remains immutable.  This script creates a new
versioned input contract from the already-frozen historical hybrid-RRF Top-20
slates, without rerunning retrieval or reading labels into selector inputs.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "v2_1_replication"
OUT = BASE / "stage0_preflight_v2_hybrid"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
L = 8192
RHO = 128
ORDERS = ["original", "reversed", "perm_seed_20261020", "perm_seed_20261021"]

ENT_Q = ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"
ENT_G = ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl"
ENT_S = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_RETRIEVAL_SLATES.jsonl"
ENT_C = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_CHUNK_MANIFEST.jsonl"
ENT_ZIP = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0/all_documents.zip"
MUS_Q = ROOT / "data/raw/vendor/musique/musique_ans_v1.0_dev.jsonl"
MUS_IDS = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"
MUS_S = ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_RETRIEVAL_SLATES.jsonl"
CORPUS = ROOT / "data/corpus/passages.jsonl"


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def sha_bytes(x: bytes):
    return hashlib.sha256(x).hexdigest()


def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def utc():
    return datetime.now(timezone.utc).isoformat()


def order_docs(docs, order_id):
    if order_id == "original":
        return list(docs)
    if order_id == "reversed":
        return list(reversed(docs))
    out = list(docs)
    random.Random(int(order_id.rsplit("_", 1)[1])).shuffle(out)
    return out


def make_row(query, docs, codes):
    return {
        "state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {query}",
            "criteria": {
                codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}"
                for i, d in enumerate(docs)
            },
        },
    }


def token_count(processor, decision_messages, row, codes):
    messages = decision_messages(row, codes)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # The frozen requests are text-only.  Calling the tokenizer directly is
    # equivalent to the processor's text branch and avoids constructing a
    # tensor for every CPU-only manifest row.
    encoded = processor.tokenizer(text, add_special_tokens=False, return_attention_mask=True)
    return len(encoded["input_ids"])


def prefix(tokenizer, text, cap):
    ids = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=cap,
        return_attention_mask=False
    )["input_ids"]
    return tokenizer.decode(ids[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def budget_docs(query, docs, processor, decision_messages, tokenizer):
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    empty = [{"candidate_id": d["candidate_id"], "content": ""} for d in docs]
    overhead = token_count(processor, decision_messages, make_row(query, empty, codes), codes)
    available = L - RHO - overhead
    k = len(docs)
    if available < k:
        return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}
    initial = max(1, available // k)
    cap = initial
    reductions = 0
    while cap >= 1:
        budgeted = [{"candidate_id": d["candidate_id"], "content": prefix(tokenizer, d["content"], cap)} for d in docs]
        total = token_count(processor, decision_messages, make_row(query, budgeted, codes), codes)
        if total <= L - RHO:
            return budgeted, {
                "overhead_tokens": overhead,
                "effective_content_budget": available,
                "initial_per_passage_cap": initial,
                "final_per_passage_cap": cap,
                "deterministic_cap_reductions": reductions,
                "serialized_tokens": total,
            }
        cap -= 1
        reductions += 1
    return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}


def cached_budget(query, docs, processor, decision_messages, tokenizer, cache):
    """Budget a candidate set once; reuse its exact per-candidate strings.

    The equal-prefix policy depends on the query and candidate set, not on the
    order in which the same candidates are presented. The returned strings are
    reordered to the caller's frozen order before request hashing.
    """
    key = (query, tuple(sorted(d["candidate_id"] for d in docs)))
    if key not in cache:
        canonical_docs = sorted(docs, key=lambda d: d["candidate_id"])
        budgeted, info = budget_docs(query, canonical_docs, processor, decision_messages, tokenizer)
        cache[key] = (budgeted, info)
    budgeted, info = cache[key]
    if budgeted is None:
        return None, info
    by_id = {d["candidate_id"]: d for d in budgeted}
    return [{"candidate_id": d["candidate_id"], "content": by_id[d["candidate_id"]]["content"]} for d in docs], info


def load_pplx_cpu():
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
    return processor, processor.tokenizer, decision_messages


def enterprise_rows():
    sys.path.insert(0, str(ROOT / "scripts"))
    from v11_track_b_index import make_chunks, normalize_document

    questions = {x["question_id"]: x for x in read_jsonl(ENT_Q)}
    slates = read_jsonl(ENT_S)
    chunks = {x["internal_doc_id"]: x for x in read_jsonl(ENT_C)}
    needed_paths = {
        chunks[cid]["archive_path"]
        for slate in slates
        for cid in slate["candidate_ids"]
    }
    archive_by_path = {}
    with zipfile.ZipFile(ENT_ZIP) as zf:
        # Walk the central directory once and decompress only referenced
        # members. Repeated ZipFile.read(name) calls rescan a very large
        # archive and can look like a hung process.
        found = set()
        for info in zf.infolist():
            name = info.filename
            if name not in needed_paths:
                continue
            archive_by_path[name] = zf.read(info).decode("utf-8", errors="strict")
            found.add(name)
        missing = sorted(needed_paths - found)
        if missing:
            raise RuntimeError(f"missing archive members: {len(missing)}; first={missing[:3]}")
    rows = []
    content_checks = []
    for slate in sorted(slates, key=lambda x: x["query_id"]):
        qid = slate["query_id"]
        docs = []
        for cid in slate["candidate_ids"]:
            chunk = chunks[cid]
            path = chunk["archive_path"]
            content = archive_by_path.get(path)
            if content is None:
                raise RuntimeError(f"missing archive member for {cid}: {path}")
            _, body = normalize_document(content.encode("utf-8"))
            chunk_index = int(str(chunk["chunk_id"]).rsplit("#chunk-", 1)[1])
            chunks_for_doc = make_chunks(body)
            if chunk_index >= len(chunks_for_doc):
                raise RuntimeError(f"missing normalized chunk {chunk_index} for {cid}")
            normalized_chunk = chunks_for_doc[chunk_index][1]
            observed = sha_bytes(normalized_chunk.encode("utf-8"))
            if observed != chunk["text_sha256"]:
                raise RuntimeError(f"archive text hash mismatch for {cid}: {observed} != {chunk['text_sha256']}")
            docs.append({"candidate_id": cid, "content": normalized_chunk, "archive_path": path, "source_type": chunk.get("source_type")})
            content_checks.append({"candidate_id": cid, "content_sha256": observed})
        rows.append({
            "dataset": "enterprise",
            "canonical_id": qid,
            "query_id": qid,
            "query": questions[qid]["question"],
            "group_id": "enterprise_source:" + qid,
            "source_slate_sha256": sha(slate["candidate_ids"]),
            "candidate_ids": slate["candidate_ids"],
            "candidates": docs,
        })
    return rows, content_checks


def musique_rows():
    selected = {x["source_record_id"] for x in read_jsonl(MUS_IDS) if x.get("dataset") == "musique"}
    dev = {x["id"]: x for x in read_jsonl(MUS_Q)}
    corpus = {x["doc_id"]: x for x in read_jsonl(CORPUS)}
    slates = [x for x in read_jsonl(MUS_S) if x.get("dataset") == "musique" and x["query_id"].split(":", 1)[1] in selected]
    rows = []
    for slate in sorted(slates, key=lambda x: x["query_id"]):
        rid = slate["query_id"].split(":", 1)[1]
        rec = dev[rid]
        docs = []
        for cid in slate["candidate_ids"]:
            c = corpus[cid]
            docs.append({"candidate_id": cid, "content": c["full_text"], "title": c.get("title", "")})
        rows.append({
            "dataset": "musique",
            "canonical_id": slate["query_id"],
            "query_id": rid,
            "query": rec["question"],
            "group_id": "musique_source:" + rid,
            "source_slate_sha256": sha(slate["candidate_ids"]),
            "candidate_ids": slate["candidate_ids"],
            "candidates": docs,
        })
    if len(rows) != len(selected):
        raise RuntimeError(f"MuSiQue slate count mismatch: {len(rows)} != {len(selected)}")
    return rows, []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    processor, tokenizer, decision_messages = load_pplx_cpu()
    enterprise, enterprise_checks = enterprise_rows()
    print(json.dumps({"phase": "enterprise_text_resolved", "rows": len(enterprise), "content_checks": len(enterprise_checks)}), flush=True)
    musique, _ = musique_rows()
    print(json.dumps({"phase": "musique_text_resolved", "rows": len(musique)}), flush=True)
    all_rows = enterprise + musique
    manifest_rows = []
    order_rows = []
    budget_cache = {}
    for item in all_rows:
        for order_id in ORDERS:
            docs = order_docs(item["candidates"], order_id)
            ordered = [{"candidate_id": d["candidate_id"], "content": d["content"]} for d in docs]
            codes20 = [f"candidate_{i:03d}" for i in range(20)]
            budgeted, binfo = cached_budget(item["query"], ordered, processor, decision_messages, tokenizer, budget_cache)
            win_a, win_a_info = cached_budget(item["query"], ordered[:10], processor, decision_messages, tokenizer, budget_cache)
            win_b, win_b_info = cached_budget(item["query"], ordered[10:], processor, decision_messages, tokenizer, budget_cache)
            full_req = make_row(item["query"], ordered, codes20)
            budget_req = make_row(item["query"], budgeted or [], codes20) if budgeted else None
            manifest_rows.append({
                "dataset": item["dataset"], "canonical_id": item["canonical_id"], "query_id": item["query_id"],
                "group_id": item["group_id"], "order_id": order_id, "query": item["query"],
                "source_slate_sha256": item["source_slate_sha256"], "candidate_ids": [d["candidate_id"] for d in ordered],
                "candidates": ordered, "candidate_content_sha256": [sha_bytes(d["content"].encode()) for d in ordered],
                "native_request_sha256": sha(full_req), "native_input_tokens": token_count(processor, decision_messages, full_req, codes20) if order_id == "original" else None,
                "budgeted_candidates": budgeted, "budget_info": binfo, "budgeted_request_sha256": sha(budget_req) if budget_req else None,
                "budgeted_input_tokens": token_count(processor, decision_messages, budget_req, codes20) if budget_req and order_id == "original" else None,
                "window_a_candidates": win_a, "window_a_info": win_a_info,
                "window_b_candidates": win_b, "window_b_info": win_b_info,
                "window_a_request_sha256": sha(make_row(item["query"], win_a, [f"candidate_{i:03d}" for i in range(10)])) if win_a else None,
                "window_b_request_sha256": sha(make_row(item["query"], win_b, [f"candidate_{i:03d}" for i in range(10)])) if win_b else None,
                "label_blind": True,
            })
            if len(manifest_rows) % 100 == 0:
                print(json.dumps({"phase": "input_manifest_progress", "rows": len(manifest_rows)}), flush=True)
    manifest_rows.sort(key=lambda x: (x["dataset"], x["canonical_id"], ORDERS.index(x["order_id"])))
    manifest_path = OUT / "INPUT_MANIFEST_V2_HYBRID.jsonl"
    manifest_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in manifest_rows), encoding="utf-8")
    input_hash = sha_file(manifest_path)
    source_paths = [ENT_Q, ENT_G, ENT_S, ENT_C, ENT_ZIP, MUS_Q, MUS_IDS, MUS_S, CORPUS]
    protocol = {
        "protocol_id": "decisionrag-v2.1-controlled-replication",
        "protocol_version": "V2-HYBRID-RRF-AMENDMENT-1",
        "amendment_status": "OWNER_APPROVED_VERSIONED_INPUT_CONTRACT",
        "amendment_reason": "Owner approved use of the existing preserved hybrid-RRF Top-20 slates because exact BM25-only Top-20 slates are unavailable; original blocked Stage 0 remains immutable.",
        "candidate_contract": "existing historical hybrid BM25-top100 + dense-top100 + RRF@60 Top-20 slates; exact original RRF ordering and archived source text; no retrieval rerun",
        "source_text_contract": "Enterprise archive member UTF-8 bytes mapped by path-qualified internal_doc_id; MuSiQue passages.jsonl full_text",
        "models": {"pplx_decider": {"revision": PPLX_REV, "native_contract": "frozen PPLX two-stage and single-choice interfaces"}, "jina_v3_5": {"revision": JINA_REV, "native_contract": "frozen Jina listwise rerank interface"}},
        "populations": {"enterprise_requests": len(enterprise), "enterprise_quality_labeled": 443, "musique_requests": len(musique)},
        "orders": ORDERS,
        "seeds": {"bootstrap": 731, "shuffle_1": 20261020, "shuffle_2": 20261021, "determinism": 20261022, "cohort_enterprise": 20261020, "cohort_musique": 20261021},
        "context": {"max_native_context_tokens": L, "safety_margin_tokens": RHO, "budget_policy": "equal-prefix per candidate, decrement cap until exact serialized request <= L-rho"},
        "policies": ["P0_bm25_order", "P1_jina_full", "P2_jina_budget", "P3_jina_two_stage_matched", "P4_pplx_native", "P5_pplx_budget", "P6_pplx_two_stage"],
        "stage2_text_rule": "Each selector chooses its own five survivors per 10-candidate window using the shared PPLX-generated window strings; final stage uses the PPLX-generated strings for the selected survivors, preserving candidate IDs and original order tie breaks.",
        "label_blind_inference": True,
        "labels_joined_after_raw_trace_freeze": True,
        "no_retries_or_fill_in": True,
        "input_manifest": "stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl",
        "input_manifest_sha256": input_hash,
        "source_hashes": {str(p.relative_to(ROOT)): sha_file(p) for p in source_paths},
        "fixed_before_outcomes": True,
        "created_utc": utc(),
    }
    protocol_path = BASE / "PROTOCOL_V2.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol_hash = sha_file(protocol_path)
    (BASE / "PROTOCOL_V2_SHA256.json").write_text(json.dumps({"path": "PROTOCOL_V2.json", "sha256": protocol_hash, "frozen_before_model_calls": True}, indent=2) + "\n", encoding="utf-8")
    (OUT / "INPUT_MANIFEST_SHA256.json").write_text(json.dumps({"path": "stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl", "sha256": input_hash, "rows": len(manifest_rows), "labels_included": False}, indent=2) + "\n")
    (OUT / "AMENDMENT_RECEIPT.json").write_text(json.dumps({"status": "FROZEN", "amendment": protocol["amendment_status"], "original_protocol_preserved": True, "protocol_v2_sha256": protocol_hash, "input_manifest_sha256": input_hash, "enterprise_rows": len(enterprise), "musique_rows": len(musique), "archive_text_hash_checks": len(enterprise_checks), "created_utc": utc()}, indent=2) + "\n")
    (OUT / "AMENDMENT.md").write_text("""# DecisionRAG v2.1 versioned input amendment\n\nThe owner approved a versioned amendment allowing the already preserved hybrid BM25-top100 + dense-top100 + RRF@60 Top-20 slates. No new retrieval was run. The original Stage-0 blocker and protocol remain unchanged. Enterprise candidate text was reconstructed from the official archive using path-qualified internal IDs and verified against the chunk-manifest text hashes. MuSiQue candidate text was resolved from the pinned passages corpus. Labels are not in the input manifest and are joined only after raw selector outputs are frozen.\n\nThis amendment changes the input contract, not models, prompts, policies, scoring, sampling, or statistical procedures.\n""", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "protocol_v2_sha256": protocol_hash, "input_manifest_sha256": input_hash, "rows": len(manifest_rows), "enterprise": len(enterprise), "musique": len(musique)}, indent=2))


if __name__ == "__main__":
    main()
