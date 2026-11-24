#!/usr/bin/env python3
"""Prepare the frozen prospective ArguAna validation population.

This script is label-blind. It freezes dataset provenance, official BM25
Top-20 candidate lists, PPLX-tokenizer length strata, and the exact R1
decrement-to-fit request views before any selector inference.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/frozen_prospective_validation"
CACHE = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
ARG = CACHE / "datasets--BeIR--arguana/snapshots/9bcf8fc0320c4e3860e25b8b85fa205fcdd7078f"
QREL = CACHE / "datasets--BeIR--arguana-qrels/snapshots/ae5468c6f1c198109a8af5f0d4dc58bd18b6fea7/test.tsv"
PPLX = CACHE / "models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce"
LENGTHS = (4096, 6144, 8192)
RHO = 128
PROTOCOL_SHA = "06ea012a3bd555454fe6a498ecada087f6affcb62bcab7915fe37a34d62df92c"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_obj(obj: Any) -> str:
    return sha_bytes(canon(obj).encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def make_row(query: str, docs: list[dict[str, str]], codes: list[str]) -> dict[str, Any]:
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


def load_exact_builder():
    src = ROOT / "scripts/v2_1_hybrid_amendment_preflight.py"
    spec = importlib.util.spec_from_file_location("frozen_bright_builder", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen builder: {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.insert(0, str(PPLX / "source/src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(PPLX), local_files_only=True)
    return mod, processor, processor.tokenizer, decision_messages


def exact_token_count(processor, decision_messages, row, codes: list[str]) -> int:
    messages = decision_messages(row, codes)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = processor.tokenizer(text, add_special_tokens=False, return_attention_mask=True)
    return len(encoded["input_ids"])


def prefix(tokenizer, text: str, cap: int) -> str:
    ids = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=cap,
        return_attention_mask=False
    )["input_ids"]
    return tokenizer.decode(ids[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def build_budget(query: str, docs: list[dict[str, str]], L: int, processor, decision_messages, tokenizer):
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    empty = [{"candidate_id": d["candidate_id"], "content": ""} for d in docs]
    overhead = exact_token_count(processor, decision_messages, make_row(query, empty, codes), codes)
    available = L - RHO - overhead
    if available < len(docs):
        return None, {"builder_status": "FAIL", "error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}
    initial = max(1, available // len(docs))
    cap = initial
    reductions = 0
    while cap >= 1:
        candidate_view = [{"candidate_id": d["candidate_id"], "content": prefix(tokenizer, d["content"], cap)} for d in docs]
        total = exact_token_count(processor, decision_messages, make_row(query, candidate_view, codes), codes)
        if total <= L - RHO:
            return candidate_view, {
                "builder_status": "PASS", "overhead_tokens": overhead,
                "effective_content_budget": available, "initial_per_passage_cap": initial,
                "final_per_passage_cap": cap, "deterministic_cap_reductions": reductions,
                "serialized_tokens": total,
            }
        cap -= 1
        reductions += 1
    return None, {"builder_status": "FAIL", "error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}


def official_bm25(corpus_rows, query_rows):
    import os
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    doc_ids = [str(x["_id"]) for x in corpus_rows]
    docs = [str(x.get("text") or "") for x in corpus_rows]
    analyzed = [analyzer.analyze(x) for x in docs]
    dictionary = Dictionary(analyzed)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, analyzed))]
    index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(docs), num_terms=len(dictionary), normalize_queries=False, normalize_documents=False)
    out = {}
    for q in query_rows:
        qid = str(q["_id"])
        toks = analyzer.analyze(str(q.get("text") or ""))
        similarities = index[model[dictionary.doc2bow(toks)]].tolist()
        pairs = [(did, float(score), i) for i, (did, score) in enumerate(zip(doc_ids, similarities))]
        # The frozen BRIGHT implementation uses reverse score sorting with
        # Python's stable tie behavior, preserving original corpus order.
        pairs.sort(key=lambda x: x[1], reverse=True)
        out[qid] = pairs[:20]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "PROTOCOL_FROZEN.md"
    if sha_file(protocol) != PROTOCOL_SHA:
        raise RuntimeError("PROTOCOL_HASH_MISMATCH_BEFORE_PREPARATION")
    if not ARG.exists() or not QREL.exists():
        raise RuntimeError("ARGUANA_SNAPSHOT_OR_QRELS_MISSING")
    import pyarrow.parquet as pq
    corpus_path = ARG / "corpus/corpus-00000-of-00001.parquet"
    queries_path = ARG / "queries/queries-00000-of-00001.parquet"
    corpus_rows = pq.read_table(corpus_path).to_pylist()
    query_rows = pq.read_table(queries_path).to_pylist()
    query_rows.sort(key=lambda x: str(x["_id"]))
    if len(query_rows) < 1000:
        raise RuntimeError(f"QUERY_MINIMUM_NOT_MET:{len(query_rows)}")
    if len({str(x["_id"]) for x in query_rows}) != len(query_rows):
        raise RuntimeError("DUPLICATE_QUERY_IDS")
    if len({str(x["_id"]) for x in corpus_rows}) != len(corpus_rows):
        raise RuntimeError("DUPLICATE_CORPUS_IDS")
    builder, processor, tokenizer, decision_messages = load_exact_builder()
    scored = official_bm25(corpus_rows, query_rows)
    corpus_by_id = {str(x["_id"]): str(x.get("text") or "") for x in corpus_rows}
    slates = []
    for q in query_rows:
        qid = str(q["_id"])
        top = scored[qid]
        if len(top) != 20 or len({x[0] for x in top}) != 20:
            raise RuntimeError(f"INVALID_TOP20:{qid}")
        candidates = [{"candidate_id": did, "content": corpus_by_id[did], "retrieval_score": score, "retrieval_rank": rank + 1, "content_sha256": sha_bytes(corpus_by_id[did].encode("utf-8"))} for rank, (did, score, _idx) in enumerate(top)]
        slates.append({
            "dataset": "arguana", "query_id": qid, "canonical_id": f"arguana:{qid}",
            "query": str(q.get("text") or ""), "query_sha256": sha_bytes(str(q.get("text") or "").encode("utf-8")),
            "candidate_ids": [x["candidate_id"] for x in candidates], "candidates": candidates,
            "retrieval": {"method": "official_bright_bm25", "analyzer": "pyserini Lucene analyzer", "k1": 0.9, "b": 0.4, "candidate_pool_size": 20, "global_corpus": True, "gold_in_input": False},
            "slate_sha256": sha_obj([x["candidate_id"] for x in candidates]),
            "input_sha256": sha_obj({"dataset": "arguana", "query_id": qid, "query": str(q.get("text") or ""), "candidates": [(x["candidate_id"], x["content_sha256"]) for x in candidates], "protocol": "decisionrag-v3.6-frozen-prospective-validation-v1"}),
        })
    slate_path = OUT / "ARGUANA_FROZEN_TOP20_SLATES.jsonl"
    slate_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in slates), encoding="utf-8")
    lengths = []
    for x in slates:
        q_tokens = len(tokenizer.encode(x["query"], add_special_tokens=False))
        passage_lengths = [len(tokenizer.encode(d["content"], add_special_tokens=False)) for d in x["candidates"]]
        lengths.append({"canonical_id": x["canonical_id"], "query_tokens": q_tokens, "passage_lengths": passage_lengths, "mean_passage_tokens": sum(passage_lengths) / len(passage_lengths), "max_passage_tokens": max(passage_lengths)})
    mean_values = sorted(x["mean_passage_tokens"] for x in lengths)
    q1 = mean_values[len(mean_values) // 3]
    q2 = mean_values[(2 * len(mean_values)) // 3]
    for x in lengths:
        x["length_tercile"] = "short" if x["mean_passage_tokens"] <= q1 else ("medium" if x["mean_passage_tokens"] <= q2 else "long")
    (OUT / "ARGUANA_QUERY_LENGTHS.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lengths), encoding="utf-8")
    tercile_counts = {k: sum(x["length_tercile"] == k for x in lengths) for k in ("short", "medium", "long")}
    prepared_summary = []
    for L in LENGTHS:
        prepared = []
        for x in slates:
            docs = [{"candidate_id": d["candidate_id"], "content": d["content"]} for d in x["candidates"]]
            budgeted, info = build_budget(x["query"], docs, L, processor, decision_messages, tokenizer)
            prepared.append({"dataset": "arguana", "canonical_id": x["canonical_id"], "query_id": x["query_id"], "query": x["query"], "candidate_ids": x["candidate_ids"], "candidates": docs, "budgeted_candidates": budgeted or [], "budget_info": info, "serialized_tokens": info.get("serialized_tokens"), "length_limit": L, "safety_margin_tokens": RHO, "source_slate_sha256": x["slate_sha256"], "source_input_sha256": x["input_sha256"], "labels_joined": False, "input_label_blind": True})
        path = OUT / f"prepared_L{L}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in prepared), encoding="utf-8")
        passes = [x for x in prepared if x["budget_info"].get("builder_status") == "PASS"]
        if any(int(x["serialized_tokens"]) > L - RHO for x in passes):
            raise RuntimeError(f"SERIALIZED_TOKEN_LIMIT_VIOLATION:{L}")
        prepared_summary.append({"L": L, "n": len(prepared), "builder_pass_n": len(passes), "builder_failure_n": len(prepared) - len(passes), "mean_serialized_tokens": sum(int(x["serialized_tokens"]) for x in passes) / len(passes) if passes else None, "max_serialized_tokens": max((int(x["serialized_tokens"]) for x in passes), default=None), "sha256": sha_file(path)})
    files = {}
    for p in [slate_path, OUT / "ARGUANA_QUERY_LENGTHS.jsonl", *(OUT / f"prepared_L{L}.jsonl" for L in LENGTHS)]:
        files[str(p.relative_to(ROOT))] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    dataset_files = {
        "arguana_snapshot_ref": {"path": "<cache>/datasets--BeIR--arguana/refs/main", "revision": "9bcf8fc0320c4e3860e25b8b85fa205fcdd7078f", "sha256": sha_file(CACHE / "datasets--BeIR--arguana/refs/main")},
        "corpus": {"path": "<cache>/datasets--BeIR--arguana/snapshots/9bcf8fc0320c4e3860e25b8b85fa205fcdd7078f/corpus/corpus-00000-of-00001.parquet", "sha256": sha_file(corpus_path), "bytes": corpus_path.stat().st_size},
        "queries": {"path": "<cache>/datasets--BeIR--arguana/snapshots/9bcf8fc0320c4e3860e25b8b85fa205fcdd7078f/queries/queries-00000-of-00001.parquet", "sha256": sha_file(queries_path), "bytes": queries_path.stat().st_size},
        "qrels": {"path": "<cache>/datasets--BeIR--arguana-qrels/snapshots/ae5468c6f1c198109a8af5f0d4dc58bd18b6fea7/test.tsv", "sha256": sha_file(QREL), "bytes": QREL.stat().st_size, "used_in_serving": False},
        "dataset_readme": {"path": "<cache>/datasets--BeIR--arguana/snapshots/9bcf8fc0320c4e3860e25b8b85fa205fcdd7078f/README.md", "sha256": sha_file(ARG / "README.md"), "bytes": (ARG / "README.md").stat().st_size},
    }
    (OUT / "DATASET_FILE_MANIFEST.json").write_text(json.dumps(dataset_files, indent=2, sort_keys=True) + "\n")
    admission = {
        "status": "PASS", "selection_rule": "first locally complete/downloadable eligible dataset in frozen order", "tried_order": [
            {"dataset": "CQADupStack", "status": "NOT_LOCALLY_AVAILABLE", "checked_cache": "<cache>/datasets--BeIR--cqadupstack"},
            {"dataset": "ArguAna", "status": "SELECTED", "reason": "complete local corpus, queries and test qrels; 1406 queries >= 1000 minimum"},
            {"dataset": "SciDocs", "status": "NOT_TRIED_AFTER_FIRST_ELIGIBLE"}, {"dataset": "FiQA-2018", "status": "NOT_TRIED_AFTER_FIRST_ELIGIBLE"}, {"dataset": "Natural Questions (BEIR)", "status": "NOT_TRIED_AFTER_FIRST_ELIGIBLE"}],
        "dataset": "BEIR/ArguAna", "excluded_benchmarks": ["2WikiMultiHopQA", "BRIGHT", "CRAG", "EnterpriseRAG-Bench", "FinanceBench", "garage", "HotpotQA", "MultiHop-RAG", "MuSiQue", "QASPER", "SciFact"],
        "corpus_documents": len(corpus_rows), "queries": len(query_rows), "qrels_file_entries_including_header": sum(1 for _ in QREL.open("rb")), "candidate_slate_queries": len(slates), "candidate_pool_size": 20, "sampling_seed": 731, "sampling": "complete 1406-query population; no sampling required", "labels_joined": False, "protocol_sha256": PROTOCOL_SHA,
    }
    (OUT / "DATASET_ADMISSION.json").write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    summary = {"status": "PREPARED_BEFORE_INFERENCE", "protocol_sha256": PROTOCOL_SHA, "slate_count": len(slates), "slate_sha256": sha_file(slate_path), "length_tercile_boundaries_mean_passage_tokens": {"short_upper": q1, "medium_upper": q2}, "length_tercile_counts": tercile_counts, "prepared": prepared_summary, "files": files, "label_blind": True}
    (OUT / "PREPARATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
