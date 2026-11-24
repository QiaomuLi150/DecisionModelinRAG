#!/usr/bin/env python3
"""Freeze CQADupStack v2 candidate lists and exact PPLX builder views."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/frozen_prospective_validation_v2"
DATA = ROOT / "data/external/v3_6_cqadupstack/release/cqadupstack"
PPLX = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce")
PROTOCOL_SHA = "f15cb0f6001327bfe47e0b1d605f73f7beefc3c6a5fc195689cd5ca1652ce86e"
FORUMS = ["android", "english", "gaming", "gis", "mathematica", "physics", "programmers", "stats", "tex", "unix", "webmasters", "wordpress"]
LENGTHS = (4096, 6144, 8192)
RHO = 128


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_obj(x: Any) -> str:
    return sha_bytes(canon(x).encode("utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_builder():
    src = ROOT / "scripts/v2_1_hybrid_amendment_preflight.py"
    spec = importlib.util.spec_from_file_location("frozen_builder_v2", src)
    if spec is None or spec.loader is None:
        raise RuntimeError("BUILDER_IMPORT_FAILED")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.insert(0, str(PPLX / "source/src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(PPLX), local_files_only=True)
    return mod, proc, proc.tokenizer, decision_messages


def make_row(q: str, docs: list[dict[str, str]], codes: list[str]):
    return {"state": f"Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {q}", "criteria": {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}}}


def token_count(proc, decision_messages, row, codes):
    messages = decision_messages(row, codes)
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(proc.tokenizer(text, add_special_tokens=False, return_attention_mask=True)["input_ids"])


def prefix(tok, text: str, cap: int) -> str:
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=cap, return_attention_mask=False)["input_ids"]
    return tok.decode(ids[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def budget(query: str, docs: list[dict[str, str]], L: int, proc, decision_messages, tok):
    codes = [f"candidate_{i:03d}" for i in range(20)]
    empty = [{"candidate_id": d["candidate_id"], "content": ""} for d in docs]
    overhead = token_count(proc, decision_messages, make_row(query, empty, codes), codes)
    available = L - RHO - overhead
    if available < 20:
        return None, {"builder_status": "FAIL", "error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}
    initial = max(1, available // 20)
    cap, reductions = initial, 0
    while cap >= 1:
        view = [{"candidate_id": d["candidate_id"], "content": prefix(tok, d["content"], cap)} for d in docs]
        total = token_count(proc, decision_messages, make_row(query, view, codes), codes)
        if total <= L - RHO:
            return view, {"builder_status": "PASS", "overhead_tokens": overhead, "effective_content_budget": available, "initial_per_passage_cap": initial, "final_per_passage_cap": cap, "deterministic_cap_reductions": reductions, "serialized_tokens": total}
        cap -= 1
        reductions += 1
    return None, {"builder_status": "FAIL", "error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}


def official_bm25(corpus_rows, query_rows):
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    ids = [str(x["_id"]) for x in corpus_rows]
    texts = [str(x.get("text") or "") for x in corpus_rows]
    analyzed = [analyzer.analyze(x) for x in texts]
    dictionary = Dictionary(analyzed)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    index = SparseMatrixSimilarity(model[list(map(dictionary.doc2bow, analyzed))], num_docs=len(texts), num_terms=len(dictionary), normalize_queries=False, normalize_documents=False)
    out = {}
    for i, q in enumerate(query_rows, 1):
        qid = str(q.get("_id", q.get("id")))
        sims = index[model[dictionary.doc2bow(analyzer.analyze(str(q.get("text") or "")))]] .tolist()
        pairs = [(did, float(score), j) for j, (did, score) in enumerate(zip(ids, sims))]
        pairs.sort(key=lambda x: x[1], reverse=True)
        out[qid] = pairs[:20]
        if i % 250 == 0:
            print(json.dumps({"phase": "bm25", "queries": i, "total": len(query_rows)}), flush=True)
    return out


def file_record(path: Path, root: Path, kind: str):
    lines = 0
    ids = set()
    with path.open("rb") as f:
        for line in f:
            if not line:
                break
            lines += 1
            if kind == "queries":
                try: ids.add(str(json.loads(line)["id"]))
                except Exception: pass
            elif kind == "qrels":
                if lines > 1:
                    ids.add(line.split(b"\t", 1)[0].decode("utf-8"))
    out = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "lines": lines, "sha256": sha_file(path)}
    if ids: out["unique_query_ids"] = len(ids)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "PROTOCOL_FROZEN.md"
    if sha_file(protocol) != PROTOCOL_SHA:
        raise RuntimeError("PROTOCOL_HASH_MISMATCH")
    builder, proc, tok, decision_messages = load_builder()
    manifest = {}
    all_queries = []
    all_slates = []
    qrel_counts = {}
    for forum in FORUMS:
        froot = DATA / forum
        corpus_path, queries_path, qrels_path = froot / "corpus.jsonl", froot / "queries.jsonl", froot / "qrels/test.tsv"
        manifest[forum] = {"corpus": file_record(corpus_path, DATA.parent.parent.parent, "corpus"), "queries": file_record(queries_path, DATA.parent.parent.parent, "queries"), "qrels": file_record(qrels_path, DATA.parent.parent.parent, "qrels")}
        corpus = list(read_jsonl(corpus_path))
        queries = list(read_jsonl(queries_path))
        qrels = set()
        with qrels_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("query-id"):
                    qrels.add(line.split("\t", 1)[0])
        qrel_counts[forum] = len(qrels)
        by_id = {str(x["_id"]): str(x.get("text") or "") for x in corpus}
        ranked = official_bm25(corpus, queries)
        for q in sorted(queries, key=lambda x: str(x.get("_id", x.get("id")))):
            qid = str(q.get("_id", q.get("id")))
            top = ranked[qid]
            candidates = [{"candidate_id": f"{forum}:{did}", "source_id": did, "content": by_id[did], "retrieval_score": score, "retrieval_rank": rank + 1, "content_sha256": sha_bytes(by_id[did].encode("utf-8"))} for rank, (did, score, _idx) in enumerate(top)]
            if len(candidates) != 20 or len({x["candidate_id"] for x in candidates}) != 20:
                raise RuntimeError(f"INVALID_TOP20:{forum}:{qid}")
            all_queries.append({"canonical_id": f"{forum}:{qid}", "forum": forum, "query_id": qid, "query": str(q.get("text") or "")})
            all_slates.append({"dataset": "cqadupstack", "forum": forum, "canonical_id": f"{forum}:{qid}", "query_id": qid, "query": str(q.get("text") or ""), "query_sha256": sha_bytes(str(q.get("text") or "").encode("utf-8")), "candidate_ids": [x["candidate_id"] for x in candidates], "candidates": candidates, "retrieval": {"method": "official_bright_bm25", "analyzer": "pyserini Lucene analyzer", "k1": 0.9, "b": 0.4, "candidate_pool_size": 20, "forum_local_corpus": True, "gold_in_input": False}, "slate_sha256": sha_obj([x["candidate_id"] for x in candidates]), "input_sha256": sha_obj({"canonical_id": f"{forum}:{qid}", "query": str(q.get("text") or ""), "candidates": [(x["candidate_id"], x["content_sha256"]) for x in candidates], "protocol": "decisionrag-v3.6-frozen-prospective-validation-v2"})})
        del corpus, queries, ranked
        print(json.dumps({"phase": "forum_retrieved", "forum": forum, "queries": len([x for x in all_slates if x["forum"] == forum])}), flush=True)
    all_slates.sort(key=lambda x: x["canonical_id"])
    if len(all_slates) != 13145:
        raise RuntimeError(f"CQADUPSTACK_QUERY_COUNT:{len(all_slates)}")
    # The dry IDs are the final 20 sorted IDs from the sparsest-qrel forum.
    sparse_forum = min(qrel_counts, key=lambda k: (qrel_counts[k], k))
    sparse_ids = sorted(x["canonical_id"] for x in all_slates if x["forum"] == sparse_forum)
    dry_ids = sparse_ids[-20:]
    hashed = sorted((x for x in all_slates if x["canonical_id"] not in set(dry_ids)), key=lambda x: sha_bytes((x["canonical_id"] + "731").encode("utf-8")))
    latency_ids = [x["canonical_id"] for x in hashed[:2000]]
    dry_set, latency_set = set(dry_ids), set(latency_ids)
    if dry_set & latency_set:
        raise RuntimeError("DRY_LATENCY_OVERLAP")
    OUT.joinpath("ARGUANA_SUPERSEDED_NOTICE.md").write_text("The prior v1 ArguAna preparation and interrupted dry trace are preserved in the sibling frozen_prospective_validation directory and are not part of v2.\n", encoding="utf-8")
    slate_path = OUT / "CQADUPSTACK_FROZEN_TOP20_SLATES.jsonl"
    slate_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in all_slates), encoding="utf-8")
    OUT.joinpath("LATENCY_COHORT_IDS.jsonl").write_text("".join(json.dumps({"canonical_id": x, "rank": i + 1}, sort_keys=True) + "\n" for i, x in enumerate(latency_ids)), encoding="utf-8")
    OUT.joinpath("DRY_RUN_IDS.jsonl").write_text("".join(json.dumps({"canonical_id": x, "forum": x.split(":", 1)[0]}, sort_keys=True) + "\n" for x in dry_ids), encoding="utf-8")
    lengths = []
    for x in all_slates:
        ps = [len(tok.encode(d["content"], add_special_tokens=False)) for d in x["candidates"]]
        lengths.append({"canonical_id": x["canonical_id"], "forum": x["forum"], "query_tokens": len(tok.encode(x["query"], add_special_tokens=False)), "passage_lengths": ps, "mean_passage_tokens": sum(ps) / 20, "max_passage_tokens": max(ps)})
    vals = sorted(x["mean_passage_tokens"] for x in lengths)
    t1, t2 = vals[len(vals) // 3], vals[(2 * len(vals)) // 3]
    for x in lengths: x["length_tercile"] = "short" if x["mean_passage_tokens"] <= t1 else ("medium" if x["mean_passage_tokens"] <= t2 else "long")
    OUT.joinpath("QUERY_LENGTHS.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lengths), encoding="utf-8")
    length_by_id = {x["canonical_id"]: x for x in lengths}
    prep_summary = []
    for L in LENGTHS:
        out_rows = []
        for i, x in enumerate(all_slates, 1):
            docs = [{"candidate_id": d["candidate_id"], "content": d["content"]} for d in x["candidates"]]
            bdocs, info = budget(x["query"], docs, L, proc, decision_messages, tok)
            out_rows.append({"dataset": "cqadupstack", "forum": x["forum"], "canonical_id": x["canonical_id"], "query_id": x["query_id"], "query": x["query"], "candidate_ids": x["candidate_ids"], "candidates": docs, "budgeted_candidates": bdocs or [], "budget_info": info, "serialized_tokens": info.get("serialized_tokens"), "length_limit": L, "safety_margin_tokens": RHO, "source_slate_sha256": x["slate_sha256"], "source_input_sha256": x["input_sha256"], "labels_joined": False, "input_label_blind": True, "is_dry_run": x["canonical_id"] in dry_set, "is_latency_query": x["canonical_id"] in latency_set})
            if i % 500 == 0: print(json.dumps({"phase": "builder", "L": L, "rows": i, "total": len(all_slates)}), flush=True)
        p = OUT / f"prepared_L{L}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in out_rows), encoding="utf-8")
        passed = [x for x in out_rows if x["budget_info"].get("builder_status") == "PASS"]
        if len(passed) != len(out_rows) or any(int(x["serialized_tokens"]) > L - RHO for x in passed):
            raise RuntimeError(f"BUILDER_FAILURE_OR_LIMIT:{L}")
        prep_summary.append({"L": L, "n": len(out_rows), "builder_pass_n": len(passed), "builder_failure_n": len(out_rows) - len(passed), "mean_serialized_tokens": sum(int(x["serialized_tokens"]) for x in passed) / len(passed), "max_serialized_tokens": max(int(x["serialized_tokens"]) for x in passed), "sha256": sha_file(p)})
    files = {str(p.relative_to(ROOT)): {"sha256": sha_file(p), "bytes": p.stat().st_size} for p in [slate_path, OUT / "LATENCY_COHORT_IDS.jsonl", OUT / "DRY_RUN_IDS.jsonl", OUT / "QUERY_LENGTHS.jsonl", *(OUT / f"prepared_L{L}.jsonl" for L in LENGTHS)]}
    manifest_out = {"archive": {"path": "data/external/v3_6_cqadupstack/cqadupstack.zip", "sha256": sha_file(ROOT / "data/external/v3_6_cqadupstack/cqadupstack.zip"), "official_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"}, "forums": manifest, "total_queries": len(all_slates), "qrel_query_counts": qrel_counts, "forum_order": FORUMS}
    OUT.joinpath("DATASET_FILE_MANIFEST.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    args = {"code_commit": "246ff54e66b178eadd350351a5544879f5484ed2", "protocol_sha256": PROTOCOL_SHA, "pplx_revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "jina_revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "pplx_template": "autojev.model.decision_messages + processor.apply_chat_template(add_generation_prompt=True, enable_thinking=False)", "builder": "equal prefix; decrement until serialized <= L-128", "batch_size": 1, "prefix_caching": False, "retries": 0, "labels_in_serving": False, "libraries": {"gensim": "4.4.0", "transformers": "5.16.1", "torch": "2.13.0+cu130"}}
    OUT.joinpath("SERVING_ARGS.json").write_text(json.dumps(args, indent=2, sort_keys=True) + "\n")
    OUT.joinpath("val_candidates.sha256").write_text(f"{sha_file(slate_path)}  CQADUPSTACK_FROZEN_TOP20_SLATES.jsonl\n", encoding="utf-8")
    summary = {"status": "PREPARED_BEFORE_CONFIRMATORY_INFERENCE", "protocol_sha256": PROTOCOL_SHA, "total_queries": len(all_slates), "dry_run_n": len(dry_ids), "confirmatory_quality_n": len(all_slates) - len(dry_ids), "latency_n": len(latency_ids), "dry_forum": sparse_forum, "latency_cohort_sha256": sha_obj(latency_ids), "length_tercile_boundaries": {"short_upper": t1, "medium_upper": t2}, "length_tercile_counts": {k: sum(x["length_tercile"] == k for x in lengths) for k in ("short", "medium", "long")}, "preparation": prep_summary, "files": files, "labels_joined": False}
    OUT.joinpath("PREPARATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
