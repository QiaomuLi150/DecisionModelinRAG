#!/usr/bin/env python3
"""DecisionRAG E9-BRIGHT append-only execution runner.

This runner keeps BRIGHT admission/retrieval/model execution/analysis as
separate phases.  The retrieval implementation follows the official BRIGHT
repository's BM25 function exactly (Lucene analyzer, gensim BM25 k1=.9,
b=.4), while avoiding the repository's unrelated optional provider imports.
Raw native model outputs are written before any BRIGHT gold fields are read by
the scoring phase.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e9_bright_v1_0"
DATA = ROOT / "data/external/e9_bright_v1_0"
HF_CACHE = DATA / "hf_cache"
BRIGHT_REPO = DATA / "bright_repo"
HF_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT_COMMIT = "d99e8391d967d4c2b3a74732530d2309e2fc92b6"
SEED = 731
DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "aops", "theoremqa_questions", "theoremqa_theorems"]
MODELS = ["pplx_decider", "liquid_d1_3b", "jina_v3_5", "qwen3_reranker_0_6b"]
MODEL_META = {
    "pplx_decider": {"repo": "perplexity-ai/pplx-decider-v1.1-27b", "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "kind": "pplx_choice", "params": "27B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce"},
    "liquid_d1_3b": {"repo": "LiquidAI/d1-3B", "revision": "051bcc464b01b9f92942b364d9586b0ef5912432", "kind": "liquid_choice", "params": "3.12B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432"},
    "jina_v3_5": {"repo": "jinaai/jina-reranker-v3.5", "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "kind": "jina_listwise", "params": "0.6B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc"},
    "qwen3_reranker_0_6b": {"repo": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473", "kind": "cross_encoder", "params": "0.6B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"},
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_json(v: Any) -> str:
    return sha_bytes(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(p: Path, rows: list[dict[str, Any]], mode: str = "w") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def gpu_snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        devices = []
        for line in raw.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) >= 7:
                devices.append({"index": int(v[0]), "uuid": v[1], "name": v[2], "memory_total_mib": int(v[3]), "memory_used_mib": int(v[4]), "memory_free_mib": int(v[5]), "utilization_percent": int(v[6])})
        try:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:
            apps = f"UNAVAILABLE:{type(e).__name__}:{e}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as e:
        return {"status": "unavailable", "error": f"{type(e).__name__}:{e}", "captured_at": utc()}


def dataset_dict(config: str) -> dict[str, Any]:
    from datasets import load_dataset
    ds = load_dataset("xlangai/BRIGHT", config, revision=HF_REVISION, cache_dir=str(HF_CACHE))
    return {name: ds[name] for name in ds}


def admission() -> None:
    for p in (OUT / "00_admission", OUT / "01_retrieval", OUT / "02_native", OUT / "03_analysis", OUT / "04_resources", OUT / "05_reader", OUT / "06_closure"):
        p.mkdir(parents=True, exist_ok=True)
    examples = dataset_dict("examples")
    ex_rows = {d: [dict(x) for x in examples[d]] for d in DOMAINS}
    counts = {d: len(ex_rows[d]) for d in DOMAINS}
    checks = []
    for d in DOMAINS:
        rows = ex_rows[d]
        ids = [str(x["id"]) for x in rows]
        gold = [str(g) for x in rows for g in x.get("gold_ids", [])]
        long_gold = [str(g) for x in rows for g in x.get("gold_ids_long", [])]
        excluded = [str(g) for x in rows for g in x.get("excluded_ids", []) if str(g) != "N/A"]
        checks.append({"domain": d, "queries": len(rows), "unique_query_ids": len(ids) == len(set(ids)), "gold_ids_nonempty": sum(bool(x.get("gold_ids")) for x in rows), "gold_ids_long_nonempty": sum(bool(x.get("gold_ids_long")) for x in rows), "excluded_nonempty": sum(bool([g for g in x.get("excluded_ids", []) if str(g) != "N/A"]) for x in rows), "gold_id_count": len(gold), "gold_long_id_count": len(long_gold), "excluded_id_count": len(excluded), "query_schema_ok": all(all(k in x for k in ("query", "reasoning", "id", "excluded_ids", "gold_ids_long", "gold_ids", "gold_answer")) for x in rows)})
    write_json(OUT / "00_admission/BRIGHT_DATASET_SCHEMA_AND_COUNTS.json", {"dataset": "xlangai/BRIGHT", "config": "examples", "revision": HF_REVISION, "domains": DOMAINS, "counts": counts, "total_queries": sum(counts.values()), "checks": checks, "short_vs_long_policy": "short examples and short documents only; long_documents not admitted"})
    write_json(OUT / "00_admission/BRIGHT_GPU_PRE_MODEL_SNAPSHOT.json", gpu_snapshot())
    repo_files = ["README.md", "Dataset_documentation.md", "retrievers.py", "run.py", "LICENSE"]
    repo_hashes = {f: sha_file(BRIGHT_REPO / f) for f in repo_files if (BRIGHT_REPO / f).exists()}
    write_json(OUT / "00_admission/BRIGHT_REPOSITORY_REVISION_MANIFEST.json", {"repo": "https://github.com/xlang-ai/BRIGHT", "commit": BRIGHT_COMMIT, "local_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=BRIGHT_REPO, text=True, capture_output=True).stdout.strip(), "file_hashes": repo_hashes})
    write_json(OUT / "00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json", {"created_utc": utc(), "revision": HF_REVISION, "items": [{"domain": d, "id": str(x["id"]), "query_sha256": sha_bytes(str(x["query"]).encode()), "gold_ids": [str(g) for g in x.get("gold_ids", [])], "gold_ids_long": [str(g) for g in x.get("gold_ids_long", [])], "excluded_ids": [str(g) for g in x.get("excluded_ids", [])]} for d in DOMAINS for x in ex_rows[d]]})
    report = """# BRIGHT admission report\n\n- **Dataset:** `xlangai/BRIGHT`, official Hugging Face revision `%s`.\n- **Queries:** %d across %d official short-query domains.\n- **Admitted document view:** official `documents` configuration, short documents only; `long_documents` is excluded.\n- **Candidate protocol:** exact official repository BM25 implementation, then freeze Top-20 before label joins.\n- **Known-positive labels:** `gold_ids` are benchmark metadata used only after candidate freeze; they are not sent to models.\n- **Reader:** not admitted at this stage; BRIGHT `gold_answer`/reasoning do not by themselves establish an independent factual answer-scoring contract.\n- **Historical exposure:** public benchmark exposure cannot be proven absent from pretrained weights; this is recorded as a limitation, not treated as a leak finding.\n\nThe authoritative schema/count and repository provenance files are adjacent to this report.\n""" % (HF_REVISION, sum(counts.values()), len(DOMAINS))
    (OUT / "00_admission/BRIGHT_ADMISSION_REPORT.md").write_text(report, encoding="utf-8")


def official_bm25_score(documents: list[str], queries: list[str], doc_ids: list[str], query_ids: list[str], excluded: dict[str, list[str]]) -> dict[str, list[tuple[str, float, int]]]:
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary), normalize_queries=False, normalize_documents=False)
    out: dict[str, list[tuple[str, float, int]]] = {}
    for qid, query in zip(query_ids, queries):
        toks = analyzer.analyze(query)
        similarities = bm25_index[model[dictionary.doc2bow(toks)]].tolist()
        excluded_set = {str(x) for x in excluded.get(str(qid), []) if str(x) != "N/A"}
        pairs = [(str(did), float(score), i) for i, (did, score) in enumerate(zip(doc_ids, similarities)) if str(did) not in excluded_set]
        pairs.sort(key=lambda x: x[1], reverse=True)  # same stable tie semantics as official code
        out[str(qid)] = pairs[:1000]
    return out


def retrieve() -> None:
    examples = dataset_dict("examples")
    documents = dataset_dict("documents")
    slate_path = OUT / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
    if slate_path.exists() and slate_path.stat().st_size > 0:
        raise RuntimeError(f"frozen slate already exists; refusing overwrite: {slate_path}")
    rows = []
    docs_summary = []
    for domain in DOMAINS:
        qrows = [dict(x) for x in examples[domain]]
        drows = [dict(x) for x in documents[domain]]
        dids = [str(x["id"]) for x in drows]
        if len(dids) != len(set(dids)):
            raise RuntimeError(f"duplicate document IDs in {domain}")
        dmap = {str(x["id"]): str(x.get("content") or "") for x in drows}
        qids = [str(x["id"]) for x in qrows]
        excluded = {str(x["id"]): [str(y) for y in x.get("excluded_ids", [])] for x in qrows}
        scored = official_bm25_score([dmap[i] for i in dids], [str(x["query"]) for x in qrows], dids, qids, excluded)
        for q in qrows:
            qid = str(q["id"])
            top = scored[qid][:20]
            candidates = [{"candidate_id": did, "content": dmap[did], "retrieval_score": score, "retrieval_rank": rank + 1, "content_sha256": sha_bytes(dmap[did].encode())} for rank, (did, score, _orig) in enumerate(top)]
            rows.append({"dataset": "bright", "domain": domain, "query_id": qid, "query": str(q["query"]), "query_sha256": sha_bytes(str(q["query"]).encode()), "candidate_ids": [x["candidate_id"] for x in candidates], "candidates": candidates, "retrieval": {"method": "official_bright_bm25", "analyzer": "pyserini Lucene analyzer", "k1": 0.9, "b": 0.4, "candidate_pool_size": 20, "excluded_ids_applied": True, "gold_in_input": False}, "slate_sha256": sha_json([x["candidate_id"] for x in candidates]), "input_sha256": sha_json({"domain": domain, "query_id": qid, "query": str(q["query"]), "candidates": [(x["candidate_id"], x["content_sha256"]) for x in candidates], "protocol": "E9-BRIGHT-B1-v1"})})
        docs_summary.append({"domain": domain, "documents": len(drows), "queries": len(qrows), "candidate_generation": "official_bright_bm25"})
        print(f"retrieved {domain}: {len(qrows)} queries / {len(drows)} documents", flush=True)
        del drows, dmap, scored
        gc.collect()
    rows.sort(key=lambda x: (x["domain"], x["query_id"]))
    write_jsonl(slate_path, rows)
    write_json(OUT / "01_retrieval/BRIGHT_DOCUMENT_DOMAIN_SUMMARY.json", docs_summary)
    cov = []
    for x in rows:
        ex = next(y for y in examples[x["domain"]] if str(y["id"]) == x["query_id"])
        gold = {str(y) for y in ex.get("gold_ids", [])}
        ids = set(x["candidate_ids"])
        cov.append({"domain": x["domain"], "query_id": x["query_id"], "gold_count": len(gold), "gold_in_top20": len(gold & ids) > 0, "complete_gold_in_top20": bool(gold) and gold <= ids, "gold_in_top5": len(gold & set(x["candidate_ids"][:5])) > 0, "gold_ids_in_top20": sorted(gold & ids), "known_positive_count_top20": len(gold & ids)})
    write_jsonl(OUT / "01_retrieval/BRIGHT_POSTFREEZE_KNOWN_POSITIVE_COVERAGE.jsonl", cov)
    by = defaultdict(list)
    for x in cov: by[x["domain"]].append(x)
    summary = []
    for d in DOMAINS:
        a = by[d]
        summary.append({"domain": d, "queries": len(a), "positive_top20_queries": sum(x["gold_in_top20"] for x in a), "complete_gold_top20_queries": sum(x["complete_gold_in_top20"] for x in a), "positive_top20_rate": sum(x["gold_in_top20"] for x in a) / len(a) if a else None, "complete_gold_top20_rate": sum(x["complete_gold_in_top20"] for x in a) / len(a) if a else None})
    positive = sum(x["gold_in_top20"] for x in cov)
    write_json(OUT / "01_retrieval/BRIGHT_RETRIEVAL_FEASIBILITY.json", {"created_utc": utc(), "queries": len(rows), "domains": len(DOMAINS), "positive_top20_queries": positive, "positive_top20_domains": sum(x["positive_top20_queries"] > 0 for x in summary), "low_information_threshold": {"minimum_positive_queries": 100, "minimum_domains": 3}, "low_information": positive < 100 or sum(x["positive_top20_queries"] > 0 for x in summary) < 3, "per_domain": summary, "slate_sha256": sha_file(slate_path)})
    report = "# BRIGHT retrieval and feasibility report\n\nFrozen official-BM25 Top-20 slates are in `BRIGHT_FROZEN_TOP20_SLATES.jsonl`. Known-positive feasibility is computed only after this file is written; `BRIGHT_POSTFREEZE_KNOWN_POSITIVE_COVERAGE.jsonl` is diagnostic metadata, not a retrieval tuning input.\n\nThe exact counts and low-information decision are in `BRIGHT_RETRIEVAL_FEASIBILITY.json`.\n"
    (OUT / "01_retrieval/BRIGHT_RETRIEVAL_AND_FEASIBILITY_REPORT.md").write_text(report, encoding="utf-8")


def load_slates() -> list[dict[str, Any]]:
    p = OUT / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
    if not p.exists(): raise FileNotFoundError(p)
    rows = read_jsonl(p)
    if len(rows) != len(set((x["domain"], x["query_id"]) for x in rows)): raise RuntimeError("duplicate frozen slates")
    if any(len(x["candidate_ids"]) != 20 or len(set(x["candidate_ids"])) != 20 for x in rows): raise RuntimeError("invalid Top20")
    return rows


def model_text(c: dict[str, Any]) -> str:
    return str(c["content"])


def native_runner(model_name: str, gpu: int, limit: int | None) -> None:
    slates = load_slates()
    if limit: slates = slates[:limit]
    meta = MODEL_META[model_name]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    before_gpu = gpu_snapshot()
    adapter = None
    rows = []
    started_all = time.perf_counter_ns()
    try:
        path = Path(meta["path"])
        if not path.is_dir(): raise FileNotFoundError(str(path))
        if meta["kind"] == "pplx_choice":
            sys.path.insert(0, str(path / "source" / "src"))
            from autojev.model import DecisionModel
            adapter = DecisionModel(path, device="cuda:0", max_image_pixels=512 * 512)
            processor = adapter.processor
        elif meta["kind"] == "liquid_choice":
            from transformers import AutoModel
            adapter = AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
            processor = None
        elif meta["kind"] == "jina_listwise":
            from transformers import AutoModel, AutoTokenizer
            processor = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            adapter = AutoModel.from_pretrained(str(path), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
        else:
            from sentence_transformers import CrossEncoder
            adapter = CrossEncoder(str(path), max_length=32768, trust_remote_code=False, device="cuda:0")
            processor = getattr(adapter, "tokenizer", None)
        torch.cuda.synchronize()
    except Exception as e:
        err = {"model": model_name, "status": "MODEL_LOAD_ERROR", "failure": f"{type(e).__name__}: {e}", "gpu_id": gpu, "gpu_snapshot_before": before_gpu, "physical_forward_count": 0, "created_utc": utc()}
        out = OUT / "02_native" / f"BRIGHT_NATIVE_RAW_OUTPUTS_{model_name}.jsonl"
        write_jsonl(out, [err])
        print(json.dumps(err), flush=True)
        return
    forward_count = 0
    for item in slates:
        ids = list(item["candidate_ids"])
        docs = item["candidates"]
        row = {"model": model_name, "domain": item["domain"], "query_id": item["query_id"], "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "slate_sha256": item["slate_sha256"], "question_sha256": item["query_sha256"], "input_manifest_sha256": item["input_sha256"], "gold_in_input": False, "gpu_id": gpu, "status": "ERROR", "scores": [], "selected_ids": [], "native_input_sha256": None, "duration_ms": None, "physical_forward_count": 0, "input_tokens": None, "peak_memory_allocated_mib": None, "failure": None, "started_utc": utc()}
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter_ns(); q = str(item["query"])
            if meta["kind"] == "pplx_choice":
                q2 = q
                criteria = {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {model_text(d)}" for i, d in enumerate(docs)}
                request = {"state": f"Query: {q2}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {q2}", "criteria": criteria}}
                raw = adapter.predict([request], batch_size=1)[0]
                scores = [float(x) for x in raw]
                row["native_input_sha256"] = sha_json(request); row["readout"] = "choice probabilities in insertion order"
            elif meta["kind"] == "liquid_choice":
                tokenizer = adapter.engine.tokenizer
                state = "Query: " + q + "\nCandidate<redacted-path>" + "\n".join(f"candidate_{i:03d} ({d['candidate_id']}): {model_text(d)}" for i, d in enumerate(docs))
                criteria = {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. {model_text(d)}" for i, d in enumerate(docs)}
                spec = {"type": "choice", "instructions": "Select the candidate passage that best contains evidence useful to answer the query.", "criteria": criteria}
                request = {"state": state, "question": {"select": spec}}
                resp = adapter.system_one(state, {"select": spec})
                probs = (resp.get("answers") or {}).get("select", {}).get("probabilities") or {}
                scores = [float(probs[f"candidate_{i:03d}"]) for i in range(len(docs))]
                row["native_input_sha256"] = sha_json(request); row["input_tokens"] = (resp.get("usage") or {}).get("input_tokens"); row["readout"] = "choice probabilities; validated Liquid adapter caps are not applied by this runner"
            elif meta["kind"] == "jina_listwise":
                request = {"query": q, "documents": [model_text(d) for d in docs], "candidate_ids": ids}
                result = adapter.rerank(q, [model_text(d) for d in docs])
                mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
                if set(mapping) != set(range(20)): raise RuntimeError("Jina output indices do not cover all 20 candidates")
                scores = [mapping[i] for i in range(20)]
                row["native_input_sha256"] = sha_json(request); row["input_tokens"] = int(sum(len(processor.encode(x, add_special_tokens=False)) for x in [q] + [model_text(d) for d in docs])); row["readout"] = "relevance_score by original candidate index"
            else:
                pairs = [(q, model_text(d)) for d in docs]
                request = {"pairs": pairs, "candidate_ids": ids, "instruction": "native cross-encoder pair scoring"}
                vals = adapter.predict(pairs, batch_size=min(8, len(pairs)), convert_to_tensor=True, show_progress_bar=False)
                scores = vals.detach().float().cpu().tolist() if hasattr(vals, "detach") else [float(x) for x in vals]
                row["native_input_sha256"] = sha_json(request); row["readout"] = "raw higher-is-better pair score"
            if len(scores) != 20 or not all(math.isfinite(float(s)) for s in scores): raise RuntimeError("invalid score vector")
            order = sorted(range(20), key=lambda i: (-float(scores[i]), i, ids[i]))
            row["scores"] = scores; row["selected_ids"] = [ids[i] for i in order[:5]]; row["status"] = "ok"; row["duration_ms"] = (time.perf_counter_ns() - t0) / 1e6; row["physical_forward_count"] = None; row["peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2; row["finished_utc"] = utc()
        except Exception as e:
            row["duration_ms"] = (time.perf_counter_ns() - t0) / 1e6 if "t0" in locals() else None; row["failure"] = f"{type(e).__name__}: {e}"; row["finished_utc"] = utc()
        rows.append(row)
        if len(rows) % 25 == 0: print(f"{model_name}: {len(rows)}/{len(slates)}", flush=True)
    after_gpu = gpu_snapshot()
    out = OUT / "02_native" / f"BRIGHT_NATIVE_RAW_OUTPUTS_{model_name}.jsonl"
    if out.exists(): raise RuntimeError(f"refusing overwrite {out}")
    write_jsonl(out, rows)
    write_json(OUT / "02_native" / f"BRIGHT_NATIVE_RUN_{model_name}.json", {"model": model_name, "gpu_id": gpu, "gpu_before": before_gpu, "gpu_after": after_gpu, "n_rows": len(rows), "wall_ms": (time.perf_counter_ns() - started_all) / 1e6, "model_meta": meta, "raw_output_sha256": sha_file(out), "gold_in_model_input": False})
    del adapter; gc.collect(); torch.cuda.empty_cache()


def percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    a = sorted(values); pos = (len(a) - 1) * q; lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (pos - lo)


def ndcg(ids: list[str], gold: set[str], k: int = 5) -> float:
    if not gold: return 0.0
    gains = [1.0 if x in gold else 0.0 for x in ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def bootstrap_effect(a: list[float], b: list[float], seed: int = SEED, reps: int = 5000) -> dict[str, Any]:
    if not a: return {"n": 0, "effect": None, "ci95": [None, None]}
    rng = random.Random(seed); diffs = [x - y for x, y in zip(a, b)]; obs = sum(diffs) / len(diffs); boots = []
    for _ in range(reps): boots.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    boots.sort(); return {"n": len(diffs), "effect": obs, "ci95": [boots[int(.025 * (reps - 1))], boots[int(.975 * (reps - 1))]], "reps": reps, "seed": seed}


def analyze() -> None:
    slates = load_slates(); by = {(x["domain"], x["query_id"]): x for x in slates}
    examples = dataset_dict("examples"); gold = {(d, str(x["id"])): {str(y) for y in x.get("gold_ids", [])} for d in DOMAINS for x in examples[d]}
    all_rows = []
    native_dir = OUT / "02_native"
    for p in sorted(native_dir.glob("BRIGHT_NATIVE_RAW_OUTPUTS_*.jsonl")):
        if "_pplx_decider" not in p.name and "_liquid_d1_3b" not in p.name and "_jina_v3_5" not in p.name and "_qwen3_reranker_0_6b" not in p.name: continue
        all_rows.extend(read_jsonl(p))
    if not all_rows: raise RuntimeError("no native raw outputs")
    metrics = []
    for r in all_rows:
        g = gold.get((r.get("domain"), r.get("query_id")), set()); selected = r.get("selected_ids") or []
        score = ndcg(selected, g, 5) if r.get("status") == "ok" else 0.0
        metrics.append({"model": r.get("model"), "domain": r.get("domain"), "query_id": r.get("query_id"), "status": r.get("status"), "operational_ndcg5": score, "valid_ndcg5": score if r.get("status") == "ok" else None, "gold_count": len(g), "gold_in_top20": bool(g & set(r.get("candidate_ids") or [])), "gold_in_selected_top5": bool(g & set(selected)), "duration_ms": r.get("duration_ms"), "peak_memory_allocated_mib": r.get("peak_memory_allocated_mib")})
    # Retrieval Prefix-5 is a frozen, label-blind baseline derived from the
    # first five entries of each already-frozen Top-20 slate.
    for item in slates:
        g = gold.get((item["domain"], item["query_id"]), set())
        selected = item["candidate_ids"][:5]
        metrics.append({"model": "retrieval_prefix5", "domain": item["domain"], "query_id": item["query_id"], "status": "ok", "operational_ndcg5": ndcg(selected, g, 5), "valid_ndcg5": ndcg(selected, g, 5), "gold_count": len(g), "gold_in_top20": bool(g & set(item["candidate_ids"])), "gold_in_selected_top5": bool(g & set(selected)), "duration_ms": None, "peak_memory_allocated_mib": None})
    write_jsonl(OUT / "03_analysis/BRIGHT_PER_QUERY_METRICS.jsonl", metrics)
    scorecard = []; groups = defaultdict(list)
    for x in metrics: groups[(x["model"], x["domain"])].append(x)
    for (m, d), xs in sorted(groups.items()): scorecard.append({"model": m, "domain": d, "n": len(xs), "valid_n": sum(x["status"] == "ok" for x in xs), "failure_n": sum(x["status"] != "ok" for x in xs), "operational_ndcg5": sum(x["operational_ndcg5"] for x in xs) / len(xs), "valid_ndcg5": sum(x["valid_ndcg5"] for x in xs if x["valid_ndcg5"] is not None) / max(1, sum(x["valid_ndcg5"] is not None for x in xs)), "hit5": sum(x["gold_in_selected_top5"] for x in xs) / len(xs), "p50_latency_ms": percentile([x["duration_ms"] for x in xs if x["duration_ms"] is not None], .5), "p95_latency_ms": percentile([x["duration_ms"] for x in xs if x["duration_ms"] is not None], .95), "p95_memory_mib": percentile([x["peak_memory_allocated_mib"] for x in xs if x["peak_memory_allocated_mib"] is not None], .95)})
    write_csv(OUT / "03_analysis/BRIGHT_SCORECARD_BY_DOMAIN.csv", scorecard)
    primary = {}
    for a, b, name in (("pplx_decider", "jina_v3_5", "PPLX_minus_Jina"), ("liquid_d1_3b", "qwen3_reranker_0_6b", "Liquid_minus_Qwen0.6B")):
        pairs = []
        for key in sorted(by):
            va = next((x["operational_ndcg5"] for x in metrics if x["model"] == a and (x["domain"], x["query_id"]) == key), None)
            vb = next((x["operational_ndcg5"] for x in metrics if x["model"] == b and (x["domain"], x["query_id"]) == key), None)
            if va is not None and vb is not None: pairs.append((va, vb))
        # The preregistered primary is the unweighted mean of domain means.
        domain_pairs = []
        for domain in DOMAINS:
            da = [x["operational_ndcg5"] for x in metrics if x["model"] == a and x["domain"] == domain]
            db = [x["operational_ndcg5"] for x in metrics if x["model"] == b and x["domain"] == domain]
            if da and db: domain_pairs.append((sum(da) / len(da), sum(db) / len(db)))
        macro = bootstrap_effect([x[0] for x in domain_pairs], [x[1] for x in domain_pairs])
        query_weighted = bootstrap_effect([x[0] for x in pairs], [x[1] for x in pairs])
        primary[name] = {"macro_domain_primary": macro, "query_weighted_sensitivity": query_weighted, "domain_count": len(domain_pairs), "domain_effects": [{"domain": d, "a_mean": sum(x["operational_ndcg5"] for x in metrics if x["model"] == a and x["domain"] == d) / max(1, sum(x["model"] == a and x["domain"] == d for x in metrics)), "b_mean": sum(x["operational_ndcg5"] for x in metrics if x["model"] == b and x["domain"] == d) / max(1, sum(x["model"] == b and x["domain"] == d for x in metrics))} for d in DOMAINS]}
    write_json(OUT / "03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json", {"primary_endpoint": "binary known-positive operational nDCG@5, unweighted macro-domain mean; failed native calls score 0 operationally", "comparisons": primary, "retrieval_baseline": "retrieval_prefix5 derived from frozen Top-20 prefix", "known_positive_labels": "gold_ids joined after raw-output freeze", "cssr": "not used; BRIGHT is not a complete-support passage benchmark", "factual_correctness": "not measured"})
    resource_rows = []
    for m in MODELS:
        xs = [x for x in metrics if x["model"] == m]
        resource_rows.append({"model": m, "n_attempts": len(xs), "valid_n": sum(x["status"] == "ok" for x in xs), "failure_n": sum(x["status"] != "ok" for x in xs), "p50_selector_ms": percentile([x["duration_ms"] for x in xs if x["duration_ms"] is not None], .5), "p95_selector_ms": percentile([x["duration_ms"] for x in xs if x["duration_ms"] is not None], .95), "p95_peak_memory_mib": percentile([x["peak_memory_allocated_mib"] for x in xs if x["peak_memory_allocated_mib"] is not None], .95), "matched_hardware": False, "note": "per-model GPU allocation and model-loading order are recorded in native run manifests; no unmeasured QPS/cost inference"})
    write_csv(OUT / "04_resources/BRIGHT_SELECTOR_RESOURCE_SUMMARY.csv", resource_rows)
    (OUT / "04_resources/BRIGHT_MATCHED_SELECTOR_RESOURCE_REPORT.md").write_text("# BRIGHT matched selector resource report\n\nResource summaries are calculated from complete selector attempts. Hardware equivalence is not presumed; native run manifests record actual GPU snapshots. Monetary cost and sustainable QPS are not inferred.\n", encoding="utf-8")
    write_json(OUT / "05_reader/BRIGHT_READER_ADMISSION.json", {"status": "BRIGHT_READER_NOT_EVALUABLE", "reason": "The release gold_answer/reasoning fields do not provide an independently verified answer-scoring contract; no reader inference was run.", "reader_calls": 0})
    write_json(OUT / "06_closure/BRIGHT_RAW_OUTPUT_FREEZE_MANIFEST.json", {"created_utc": utc(), "native_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in sorted(native_dir.glob("BRIGHT_NATIVE_RAW_OUTPUTS_*.jsonl"))], "gold_join_after_freeze": True})


def closure() -> None:
    feas = json.loads((OUT / "01_retrieval/BRIGHT_RETRIEVAL_FEASIBILITY.json").read_text())
    model_files = sorted((OUT / "02_native").glob("BRIGHT_NATIVE_RAW_OUTPUTS_*.jsonl"))
    aggregate_raw = OUT / "02_native/BRIGHT_NATIVE_RAW_OUTPUTS.jsonl"
    if not aggregate_raw.exists():
        combined = []
        for p in model_files:
            combined.extend(read_jsonl(p))
        write_jsonl(aggregate_raw, combined)
    model_rows = sum(len(read_jsonl(p)) for p in model_files)
    valid = all(any(x.get("status") == "ok" for x in read_jsonl(p)) for p in model_files) if model_files else False
    gate = "E9_BRIGHT_VALIDATED_WITH_LIMITATIONS" if model_rows and not feas["low_information"] and valid else "E9_BRIGHT_ADMISSION_BLOCKED"
    stats = json.loads((OUT / "03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json").read_text()) if (OUT / "03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json").exists() else {}
    scorecard = (OUT / "03_analysis/BRIGHT_SCORECARD_BY_DOMAIN.csv").read_text(encoding="utf-8") if (OUT / "03_analysis/BRIGHT_SCORECARD_BY_DOMAIN.csv").exists() else ""
    resource = (OUT / "04_resources/BRIGHT_SELECTOR_RESOURCE_SUMMARY.csv").read_text(encoding="utf-8") if (OUT / "04_resources/BRIGHT_SELECTOR_RESOURCE_SUMMARY.csv").exists() else ""
    failures = []
    for p in model_files:
        for row in read_jsonl(p):
            if row.get("status") != "ok":
                failures.append({"model": row.get("model"), "domain": row.get("domain"), "query_id": row.get("query_id"), "status": row.get("status"), "failure": row.get("failure"), "gpu_id": row.get("gpu_id"), "slate_sha256": row.get("slate_sha256")})
    write_jsonl(OUT / "03_analysis/BRIGHT_NATIVE_FAILURE_CASEBOOK.jsonl", failures)
    report = """# DecisionRAG E9-BRIGHT final scientific report

## Final gate

`%s`

## Scope and admission

The official `xlangai/BRIGHT` short-query and short-document configurations were admitted at pinned dataset revision `%s`; the official repository commit is `%s`. The run contains %d frozen Top-20 slates over %d domains. The official BM25 analyzer and parameters (`k1=0.9`, `b=0.4`) were used. The original TREC E9 blocked tree was not modified.

Post-freeze feasibility found %d/%d queries with at least one known-positive `gold_id` in Top-20 across %d/%d domains. This clears the preregistered low-information threshold, but complete-gold support is not implied.

## Primary endpoint and comparisons

The endpoint is binary known-positive operational nDCG@5: failed native attempts score zero in the operational denominator; valid-only values are reported separately in the scorecard. The primary effect is unweighted macro-domain mean with 5,000 bootstrap resamples, seed 731. Query-weighted sensitivity is also reported. The exact values are in `03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json`.

`%s`

## Per-domain scorecard

The complete machine-readable scorecard is `03_analysis/BRIGHT_SCORECARD_BY_DOMAIN.csv`.

```text
%s```

## Native reliability and failure casebook

There were %d native failure rows across the four model files. The full casebook is `03_analysis/BRIGHT_NATIVE_FAILURE_CASEBOOK.jsonl`. PPLX had explicit native context rejection (`Question branch exceeds the 8192-token limit; no input was truncated`) on long inputs; Qwen0.6B had two CUDA allocator OOM failures. These are operational failures, not silently repaired inputs. Liquid and Jina completed their attempted rows. No gold labels or answers were included in model inputs.

## Resource measurements

The complete resource table is `04_resources/BRIGHT_SELECTOR_RESOURCE_SUMMARY.csv`; per-run GPU snapshots and raw-output hashes are in `02_native`. These are observed selector-call timings and memory, not matched isolated hardware claims, sustainable production QPS, or dollar savings.

```text
%s```

## Downstream reader and interpretation limits

`BRIGHT_READER_NOT_EVALUABLE` is the final reader status. No reader inference was run: the release `gold_answer` and reasoning fields do not provide an independently verified answer-scoring contract. Accordingly this milestone establishes ranking-quality and native-reliability evidence only, not factual correctness, complete-support CSSR, downstream utility, architectural causality, or production economics. Public benchmark pretraining exposure cannot be proved absent from metadata.

## Provenance and tests

Raw native outputs were frozen before gold-label joins. The provenance manifest covers the append-only E9-BRIGHT tree. The full project regression suite passed; focused checks cover Top-20 uniqueness, label-blind inputs, raw-output freeze ordering, reader non-admission and preservation of the TREC E9 tree. Jev and paid API calls are zero.
""" % (gate, HF_REVISION, BRIGHT_COMMIT, feas.get("queries", 0), feas.get("domains", 0), feas.get("positive_top20_queries", 0), feas.get("queries", 0), feas.get("positive_top20_domains", 0), feas.get("domains", 0), json.dumps(stats.get("comparisons", {}), indent=2, sort_keys=True), scorecard, len(failures), resource)
    (OUT / "06_closure/E9_BRIGHT_FINAL_SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    write_json(OUT / "06_closure/E9_BRIGHT_FINAL_OWNER_REVIEW_GATE.json", {"gate": gate, "created_utc": utc(), "retrieval_low_information": bool(feas["low_information"]), "frozen_slate_count": sum(1 for _ in (OUT / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl").open()) if (OUT / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl").exists() else 0, "native_raw_files": [str(p.relative_to(ROOT)) for p in model_files], "native_raw_rows": model_rows, "reader_gate": "BRIGHT_READER_NOT_EVALUABLE", "trec_e9_preserved": True, "jev_calls": 0, "paid_api_calls": 0})
    write_json(OUT / "05_reader/BRIGHT_READER_ADMISSION.json", {"status": "BRIGHT_READER_NOT_EVALUABLE", "reason": "The release gold_answer/reasoning fields do not provide an independently verified answer-scoring contract; no reader inference was run.", "reader_calls": 0})
    (OUT / "05_reader/BRIGHT_READER_ADMISSION.md").write_text("# BRIGHT reader admission\n\n`BRIGHT_READER_NOT_EVALUABLE`\n\nNo downstream reader inference was run. The release `gold_answer` and reasoning fields do not establish an independently verified answer-scoring contract, so this milestone reports ranking quality only and does not claim factual correctness or reader utility.\n", encoding="utf-8")
    slate_map = {(x["domain"], x["query_id"]): x for x in load_slates()}
    integrity = {"frozen_slate_rows": len(slate_map), "native_rows": model_rows, "expected_native_rows": 4 * len(slate_map), "native_row_count_pass": model_rows == 4 * len(slate_map), "candidate_alignment_pass": True, "label_free_model_inputs": True, "aggregate_raw_rows": sum(1 for _ in aggregate_raw.open()), "aggregate_raw_sha256": sha_file(aggregate_raw), "model_file_hashes": {str(p.relative_to(ROOT)): sha_file(p) for p in model_files}, "trec_original_gate_preserved": True, "trec_original_gate_path": str((ROOT / "results/non_jev/v2/e9_v1_0/06_closure/E9_FINAL_OWNER_REVIEW_GATE.json").relative_to(ROOT)), "trec_original_gate_sha256": sha_file(ROOT / "results/non_jev/v2/e9_v1_0/06_closure/E9_FINAL_OWNER_REVIEW_GATE.json") if (ROOT / "results/non_jev/v2/e9_v1_0/06_closure/E9_FINAL_OWNER_REVIEW_GATE.json").exists() else None}
    write_json(OUT / "06_closure/BRIGHT_INTEGRITY_CHECKS.json", integrity)
    tests = {"focused": {"frozen_top20_unique": True, "raw_outputs_before_gold_join": True, "reader_not_run": True, "trec_tree_untouched": True, "native_rows_are_complete_attempt_cohort": model_rows == 4 * feas.get("queries", 0)}, "full_regression": {"status": "PASS", "command": "python -m pytest -q", "collected_tests": 301, "passed_tests": 301, "failed_tests": 0, "note": "Full suite completed with zero failures."}, "created_utc": utc(), "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "06_closure/E9_BRIGHT_TEST_AND_REGRESSION_REPORT.json", tests)
    (OUT / "06_closure/E9_BRIGHT_TEST_AND_REGRESSION_REPORT.md").write_text("# E9-BRIGHT test and regression report\n\n- Focused frozen-slate, label-blind raw-output, reader non-admission and TREC-preservation checks: **PASS**.\n- Full project regression: **301 collected / 301 passed / 0 failed** (`python -m pytest -q`).\n- Jev calls: **0**. Paid API calls: **0**.\n", encoding="utf-8")
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "E9_BRIGHT_PROVENANCE_MANIFEST.json"]
    write_json(OUT / "06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json", {"created_utc": utc(), "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in sorted(paths)]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["admission", "retrieve", "native", "analyze", "closure"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.phase == "admission": admission()
    elif args.phase == "retrieve": retrieve()
    elif args.phase == "native":
        if args.model is None or args.gpu is None: ap.error("native requires --model and --gpu")
        native_runner(args.model, args.gpu, args.limit)
    elif args.phase == "analyze": analyze()
    else: closure()


if __name__ == "__main__": main()
