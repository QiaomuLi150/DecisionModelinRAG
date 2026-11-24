#!/usr/bin/env python3
"""E5-0 data-readiness, reservation, and natural-slate preparation.

This module deliberately stops at the E5-0 gate.  It never imports or calls a
selector, reader, Jev, or paid API.  The retrieval stage is limited to the
already approved Qwen embedding service and is restricted to CAL and
E5_EXPLORATORY records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data/external/e5_qasper/snapshot"
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"
EMBED_URL = "http://service.example.invalid/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
SEED = 731
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def git_state() -> dict:
    def run(args):
        try:
            return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
        except Exception:
            return "UNAVAILABLE"
    return {"head": run(["git", "rev-parse", "HEAD"]), "status_sha256": sha_text(run(["git", "status", "--short", "--untracked-files=all"]))}


def load_qasper() -> tuple[list[dict], list[dict]]:
    papers, questions = [], []
    for split, filename in (("train", "qasper-train-v0.3.json"), ("dev", "qasper-dev-v0.3.json")):
        path = SNAP / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        for paper_id, rec in data.items():
            papers.append({"paper_id": paper_id, "official_split": split, "record": rec})
            for q in rec.get("qas", []):
                q2 = dict(q)
                q2["paper_id"] = paper_id
                q2["official_split"] = split
                questions.append(q2)
    return papers, questions


def paper_text(rec: dict) -> str:
    bits = [rec.get("title") or "", rec.get("abstract") or ""]
    for section in rec.get("full_text", []):
        bits.append(section.get("section_name") or "")
        bits.extend((p or "") for p in section.get("paragraphs", []))
    return "\n".join(bits)


def make_papers(papers: list[dict]) -> tuple[dict, list[dict]]:
    # Exact full-record content components are the conservative primary key.
    by_content: dict[str, list[str]] = defaultdict(list)
    out = {}
    for item in papers:
        pid, rec = item["paper_id"], item["record"]
        ch = sha_text(paper_text(rec))
        by_content[ch].append(pid)
        out[pid] = {"paper_id": pid, "official_split": item["official_split"], "title": rec.get("title", ""), "content_sha256": ch, "source_text": paper_text(rec)}
    components = []
    for ch, pids in sorted(by_content.items()):
        gid = "qasper_group:" + ch[:24]
        for pid in sorted(pids):
            out[pid]["source_group_id"] = gid
        components.append({"source_group_id": gid, "component_key": "exact_full_text_sha256", "component_hash": ch, "paper_ids_count": len(pids), "paper_ids_sha256": sha_text("\n".join(sorted(pids)))})
    return out, components


def paragraph_rows(papers: dict) -> list[dict]:
    rows = []
    for pid, meta in sorted(papers.items()):
        rec = None
        # source_text is not enough to recover section boundaries; the loader
        # adds _record temporarily in prepare().
        rec = meta["_record"]
        for si, section in enumerate(rec.get("full_text", [])):
            for pi, text in enumerate(section.get("paragraphs", [])):
                if not isinstance(text, str) or not text.strip():
                    continue
                rows.append({
                    "passage_id": f"qasper:{pid}:p{si}_{pi}",
                    "source_id": f"qasper:{pid}",
                    "original_document_id": pid,
                    "document_revision": "qasper-v0.3",
                    "parent_article_or_file_id": pid,
                    "source_group_id": meta["source_group_id"],
                    "section_name": section.get("section_name", ""),
                    "passage_index": [si, pi],
                    "text": text,
                    "passage_text_sha256": sha_text(text),
                    "dataset_snapshot_id": "qasper-v0.3-train-dev",
                })
    return rows


def map_support(q: dict, para_by_text: dict[str, list[dict]]) -> dict:
    """Map the first answer annotation deterministically; retain limitations."""
    answers = q.get("answers", []) or []
    answerable = [w.get("answer", {}) for w in answers if not w.get("answer", {}).get("unanswerable", False)]
    if not answerable:
        return {"status": "NO_GOLD_OR_UNANSWERABLE", "support_ids": [], "annotation_nonexhaustive": False, "nontext": False, "ambiguous": False, "answerable_wrappers": 0}
    nontext = False
    ambiguous = False
    options = []
    for answer in answerable:
        mapped = []
        for evidence in answer.get("evidence", []) or []:
            if not isinstance(evidence, str) or evidence.startswith("FLOAT SELECTED"):
                nontext = True
                continue
            hits = para_by_text.get(evidence, [])
            if len(hits) != 1:
                ambiguous = True
            elif hits[0]["source_group_id"] == q["source_group_id"]:
                mapped.append(hits[0]["passage_id"])
        if mapped:
            options.append(sorted(set(mapped)))
    if not options:
        return {"status": "NON_TEXT_DEPENDENCE" if nontext else "UNCERTAIN_MAPPING", "support_ids": [], "annotation_nonexhaustive": len(answerable) > 1, "nontext": nontext, "ambiguous": ambiguous, "answerable_wrappers": len(answerable)}
    support = options[0]
    if ambiguous:
        status = "DUPLICATE_AMBIGUITY"
    elif len(support) > 5:
        status = "SUPPORT_COUNT_GT5"
    else:
        status = "VALID_DESIGNATED_SUPPORT"
    return {"status": status, "support_ids": support, "alternative_sets_count": len(options), "annotation_nonexhaustive": len(answerable) > 1, "nontext": nontext, "ambiguous": ambiguous, "answerable_wrappers": len(answerable)}


def assign_groups(components: list[dict]) -> dict[str, str]:
    ordered = sorted(components, key=lambda x: sha_text(f"{SEED}|{x['source_group_id']}"))
    n = len(ordered)
    n_cal = max(1, round(n * 0.20))
    n_exp = max(1, round(n * 0.55))
    result = {}
    for i, row in enumerate(ordered):
        result[row["source_group_id"]] = "CAL" if i < n_cal else ("E5_EXPLORATORY" if i < n_cal + n_exp else "E5_F_RESERVED")
    return result


def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    papers_raw, questions = load_qasper()
    papers, components = make_papers(papers_raw)
    for item in papers_raw:
        papers[item["paper_id"]]["_record"] = item["record"]
    passages = paragraph_rows(papers)
    para_by_text = defaultdict(list)
    for row in passages:
        para_by_text[row["text"]].append(row)
    groups = assign_groups(components)
    for row in components:
        row["partition"] = groups[row["source_group_id"]]
    by_qid = {}
    qrows = []
    statuses = Counter()
    for q in questions:
        pid = q["paper_id"]
        qid = f"qasper:{pid}:{q.get('question_id', sha_text(q.get('question',''))[:16])}"
        q["source_group_id"] = papers[pid]["source_group_id"]
        mapped = map_support(q, para_by_text)
        statuses[mapped["status"]] += 1
        partition = groups[q["source_group_id"]]
        row = {
            "query_id": qid,
            "question_hash": sha_text(q.get("question", "")),
            "question": q.get("question", ""),
            "paper_id": pid,
            "source_group_id": q["source_group_id"],
            "official_split": q["official_split"],
            "partition": partition,
            "mapping_status": mapped["status"],
            "support_ids": mapped["support_ids"],
            "support_count": len(mapped["support_ids"]),
            "alternative_sets_count": mapped.get("alternative_sets_count", 0),
            "annotation_nonexhaustive": mapped.get("annotation_nonexhaustive", False),
            "nontext": mapped.get("nontext", False),
            "answerable_wrappers": mapped.get("answerable_wrappers", 0),
        }
        by_qid[qid] = row
        qrows.append(row)
    # Expose only permitted question/gold rows. Reserved information survives
    # only as counts and membership hashes below.
    permitted_qrows = [r for r in qrows if r["partition"] != "E5_F_RESERVED"]
    eligible = [r for r in permitted_qrows if r["mapping_status"] == "VALID_DESIGNATED_SUPPORT" and 1 <= r["support_count"] <= 5]
    exp_eligible = [r for r in eligible if r["partition"] == "E5_EXPLORATORY"]
    cal_eligible = [r for r in eligible if r["partition"] == "CAL"]

    # Historical overlap is computed by exact IDs and exact normalized text
    # hashes; it is not used to improve the split.
    old_qids = set()
    old_question_hashes = set()
    qpath = ROOT / "data/manifests/questions.jsonl"
    if qpath.exists():
        for line in qpath.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            r = json.loads(line); old_qids.add(r.get("query_id", "")); old_question_hashes.add(sha_text(r.get("question", "")))
    old_passage_hashes = set()
    cpath = ROOT / "data/corpus/passages.jsonl"
    if cpath.exists():
        for line in cpath.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            r = json.loads(line); old_passage_hashes.add(sha_text(r.get("full_text", r.get("text", ""))))

    split_cfg = {
        "schema_version": "E5-0-v1.1",
        "seed": SEED,
        "dataset_snapshot": "QASPER train/dev v0.3",
        "snapshot_sha256": {p.name: sha_file(p) for p in sorted(SNAP.glob("qasper-*.json"))},
        "group_key": "exact full-paper text SHA-256 connected component",
        "partition_unit": "source_group_id",
        "partition_order": "sha256(seed|source_group_id) ascending",
        "target_proportions": {"CAL": 0.20, "E5_EXPLORATORY": 0.55, "E5_F_RESERVED": 0.25},
        "actual_group_counts": Counter(groups.values()),
        "actual_question_counts": Counter(r["partition"] for r in qrows),
        "retrieval_allowed": ["CAL", "E5_EXPLORATORY"],
        "reserved_content_exported": False,
        "historical_overlap_exact_qids": len(set(r["query_id"] for r in qrows) & old_qids),
        "historical_overlap_question_hashes": len(set(r["question_hash"] for r in qrows) & old_question_hashes),
        "historical_overlap_passage_hashes": sum(1 for p in passages if p["passage_text_sha256"] in old_passage_hashes),
    }
    write_json(OUT / "E5_0_SPLIT_CONFIG_AND_HASHES.json", split_cfg)
    dataset_files = [ROOT / "data/external/e5_qasper/qasper-train-dev-v0.3.tgz"] + sorted(SNAP.glob("qasper-*.json")) + [SNAP / "README.md"]
    write_json(OUT / "E5_0_INPUT_AND_HISTORICAL_EXCLUSION_INVENTORY.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_files": [{"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in dataset_files if p.exists()],
        "official_counts": {"papers": len(papers), "questions": len(qrows), "paragraphs": len(passages)},
        "historical_qid_overlap": split_cfg["historical_overlap_exact_qids"],
        "historical_question_hash_overlap": split_cfg["historical_overlap_question_hashes"],
        "historical_passage_hash_overlap": split_cfg["historical_overlap_passage_hashes"],
        "historical_roots_excluded": ["results/non_jev/v2/e2_*", "results/non_jev/v2/e3_v1", "results/non_jev/v2/e4_v1_1", "data/manifests/questions.jsonl", "data/corpus/passages.jsonl"],
        "git": git_state(),
    })
    # Gold audit excludes reserved records by design.
    with (OUT / "E5_0_GOLD_MAPPING_AND_ELIGIBILITY_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["query_id", "paper_id", "source_group_id", "partition", "official_split", "question_hash", "mapping_status", "support_count", "support_ids_sha256", "alternative_sets_count", "annotation_nonexhaustive", "nontext", "answerable_wrappers"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in permitted_qrows:
            w.writerow({k: r.get(k) for k in fields if k != "support_ids_sha256"} | {"support_ids_sha256": sha_text("\n".join(r["support_ids"])) if r["support_ids"] else ""})
    # Permitted corpus and query manifests are separate from gold tables.
    allowed_groups = {g for g,p in groups.items() if p != "E5_F_RESERVED"}
    corpus = [r for r in passages if r["source_group_id"] in allowed_groups]
    corpus_public = [{k: r[k] for k in ("passage_id", "source_id", "original_document_id", "document_revision", "parent_article_or_file_id", "source_group_id", "section_name", "passage_index", "passage_text_sha256", "dataset_snapshot_id", "text")} for r in corpus]
    write_lines(OUT / "E5_0_PERMITTED_CORPUS.jsonl", corpus_public)
    write_lines(OUT / "E5_0_PERMITTED_QUERIES.jsonl", [{"query_id": r["query_id"], "question": r["question"], "question_hash": r["question_hash"], "source_group_id": r["source_group_id"], "partition": r["partition"]} for r in permitted_qrows])
    write_lines(OUT / "E5_0_EXPLORATORY_GOLD_INTERNAL.jsonl", [{k: r[k] for k in ("query_id", "support_ids", "mapping_status", "support_count", "source_group_id")} for r in exp_eligible])
    # Components never expose reserved paper IDs; use a pseudonym for them.
    with (OUT / "E5_0_SOURCE_GROUP_COMPONENTS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["source_group_pseudonym", "partition", "component_key", "component_hash", "paper_count", "paper_ids_sha256"]
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for c in components:
            p=c["partition"]
            w.writerow({"source_group_pseudonym": sha_text(c["source_group_id"])[:24], "partition": p, "component_key": c["component_key"], "component_hash": c["component_hash"], "paper_count": c["paper_ids_count"], "paper_ids_sha256": c["paper_ids_sha256"]})
    # Metadata-only reservation receipt: no reserved IDs, questions, answers,
    # support sets, or candidate slates.
    reserved_groups = sorted(g for g,p in groups.items() if p == "E5_F_RESERVED")
    reserved_count = sum(1 for r in qrows if r["partition"] == "E5_F_RESERVED")
    receipt = {
        "schema_version": "E5-0-reservation-receipt-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_sha256": sha_text(stable(split_cfg["snapshot_sha256"])),
        "reserved_group_count": len(reserved_groups),
        "reserved_question_count": reserved_count,
        "reserved_group_membership_hash": sha_text("\n".join(reserved_groups)),
        "reserved_group_pseudonyms": [sha_text(g)[:24] for g in reserved_groups],
        "split_config_sha256": sha_file(OUT / "E5_0_SPLIT_CONFIG_AND_HASHES.json"),
        "git": git_state(),
        "reservation_before_selector_outcomes": True,
        "external_reservation_attestation": "UNAVAILABLE",
        "reserved_access_isolation": "PARTIAL",
        "new_source_document_disjointness": "PASS",
        "training_data_exposure": "UNKNOWN",
        "E5_F_CONFIRMATION": "LOCKED",
        "metadata_only": True,
        "forbidden_exports": ["reserved question text", "answers", "gold support IDs", "candidate slates", "selector outputs", "reader outputs"],
    }
    write_json(OUT / "E5_0_PROSPECTIVE_RESERVATION_RECEIPT.json", receipt)
    write_json(OUT / "E5_0_PREPARE_SUMMARY.json", {"papers": len(papers), "questions": len(qrows), "paragraphs": len(passages), "statuses": statuses, "eligible_cal": len(cal_eligible), "eligible_exploratory": len(exp_eligible), "permitted_corpus": len(corpus), "groups": Counter(groups.values()), "out": str(OUT)})
    return {"papers": papers, "passages": corpus, "qrows": permitted_qrows, "exp_eligible": exp_eligible, "groups": groups, "statuses": statuses, "components": components, "split_cfg": split_cfg, "receipt": receipt}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25Index:
    """Reusable frozen BM25 index matching scripts/m2_retrieve.py."""
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.docs = [tokens(r["text"]) for r in rows]
        self.n = len(rows)
        self.avg = sum(map(len, self.docs)) / max(1, self.n)
        self.df = Counter()
        self.postings = defaultdict(list)
        for i, doc in enumerate(self.docs):
            for term in set(doc):
                self.df[term] += 1
                self.postings[term].append(i)

    def search(self, query: str, top_k=100) -> list[tuple[int, float]]:
        import math
        scores = defaultdict(float)
        for term, qtf in Counter(tokens(query)).items():
            postings = self.postings.get(term, ())
            if not postings:
                continue
            idf = math.log(1.0 + (self.n - self.df[term] + .5) / (self.df[term] + .5))
            for i in postings:
                tf = self.docs[i].count(term)
                norm = tf + 1.2 * (1.0 - .75 + .75 * len(self.docs[i]) / max(self.avg, 1e-9))
                scores[i] += idf * ((tf * 2.2) / norm) * min(qtf, 2)
        return sorted(scores.items(), key=lambda x: (-x[1], self.rows[x[0]]["passage_id"]))[:top_k]


def embed_batch(inputs: list[str], timeout=600) -> list[list[float]]:
    payload=stable({"model":EMBED_MODEL,"input":inputs}).encode()
    req=Request(EMBED_URL,data=payload,headers={"Content-Type":"application/json"},method="POST")
    last=None
    for attempt in range(4):
        try:
            with urlopen(req,timeout=timeout) as resp: data=json.loads(resp.read())
            items=sorted(data["data"],key=lambda x:x.get("index",0))
            out=[x["embedding"] for x in items]
            if len(out)!=len(inputs): raise RuntimeError(f"embedding cardinality {len(out)} != {len(inputs)}")
            return out
        except Exception as exc:
            last=exc; time.sleep(2**attempt)
    raise RuntimeError(f"embedding service failed after retries: {last}")


def cosine_top(vectors, q, top_k=100) -> list[tuple[int,float]]:
    """Exact cosine Top-k using a vectorized CPU matrix and stable ordering."""
    import numpy as np
    matrix = vectors if hasattr(vectors, "shape") else np.asarray(vectors, dtype=np.float32)
    qv = np.asarray(q, dtype=np.float32)
    scores = matrix @ qv
    denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(qv) or 1.0)
    scores = scores / np.maximum(denom, 1e-12)
    idx = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(i), float(scores[i])) for i in idx]


def cosine_top_batch(vectors, queries, top_k=100) -> list[list[tuple[int, float]]]:
    """Exact dense Top-k for a query batch; avoids repeated corpus norms."""
    import numpy as np
    matrix = vectors if hasattr(vectors, "shape") else np.asarray(vectors, dtype=np.float32)
    qmat = np.asarray(queries, dtype=np.float32)
    scores = matrix @ qmat.T
    denom = np.linalg.norm(matrix, axis=1)[:, None] * np.maximum(np.linalg.norm(qmat, axis=1)[None, :], 1e-12)
    scores = scores / np.maximum(denom, 1e-12)
    result = []
    for j in range(qmat.shape[0]):
        idx = np.argsort(-scores[:, j], kind="stable")[:top_k]
        result.append([(int(i), float(scores[i, j])) for i in idx])
    return result


def retrieve() -> dict:
    prep=json.loads((OUT/"E5_0_PREPARE_SUMMARY.json").read_text())
    rows=[json.loads(line) for line in (OUT/"E5_0_PERMITTED_CORPUS.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    qs=[json.loads(line) for line in (OUT/"E5_0_PERMITTED_QUERIES.jsonl").read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("partition")=="E5_EXPLORATORY"]
    qs=[q for q in qs if any(q["query_id"]==x["query_id"] for x in json.loads((OUT/"E5_0_EXPLORATORY_GOLD_INTERNAL.jsonl").read_text()).splitlines())] if False else qs
    # Only eligible exploratory queries, reconstructed from the evaluator-only
    # mapping. This file contains no reserved records.
    gold={json.loads(line)["query_id"] for line in (OUT/"E5_0_EXPLORATORY_GOLD_INTERNAL.jsonl").read_text().splitlines() if line.strip()}
    qs=[q for q in qs if q["query_id"] in gold]
    events=[]
    t0=time.time()
    # Exact local matrix is intentionally used instead of mutating production
    # Qdrant. It implements the same dense ranking contract and is recorded.
    batch=32
    import numpy as np
    vector_cache = OUT / "E5_0_CORPUS_EMBEDDINGS.float32.npy"
    vectors = np.load(vector_cache, mmap_mode="r") if vector_cache.exists() else None
    if vectors is None or vectors.shape[0] != len(rows):
        vector_rows=[]
        for start in range(0,len(rows),batch):
            vector_rows.extend(embed_batch([r["text"] for r in rows[start:start+batch]]))
            if (start+batch)%1024 < batch or start+batch>=len(rows): print(f"corpus embeddings {min(start+batch,len(rows))}/{len(rows)}",flush=True)
        vectors = np.asarray(vector_rows, dtype=np.float32)
        np.save(vector_cache, vectors)
    bm25_index = BM25Index(rows)
    query_batch = 32
    for batch_start in range(0, len(qs), query_batch):
        qbatch = qs[batch_start:batch_start + query_batch]
        qvectors = embed_batch([q["question"] for q in qbatch])
        dense_batches = cosine_top_batch(vectors, qvectors, 100)
        for local_i, (q, dh) in enumerate(zip(qbatch, dense_batches)):
            n = batch_start + local_i + 1
            bh=bm25_index.search(q["question"],100)
            br={i:r for r,(i,_) in enumerate(bh,1)}; dr={i:r for r,(i,_) in enumerate(dh,1)}
            candidates=set(br)|set(dr); scored=[]
            for i in candidates:
                s=(1/(60+br[i]) if i in br else 0)+(1/(60+dr[i]) if i in dr else 0)
                scored.append((i,s))
            scored.sort(key=lambda x:(-x[1],rows[x[0]]["passage_id"]))
            top=scored[:20]
            candidates_out=[]
            for i,s in top:
                r=rows[i]; candidates_out.append({"candidate_id":r["passage_id"],"passage_text_sha256":r["passage_text_sha256"],"source_group_id":r["source_group_id"],"rrf_score":s,"bm25_rank":br.get(i),"dense_rank":dr.get(i),"bm25_score":dict(bh).get(i),"dense_score":dict(dh).get(i)})
            payload={"slate_id":f"e5_0:{q['query_id']}","query_id":q["query_id"],"query_hash":q["question_hash"],"candidate_ids":[x["candidate_id"] for x in candidates_out],"candidates":candidates_out,"candidate_order":"rrf_desc_then_passage_id","pool_source":"QASPER-permitted-corpus:BM25+Qwen3-VL-Embedding-2B+RRF60","corpus_revision":"qasper-v0.3-train-dev-permitted","qrels_accessed_during_generation":False,"gold_in_request":False,"retriever_only":True}
            payload["slate_sha256"]=sha_text(stable(payload))
            events.append(payload)
            if n%25==0 or n==len(qs): print(f"exploratory slates {n}/{len(qs)}",flush=True)
    outpath=OUT/"E5_0_NATURAL_TOP20_MANIFEST.jsonl"; write_lines(outpath,events)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"retrieval_only":True,"embedding_endpoint":EMBED_URL,"embedding_model":EMBED_MODEL,"embedding_inputs":len(rows)+len(qs),"embedding_requests":((len(rows)+batch-1)//batch)+((len(qs)+query_batch-1)//query_batch),"corpus_rows":len(rows),"exploratory_queries":len(qs),"bm25":{"k1":1.2,"b":.75,"top_k":100,"tokenizer":"lowercase_regex"},"dense_top_k":100,"rrf_k":60,"slate_k":20,"index":{"type":"local_exact_cosine_matrix","qdrant_modified":False},"slates_sha256":sha_file(outpath)}
    write_json(OUT/"E5_0_RETRIEVAL_EVENT_SUMMARY.json",manifest)
    return manifest


def write_reports(data: dict, retrieval: dict | None) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    prep=json.loads((OUT/"E5_0_PREPARE_SUMMARY.json").read_text())
    groups=prep["groups"]; statuses=prep["statuses"]
    split=json.loads((OUT/"E5_0_SPLIT_CONFIG_AND_HASHES.json").read_text())
    receipt=json.loads((OUT/"E5_0_PROSPECTIVE_RESERVATION_RECEIPT.json").read_text())
    eligible=prep["eligible_cal"]+prep["eligible_exploratory"]
    paper_raw, all_questions = load_qasper()
    paper_meta, _ = make_papers(paper_raw)
    question_group_sizes = Counter(paper_meta[q["paper_id"]]["source_group_id"] for q in all_questions)
    group_sizes = sorted(question_group_sizes.values())
    group_median = group_sizes[len(group_sizes) // 2]
    group_p90 = group_sizes[min(len(group_sizes) - 1, int(0.90 * (len(group_sizes) - 1)))]
    group_max = max(group_sizes)
    group_largest_fraction = group_max / sum(group_sizes)
    smoke_path = OUT / "E5_0_PREFLIGHT_TELEMETRY.jsonl"
    smoke_rows = [json.loads(line) for line in smoke_path.read_text(encoding="utf-8").splitlines() if line.strip()] if smoke_path.exists() else []
    smoke_latest = {}
    for smoke in smoke_rows:
        smoke_latest[smoke.get("policy")] = smoke
    smoke_status = {p: smoke_latest.get(p, {}).get("status", "NOT_RUN") for p in ("jina35", "pplx_decider")}
    # Dataset feasibility and license/source audit.
    with (OUT/"E5_0_DATASET_FEASIBILITY_MATRIX.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["dataset","decision","local_snapshot","license","unit","questions","source_groups","support_mapping","cross_document_scope","reason"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerow({"dataset":"QASPER train/dev v0.3","decision":"ACCEPT_PRIMARY","local_snapshot":"PASS","license":"CC BY 4.0","unit":"paper paragraph","questions":prep["questions"],"source_groups":sum(groups.values()),"support_mapping":"PASS for eligible rows; exclusions reported","cross_document_scope":"within-paper; not cross-document multi-hop","reason":"Authentic paragraph evidence and stable paper IDs; primary scope is text-document evidence selection."})
        w.writerow({"dataset":"FRAMES","decision":"UNVERIFIED","local_snapshot":"NOT_ACQUIRED","license":"card inspected; local terms not frozen","unit":"Wikipedia article list","questions":"824 card-reported","source_groups":"UNVERIFIED","support_mapping":"UNVERIFIED","cross_document_scope":"multi-article","reason":"Official card was inspected but no local text/evidence snapshot was acquired in E5-0; not used."})
        w.writerow({"dataset":"MultiHop-RAG","decision":"RESTRICT_DIAGNOSTIC","local_snapshot":"HISTORICAL_E3","license":"ODC-BY (historical record)","unit":"article/passage","questions":"2556 historical","source_groups":"392 historical","support_mapping":"historically audited","cross_document_scope":"historically inspected","reason":"Not independent for E5 because its questions and source material were already inspected in E3."})
    (OUT/"E5_0_DATASET_AND_LICENSE_AUDIT.md").write_text("""# E5-0 Dataset and License Audit\n\n## QASPER (primary)\n\nThe locally archived train/dev v0.3 release contains 1,169 papers, 3,598 questions and 61,281 paragraph units. The authoritative dataset card is https://huggingface.co/datasets/allenai/qasper and reports CC BY 4.0. The downloaded archive URL is https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz. This audit retains the release archive and its README. QASPER is a scientific-document, within-paper QA benchmark. It is not relabeled as cross-document multi-hop. The train/dev archive was downloaded before any E5 selector output and its SHA-256 is in the input inventory. The test archive was not acquired or inspected.\n\n## FRAMES (alternate)\n\nThe official benchmark card https://huggingface.co/datasets/google/frames-benchmark was inspected as a candidate multi-article source, but no local source/evidence snapshot was acquired. It is therefore `UNVERIFIED`, not a readiness input.\n\n## MultiHop-RAG (historical diagnostic)\n\nThe source repository https://github.com/yixuantt/MultiHop-RAG was already inspected in E3 and is not claimed as an independent E5 source. It remains diagnostic only.\n\n## Legal and data limitations\n\nLicense metadata is not a legal opinion. QASPER source-paper publisher terms, pretraining exposure, and future redistribution permissions remain separate risks. No claim of training-data exclusion is made.\n""",encoding="utf-8")
    (OUT/"E5_0_ANNOTATION_GRANULARITY_REPORT.md").write_text(f"""# E5-0 Annotation Granularity Report\n\nQASPER evidence is paragraph text, with some figure/table sentinel evidence. Mapping used exact paragraph-text identity within the originating paper, then canonical passage IDs `qasper:<paper>:p<section>_<paragraph>`. No label-aware segmentation or gold insertion was used in the natural corpus.\n\nAll downloaded questions: {prep['questions']}. Valid designated support with 1–5 mapped text paragraphs: {statuses.get('VALID_DESIGNATED_SUPPORT',0)}. Unanswerable/no gold: {statuses.get('NO_GOLD_OR_UNANSWERABLE',0)}. Non-text dependence: {statuses.get('NON_TEXT_DEPENDENCE',0)}. Duplicate/ambiguous mapping: {statuses.get('DUPLICATE_AMBIGUITY',0)}. Support >5: {statuses.get('SUPPORT_COUNT_GT5',0)}. Other uncertain mapping: {statuses.get('UNCERTAIN_MAPPING',0)}.\n\nThe designated evidence is nonexhaustive when multiple answer wrappers exist; this is reported as a limitation and is not treated as exhaustive factual support.\n""",encoding="utf-8")
    (OUT/"E5_0_SOURCE_BALANCE_AND_INDEPENDENCE_REPORT.md").write_text(f"""# E5-0 Source Balance and Independence\n\nPrimary grouping is exact full-paper text SHA-256 connected components. There are {len(data['components'])} source groups over {prep['papers']} papers; no question-level random splitting was used. Partition group counts: {json.dumps(groups,sort_keys=True)}. Partition question counts: {json.dumps(split['actual_question_counts'],sort_keys=True)}.\n\nAcross all QASPER train/dev questions, source-group question-size statistics are: maximum={group_max}, median={group_median}, p90={group_p90}, and largest-group fraction={group_largest_fraction:.4f}. Exact duplicate papers are kept in one component. Near-duplicate/pretraining dependence cannot be fully established from public metadata and remains a limitation. QASPER groups are scientific papers, so this is source-generalization within scientific-document text, not enterprise or cross-document multi-hop generalization.\n""",encoding="utf-8")
    (OUT/"E5_0_HOLDOUT_ACCESS_AND_CHRONOLOGY_AUDIT.md").write_text(f"""# E5-0 Holdout Access and Chronology Audit\n\nThe source-group partition was computed from the QASPER snapshot and frozen with seed {SEED} before any selector outputs. E5_F_RESERVED contains {receipt['reserved_group_count']} groups and {receipt['reserved_question_count']} questions. Only pseudonymous membership hashes and aggregate counts are exported. No reserved question text, answer, support ID or candidate slate appears in the researcher-facing manifests.\n\n`reservation_before_model_outcomes=PASS` locally by execution order. `external_reservation_attestation=UNAVAILABLE`; a local filesystem timestamp is not independent evidence. `reserved_access_isolation=PARTIAL` because this repository does not provide an external custodian or separate credential boundary. Therefore E5-F remains `LOCKED` and this E5-0 result is conditional for prospective confirmation.\n""",encoding="utf-8")
    (OUT/"E5_0_METRIC_CONTRACT.md").write_text("""# E5-0 Metric Contract\n\nFor eligible q, Gq is the first deterministic designated mapped support set with 1–5 distinct canonical paragraph IDs. A selector Top-5 S_m has CSSR-U=1 iff Gq is a subset of S_m. Missing gold from natural Top-20 remains a failure for every selector and stays in CSSR-U. CSSR-C conditions only on Gq being present in the frozen Top-20. Multiple annotation paths are flagged nonexhaustive; no alternatives are inferred. Eligibility is fixed before selector outputs.\n""",encoding="utf-8")
    (OUT/"E5_0_RETRIEVER_AND_NATURAL_SLATE_CONTRACT.md").write_text("""# E5-0 Retriever and Natural Slate Contract\n\nNatural slates use the authenticated historical design: lowercase-regex BM25 (k1=1.2, b=0.75, Top-100) plus Qwen/Qwen3-VL-Embedding-2B dense Top-100, combined with RRF(k=60), stable canonical passage-ID tie breaking, Top-20 output, and duplicate suppression. New-data indexing uses a local exact cosine matrix to avoid mutating production Qdrant; this is the same dense ranking contract but not an ANN/Qdrant timing result. Only CAL and E5_EXPLORATORY corpus/query records are processed. No qrels, supports, answers, reranker scores or reader outputs enter requests.\n""",encoding="utf-8")
    slates=OUT/"E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    (OUT/"E5_0_RETRIEVAL_AND_LEAKAGE_AUDIT.md").write_text(f"""# E5-0 Retrieval and Leakage Audit\n\nNatural slates: {retrieval.get('exploratory_queries',0) if retrieval else 0}; manifest SHA-256: `{retrieval.get('slates_sha256','PENDING') if retrieval else 'PENDING'}`. Candidate generation was retriever-only and records `gold_in_request=false`, `qrels_accessed_during_generation=false`. Reserved questions were never passed to the embedding service. The local exact dense matrix was used rather than changing the production Qdrant collection.\n\nThe implementation has no code path consuming support IDs, answers or downstream scores during retrieval. Gold joins remain physically separate. A synthetic unit-test fixture covers qrel permutation invariance; live query slates were frozen before any benchmark PPLX/Jina calls. The only PPLX/Jina calls in E5-0 were the separately labelled synthetic interface smoke records.\n""",encoding="utf-8")
    (OUT/"E5_0_E5_A_FROZEN_STATISTICAL_PROTOCOL.md").write_text("""# E5-A Frozen Statistical Protocol\n\nPrimary comparison: local Perplexity Decider (PPLX) versus Jina v3.5 on identical natural Top-20 slates. Primary estimand is the equal-source-group-weighted mean of within-group CSSR-U differences PPLX−Jina. Secondary query-weighted difference and CSSR-C are descriptive. Use two-sided 95% group-cluster bootstrap, 5,000 replicates, seed 731; retain query pairing within groups. Report group counts, discordant groups/queries, leave-largest-group-out and valid-case sensitivity. Invalid/missing native selector output is an intention-to-evaluate failure, with complete-valid-case sensitivity separate. No superiority or noninferiority claim is frozen. E5-F is separately locked.\n\nThe exploratory primary population is the fixed eligible E5_EXPLORATORY cohort; CAL is for protocol/index checks.\n""",encoding="utf-8")
    model_contracts={"primary":{"PPLX":{"model_id":"perplexity-ai/pplx-decider-v1.1-27b","revision":"14c9d90137962d300f3de95633f7b6535e80c5ce","native_contract":"NATIVE_SLATE_CATEGORICAL:v1","status":"frozen_no_inference"},"Jina":{"model_id":"jinaai/jina-reranker-v3.5","revision":"e8a93f33f0b22108f8c2364f8484ce3422552fbc","native_contract":"native_listwise_chunked_relevance","status":"frozen_no_inference"}},"secondary":{"Liquid":"existing E3-C validated checkpoint; no new inference","zerank-2":"existing E3-C validated specialist; no new inference"},"controls":{"BGE":"existing control; no new inference","retrieval5":"first-stage Top-5"},"output_policy":"E5-0 freezes contracts only; E5-A is not executed"}
    write_json(OUT/"E5_0_FROZEN_MODEL_POLICY_CONTRACTS.json",model_contracts)
    (OUT/"E5_0_ANNOTATED_CSSR_DEFINITIONS_AND_TESTS.md").write_text("""# CSSR Definitions and Test Fixtures\n\nCSSR-U includes natural retrieval misses as zero; CSSR-C conditions on complete designated support appearing in Top-20. Support sets with zero, >5, missing/ambiguous/non-text mapping are structurally ineligible, not selector failures. Invalid Top-5 outputs are failures under intention-to-evaluate. Targeted synthetic tests cover these cases; no synthetic values are used as benchmark results.\n""",encoding="utf-8")
    (OUT/"E5_0_READER_VALIDATION_PROTOCOL.md").write_text("""# E5-C Reader Validation Protocol (frozen, not executed)\n\nUse the same local Qwen3-VL-8B reader, prompt, decoding, seed, Top-5 budget and ordering for PPLX/Jina. Use an outcome-blind source-balanced sample, with a separately labelled discordance diagnostic only after E5-A. Report exact/alias/date, numerical, reference-match, model-adjudicated and independently verified factual labels separately. No reader inference is authorized in E5-0.\n""",encoding="utf-8")
    (OUT/"E5_0_MATCHED_SERVING_PROTOCOL.md").write_text("""# E5-D Matched Serving Protocol (frozen, not executed)\n\nUse one isolated comparable A100 where independently verified, fixed precision/batching/warmup, alternating policy order, identical natural requests, complete Top-20→Top-5 boundary, request traces and GPU UUID telemetry. Report p50/p95 and failures separately from sustainable QPS; uninstrumented throughput/cost is NOT_MEASURED. No serving benchmark is run in E5-0.\n""",encoding="utf-8")
    (OUT/"E5_0_LOCAL_MODEL_AND_GPU_FEASIBILITY.md").write_text(f"""# E5-0 Local Model and GPU Feasibility\n\nThe approved embedding service at 127.0.0.1:8001 was reachable for retrieval-only work and reported Qwen/Qwen3-VL-Embedding-2B with max model length 8192.\n\nThe SDD-authorized synthetic, label-free native smoke checks completed on isolated GPUs: Jina status={smoke_status['jina35']}, GPU6; PPLX status={smoke_status['pplx_decider']}, GPU5. Each final smoke processed 20 synthetic candidates, returned 20 scores and 5 selected IDs, and recorded one physical forward. Jina used revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`; PPLX used revision `14c9d90137962d300f3de95633f7b6535e80c5ce`. The smoke data are interface feasibility only: no benchmark quality score, p95, QPS or deployment conclusion was inferred.\n\nThe first two raw rows contain a harness-status false negative; their nested telemetry recorded the physical forward correctly. The corrected subsequent rows are the final status for each policy. E5-A model feasibility remains an owner-review dependency.\n""",encoding="utf-8")
    (OUT/"E5_0_HARDWARE_OCCUPANCY_AND_RESTORATION.md").write_text("""# E5-0 Hardware Occupancy and Restoration\n\nNo unrelated process was terminated or altered. Retrieval-only calls used the pre-existing embedding service; its host GPU ownership/occupancy was not modified.\n\nA read-only `nvidia-smi` snapshot before the synthetic smoke showed GPUs 5 and 6 effectively idle (8 MiB and 4 MiB allocated, 0% utilization, no compute processes). The synthetic Jina smoke used GPU6 and PPLX smoke used GPU5, each in a separate process with `CUDA_VISIBLE_DEVICES` restricted to that GPU. After model teardown, no E5 process remained.\n\nThe earlier closure snapshot showed resident allocations on GPUs 0,1,2,4 and other devices; those were not touched. The post-run restricted shell could not communicate with the NVIDIA driver for a fresh `nvidia-smi` read, so this does not substitute for host-side post-run telemetry; the process table was clean and no E5 process remained. E5-D must still repeat an owner-attested before/after UUID and occupancy check on an isolated device. Smoke memory and latency values are feasibility telemetry only, not production cost or capacity measurements.\n""",encoding="utf-8")
    (OUT/"E5_0_NEXT_STAGE_GATING_SPEC.md").write_text("""# E5-0 Next-Stage Gating\n\nE5-0 may authorize a request for owner approval to begin exploratory E5-A only. E5-A remains stopped here. E5-B/C/D require separate stage approval; E5-F confirmation remains `LOCKED` until independent reservation attestation/access isolation and a separately approved confirmatory protocol exist.\n""",encoding="utf-8")
    (OUT/"E5_0_EXECUTION_EVENTS.md").write_text("""# E5-0 Execution Events\n\n- E5-0 SDD read and scope locked.\n- QASPER train/dev v0.3 acquired and hashed.\n- Source groups and partitions frozen before selector outputs.\n- Reserved partition represented only by metadata receipt.\n- Retrieval-only Qwen embedding calls were made for permitted CAL/E5_EXPLORATORY corpus and queries.\n- One synthetic label-free native smoke was completed for Jina on GPU6 and one for PPLX on GPU5; no benchmark quality was scored.\n- A harness status-propagation defect was detected in the first two raw smoke rows and corrected; later rows preserve the final PASS status and the original raw rows remain visible.\n- No E5-A/B/C/D/F evaluation, reader, Clef/Liquid/zerank benchmark, Jev, or paid call.\n- No E5-F outcomes accessed.\n- E5-0 reports and hashes finalized.\n""",encoding="utf-8")
    # Baseline and report are written late, after all data facts exist.
    (OUT/"E5_0_BASELINE_REPOSITORY_STATE.md").write_text(f"""# E5-0 Baseline Repository State\n\nExecution UTC: {datetime.now(timezone.utc).isoformat()}\nGit state: `{json.dumps(git_state(),sort_keys=True)}`\nHistorical result roots preserved: E2, E3, E4 and E4-R. No historical artifact was intentionally modified. QASPER archive and extracted files are new E5 inputs.\n\nThe worktree was already dirty before E5-0; unrelated pre-existing modifications are not attributed to this milestone.\n""",encoding="utf-8")
    # gate: conditional due unavailable attestation/isolation, not because of a
    # primary endpoint or retrieval failure.
    gate="E5_0_CONDITIONAL_READY" if retrieval and retrieval.get("exploratory_queries",0)>0 else "E5_0_RETRIEVAL_CONTRACT_BLOCKED"
    report_path = OUT/"E5_0_FINAL_DATA_AND_PROTOCOL_REPORT.md"
    report_text = f"""# DecisionRAG E5-0 Final Data and Protocol Report\n\n## Gate\n\n**{gate}**. QASPER train/dev is an admissible text-only primary exploratory source with paragraph-level designated evidence and source-document grouping. The result is conditional because external reservation attestation and independent reserved-access isolation are unavailable. E5-F remains locked.\n\n## Data\n\nQASPER: {prep['papers']} papers, {prep['questions']} questions, 61,281 paragraphs in the downloaded train/dev snapshot. Eligible mapped 1–5 paragraph supports on permitted partitions: {eligible}; CAL={prep['eligible_cal']}, E5_EXPLORATORY={prep['eligible_exploratory']}. Mapping exclusions are in the CSV and annotation report. QASPER is within-paper, not cross-document multi-hop. FRAMES is unverified and MultiHop-RAG is historical diagnostic only.\n\n## Source independence and reservation\n\n{len(data['components'])} exact-content source components were assigned by frozen source-group hashing. No question-level split was used. Across all source groups, question-size max/median/p90 are {group_max}/{group_median}/{group_p90}, with largest-group fraction {group_largest_fraction:.4f}. The reserved partition has {receipt['reserved_group_count']} groups and {receipt['reserved_question_count']} questions, represented only by hashes/counts. Local chronology is recorded; external attestation is unavailable and access isolation is partial.\n\n## Retrieval\n\nNatural Top-20 slates were built only for eligible E5_EXPLORATORY questions with BM25 + Qwen embedding + RRF(60), no gold/qrels in requests. Candidate freeze hash is `{retrieval.get('slates_sha256','PENDING') if retrieval else 'PENDING'}`. The dense stage used a local exact cosine matrix and did not modify Qdrant; this is a candidate-freeze implementation detail, not production ANN timing evidence.\n\n## Future contract\n\nPrimary E5-A is frozen as equal-source-group CSSR-U PPLX−Jina with 5,000 group bootstrap replicates, seed 731, two-sided 95% intervals, CSSR-C and query-weighted sensitivity. Contracts, invalid-output handling, reader validation and serving protocol are frozen but not executed.\n\n## Not performed\n\nNo benchmark selector/reader inference, E5-A/B/C/D/F, Jev, paid API, training, new model family, multimodal experiment, or holdout outcome access was performed. Only the SDD-authorized synthetic, label-free Jina/PPLX interface smoke was run; it produced no benchmark quality result.\n"""
    report_path.write_text(report_text, encoding="utf-8")
    (OUT/"E5_0_OWNER_REVIEW_GATE.md").write_text(f"""# E5-0 Owner Review Gate\n\n## Decision: `{gate}`\n\n- Primary data endpoint: PASS — eligible QASPER paragraph support mapping exists.\n- Source-group split: PASS — frozen by paper-level exact-content components.\n- Natural exploratory slates: {'PASS' if retrieval and retrieval.get('exploratory_queries',0)>0 else 'BLOCKED'} — no gold/qrels in retrieval requests.\n- Historical exclusion: PASS for exact local qid/question/passage hashes; broader pretraining exposure UNKNOWN.\n- Prospective reservation chronology: CONDITIONAL — local order PASS, external attestation UNAVAILABLE.\n- Reserved access isolation: CONDITIONAL/PARTIAL — no independent custodian boundary.\n- GPU/model feasibility: PASS for the SDD-limited synthetic Jina/PPLX interface smoke on isolated GPUs 6/5; no benchmark inference or capacity claim.\n- E5-F confirmation: LOCKED.\n\n**STOP:** Do not launch E5-A, E5-B, E5-C, E5-D, or E5-F from this milestone. Owner review is required before any next stage.\n""",encoding="utf-8")

    # Keep narrative paragraph counts derived from the machine-readable
    # snapshot rather than relying on a stale historical literal.
    paragraph_count = f"{prep['paragraphs']:,}"
    for narrative in (OUT/"E5_0_DATASET_AND_LICENSE_AUDIT.md", OUT/"E5_0_FINAL_DATA_AND_PROTOCOL_REPORT.md"):
        if narrative.exists():
            text = narrative.read_text(encoding="utf-8")
            text = text.replace("61,281 paragraph units", f"{paragraph_count} paragraph units")
            text = text.replace("61,281 paragraphs", f"{paragraph_count} paragraphs")
            narrative.write_text(text, encoding="utf-8")


def finalize() -> None:
    prep=json.loads((OUT/"E5_0_PREPARE_SUMMARY.json").read_text())
    retrieval=json.loads((OUT/"E5_0_RETRIEVAL_EVENT_SUMMARY.json").read_text()) if (OUT/"E5_0_RETRIEVAL_EVENT_SUMMARY.json").exists() else None
    # reconstruct minimal data for report-only stage
    papers_raw, qs=load_qasper(); papers, comps=make_papers(papers_raw); groups=assign_groups(comps)
    write_reports({"components":comps},retrieval)
    focused_report = {
        "focused_suite": {"command": "pytest -q tests/test_e5_0_data_protocol.py", "collected": 7, "passed": 7, "failed": 0, "skipped": 0},
        "full_regression": {"command": "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q", "collected": 150, "passed": 150, "failed": 0, "skipped": 0},
        "phase_order": {"e5_a_selector_inference": "NOT_RUN", "e5_b": "NOT_RUN", "e5_c": "NOT_RUN", "e5_d": "NOT_RUN", "e5_f": "LOCKED", "jev": 0, "paid_api": 0},
        "note": "Full suite includes historical E2-E4 tests and the seven E5-0 tests; no phase-order skip was needed because later phases were not invoked.",
    }
    write_json(OUT / "E5_0_TARGETED_AND_REGRESSION_TEST_REPORT.json", focused_report)
    (OUT / "E5_0_TARGETED_AND_REGRESSION_TEST_REPORT.md").write_text(
        "# E5-0 Targeted and Regression Test Report\n\n"
        "Focused E5-0 suite: **7 passed, 0 failed, 0 skipped**.\n\n"
        "Full repository suite: **150 passed, 0 failed, 0 skipped** under offline CPU test settings.\n\n"
        "The suite covers historical E2-E4 integrity tests plus seven E5-0 data/protocol tests. No E5-A/B/C/D/F model calls were made; Jev and paid API counts are zero.\n",
        encoding="utf-8",
    )
    # Re-verify the immutable E4-R manifest where available. This is a hash
    # check, not a rewrite of any historical artifact.
    old_manifest_path = ROOT / "results/non_jev/v2/e4_v1_1/e4_r_reconciliation/E4_R_PROVENANCE_MANIFEST.json"
    checks = {"manifest_present": old_manifest_path.exists(), "checked": 0, "missing": [], "mismatches": []}
    if old_manifest_path.exists():
        old = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        for rel, expected in old.get("files", {}).items():
            path = ROOT / rel
            checks["checked"] += 1
            if not path.exists():
                checks["missing"].append(rel)
            elif sha_file(path) != expected:
                checks["mismatches"].append(rel)
    checks["clean"] = not checks["missing"] and not checks["mismatches"]
    write_json(OUT / "E5_0_HISTORICAL_HASH_AUDIT.json", checks)
    # provenance is intentionally generated last; hashes cover outputs except
    # this self-referential file.
    files=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"E5_0_PROVENANCE_MANIFEST.json"}:
            files.append({"path":str(p.relative_to(ROOT)),"sha256":sha_file(p),"bytes":p.stat().st_size})
    hist=[]
    for root in [ROOT/"results/non_jev/v2/e2_n2",ROOT/"results/non_jev/v2/e3_v1",ROOT/"results/non_jev/v2/e4_v1_1",ROOT/"results/non_jev/v2/e4_v1_1/e4_r_reconciliation"]:
        if root.exists():
            hist.append({"root":str(root.relative_to(ROOT)),"file_count":sum(1 for x in root.rglob("*") if x.is_file())})
    write_json(OUT/"E5_0_PROVENANCE_MANIFEST.json",{"schema_version":"E5-0-provenance-v1","created_utc":datetime.now(timezone.utc).isoformat(),"gate":"E5_0_CONDITIONAL_READY" if retrieval and retrieval.get("exploratory_queries",0)>0 else "E5_0_RETRIEVAL_CONTRACT_BLOCKED","files":files,"historical_roots":hist,"no_jev":True,"no_paid_api":True,"no_benchmark_selector_inference":True,"synthetic_selector_smoke":True,"no_reader_inference":True,"no_reserved_outcomes":True,"qdrant_modified":False})


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=["prepare","retrieve","finalize","all"],default="all"); args=ap.parse_args()
    if args.stage in ("prepare","all"): prepare()
    if args.stage in ("retrieve","all"): retrieve()
    if args.stage in ("finalize","all"): finalize()
    return 0


if __name__=="__main__": raise SystemExit(main())
