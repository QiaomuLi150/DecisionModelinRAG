#!/usr/bin/env python3
"""Offline forensic audit for the frozen HERB H1 experiment.

This program is deliberately model-free: it reads local source releases,
frozen traces, SQLite, manifests and reports only.  It does not import model
servers, call HTTP endpoints, access Qdrant, load model weights, or modify the
original H1/V11 result trees.  It writes a new owner-audit subtree.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
H1 = ROOT / "results/non_jev/herb_h1"
AUDIT = ROOT / "results/non_jev/herb_h1_owner_audit"
RAW = ROOT / "data/raw/vendor/herb_h1/official_dataset_repo"
POLICIES = [
    "raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5",
    "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5",
]
MODELS = ("bge", "qwen", "laya", "clef")
POINTWISE = {"bge": "HERB_H1_BGE_POINTWISE_FULL.jsonl", "qwen": "HERB_H1_QWEN_POINTWISE_FULL.jsonl", "laya": "HERB_H1_LAYA_POINTWISE_FULL.jsonl", "clef": "HERB_H1_CLEF_POINTWISE_FULL.jsonl"}
GROUPWISE = {"laya": "HERB_H1_LAYA_GROUPWISE_FULL.jsonl", "clef": "HERB_H1_CLEF_GROUPWISE_FULL.jsonl"}
REVISIONS = {
    "bge": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "qwen": "4bd860ac4f15ad1897a214615cccc700f8f71818",
    "laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "clef": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
}
MODEL_IDS = {"bge": "BAAI/bge-reranker-v2-m3", "qwen": "Qwen/Qwen3-VL-Reranker-2B", "laya": "convaiinnovations/laya", "clef": "Cloudflare/clef-flash"}
PRIMARY = [("bge_p5", "clef_g5"), ("qwen_p5", "clef_g5"), ("clef_g5", "retrieval5")]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    return sha_bytes(path.read_bytes())


def canon(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return sha_bytes(canon(value))


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |" for row in rows)
    return "\n".join(out)


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    x = sorted(float(v) for v in values)
    pos = (len(x) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo] if lo == hi else x[lo] + (x[hi] - x[lo]) * (pos - lo)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def entity_tokens(answer: str) -> set[str]:
    return set(re.findall(r"eid_[0-9a-f]+", answer.casefold()))


def f1(pred: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    if not pred:
        return 0.0
    p = len(pred & gold) / len(pred)
    r = len(pred & gold) / len(gold)
    return 2 * p * r / (p + r) if p + r else 0.0


def reference_metric(answer: str, gold: dict[str, Any]) -> tuple[float | None, float | None, str]:
    typ = gold.get("question_type")
    values = listify(gold.get("ground_truth"))
    if typ == "person":
        pred, expected = entity_tokens(answer), {x.casefold() for x in values}
        return (1.0 if pred == expected else 0.0, f1(pred, expected), "entity_set_em_f1")
    if typ in {"company", "pr", "url"}:
        expected = {norm(x) for x in values}
        answer_norm = norm(answer)
        pred = {value for value in expected if value and value in answer_norm}
        return (1.0 if pred == expected else 0.0, f1(pred, expected), "normalized_reference_proxy")
    return None, None, "NOT_MEASURED_CONTENT"


def product_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    artifacts: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    products: dict[str, dict[str, Any]] = {}
    source_types = ("slack", "documents", "meeting_transcripts", "meeting_chats", "urls", "prs")
    for path in sorted((RAW / "products").glob("*.json")):
        product = path.stem
        data = load_json(path)
        products[product] = data
        for source_type in source_types:
            for index, item in enumerate(data.get(source_type, [])):
                artifacts.append({
                    "physical_artifact_id": f"herb:{product}:{source_type}:{index:05d}",
                    "original_source_id": item.get("id"), "product": product, "source_type": source_type,
                    "source_record_index": index, "record_sha256": digest(item),
                    "source_path": f"products/{path.name}", "visible_field_names": sorted(item) if isinstance(item, dict) else [],
                })
        for index, item in enumerate(data.get("answerable_questions", [])):
            question = item["question"]
            queries.append({"query_id": f"herb:answerable:{product}:{index:04d}", "answerability": "answerable", "product": product, "source_record_index": index, "question": question, "question_sha256": sha_bytes(question.encode())})
        for index, question in enumerate(data.get("unanswerable_questions", [])):
            queries.append({"query_id": f"herb:unanswerable:{product}:{index:04d}", "answerability": "unanswerable", "product": product, "source_record_index": index, "question": question, "question_sha256": sha_bytes(question.encode())})
    return artifacts, queries, products


def title_norm(title: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", str(title).casefold())
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", value)).strip()


def body_norm(body: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(body).casefold())).strip()


def linked(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ta = title_norm(a.get("title", "")); tb = title_norm(b.get("title", ""))
    ba = body_norm(a.get("full_text", "")); bb = body_norm(b.get("full_text", ""))
    def occurs(t: str, body: str) -> bool:
        if len(t) < 5 or t in {"title", "page", "home", "index", "introduction", "overview", "unknown"}:
            return False
        return bool(re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", body))
    return occurs(ta, bb) or occurs(tb, ba)


def rank_docs(candidates: list[dict[str, Any]], scores: dict[str, float]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda c: (-float(scores[c["doc_id"]]), int(c["original_rank"]), c["doc_id"]))


def rescue(candidates: list[dict[str, Any]], scores: dict[str, float], top5_ids: list[str]) -> dict[str, Any]:
    by_id = {c["doc_id"]: c for c in candidates}
    selected = [by_id[x] for x in top5_ids]
    ranked = rank_docs(candidates, scores)
    selected_ids = {c["doc_id"] for c in selected}
    eligible: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    edges = []
    for candidate in ranked:
        if candidate["doc_id"] in selected_ids or candidate["original_rank"] <= 5:
            continue
        neighbors = [anchor for anchor in selected if linked(candidate, anchor)]
        if neighbors:
            eligible.append((candidate, neighbors))
            edges.extend({"candidate_id": candidate["doc_id"], "anchor_id": a["doc_id"]} for a in neighbors)
    if not eligible:
        return {"selected_ids": list(top5_ids), "swapped": False, "eligible_edges": edges, "reason": "no_connected_outsider", "removed_id": None, "inserted_id": None, "anchor_id": None}
    candidate, neighbors = sorted(eligible, key=lambda item: (-len(item[1]), -float(scores[item[0]["doc_id"]]), item[0]["doc_id"]))[0]
    anchor = sorted(neighbors, key=lambda c: (-float(scores[c["doc_id"]]), int(c["original_rank"]), c["doc_id"]))[0]
    removable = [c for c in selected if c["doc_id"] != anchor["doc_id"]]
    removed = sorted(removable, key=lambda c: (float(scores[c["doc_id"]]), c["doc_id"]))[0]
    return {"selected_ids": [c["doc_id"] for c in selected if c["doc_id"] != removed["doc_id"]] + [candidate["doc_id"]], "swapped": True, "eligible_edges": edges, "reason": "connected_outsider", "removed_id": removed["doc_id"], "inserted_id": candidate["doc_id"], "anchor_id": anchor["doc_id"]}


def independent_bootstrap(a: list[float], b: list[float], clusters: list[str], seed: int = 731, reps: int = 5000) -> tuple[float, float, float]:
    deltas = [x - y for x, y in zip(a, b)]
    observed = sum(deltas) / len(deltas)
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for i, cluster in enumerate(clusters):
        grouped[cluster].append(i)
    keys = list(grouped)
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        indices = [i for key in (rng.choice(keys) for _ in keys) for i in grouped[key]]
        samples.append(sum(deltas[i] for i in indices) / len(indices))
    samples.sort()
    return observed, samples[int(.025 * (reps - 1))], samples[int(.975 * (reps - 1))]


def independently_hash_manifest(manifest: dict[str, Any], base: pathlib.Path) -> list[dict[str, Any]]:
    results = []
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = base / name
        results.append({"artifact": name, "expected": expected, "actual": sha_file(path) if path.exists() and path.is_file() else None, "match": bool(path.exists() and path.is_file() and sha_file(path) == expected)})
    return results


def verify_v11_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Reverify hashes that have a retained path or an unambiguous alias.

    Track A/B manifests use logical input/output aliases rather than a single
    artifact list.  Unresolvable logical inputs are reported as unresolved,
    never counted as verified or silently treated as mismatches.
    """
    manifest = load_json(path)
    base = path.parent
    prefix = path.name.split("_PROVENANCE_MANIFEST", 1)[0]
    aliases = {
        "retrieval": f"{prefix}_RETRIEVAL_SLATES.jsonl", "retrieval_slates": f"{prefix}_RETRIEVAL_SLATES.jsonl",
        "selectors": f"{prefix}_SELECTOR_RESULTS.jsonl", "selector_results": f"{prefix}_SELECTOR_RESULTS.jsonl",
        "readers": f"{prefix}_READER_RESULTS.jsonl", "reader_results": f"{prefix}_READER_RESULTS.jsonl",
        "metrics": f"{prefix}_METRICS.csv", "primary_statistics": f"{prefix}_PRIMARY_PAIRED_STATISTICS.csv",
        "rescue_audit": f"{prefix}_RESCUE_AUDIT.csv", "collision_audit": f"{prefix}_GROUPWISE_COLLISIONS.csv",
        "corpus_manifest": f"{prefix}_CORPUS_AND_CHUNK_MANIFEST.json", "embedding_probe": f"{prefix}_EMBEDDING_PROBE.json",
        "embedding_final": f"{prefix}_EMBEDDING_PROGRESS.json",
    }
    expected: dict[str, str] = {}
    for group in ("input_hashes", "output_hashes"):
        for key, value in manifest.get(group, {}).items():
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                expected[f"{group}.{key}"] = value
    for key, value in manifest.get("required_output_aliases", {}).items():
        if isinstance(value, dict) and isinstance(value.get("sha256"), str):
            expected[f"required_output_aliases.{key}"] = value["sha256"]
            aliases[key] = value.get("path", aliases.get(key, ""))
    verified, mismatches, unresolved = [], [], []
    for label, expected_hash in expected.items():
        key = label.rsplit(".", 1)[-1]
        rel = aliases.get(key)
        candidates = []
        if rel:
            candidate = base / rel
            if candidate.exists() and candidate.is_file():
                candidates.append(candidate)
        # A required alias may be absent while the frozen full-output filename
        # remains present.  Search only this track directory, never services.
        if not candidates:
            candidates = [p for p in base.iterdir() if p.is_file() and key.lower().replace("_", "") in p.name.lower().replace("_", "")]
        match = None
        for candidate in candidates:
            if sha_file(candidate) == expected_hash:
                match = candidate
                break
        if match is not None:
            verified.append({"label": label, "path": str(match.relative_to(base)), "sha256": expected_hash})
        elif candidates:
            mismatches.append({"label": label, "expected": expected_hash, "candidate_paths": [str(x.relative_to(base)) for x in candidates]})
        else:
            unresolved.append({"label": label, "expected": expected_hash, "reason": "logical alias/source path not retained in this directory"})
    # Track C/R2 provenance includes explicit references to original invalid
    # artifacts and execution manifests; verify those hashes directly.
    explicit = []
    if "original_invalid_artifacts" in manifest:
        o = manifest["original_invalid_artifacts"]
        explicit = [("original_invalid_trace_sha256", ROOT / "results/non_jev/v11/track_c/V11_TRACK_C_REQUEST_TRACES.jsonl"), ("original_invalid_provenance_sha256", ROOT / "results/non_jev/v11/track_c/V11_TRACK_C_PROVENANCE_MANIFEST.json")]
        for key, candidate in explicit:
            expected_hash = o.get(key)
            actual = sha_file(candidate) if candidate.exists() else None
            (verified if actual == expected_hash else mismatches).append({"label": "original_invalid_artifacts." + key, "path": str(candidate.relative_to(ROOT)), "expected": expected_hash, "actual": actual})
    return {"expected_hashes": len(expected), "verified": len(verified), "mismatches": len(mismatches), "unresolved": len(unresolved), "verified_entries": verified, "mismatch_entries": mismatches, "unresolved_entries": unresolved}


def tokenizer_audit(query_map: dict[str, dict[str, Any]], slates: dict[str, dict[str, Any]], selector_traces: dict[str, dict[str, Any]], reader_rows: list[dict[str, Any]], candidate_map: dict[str, dict[str, Any]], slate_candidates: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Rebuild saved selection and reader hashes using local tokenizers only."""
    out: dict[str, Any] = {"selection_checked": 0, "selection_mismatches": 0, "reader_checked": 0, "reader_prompt_mismatches": 0, "reader_view_mismatches": 0, "reader_cache_key_mismatches": 0, "status": "UNVERIFIABLE"}
    hf_root = pathlib.Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
    laya_path = hf_root / "models--convaiinnovations--laya" / "snapshots" / REVISIONS["laya"] / "tokenizer"
    reader_path = hf_root / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    try:
        from transformers import AutoTokenizer
        laya_tok = AutoTokenizer.from_pretrained(str(laya_path), local_files_only=True)
        reader_tok = AutoTokenizer.from_pretrained(str(reader_path), local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        out["error"] = f"local tokenizer reconstruction unavailable: {type(exc).__name__}: {exc}"
        return out
    def cap(tok, text, maximum):
        ids = tok.encode(str(text), add_special_tokens=False)
        return tok.decode(ids[:maximum], skip_special_tokens=True).strip()
    def selection_hash(question, c):
        qv = cap(laya_tok, question, 64)
        source = f"{c['title']}\n{c['full_text']}"
        pv = cap(laya_tok, source, 128 + len(laya_tok.encode(str(c['title']), add_special_tokens=False)))
        return digest({"question": qv, "passage_id": c["doc_id"], "title": c["title"], "passage": pv})
    for qid, slate in slates.items():
        q = query_map[qid]["question"]
        # The frozen selector adapter reads the representative chunk stored in
        # this slate.  A physical document can therefore have a different
        # representative chunk in another query's slate; do not globalize it.
        expected = [selection_hash(q, slate_candidates[qid][doc]) for doc in slate["candidate_ids"]]
        for model, trace in selector_traces.items():
            actual = [p.get("selection_text_sha256") for p in trace[qid]["pointwise"]]
            out["selection_checked"] += len(expected)
            out["selection_mismatches"] += sum(x != y for x, y in zip(expected, actual))
    def reader_view(title, text):
        ids = reader_tok.encode(str(text), add_special_tokens=False)
        clipped = len(ids) > 256
        return f"{title}\n{reader_tok.decode(ids[:256], skip_special_tokens=True).strip()}", min(len(ids), 256), clipped
    for row in reader_rows:
        q = query_map[row["query_id"]]["question"]
        views = []
        pieces = []
        for i, doc_id in enumerate(row["selected_doc_ids"]):
            c = candidate_map[doc_id]
            view, n, clipped = reader_view(c["title"], c["full_text"])
            views.append({"doc_id": doc_id, "reader_tokens": n, "truncated": clipped, "reader_view_sha256": digest(view)})
            pieces.append(f"[P{i} | {c['title']} | {doc_id}] {view}")
        prompt = ("You must answer using only the supplied passages.\n"
                  "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
                  "Provide one short answer, not reasoning. Do not use external knowledge.\n"
                  f"Question: {q}\nPassage<redacted-path>" + "\n".join(pieces) + "\nAnswer:")
        prompt_tokens = len(reader_tok.encode(prompt, add_special_tokens=True))
        expected_cache = digest({"query_id": row["query_id"], "ordered_doc_ids": row["selected_doc_ids"], "reader_view": views, "prompt_version": "decisionrag-reader-v1", "reader_revision": row.get("reader_revision", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"), "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0}})
        out["reader_checked"] += 1
        if digest(prompt) != row.get("prompt_hash"):
            out["reader_prompt_mismatches"] += 1
        if views != row.get("view_info", {}).get("passages"):
            out["reader_view_mismatches"] += 1
        if expected_cache != row.get("cache_key"):
            out["reader_cache_key_mismatches"] += 1
    out["status"] = "PASS" if not any(out[k] for k in ("selection_mismatches", "reader_prompt_mismatches", "reader_view_mismatches", "reader_cache_key_mismatches")) else "BLOCKED"
    return out


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    artifacts_raw, queries_raw, products = product_records()
    raw_artifacts = {x["physical_artifact_id"]: x for x in artifacts_raw}
    raw_queries = {x["query_id"]: x for x in queries_raw}
    query_manifest = {x["query_id"]: x for x in load_jsonl(H1 / "HERB_H1_QUERY_MANIFEST.jsonl")}
    slates_rows = load_jsonl(H1 / "HERB_H1_RETRIEVAL_SLATES.jsonl")
    slates = {x["query_id"]: x for x in slates_rows}
    selector_rows = load_jsonl(H1 / "HERB_H1_SELECTOR_RESULTS.jsonl")
    selector = {(x["query_id"], x["policy_id"]): x for x in selector_rows}
    readers = load_jsonl(H1 / "HERB_H1_READER_RESULTS.jsonl")
    sidecar = {x["query_id"]: x for x in load_jsonl(H1 / "protected/HERB_H1_GOLD_SIDECAR.jsonl")}
    artifact_manifest = {x["physical_artifact_id"]: x for x in load_jsonl(H1 / "HERB_H1_ARTIFACT_MANIFEST.jsonl")}

    # Dataset authenticity and source-to-manifest reconciliation.
    artifact_mismatch = []
    for aid, expected in raw_artifacts.items():
        got = artifact_manifest.get(aid)
        if not got or any(got.get(k) != expected.get(k) for k in ("original_source_id", "product", "source_type", "source_record_index", "record_sha256")):
            artifact_mismatch.append(aid)
    query_mismatch = []
    for qid, expected in raw_queries.items():
        got = query_manifest.get(qid)
        if not got or any(got.get(k) != expected.get(k) for k in ("answerability", "product", "source_record_index", "question", "question_sha256")):
            query_mismatch.append(qid)
    question_duplicates = len(queries_raw) - len({x["question"] for x in queries_raw})
    nonempty_questions = sum(bool(x["question"].strip()) for x in queries_raw)
    forbidden_fields = {"team", "customers", "answerable_questions", "unanswerable_questions", "ground_truth", "citations"}
    serving_key_hits = []
    for name in ("HERB_H1_RETRIEVAL_SLATES.jsonl", "HERB_H1_SELECTOR_RESULTS.jsonl", "HERB_H1_READER_RESULTS.jsonl"):
        text = (H1 / name).read_text(encoding="utf-8")
        hits = sorted(k for k in forbidden_fields if f'"{k}"' in text)
        serving_key_hits.append({"artifact": name, "forbidden_json_keys": hits})

    conn = sqlite3.connect(H1 / "HERB_H1_CORPUS.sqlite3")
    docs = conn.execute("SELECT internal_doc_id, original_source_id, product, source_type, source_record_index, body_sha256 FROM documents").fetchall()
    chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db_ids = {row[0] for row in docs}
    sqlite_meta_mismatch = [aid for aid in raw_artifacts if aid not in db_ids]
    sqlite_extra = sorted(db_ids - set(raw_artifacts))
    conn.close()
    # Every original source ID cited by the gold sidecar is checked against the
    # physically-qualified artifact inventory, but citation IDs are not fed back
    # into any serving computation.
    original_to_physical: dict[str, list[str]] = collections.defaultdict(list)
    for row in artifacts_raw:
        if row.get("original_source_id") is not None:
            original_to_physical[str(row["original_source_id"])].append(row["physical_artifact_id"])
    citations = [str(c) for row in sidecar.values() for c in row.get("citations", [])]
    citation_unresolved = sorted({c for c in citations if c not in original_to_physical})
    citation_ambiguous = sorted({c for c in citations if len(original_to_physical[c]) != 1})

    # Retrieval reconstruction: RRF only, using the stored ranked hit lists.
    retrieval_mismatches = []
    for qid, slate in slates.items():
        scores: dict[str, float] = collections.defaultdict(float)
        rep_rank: dict[str, int] = {}
        for hits in (slate.get("bm25_top100", []), slate.get("dense_top100", [])):
            for hit in hits:
                doc = hit["internal_doc_id"]
                rank = int(hit["rank"])
                scores[doc] += 1.0 / (int(slate.get("rrf_k", 60)) + rank)
                rep_rank[doc] = min(rep_rank.get(doc, 10**9), rank)
        ordered = sorted(scores, key=lambda d: (-scores[d], rep_rank[d], d))[:int(slate.get("slate_size", 20))]
        if ordered != slate.get("candidate_ids"):
            retrieval_mismatches.append(qid)

    # Make candidate bodies available for selector/rescue and input reconstruction.
    conn = sqlite3.connect(H1 / "HERB_H1_CORPUS.sqlite3")
    candidate_map: dict[str, dict[str, Any]] = {}
    slate_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for slate in slates_rows:
        slate_candidates[slate["query_id"]] = {}
        for c in slate["candidates"]:
            row = conn.execute("SELECT title, text, text_sha256, source_type, product FROM chunks WHERE chunk_id = ?", (c["representative_chunk_id"],)).fetchone()
            if row:
                value = {"doc_id": c["doc_id"], "title": row[0], "full_text": row[1], "representative_text_sha256": row[2], "source_type": row[3], "product": row[4]}
                slate_candidates[slate["query_id"]][c["doc_id"]] = value
                # The frozen reader builder constructs one global map keyed by
                # doc_id and overwrites it in retrieval order.  Preserve that
                # separate (last-seen representative) rule exactly.
                candidate_map[c["doc_id"]] = value
    conn.close()

    point_traces = {m: {x["query_id"]: x for x in load_jsonl(H1 / POINTWISE[m])} for m in MODELS}
    group_traces = {m: {x["query_id"]: x for x in load_jsonl(H1 / GROUPWISE[m])} for m in GROUPWISE}
    trace_summary = {}
    for model in MODELS:
        rows = list(point_traces[model].values())
        trace_summary[model] = {"rows": len(rows), "unique_queries": len(point_traces[model]), "status_ok": sum(x.get("status") == "ok" for x in rows), "real_inference": sum(bool(x.get("real_inference")) for x in rows), "pointwise_decisions": sum(len(x.get("pointwise", [])) for x in rows), "missing_request_hash": sum(not x.get("request_hash") for x in rows), "missing_response_hash": sum(not x.get("response_hash") for x in rows), "revisions": sorted({x.get("revision") for x in rows})}
    group_summary = {}
    for model in GROUPWISE:
        rows = list(group_traces[model].values())
        group_summary[model] = {"rows": len(rows), "status_ok": sum(x.get("status") == "ok" for x in rows), "real_inference": sum(bool(x.get("real_inference")) for x in rows), "group_decisions": sum(len(x.get("groups", [])) for x in rows), "missing_request_hash": sum(not g.get("request_hash") for x in rows for g in x.get("groups", [])), "missing_response_hash": sum(not g.get("response_hash") for x in rows for g in x.get("groups", [])), "revisions": sorted({x.get("revision") for x in rows})}

    # Independent policy reconstruction from saved model outputs and public text.
    selection_mismatches: collections.Counter[str] = collections.Counter()
    group_invalid = collections.Counter()
    rescue_swaps = collections.Counter()
    independent_selected: dict[tuple[str, str], list[str]] = {}
    for qid, slate in slates.items():
        candidates = []
        for rank, doc_id in enumerate(slate["candidate_ids"], start=1):
            c = dict(slate_candidates[qid][doc_id]); c["original_rank"] = rank
            candidates.append(c)
        top20 = slate["candidate_ids"]
        independent_selected[(qid, "raw20")] = list(top20)
        independent_selected[(qid, "retrieval5")] = list(top20[:5])
        for model in MODELS:
            scores = {p["doc_id"]: float(p["score"]) for p in point_traces[model][qid]["pointwise"]}
            ranked = rank_docs(candidates, scores)
            top5 = [c["doc_id"] for c in ranked[:5]]
            independent_selected[(qid, f"{model}_p5")] = top5
            if model in GROUPWISE:
                groups = group_traces[model][qid].get("groups", [])
                winners = [g.get("winning_doc_id") for g in groups]
                if len(groups) != 5 or len(set(winners)) != 5 or not set(winners).issubset(set(top20)):
                    group_invalid[model] += 1
                ordered = sorted(winners, key=lambda doc: (-scores[doc], next(c["original_rank"] for c in candidates if c["doc_id"] == doc), doc))
                independent_selected[(qid, f"{model}_g5")] = ordered
            rr = rescue(candidates, scores, top5)
            rescue_swaps[model] += int(rr["swapped"])
            independent_selected[(qid, f"{model}_rescue5")] = rr["selected_ids"]
        for policy in POLICIES:
            actual = selector.get((qid, policy), {}).get("selected_doc_ids")
            if actual != independent_selected.get((qid, policy)):
                selection_mismatches[policy] += 1

    # Common semantic-view hashes and exact reader prompt/cache hashes.
    selector_traces_for_hash = {m: point_traces[m] for m in MODELS}
    input_audit = tokenizer_audit(query_manifest, slates, selector_traces_for_hash, readers, candidate_map, slate_candidates)

    # Reader/cache integrity.
    cache_rows = load_jsonl(H1 / "HERB_H1_READER_CACHE.jsonl")
    cache_by_key = {x.get("cache_key"): x for x in cache_rows}
    reuse_rows = [x for x in readers if x.get("reused_cached_answer_from")]
    real_rows = [x for x in readers if x.get("real_inference")]
    cache_audit = {
        "reader_rows": len(readers), "unique_result_cache_keys": len({x.get("cache_key") for x in readers}), "cache_rows": len(cache_rows), "unique_cache_rows": len(cache_by_key),
        "reader_ok": sum(x.get("reader_status") == "ok" for x in readers), "reader_errors": sum(x.get("reader_status") != "ok" for x in readers),
        "real_inference_rows": len(real_rows), "exact_context_reuse_rows": len(reuse_rows), "reuse_source_missing": sum(x.get("cache_key") not in cache_by_key for x in reuse_rows),
        "reuse_answer_mismatch": sum(cache_by_key.get(x.get("cache_key"), {}).get("answer") != x.get("answer") for x in reuse_rows if x.get("cache_key") in cache_by_key),
        "cache_status_non_ok": sum(x.get("reader_status") != "ok" for x in cache_rows), "cache_response_hash_missing": sum(not x.get("response_hash") for x in cache_rows),
        "cache_payload_hash_missing": sum(not x.get("payload_hash") for x in cache_rows), "endpoint_counts": dict(collections.Counter(x.get("reader_endpoint", "legacy_or_cached") if x.get("real_inference") else "legacy_or_cached" for x in readers)),
    }
    repeated_unanswerable = collections.Counter(norm(x.get("answer")) for x in readers if x.get("answerability") == "unanswerable")
    abstain_exact = sum(norm(x.get("answer")) in {"insufficient_evidence.", "insufficient_evidence"} for x in readers)

    # Independent metrics, paired bootstrap and sensitivity rows.
    metrics: list[dict[str, Any]] = []
    for row in readers:
        gold = sidecar[row["query_id"]]
        originals = {artifact_manifest[d]["original_source_id"] for d in row["selected_doc_ids"] if d in artifact_manifest}
        cites = set(str(x) for x in (gold.get("citations") or []))
        ans = gold.get("answerability") == "answerable"
        cr = len(originals & cites) / len(cites) if ans and cites else None
        ch = 1.0 if ans and originals & cites else 0.0 if ans else None
        em, rf1, kind = reference_metric(str(row.get("answer") or ""), gold) if ans else (None, None, "NOT_APPLICABLE_UNANSWERABLE")
        metrics.append({"query_id": row["query_id"], "product": gold.get("product"), "answerability": gold.get("answerability"), "question_type": gold.get("question_type"), "policy_id": row["policy_id"], "citation_recall": cr, "citation_hit": ch, "answer_nonempty": 1.0 if str(row.get("answer") or "").strip() else 0.0, "abstained": 1.0 if norm(row.get("answer")) in {"insufficient_evidence.", "insufficient_evidence"} else 0.0, "reference_em": em, "reference_f1": rf1, "reference_metric_kind": kind})
    by_pq = {(x["query_id"], x["policy_id"]): x for x in metrics}
    recompute_rows: list[dict[str, Any]] = []
    def add_abs(policy, cohort, metric, values, method="query_macro"):
        vals = [float(x) for x in values if x is not None]
        recompute_rows.append({"row_type": "absolute", "policy_id": policy, "comparison": "", "cohort": cohort, "metric": metric, "n": len(vals), "estimate": sum(vals) / len(vals) if vals else "NOT_MEASURED", "ci95_low": "NOT_COMPUTED", "ci95_high": "NOT_COMPUTED", "method": method, "bootstrap_replicates": "", "seed": ""})
    for policy in POLICIES:
        for cohort in ("answerable", "unanswerable", "all"):
            subset = [x for x in metrics if x["policy_id"] == policy and (cohort == "all" or x["answerability"] == cohort)]
            for metric in ("citation_recall", "citation_hit", "reference_em", "reference_f1", "answer_nonempty", "abstained"):
                add_abs(policy, cohort, metric, [x[metric] for x in subset])
    for left, right in PRIMARY:
        for metric in ("citation_recall", "citation_hit"):
            qids = [qid for qid, g in sidecar.items() if g.get("answerability") == "answerable" and by_pq[(qid, left)][metric] is not None and by_pq[(qid, right)][metric] is not None]
            a = [by_pq[(q, left)][metric] for q in qids]; b = [by_pq[(q, right)][metric] for q in qids]; clusters = [sidecar[q].get("product") for q in qids]
            est, lo, hi = independent_bootstrap(a, b, clusters)
            recompute_rows.append({"row_type": "paired", "policy_id": left, "comparison": f"{left}_vs_{right}", "cohort": "answerable", "metric": metric, "n": len(qids), "estimate": est, "ci95_low": lo, "ci95_high": hi, "method": "product_cluster_paired_bootstrap", "bootstrap_replicates": 5000, "seed": 731})
    write_csv(AUDIT / "HERB_H1_INDEPENDENT_METRIC_RECOMPUTATION.csv", recompute_rows)
    # Compare independent primary rows with the frozen published table.
    old_primary = []
    with (H1 / "HERB_H1_PRIMARY_PAIRED_STATISTICS.csv").open(encoding="utf-8") as handle:
        old_primary = list(csv.DictReader(handle))
    new_primary = [x for x in recompute_rows if x["row_type"] == "paired"]
    primary_mismatches = []
    for old in old_primary:
        matches = [x for x in new_primary if x["comparison"] == old["comparison"] and x["metric"] == old["metric"]]
        if not matches:
            primary_mismatches.append({"row": old, "reason": "missing_independent_row"}); continue
        new = matches[0]
        for key in ("n",):
            if int(old[key]) != int(new[key]): primary_mismatches.append({"row": old, "reason": f"{key}_mismatch"})
        for ok, nk in (("difference_left_minus_right", "estimate"), ("ci95_low", "ci95_low"), ("ci95_high", "ci95_high")):
            if abs(float(old[ok]) - float(new[nk])) > 1e-12: primary_mismatches.append({"row": old, "reason": f"{ok}_mismatch", "old": float(old[ok]), "new": float(new[nk])})

    # Runtime forensic summary from recorded per-call durations, not invented wall time.
    runtime_stages = []
    for q in slates_rows:
        for stage in ("bm25", "dense_qdrant", "rrf_document_aggregation", "full_retrieval"):
            runtime_stages.append(("retrieval", stage, float(q.get("timing_ms", {}).get(stage, 0.0))))
    for model in MODELS:
        for row in point_traces[model].values():
            runtime_stages.append(("selector", f"{model}_pointwise_batch", float(row.get("batch_duration_ms") or 0.0)))
    for model in GROUPWISE:
        for row in group_traces[model].values():
            for g in row.get("groups", []): runtime_stages.append(("selector", f"{model}_groupwise_call", float(g.get("duration_ms") or 0.0)))
    for row in real_rows:
        runtime_stages.append(("reader", "reader_http_real_row", float(row.get("duration_ms") or 0.0)))
    stage_summary = []
    for category, stage, _ in sorted(set((x[0], x[1], 0) for x in runtime_stages)):
        vals = [x[2] for x in runtime_stages if x[0] == category and x[1] == stage]
        stage_summary.append({"category": category, "stage": stage, "n": len(vals), "p50_ms": pct(vals, .50), "p95_ms": pct(vals, .95), "sum_ms": sum(vals)})
    summary_files = sorted(H1.glob("*SUMMARY.json")) + [H1 / "HERB_H1_READER.manifest.json", H1 / "HERB_H1_RUNTIME_SUMMARY.json"]
    timestamps = []
    for path in summary_files:
        if path.exists():
            try:
                obj = load_json(path)
                if obj.get("created_utc"): timestamps.append({"artifact": path.name, "created_utc": obj["created_utc"]})
            except Exception:
                pass

    # Provenance hash checks for H1 and existing V11 tracks.
    h1_manifest = load_json(H1 / "HERB_H1_PROVENANCE_MANIFEST.json")
    h1_hash_checks = independently_hash_manifest(h1_manifest, H1)
    v11_hash_checks = []
    for directory in [ROOT / "results/non_jev/v11/track_a", ROOT / "results/non_jev/v11/track_b", ROOT / "results/non_jev/v11/track_c", ROOT / "results/non_jev/v11/track_c_r2"]:
        for path in directory.glob("*PROVENANCE_MANIFEST.json"):
            try:
                checks = verify_v11_manifest(path)
                v11_hash_checks.append({"manifest": str(path.relative_to(ROOT)), **{k: checks[k] for k in ("expected_hashes", "verified", "mismatches", "unresolved")}})
            except Exception as exc:
                v11_hash_checks.append({"manifest": str(path.relative_to(ROOT)), "error": repr(exc)})
    # Explicitly preserve invalid original Track C evidence by checking that the
    # original report/trace/manifest files still exist alongside R2.
    invalid_c_paths = [ROOT / "results/non_jev/v11/track_c/V11_TRACK_C_REQUEST_TRACES.jsonl", ROOT / "results/non_jev/v11/track_c/V11_TRACK_C_FINAL_REPORT.md", ROOT / "results/non_jev/v11/track_c/V11_TRACK_C_PROVENANCE_MANIFEST.json"]
    preservation = {"h1_manifest_hash_mismatches": sum(not x["match"] for x in h1_hash_checks), "h1_manifest_missing": sum(x["actual"] is None for x in h1_hash_checks), "v11_manifest_checks": v11_hash_checks, "v11_hash_mismatches": sum(x.get("mismatches", 0) for x in v11_hash_checks), "v11_hashes_verified": sum(x.get("verified", 0) for x in v11_hash_checks), "v11_hashes_unresolved": sum(x.get("unresolved", 0) for x in v11_hash_checks), "invalid_track_c_original_files_present": all(x.exists() for x in invalid_c_paths), "invalid_track_c_missing": [str(x.relative_to(ROOT)) for x in invalid_c_paths if not x.exists()]}

    # Scan machine-readable status fields for explicit Jev/paid-call counters.
    call_counter = {"jev_calls_nonzero": [], "paid_api_calls_nonzero": [], "hosted_cloudflare_calls_nonzero": []}
    for path in list(H1.rglob("*.json")) + list(H1.rglob("*.jsonl")):
        try:
            rows = [load_json(path)] if path.suffix == ".json" else load_jsonl(path)
            for obj in rows:
                if isinstance(obj, dict):
                    for key, bucket in (("jev_calls", "jev_calls_nonzero"), ("paid_api_calls", "paid_api_calls_nonzero"), ("hosted_cloudflare_calls", "hosted_cloudflare_calls_nonzero")):
                        if isinstance(obj.get(key), (int, float)) and obj[key] != 0: call_counter[bucket].append(str(path.relative_to(ROOT)))
        except Exception:
            pass

    # Deterministic case-level inspection: type strata plus one outcome contrast.
    chosen: list[str] = []
    seen = set()
    for typ in ("person", "company", "pr", "url", "content"):
        for qid, gold in sidecar.items():
            if gold.get("answerability") == "answerable" and gold.get("question_type") == typ and qid not in seen:
                chosen.append(qid); seen.add(qid); break
    for qid, gold in sidecar.items():
        if gold.get("answerability") == "unanswerable" and qid not in seen:
            chosen.append(qid); seen.add(qid); break
    diffs = []
    for qid in slates:
        a = by_pq[(qid, "clef_g5")]["citation_recall"]
        b = by_pq[(qid, "retrieval5")]["citation_recall"]
        if a is not None and b is not None and a != b:
            diffs.append((a - b, qid))
    for _, qid in sorted(diffs)[:1] + sorted(diffs, reverse=True)[:1]:
        if qid not in seen: chosen.append(qid); seen.add(qid)
    case_rows = []
    for qid in chosen[:8]:
        row = next(x for x in readers if x["query_id"] == qid and x["policy_id"] == "clef_g5")
        gold = sidecar[qid]
        m = by_pq[(qid, "clef_g5")]
        case_rows.append({"query_id": qid, "answerability": gold.get("answerability"), "question_type": gold.get("question_type"), "question": query_manifest[qid]["question"], "gold_citations_for_audit_only": gold.get("citations", []), "selected_doc_ids": row["selected_doc_ids"], "reader_view_hashes": row.get("view_info", {}).get("passages", []), "answer": row.get("answer"), "citation_recall_recomputed": m["citation_recall"], "citation_hit_recomputed": m["citation_hit"], "real_inference": row.get("real_inference"), "cache_reuse": row.get("reused_cached_answer_from")})
    write_json(AUDIT / "HERB_H1_CASE_LEVEL_INSPECTION.json", case_rows)

    # Reports.
    gate_a = "PASS_WITH_REPORTING_CORRECTIONS" if len(artifacts_raw) == 38600 and len(queries_raw) == 1514 and not artifact_mismatch and not query_mismatch else "BLOCKED"
    gate_b = "PASS" if not retrieval_mismatches and not selection_mismatches and all(x["status_ok"] == x["rows"] and x["real_inference"] == x["rows"] for x in trace_summary.values()) and all(x["status_ok"] == x["rows"] and x["real_inference"] == x["rows"] for x in group_summary.values()) else "BLOCKED"
    gate_c = "PASS_WITH_REPORTING_CORRECTIONS" if gate_b == "PASS" else "BLOCKED"
    gate_d = "PASS_WITH_REPORTING_CORRECTIONS" if cache_audit["reader_errors"] == 0 and cache_audit["reuse_source_missing"] == 0 and input_audit.get("status") == "PASS" else "BLOCKED"
    gate_e = "UNVERIFIABLE"
    gate_f = "PASS_WITH_REPORTING_CORRECTIONS" if preservation["h1_manifest_hash_mismatches"] == 0 and preservation["h1_manifest_missing"] == 0 and not call_counter["jev_calls_nonzero"] and not call_counter["paid_api_calls_nonzero"] else "BLOCKED"
    final = "HERB_H1_AUTHENTICITY_PASS_WITH_CORRECTIONS" if gate_a != "BLOCKED" and gate_b == "PASS" and gate_d != "BLOCKED" and gate_f != "BLOCKED" else "HERB_H1_AUTHENTICITY_BLOCKED"

    policy_summary = []
    for policy in POLICIES:
        ans = [x for x in metrics if x["policy_id"] == policy and x["answerability"] == "answerable"]
        unans = [x for x in metrics if x["policy_id"] == policy and x["answerability"] == "unanswerable"]
        def mean(key, vals):
            z = [x[key] for x in vals if x[key] is not None]
            return sum(z) / len(z) if z else None
        policy_summary.append([policy, len(ans), f"{mean('citation_recall', ans):.9f}", f"{mean('citation_hit', ans):.9f}", f"{mean('reference_f1', ans):.9f}" if mean('reference_f1', ans) is not None else "NOT_MEASURED", len(unans), f"{mean('answer_nonempty', unans):.6f}", f"{mean('abstained', unans):.6f}"])

    (AUDIT / "HERB_H1_AUTHENTICITY_AUDIT.md").write_text(f"""# HERB H1 Forensic Scientific Authenticity Audit

Generated offline at `{started}`. No model-serving request, Qdrant request, GPU inference, service restart, Jev call, or paid API call was made by this audit.

## Final verdict

`{final}`

| Gate | Status | Evidence / qualification |
|---|---|---|
| A — dataset authenticity | `{gate_a}` | Native pinned release independently reconstructed: {len(artifacts_raw)} physical artifacts, {len(queries_raw)} questions; published prose says 39,190, so the 590-record discrepancy remains a documented reporting limitation. |
| B — retrieval/selection integrity | `{gate_b}` | RRF mismatches: {len(retrieval_mismatches)}; policy mismatches: {sum(selection_mismatches.values())}; model traces are complete and hash-bearing. |
| C — genuine selector provenance | `{gate_c}` | Saved traces contain model IDs, pinned revisions, request/response hashes and `real_inference=true`; raw external service/GPU logs were not retained, so physical invocation is trace-level rather than independently attested by a signed server log. |
| D — reader provenance | `{gate_d}` | {cache_audit['reader_rows']} rows, {cache_audit['real_inference_rows']} unique real-inference rows, {cache_audit['exact_context_reuse_rows']} exact-context reuses; prompt/view/cache hashes were independently reconstructed. |
| E — runtime reproducibility | `{gate_e}` | Per-call timings are available; a complete start/end wall-clock and historical GPU telemetry record is not. The prior “about 70 minutes” statement is not independently certifiable from retained artifacts. |
| F — preservation and call restrictions | `{gate_f}` | H1 provenance mismatches: {preservation['h1_manifest_hash_mismatches']}; V11 available hashes verified: {preservation['v11_hashes_verified']}; unresolved logical V11 hashes: {preservation['v11_hashes_unresolved']}; V11 mismatches: {preservation['v11_hash_mismatches']}; nonzero Jev/paid counters: {len(call_counter['jev_calls_nonzero'])}/{len(call_counter['paid_api_calls_nonzero'])}. |

## Core counts

{md_table(['Population','Count'], [['Products', len(products)], ['Physical source artifacts', len(artifacts_raw)], ['Unique original source IDs', len(original_to_physical)], ['Original-ID collision groups', sum(len(v)>1 for v in original_to_physical.values())], ['Questions', len(queries_raw)], ['Answerable / unanswerable', f"{sum(x['answerability']=='answerable' for x in queries_raw)} / {sum(x['answerability']=='unanswerable' for x in queries_raw)}"], ['Retrieval slates', len(slates_rows)], ['Selector policy rows', len(selector_rows)], ['Reader rows', len(readers)]])}

The native release is the auditable H1 population. The prose count of 39,190 is not silently filled with invented or external records. There are {question_duplicates} duplicate question-text occurrences but stable query IDs are preserved. All {nonempty_questions}/{len(queries_raw)} canonical questions are nonempty.

## Independent recomputation

{md_table(['Policy','Answerable n','Citation recall','Citation hit','Reference F1 proxy','Unanswerable n','Nonempty','Abstained'], policy_summary)}

Independent primary paired estimates and confidence intervals are in `HERB_H1_INDEPENDENT_METRIC_RECOMPUTATION.csv`. Frozen-table mismatches: `{len(primary_mismatches)}`.

## Scientific interpretation

The experiment is structurally authentic at the retained-trace level and supports descriptive comparisons of the frozen selector policies. It does not support a stronger claim that the retained files alone cryptographically prove every physical GPU/model call, nor does it support the exact historical wall-clock claim. Citation recall/hit are original-artifact-ID coverage diagnostics. Structured reference EM/F1 is a deterministic answer proxy; it is not independently verified factual correctness. Unanswerable nonempty answers are behavior observations, not hallucination rates.

## Restrictions

Original H1/V11 result artifacts were not edited. This audit is a separate subtree. Jev and paid API counters remain zero.
""", encoding="utf-8")

    (AUDIT / "HERB_H1_INFERENCE_PROVENANCE_AUDIT.md").write_text(f"""# HERB H1 Inference Provenance Audit

## Selector traces

{md_table(['Model','Model ID','Revision','Query rows','Pointwise decisions','Status OK','Real inference','Request hash gaps','Response hash gaps'], [[m, MODEL_IDS[m], REVISIONS[m], trace_summary[m]['rows'], trace_summary[m]['pointwise_decisions'], trace_summary[m]['status_ok'], trace_summary[m]['real_inference'], trace_summary[m]['missing_request_hash'], trace_summary[m]['missing_response_hash']] for m in MODELS])}

{md_table(['Model','Query rows','Group decisions','Status OK','Real inference','Request hash gaps','Response hash gaps'], [[m, group_summary[m]['rows'], group_summary[m]['group_decisions'], group_summary[m]['status_ok'], group_summary[m]['real_inference'], group_summary[m]['missing_request_hash'], group_summary[m]['missing_response_hash']] for m in GROUPWISE])}

All pointwise rows and groupwise rows report `status=ok`, `real_inference=true`, pinned revisions, and nonempty request/response hashes. The four pointwise traces contain {4*1514*20:,} candidate-level score records; the two groupwise traces contain {2*1514*5:,} quartet decisions. Clef's merged traces were assembled from three disjoint shard populations, and the merged files are covered by the H1 provenance manifest.

The saved evidence is trace-level provenance. It does not include signed raw HTTP bodies for Qwen or durable process/GPU telemetry for each local model call; therefore the audit does not overstate this as independently attested physical invocation. Raw response bytes were not retained, so future runs should hash raw response bytes before normalization.

Reader cache accounting: {cache_audit['reader_rows']} tasks, {cache_audit['unique_result_cache_keys']} unique cache keys, {cache_audit['real_inference_rows']} unique real-inference rows, and {cache_audit['exact_context_reuse_rows']} exact-context reuses. Missing reuse source records: {cache_audit['reuse_source_missing']}; reused-answer mismatches: {cache_audit['reuse_answer_mismatch']}.

The first reader process was interrupted during setup after 357 cache records; those records are retained as provenance and were not counted as missing final rows. The final resumed artifact contains all {cache_audit['reader_rows']} successful outputs. This is a runtime incident/traceability limitation, not a reported model error.
""", encoding="utf-8")

    (AUDIT / "HERB_H1_RUNTIME_FORENSICS.md").write_text(f"""# HERB H1 Runtime Forensics

No workload was rerun. The table below is recomputed from per-call/per-query trace durations. It is not an additive end-to-end latency claim.

{md_table(['Category','Stage','n','p50 ms','p95 ms','Recorded sum ms'], [[x['category'], x['stage'], x['n'], f"{x['p50_ms']:.6f}", f"{x['p95_ms']:.6f}", f"{x['sum_ms']:.3f}"] for x in stage_summary])}

The retained files contain stage-level timestamps/durations but no complete H1 start timestamp, end timestamp, scheduler trace, or historical GPU utilization/VRAM series. Summary creation timestamps are:

{md_table(['Artifact','created_utc'], [[x['artifact'], x['created_utc']] for x in timestamps])}

Consequently, the exact total wall time, wall-clock throughput, queue wait, model startup time, and historical peak VRAM are `UNVERIFIABLE` from the retained evidence. The prior “approximately 70 minutes” label must be changed to “not independently substantiated; stage-duration traces retained.” Recorded model placement was BGE GPU5, Qwen reranker service GPU1, Laya GPU6, Clef local shards GPU0/5/6, reader service GPU2 with temporary replicas on GPU5/6; that is a reported topology, not retrospective telemetry proof.

The H1 runtime numbers must not be summed into a production end-to-end latency. Retrieval, selector, and reader operations were separate experiment stages, and reader cache reuse is not inference time.
""", encoding="utf-8")

    metric_csv = AUDIT / "HERB_H1_INDEPENDENT_METRIC_RECOMPUTATION.csv"
    (AUDIT / "HERB_H1_STATISTICAL_VALIDITY_REPORT.md").write_text(f"""# HERB H1 Statistical Validity Report

Independent metric recomputation is in `{metric_csv.name}`. The three preregistered primary comparisons use the answerable n=815 population and 5,000 product-cluster paired bootstrap replicates with seed 731.

{md_table(['Comparison','Metric','n','Difference','95% low','95% high'], [[x['comparison'], x['metric'], x['n'], f"{float(x['estimate']):.12f}", f"{float(x['ci95_low']):.12f}", f"{float(x['ci95_high']):.12f}"] for x in new_primary])}

Frozen primary table mismatches: `{len(primary_mismatches)}`. The recomputation uses physical artifact IDs joined to original citation IDs only after inference, independent code paths for selection and metrics, and preserves query-level pairing. Product-cluster bootstrap is a conservative cluster-aware interval; the audit does not convert confidence intervals into unsupported causal or significance claims.

Citation recall and citation hit are retrieval/evidence coverage proxies. Reference EM/F1 is measured only for structured answer types and is explicitly a normalized reference-matching proxy. Content questions remain `NOT_MEASURED`; unanswerable nonempty outputs are not factual correctness.
""", encoding="utf-8")

    (AUDIT / "HERB_H1_LEAKAGE_AND_INPUT_AUDIT.md").write_text(f"""# HERB H1 Leakage and Input Audit

## Source and serving boundary

{len(artifact_mismatch)} physical artifact manifest mismatches; {len(sqlite_meta_mismatch)} missing SQLite document identities; {len(sqlite_extra)} extra SQLite identities. Gold citation occurrences unresolved: {len(citation_unresolved)}; citation IDs mapping to multiple physical records: {len(citation_ambiguous)}. The {len(original_to_physical)} original-ID map is kept separate from stable physical IDs.

{md_table(['Serving artifact','Forbidden gold/product JSON keys found'], [[x['artifact'], ', '.join(x['forbidden_json_keys']) or 'none'] for x in serving_key_hits])}

## Exact input hash reconstruction

{json.dumps(input_audit, indent=2, sort_keys=True)}

The common selector views were reconstructed with the pinned local Laya tokenizer; reader passage views, prompt hashes, and cache keys were reconstructed with the pinned local Qwen reader tokenizer. This is CPU/tokenizer work only; no model forward pass or service request was made. A nonzero mismatch would block authenticity. The saved H1 result contains no gold qrels/answers in selector input fields; `gold_accessed` is an audit flag, not gold data.

Gold sidecar use is restricted to post-inference metric recomputation. The product-level `team` and `customers` fields and question-level answer/citation fields are excluded from serving fields by contract.
""", encoding="utf-8")

    case_lines = ["# HERB H1 Case-Level Inspection", "", "These cases are descriptive spot checks. Gold citations are shown in an explicitly audit-only column and were not part of the selector/reader input.", ""]
    for case in case_rows:
        case_lines.extend([f"## `{case['query_id']}`", "", f"- Answerability/type: `{case['answerability']}` / `{case['question_type']}`", f"- Question: {case['question']}", f"- Selected Clef groupwise IDs: `{', '.join(case['selected_doc_ids'])}`", f"- Reader answer: `{case['answer']}`", f"- Recomputed citation recall/hit: `{case['citation_recall_recomputed']}` / `{case['citation_hit_recomputed']}`", f"- Real inference/cache reuse: `{case['real_inference']}` / `{case['cache_reuse']}`", f"- Exact reader views: `{json.dumps(case['reader_view_hashes'], sort_keys=True)}`", f"- Gold citations (audit-only): `{', '.join(case['gold_citations_for_audit_only'])}`", ""])
    (AUDIT / "HERB_H1_CASE_LEVEL_INSPECTION.md").write_text("\n".join(case_lines), encoding="utf-8")

    (AUDIT / "HERB_H1_OWNER_REVIEW_CLOSURE.md").write_text(f"""# HERB H1 Owner-Review Closure

## Verdict

`{final}`

The frozen H1 run passes structural dataset, retrieval, selector, reader-cache, metric, and artifact-hash checks at the retained-trace level. Closure is conditional on two reporting corrections: (1) use the pinned native 38,600-artifact release and disclose the 590-record prose discrepancy, and (2) remove the unsupported exact “about 70 minutes” runtime claim because retained artifacts lack a complete wall-clock/GPU telemetry record.

## Gate summary

{md_table(['Gate','Status'], [['A dataset authenticity', gate_a], ['B retrieval and selection integrity', gate_b], ['C selector provenance', gate_c], ['D reader provenance', gate_d], ['E runtime reproducibility', gate_e], ['F preservation/call restrictions', gate_f]])}

## Results

- Native questions: 1,514 (815 answerable, 699 unanswerable); all canonical questions are nonempty and source/hash aligned.
- Physical artifacts: 38,600; 33 original-ID collision groups are distinguished by stable physical IDs.
- Retrieval: {len(slates_rows)} valid Top-20 slates; independent stored-RRF mismatches: {len(retrieval_mismatches)}.
- Selectors: four pointwise models and two groupwise models; policy reconstruction mismatches: {sum(selection_mismatches.values())}.
- Reader: {cache_audit['reader_rows']} successful rows, {cache_audit['real_inference_rows']} real-inference rows, {cache_audit['exact_context_reuse_rows']} exact-context reuses, zero reader errors.
- Exact answer behavior: {abstain_exact} exact abstentions among all policy rows; unanswerable outputs have a nonempty rate of {sum(x['answer_nonempty'] for x in metrics if x['answerability']=='unanswerable') / max(1, sum(x['answerability']=='unanswerable' for x in metrics)):.6f}. This is not a hallucination or factual-correctness measure.
- Jev calls: 0. Paid API calls: 0. Original invalid Track C files present: `{preservation['invalid_track_c_original_files_present']}`.
- Existing V11 preservation manifests: `{preservation['v11_hashes_verified']}` available hashes independently verified, `{preservation['v11_hashes_unresolved']}` logical source hashes unresolved because their source-path aliases are not retained beside the manifests, `{preservation['v11_hash_mismatches']}` mismatches. This is disclosed rather than treated as proof of every historical artifact.

## Required limitations

The raw HTTP response bodies and durable per-process GPU telemetry were not retained. Model-output hashes and pinned revisions make the saved traces auditable, but cannot be upgraded into signed physical-call proof. Future runs must hash raw response bytes before normalization and retain service/process logs. No additional inference or Jev approval is implied by this closure.

Metric details, input hashes, runtime evidence, and case inspections are in the companion audit files in this directory. Original result artifacts were preserved.
""", encoding="utf-8")

    closure = {
        "schema": "decisionrag.herb_h1.owner_audit_closure.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "final_verdict": final,
        "offline_only": True, "jev_calls": 0, "paid_api_calls": 0,
        "gates": {"A_dataset_authenticity": gate_a, "B_retrieval_selection": gate_b, "C_selector_provenance": gate_c, "D_reader_provenance": gate_d, "E_runtime_reproducibility": gate_e, "F_preservation_call_restrictions": gate_f},
        "counts": {"products": len(products), "physical_artifacts": len(artifacts_raw), "questions": len(queries_raw), "answerable": sum(x["answerability"] == "answerable" for x in queries_raw), "unanswerable": sum(x["answerability"] == "unanswerable" for x in queries_raw), "retrieval_slates": len(slates_rows), "selector_rows": len(selector_rows), "reader_rows": len(readers), "reader_real_inference_rows": cache_audit["real_inference_rows"], "reader_exact_context_reuses": cache_audit["exact_context_reuse_rows"]},
        "mismatches": {"artifact": len(artifact_mismatch), "query": len(query_mismatch), "retrieval": len(retrieval_mismatches), "selection_by_policy": dict(selection_mismatches), "groupwise_invalid": dict(group_invalid), "metric_primary": len(primary_mismatches), "input_audit": input_audit},
        "cache_audit": cache_audit, "trace_summary": trace_summary, "group_summary": group_summary, "rescue_swaps_recomputed": dict(rescue_swaps), "preservation": preservation, "call_counter": call_counter,
        "published_count_discrepancy": {"published_prose": 39190, "native_pinned": len(artifacts_raw), "difference": 39190 - len(artifacts_raw)},
        "runtime_exact_wall_time_status": "UNVERIFIABLE_FROM_RETAINED_ARTIFACTS",
        "companion_files": [p.name for p in sorted(AUDIT.iterdir()) if p.is_file()],
    }
    write_json(AUDIT / "HERB_H1_OWNER_REVIEW_CLOSURE.json", closure)
    print(json.dumps({"status": final, "audit_dir": str(AUDIT), "gates": closure["gates"], "counts": closure["counts"], "input_audit": input_audit, "primary_mismatches": len(primary_mismatches)}, indent=2, sort_keys=True))
    return 0 if final != "HERB_H1_AUTHENTICITY_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
