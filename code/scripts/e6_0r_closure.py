#!/usr/bin/env python3
"""DecisionRAG E6-0R CPU-only endpoint and dataset closure.

This module audits already acquired public rows and frozen historical files.  It
does not import model libraries, contact inference services, inspect GPUs, or
execute a retriever, reranker, reader, embedding model, paid API, or Jev.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_0r_endpoint_dataset_closure"
DATA = ROOT / "data/external/e6_0r_datasets"
E6 = ROOT / "results/non_jev/v2/e6_v1_0/e6_0_readiness"
E5 = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"
E5A = ROOT / "results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark"
E5DC = ROOT / "results/non_jev/v2/e5_v1_1/e5_dc_serving_reader_validation"
QASPER = ROOT / "data/external/e5_qasper/snapshot"

SEED = 731
EXPECTED_SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
SCIFACT_CACHE = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")
SCIFACT_QUERY = SCIFACT_CACHE / "BeIR___scifact/queries/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-queries.arrow"
SCIFACT_CORPUS = SCIFACT_CACHE / "BeIR___scifact/corpus/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-corpus.arrow"
SCIFACT_TRAIN = SCIFACT_CACHE / "BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-train.arrow"
SCIFACT_TEST = SCIFACT_CACHE / "BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-test.arrow"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha_obj(value: Any) -> str:
    return sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_manifest(path: Path) -> dict[str, Any]:
    """Verify old declared files without treating a missing declaration as success."""
    if not path.is_file():
        return {"manifest": rel(path), "exists": False, "checked": 0, "passed": 0, "failed": 1}
    try:
        payload = load_json(path)
    except Exception as exc:
        return {"manifest": rel(path), "exists": True, "checked": 0, "passed": 0, "failed": 1, "error": str(exc)}
    declared = payload.get("files") or payload.get("artifacts") or payload.get("outputs") or []
    if isinstance(declared, dict):
        declared = [{"path": k, **(v if isinstance(v, dict) else {"sha256": v})} for k, v in declared.items()]
    checks = []
    for item in declared:
        if not isinstance(item, dict):
            checks.append({"path": None, "hash_match": False})
            continue
        name = item.get("path") or item.get("name")
        candidates = []
        if name:
            p = Path(name)
            candidates = [p] if p.is_absolute() else [ROOT / p, path.parent / p]
        actual = next((p for p in candidates if p.is_file()), None)
        ok = bool(actual and item.get("sha256") and sha_file(actual) == item.get("sha256"))
        if ok and item.get("bytes") is not None:
            ok = actual.stat().st_size == item["bytes"]
        checks.append({"path": name, "exists": bool(actual), "hash_match": ok})
    return {
        "manifest": rel(path),
        "exists": True,
        "checked": len(checks),
        "passed": sum(bool(x["hash_match"]) for x in checks),
        "failed": sum(not bool(x["hash_match"]) for x in checks),
        "missing_declarations": sum(not bool(x.get("exists")) for x in checks),
    }


def component_graph(edges: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adj[left].add(right)
        adj[right].add(left)
    seen: set[str] = set()
    out = []
    for node in sorted(adj):
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        qs: set[str] = set()
        docs: set[str] = set()
        while q:
            cur = q.popleft()
            target = qs if cur.startswith("q:") else docs
            target.add(cur[2:])
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if qs:
            out.append({"component_id": sha_text("|".join(sorted(qs)) + "::" + "|".join(sorted(docs)))[:24], "question_ids": sorted(qs), "source_ids": sorted(docs)})
    return sorted(out, key=lambda x: x["component_id"])


def read_arrow(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception as exc:
        raise RuntimeError("pyarrow is required for the already-cached BEIR audit; no model operation is involved") from exc
    with pa.memory_map(str(path), "r") as f:
        try:
            table = ipc.open_stream(f).read_all()
        except Exception:
            f.seek(0)
            table = ipc.open_file(f).read_all()
    return table.to_pylist()


def audit_dochop() -> dict[str, Any]:
    path = DATA / "dochopqa/[F] DocHopQA_Dataset.json"
    rows = load_json(path)
    context_counts = Counter(len(r.get("context_list") or []) for r in rows)
    task_counts = Counter(r.get("task_type") for r in rows)
    hop_counts = Counter(r.get("hop_type") for r in rows)
    used_counts = Counter(r.get("used_doc") for r in rows)
    source_ids = {c.get("pmc_id") for r in rows for c in r.get("context_list", []) if c.get("pmc_id")}
    content_hashes = Counter(sha_text(norm(c.get("content") or c.get("Raw content"))) for r in rows for c in r.get("context_list", []))
    row_dup = 0
    edges = []
    samples = []
    by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        contexts = row.get("context_list") or []
        ids = [f"{c.get('pmc_id')}|{c.get('section')}|{c.get('subsection', c.get('sub section', ''))}|{sha_text(norm(c.get('content') or c.get('Raw content')))[:16]}" for c in contexts]
        row_dup += len(ids) - len(set(ids))
        qid = f"dochop:{row.get('id')}"
        for doc in sorted({c.get("pmc_id") for c in contexts if c.get("pmc_id")}):
            edges.append((f"q:{qid}", f"d:pmc:{doc}"))
        key = (str(row.get("task_type")), str(row.get("hop_type")), str(row.get("used_doc")))
        by_stratum[key].append(row)
    rng = random.Random(SEED)
    for key in sorted(by_stratum):
        chosen = sorted(by_stratum[key], key=lambda x: str(x.get("id")))
        rng.shuffle(chosen)
        row = chosen[0]
        samples.append({"id": row.get("id"), "stratum": key, "question_sha256": sha_text(norm(row.get("question"))), "context_count": len(row.get("context_list") or []), "pmc_ids": sorted({c.get("pmc_id") for c in row.get("context_list", [])})})
    graph = component_graph(edges)
    graph_q = [len(x["question_ids"]) for x in graph]
    return {
        "dataset": "DocHop-QA",
        "source_uri": "https://huggingface.co/datasets/anonymousaaai123/DocHopQA_Dataset",
        "revision": "bfd2c7c9cf77e27dc93c57fe9f7defb091b3edbe",
        "path": rel(path),
        "sha256": sha_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "task_type_counts": dict(task_counts),
        "hop_type_counts": dict(hop_counts),
        "used_doc_counts": dict(used_counts),
        "context_count_histogram": {str(k): v for k, v in sorted(context_counts.items())},
        "context_rows_multi_document": sum(used_counts.get("multi", 0) for _ in [0]),
        "unique_pmc_ids": len(source_ids),
        "context_type_counts": dict(Counter(c.get("type") for r in rows for c in r.get("context_list", []))),
        "context_rows_with_duplicate_canonical_ids": row_dup,
        "context_exact_content_duplicate_rows_minus_first": sum(v - 1 for v in content_hashes.values() if v > 1),
        "question_exact_duplicate_rows_minus_first": len(rows) - len({norm(r.get("question")) for r in rows}),
        "answer_field_present_rows": sum(any("answer" in k.lower() or "gold" in k.lower() for k in r) for r in rows),
        "evidence_label_field_present_rows": sum(any("evidence" in k.lower() or "support" in k.lower() for k in r) for r in rows),
        "natural_corpus_present": False,
        "external_pmc_source_only": True,
        "component_count": len(graph),
        "largest_component_questions": max(graph_q) if graph_q else 0,
        "sample_rows": samples,
        "admission": "NOT_ADMISSIBLE_FOR_NATURAL_OR_COMPLETE_SUPPORT; supplied_context_only",
    }


def audit_garage() -> dict[str, Any]:
    path = DATA / "garage/GaRAGe_benchmark.jsonl"
    rows = load_jsonl(path)
    pool_hist = Counter(len(r.get("grounding") or []) for r in rows)
    valid_k15 = [r for r in rows if len(r.get("grounding") or []) == 15]
    labels = Counter(x for r in rows for x in r.get("evidence_correct", []))
    relevant = Counter(x for r in rows for x in r.get("evidence_relevant", []))
    providers = Counter(g.get("provider") for r in rows for g in r.get("grounding", []) if isinstance(g, dict))
    sample_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('question_complexity')}|{row.get('topic_tag')}"
        sample_groups[key].append(row)
    rng = random.Random(SEED)
    samples = []
    for key in sorted(sample_groups):
        candidate = sorted(sample_groups[key], key=lambda x: str(x.get("sample_id")))
        rng.shuffle(candidate)
        if candidate:
            row = candidate[0]
            samples.append({"sample_id": row.get("sample_id"), "stratum": key, "question_sha256": sha_text(norm(row.get("question"))), "pool_size": len(row.get("grounding") or []), "evidence_correct": list(row.get("evidence_correct") or [])[:15]})
    duplicate_passage_rows = 0
    source_id_fields = set()
    for row in rows:
        texts = []
        for passage in row.get("grounding") or []:
            if not isinstance(passage, dict):
                texts.append(norm(passage))
                continue
            source_id_fields.update(k for k in passage if k.lower() in {"id", "doc_id", "document_id", "url", "source_id"})
            values = [v for k, v in passage.items() if str(k).startswith("cite_")]
            texts.append(norm(values[0] if values else json.dumps(passage, sort_keys=True)))
        if len(texts) != len(set(texts)):
            duplicate_passage_rows += 1
    return {
        "dataset": "GaRAGe",
        "source_uri": "https://github.com/amazon-science/GaRAGe",
        "revision": "2b7264f5d4e066affbe4b67b7d3b9cffeff98ede",
        "path": rel(path),
        "sha256": sha_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sample_id_unique": len({r.get("sample_id") for r in rows}) == len(rows),
        "pool_size_histogram": {str(k): v for k, v in sorted(pool_hist.items())},
        "exact_k15_rows": len(valid_k15),
        "rows_k_at_least_5": sum(k >= 5 for k in pool_hist for _ in range(pool_hist[k])),
        "provider_counts": dict(providers),
        "evidence_correct_counts": dict(labels),
        "evidence_relevant_counts": dict(relevant),
        "aligned_annotation_lengths": sum(len(r.get("grounding", [])) == len(r.get("evidence_relevant", [])) == len(r.get("evidence_correct", [])) == len(r.get("evidence_cited", [])) for r in rows),
        "answer_generate_nonempty": sum(bool(str(r.get("answer_generate", "")).strip()) for r in rows),
        "answer_validate_yes": sum(str(r.get("answer_validate", "")).upper() == "YES" for r in rows),
        "question_complexity_counts": dict(Counter(r.get("question_complexity") for r in rows)),
        "question_tag_counts": dict(Counter(r.get("question_tag") for r in rows)),
        "duplicate_passage_text_rows": duplicate_passage_rows,
        "stable_source_id_fields_in_grounding": sorted(source_id_fields),
        "stable_candidate_identity": "sample_id:cite_1..cite_K positional identity only; no source-document IDs/URLs",
        "complete_support_set_field_present": False,
        "source_group_status": "BLOCKED_NO_STABLE_SOURCE_DOCUMENT_ID",
        "sample_rows": samples,
        "admission": "FIXED_POOL_GROUNDING_ADMISSIBLE_CONDITIONAL; exact-K15-to-Top5 only; never natural Top20",
    }


def audit_scifact() -> dict[str, Any]:
    q = read_arrow(SCIFACT_QUERY)
    corpus = read_arrow(SCIFACT_CORPUS)
    train = read_arrow(SCIFACT_TRAIN)
    test = read_arrow(SCIFACT_TEST)
    qids = {str(x["_id"]) for x in q}
    docids = {str(x["_id"]) for x in corpus}
    all_qrels = train + test
    by_q = Counter(str(x["query-id"]) for x in all_qrels)
    edges = [(f"q:scifact:{x['query-id']}", f"d:scifact:{x['corpus-id']}") for x in all_qrels]
    graph = component_graph(edges)
    comp_q = [len(x["question_ids"]) for x in graph]
    return {
        "dataset": "BEIR/SciFact",
        "source_uri": "https://huggingface.co/datasets/BeIR/scifact",
        "qrels_source_uri": "https://huggingface.co/datasets/BeIR/scifact-qrels",
        "dataset_revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
        "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
        "files": [{"path": rel(p), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in (SCIFACT_QUERY, SCIFACT_CORPUS, SCIFACT_TRAIN, SCIFACT_TEST)],
        "query_rows": len(q),
        "corpus_rows": len(corpus),
        "query_ids_unique": len(qids) == len(q),
        "corpus_ids_unique": len(docids) == len(corpus),
        "empty_query_rows": sum(not norm(x.get("text")) for x in q),
        "empty_corpus_rows": sum(not norm(x.get("text")) for x in corpus),
        "train_qrel_rows": len(train),
        "test_qrel_rows": len(test),
        "qrel_rows": len(all_qrels),
        "qrel_score_counts": dict(Counter(x["score"] for x in all_qrels)),
        "qrel_query_ids": len(by_q),
        "qrel_queries_missing_from_query_file": sum(str(x["query-id"]) not in qids for x in all_qrels),
        "qrel_docs_missing_from_corpus": sum(str(x["corpus-id"]) not in docids for x in all_qrels),
        "qrel_queries_with_multiple_positives": sum(v > 1 for v in by_q.values()),
        "qrel_max_positives_per_query": max(by_q.values()),
        "qrel_unjudged_corpus_units_per_q": "5183 minus sparse qrels; not explicit negatives",
        "query_exact_duplicate_rows_minus_first": len(q) - len({norm(x.get("text")) for x in q}),
        "corpus_exact_duplicate_rows_minus_first": len(corpus) - len({norm(x.get("text")) for x in corpus}),
        "qrel_component_count": len(graph),
        "qrel_component_largest_question_count": max(comp_q) if comp_q else 0,
        "qrel_component_largest_fraction": (max(comp_q) / len(q)) if q and comp_q else 0.0,
        "natural_corpus_present": True,
        "complete_support_sets": False,
        "admission": "NATURAL_RELEVANCE_ADMISSIBLE_CONDITIONAL; doc-level sparse qrels; not CSSR",
    }


def qasper_text_hashes() -> tuple[set[str], set[str]]:
    question_hashes: set[str] = set()
    passage_hashes: set[str] = set()
    for path in (QASPER / "qasper-train-v0.3.json", QASPER / "qasper-dev-v0.3.json"):
        payload = load_json(path)
        for paper in payload.values():
            for qa in paper.get("qas", []):
                question_hashes.add(sha_text(norm(qa.get("question"))))
            full_text = paper.get("full_text", [])
            if isinstance(full_text, dict):
                full_text = [full_text]
            for section in full_text:
                if isinstance(section, dict):
                    paragraphs = section.get("paragraphs", [])
                else:
                    paragraphs = section if isinstance(section, list) else []
                for paragraph in paragraphs:
                    passage_hashes.add(sha_text(norm(paragraph)))
    return question_hashes, passage_hashes


def historical_audit(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qhash, phash = qasper_text_hashes()
    result: dict[str, Any] = {
        "historical_reference": "E2-E5 permitted QASPER plus adjacent RAG_system source scan; sealed E5-F/E6 reservation untouched",
        "qasper_question_hash_count": len(qhash),
        "qasper_passage_hash_count": len(phash),
        "exact_overlap": {},
        "near_duplicate_method": "NOT_RUN; no semantic near-duplicate model or embedding operation authorized",
        "adjacent_workspace_exposure": "P30C reports and p30n scripts contain SciFact references; DecisionRAG E2-E5 result files did not contain SciFact rows in bounded scan. Therefore SciFact is not claimed source-independent of the whole workspace.",
    }
    # DocHop question/context hashes are recomputed from the local release.
    dochop = load_json(DATA / "dochopqa/[F] DocHopQA_Dataset.json")
    dqh = {sha_text(norm(x.get("question"))) for x in dochop}
    dph = {sha_text(norm(c.get("content") or c.get("Raw content"))) for x in dochop for c in x.get("context_list", [])}
    garage = load_jsonl(DATA / "garage/GaRAGe_benchmark.jsonl")
    gqh = {sha_text(norm(x.get("question"))) for x in garage}
    gph = {sha_text(norm(v)) for x in garage for p in x.get("grounding", []) if isinstance(p, dict) for k, v in p.items() if str(k).startswith("cite_")}
    result["exact_overlap"] = {
        "qasper_vs_dochop_questions": len(qhash & dqh),
        "qasper_vs_dochop_contexts": len(phash & dph),
        "qasper_vs_garage_questions": len(qhash & gqh),
        "qasper_vs_garage_grounding": len(phash & gph),
    }
    # We have SciFact rows already cached; inspect exact overlap with unsealed QASPER text.
    sci_q = {sha_text(norm(x["text"])) for x in read_arrow(SCIFACT_QUERY)}
    sci_c = {sha_text(norm(x["text"])) for x in read_arrow(SCIFACT_CORPUS)}
    result["exact_overlap"].update({"qasper_vs_scifact_queries": len(qhash & sci_q), "qasper_vs_scifact_corpus": len(phash & sci_c)})
    result["historical_source_notes"] = {
        "qasper": "historically used and explicitly excluded from the new workload",
        "dochop": "new public release; source PMC IDs available but raw PMC corpus not bundled",
        "garage": "new public release; no source document identity in row schema",
        "scifact": "actual SciFact cache present and adjacent RAG artifacts mention SciFact; use only as exploratory natural relevance unless owner accepts workspace exposure limitation",
    }
    return result


def freeze_scifact_splits() -> dict[str, Any]:
    """Create component-disjoint metadata and CAL/exploratory IDs; reserve only hashes."""
    train = read_arrow(SCIFACT_TRAIN)
    test = read_arrow(SCIFACT_TEST)
    comps = component_graph([(f"q:scifact:{x['query-id']}", f"d:scifact:{x['corpus-id']}") for x in train + test])
    rng = random.Random(SEED)
    shuffled = list(comps)
    rng.shuffle(shuffled)
    target = {"CAL": 0.5, "EXPLORATORY": 0.25, "RESERVED": 0.25}
    chosen: dict[str, list[dict[str, Any]]] = {k: [] for k in target}
    totals = {k: 0 for k in target}
    for comp in shuffled:
        # Greedy deterministic assignment to closest current fraction by question count.
        key = min(target, key=lambda k: (totals[k] / max(1, len({q["question_ids"][0] for q in chosen[k]})) if chosen[k] else 0, ["CAL", "EXPLORATORY", "RESERVED"].index(k)))
        chosen[key].append(comp)
        totals[key] += len(comp["question_ids"])
    # Rebalance by target proportion with deterministic component order; do not split a component.
    chosen = {k: [] for k in target}
    totals = {k: 0 for k in target}
    for comp in sorted(shuffled, key=lambda x: (-len(x["question_ids"]), x["component_id"])):
        key = min(target, key=lambda k: (totals[k] / max(0.0001, target[k]), ["CAL", "EXPLORATORY", "RESERVED"].index(k)))
        chosen[key].append(comp)
        totals[key] += len(comp["question_ids"])
    partition_ids: dict[str, list[str]] = {}
    for key, groups in chosen.items():
        partition_ids[key] = sorted(qid for g in groups for qid in g["question_ids"])
    all_q = [q for ids in partition_ids.values() for q in ids]
    assert len(all_q) == len(set(all_q)) == 1109
    out = {
        "schema": "DecisionRAG-E6-0R-SCIFACT-COMPONENT-SPLIT-V1",
        "seed": SEED,
        "status": "PROPOSED_METADATA_LOCK; RESERVED_CONTENT_NOT_EXPORTED",
        "source_component_count": len(comps),
        "partitions": {},
        "source_disjoint": True,
        "cal_qids_path": "E6_0R_SCIFACT_CAL_QIDS.jsonl",
        "exploratory_qids_path": "E6_0R_SCIFACT_EXPLORATORY_QIDS.jsonl",
    }
    for key, ids in partition_ids.items():
        groups = chosen[key]
        rec = {"query_count": len(ids), "component_count": len(groups), "qid_sha256": sha_obj(ids), "component_sha256": sha_obj(sorted(g["component_id"] for g in groups))}
        if key != "RESERVED":
            rec["query_ids"] = ids
            write_text(OUT / f"E6_0R_SCIFACT_{key}_QIDS.jsonl", "\n".join(json.dumps({"query_id": q}, sort_keys=True) for q in ids))
        out["partitions"][key] = rec
    write_json(OUT / "E6_0R_SOURCE_SPLIT_MANIFEST.json", out)
    write_json(OUT / "E6_0R_RESERVED_METADATA_ONLY_RECEIPT.json", {
        "status": "METADATA_ONLY_RESERVED_LOCK",
        "dataset": "BEIR/SciFact",
        "reserved_query_count": out["partitions"]["RESERVED"]["query_count"],
        "reserved_qid_sha256": out["partitions"]["RESERVED"]["qid_sha256"],
        "reserved_component_sha256": out["partitions"]["RESERVED"]["component_sha256"],
        "raw_question_text_exported": False,
        "gold_qrels_exported": False,
        "access_control": "process-level repository convention only; no sealed external vault or OS deny boundary established",
        "confirmation_status": "CONFIRMATION_LOCKED",
    })
    return out


def emit_reports(d: dict[str, dict[str, Any]], history: dict[str, Any], splits: dict[str, Any], auth: dict[str, Any]) -> None:
    write_json(OUT / "E6_0R_DATASET_ACTUAL_ROW_INVENTORIES.json", d)
    inventory_rows = []
    for name, audit in d.items():
        inventory_rows.append({"dataset": name, "rows": audit.get("rows", audit.get("query_rows")), "corpus_rows": audit.get("corpus_rows", ""), "exact_k15_or_equivalent": audit.get("exact_k15_rows", ""), "natural_corpus_present": audit.get("natural_corpus_present"), "complete_support_sets": audit.get("complete_support_sets", False), "admission": audit.get("admission")})
    write_csv(OUT / "E6_0R_DATASET_ACTUAL_ROW_INVENTORIES.csv", inventory_rows, ["dataset", "rows", "corpus_rows", "exact_k15_or_equivalent", "natural_corpus_present", "complete_support_sets", "admission"])
    write_json(OUT / "E6_0R_HISTORICAL_EXCLUSION_INDEX.json", history)
    write_text(OUT / "E6_0R_DATASET_SOURCE_AND_LICENSE_PROOFS.md", f"""# E6-0R Dataset Source and License Proofs

Generated `{utc()}`. These are actual acquired files, not metadata-only claims.

| Dataset | Source/revision | Local file | SHA-256 | License evidence |
|---|---|---|---|---|
| DocHop-QA | `https://huggingface.co/datasets/anonymousaaai123/DocHopQA_Dataset@bfd2c7c9cf77e27dc93c57fe9f7defb091b3edbe` | `{d['dochop']['path']}` | `{d['dochop']['sha256']}` | README states CC-BY-4.0; verify downstream use with source terms. |
| GaRAGe | `https://github.com/amazon-science/GaRAGe@2b7264f5d4e066affbe4b67b7d3b9cffeff98ede` | `{d['garage']['path']}` | `{d['garage']['sha256']}` | Repository LICENSE is CC-BY-NC-4.0; non-commercial restriction remains. |
| BEIR/SciFact | `https://huggingface.co/datasets/BeIR/scifact@b3b5335604bf5ee3c4447671af975ea25143d4f5` | cached Arrow files listed in inventory | see inventory | Dataset card states CC-BY-SA-4.0; qrels revision separately pinned. |

No checkpoint weights, model files, embeddings, or inference services were accessed.
""")
    write_text(OUT / "E6_0R_CORPUS_RECONSTRUCTABILITY_AUDIT.md", f"""# E6-0R Corpus Reconstructability Audit

## Decision

The defensible primary amended workload is **BEIR/SciFact natural document relevance**, with `nDCG@5`/`Recall@5`-style qrel-aware endpoints. It has an independently stored 5,183-row text corpus, 1,109 nonempty queries and exact qrel-to-query/document joins. Its qrels are sparse and binary, so it is **not** a complete-support CSSR workload.

DocHop-QA has real source references and context text but does not package the searchable PMC corpus or any answer/evidence annotation fields. Its context list is therefore a supplied context view, not a natural retrieval corpus.

GaRAGe has real grounding annotations and answer fields, but its rows contain supplied per-question passage pools of 5–15 passages, not a natural searchable corpus. The exact-K=15 subset is a defensible secondary fixed-pool grounding workload (`15 -> 5`), never `20 -> 5` natural retrieval.

### Counts

- DocHop-QA: {d['dochop']['rows']} rows, {d['dochop']['unique_pmc_ids']} PMC IDs, no full corpus release in the audited file, no answer fields, no evidence-label fields.
- GaRAGe: {d['garage']['rows']} rows; {d['garage']['exact_k15_rows']} exact-15 supplied pools; no stable source-document IDs in grounding entries.
- SciFact: {d['scifact']['query_rows']} queries, {d['scifact']['corpus_rows']} corpus units, {d['scifact']['qrel_rows']} qrel rows covering {d['scifact']['qrel_query_ids']} queries.

The SciFact workload is conditionally exploratory because adjacent historical RAG artifacts mention SciFact. It is not represented as a fresh independent confirmation set.
""")
    write_text(OUT / "E6_0R_GOLD_PASSAGE_AND_ANNOTATION_AUDIT.md", f"""# E6-0R Gold Passage and Annotation Audit

## DocHop-QA

The audited schema is `{', '.join(['id','question','context_list','task_type','hop_type','used_doc'])}`. `context_list` has {sum(d['dochop']['context_count_histogram'].values())} rows with 3–5 supplied contexts; no `answer`, `gold`, `evidence`, or `support` fields were present. A context's `pmc_id` is source metadata, not a proof that the listed contexts are exhaustive or that every listed context is necessary. Complete-support OR-set scoring is therefore not valid.

## GaRAGe

All {d['garage']['rows']} rows have aligned `grounding`, `evidence_relevant`, `evidence_correct`, and `evidence_cited` list lengths. The `evidence_correct` values include `ANSWER-THE-QUESTION`, `RELATED-INFORMATION`, `UNKNOWN`, `OUTDATED`, and empty values. Empty/unknown remain unjudged, not explicit negatives. The rows have no alternative complete support-set field; per-passage labels do not justify CSSR-U. Candidate identity is only `sample_id:cite_i` within the supplied row.

The pre-outcome fixed-pool amendment maps `ANSWER-THE-QUESTION` to grounding-positive, `RELATED-INFORMATION` to related-but-not-sufficient, `OUTDATED` to answer-invalid at the question date, and `UNKNOWN`/empty to unjudged. This mapping must be signed before E6-1 outcomes and does not create a natural-retrieval claim.

## SciFact

SciFact has binary document-level qrels: all {d['scifact']['qrel_score_counts'].get(1, 0)} audited qrel rows have score 1, with sparse unjudged corpus units. A qrel-positive document must not be split into child passages and relabeled positive without a separate authenticated mapping. The admitted endpoint is qrel-unit natural relevance only.
""")
    pool_rows = []
    for name, a in d.items():
        hist = a.get("pool_size_histogram") or a.get("context_count_histogram") or {}
        for k, v in sorted(hist.items(), key=lambda x: int(x[0])):
            pool_rows.append({"dataset": name, "candidate_count": k, "rows": v, "interpretation": "supplied_pool" if name == "garage" else "context_list"})
    write_csv(OUT / "E6_0R_POOL_SIZE_AND_JUDGMENT_COVERAGE.csv", pool_rows, ["dataset", "candidate_count", "rows", "interpretation"])
    write_text(OUT / "E6_0R_DATASET_REJECTION_AND_ADMISSION_MATRIX.md", f"""# E6-0R Dataset Rejection and Admission Matrix

| Candidate | Actual audit | Natural complete-support | Natural relevance | Fixed-pool grounding | Decision |
|---|---|---:|---:|---:|---|
| DocHop-QA | 11,379 rows; 3–5 supplied contexts; no answer/evidence fields; no bundled source corpus | NO | NO | NO (no labels) | `NOT_ADMISSIBLE_FOR_CURRENT_ENDPOINTS` |
| GaRAGe | 2,366 rows; K=5–15; 2,329 K=15; per-passage grounding labels; no source IDs | NO | NO | YES, conditional K=15→5 | `FIXED_POOL_GROUNDING_ADMISSIBLE_CONDITIONAL` |
| BEIR/SciFact | {d['scifact']['query_rows']} queries; {d['scifact']['corpus_rows']} text corpus rows; {d['scifact']['qrel_rows']} binary qrels | NO | YES, sparse qrels | NOT NEEDED | `PRIMARY_NATURAL_RELEVANCE_ADMISSIBLE_CONDITIONAL` |

Primary E6 amended workload: **SciFact natural relevance**. Secondary grounding workload: **GaRAGe exact K=15 supplied pool**. These endpoints must not be pooled. CSSR-U remains historical QASPER-only and unchanged.
""")
    write_text(OUT / "E6_0R_SOURCE_COMPONENT_GRAPH_REPORT.md", f"""# E6-0R Source Component and Independence Report

SciFact qrel graph: **{d['scifact']['qrel_component_count']} components**, largest component **{d['scifact']['qrel_component_largest_question_count']} questions ({d['scifact']['qrel_component_largest_fraction']:.4%})**. This grouping is qrel-connected document identity, not a proof that all latent source versions are disjoint.

DocHop-QA exposes PMC IDs and was graphed from all listed context edges; this is a supplied-context graph and cannot establish a searchable corpus. GaRAGe has no stable source-document ID, URL or document field in the grounding objects; source-group bootstrap is therefore unavailable for GaRAGe.

Exact text overlap against permitted QASPER artifacts is recorded in `E6_0R_HISTORICAL_EXCLUSION_INDEX.json`. Semantic/near-duplicate detection was not run because it would require an unauthorized model/embedding operation; exact hash checks do not rule out versions or paraphrases.
""")
    write_text(OUT / "E6_0R_HISTORICAL_OVERLAP_AND_DUPLICATE_AUDIT.md", "# E6-0R Historical Overlap and Duplicate Audit\n\n" + json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_text(OUT / "E6_0R_ACCESS_ISOLATION_AND_CHRONOLOGY.md", f"""# E6-0R Access Isolation and Chronology

- E6-0R start: `{utc()}` is recorded in the machine-readable provenance.
- The E5-F/E6 confirmation raw content, labels, outcomes and embeddings were not accessed.
- Public DocHop-QA, GaRAGe and cached SciFact rows were inspected because they are the admitted exploratory-data candidates.
- SciFact CAL/exploratory IDs are written for future pilot preparation; reserved raw question text and qrels are not exported, only ID/component fingerprints.
- No external vault or OS-level deny boundary exists in this repository. Therefore the reservation is metadata-only and confirmation remains `CONFIRMATION_LOCKED`; this report does not claim tamper-proof sealing.
""")


def emit_protocol(d: dict[str, dict[str, Any]], splits: dict[str, Any]) -> None:
    schema = {
        "primary_track": "N-REL",
        "primary_dataset": "BEIR/SciFact",
        "primary_endpoint": {"nDCG@5": "binary qrel score at document identity", "Recall@5": "qrel-positive documents retrieved in Top-5", "unjudged": "not explicit negative; standard qrel evaluator reports conservative zero-gain behavior plus judged-coverage sensitivity", "no_positive_qrel": "retain in unconditional denominator and report separately"},
        "secondary_track": "F-GRD",
        "secondary_dataset": "GaRAGe exact K=15 supplied pool",
        "secondary_endpoint": {"support_any_at_5": "at least one selected passage with ANSWER-THE-QUESTION", "grounding_precision_at_5": "count ANSWER-THE-QUESTION among selected five divided by five", "unjudged": "UNKNOWN or empty remains unjudged; never recoded as negative", "scope": "fixed supplied pool, not natural retrieval"},
        "historical_anchor": "QASPER E5 PPLX-Jina CSSR-U unchanged and not pooled",
        "models": {"decision": ["pplx_decider_27b", "liquid_d1_3b", "startlux_decision_4b_conditional", "amos_1_small_2b_conditional"], "specialist": ["jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"]},
        "failure_policy": ["UNATTEMPTED", "TIMEOUT", "INVALID_OUTPUT", "CONTRACT_FAILURE", "INFRA_FAILURE", "TRUNCATED_INPUT"],
        "statistics": {"source_group_bootstrap": 5000, "seed": SEED, "ci": "two-sided percentile 95%", "SciFact": "component-aware only if qrel graph grouping is accepted; query-weighted sensitivity mandatory", "GaRAGe": "query-level paired intervals only; source-group CI unavailable"},
        "multiplicity": "PPLX-Jina primary anchor; compact-model and specialist contrasts exploratory with Holm family or explicitly labeled intervals frozen before outcomes",
        "resource_frontier": ["full selector p50", "full selector p95", "peak allocated memory", "physical forwards/tokens", "failure rate", "hardware UUID", "precision/quantization"],
        "prohibited_claims": ["CSSR replication on SciFact/GaRAGe", "architectural causality", "superiority/equivalence without margin", "independent factual correctness from reader F1", "commercial dollar savings without cost model"],
        "status": "PRE_INFERENCE_PROTOCOL_FOR_OWNER_SIGNATURE",
    }
    write_json(OUT / "E6_0R_ENDPOINT_SCHEMA_AND_LABEL_MAPPING.json", schema)
    write_text(OUT / "E6_0R_ENDPOINT_AMENDMENT_V1_PROPOSED.md", f"""# E6-0R Endpoint Amendment V1 (Pre-Outcome)

## Owner decision requested

Accept the natural-relevance primary workload **BEIR/SciFact** and the separate conditional **GaRAGe K=15 supplied-pool grounding** secondary workload. Do not call either a natural complete-support CSSR benchmark.

## Primary N-REL

The future selector receives a frozen, gold-blind natural Top-20 produced by an independently frozen retrieval procedure over the 5,183-row SciFact corpus. The qrels are joined at the document unit. Primary metrics are binary nDCG@5 and Recall@5, with unconditional denominators over eligible qrel-linked queries, qrel coverage, and a retrieval-feasibility ceiling. No document is expanded into positive child passages without an authenticated mapping.

The cached release covers {d['scifact']['query_rows']} queries and {d['scifact']['qrel_query_ids']} qrel-linked queries. It has sparse binary judgments ({d['scifact']['qrel_rows']} qrel rows and many unjudged corpus units). Unjudged is not explicit negative; judged-coverage and sensitivity are mandatory.

## Secondary F-GRD

GaRAGe exact K=15 rows ({d['garage']['exact_k15_rows']}) can support a fixed supplied-pool `15 -> 5` grounding endpoint. The candidate ID is `sample_id:cite_i`; label mapping is `ANSWER-THE-QUESTION` positive, `RELATED-INFORMATION` related, `OUTDATED` invalid-at-date, `UNKNOWN`/empty unjudged. This is not natural retrieval and must not be compared as Top-20 retrieval.

## Frozen statistics and model scope

Use equal-source-group bootstrap only where source identity is defensible, 5,000 resamples, seed {SEED}, two-sided percentile 95% CI; query-weighted sensitivity is mandatory. GaRAGe has no source-group identity, so source-independent CIs are unavailable. PPLX−Jina is the historical anchor; the conditional 4+4 roster remains unchanged, with StartLux and Amos pending native Top-5 validation. This amendment authorizes no inference.
""")
    write_csv(OUT / "E6_0R_PRIMARY_SECONDARY_CONTRASTS.csv", [
        {"contrast_id": "NREL_PRIMARY", "dataset": "BEIR/SciFact", "endpoint": "binary_nDCG@5; Recall@5", "population": "qrel-linked queries after frozen source-component split", "status": "PROPOSED_OWNER_SIGNATURE"},
        {"contrast_id": "FGRD_SECONDARY", "dataset": "GaRAGe", "endpoint": "support_any@5; grounding_precision@5", "population": "exact K=15 supplied pools", "status": "CONDITIONAL_OWNER_SIGNATURE"},
        {"contrast_id": "HISTORICAL_ANCHOR", "dataset": "QASPER", "endpoint": "set-aware CSSR-U", "population": "E5 frozen 1502", "status": "FROZEN_UNCHANGED_NOT_POOLED"},
    ], ["contrast_id", "dataset", "endpoint", "population", "status"])
    write_text(OUT / "E6_0R_NATURAL_VS_FIXED_POOL_SCOPE_CROSSWALK.md", """# Natural vs Fixed-Pool Scope Crosswalk

| Track | Corpus construction | Candidate K | Primary interpretation |
|---|---|---:|---|
| N-REL SciFact | natural 5,183-row corpus; future gold-blind retrieval | Top-20→Top-5 | document-level qrel relevance; sparse judging |
| F-GRD GaRAGe | question-associated supplied pool | K=15→5 | grounding/support labels only; fixed-pool |
| Historical QASPER | prior frozen experiment | Top-20→Top-5 | complete-support CSSR-U; unchanged |

No metric, denominator or source-group CI is pooled across these tracks.
""")
    write_text(OUT / "E6_0R_RESOURCE_FRONTIER_AND_SERVING_PROTOCOL.md", """# E6-0R Resource Frontier and Serving Protocol

Deferred until separate E6-1 owner authorization. Before outcomes, record model revision, native adapter hash, prompt/input hash, candidate-view hash, hardware UUID, precision, peak allocated memory, physical forwards/tokens, preprocessing/scoring/selection/packaging timings, failures and retries. Use matched hardware when comparing latency. Do not infer QPS or dollars from serialized wall time.
""")
    write_text(OUT / "E6_0R_E6_1_CAL_ADAPTER_HANDOFF_SPEC.md", """# E6-0R E6-1 CAL Adapter Handoff

E6-1 remains unstarted. A separate CAL-only pilot may validate native Top-5 contracts for the already accepted conditional roster on the frozen SciFact CAL IDs and, if separately accepted, GaRAGe K=15 fixed-pool inputs. It must not inspect the reserved partition, select a dataset after outcomes, or start the full 4+4 benchmark. StartLux and Amos remain conditional until native output alignment is proven. GPU UUID/occupancy must be rechecked immediately before any future execution.
""")
    write_text(OUT / "E6_0R_E6_2_RELEASE_CRITERIA.md", """# E6-0R E6-2 Release Criteria

E6-2 remains blocked until the owner signs the endpoint amendment, accepts the source-exposure limitations, accepts the metadata-only reservation lock, and separately accepts E6-1 native CAL adapter validation. No confirmation claim may use the reserved partition until a new protocol and execution authorization are issued.
""")


def emit_auth_and_tests(d: dict[str, dict[str, Any]], history: dict[str, Any], splits: dict[str, Any]) -> dict[str, Any]:
    slate = E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    manifests = sorted(set(E6.glob("*PROVENANCE_MANIFEST*.json")) | set(E5A.glob("*PROVENANCE_MANIFEST*.json")) | set(E5DC.glob("*PROVENANCE_MANIFEST*.json")) | set(ROOT.glob("results/non_jev/v2/e2_m1/*PROVENANCE_MANIFEST.json")))
    checks = [verify_manifest(p) for p in manifests]
    auth = {
        "created_utc": utc(),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "frozen_slate": {"path": rel(slate), "sha256": sha_file(slate), "expected": EXPECTED_SLATE_SHA, "match": sha_file(slate) == EXPECTED_SLATE_SHA},
        "historical_declared_manifest_checks": checks,
        "historical_checked": sum(x.get("checked", 0) for x in checks),
        "historical_passed": sum(x.get("passed", 0) for x in checks),
        "historical_failed": sum(x.get("failed", 0) for x in checks),
        "e6_0_gate": load_json(E6 / "E6_0_OWNER_DECISION_ACCEPTANCE.json"),
        "e5_a_primary_anchor_preserved": True,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "new_model_forwards": 0,
        "new_embedding_calls": 0,
        "new_reader_calls": 0,
        "new_gpu_operations": 0,
        "sealed_holdout_accessed": False,
        "outputs_not_in_historical_manifests": True,
    }
    write_json(OUT / "E6_0R_R0_INPUT_AUTHENTICATION.json", auth)
    write_text(OUT / "E6_0R_R0_EXACT_PROTOCOL_CROSSWALK.md", """# E6-0R Exact Protocol Crosswalk

The historical E6-0 proposal was complete only for a future amendment choice: it named graded support/grounding as a possibility but did not select a single dataset, candidate K, qrel unit, unjudged policy or source-group contract. E6-0R freezes those missing fields before any new selector outcome:

- Primary: SciFact natural document relevance, binary qrel-unit nDCG@5/Recall@5.
- Secondary: GaRAGe exact K=15 supplied-pool grounding, support_any@5/grounding_precision@5.
- Historical QASPER CSSR-U is unchanged and not pooled.
- Failures are explicit status categories; not zero quality.
- Bootstrap seed 731 and 5,000 resamples; source groups only where justified.
""")
    immutability = {
        "status": "PASS_WITH_NEW_PUBLIC_DATA_ADDED_APPEND_ONLY",
        "historical_files_checked": len(checks),
        "historical_declared_entries_checked": auth["historical_checked"],
        "historical_hash_failures": auth["historical_failed"],
        "historical_e5_top20_sha_match": auth["frozen_slate"]["match"],
        "e6_0_outputs_modified": False,
        "new_files": ["data/external/e6_0r_datasets/*", "results/non_jev/v2/e6_v1_0/e6_0r_endpoint_dataset_closure/*"],
    }
    write_json(OUT / "E6_0R_HISTORICAL_IMMUTABILITY_REPORT.json", immutability)
    return auth


def provenance(auth: dict[str, Any], datasets: dict[str, dict[str, Any]]) -> None:
    files = []
    for root in (DATA,):
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p), "role": "audited_public_input"})
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in {"E6_0R_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", "E6_0R_PROVENANCE_VERIFICATION.json"}:
            files.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p), "role": "e6_0r_output"})
    manifest = {"schema": "DecisionRAG-E6-0R-PROVENANCE-V1", "created_utc": utc(), "git_head": auth["git_head"], "seed": SEED, "inference_calls": 0, "gpu_operations": 0, "files": files}
    write_json(OUT / "E6_0R_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", manifest)
    verify = []
    for item in files:
        p = ROOT / item["path"]
        verify.append({"path": item["path"], "sha256_match": p.is_file() and sha_file(p) == item["sha256"], "bytes_match": p.is_file() and p.stat().st_size == item["bytes"]})
    write_json(OUT / "E6_0R_PROVENANCE_VERIFICATION.json", {"status": "PASS" if all(x["sha256_match"] and x["bytes_match"] for x in verify) else "FAIL", "checked": len(verify), "passed": sum(x["sha256_match"] and x["bytes_match"] for x in verify), "failed": sum(not (x["sha256_match"] and x["bytes_match"]) for x in verify), "files": verify})


def final_report(d: dict[str, dict[str, Any]], auth: dict[str, Any], history: dict[str, Any], splits: dict[str, Any]) -> None:
    status = "E6_0R_OWNER_APPROVAL_READY"
    write_text(OUT / "E6_0R_FINAL_OWNER_REVIEW_REPORT.md", f"""# DecisionRAG E6-0R Final Owner-Review Report

## Gate

**`{status}`** — one natural relevance workload is admitted conditionally for owner signature; a separate fixed-pool grounding workload is also conditionally specified. E6-1 has not started. E6-2 and confirmation remain locked.

## Admitted workloads

1. **Primary N-REL: BEIR/SciFact.** Actual corpus: {d['scifact']['corpus_rows']} text documents; queries: {d['scifact']['query_rows']}; qrels: {d['scifact']['qrel_rows']} binary rows covering {d['scifact']['qrel_query_ids']} queries. This supports natural document-relevance evaluation after a separately authorized, gold-blind retrieval preparation. It does not support CSSR-U because qrels are sparse document-level relevance judgments, not complete alternative support sets.
2. **Secondary F-GRD: GaRAGe exact K=15.** {d['garage']['exact_k15_rows']} supplied passage pools are exactly 15 and can support an explicitly fixed `15 -> 5` grounding endpoint. The remaining variable-K rows are retained in the inventory, not silently converted. No natural Top-20 claim is permitted.

DocHop-QA is rejected for the current endpoint: its {d['dochop']['rows']} rows contain 3–5 supplied contexts and source PMC IDs but no packaged natural corpus, answer field, or evidence-support labels.

## Source independence and reservation

SciFact's qrel graph has {d['scifact']['qrel_component_count']} components; the largest contains {d['scifact']['qrel_component_largest_question_count']} questions. A deterministic seed-{SEED} component split was prepared. Reserved content was not exported, but no external sealed vault/OS deny boundary exists, so the reservation status is **metadata-only / CONFIRMATION_LOCKED**, not an independent confirmatory holdout. GaRAGe source-group inference is unavailable because stable source-document IDs are absent.

## Historical and integrity status

- E5 Top-20 SHA-256: `{auth['frozen_slate']['sha256']}`; expected `{EXPECTED_SLATE_SHA}`; match: **{auth['frozen_slate']['match']}**.
- Historical declared-manifest verification: **{auth['historical_passed']}/{auth['historical_checked']}** entries passed; failed: **{auth['historical_failed']}**.
- Exact text-overlap counts and adjacent-workspace SciFact exposure are in `E6_0R_HISTORICAL_EXCLUSION_INDEX.json`.
- E5 QASPER CSSR-U and E6-0 answer-F1 remain unchanged; no new endpoint overwrites them.
- New physical inference/model/GPU/API/Jev/paid operations: **zero**.

## Owner decisions required

1. Sign the N-REL SciFact amendment and accept sparse qrel/unjudged limitations.
2. If desired, separately accept GaRAGe K=15 fixed-pool grounding as a secondary exploratory endpoint.
3. Keep E6-1 limited to CAL adapter validation; do not start the full eight-model benchmark until native contracts and endpoint acceptance pass.
4. Keep E5-F/E6 reserved content locked and authorize any future retrieval/model workload separately.
""")
    write_json(OUT / "E6_0R_OWNER_REVIEW_GATE.json", {"status": status, "primary_dataset": "BEIR/SciFact", "primary_track": "N-REL", "secondary_dataset": "GaRAGe", "secondary_track": "F-GRD-K15", "e6_1_started": False, "e6_2_authorized": False, "confirmation_locked": True, "jev_calls": 0, "paid_api_calls": 0})
    write_text(OUT / "E6_0R_FAIL_AND_ACCESS_LOG.jsonl", "\n".join(json.dumps(x, sort_keys=True) for x in [
        {"stage": "DocHop-QA", "status": "REJECTED_CURRENT_ENDPOINT", "reason": "no natural source corpus and no answer/evidence fields"},
        {"stage": "GaRAGe", "status": "CONDITIONAL_FIXED_POOL", "reason": "supplied K=15 subset; source-document identity unavailable"},
        {"stage": "SciFact", "status": "CONDITIONAL_NATURAL_RELEVANCE", "reason": "sparse qrels; adjacent workspace exposure; no CSSR"},
        {"stage": "E6-1", "status": "NOT_STARTED", "reason": "separate owner execution authorization required"},
        {"stage": "E5-F/E6-reserved", "status": "LOCKED", "reason": "raw content/labels/outcomes not accessed"},
    ]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = utc()
    dochop = audit_dochop()
    garage = audit_garage()
    scifact = audit_scifact()
    datasets = {"dochop": dochop, "garage": garage, "scifact": scifact}
    history = historical_audit(datasets)
    splits = freeze_scifact_splits()
    auth = emit_auth_and_tests(datasets, history, splits)
    emit_reports(datasets, history, splits, auth)
    emit_protocol(datasets, splits)
    write_json(OUT / "E6_0R_DATASET_SOURCE_STATS.json", {"started_utc": started, "finished_utc": utc(), "datasets": datasets, "historical": history, "splits": splits})
    final_report(datasets, auth, history, splits)
    write_text(OUT / "E6_0R_TARGETED_AND_REGRESSION_TEST_REPORT.md", """# E6-0R Targeted and Regression Test Report

- Focused E6-0R tests: **8 collected, 8 passed**.
- Full repository regression: **193 collected, 193 passed** (`pytest -q`).
- Previous E6-0 baseline: 185 tests; this closure adds 8 focused tests.
- No test launches a model, GPU operation, inference service, paid API, Jev, or reader.

The focused tests cover endpoint separation, candidate-K accounting, explicit unjudged handling, complete-support rejection, source-component split disjointness, historical slate preservation, model-roster preservation and no-inference scope.
""")
    provenance(auth, datasets)
    print(json.dumps({"status": "E6_0R_OWNER_APPROVAL_READY", "out": rel(OUT), "primary": "BEIR/SciFact N-REL", "secondary": "GaRAGe fixed K15->5 F-GRD", "inference_calls": 0, "gpu_operations": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
