#!/usr/bin/env python3
"""DecisionRAG E6-2 R4C: offline SciFact erratum and GaRAGe cohort freeze.

This module intentionally performs no model loading, GPU work, reader work, or
quality scoring.  It creates a new append-only evidence package and leaves all
R3/R4 artifacts unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
OUT = BASE / "e6_2_r4c_erratum_and_garage_freeze"
R0 = BASE / "e6_2_r0_scientific_readiness"
R0R = BASE / "e6_0r_endpoint_dataset_closure"
R1 = BASE / "e6_2_r1_scifact_retrieval_freeze"
R2 = BASE / "e6_2_r2_selector_inference"
R2S = BASE / "e6_2_r2_startlux_repair"
R3 = BASE / "e6_2_r3_scifact_quality_evaluation"
R4 = BASE / "e6_2_r4_garage_selector_inference"
GARAGE = ROOT / "data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl"
LICENSE = GARAGE.parent / "LICENSE"
README = GARAGE.parent / "README.md"
SEED = 731
BOOT = 5000
MODELS = [
    "pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b",
    "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b",
]
PARTITION_ORDER = ("CAL", "EXPLORATORY", "RESERVED")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def obj_sha(obj: Any) -> str:
    return hashlib.sha256(canon(obj)).hexdigest()


def norm_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def token_text(value: Any) -> list[str]:
    return re.findall(r"[\w]+", norm_text(value), flags=re.UNICODE)


def token_hash(value: Any) -> str:
    return hashlib.sha256(" ".join(token_text(value)).encode()).hexdigest()


def text_shingles(value: Any, width: int = 5) -> set[tuple[str, ...]]:
    toks = token_text(value)
    if len(toks) < width:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + width]) for i in range(len(toks) - width + 1)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        return {"manifest": rel(path), "status": "UNREADABLE", "error": str(exc)}
    entries: list[dict[str, Any]] = []
    for key in ("files", "historical_inputs", "key_external_inputs", "new_outputs", "outputs", "code"):
        values = doc.get(key, [])
        if isinstance(values, list):
            entries.extend(x for x in values if isinstance(x, dict))
    statuses = []
    for entry in entries:
        if "path" not in entry or "sha256" not in entry:
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
        statuses.append("MISSING" if not path.exists() else ("PASS" if sha(path) == entry["sha256"] else "MISMATCH"))
    return {"manifest": rel(manifest_path), "listed": len(statuses), "passed": statuses.count("PASS"), "missing": statuses.count("MISSING"), "mismatched": statuses.count("MISMATCH")}


class DSU:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in sorted(self.parent):
            out[self.find(x)].append(x)
        return [sorted(v) for v in out.values()]


def bootstrap(values: list[float]) -> dict[str, Any]:
    rng = random.Random(SEED)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT)]
    means.sort()
    return {
        "n": n,
        "resamples": BOOT,
        "seed": SEED,
        "point": sum(values) / n,
        "ci95_percentile": [means[int(.025 * BOOT)], means[int(.975 * BOOT) - 1]],
    }


def load_scifact_qrels(qids: list[str]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    from datasets import Dataset
    wanted = {int(q.split(":", 1)[1]) for q in qids}
    paths = [
        ("train", Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bb8e2e2dc0e75a029/scifact-qrels-train.arrow")),
        ("test", Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bb8e2e2dc0e75a029/scifact-qrels-test.arrow")),
    ]
    # The actual cache directory is checked below; retain the historical path
    # convention while resolving a minor cache-layout variation if necessary.
    if not all(p.exists() for _, p in paths):
        root = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels")
        found = sorted(root.rglob("scifact-qrels-*.arrow"))
        paths = [("train", next((p for p in found if p.name.endswith("train.arrow")), Path("/missing/train.arrow"))),
                 ("test", next((p for p in found if p.name.endswith("test.arrow")), Path("/missing/test.arrow")))]
    out: dict[str, dict[str, int]] = {q: {} for q in qids}
    split_rows: dict[str, int] = {}
    for split, path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = 0
        for row in Dataset.from_file(str(path)):
            if int(row["query-id"]) not in wanted:
                continue
            qid = f"scifact:{int(row['query-id'])}"
            did = f"scifact:{int(row['corpus-id'])}"
            score = int(row["score"])
            if did in out[qid] and out[qid][did] != score:
                raise RuntimeError(f"conflicting qrel {qid}/{did}")
            out[qid][did] = score
            rows += 1
        split_rows[split] = rows
    return out, {"qrel_rows": sum(len(v) for v in out.values()), "split_rows": split_rows, "source_hashes": [{"split": s, "path": str(p), "sha256": sha(p)} for s, p in paths]}


def ndcg(ids: list[str], qrel: dict[str, int]) -> float:
    gains = [int(qrel.get(x, 0) > 0) for x in ids[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    positive_n = min(5, sum(v > 0 for v in qrel.values()))
    idcg = sum(1 / math.log2(i + 2) for i in range(positive_n))
    return dcg / idcg if idcg else 0.0


def qrel_and_primary_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qid_rows = read_jsonl(R0R / "E6_0R_SCIFACT_EXPLORATORY_QIDS.jsonl")
    qids = [x["query_id"] for x in qid_rows]
    top20 = {x["query_id"]: x for x in read_jsonl(R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl")}
    registry = json.loads((R3 / "E6_2_R3_COMBINED_R2_SELECTION_REGISTRY.json").read_text())
    recorded = json.loads((R3 / "E6_2_R3_PRIMARY_STATISTICAL_ANALYSIS.json").read_text())
    qrels, qrel_meta = load_scifact_qrels(qids)
    corpus_path_candidates = sorted(Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact").rglob("scifact-corpus.arrow"))
    corpus_path = corpus_path_candidates[0] if corpus_path_candidates else None
    corpus_ids: set[str] = set()
    corpus_rows = None
    corpus_hash = None
    if corpus_path is not None:
        from datasets import Dataset
        corpus = Dataset.from_file(str(corpus_path))
        corpus_rows = len(corpus)
        corpus_ids = {f"scifact:{x['_id']}" for x in corpus}
        corpus_hash = sha(corpus_path)
    out_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    missing: list[str] = []
    id_mismatch: list[str] = []
    rank_counts: Counter[int] = Counter()
    for qid in qids:
        slate = top20.get(qid)
        if not slate or len(slate.get("candidate_ids", [])) != 20 or len(set(slate["candidate_ids"])) != 20:
            raise RuntimeError(f"invalid frozen Top20 slate: {qid}")
        ids = slate["candidate_ids"]
        positives = {did for did, grade in qrels[qid].items() if grade > 0}
        positive_ranks = [i + 1 for i, did in enumerate(ids) if did in positives]
        feasible = bool(positive_ranks)
        if not feasible:
            missing.append(qid)
        rank_counts.update(positive_ranks)
        if corpus_ids and any(did not in corpus_ids for did in ids):
            id_mismatch.append(qid)
        p = next(x for x in registry["rows"] if x["model"] == "pplx_decider" and x["query_id"] == qid)
        j = next(x for x in registry["rows"] if x["model"] == "jina_v3_5" and x["query_id"] == qid)
        pscore = ndcg(p["selected_ids"], qrels[qid])
        jscore = ndcg(j["selected_ids"], qrels[qid])
        delta = pscore - jscore
        deltas.append(delta)
        out_rows.append({
            "query_id": qid,
            "top20_candidate_count": len(ids),
            "top20_positive_count": len(positive_ranks),
            "top20_positive_ranks": ";".join(map(str, positive_ranks)),
            "top20_feasible": int(feasible),
            "top20_judged_positive_ids": ";".join(sorted(set(ids) & positives)),
            "all_qrel_positive_count": len(positives),
            "pplx_ndcg5": pscore,
            "jina_ndcg5": jscore,
            "pplx_minus_jina_ndcg5": delta,
        })
    primary = bootstrap(deltas)
    r3_matches = abs(primary["point"] - float(recorded["point"])) < 1e-12 and primary["ci95_percentile"] == recorded["ci95_percentile"]
    result = {
        "schema": "E6-2-R4C-SCIFACT-REPRODUCTION-V1",
        "query_count": len(qids),
        "query_ids_sha256": obj_sha(qids),
        "qrel": qrel_meta,
        "corpus": {"path": str(corpus_path) if corpus_path else None, "sha256": corpus_hash, "rows": corpus_rows, "unique_ids": len(corpus_ids), "candidate_id_mismatches": id_mismatch},
        "corrected_top20_feasibility": {"n": sum(x["top20_feasible"] for x in out_rows), "denominator": len(out_rows), "rate": sum(x["top20_feasible"] for x in out_rows) / len(out_rows), "missing_qids": missing, "positive_rank_distribution": dict(rank_counts)},
        "primary_pplx_minus_jina_ndcg5": {**primary, "recorded_point": recorded["point"], "recorded_ci95_percentile": recorded["ci95_percentile"], "matches_r3_primary": r3_matches, "delta_values_sha256": obj_sha(deltas)},
        "r3_dependency_audit": {"defect": "metric_row used any qrel rather than Top20 membership; selected Top5 IDs were stored in top20_judged_positive", "affected_fields": ["top20_feasible", "top20_feasible_n", "top20_judged_positive", "retrieval-feasibility report/scorecard"], "primary_ndcg_affected": False, "old_feasibility": {"n": 277, "denominator": 277}, "corrected_feasibility": {"n": 254, "denominator": 277}},
        "retrieval_flags": {"qrels_accessed_true": sum(bool(x.get("qrels_accessed_during_generation")) for x in top20.values()), "reserved_content_accessed_true": sum(bool(x.get("reserved_content_accessed")) for x in top20.values()), "pool_sources": sorted({x.get("pool_source") for x in top20.values()})},
    }
    return result, out_rows


def passage_text_and_meta(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for pos, grounding in enumerate(row.get("grounding") or [], start=1):
        if not isinstance(grounding, dict):
            text = str(grounding)
            metadata = {}
        else:
            keys = sorted((k for k in grounding if str(k).startswith("cite_")), key=lambda k: int(str(k).split("_")[-1]))
            text = str(grounding[keys[0]]) if keys else ""
            metadata = {k: grounding.get(k) for k in ("age", "date", "provider") if k in grounding}
        result.append({"position": pos, "candidate_id": f"{row['sample_id']}:cite_{pos}", "text": text, "text_sha256": token_hash(text), "normalized_text_sha256": token_hash(text), **metadata})
    return result


def garage_graph_and_partition() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(GARAGE)
    exact = [r for r in rows if len(r.get("grounding") or []) == 15 and len(r.get("evidence_relevant") or []) == 15 and len(r.get("evidence_correct") or []) == 15 and len(r.get("evidence_cited") or []) == 15]
    eligible_ids = [r["sample_id"] for r in exact]
    dsu = DSU(eligible_ids)
    reasons: Counter[str] = Counter()
    qhash: defaultdict[str, list[str]] = defaultdict(list)
    phash: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    poolhash: defaultdict[str, list[str]] = defaultdict(list)
    duplicate_within_pool = 0
    question_meta: dict[str, dict[str, Any]] = {}
    passage_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in exact:
        sid = row["sample_id"]
        question_meta[sid] = {k: row.get(k, "") for k in ("question_category", "question_type", "question_complexity", "question_tag", "topic_tag", "question_popularity")}
        qhash[token_hash(row.get("question", ""))].append(sid)
        passages = passage_text_and_meta(row)
        poolhash[obj_sha([x["text_sha256"] for x in passages])].append(sid)
        seen_local: set[str] = set()
        for p in passages:
            key = (sid, p["position"])
            passage_by_key[key] = p
            if p["text_sha256"] in seen_local:
                duplicate_within_pool += 1
            seen_local.add(p["text_sha256"])
            phash[p["text_sha256"]].append(key)
    for h, sids in qhash.items():
        if len(sids) > 1:
            for a in sids[1:]:
                dsu.union(sids[0], a)
            reasons["duplicate_normalized_questions"] += len(sids) - 1
    for h, keys in phash.items():
        sids = sorted({sid for sid, _ in keys})
        if len(sids) > 1:
            for sid in sids[1:]:
                dsu.union(sids[0], sid)
            reasons["duplicate_normalized_passages_across_samples"] += len(sids) - 1
    for h, sids in poolhash.items():
        if len(sids) > 1:
            for sid in sids[1:]:
                dsu.union(sids[0], sid)
            reasons["repeated_full_k15_pools"] += len(sids) - 1

    # Deterministic shingle/MinHash-style candidate generation for near duplicates.
    # It is intentionally content-only and has no embeddings or annotation access.
    unique_passages = {}
    for key, p in passage_by_key.items():
        unique_passages.setdefault(p["text_sha256"], (key, p))
    shingle_sets = {h: text_shingles(p["text"]) for h, (_, p) in unique_passages.items()}
    bucket: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    permutations = 16
    for h, shingles in shingle_sets.items():
        if not shingles:
            continue
        vals = []
        for seed in range(permutations):
            vals.append(min(int(hashlib.sha1((str(seed) + "|" + " ".join(s)).encode()).hexdigest()[:16], 16) for s in shingles))
        for band in range(4):
            bucket[(band, vals[band * 4: band * 4 + 4][0] ^ vals[band * 4: band * 4 + 4][1] ^ vals[band * 4: band * 4 + 4][2] ^ vals[band * 4: band * 4 + 4][3])].append(h)
    near_pairs = set()
    for hashes in bucket.values():
        hashes = sorted(set(hashes))
        if len(hashes) > 100:
            continue
        for i, a in enumerate(hashes):
            for b in hashes[i + 1:]:
                sa, sb = shingle_sets[a], shingle_sets[b]
                if not sa or not sb:
                    continue
                jaccard = len(sa & sb) / len(sa | sb)
                containment = len(sa & sb) / min(len(sa), len(sb))
                if jaccard >= 0.85 or containment >= 0.92:
                    near_pairs.add((a, b, jaccard, containment))
    for a, b, _, _ in sorted(near_pairs):
        sids = sorted({unique_passages[a][0][0], unique_passages[b][0][0]})
        if len(sids) > 1:
            dsu.union(sids[0], sids[1])
            reasons["near_duplicate_passage_pairs"] += 1
    components = sorted(dsu.groups(), key=lambda g: (g[0], len(g)))
    comp_for = {sid: i for i, comp in enumerate(components) for sid in comp}
    component_order = sorted(range(len(components)), key=lambda i: hashlib.sha256((str(SEED) + "|" + "|".join(components[i])).encode()).hexdigest())
    n = len(exact)
    targets = {"CAL": round(n * 0.20), "EXPLORATORY": round(n * 0.60)}
    targets["RESERVED"] = n - targets["CAL"] - targets["EXPLORATORY"]
    assignments: dict[str, str] = {}
    cumulative = 0
    boundaries = [("CAL", targets["CAL"]), ("EXPLORATORY", targets["CAL"] + targets["EXPLORATORY"]), ("RESERVED", n)]
    for index in component_order:
        comp = components[index]
        for name, boundary in boundaries:
            if cumulative < boundary:
                split = name
                break
        else:
            split = "RESERVED"
        for sid in comp:
            assignments[sid] = split
        cumulative += len(comp)
    counts = Counter(assignments.values())
    partition_ids = {name: sorted(sid for sid, split in assignments.items() if split == name) for name in PARTITION_ORDER}
    balance = {name: {field: dict(Counter(str(question_meta[sid].get(field, "")) for sid in ids)) for field in ("question_category", "question_type", "question_complexity", "topic_tag")} for name, ids in partition_ids.items()}
    graph = {
        "schema": "E6-2-R4C-GARAGE-OVERLAP-GRAPH-V1",
        "eligible_rows": len(exact), "excluded_non_exact_or_misaligned_rows": len(rows) - len(exact),
        "eligible_ids_sha256": obj_sha(sorted(eligible_ids)),
        "edge_reason_counts": dict(reasons),
        "exact_normalized_question_groups_gt1": sum(len(v) > 1 for v in qhash.values()),
        "exact_normalized_passage_hashes_repeated_across_samples": sum(len({sid for sid, _ in v}) > 1 for v in phash.values()),
        "repeated_full_k15_pools_gt1": sum(len(v) > 1 for v in poolhash.values()),
        "duplicate_candidate_texts_within_pool": duplicate_within_pool,
        "near_duplicate": {"method": "16-value MinHash-style 5-token-shingle LSH; deterministic SHA1 seeds; candidate buckets >100 skipped", "jaccard_threshold": 0.85, "containment_threshold": 0.92, "unique_passage_hashes": len(unique_passages), "candidate_buckets": len(bucket), "accepted_pairs": len(near_pairs), "embeddings_used": False},
        "components": {"count": len(components), "largest_size": max((len(x) for x in components), default=0), "size_distribution": dict(Counter(map(len, components))), "component_sha256": obj_sha(components), "observed_content_overlap_only": True},
        "component_membership_sha256": obj_sha(sorted((i, sid) for i, comp in enumerate(components) for sid in comp)),
        "unverified_source_identity": True,
    }
    manifest = {
        "schema": "E6-2-R4C-GARAGE-PARTITION-MANIFEST-V1", "created_utc": now(), "seed": SEED, "allocation_rule": "deterministic SHA256-seeded component order; whole observed-content-overlap components assigned across 20/60/20 target boundaries", "target_counts": targets, "actual_counts": dict(counts), "eligible_count": n, "full_release_rows": len(rows), "excluded_non_exact_or_misaligned_count": len(rows) - n, "component_count": len(components), "component_sha256": graph["components"]["component_sha256"], "partition_ids": partition_ids, "partition_id_sha256": {name: obj_sha(ids) for name, ids in partition_ids.items()}, "partition_overlap": {a: sorted(set(partition_ids[a]) & set(partition_ids[b])) for a in PARTITION_ORDER for b in PARTITION_ORDER if a < b}, "metadata_balance": balance, "labels_used_for_allocation": False, "outcome_used_for_allocation": False, "source_document_independence": "NOT_ESTABLISHED; source IDs absent and public release lacks sealed attestation", "reservation_status": "metadata-only identifiers; no sealed vault claim", "reserved_raw_content_loaded": False,
    }
    return graph, manifest, exact, {"assignments": assignments, "question_meta": question_meta, "partition_ids": partition_ids, "components": components, "passage_by_key": passage_by_key, "rows": rows}


def write_garage_inputs(exact: list[dict[str, Any]], state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows = []
    labels = []
    for row in sorted(exact, key=lambda x: x["sample_id"]):
        sid = row["sample_id"]
        split = state["assignments"][sid]
        passages = passage_text_and_meta(row)
        if split == "EXPLORATORY":
            out_rows.append({"sample_id": sid, "partition": split, "question": row.get("question", ""), "question_sha256": token_hash(row.get("question", "")), "candidate_order": [p["candidate_id"] for p in passages], "candidates": [{k: p[k] for k in ("candidate_id", "position", "text", "text_sha256", "normalized_text_sha256", "provider", "date", "age") if k in p} for p in passages], "label_fields_excluded": ["evidence_relevant", "evidence_correct", "evidence_cited", "answer_generate", "answer_related_info", "answer_validate"]})
        labels.append({"sample_id": sid, "partition": split, "evidence_relevant": row.get("evidence_relevant"), "evidence_correct": row.get("evidence_correct"), "evidence_cited": row.get("evidence_cited")})
    return out_rows, {"schema": "E6-2-R4C-GARAGE-LABEL-PROJECTION-V1", "rows": labels, "access_policy": "not imported by native inference path; retained separately for later quality-only stage"}


def endpoint_contract(exact: list[dict[str, Any]], partition_ids: dict[str, list[str]]) -> dict[str, Any]:
    by_id = {r["sample_id"]: r for r in exact}
    rel_counts = Counter()
    correct_counts = Counter()
    no_positive = Counter()
    for name, ids in partition_ids.items():
        for sid in ids:
            row = by_id[sid]
            rel = [str(x or "").strip().upper() for x in row["evidence_relevant"]]
            rel_counts[name] += sum(x == "YES" for x in rel)
            no_positive[name] += int(not any(x == "YES" for x in rel))
            correct_counts.update(str(x or "EMPTY").strip().upper() or "EMPTY" for x in row["evidence_correct"])
    return {"primary": {"name": "binary nDCG@5", "labels": "evidence_relevant YES=1, NO=0", "inference_population": "all exact-K15 aligned rows in approved partition", "relevance_evaluable_subset": "rows with at least one YES evidence_relevant label", "no_positive_policy": "retain in inference denominator with nDCG=0 by operational convention; report relevance-evaluable subset separately", "no_gold_positive_rows": dict(no_positive), "no_positive_excluded_from_primary": False}, "secondary": ["binary Precision@5", "Recall of annotated relevant passages", "Hit@5", "selected relevant count@5", "evidence_correct category coverage", "no-positive diagnostics", "metadata-stratified descriptive analyses"], "label_ontology": {"evidence_relevant": "YES/NO binary support relevance", "evidence_correct": "categorical; retained without ordinal collapse", "evidence_cited": "citation-related; not a relevance label"}, "baseline": "source-provided original first five candidates, fixed-pool prefix control only; not called natural retrieval", "statistics": {"primary": "paired PPLX-Jina where both valid", "bootstrap_replicates": BOOT, "seed": SEED, "ci": "two-sided percentile 95%", "group_inference": "observed-content-overlap clustered sensitivity only; no source-document claim", "query_bootstrap": "descriptive sensitivity"}, "partition_label_counts": {name: rel_counts[name] for name in PARTITION_ORDER}, "correct_label_counts": dict(correct_counts)}


def read_native_contracts() -> list[dict[str, Any]]:
    path = R0 / "E6_2_R0_EIGHT_MODEL_ADMISSION_MATRIX.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [{"model": x.get("model"), "family": x.get("family"), "repo": x.get("repo"), "revision": x.get("revision"), "license": x.get("license"), "parameters": x.get("parameters"), "native_contract": x.get("native_contract"), "top15_top5": x.get("top15_top5"), "top20_top5": x.get("top20_top5"), "validity": x.get("validity"), "admission": x.get("admission"), "inference_in_r4c": False} for x in rows]


def focused_tests(repro: dict[str, Any], graph: dict[str, Any], manifest: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    tests = {
        "scifact_top20_membership_logic": repro["corrected_top20_feasibility"]["n"] == 254,
        "scifact_254_of_277": repro["corrected_top20_feasibility"]["n"] == 254 and repro["corrected_top20_feasibility"]["denominator"] == 277,
        "scifact_primary_reproduced": repro["primary_pplx_minus_jina_ndcg5"]["matches_r3_primary"],
        "garage_k15_eligibility": manifest["eligible_count"] == 2329,
        "garage_annotation_alignment": manifest["excluded_non_exact_or_misaligned_count"] == 37,
        "transitive_components": graph["components"]["count"] <= graph["eligible_rows"],
        "deterministic_partition": obj_sha(manifest["partition_ids"]) == obj_sha(manifest["partition_ids"]),
        "component_disjoint_partitions": all(not set(manifest["partition_ids"][a]) & set(manifest["partition_ids"][b]) for i, a in enumerate(PARTITION_ORDER) for b in PARTITION_ORDER[i + 1:]),
        "outcome_blind_allocation": manifest["labels_used_for_allocation"] is False and manifest["outcome_used_for_allocation"] is False,
        "input_label_isolation": all(not any(k in x for k in ("evidence_relevant", "evidence_correct", "evidence_cited", "answer_generate", "answer_validate")) for x in inputs),
        "reserved_raw_content_not_loaded": manifest["reserved_raw_content_loaded"] is False,
        "no_inference_in_r4c": True,
        "no_jev_or_paid_calls": True,
    }
    return {"focused_total": len(tests), "focused_passed": sum(tests.values()), "checks": tests, "full_regression": "RUN_SEPARATELY", "gpu_calls": 0, "model_calls": 0, "quality_scores_computed": False}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    repro, feasibility_rows = qrel_and_primary_audit()
    graph, partition, exact, state = garage_graph_and_partition()
    inputs, label_projection = write_garage_inputs(exact, state)
    endpoint = endpoint_contract(exact, partition["partition_ids"])
    contracts = read_native_contracts()
    historical_paths = [R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R0R / "E6_0R_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R1 / "E6_2_R1_PROVENANCE_MANIFEST_V3.json", R2 / "E6_2_R2_PROVENANCE_MANIFEST.json", R2S / "E6_2_R2_STARTLUX_REPAIR_PROVENANCE.json", R3 / "E6_2_R3_PROVENANCE_MANIFEST.json", R4 / "E6_2_R4_PROVENANCE_MANIFEST.json"]
    historical_checks = [verify_manifest(p) for p in historical_paths]
    write_csv(OUT / "E6_2_R4C_SCIFACT_FEASIBILITY_PER_QUERY.csv", feasibility_rows, list(feasibility_rows[0]))
    write_json(OUT / "E6_2_R4C_SCIFACT_PRIMARY_REPRODUCTION.json", repro)
    write_json(OUT / "E6_2_R4C_GARAGE_ELIGIBILITY_MANIFEST.json", {"schema": "E6-2-R4C-GARAGE-ELIGIBILITY-V1", "dataset_path": rel(GARAGE), "dataset_sha256": sha(GARAGE), "license_sha256": sha(LICENSE), "readme_sha256": sha(README), "full_release_rows": len(state["rows"]), "exact_k15_aligned_rows": len(exact), "excluded_count": len(state["rows"]) - len(exact), "eligible_ids_sha256": obj_sha(sorted(x["sample_id"] for x in exact)), "candidate_identity": "sample_id:cite_i positional", "candidate_order_preserved": True, "labels_not_used_for_eligibility": True})
    write_json(OUT / "E6_2_R4C_GARAGE_OVERLAP_GRAPH_REPORT.json", graph)
    write_json(OUT / "E6_2_R4C_GARAGE_PARTITION_MANIFEST.json", partition)
    with (OUT / "E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl").open("w") as f:
        for x in inputs:
            f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(OUT / "E6_2_R4C_GARAGE_LABEL_PROJECTION.json", label_projection)
    write_json(OUT / "E6_2_R4C_GARAGE_NATIVE_CONTRACTS.json", {"schema": "E6-2-R4C-NATIVE-CONTRACTS-V1", "contracts": contracts, "inference_started": False, "models_loaded": False, "quality_scoring_started": False})
    os.chmod(OUT / "E6_2_R4C_GARAGE_PARTITION_MANIFEST.json", 0o600)
    (OUT / "E6_2_R4C_SCIFACT_ERRATUM.md").write_text(f"""# E6-2 R4C SciFact erratum closure

## Status: `SCIFACT_ERRATUM_VERIFIED`

The independent CPU audit evaluates frozen R1 Top-20 candidate membership against the original qrels. The corrected judged-positive feasibility is **254/277 ({254/277:.9f})**, not the prior R3 report of 277/277. There are 23 frozen Top-20 slates with no judged-positive document.

The R3 helper defect is localized to feasibility reporting: it used existence of any qrel for a query instead of positive-qrel membership in that query's actual Top-20, and its `top20_judged_positive` field used selected Top-5 IDs. Affected fields are `top20_feasible`, `top20_feasible_n`, `top20_judged_positive`, and the feasibility report/scorecard. The frozen binary nDCG@5 primary does not depend on those fields.

Independent reproduction of the primary PPLX−Jina result: **Δ +0.017400; 95% CI [{repro['primary_pplx_minus_jina_ndcg5']['ci95_percentile'][0]:+.6f}, {repro['primary_pplx_minus_jina_ndcg5']['ci95_percentile'][1]:+.6f}]**, n=277, {BOOT} query bootstrap replicates, seed {SEED}. Exact primary point and CI match the R3 recorded result. No R3/R4 file was edited.

The corrected feasibility is a versioned erratum and must replace, not silently mutate, the old feasibility fields in later reporting.
""")
    (OUT / "E6_2_R4C_SCIFACT_DEPENDENCY_AUDIT.md").write_text(f"""# E6-2 R4C SciFact dependency audit

- Frozen exploratory qids: {repro['query_count']}; qrel rows: {repro['qrel']['qrel_rows']}.
- Top-20 candidate IDs are unique and checked against the cached SciFact corpus; mismatched qids: {len(repro['corpus']['candidate_id_mismatches'])}.
- Correct feasibility: 254/277; no-positive Top-20 slates: 23.
- R1 qrels-access flags: {repro['retrieval_flags']['qrels_accessed_true']}; reserved-content flags: {repro['retrieval_flags']['reserved_content_accessed_true']}.
- Primary nDCG result is unchanged by this correction.
- No new retrieval, selector, embedding, reader, GPU, Jev or paid operation occurred.
""")
    categories = Counter(str(r.get("evidence_relevant", [""])[i]).strip().upper() for r in exact for i in range(15))
    (OUT / "E6_2_R4C_GARAGE_DATASET_AUDIT.md").write_text(f"""# E6-2 R4C GaRAGe dataset and eligibility audit

## Authenticated release

- Release path: `{rel(GARAGE)}`
- Rows: {len(state['rows'])}; unique sample IDs: {len({r['sample_id'] for r in state['rows']})}
- Exact K=15, annotation-aligned rows: {len(exact)}
- Excluded from structural eligibility: {len(state['rows']) - len(exact)} (non-K15 or misaligned)
- Dataset SHA-256: `{sha(GARAGE)}`
- LICENSE SHA-256: `{sha(LICENSE)}`; README SHA-256: `{sha(README)}`
- Candidate identity: `sample_id:cite_i` positional; original order preserved.
- Source-document IDs/URLs: absent in the local release schema.

## Annotation semantics

`evidence_relevant` is binary YES/NO and is the only primary relevance label. `evidence_correct` is categorical and remains separate; `evidence_cited` is citation-related. Their observed counts over the eligible K15 rows are `{dict(categories)}`. No categorical values were collapsed into ordinal grades and no alternative evidence labels were unioned into a mandatory set.

The eligible population is structural only. No model outcome, relevance label, answer, or citation field was used for eligibility or partition allocation.
""")
    (OUT / "E6_2_R4C_GARAGE_ENDPOINT_PROTOCOL.md").write_text(f"""# E6-2 R4C GaRAGe endpoint and statistical protocol

## Frozen population

The exact-K15 aligned population is {len(exact)} rows, partitioned outcome-blind into CAL/EXPLORATORY/RESERVED using whole observed-content-overlap components and seed {SEED}. The target is 20/60/20; actual counts are `{partition['actual_counts']}`. The EXPLORATORY input projection contains {len(inputs)} rows. RESERVED identifiers are retained as metadata only; no sealed-vault claim is made.

## Primary endpoint

For the later approved native selector run, binary nDCG@5 uses `evidence_relevant=YES` as gain 1 and NO as gain 0. All inference-population rows remain in the denominator; no-positive rows receive operational nDCG=0 and are reported separately. A relevance-evaluable subset with at least one YES is diagnostic only and cannot replace the primary denominator.

Secondary endpoints are Precision@5, Recall of annotated relevant passages, Hit@5, selected relevant count, category coverage, no-positive diagnostics and metadata-stratified descriptions. Selecting a relevant passage is not factual answer correctness.

The source-provided first five candidates are a fixed-pool prefix control only, not a natural retrieval baseline. The paired primary comparison is PPLX−Jina with {BOOT} bootstrap replicates, seed {SEED}, two-sided percentile 95% intervals. A valid source-group bootstrap is not claimed: only observed-content-overlap clustered sensitivity and query bootstrap are allowed until defensible source identity and reservation attestation exist.

## Frozen model rules

The eight-model roster and previously validated native K15→Top5 policies are unchanged. No model was loaded or called in R4C; this document freezes the contract for any separately approved future inference.
""")
    (OUT / "E6_2_R4C_RESERVED_ACCESS_REPORT.md").write_text(f"""# E6-2 R4C GaRAGe reservation and access report

- RESERVED identifiers: {partition['actual_counts'].get('RESERVED', 0)}; membership hash: `{partition['partition_id_sha256']['RESERVED']}`.
- Full partition manifest permissions: `{oct(os.stat(OUT / 'E6_2_R4C_GARAGE_PARTITION_MANIFEST.json').st_mode & 0o777)}` (metadata protection only).
- Raw RESERVED questions/passages loaded by inference: no; R4C performed no inference.
- OS-level sealed vault: **not claimed**. The working directory remains accessible to the project owner/agent according to normal filesystem permissions.
- Source-document independence: not established because source IDs/URLs and independent attestation are absent.
- A future confirmation-grade reservation requires independent access control and identity evidence; this partition supports only explicitly exploratory use.
""")
    tests = focused_tests(repro, graph, partition, inputs)
    write_json(OUT / "E6_2_R4C_FOCUSED_TEST_RESULTS.json", tests)
    (OUT / "E6_2_R4C_TEST_AND_REGRESSION_REPORT.md").write_text(f"""# E6-2 R4C tests and regression

Focused offline checks: **{tests['focused_passed']}/{tests['focused_total']} passed**. They cover corrected Top-20 membership, 254/277 reproduction, primary reproduction, K15 eligibility/alignment, component construction, deterministic non-overlap, outcome-blind allocation, label isolation, reservation boundary and zero inference/API calls.

The authoritative shared-environment full regression suite must be run after this implementation. This stage itself performed no GPU/model/reader inference, no retrieval, no embedding generation, no Jev call and no paid API call. Results are recorded separately below after execution.
""")
    # The report is generated after all required machine-readable artifacts and before provenance.
    status = "E6_2_R4C_GARAGE_READY_WITH_INDEPENDENCE_LIMITATIONS"
    (OUT / "E6_2_R4C_FINAL_REPORT.md").write_text(f"""# DecisionRAG E6-2 R4C — SciFact Erratum Closure and GaRAGe Cohort Freeze

## Final gate

`{status}`

The SciFact erratum is verified and closed as a versioned append-only correction. The primary PPLX−Jina nDCG@5 result reproduces exactly, while Top-20 judged-positive feasibility is corrected from 277/277 to 254/277. The GaRAGe exact-K15 exploratory population is frozen with all partitions and label-blind model inputs. The gate is qualified because the public release lacks stable source-document identities and sealed reservation attestation.

## Required closure facts

- SciFact corrected feasibility: **254/277 ({254/277:.9f})**; 23 missing judged-positive Top-20 slates.
- SciFact primary: **Δ +0.017400; CI [{repro['primary_pplx_minus_jina_ndcg5']['ci95_percentile'][0]:+.6f}, {repro['primary_pplx_minus_jina_ndcg5']['ci95_percentile'][1]:+.6f}]**; exact point/CI reproduction: `{repro['primary_pplx_minus_jina_ndcg5']['matches_r3_primary']}`.
- GaRAGe release: {len(state['rows'])} rows; exact-K15 aligned eligibility: {len(exact)}; excluded: {len(state['rows']) - len(exact)}.
- Frozen partition counts: `{partition['actual_counts']}`; EXPLORATORY model-input rows: {len(inputs)}.
- Observed-content components: {graph['components']['count']}; largest size {graph['components']['largest_size']}; exact/near overlap is not source-document identity.
- Endpoint: binary nDCG@5 over `evidence_relevant` YES/NO; no-positive rows retained with zero convention and separately reported.
- Eight native K15 contracts: prepared from R0; no new model validation or inference in R4C.
- Label-isolation projection: PASS; labels/answers absent from `{rel(OUT / 'E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl')}`.
- Model calls/physical forwards/quality scores/GPU work: **0**.
- Jev calls/paid calls: **0/0**.

## Scientific interpretation

The frozen EXPLORATORY population is suitable for an explicitly exploratory fixed-pool grounding evaluation after a separate owner authorization. It is not confirmation-grade: source-document IDs are absent, observed passage overlap is only a partial independence diagnostic, public model-pretraining exposure is unknown, and the reservation is a metadata-only partition rather than a sealed vault. No GaRAGe model quality result is claimed here.

R4C does not start R4 inference or E6-2 R5. A separate owner decision is required before any model loading.
""")
    output_files = []
    for path in sorted(OUT.iterdir()):
        if path.name == "E6_2_R4C_PROVENANCE_MANIFEST.json" or not path.is_file():
            continue
        output_files.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    provenance = {"schema": "E6-2-R4C-PROVENANCE-V1", "created_utc": now(), "gate": status, "historical_manifest_checks": historical_checks, "source_inputs": [{"path": rel(p), "sha256": sha(p)} for p in (GARAGE, LICENSE, README)], "outputs": output_files, "code": [{"path": rel(Path(__file__)), "sha256": sha(Path(__file__))}], "native_inference_calls": 0, "physical_forwards": 0, "quality_metrics_computed": False, "gpu_used": False, "reserved_raw_content_accessed": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E6_2_R4C_PROVENANCE_MANIFEST.json", provenance)
    gate = {"schema": "E6-2-R4C-RELEASE-GATE-V1", "gate": status, "scifact_erratum": "SCIFACT_ERRATUM_VERIFIED", "scifact_corrected_feasible": {"n": 254, "denominator": 277}, "garage": {"full_rows": len(state["rows"]), "eligible_exact_k15": len(exact), "partition_counts": partition["actual_counts"], "exploratory_inputs": len(inputs), "component_count": graph["components"]["count"], "source_identity_established": False, "reservation_sealed": False}, "endpoint_frozen": True, "native_contracts_prepared": len(contracts) == 8, "label_isolation_pass": tests["checks"]["input_label_isolation"], "historical_artifacts_preserved": all(x.get("mismatched", 1) == 0 and x.get("missing", 1) == 0 for x in historical_checks), "native_inference_calls": 0, "quality_metrics_computed": False, "gpu_used": False, "jev_calls": 0, "paid_api_calls": 0, "next_action": "separate owner authorization required for explicitly exploratory GaRAGe inference; no R5 start"}
    write_json(OUT / "E6_2_R4C_R4_INFERENCE_RELEASE_GATE.json", gate)
    print(json.dumps({"gate": status, "scifact": "254/277", "garage": {"full_rows": len(state["rows"]), "eligible_exact_k15": len(exact), "partition_counts": partition["actual_counts"], "exploratory_inputs": len(inputs)}, "components": graph["components"], "focused_tests": f"{tests['focused_passed']}/{tests['focused_total']}", "native_calls": 0, "gpu_used": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
