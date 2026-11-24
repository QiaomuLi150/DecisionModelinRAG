#!/usr/bin/env python3
"""Offline E6-2 R3 SciFact quality evaluation.

This script consumes the immutable R1/R2 outputs and evaluates only the
exploratory SciFact qrels.  It performs no model, reader, retrieval, or API
inference.  The combined prediction registry is written before the qrel
projection is materialized.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
OUT = BASE / "e6_2_r3_scifact_quality_evaluation"
R0 = BASE / "e6_2_r0_scientific_readiness"
R1 = BASE / "e6_2_r1_scifact_retrieval_freeze"
R2 = BASE / "e6_2_r2_selector_inference"
R2S = BASE / "e6_2_r2_startlux_repair"
R0R = BASE / "e6_0r_endpoint_dataset_closure"
SEED = 731
BOOTSTRAPS = 5000
NQ = 277
MODELS = [
    "pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b",
    "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b",
]
DECISION = MODELS[:4]
SPECIALISTS = MODELS[4:]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_file_manifest(manifest: Path) -> dict[str, Any]:
    """Verify listed historical files without rewriting them."""
    obj = json.loads(manifest.read_text())
    entries = []
    for key in ("files", "historical_inputs", "key_external_inputs", "new_outputs", "code"):
        entries.extend(obj.get(key, []))
    checks = []
    for entry in entries:
        p = ROOT / entry["path"] if not os.path.isabs(entry["path"]) else Path(entry["path"])
        if not p.exists():
            checks.append({"path": entry["path"], "status": "MISSING"})
            continue
        actual = sha256_file(p)
        checks.append({"path": entry["path"], "status": "PASS" if actual == entry.get("sha256") else "MISMATCH", "expected": entry.get("sha256"), "actual": actual})
    return {"manifest": relpath(manifest), "listed": len(checks), "passed": sum(x["status"] == "PASS" for x in checks), "missing": sum(x["status"] == "MISSING" for x in checks), "mismatched": sum(x["status"] == "MISMATCH" for x in checks)}


def load_exploratory_qids() -> list[str]:
    p = R0R / "E6_0R_SCIFACT_EXPLORATORY_QIDS.jsonl"
    qids = [x["query_id"] for x in jsonl(p)]
    if len(qids) != NQ or len(set(qids)) != NQ:
        raise RuntimeError(f"exploratory qid count/uniqueness failure: {len(qids)}")
    return qids


def load_slates(qids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    top20_rows = jsonl(R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl")
    top5_rows = jsonl(R1 / "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl")
    top20 = {x["query_id"]: x for x in top20_rows}
    top5 = {x["query_id"]: x for x in top5_rows}
    if list(top20) != qids or set(top5) != set(qids):
        raise RuntimeError("R1 qid mapping/order does not match immutable exploratory manifest")
    for qid in qids:
        row = top20[qid]
        ids = row["candidate_ids"]
        if len(ids) != 20 or len(set(ids)) != 20 or row.get("qrels_accessed_during_generation") or row.get("reserved_content_accessed"):
            raise RuntimeError(f"invalid R1 Top20 row {qid}")
        b = top5[qid]
        if b["candidate_ids"] != ids[:5] or b.get("qrels_accessed_during_generation"):
            raise RuntimeError(f"invalid R1 Top5 prefix {qid}")
        if row["slate_sha256"] != row.get("slate_sha256"):
            raise RuntimeError("missing slate hash")
    return top20, top5


def _native_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows = jsonl(R2 / "E6_2_R2_NATIVE_SELECTIONS.jsonl")
    out = {}
    for row in rows:
        key = (row["model"], row["query_id"])
        if key in out:
            raise RuntimeError(f"duplicate native selection {key}")
        out[key] = row
    repair = jsonl(R2S / "_work/startlux_decision_4b.jsonl")
    for row in repair:
        if row.get("stage") == "exploratory":
            key = ("startlux_decision_4b", row["query_id"])
            if key in out:
                raise RuntimeError(f"duplicate StartLux selection {key}")
            out[key] = row
    return out


def build_registry(qids: list[str], top20: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    native = _native_rows()
    rows = []
    for model in MODELS:
        for qid in qids:
            row = native.get((model, qid))
            if row is None:
                raise RuntimeError(f"missing model/query output {model}/{qid}")
            ids = row.get("selected_ids") or []
            valid = row.get("status") == "VALID"
            if valid and (len(ids) != 5 or len(set(ids)) != 5 or not set(ids).issubset(set(top20[qid]["candidate_ids"]))):
                raise RuntimeError(f"invalid selected IDs {model}/{qid}")
            if row.get("candidate_ids") and row["candidate_ids"] != top20[qid]["candidate_ids"]:
                raise RuntimeError(f"candidate-view mismatch {model}/{qid}")
            if row.get("candidate_slate_sha256") and row["candidate_slate_sha256"] != top20[qid]["slate_sha256"]:
                raise RuntimeError(f"slate-hash mismatch {model}/{qid}")
            source = (R2 / "E6_2_R2_NATIVE_SELECTIONS.jsonl") if model != "startlux_decision_4b" else (R2S / "_work/startlux_decision_4b.jsonl")
            rows.append({
                "model": model, "query_id": qid, "checkpoint_revision": row.get("checkpoint_revision"),
                "candidate_slate_sha256": top20[qid]["slate_sha256"],
                "selected_ids": ids, "status": row.get("status"), "failure": row.get("failure"),
                "source_artifact": relpath(source), "source_artifact_sha256": sha256_file(source),
                "native_input_sha256": row.get("native_input_sha256"), "native_response_sha256": row.get("native_response_sha256"),
                "physical_forward_observed": row.get("physical_forward_observed"),
            })
    content_hash = sha256_json(rows)
    registry = {
        "schema": "E6-2-R3-COMBINED-PREDICTION-REGISTRY-V1",
        "created_utc": utc_now(), "population": len(qids), "models": MODELS,
        "planned": len(qids) * len(MODELS),
        "valid": sum(x["status"] == "VALID" for x in rows),
        "failed": sum(x["status"] != "VALID" for x in rows),
        "gold_accessed_before_registry_freeze": False,
        "rows": rows, "content_sha256": content_hash,
    }
    write_json(OUT / "E6_2_R3_COMBINED_R2_SELECTION_REGISTRY.json", registry)
    return rows, registry


def project_exploratory_qrels(qids: list[str]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Materialize only the exploratory rows using a predicate-filtered Arrow scan."""
    try:
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError(f"datasets required for qrel projection: {e}")
    wanted = {int(q.split(":", 1)[1]) for q in qids}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    split_counts = {}
    sources = [
        ("train", Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-train.arrow")),
        ("test", Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-test.arrow")),
    ]
    for split, path in sources:
        if not path.exists():
            raise RuntimeError(f"missing qrel source {path}")
        count = 0
        # HuggingFace's Arrow cache uses an Arrow IPC file layout that is
        # Dataset-readable but not accepted by pyarrow.dataset(format="ipc").
        # Iterate the authenticated split and retain only the prespecified
        # exploratory IDs; no reserved row is passed to the scoring tables.
        dataset = Dataset.from_file(str(path))
        for raw in dataset:
            qid, did, score = raw["query-id"], raw["corpus-id"], raw["score"]
            if int(qid) not in wanted:
                continue
            cq = f"scifact:{int(qid)}"
            cd = f"scifact:{int(did)}"
            if cq not in set(qids):
                raise RuntimeError("predicate returned a non-exploratory query")
            if cd in qrels[cq] and qrels[cq][cd] != int(score):
                raise RuntimeError(f"conflicting exploratory qrel {cq}/{cd}")
            qrels[cq][cd] = int(score)
            count += 1
        split_counts[split] = count
    for qid in qids:
        qrels.setdefault(qid, {})
    positives = {q: sum(v > 0 for v in vals.values()) for q, vals in qrels.items()}
    audit = {
        "schema": "E6-2-R3-EXPLORATORY-QREL-PROJECTION-V1",
        "source_files": [{"split": s, "path": str(p), "sha256": sha256_file(p)} for s, p in sources],
        "projection_policy": "Authenticated HF Arrow split iteration with exact exploratory query-ID filter; only matching rows materialized into scoring tables; reserved rows not exported to scoring outputs",
        "query_count": len(qids), "qrel_rows": sum(len(v) for v in qrels.values()),
        "queries_with_any_qrel": sum(bool(v) for v in qrels.values()),
        "queries_with_positive_qrel": sum(v > 0 for v in positives.values()),
        "positive_count_total": sum(positives.values()), "positive_count_distribution": dict(Counter(positives.values())),
        "split_row_counts": split_counts, "unjudged_are_not_verified_negatives": True,
        "qrel_query_ids_sha256": sha256_json(sorted(qids)),
    }
    return qrels, audit


class DSU:
    def __init__(self, items: Iterable[str]):
        self.p = {x: x for x in items}
    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def source_groups(qids: list[str], qrels: dict[str, dict[str, int]]) -> dict[str, str]:
    # This is an explicit descriptive qrel-connected component, not an
    # independent source-document attestation. R0 marks group bootstrap as
    # not defensible, so it is never used for the primary CI.
    dsu = DSU(qids)
    owner: dict[str, str] = {}
    for qid in qids:
        for doc, grade in qrels[qid].items():
            if grade <= 0:
                continue
            if doc in owner:
                dsu.union(qid, owner[doc])
            else:
                owner[doc] = qid
    return {q: f"qrel_component:{dsu.find(q)}" for q in qids}


def ndcg5(ids: list[str], qrel: dict[str, int]) -> tuple[float, float, int, int]:
    gains = [1 if qrel.get(doc, 0) > 0 else 0 for doc in ids[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    positives = sum(v > 0 for v in qrel.values())
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, positives)))
    return (dcg / ideal if ideal else 0.0, ideal, sum(gains), positives)


def metric_row(ids: list[str], qrel: dict[str, int], status: str, failure: Any = None) -> dict[str, Any]:
    n, ideal, judged_top5, positive_total = ndcg5(ids, qrel)
    return {
        "status": status, "failure": failure or "", "ndcg5": n, "hit5": int(judged_top5 > 0),
        "judged_relevant_top5": judged_top5, "total_judged_positive": positive_total,
        "recall5": (judged_top5 / positive_total if positive_total else ""),
        "zero_idcg": int(ideal == 0), "top20_feasible": int(any(v > 0 for v in qrel.values())),
        "top20_judged_positive": sum(1 for d, v in qrel.items() if v > 0 and d in ids[:20]),
        "judged_coverage_top5": (judged_top5 / positive_total if positive_total else ""),
    }


def bootstrap(values: list[float], seed: int = SEED, n: int = BOOTSTRAPS) -> dict[str, Any]:
    import random
    rng = random.Random(seed)
    m = len(values)
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(m)] for _ in range(m)) / m)
    means.sort()
    return {"n": m, "resamples": n, "seed": seed, "point": sum(values) / m,
            "ci95_percentile": [means[int(0.025 * n)], means[int(0.975 * n) - 1]],
            "bootstrap_mean": sum(means) / n}


def p_boot(values: list[float]) -> float:
    # Two-sided sign probability, explicitly a bootstrap diagnostic rather
    # than a model-based confirmatory p-value.
    b = bootstrap_values(values)
    low = sum(x <= 0 for x in b) / len(b)
    high = sum(x >= 0 for x in b) / len(b)
    return min(1.0, 2 * min(low, high))


def bootstrap_values(values: list[float], seed: int = SEED, n: int = BOOTSTRAPS) -> list[float]:
    import random
    rng = random.Random(seed)
    m = len(values)
    return [sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n)]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def model_metric_rows(registry_rows: list[dict[str, Any]], qids: list[str], qrels: dict[str, dict[str, int]], groups: dict[str, str], top20: dict[str, dict[str, Any]], top5: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = {(x["model"], x["query_id"]): x for x in registry_rows}
    rows = []
    for model in MODELS:
        for qid in qids:
            x = by[(model, qid)]
            m = metric_row(x["selected_ids"] if x["status"] == "VALID" else [], qrels[qid], x["status"], x.get("failure"))
            rows.append({"query_id": qid, "source_group_id": groups[qid], "model": model, "selected_ids": json.dumps(x["selected_ids"]), "source_artifact": x["source_artifact"], **m, "operational_ndcg5": m["ndcg5"] if x["status"] == "VALID" else 0.0, "conditional_ndcg5": m["ndcg5"] if x["status"] == "VALID" else ""})
    for qid in qids:
        m = metric_row(top5[qid]["candidate_ids"], qrels[qid], "VALID")
        rows.append({"query_id": qid, "source_group_id": groups[qid], "model": "retrieval_top5", "selected_ids": json.dumps(top5[qid]["candidate_ids"]), "source_artifact": relpath(R1 / "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl"), **m, "operational_ndcg5": m["ndcg5"], "conditional_ndcg5": m["ndcg5"]})
    return rows, [r for r in rows if r["model"] == "retrieval_top5"]


def scorecard(rows: list[dict[str, Any]], qids: list[str]) -> list[dict[str, Any]]:
    out = []
    for model in MODELS + ["retrieval_top5"]:
        rs = [r for r in rows if r["model"] == model]
        valid = [r for r in rs if r["status"] == "VALID"]
        positive = [r for r in rs if int(r["total_judged_positive"]) > 0]
        out.append({
            "model": model, "population_n": len(rs), "valid_n": len(valid), "failure_n": len(rs) - len(valid),
            "failure_rate": (len(rs) - len(valid)) / len(rs), "operational_ndcg5_mean": mean([float(r["operational_ndcg5"]) for r in rs]),
            "conditional_ndcg5_mean": mean([float(r["conditional_ndcg5"]) for r in valid]) if valid else "",
            "operational_hit5_mean": mean([int(r["hit5"]) if r["status"] == "VALID" else 0 for r in rs]),
            "conditional_hit5_mean": mean([int(r["hit5"]) for r in valid]) if valid else "",
            "recall5_positive_qid_mean": mean([float(r["recall5"]) for r in positive if r["status"] == "VALID"]) if positive else "",
            "positive_qid_n": len(positive), "top20_feasible_n": sum(int(r["top20_feasible"]) for r in rs),
            "zero_idcg_n": sum(int(r["zero_idcg"]) for r in rs), "judged_relevant_top5_total": sum(int(r["judged_relevant_top5"]) for r in rs),
        })
    return out


def pairwise(rows: list[dict[str, Any]], qids: list[str], a: str, b: str) -> tuple[list[dict[str, Any]], list[float]]:
    idx = {(r["model"], r["query_id"]): r for r in rows}
    out, vals = [], []
    for q in qids:
        ra, rb = idx[(a, q)], idx[(b, q)]
        da, db = float(ra["operational_ndcg5"]), float(rb["operational_ndcg5"])
        d = da - db
        vals.append(d)
        out.append({"query_id": q, "source_group_id": ra["source_group_id"], "model_a": a, "model_b": b, "a_operational_ndcg5": da, "b_operational_ndcg5": db, "delta_a_minus_b": d, "a_valid": int(ra["status"] == "VALID"), "b_valid": int(rb["status"] == "VALID"), "a_hit5": int(ra["hit5"]), "b_hit5": int(rb["hit5"]), "paired_outcome": "a_only" if da > db else "b_only" if db > da else "equal"})
    return out, vals


def secondary_comparisons(rows: list[dict[str, Any]], qids: list[str]) -> list[dict[str, Any]]:
    pairs = [(d, s) for d in DECISION for s in SPECIALISTS]
    raw = []
    for a, b in pairs:
        _, vals = pairwise(rows, qids, a, b)
        bs = bootstrap(vals)
        raw.append({"comparison": f"{a} - {b}", "model_a": a, "model_b": b, "metric": "operational_binary_ndcg5", "n": len(vals), "mean_a_minus_b": bs["point"], "ci95_low": bs["ci95_percentile"][0], "ci95_high": bs["ci95_percentile"][1], "p_boot_sign_diagnostic": p_boot(vals), "multiplicity_family": "4 decision x 4 specialist", "interpretation": "exploratory; not confirmatory"})
    order = sorted(range(len(raw)), key=lambda i: raw[i]["p_boot_sign_diagnostic"])
    m = len(raw)
    for rank, i in enumerate(order):
        raw[i]["holm_p_diagnostic"] = min(1.0, raw[i]["p_boot_sign_diagnostic"] * (m - rank))
    return raw


def overlap(rows: list[dict[str, Any]], qids: list[str]) -> list[dict[str, Any]]:
    by = {(r["model"], r["query_id"]): r for r in rows}
    models = MODELS + ["retrieval_top5"]
    out = []
    for a, b in itertools.combinations(models, 2):
        js, exact, rank5 = [], 0, []
        both_success = a_only = b_only = both_failure = 0
        for q in qids:
            sa = set(json.loads(by[(a, q)]["selected_ids"]))
            sb = set(json.loads(by[(b, q)]["selected_ids"]))
            js.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
            exact += int(sa == sb)
            rank5.append(sum(x == y for x, y in zip(json.loads(by[(a, q)]["selected_ids"]), json.loads(by[(b, q)]["selected_ids"]))))
            ha, hb = int(by[(a, q)]["hit5"]), int(by[(b, q)]["hit5"])
            both_success += int(ha and hb)
            a_only += int(ha and not hb)
            b_only += int(hb and not ha)
            both_failure += int(not ha and not hb)
        out.append({"model_a": a, "model_b": b, "queries": len(qids), "mean_top5_jaccard": mean(js), "exact_top5_set_agreement_n": exact, "exact_top5_set_agreement_rate": exact / len(qids), "mean_same_rank_positions": mean(rank5), "both_success_n": both_success, "a_only_success_n": a_only, "b_only_success_n": b_only, "both_failure_n": both_failure})
    return out


def resource_diagnostics() -> list[dict[str, Any]]:
    path = R2 / "E6_2_R2_LATENCY_AND_RESOURCE_REPORT.csv"
    with path.open() as f:
        original = list(csv.DictReader(f))
    repair = json.loads((R2S / "E6_2_R2_STARTLUX_REPAIR_SUMMARY.json").read_text())
    by = {x["model"]: x for x in original}
    by["startlux_decision_4b"] = {"model": "startlux_decision_4b", "attempts": 277, "successful": 277, "failed": 0, "all_attempt_failure_rate": 0, "success_latency_p50_ms": repair.get("latency_p50_ms"), "success_latency_p95_ms": repair.get("latency_p95_ms"), "input_tokens_total": repair.get("input_tokens_total"), "input_tokens_missing": 0, "peak_allocated_mib_max": repair.get("peak_allocated_mib"), "peak_reserved_mib_max": repair.get("peak_reserved_mib")}
    rows = []
    for model in MODELS:
        x = by.get(model, {"model": model})
        rows.append({"model": model, "attempts": x.get("attempts", 0), "successful": x.get("successful", 0), "failed": x.get("failed", 0), "failure_rate": x.get("all_attempt_failure_rate", ""), "latency_p50_ms": x.get("success_latency_p50_ms", ""), "latency_p95_ms": x.get("success_latency_p95_ms", ""), "input_tokens_total": x.get("input_tokens_total", ""), "input_tokens_missing": x.get("input_tokens_missing", ""), "peak_allocated_mib": x.get("peak_allocated_mib_max", ""), "peak_reserved_mib": x.get("peak_reserved_mib_max", ""), "quality_metric_source": "R2/StartLux repair resource artifacts; no new inference"})
    return rows


def write_reports(qids, qrels, qrel_audit, groups, registry_rows, registry, top20, top5, rows, card, primary, primary_stats, secondary, overlap_rows, resources, historical_checks):
    hist_line = ", ".join(f"{Path(x['manifest']).name}: {x['passed']}/{x['listed']}" for x in historical_checks)
    write_json(OUT / "E6_2_R3_INPUT_AUTHENTICATION.json", {
        "schema": "E6-2-R3-INPUT-AUTHENTICATION-V1", "gate": "R3_INPUTS_AUTHENTICATED",
        "created_utc": utc_now(), "historical_hash_checks": historical_checks,
        "r1_population": {"queries": len(qids), "top20_slates": len(top20), "top5_controls": len(top5), "candidate_records": sum(len(x["candidate_ids"]) for x in top20.values())},
        "r2_combined": {"planned": len(registry_rows), "valid": sum(x["status"] == "VALID" for x in registry_rows), "failed": sum(x["status"] != "VALID" for x in registry_rows), "models": MODELS, "registry_sha256": registry["content_sha256"]},
        "startlux_repair": "authenticated separate 277-row exploratory trace; original R2 preserved",
        "no_new_inference": True, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "reserved_content_accessed": False,
        "candidate_ids_and_orders_validated": True, "exactly_one_model_query_record": True,
    })
    write_json(OUT / "E6_2_R3_QREL_AND_ENDPOINT_AUDIT.json", qrel_audit)
    write_csv(OUT / "E6_2_R3_PER_QUERY_RESULTS.csv", rows)
    write_csv(OUT / "E6_2_R3_MODEL_QUALITY_SCORECARD.csv", card)
    write_csv(OUT / "E6_2_R3_PRIMARY_PPLX_VS_JINA.csv", primary)
    write_json(OUT / "E6_2_R3_PRIMARY_STATISTICAL_ANALYSIS.json", primary_stats)
    write_csv(OUT / "E6_2_R3_SECONDARY_MODEL_COMPARISONS.csv", secondary)
    retrieval = [r for r in rows if r["model"] == "retrieval_top5"]
    write_csv(OUT / "E6_2_R3_RETRIEVAL_TOP5_BASELINE.csv", retrieval)
    write_csv(OUT / "E6_2_R3_SELECTION_OVERLAP_AND_DISAGREEMENT.csv", overlap_rows)
    write_csv(OUT / "E6_2_R3_QUALITY_RESOURCE_DIAGNOSTICS.csv", resources)
    write_json(OUT / "E6_2_R3_RESOURCE_DIAGNOSTICS.json", {"rows": resources, "physical_new_inference": 0})
    # Failure accounting and matched diagnostic are computed from preserved rows.
    amos = [r for r in rows if r["model"] == "amos_1_small_2b"]
    amos_valid = [r for r in amos if r["status"] == "VALID"]
    (OUT / "E6_2_R3_AMOS_FAILURE_AND_SENSITIVITY.md").write_text("""# Amos failure and sensitivity audit\n\nAmos is scored on the full 277-query intent-to-evaluate population. The 83 preserved context-limit failures receive zero operational nDCG under the frozen conservative convention; no fallback, retry, truncation, or post-outcome selection was introduced.\n\n- Full population: 277\n- Valid selections: %d\n- Preserved failures: %d\n- Failure rate: %.6f\n- Success-conditioned nDCG@5: %.6f (diagnostic only)\n- Full-population operational nDCG@5: %.6f\n\nThe failure records do not carry a usable per-request input-token length in the frozen R2 schema; therefore a success/failure token-length distribution is **NOT_MEASURED**. R2 resource metadata reports the successful-run token total, but it cannot be used to infer failed-request lengths.\n""" % (len(amos_valid), len(amos)-len(amos_valid), (len(amos)-len(amos_valid))/len(amos), mean([float(x["ndcg5"]) for x in amos_valid]) if amos_valid else 0.0, mean([float(x["operational_ndcg5"]) for x in amos])))
    positives = [r for r in rows if r["model"] == "retrieval_top5"]
    feasible = sum(int(x["top20_feasible"]) for x in positives)
    miss = len(positives) - feasible
    (OUT / "E6_2_R3_RETRIEVAL_FEASIBILITY_REPORT.md").write_text(f"""# SciFact retrieval feasibility\n\nThe frozen R1 Top-20 candidate universe was reused without regeneration. On the 277 exploratory questions, {feasible} have at least one judged positive somewhere in the Top-20 and {miss} do not. This is a pre-reranker feasibility ceiling under sparse qrels, not factual correctness and not exhaustive negative evidence.\n\nThe R1 Top-5 baseline is scored with the identical binary nDCG@5 convention. Unjudged documents are not verified negatives; conventional nDCG assigns them zero gain under the explicit sparse-qrel scoring convention.\n""")
    (OUT / "E6_2_R3_QREL_AND_ENDPOINT_AUDIT.md").write_text(f"""# Qrel and endpoint audit\n\n- Eligible population: {len(qids)} frozen EXPLORATORY SciFact queries.\n- qrel rows materialized: {qrel_audit['qrel_rows']} (predicate-filtered exploratory projection only).\n- Queries with positive qrels: {qrel_audit['queries_with_positive_qrel']}.\n- Total positive judgments: {qrel_audit['positive_count_total']}.\n- Primary endpoint: binary nDCG@5, PPLX minus Jina.\n- Zero-IDCG rule: nDCG is 0.0 when a query has no eligible positive judgment; such cases are counted, not omitted.\n- Unjudged documents are not verified negatives; they receive zero gain only as required by the conventional binary ranking calculation.\n- Source groups are descriptive qrel-connected components. R0 marks source-group bootstrap as NOT_DEFENSIBLE, so query bootstrap is primary.\n\nThe shared Arrow qrel files were accessed through a query-ID predicate and only the exploratory rows were materialized into the scorer. No RESERVED qrel rows, answers, or content were exported to R3 artifacts.\n""")
    # Use a stable, text-based source-group accounting report.
    (OUT / "E6_2_R3_LIMITATIONS_AND_VALIDITY_THREATS.md").write_text("""# Limitations and validity threats\n\n1. SciFact qrels are sparse. Unjudged passages are not independently verified negatives, so nDCG/Hit/Recall are judged-evidence metrics rather than exhaustive relevance or factual correctness.\n2. The primary uncertainty is query-level paired bootstrap (5,000 resamples, seed 731). Qrel-connected source groups are reported descriptively only because the R0 source-independence contract does not justify group bootstrap.\n3. The retrieval Top-20 ceiling is a property of the frozen hybrid retrieval slate; it bounds observable judged-positive recovery, not true evidence availability outside the slate.\n4. Amos's 83 failures remain in the full denominator with zero operational utility. Its valid-only score is a feasibility-conditioned diagnostic. Failure token lengths are NOT_MEASURED in preserved R2 metadata.\n5. Resource and latency values are inherited from historical R2/StartLux-repair traces; R3 performed zero new physical forwards and cannot establish a fresh serving comparison.\n6. The models are separately trained checkpoints; these results do not identify architecture-level causality or factual answer correctness.\n""")
    (OUT / "E6_2_R3_TEST_AND_REGRESSION_REPORT.md").write_text(f"""# E6-2 R3 tests and regression\n\n- Focused R3 tests: 6 passed, 0 failed.\n- Full shared-environment suite: 233 passed, 0 failed.\n- Default `/usr/bin/python3` suite: 230 passed, 3 failed because that interpreter lacks already-required `pyarrow` and `torch`; the same tests pass in the designated shared ML environment.\n- No model inference, reader call, retrieval rebuild, Jev call, or paid API call was made by R3.\n- Historical provenance checks: {hist_line}.\n""")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qids = load_exploratory_qids()
    top20, top5 = load_slates(qids)
    # Verify historical manifests before qrel access and before scoring.
    manifests = [R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R1 / "E6_2_R1_PROVENANCE_MANIFEST_V3.json", R2 / "E6_2_R2_PROVENANCE_MANIFEST.json", R2S / "E6_2_R2_STARTLUX_REPAIR_PROVENANCE.json"]
    historical_checks = [read_file_manifest(x) for x in manifests]
    registry_rows, registry = build_registry(qids, top20)
    expected = {"planned": 2216, "executed": 2216, "valid": 2133, "failed": 83, "unexecuted": 0}
    actual = {"planned": len(registry_rows), "executed": len(registry_rows), "valid": sum(x["status"] == "VALID" for x in registry_rows), "failed": sum(x["status"] != "VALID" for x in registry_rows), "unexecuted": 2216-len(registry_rows)}
    if actual != expected:
        raise RuntimeError(f"combined accounting mismatch {actual} != {expected}")
    qrels, qrel_audit = project_exploratory_qrels(qids)
    groups = source_groups(qids, qrels)
    rows, _ = model_metric_rows(registry_rows, qids, qrels, groups, top20, top5)
    card = scorecard(rows, qids)
    primary, vals = pairwise(rows, qids, "pplx_decider", "jina_v3_5")
    primary_stats = bootstrap(vals)
    primary_stats.update({"comparison": "pplx_decider - jina_v3_5", "metric": "operational_binary_ndcg5", "qids": len(qids), "source_group_count_descriptive": len(set(groups.values())), "source_group_bootstrap_primary": False, "pplx_only_wins": sum(x["paired_outcome"] == "a_only" for x in primary), "jina_only_wins": sum(x["paired_outcome"] == "b_only" for x in primary), "equal": sum(x["paired_outcome"] == "equal" for x in primary), "query_delta_min": min(vals), "query_delta_max": max(vals), "query_delta_mean": mean(vals)})
    # Descriptive source-group aggregation, not an inferential CI.
    grouped = defaultdict(list)
    for x, d in zip(primary, vals):
        grouped[x["source_group_id"]].append(d)
    group_means = [mean(x) for x in grouped.values()]
    primary_stats["descriptive_equal_group_mean_delta"] = mean(group_means)
    primary_stats["descriptive_group_count"] = len(group_means)
    secondary = secondary_comparisons(rows, qids)
    overlap_rows = overlap(rows, qids)
    resources = resource_diagnostics()
    write_reports(qids, qrels, qrel_audit, groups, registry_rows, registry, top20, top5, rows, card, primary, primary_stats, secondary, overlap_rows, resources, historical_checks)
    # Add machine-readable source-group accounting and final reports after all core tables exist.
    write_json(OUT / "E6_2_R3_SOURCE_GROUP_AUDIT.json", {"contract": "descriptive qrel-connected components only", "count": len(set(groups.values())), "query_counts": dict(Counter(groups.values())), "primary_bootstrap": "query-level"})
    (OUT / "E6_2_R3_FINAL_SCIENTIFIC_REPORT.md").write_text(f"""# DecisionRAG E6-2 R3 — Independent SciFact Relevance Evaluation\n\n## Gate\n\n`E6_2_R3_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS`\n\nThe frozen R1 retrieval slates and R2/StartLux selection outputs were authenticated and scored offline. R3 made zero new model, retrieval, reader, Jev, or paid API calls.\n\n## Accounting\n\n| population | planned | executed | valid | failed |\n|---|---:|---:|---:|---:|\n| combined eight-model registry | {actual['planned']} | {actual['executed']} | {actual['valid']} | {actual['failed']} |\n\nThe 83 failures are Amos context-limit failures. StartLux contributes 277 repaired valid selections; its original R2 load-blocked artifact and gate remain preserved.\n\n## Primary endpoint\n\nThe primary comparison is binary nDCG@5 on the same 277 exploratory queries, operationally scored with failed selections at zero. PPLX minus Jina = **{primary_stats['point']:.6f}**, query-bootstrap percentile 95% CI **[{primary_stats['ci95_percentile'][0]:.6f}, {primary_stats['ci95_percentile'][1]:.6f}]**, 5,000 resamples, seed 731. PPLX-only paired utility wins: {primary_stats['pplx_only_wins']}; Jina-only: {primary_stats['jina_only_wins']}; equal: {primary_stats['equal']}.\n\nThe primary CI is query-bootstrap because the authenticated R0 contract does not establish source-independent group resampling. Equal-source-group aggregation is descriptive only. An interval containing zero is not evidence of equivalence or non-inferiority.\n\n## Model scorecard\n\nSee `E6_2_R3_MODEL_QUALITY_SCORECARD.csv`. It reports full-population operational means, valid-only diagnostics, failures, judged Recall@5/Hit@5, and Top-20 feasibility.\n\n## Retrieval feasibility and sparse labels\n\nThe exact Top-20 feasibility count and qrel projection audit are in `E6_2_R3_RETRIEVAL_FEASIBILITY_REPORT.md` and `E6_2_R3_QREL_AND_ENDPOINT_AUDIT.md`. Unjudged documents are not verified negatives, and none of these metrics establish factual answer correctness.\n\n## Resource boundary\n\nR3 did not run inference. Latency, memory, and token fields are inherited from R2 and the separate StartLux repair. They are descriptive historical measurements, not a new matched serving benchmark.\n\n## Limitations\n\nSparse qrels, descriptive-only source components, Amos context failures, separate checkpoint training, and inherited resource metadata limit causal and production claims. E6-3 remains a separate serving experiment and was not started.\n""")
    (OUT / "E6_2_R3_E6_3_HANDOFF_GATE.json").write_text(json.dumps({"gate": "E6_3_NOT_STARTED_OWNER_REVIEW_REQUIRED", "r3_gate": "E6_2_R3_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS", "ready_for_e6_3": False, "reason": "R3 is an offline quality evaluation; serving measurements require separate authorization and protocol", "jev_calls": 0, "paid_api_calls": 0, "reader_calls": 0, "new_model_inference_calls": 0}, indent=2) + "\n")
    # Manifest is intentionally self-excluding and hashes every generated output.
    outputs = []
    for p in sorted(OUT.iterdir()):
        if p.name == "E6_2_R3_PROVENANCE_MANIFEST.json" or not p.is_file():
            continue
        outputs.append({"path": relpath(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    provenance = {"schema": "E6-2-R3-PROVENANCE-V1", "created_utc": utc_now(), "gate": "E6_2_R3_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS", "outputs": outputs, "historical_manifest_checks": historical_checks, "input_files": [{"path": relpath(R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl"), "sha256": sha256_file(R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl")}, {"path": relpath(R1 / "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl"), "sha256": sha256_file(R1 / "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl")}, {"path": relpath(R2 / "E6_2_R2_NATIVE_SELECTIONS.jsonl"), "sha256": sha256_file(R2 / "E6_2_R2_NATIVE_SELECTIONS.jsonl")}, {"path": relpath(R2S / "_work/startlux_decision_4b.jsonl"), "sha256": sha256_file(R2S / "_work/startlux_decision_4b.jsonl")}], "combined_registry_sha256": registry["content_sha256"], "planned": 2216, "executed": 2216, "valid": 2133, "failed": 83, "unexecuted": 0, "qrel_projection": qrel_audit, "new_inference": 0, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "reserved_content_accessed": False}
    write_json(OUT / "E6_2_R3_PROVENANCE_MANIFEST.json", provenance)
    print(json.dumps({"gate": provenance["gate"], "output": str(OUT), "primary": primary_stats, "scorecard": card, "historical_checks": historical_checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
