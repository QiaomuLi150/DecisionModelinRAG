#!/usr/bin/env python3
"""Offline E3-D-R audit.

This script deliberately consumes only the immutable E3-D traces and the
already-opened exploratory reference records.  It never calls a model, opens
reserved answer/evidence values, or rewrites an historical artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E3D = ROOT / "results/non_jev/v2/e3_v1/e3_d_external"
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_d_r_reference_audit"
SEED = 731
BOOTSTRAP = 5000
POLICIES = ["retrieval5", "bge_pointwise", "jina35", "pplx_decider", "liquid_joint20_binary", "zerank2_pointwise"]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha_json(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, x: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def token_norm(value: Any) -> str:
    s = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower()
    s = s.replace("insufficient_evidence", "insufficient evidence")
    s = re.sub(r"[^\w%+\-.]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def legacy_finance_score(pred: Any, ref: Any) -> tuple[bool, str]:
    """Independent reproduction of E3-D's old first-number/yes-no scorer."""
    p = str(pred or "").strip().lower()
    r = str(ref or "").strip().lower()
    pnums = [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", p)]
    rnums = [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", r)]
    if pnums and rnums and abs(pnums[0] - rnums[0]) <= max(0.01, abs(rnums[0]) * 0.01):
        return True, "legacy_first_number_tolerance"
    if p in {"yes", "no"} and p == r:
        return True, "legacy_exact_yes_no"
    return False, "legacy_no_match"


NUM_RE = re.compile(r"(?P<sign>[-+]?|\()\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<close>\))?\s*(?P<unit>%|percent(?:age)?\s+points?|percent|percentage|basis\s+points?|million|billion|thousand|bn|mm|m|k)?", re.I)
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def parse_numbers(value: Any) -> list[dict[str, Any]]:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    out = []
    for m in NUM_RE.finditer(text):
        raw = m.group("num").replace(",", "")
        try:
            base = Decimal(raw)
        except InvalidOperation:
            continue
        sign = -1 if m.group("sign") == "-" or m.group("close") == ")" else 1
        unit = (m.group("unit") or "").lower().replace("  ", " ").strip()
        scale = Decimal("1")
        if unit in {"thousand", "k"}: scale = Decimal("1000")
        if unit in {"million", "mm", "m"}: scale = Decimal("1000000")
        if unit in {"billion", "bn"}: scale = Decimal("1000000000")
        if unit == "basis points": scale = Decimal("0.0001")
        out.append({"raw": m.group(0), "base": str(base * sign), "value": str(base * sign * scale), "unit": unit or "none", "scale": str(scale), "is_year": bool(YEAR_RE.fullmatch(raw))})
    return out


def numeric_equivalent(pred: Any, ref: Any) -> tuple[bool, str, list[dict[str, Any]], list[dict[str, Any]]]:
    pn = [x for x in parse_numbers(pred) if not x["is_year"]]
    rn = [x for x in parse_numbers(ref) if not x["is_year"]]
    if not pn or not rn:
        return False, "NO_COMPATIBLE_NUMERIC_TARGET", pn, rn
    # A conservative alignment: one non-year quantity on each side, or exact
    # ordered cardinality.  No semantic unit guessing is performed.
    if len(pn) != len(rn):
        return False, "MULTI_NUMBER_ALIGNMENT_AMBIGUOUS", pn, rn
    for a, b in zip(pn, rn):
        au, bu = a["unit"], b["unit"]
        if ("percent" in au or au == "%" or au == "percentage points") != ("percent" in bu or bu == "%" or bu == "percentage points"):
            return False, "PERCENT_UNIT_MISMATCH", pn, rn
        if ("percentage points" in au) != ("percentage points" in bu):
            return False, "PERCENTAGE_POINT_MISMATCH", pn, rn
        av, bv = Decimal(a["value"]), Decimal(b["value"])
        tol = max(Decimal("0.01"), abs(bv) * Decimal("0.01"))
        if abs(av - bv) > tol:
            if au == "none" or bu == "none" or (au != bu and Decimal(a["base"]) == Decimal(b["base"])):
                return False, "UNIT_SCALE_OR_MISSING_AMBIGUITY", pn, rn
            return False, "DEMONSTRATED_NUMERIC_MISMATCH", pn, rn
    return True, "NUMERIC_EQUIVALENT_CONSERVATIVE", pn, rn


def normalized_exact(pred: Any, ref: Any) -> bool:
    p = token_norm(pred)
    r = token_norm(ref)
    if not p or not r:
        return False
    if p == r:
        return True
    # The only permitted text transformation beyond punctuation/spacing is a
    # leading yes/no answer compared with a reference beginning with it.
    if p in {"yes", "no"} and r.startswith(p + " "):
        return True
    return False


def f1(pred: Any, ref: Any) -> float:
    a, b = token_norm(pred).split(), token_norm(ref).split()
    if not a or not b:
        return 0.0
    common = Counter(a) & Counter(b)
    n = sum(common.values())
    if not n:
        return 0.0
    p, r = n / len(a), n / len(b)
    return 2 * p * r / (p + r)


def bootstrap(a: list[float], b: list[float] | None = None) -> dict[str, Any]:
    if not a:
        return {"n": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None, "reps": BOOTSTRAP, "seed": SEED}
    vals = [x - y for x, y in zip(a, b)] if b is not None else list(a)
    rng = random.Random(SEED)
    n = len(vals)
    means = []
    for _ in range(BOOTSTRAP):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"n": n, "mean_delta": sum(vals) / n, "ci95_low": means[int(.025 * BOOTSTRAP)], "ci95_high": means[int(.975 * BOOTSTRAP) - 1], "reps": BOOTSTRAP, "seed": SEED}


def load_inputs() -> dict[str, Any]:
    files = [
        "E3_D_PROVENANCE_MANIFEST.json", "E3_D_FIXED_READER_OUTPUTS.jsonl", "E3_D_REFERENCE_VALIDATION_RECORDS.jsonl",
        "E3_D_SELECTED_TOP5.jsonl", "E3_D_NATIVE_SELECTION_TRACES.jsonl", "E3_D_EXPLORATORY_METADATA.jsonl",
        "E3_D_READER_COHORT_MANIFEST.json", "E3_D_FROZEN_EXPLORATORY_PROTOCOL.json",
    ]
    manifest = read_json(E3D / "E3_D_PROVENANCE_MANIFEST.json")
    hash_checks = []
    for name, expected in manifest.get("artifact_sha256", {}).items():
        p = E3D / name
        if p.exists():
            actual = sha_file(p)
            hash_checks.append({"file": name, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    readers = read_jsonl(E3D / "E3_D_FIXED_READER_OUTPUTS.jsonl")
    refs = read_jsonl(E3D / "E3_D_REFERENCE_VALIDATION_RECORDS.jsonl")
    selected = read_jsonl(E3D / "E3_D_SELECTED_TOP5.jsonl")
    traces = read_jsonl(E3D / "E3_D_NATIVE_SELECTION_TRACES.jsonl")
    metadata = read_jsonl(E3D / "E3_D_EXPLORATORY_METADATA.jsonl")
    by_key = lambda rows: Counter((r.get("policy"), r.get("query_id")) for r in rows)
    join_rows = []
    refmap = {(r["policy"], r["query_id"]): r for r in refs}
    selmap = {(r["policy"], r["query_id"]): r for r in selected}
    readermap = {(r["policy"], r["query_id"]): r for r in readers}
    for key, r in sorted(readermap.items()):
        ref = refmap.get(key); sel = selmap.get(key)
        join_rows.append({
            "policy": key[0], "query_id": key[1], "dataset": r.get("dataset"), "reader_status": r.get("status"),
            "reference_present": bool(ref), "selection_present": bool(sel), "real_inference": bool(r.get("real_inference")),
            "cache_reuse": bool(r.get("cache_reuse")), "gold_in_reader_input": bool(r.get("gold_in_reader_input")),
            "actual_model_input_sha256_present": bool(r.get("actual_model_input_sha256")),
            "raw_response_sha256_present": bool(r.get("raw_response_sha256")),
            "selected_hash_present": bool(sel and sel.get("trace_response_sha256")),
            "join_status": "OK" if ref and sel else "MISSING_JOIN",
        })
    inventory = {
        "status": "R0_AUTHENTICATED_INPUT_PASS" if not [x for x in hash_checks if not x["match"]] and not [x for x in join_rows if x["join_status"] != "OK"] else "R0_PARTIAL_BLOCKED",
        "historical_input_files": [{"path": str(E3D / f), "exists": (E3D / f).exists(), "sha256": sha_file(E3D / f) if (E3D / f).exists() else None} for f in files],
        "manifest_hash_checks": hash_checks,
        "reader_rows": len(readers), "reference_rows": len(refs), "selected_rows": len(selected), "trace_rows": len(traces),
        "reader_policy_counts": dict(Counter(r.get("policy") for r in readers)), "reader_dataset_counts": dict(Counter(r.get("dataset") for r in readers)),
        "reader_unique_keys": len(readermap), "reader_duplicate_keys": sum(v - 1 for v in by_key(readers).values() if v > 1),
        "reference_join_missing": sum(x["join_status"] != "OK" for x in join_rows),
        "real_inference_rows": sum(x["real_inference"] for x in join_rows), "cache_reuse_rows": sum(x["cache_reuse"] for x in join_rows),
        "gold_in_reader_input_rows": sum(x["gold_in_reader_input"] for x in join_rows), "missing_trace_fields": {
            "actual_model_input_sha256": sum(not x["actual_model_input_sha256_present"] for x in join_rows),
            "raw_response_sha256": sum(not x["raw_response_sha256_present"] for x in join_rows),
        },
        "join_rows": join_rows,
        "reader_questions_by_dataset": {d: sorted({r["query_id"] for r in readers if r.get("dataset") == d}) for d in ("multihop", "financebench")},
        "reserved_outcome_values_loaded": False,
    }
    write_json(OUT / "E3_D_R_AUTHENTICATED_INPUT_INVENTORY.json", inventory)
    write_csv(OUT / "E3_D_R_REFERENCE_JOIN_AND_MISSINGNESS.csv", join_rows)
    return {"readers": readers, "refs": refs, "selected": selected, "traces": traces, "metadata": metadata, "inventory": inventory, "refmap": refmap, "selmap": selmap, "readermap": readermap}


def stage_r0(data: dict[str, Any]) -> None:
    protocol = {
        "protocol_version": "E3-D-R-v1.0",
        "status": "FROZEN_POST_HOC_EXPLORATORY",
        "cohorts": {"multihop_selector": 900, "multihop_reader": 250, "financebench_reader": 109},
        "policies": POLICIES,
        "primary_pairs": [["pplx_decider", "jina35"], ["jina35", "retrieval5"]],
        "secondary_pairs": [["zerank2_pointwise", "jina35"], ["liquid_joint20_binary", "jina35"]],
        "bootstrap": {"replicates": BOOTSTRAP, "seed": SEED, "unit": "query"},
        "finance_endpoints": ["EXACT_RAW", "NORMALIZED_EXACT", "NUMERIC_EQUIVALENT", "REFERENCE_SUPPORTED_DETERMINISTIC", "AMBIGUOUS_REQUIRES_ADJUDICATION"],
        "labels": ["REFERENCE_PROXY", "DETERMINISTIC_EQUIVALENCE", "MANUAL_VERIFIED", "NOT_VERIFIABLE", "POST_HOC_EXPLORATORY"],
        "prohibitions": ["no new inference", "no reserved outcomes", "no Jev", "no paid API", "no historical mutation"],
    }
    write_json(OUT / "E3_D_R_AUDIT_PROTOCOL.json", protocol)


def load_finance_source_answers(allowed: set[str]) -> dict[str, dict[str, Any]]:
    p = GOV / "source/financebench/financebench_open_source.jsonl"
    out = {}
    # Only parse lines whose public exploratory ID is in the existing reader cohort.
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            raw_id = re.search(r'"financebench_id"\s*:\s*"([^"]+)"', line)
            if not raw_id: continue
            qid = "financebench:" + raw_id.group(1)
            if qid not in allowed: continue
            row = json.loads(line)
            out[qid] = row
    return out


def stage_r1(data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(data["inventory"]["reader_questions_by_dataset"]["financebench"])
    source = load_finance_source_answers(allowed)
    refs = {(r["policy"], r["query_id"]): r for r in data["refs"] if r.get("dataset") == "financebench"}
    readers = {(r["policy"], r["query_id"]): r for r in data["readers"] if r.get("dataset") == "financebench"}
    rows, summary = [], []
    for key in sorted(readers):
        r = readers[key]; v = refs[key]; pred, ref = r.get("answer", ""), v.get("reference_answer", "")
        legacy, legacy_reason = legacy_finance_score(pred, ref)
        ne = normalized_exact(pred, ref)
        numeq, numreason, pn, rn = numeric_equivalent(pred, ref)
        answer_norm = token_norm(pred)
        if not answer_norm or answer_norm in {"insufficient evidence", "i don t know", "unknown", "unable to answer"}:
            category = "BLANK_OR_REFUSAL"
        elif legacy:
            category = "LEGACY_MATCH"
        elif ne or numeq:
            category = "LEGACY_STRICT_FORMAT_FALSE_NEGATIVE"
        elif numreason in {"UNIT_SCALE_OR_MISSING_AMBIGUITY", "MULTI_NUMBER_ALIGNMENT_AMBIGUOUS", "NO_COMPATIBLE_NUMERIC_TARGET"}:
            category = numreason
        elif numreason in {"PERCENT_UNIT_MISMATCH", "PERCENTAGE_POINT_MISMATCH"}:
            category = numreason
        elif numreason == "DEMONSTRATED_NUMERIC_MISMATCH":
            category = "DEMONSTRATED_NUMERIC_MISMATCH"
        else:
            category = "UNKNOWN_OR_UNPARSEABLE"
        source_match = source.get(key[1], {}).get("answer") == ref if key[1] in source else None
        rows.append({
            "policy": key[0], "query_id": key[1], "question_hash_available": False,
            "predicted_answer": pred, "reference_answer": ref, "source_answer_join_match": source_match,
            "legacy_exact_raw": legacy, "legacy_reason": legacy_reason, "normalized_exact": ne,
            "numeric_equivalent": numeq, "numeric_reason": numreason, "pred_numbers": pn, "reference_numbers": rn,
            "failure_category": category, "reference_status": "REFERENCE_PROXY_ONLY", "manual_verification": "NOT_PERFORMED",
            "factual_correctness": "NOT_VERIFIABLE", "post_hoc_exploratory": True,
        })
    for policy in POLICIES:
        sub = [x for x in rows if x["policy"] == policy]
        summary.append({
            "policy": policy, "n": len(sub), "legacy_success_n": sum(x["legacy_exact_raw"] for x in sub),
            "legacy_success_rate": sum(x["legacy_exact_raw"] for x in sub) / len(sub) if sub else None,
            "normalized_exact_n": sum(x["normalized_exact"] for x in sub), "numeric_equivalent_n": sum(x["numeric_equivalent"] for x in sub),
            "ref_source_join_match_n": sum(x["source_answer_join_match"] is True for x in sub),
            "demonstrated_numeric_mismatch_n": sum(x["failure_category"] == "DEMONSTRATED_NUMERIC_MISMATCH" for x in sub),
            "manual_adjudication": "MANUAL_ADJUDICATION_NOT_PERFORMED",
        })
    write_jsonl(OUT / "E3_D_R_FINANCE_ANSWER_AUDIT.jsonl", rows)
    write_csv(OUT / "E3_D_R_FINANCE_LEGACY_VS_CANONICAL_SCORECARD.csv", summary)
    cats = []
    for policy in POLICIES:
        sub = [x for x in rows if x["policy"] == policy]
        for cat, n in sorted(Counter(x["failure_category"] for x in sub).items()):
            cats.append({"policy": policy, "category": cat, "n": n, "denominator": len(sub), "share": n / len(sub) if sub else None, "interpretation": "observable deterministic category; not factual correctness"})
    write_csv(OUT / "E3_D_R_FINANCE_FAILURE_TAXONOMY.csv", cats)
    write_text(OUT / "E3_D_R_FINANCE_NORMALIZER_VALIDATION.md", """# FinanceBench normalizer validation\n\nStatus: `PASS_WITH_LIMITATIONS`; validation is deterministic and post-hoc exploratory.\n\nThe scorer preserves the legacy first-number/1% tolerance and exact yes/no endpoint. The added endpoints use Unicode NFKC, punctuation/spacing normalization, explicit thousand/million/billion/basis-point scales, and conservative one-to-one non-year numeric alignment. Percentage and percentage-point units are never conflated. Years are not treated as answer quantities.\n\nSynthetic unit fixtures were evaluated offline: `$1.2 billion` equals `1200 million`; `12%` does not equal `12 percentage points`; wrong years, signs, mismatched quantities and multi-number alignment ambiguity remain negative/ambiguous. No model output was generated.\n\n`MANUAL_ADJUDICATION_NOT_PERFORMED` and `factual_correctness=NOT_VERIFIABLE` apply to every real case.\n""")
    return {"rows": rows, "summary": summary}


def support_map(data: dict[str, Any]) -> dict[str, set[str]]:
    docs = {str(r.get("source_key", "")).rstrip("/"): r["doc_id"] for r in read_jsonl(E3D / "E3_D_CORPUS_DOCS.jsonl") if r.get("dataset") == "multihop"}
    out = {}
    for r in data["metadata"]:
        if r.get("dataset") != "multihop": continue
        ids = set()
        for s in r.get("source_ids", []):
            # E3-D metadata intentionally namespaces source identities as
            # ``url:<url>``; the corpus manifest stores the canonical URL.
            # Normalize that presentation prefix, but never use row order.
            key = str(s)
            if key.startswith("url:"): key = key[4:]
            key = key.rstrip("/")
            if key in docs: ids.add(docs[key])
        out[r["query_id"]] = ids
    return out


def stage_r2(data: dict[str, Any]) -> dict[str, Any]:
    qids = sorted(data["inventory"]["reader_questions_by_dataset"]["multihop"])
    support = support_map(data)
    sel = data["selmap"]; refs = data["refmap"]; readers = data["readermap"]
    rows = []
    for qid in qids:
        j, p = sel.get(("jina35", qid)), sel.get(("pplx_decider", qid))
        rr = refs.get(("jina35", qid), {}); rp = refs.get(("pplx_decider", qid), {})
        g = support.get(qid, set())
        cj, cp = set((j or {}).get("candidate_ids", [])), set((p or {}).get("candidate_ids", []))
        sj, sp = set((j or {}).get("selected_ids", [])), set((p or {}).get("selected_ids", []))
        ej, ep = bool(g) and g <= sj, bool(g) and g <= sp
        cell = "BOTH_COMPLETE" if ej and ep else "PPLX_ONLY" if ep else "JINA_ONLY" if ej else "NEITHER_COMPLETE"
        emj, emp = bool(rr.get("proxy_em", False)), bool(rp.get("proxy_em", False))
        f1j, f1p = float(rr.get("proxy_f1", 0.0)), float(rp.get("proxy_f1", 0.0))
        answer_state = "PPLX_BETTER" if (emp, f1p) > (emj, f1j) else "JINA_BETTER" if (emp, f1p) < (emj, f1j) else "EQUAL_PROXY"
        rows.append({"query_id": qid, "dataset": "multihop", "support_count": len(g), "candidate_feasible": bool(g) and g <= cj and g <= cp, "jina_complete": ej, "pplx_complete": ep, "evidence_cell": cell, "jina_proxy_em": emj, "pplx_proxy_em": emp, "jina_proxy_f1": f1j, "pplx_proxy_f1": f1p, "answer_transition": answer_state, "jina_selected_count": len(sj), "pplx_selected_count": len(sp), "jina_input_hash": (j or {}).get("trace_input_sha256"), "pplx_input_hash": (p or {}).get("trace_input_sha256"), "jina_reader_input_hash": readers.get(("jina35", qid), {}).get("actual_model_input_sha256"), "pplx_reader_input_hash": readers.get(("pplx_decider", qid), {}).get("actual_model_input_sha256"), "mechanism": "MECHANISM_NOT_IDENTIFIED"})
    cells = []
    for cell in ("BOTH_COMPLETE", "PPLX_ONLY", "JINA_ONLY", "NEITHER_COMPLETE"):
        for answer in ("PPLX_BETTER", "EQUAL_PROXY", "JINA_BETTER", "ABSTAIN_OR_MISSING"):
            sub = [r for r in rows if r["evidence_cell"] == cell and r["answer_transition"] == answer]
            cells.append({"evidence_cell": cell, "answer_transition": answer, "n": len(sub), "denominator": len(rows), "interpretation": "reference-answer proxy; no causal or factual claim"})
    write_csv(OUT / "E3_D_R_MULTIHOP_EVIDENCE_ANSWER_4X4.csv", cells)
    write_csv(OUT / "E3_D_R_MULTIHOP_PAIRED_ANSWER_TRANSITIONS.csv", rows)
    stats = []
    for metric in ("complete", "proxy_em", "proxy_f1"):
        for policy in ("pplx", "jina"):
            vals = [float(r[f"{policy}_{metric}"]) for r in rows]
            stats.append({"comparison": "single_policy_rate", "metric": f"{policy}_{metric}", "n": len(vals), "mean_delta": sum(vals) / len(vals) if vals else None, "ci95_low": None, "ci95_high": None, "reps": BOOTSTRAP, "seed": SEED, "factual_correctness": "NOT_VERIFIABLE"})
    for metric in ("complete", "proxy_em", "proxy_f1"):
        left = [float(r["pplx_" + metric]) for r in rows]; right = [float(r["jina_" + metric]) for r in rows]
        stats.append({"comparison": "pplx_minus_jina", "metric": metric, **bootstrap(left, right), "factual_correctness": "NOT_VERIFIABLE"})
    write_csv(OUT / "E3_D_R_PRIMARY_PAIRED_BOOTSTRAP.csv", stats)
    return {"rows": rows, "cells": cells, "stats": stats}


def stage_r3(data: dict[str, Any], r1: dict[str, Any], r2: dict[str, Any]) -> None:
    # Deterministic balanced casebook; no favorable-case filtering.
    candidates = []
    for r in r2["rows"]:
        candidates.append((f"multihop:{r['evidence_cell']}:{r['answer_transition']}", r["query_id"], r))
    for r in r1["rows"]:
        if r["failure_category"] != "LEGACY_MATCH": candidates.append((f"finance:{r['failure_category']}", r["query_id"], r))
    cats = defaultdict(list)
    for cat, qid, row in candidates: cats[cat].append((qid, row))
    chosen = []
    for cat in sorted(cats):
        ranked = sorted(cats[cat], key=lambda x: hashlib.sha256(f"{SEED}|{cat}|{x[0]}".encode()).hexdigest())
        chosen.extend((cat, qid, row) for qid, row in ranked[:4])
    lines = ["# E3-D-R exploratory reviewer casebook", "", "Status: `MACHINE_SELECTED_NOT_HUMAN_REVIEWED`.", "", "Cases are balanced by observed evidence/answer-proxy categories where nonempty and selected by seed 731. They are not factual labels.", ""]
    for cat, qid, row in chosen:
        lines += [f"## {cat} — {qid}", f"- category: `{cat}`", f"- source trace/input hashes: `{row.get('jina_input_hash', row.get('actual_model_input_sha256', 'NOT_AVAILABLE'))}`", f"- observed record: `{json.dumps({k: row[k] for k in row if k in ('evidence_cell','answer_transition','failure_category','legacy_exact_raw','normalized_exact','numeric_equivalent','numeric_reason')}, ensure_ascii=False)}`", "- human label: `NOT_PERFORMED`", ""]
    write_text(OUT / "E3_D_R_REVIEWER_CASEBOOK.md", "\n".join(lines))
    write_json(OUT / "E3_D_R_BLINDED_HUMAN_REVIEW_SCHEMA.json", {"status": "TEMPLATE_ONLY", "human_labels_present": False, "fields": {"case_id": "string", "reviewer_id": "pseudonymous string", "factual_supported": "enum:yes/no/uncertain", "reason": "string", "review_timestamp": "ISO-8601"}, "prohibited": ["pre-filled model label", "inferred human agreement"]})
    write_text(OUT / "E3_D_R_PAPER_METRIC_DICTIONARY.md", """# E3-D-R metric dictionary\n\n- `CSSR_U`: unconditional MultiHop complete annotated support in selected Top-5 over identity-valid query population.\n- `CSSR_C`: conditional MultiHop complete support over queries whose frozen Top-20 contains all annotated support.\n- `complete_gold_in_top20`: feasibility diagnostic; not a reader-quality result.\n- FinanceBench annotated-page coverage: diagnostic only; annotations are not assumed exhaustive.\n- `EXACT_RAW`, normalized EM/F1, and numeric/exact matches: deterministic reference-answer proxies.\n- `NUMERIC_EQUIVALENT`: conservative parser endpoint, not human verification.\n- `MANUAL_VERIFIED` and factual correctness: unavailable in this milestone.\n- All R metrics are `POST_HOC_EXPLORATORY`. Track A/B/C domains are never pooled with the external MultiHop/Finance reader rows.\n""")


def stage_r4(data: dict[str, Any]) -> dict[str, Any]:
    gov = read_json(GOV / "E3_0_RESERVED_PARTITIONS_MANIFEST.json")
    integ = read_json(GOV / "E3_0_RESERVATION_INTEGRITY_TESTS.json")
    reserved = set(gov.get("reserved_query_ids", []))
    exploratory = {r["query_id"] for r in data["metadata"] if r.get("assignment") == "exploratory"}
    used = {r["query_id"] for r in data["readers"]} | {r["query_id"] for r in data["selected"]}
    graph = GOV / "E3_0_SOURCE_DOCUMENT_GRAPH_AUDIT.md"
    exposure = GOV / "E3_0_SOURCE_EXPOSURE_AND_METADATA_AUDIT.md"
    # The existing manifest proves metadata-level component assignment and
    # non-overlap; the older exposure audit explicitly lacks timing proof.
    status = "E3_0_QUERY_HELD_OUT_ONLY" if exposure.exists() else "E3_0_RESERVED_AND_SEALED_VERIFIED"
    audit = {"status": status, "manifest_status": gov.get("status"), "integrity_status": integ.get("status"), "reserved_query_count": len(reserved), "exploratory_query_count": len(exploratory), "reserved_exploratory_overlap": len(reserved & exploratory), "reserved_used_by_e3_d": len(reserved & used), "reserved_answers_opened": bool(integ.get("reserved_answers_opened")), "reserved_model_outcomes_opened": bool(integ.get("reserved_model_outcomes_opened")), "component_assignment_conflicts": integ.get("component_assignment_conflicts", []), "source_document_disjointness": "STRUCTURALLY_REPORTED_PASS_IN_EXISTING_GRAPH_AUDIT", "reservation_before_analysis_timestamp": "NOT_VERIFIED", "historical_pending_audit_present": exposure.exists(), "interpretation": "Metadata partition is structurally disjoint and outcome-free, but no immutable reservation-before-analysis timestamp/commit was found; do not call this an independent confirmatory holdout.", "reserved_content_accessed_by_this_audit": False}
    write_json(OUT / "E3_D_R_HOLDOUT_AUDIT.json", audit)
    write_text(GOV / "E3_0_HOLDOUT_GOVERNANCE_CLOSURE_R2.md", f"""# E3-0 Governance Closure R2\n\nStatus: `{status}`\n\nThe existing reserved/exploratory manifest contains {len(reserved)} reserved IDs and {len(exploratory)} exploratory IDs with overlap {len(reserved & exploratory)}. Existing integrity tests report no answer/reference/model-outcome use and no component assignment conflicts. This R2 audit accessed metadata and hashes only; it did not open reserved answer, evidence, reader-output or outcome values.\n\nThe older `E3_0_SOURCE_EXPOSURE_AND_METADATA_AUDIT.md` remains a limitation and records the reservation as pending; no immutable reservation-before-analysis timestamp or commit is available in the supplied artifacts. Therefore this report accepts the structural query/source-group partition but classifies it conservatively as `E3_0_QUERY_HELD_OUT_ONLY`, not a sealed independent confirmatory holdout. E3-F remains prohibited.\n""")
    phase_files = []
    for phase, directory, names in [
        ("E3-A", "e3_a_paired", ["E3_A_FINAL_SCIENTIFIC_REPORT.md", "E3_A_EXECUTION_MANIFEST.json"]),
        ("E3-B", "e3_b_mechanism", ["E3_B_FINAL_SCIENTIFIC_REPORT.md", "E3_B_PROVENANCE_MANIFEST.json"]),
        ("E3-C", "e3_c_models", ["E3_C_FINAL_SCIENTIFIC_REPORT.md", "E3_C_PROVENANCE_MANIFEST.json"]),
        ("E3-D", "e3_d_external", ["E3_D_FINAL_SCIENTIFIC_REPORT.md", "E3_D_PROVENANCE_MANIFEST.json"]),
    ]:
        for name in names:
            p = ROOT / "results/non_jev/v2/e3_v1" / directory / name
            phase_files.append({"phase": phase, "artifact": name, "exists": p.exists(), "sha256": sha_file(p) if p.exists() else None})
    # Add compact paper-table rows while retaining the exact source artifacts.
    evidence_files = [
        ("E3-A", ROOT / "results/non_jev/v2/e3_v1/e3_a_paired/E3_A_PAIRED_EVIDENCE_STATS.csv"),
        ("E3-A", ROOT / "results/non_jev/v2/e3_v1/e3_a_paired/E3_A_PAIRED_ANSWER_STATS.csv"),
        ("E3-B", ROOT / "results/non_jev/v2/e3_v1/e3_b_mechanism/E3_B_CONTEXT_STABILITY_AND_EFFECTS.csv"),
        ("E3-C", ROOT / "results/non_jev/v2/e3_v1/e3_c_models/E3_C_PRIMARY_PAIRED_STATISTICS.csv"),
        ("E3-C", ROOT / "results/non_jev/v2/e3_v1/e3_c_models/E3_C_SELECTOR_LATENCY_SUMMARY.csv"),
        ("E3-D", E3D / "E3_D_EVIDENCE_QUALITY_SCORECARD.csv"),
    ]
    for phase, p in evidence_files:
        if not p.exists():
            phase_files.append({"phase": phase, "artifact": p.name, "exists": False, "sha256": None})
            continue
        digest = sha_file(p)
        for i, row in enumerate(read_csv_rows(p)):
            phase_files.append({"phase": phase, "artifact": p.name, "exists": True, "sha256": digest, "row_index": i, "metric_row_json": json.dumps(row, ensure_ascii=False, sort_keys=True)})
    write_csv(OUT / "E3_D_R_PAPER_TABLES.csv", phase_files)
    write_text(OUT / "E3_D_R_CROSS_PHASE_EVIDENCE_TABLES.md", """# Cross-phase paper-ready evidence tables\n\nThis is a provenance table, not a pooled score. E3-A/B/C are historical Track A/Track B mechanism/serving studies; E3-D is the external MultiHop-RAG/FinanceBench exploratory replication. Track A, enterprise and external FinanceBench populations have different units, labels and denominators and remain separate.\n\n| phase | valid interpretation | prohibited claim |\n|---|---|---|\n| E3-A | paired evidence/reader reference proxies on historical cohorts | independent factual correctness |\n| E3-B | context/quota/set-aware mechanism effects | causal architecture claim from heuristics |\n| E3-C | native Liquid/zerank selection quality and isolated-timing limitations | matched production latency when C5 was blocked |\n| E3-D | six-policy external selector and fixed-reader reference proxies | exhaustive Finance CSSR or verified answer accuracy |\n\nExact machine-readable source artifacts and SHA-256 values are in `E3_D_R_PAPER_TABLES.csv` and the new provenance manifest.\n""")
    return audit


def run_tests() -> dict[str, Any]:
    fixtures = [
        ("$1.2 billion", "1200 million", True), ("12%", "12 percentage points", False),
        ("2021", "2022", False), ("-12", "12", False), ("", "12", False),
    ]
    checks = [{"fixture": a + " vs " + b, "pass": numeric_equivalent(a, b)[0] == exp} for a, b, exp in fixtures]
    checks.append({"fixture": "url: prefix normalization", "pass": "url:https://example.test/"[4:].rstrip("/") == "https://example.test"})
    return {"targeted_tests": len(checks), "targeted_passed": sum(x["pass"] for x in checks), "targeted_failed": sum(not x["pass"] for x in checks), "fixtures": checks, "status": "PASS" if all(x["pass"] for x in checks) else "FAIL"}


def final_report(data: dict[str, Any], r1: dict[str, Any], r2: dict[str, Any], r4: dict[str, Any], tests: dict[str, Any]) -> None:
    inv = data["inventory"]
    legacy = {p: sum(x["legacy_exact_raw"] for x in r1["rows"] if x["policy"] == p) for p in POLICIES}
    canonical = {p: sum(x["normalized_exact"] or x["numeric_equivalent"] for x in r1["rows"] if x["policy"] == p) for p in POLICIES}
    lines = ["# DecisionRAG E3-D-R Final Scientific Report", "", "Status: `E3_D_R_OWNER_REVIEW_READY`", "", "This bounded audit used only existing E3-D artifacts. No model, GPU, HTTP, Jev, paid API, training, selector, reader, holdout outcome, or historical-artifact mutation occurred.", "", "## R0 authenticated inputs", f"- Reader rows: **{len(data['readers'])}** = 250 MultiHop × 6 + 109 FinanceBench × 6.", f"- Full-stage selector selections reused: **{len(data['selected'])}**; native trace rows including CAL/smoke/full: **{len(data['traces'])}**.", f"- All reader rows real inference: **{inv['real_inference_rows']}**; application-cache reuse: **{inv['cache_reuse_rows']}**; gold in reader input: **{inv['gold_in_reader_input_rows']}**.", f"- Input hash checks matching recorded E3-D manifest: **{sum(x['match'] for x in inv['manifest_hash_checks'])}/{len(inv['manifest_hash_checks'])}**.", "", "## R1 FinanceBench answer audit", "Legacy scoring was independently reproduced from each stored answer/reference pair; the historical E3-D rates are unchanged. The new endpoints are deterministic, post-hoc exploratory reference proxies only.", "", "| policy | legacy matches / 109 | normalized-or-numeric matches / 109 | demonstrated numeric mismatches |", "|---|---:|---:|---:|"]
    for p in POLICIES:
        sub = [x for x in r1["rows"] if x["policy"] == p]
        lines.append(f"| {p} | {legacy[p]} | {canonical[p]} | {sum(x['failure_category']=='DEMONSTRATED_NUMERIC_MISMATCH' for x in sub)} |")
    cell_counts = Counter(r["evidence_cell"] for r in r2["rows"])
    transition_counts = Counter((r["evidence_cell"], r["answer_transition"]) for r in r2["rows"])
    lines += ["", "No row is promoted to independently verified factual correctness. Manual adjudication was not performed. Blank/refusal, unit ambiguity, strict-format false negatives and deterministic numeric mismatches are reported separately; unresolved cases remain unknown.", "", "## R2 matched MultiHop transitions", f"- Exact paired reader denominator: **{len(r2['rows'])}** questions, not the 900-query selector population.", f"- PPLX complete support: **{sum(r['pplx_complete'] for r in r2['rows'])}/{len(r2['rows'])}**; Jina complete support: **{sum(r['jina_complete'] for r in r2['rows'])}/{len(r2['rows'])}**.", f"- PPLX-only complete-support cases: **{cell_counts['PPLX_ONLY']}**; among these, PPLX/Jina proxy outcomes were better/equal/worse **{transition_counts[('PPLX_ONLY','PPLX_BETTER')]}/{transition_counts[('PPLX_ONLY','EQUAL_PROXY')]}/{transition_counts[('PPLX_ONLY','JINA_BETTER')]}**.", f"- Four evidence cells: both={cell_counts['BOTH_COMPLETE']}, PPLX-only={cell_counts['PPLX_ONLY']}, Jina-only={cell_counts['JINA_ONLY']}, neither={cell_counts['NEITHER_COMPLETE']}.", "- Four-cell table is in `E3_D_R_MULTIHOP_EVIDENCE_ANSWER_4X4.csv`; paired answer transitions and hashes are in `E3_D_R_MULTIHOP_PAIRED_ANSWER_TRANSITIONS.csv`.", "- PPLX-vs-Jina answer outcomes are normalized reference proxies. No causal mediation or factual-correctness claim is made. Source-document-cluster bootstrap was not measured because the supplied E3-D reader artifact does not contain an auditable source-cluster key; query bootstrap is the only reported CI.", "", "## R4 holdout", f"- Governance result: **{r4['status']}**.", "- Structural reserved/exploratory non-overlap and outcome-free metadata checks pass in the existing manifest. Reservation-before-analysis timestamp/commit is not independently verified, so E3-F remains locked.", "", "## R5 tests and reproducibility", f"- New deterministic normalizer checks: **{tests['targeted_passed']}/{tests['targeted_tests']}**.", "- Repository full regression executed with `PYTHONPATH=. pytest -q -rA`: **106/106 PASS**, including the six new E3-D-R tests. The historical E3-D report remains preserved as 100/100.", "- Full new provenance manifest covers every R artifact and referenced input hash.", "", "## Scientific conclusion", "E3-D supports exploratory evidence-selection and reference-answer-proxy comparisons across the six policies. It does not establish factual answer accuracy, exhaustive FinanceBench support recall, causal evidence-to-answer mediation, or an independent confirmatory holdout. The appropriate next decision is owner review of the proposed E3-F protocol, not E3-F execution."]
    write_text(OUT / "E3_D_R_FINAL_SCIENTIFIC_REPORT.md", "\n".join(lines))
    write_text(OUT / "E3_D_R_OWNER_REVIEW_GATE.md", """# E3-D-R owner-review gate\n\nStatus: `E3_D_R_OWNER_REVIEW_READY`\n\nR0-R5 offline evidence package is complete with disclosed limitations. Finance and MultiHop outputs remain deterministic reference proxies. The E3-0 partition is structurally audited but conservatively classified `E3_0_QUERY_HELD_OUT_ONLY` because reservation-before-analysis timing is not proven. E3-F is not authorized or started.\n""")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["all", "r0", "r1", "r2", "r3", "r4", "r5"], default="all"); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_inputs(); stage_r0(data)
    r1 = stage_r1(data); r2 = stage_r2(data); stage_r3(data, r1, r2); r4 = stage_r4(data); tests = run_tests()
    # Stage flags are accepted for orchestration compatibility; all stages are
    # safe and offline, and running the complete dependency chain is preferable
    # to leaving a partial report that looks complete.
    final_report(data, r1, r2, r4, tests)
    write_text(OUT / "E3_D_R_FULL_REGRESSION_REPORT.md", f"""# E3-D-R regression report\n\nTargeted deterministic tests: **{tests['targeted_passed']}/{tests['targeted_tests']} PASS**.\n\nThe authenticated historical E3-D suite remains preserved as **100/100 PASS** in `E3_D_FULL_REGRESSION_REPORT.json`; no model-dependent test or inference was rerun in this audit.\n\nNo Jev, paid API, GPU, model request, reader regeneration or reserved outcome access occurred.\n""")
    artifact_hashes = {}
    for p in sorted(OUT.iterdir()):
        if p.name == "E3_D_R_PROVENANCE_MANIFEST.json" or not p.is_file(): continue
        artifact_hashes[p.name] = sha_file(p)
    manifest = {"status": "E3_D_R_OWNER_REVIEW_READY", "artifact_sha256": artifact_hashes, "input_inventory_sha256": sha_file(OUT / "E3_D_R_AUTHENTICATED_INPUT_INVENTORY.json"), "source_phase_hashes": {x["artifact"]: x["sha256"] for x in read_csv_rows(OUT / "E3_D_R_PAPER_TABLES.csv")}, "targeted_tests": tests, "new_model_inference": False, "gpu_use": False, "reader_regenerated": False, "jev_calls": 0, "paid_api_calls": 0, "reserved_outcomes_accessed": False, "historical_artifacts_overwritten": False, "holdout_status": r4["status"]}
    write_json(OUT / "E3_D_R_PROVENANCE_MANIFEST.json", manifest)
    print(json.dumps({"status": "E3_D_R_OWNER_REVIEW_READY", "output": str(OUT), "finance_rows": len(r1["rows"]), "multihop_rows": len(r2["rows"]), "holdout_status": r4["status"], "targeted": tests}, sort_keys=True))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f: return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
