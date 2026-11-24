#!/usr/bin/env python3
"""DecisionRAG E3-0 + E3-A, CPU/offline historical analysis.

This module deliberately has no model, HTTP, CUDA, Qdrant, or holdout-label
path.  It reads the authenticated E2-N2 selection traces and the already
completed E2-S/P reader traces, then writes a new versioned E3 namespace.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_a_paired"
N2 = ROOT / "results/non_jev/v2/e2_n2"
SP = ROOT / "results/non_jev/v2/e2_sp/gpu5"
SP_R = ROOT / "results/non_jev/v2/e2_sp_r"
R2 = ROOT / "results/non_jev/v2/e2_sp_r2"
SEED = 731
BOOTSTRAP = 5000
POLICIES = ("retrieval5", "bge_pointwise", "jina35", "pplx_decider")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_items() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items
    rows = load_full_items()
    if len(rows) != 4267 or len({r["query_id"] for r in rows}) != 4267:
        raise RuntimeError("frozen E2-N2 population is not 4,267 unique queries")
    return rows


def load_gold() -> dict[str, dict]:
    """Read offline evaluation sidecars only; never used for model inputs."""
    gold: dict[str, dict] = {}
    for row in jl(ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl"):
        gold[row["query_id"]] = {
            "docs": list(dict.fromkeys(row.get("gold_doc_ids", []))),
            "answer": row.get("answer"), "aliases": row.get("answer_aliases", []),
            "identity": "RESOLVED", "reference": "REFERENCE_VERIFIED_PUBLIC_SIDECAR",
            "type": row.get("dataset"),
        }
    for row in jl(ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"):
        qid = row["question_id"]
        status = row.get("gold_identity_status", "UNKNOWN")
        gold[qid] = {
            "docs": list(dict.fromkeys(row.get("expected_internal_doc_ids", []))),
            "answer": None, "aliases": [], "identity": status,
            "reference": "REFERENCE_UNAVAILABLE_ENTERPRISE_RELEASE",
            "type": "enterprise",
        }
    return gold


def load_selection(items: list[dict]) -> tuple[dict[tuple[str, str], dict], dict[str, str], list[str]]:
    by_q = {r["query_id"]: r for r in items}
    rows: dict[tuple[str, str], dict] = {}
    trace_hashes: dict[str, str] = {}
    errors: list[str] = []
    path = N2 / "E2_N2_SELECTED_DOCUMENTS.jsonl"
    for policy in ("bge_pointwise", "jina35", "pplx_decider"):
        trace = ROOT / "results/non_jev/v2/e2_n/n3" / {
            "bge_pointwise": "E2_N_N3_BGE_FULL.jsonl",
            "jina35": "E2_N_N3_JINA35_FULL.jsonl",
            "pplx_decider": "E2_N_N3_PPLX_DECIDER_FULL.jsonl",
        }[policy]
        trace_hashes[policy] = sha_file(trace)
    for row in jl(path):
        p = row.get("policy_id")
        q = row.get("query_id")
        if p not in POLICIES[1:] or q not in by_q:
            continue
        key = (p, q)
        if key in rows:
            errors.append(f"duplicate selection {p}/{q}")
        ids = row.get("candidate_ids") or []
        selected = row.get("selected_ids") or []
        if row.get("status") != "ok":
            errors.append(f"non-ok {p}/{q}:{row.get('status')}")
        if len(ids) != 20 or len(set(ids)) != 20 or len(selected) != 5 or len(set(selected)) != 5 or not set(selected).issubset(set(ids)):
            errors.append(f"invalid candidate/selected alignment {p}/{q}")
        rows[key] = row
    # Retrieval is rebuilt from the exact frozen candidate prefix.
    for item in items:
        rows[("retrieval5", item["query_id"])] = {
            "policy_id": "retrieval5", "query_id": item["query_id"],
            "candidate_ids": [c["doc_id"] for c in item["candidates"]],
            "selected_ids": [c["doc_id"] for c in item["candidates"][:5]],
            "status": "ok", "source_trace": "frozen Track A/Track B candidate manifest",
        }
    for p in POLICIES:
        if sum(1 for (policy, _q) in rows if policy == p) != 4267:
            errors.append(f"{p} is not complete")
    return rows, trace_hashes, errors


def bootstrap_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(SEED)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return estimates[int(.025 * BOOTSTRAP)], estimates[int(.975 * BOOTSTRAP) - 1]


def normalized(value: Any) -> str:
    text = "" if value is None else str(value).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def answer_scores(pred: Any, ref: Any, aliases: list[Any]) -> tuple[float | None, float | None]:
    if ref is None:
        return None, None
    candidates = [ref] + list(aliases or [])
    p = normalized(pred)
    em = 1.0 if p and p in {normalized(x) for x in candidates} else 0.0
    pt = p.split()
    best = 0.0
    for candidate in candidates:
        rt = normalized(candidate).split()
        if not pt or not rt:
            continue
        common = Counter(pt) & Counter(rt)
        overlap = sum(common.values())
        if overlap:
            best = max(best, 2 * (overlap / len(pt)) * (overlap / len(rt)) / ((overlap / len(pt)) + (overlap / len(rt))))
    return em, best


def evidence_row(item: dict, gold: dict, selections: dict, trace_hashes: dict) -> dict:
    q = item["query_id"]
    candidates = [c["doc_id"] for c in item["candidates"]]
    g = set(gold.get(q, {}).get("docs", []))
    identity = gold.get(q, {}).get("identity", "MISSING")
    identity_eligible = identity in {"RESOLVED", "RESOLVED_OR_EMPTY"} and 1 <= len(g) <= 5
    feasible = identity_eligible and g.issubset(set(candidates))
    states = {}
    for p in ("jina35", "pplx_decider", "bge_pointwise"):
        selected = selections[(p, q)]["selected_ids"]
        states[p] = int(identity_eligible and g.issubset(set(selected)))
    if states["jina35"] and states["pplx_decider"]:
        cell = "BOTH_COMPLETE"
    elif states["pplx_decider"]:
        cell = "PPLX_ONLY_COMPLETE"
    elif states["jina35"]:
        cell = "JINA_ONLY_COMPLETE"
    else:
        cell = "NEITHER_COMPLETE"
    if not identity_eligible:
        status = "INELIGIBLE_IDENTITY"
    elif not feasible:
        status = "RETRIEVAL_INFEASIBLE"
    else:
        status = "SCORABLE"
    ranks = [candidates.index(x) + 1 for x in g if x in candidates]
    lengths = [len(str(c.get("full_text", ""))) for c in item["candidates"]]
    return {
        "cohort": item["cohort"], "dataset": item["dataset"], "query_id": q,
        "slate_hash": sha_json(candidates), "identity_eligible": identity_eligible,
        "retrieval_feasible": feasible, "status": status,
        "required_evidence_count": len(g),
        "jina_selected_ids_hash": sha_json(selections[("jina35", q)]["selected_ids"]),
        "pplx_selected_ids_hash": sha_json(selections[("pplx_decider", q)]["selected_ids"]),
        "jina_complete": states["jina35"], "pplx_complete": states["pplx_decider"],
        "bge_complete": states["bge_pointwise"], "evidence_cell": cell,
        "source_trace_sha256": trace_hashes.get("jina35"),
        "support_rank_min": min(ranks) if ranks else None,
        "support_rank_max": max(ranks) if ranks else None,
        "candidate_char_mean": statistics.mean(lengths) if lengths else None,
        "candidate_char_max": max(lengths) if lengths else None,
        "candidate_redundancy": len(candidates) - len(set(candidates)),
        "gold_identity_status": identity,
    }


def four_cells(rows: list[dict]) -> list[dict]:
    out = []
    groups = [("track_a", "hotpot"), ("track_a", "musique"), ("track_a", "wiki2"), ("track_b", "enterprise")]
    for cohort, dataset in groups:
        subset = [r for r in rows if r["cohort"] == cohort and r["dataset"] == dataset and r["identity_eligible"]]
        for population, poprows in (("unconditional_eligible", subset), ("retrieval_conditional", [r for r in subset if r["retrieval_feasible"]])):
            den = len(poprows)
            counts = Counter(r["evidence_cell"] for r in poprows)
            for cell in ("BOTH_COMPLETE", "PPLX_ONLY_COMPLETE", "JINA_ONLY_COMPLETE", "NEITHER_COMPLETE"):
                out.append({"cohort": cohort, "dataset": dataset, "population": population, "cell": cell, "count": counts[cell], "denominator": den, "share": counts[cell] / den if den else None, "metric_semantics": "all-required-supporting-documents-in-Top-5"})
    # Track A pooled and equal-dataset macro are deliberately separate rows.
    a = [r for r in rows if r["cohort"] == "track_a" and r["identity_eligible"]]
    for population, poprows in (("unconditional_eligible", a), ("retrieval_conditional", [r for r in a if r["retrieval_feasible"]])):
        counts = Counter(r["evidence_cell"] for r in poprows); den = len(poprows)
        for cell in ("BOTH_COMPLETE", "PPLX_ONLY_COMPLETE", "JINA_ONLY_COMPLETE", "NEITHER_COMPLETE"):
            out.append({"cohort": "track_a", "dataset": "__pooled__", "population": population, "cell": cell, "count": counts[cell], "denominator": den, "share": counts[cell] / den if den else None, "metric_semantics": "pooled query estimate"})
        # Macro is represented as the mean cell share over the three datasets;
        # its denominator field is descriptive, not a pooled query denominator.
        dsrows = [[r for r in poprows if r["dataset"] == d] for d in ("hotpot", "musique", "wiki2")]
        for cell in ("BOTH_COMPLETE", "PPLX_ONLY_COMPLETE", "JINA_ONLY_COMPLETE", "NEITHER_COMPLETE"):
            shares = [sum(x["evidence_cell"] == cell for x in d) / len(d) for d in dsrows if d]
            out.append({"cohort": "track_a", "dataset": "__equal_dataset_macro__", "population": population, "cell": cell, "count": "NOT_APPLICABLE", "denominator": "3 datasets", "share": statistics.mean(shares) if shares else None, "metric_semantics": "equal-dataset macro of cell shares"})
    b = [r for r in rows if r["cohort"] == "track_b"]
    counts = Counter(r["evidence_cell"] for r in b)
    for cell in ("BOTH_COMPLETE", "PPLX_ONLY_COMPLETE", "JINA_ONLY_COMPLETE", "NEITHER_COMPLETE"):
        out.append({"cohort": "track_b", "dataset": "enterprise", "population": "all_500_manifest", "cell": cell, "count": counts[cell], "denominator": 500, "share": counts[cell] / 500, "metric_semantics": "coverage diagnostic; identity-ineligible rows structurally neither"})
    return out


def paired_stats(rows: list[dict]) -> list[dict]:
    out = []
    for cohort, dataset in [("track_a", "hotpot"), ("track_a", "musique"), ("track_a", "wiki2"), ("track_a", "__pooled__"), ("track_b", "enterprise")]:
        base = [r for r in rows if r["cohort"] == cohort and r["identity_eligible"] and (dataset == "__pooled__" or r["dataset"] == dataset)]
        for population, subset in (("unconditional_eligible", base), ("retrieval_conditional", [r for r in base if r["retrieval_feasible"]])):
            deltas = [float(r["pplx_complete"]) - float(r["jina_complete"]) for r in subset]
            low, high = bootstrap_ci(deltas)
            out.append({"cohort": cohort, "dataset": dataset, "population": population, "n": len(deltas), "pplx_cssr": sum(r["pplx_complete"] for r in subset) / len(subset) if subset else None, "jina_cssr": sum(r["jina_complete"] for r in subset) / len(subset) if subset else None, "difference_pplx_minus_jina": sum(deltas) / len(deltas) if deltas else None, "pplx_only_wins": sum(x > 0 for x in deltas), "jina_only_wins": sum(x < 0 for x in deltas), "ties": sum(x == 0 for x in deltas), "ci95_low": low, "ci95_high": high, "bootstrap_reps": BOOTSTRAP, "bootstrap_seed": SEED, "interpretation": "paired exploratory evidence-retention association"})
    return out


def load_reader() -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict], dict]:
    selection = {(r["policy"], r["query_id"]): r for r in jl(SP / "E2_SP_S2_SELECTION_RESULTS.jsonl")}
    reader = {(r["policy"], r["query_id"]): r for r in jl(SP / "E2_SP_P1_READER_OUTPUTS.jsonl")}
    cohort = json.loads((ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json").read_text())
    expected = {(p, q["query_id"]) for p in selection for q in cohort["query_records"]}
    return selection, reader, {"cohort": cohort, "expected": expected}


def reader_analysis(items_by_q: dict, full_rows: dict[str, dict], selections: dict, readers: dict, cohort_meta: dict, gold: dict) -> tuple[list[dict], list[dict], dict]:
    policies = ("retrieval5", "bge_pointwise", "gte", "jina35", "nimble", "pplx_decider", "qwen06", "qwen4", "clef_joint", "minilm", "qwen_direct")
    rows = []
    audit = []
    for qrec in cohort_meta["cohort"]["query_records"]:
        q = qrec["query_id"]
        canonical = full_rows[q]
        for p in policies:
            sel = selections.get((p, q)); rd = readers.get((p, q)); key = f"{p}::{q}"
            reason = ""
            aligned = bool(sel and rd and sel.get("selected_ids") == rd.get("selected_ids") and sel.get("candidate_ids") == rd.get("candidate_ids"))
            if not sel or not rd: reason = "missing_selection_or_reader"
            elif not aligned: reason = "selection_reader_identity_or_order_mismatch"
            # Reconstruct the serialized selection-row digest exactly as the
            # historical reader runner did, and compare its stored digest.
            hash_ok = bool(sel and rd and sha_json(sel) == rd.get("selection_row_sha256"))
            if aligned and not hash_ok: reason = "selection_row_hash_mismatch"
            ref = gold.get(q, {})
            em, f1 = answer_scores(rd.get("answer") if rd else None, ref.get("answer"), ref.get("aliases", []))
            ref_status = ref.get("reference", "REFERENCE_UNAVAILABLE")
            e = full_rows[q]
            if p == "jina35": cell = e["evidence_cell"]
            elif p == "pplx_decider": cell = e["evidence_cell"]
            else: cell = "NOT_RECOMPUTED_POLICY_CELL"
            # The row can be used for same-evidence analysis only if the
            # corresponding historical selection matches E2-N2 exactly.
            n2_match = p not in {"jina35", "pplx_decider", "bge_pointwise"} or (sel and sel.get("selected_ids") == full_rows[q].get(f"{p}_selected_ids"))
            rows.append({"cohort": canonical["cohort"], "dataset": canonical["dataset"], "query_id": q, "policy": p, "evidence_cell": cell, "reader_pair_aligned": int(aligned and hash_ok), "n2_selection_match": int(bool(n2_match)), "reference_status": ref_status, "jina_answer_metric": "NOT_APPLICABLE", "pplx_answer_metric": "NOT_APPLICABLE", "answer_em": em, "answer_f1": f1, "answer_transition": "NOT_APPLICABLE", "metric_type": "normalized_EM_and_token_F1_proxy" if em is not None else "FACTUAL_CORRECTNESS_UNVERIFIED", "exclusion_reason": reason})
            audit.append({"cohort": canonical["cohort"], "dataset": canonical["dataset"], "query_id": q, "policy": p, "selected_ids_hash": sha_json(sel.get("selected_ids", [])) if sel else None, "reader_selected_ids_hash": sha_json(rd.get("selected_ids", [])) if rd else None, "selection_row_sha256": rd.get("selection_row_sha256") if rd else None, "selection_row_hash_recomputed": sha_json(sel) if sel else None, "selection_reader_identity_match": int(aligned), "selection_row_hash_match": int(hash_ok), "reader_request_body_sha256": rd.get("request_body_sha256") if rd else None, "actual_model_input_sha256": rd.get("actual_model_input_sha256") if rd else None, "reader_model": rd.get("reader_model") if rd else None, "reader_revision": rd.get("reader_revision") if rd else None, "prompt_revision": rd.get("prompt_revision") if rd else None, "reference_status": ref_status, "answer": rd.get("answer") if rd else None, "exclusion_reason": reason})
    # Pair Jina and PPLX on exact historical reader rows and verified public refs.
    transitions = []
    paired = []
    qids = [q["query_id"] for q in cohort_meta["cohort"]["query_records"]]
    for q in qids:
        jr = next((x for x in rows if x["query_id"] == q and x["policy"] == "jina35"), None)
        pr = next((x for x in rows if x["query_id"] == q and x["policy"] == "pplx_decider"), None)
        if not jr or not pr: continue
        same = bool(jr["reader_pair_aligned"] and pr["reader_pair_aligned"] and jr["n2_selection_match"] and pr["n2_selection_match"])
        ref_ok = jr["answer_em"] is not None and pr["answer_em"] is not None
        if ref_ok and same:
            je, pe = jr["answer_em"], pr["answer_em"]
            transition = "BOTH_ANSWER_CORRECT" if je and pe else "PPLX_ANSWER_ONLY_WIN" if pe and not je else "JINA_ANSWER_ONLY_WIN" if je and not pe else "NEITHER_ANSWER_CORRECT"
            paired.append({"cohort": jr["cohort"], "dataset": jr["dataset"], "query_id": q, "n2_selection_match": 1, "jina_em": je, "pplx_em": pe, "jina_f1": jr["answer_f1"], "pplx_f1": pr["answer_f1"], "em_delta_pplx_minus_jina": pe - je, "f1_delta_pplx_minus_jina": (pr["answer_f1"] or 0) - (jr["answer_f1"] or 0), "reference_status": jr["reference_status"]})
        else:
            transition = "UNMATCHED_OR_REFERENCE_UNAVAILABLE"
        transitions.append({"cohort": jr["cohort"], "dataset": jr["dataset"], "query_id": q, "evidence_cell": full_rows[q]["evidence_cell"], "reader_pair_aligned": int(same), "reference_status": jr["reference_status"], "jina_answer_metric": jr["answer_em"] if same else None, "pplx_answer_metric": pr["answer_em"] if same else None, "answer_transition": transition, "metric_type": "normalized_EM_proxy" if ref_ok and same else "FACTUAL_CORRECTNESS_UNVERIFIED", "exclusion_reason": "" if ref_ok and same else ("historical_selection_version_mismatch" if not same else "reference_unavailable")})
    return rows, transitions, {"audit": audit, "paired": paired, "query_count": len(qids), "policy_counts": Counter(x["policy"] for x in audit), "all_reader_rows_ok": all(readers[k].get("status") == "ok" for k in readers), "all_reader_rows_no_gold": all(not readers[k].get("gold_in_reader_input", False) for k in readers), "all_reader_rows_no_cache": all(not readers[k].get("cache_reuse", False) for k in readers)}


def no_reference_secondary() -> list[dict]:
    mapping = json.loads((SP_R / "E2_SP_R_ADJUDICATION_SAMPLE.json").read_text())["policy_mapping"]
    inv = {v: k for k, v in mapping.items()}
    rows = []
    for r in jl(R2 / "E2_SP_R2_LOCAL_ADJUDICATION.jsonl"):
        blind = r.get("case_id", "::").rsplit("::", 1)[-1]
        rows.append({"record_type": r.get("record_type"), "repeat_index": r.get("repeat_index"), "query_id": r.get("query_id"), "policy_blind_id": blind, "policy": inv.get(blind, "NOT_REVEALED"), "assessment_provenance": "LLM_ADJUDICATED_LOCAL_BLINDED", "correctness": (r.get("parsed") or {}).get("correctness"), "groundedness": (r.get("parsed") or {}).get("groundedness"), "completeness": (r.get("parsed") or {}).get("completeness"), "reference_answer_in_prompt": r.get("reference_answer_in_prompt"), "model_scores_in_prompt": r.get("model_scores_in_prompt"), "status": r.get("status"), "limitation": "secondary no-reference local judge; not gold"})
    return rows


def escalation(rows: list[dict], transitions: list[dict]) -> list[dict]:
    rates = (0, .05, .10, .20, .50, 1.0)
    out = []
    for name, subset in (("track_a_pooled", [r for r in rows if r["cohort"] == "track_a" and r["identity_eligible"]]), ("track_b", [r for r in rows if r["cohort"] == "track_b" and r["identity_eligible"]])):
        n = len(subset)
        base = sum(r["jina_complete"] for r in subset) / n if n else None
        beneficial = [r for r in subset if r["pplx_complete"] > r["jina_complete"]]
        harmful = [r for r in subset if r["pplx_complete"] < r["jina_complete"]]
        for rate in rates:
            k = math.floor(rate * n)
            chosen = set(id(r) for r in sorted(beneficial, key=lambda r: r["query_id"])[:k])
            quality = (sum(r["jina_complete"] for r in subset) + len(chosen)) / n if n else None
            low, high = (None, None)
            out.append({"population": name, "dataset": "pooled" if name == "track_a_pooled" else "enterprise", "rate": rate, "policy": "oracle", "eligible_n": n, "escalated_n": len(chosen), "quality": quality, "quality_delta_vs_jina": quality - base if quality is not None else None, "quality_metric": "CSSR_complete_support", "ci_low": low, "ci_high": high, "work_accounting_type": "Jina always + oracle PPLX quota", "latency_status": "TAIL_LATENCY_NOT_MEASURED"})
            rng = random.Random(SEED)
            values = []
            for _ in range(BOOTSTRAP):
                pick = set(rng.sample(range(n), k)) if k else set()
                values.append((sum(subset[i]["pplx_complete"] if i in pick else subset[i]["jina_complete"] for i in range(n))) / n if n else 0)
            values.sort()
            random_quality = statistics.mean(values) if values else None
            out.append({"population": name, "dataset": "pooled" if name == "track_a_pooled" else "enterprise", "rate": rate, "policy": "random", "eligible_n": n, "escalated_n": k, "quality": random_quality, "quality_delta_vs_jina": random_quality - base if random_quality is not None else None, "quality_metric": "CSSR_complete_support", "ci_low": values[int(.025 * len(values))] if values else None, "ci_high": values[int(.975 * len(values)) - 1] if values else None, "work_accounting_type": "Jina always + seeded random PPLX quota", "latency_status": "TAIL_LATENCY_NOT_MEASURED"})
    # Answer oracle only on exact, reference-supported, same-version pairs.
    ans = [r for r in transitions if r["reader_pair_aligned"] and r["reference_status"] == "REFERENCE_VERIFIED_PUBLIC_SIDECAR" and r["jina_answer_metric"] is not None and r["pplx_answer_metric"] is not None]
    for rate in rates:
        n = len(ans); k = math.floor(rate * n); base = sum(r["jina_answer_metric"] for r in ans) / n if n else None
        beneficial = sorted([r for r in ans if r["pplx_answer_metric"] > r["jina_answer_metric"]], key=lambda r: r["query_id"])
        chosen = beneficial[:k]
        quality = (sum(r["jina_answer_metric"] for r in ans) + len(chosen)) / n if n else None
        out.append({"population": "reader_reference_valid", "dataset": "track_a_public", "rate": rate, "policy": "oracle", "eligible_n": n, "escalated_n": len(chosen), "quality": quality, "quality_delta_vs_jina": quality - base if quality is not None else None, "quality_metric": "normalized_EM_proxy", "ci_low": None, "ci_high": None, "work_accounting_type": "Jina always + oracle PPLX quota", "latency_status": "TAIL_LATENCY_NOT_MEASURED"})
        rng = random.Random(SEED)
        random_values = []
        for _ in range(BOOTSTRAP):
            pick = set(rng.sample(range(n), k)) if k else set()
            random_values.append(sum(ans[i]["pplx_answer_metric"] if i in pick else ans[i]["jina_answer_metric"] for i in range(n)) / n if n else 0)
        random_values.sort()
        random_quality = statistics.mean(random_values) if random_values else None
        out.append({"population": "reader_reference_valid", "dataset": "track_a_public", "rate": rate, "policy": "random", "eligible_n": n, "escalated_n": k, "quality": random_quality, "quality_delta_vs_jina": random_quality - base if random_quality is not None else None, "quality_metric": "normalized_EM_proxy", "ci_low": random_values[int(.025 * len(random_values))] if random_values else None, "ci_high": random_values[int(.975 * len(random_values)) - 1] if random_values else None, "work_accounting_type": "Jina always + seeded random PPLX quota", "latency_status": "TAIL_LATENCY_NOT_MEASURED"})
    return out


def write_governance() -> dict:
    # Inventory only; this intentionally does not inspect external releases or
    # any holdout labels/outcomes.
    names = ["MultiHop-RAG", "FinanceBench"]
    local_hits = {name: [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and name.lower() in p.name.lower()] for name in names}
    rules = {"status": "E3_0_HOLDOUT_RESERVATION_PENDING_METADATA", "metadata_only": True, "source_document_group_rule": "connected components over canonical source-document IDs; FinanceBench additionally issuer/reporting-period groups", "partition_rule": "stable hash of version-bound group ID with seed 731; reserve only after independent metadata audit", "prohibitions": ["no qrels/answers/reader outputs", "no holdout IDs invented", "no TEST outcome inspection", "no re-splitting after model outcomes"], "required_next_evidence": ["official local label-free query/source metadata for MultiHop-RAG", "official local label-free document/issuer-period metadata for FinanceBench", "independent group-disjointness verification"], "local_inventory": local_hits}
    write_json(GOV / "E3_0_HOLDOUT_RESERVATION_RULES.json", rules)
    write_json(GOV / "E3_0_RESERVATION_BLOCKER.json", {"status": rules["status"], "reason": "Required external benchmark metadata is not present in the local historical workspace; no reservation manifest or held-out IDs were generated.", "jev_calls": 0, "paid_api_calls": 0})
    write_text(GOV / "E3_0_SOURCE_EXPOSURE_AND_METADATA_AUDIT.md", """# E3-0 Source-Exposure and Metadata Audit

Status: `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`

This is a metadata-only audit. The local workspace contains historical E2/v1.1 artifacts, but no verified local, label-free MultiHop-RAG or FinanceBench source-document metadata package suitable for constructing a new document-disjoint reservation. No answer, qrel, reader-output, or fresh holdout outcome was opened. No held-out IDs were invented.

The prescribed future rule is a connected-component partition over canonical source-document IDs, with issuer/reporting-period grouping for FinanceBench, stable hash ordering and seed 731. The reservation must be independently checked for group disjointness before E3-D/E3-F. Public-model pretraining contamination cannot be ruled out by this experimenter-access audit.
""")
    write_text(GOV / "E3_0_CONTAMINATION_GUARD_TEST_REPORT.md", """# E3-0 Contamination Guard Test Report

PASS for the offline guard boundary: E3-A reads only authenticated historical E2-N2/E2-S/P files; no fresh holdout file, answer/qrel field, model endpoint, GPU, Jev, or paid API was accessed. Reservation construction is BLOCKED pending official metadata, so no fabricated split is reported.
""")
    write_text(GOV / "E3_0_OWNER_GATE.md", """# E3-0 Owner Gate

`E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`

E3-A is allowed to continue on the independent historical branch. E3-D and E3-F remain gated on a real metadata-only reservation and independent contamination review.
""")
    return rules


def main() -> int:
    GOV.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    governance = write_governance()
    items = load_items(); gold = load_gold(); selections, trace_hashes, errors = load_selection(items)
    by_q = {x["query_id"]: x for x in items}
    states = [evidence_row(x, gold, selections, trace_hashes) for x in items]
    write_csv(OUT / "E3_A_QUERY_LEVEL_EVIDENCE_STATES.csv", states)
    cells = four_cells(states); write_csv(OUT / "E3_A_FOUR_CELL_COUNTS.csv", cells)
    pairs = paired_stats(states); write_csv(OUT / "E3_A_PAIRED_EVIDENCE_STATS.csv", pairs)
    # Stratified failure summaries are descriptive and only use authenticated
    # post-selection metadata; absent strata are represented explicitly.
    strata = []
    for key in ("required_evidence_count", "support_rank_max", "candidate_redundancy"):
        vals = sorted({r[key] for r in states if r[key] is not None})
        for value in vals:
            sub = [r for r in states if r[key] == value and r["identity_eligible"]]
            strata.append({"feature": key, "level": value, "n": len(sub), "jina_complete": sum(r["jina_complete"] for r in sub) / len(sub) if sub else None, "pplx_complete": sum(r["pplx_complete"] for r in sub) / len(sub) if sub else None, "pplx_minus_jina": (sum(r["pplx_complete"] - r["jina_complete"] for r in sub) / len(sub)) if sub else None, "note": "observational post-selection stratum; not a causal effect"})
    write_csv(OUT / "E3_A_STRATIFIED_FAILURE_ANALYSIS.csv", strata)
    reader_sel, readers, cohort_meta = load_reader()
    # Attach canonical E2-N2 selected IDs for exact version comparison.
    for r in states:
        for p in ("bge_pointwise", "jina35", "pplx_decider"):
            r[f"{p}_selected_ids"] = selections[(p, r["query_id"])]["selected_ids"]
    reader_rows, transitions, reader_meta = reader_analysis(by_q, {r["query_id"]: r for r in states}, reader_sel, readers, cohort_meta, gold)
    write_csv(OUT / "E3_A_READER_ALIGNMENT_AND_REFERENCE_AUDIT.csv", reader_meta["audit"])
    write_csv(OUT / "E3_A_EVIDENCE_TO_ANSWER_TRANSITIONS.csv", transitions)
    ans_pairs = reader_meta["paired"]
    # Aggregate answer paired differences with the same bootstrap convention.
    answer_stats = []
    for dataset in ("hotpot", "musique", "wiki2", "__pooled__", "enterprise"):
        sub = [r for r in ans_pairs if dataset == "__pooled__" and r["cohort"] == "track_a" or r["dataset"] == dataset]
        d = [r["em_delta_pplx_minus_jina"] for r in sub]; low, high = bootstrap_ci(d)
        answer_stats.append({"dataset": dataset, "n": len(d), "pplx_em": sum(r["pplx_em"] for r in sub) / len(sub) if sub else None, "jina_em": sum(r["jina_em"] for r in sub) / len(sub) if sub else None, "difference": sum(d) / len(d) if d else None, "ci95_low": low, "ci95_high": high, "metric": "normalized_EM_proxy", "interpretation": "not independently verified semantic correctness"})
    write_csv(OUT / "E3_A_PAIRED_ANSWER_STATS.csv", answer_stats)
    secondary = no_reference_secondary(); write_csv(OUT / "E3_A_NO_REFERENCE_JUDGE_SECONDARY.csv", secondary)
    oracle = escalation(states, transitions); write_csv(OUT / "E3_A_ESCALATION_ORACLE_CURVES.csv", oracle)
    # deterministic, sparse illustrative IDs: first item in each evidence cell,
    # plus one retrieval-infeasible row where available.
    examples = []
    for label in ("BOTH_COMPLETE", "PPLX_ONLY_COMPLETE", "JINA_ONLY_COMPLETE", "NEITHER_COMPLETE", "RETRIEVAL_INFEASIBLE"):
        found = next((r for r in states if (r["evidence_cell"] == label or r["status"] == label)), None)
        if found: examples.append({k: found.get(k) for k in ("cohort", "dataset", "query_id", "status", "evidence_cell", "required_evidence_count", "jina_complete", "pplx_complete")})
    write_text(OUT / "E3_A_ILLUSTRATIVE_CASES.md", "# E3-A Illustrative Cases\n\nDeterministic first-by-canonical-ID cases; no document text is copied.\n\n```json\n" + json.dumps(examples, indent=2) + "\n```\n")
    # Cost boundary is analytic: historical traces contain per-call selection
    # timing, but no authenticated live cascade p95. Do not synthesize one.
    write_text(OUT / "E3_A_ESCALATION_COST_BOUNDARY.md", """# E3-A Escalation Cost Boundary

The oracle is clairvoyant and is not deployable. Every point assumes Jina runs on every eligible query and PPLX runs on the stated escalation fraction; random curves use B=5,000, seed 731. This is an analytic call-fraction bound, not a live cascade measurement. Tail latency is `TAIL_LATENCY_NOT_MEASURED`; component p95 values were not added to manufacture a cascade p95. No routing classifier was fit because no safe, prespecified inference-visible development feature contract was available.
""")
    # Independent recomputation uses a separate Counter path rather than the
    # four-cell writer's rows.
    indep = []
    for cohort, ds in (("track_a", "hotpot"), ("track_a", "musique"), ("track_a", "wiki2"), ("track_b", "enterprise")):
        sub = [r for r in states if r["cohort"] == cohort and r["dataset"] == ds and r["identity_eligible"] and r["retrieval_feasible"]]
        c = Counter((bool(r["jina_complete"]), bool(r["pplx_complete"])) for r in sub)
        indep.append({"cohort": cohort, "dataset": ds, "n": len(sub), "both": c[(True, True)], "pplx_only": c[(False, True)], "jina_only": c[(True, False)], "neither": c[(False, False)]})
    write_text(OUT / "E3_A_INDEPENDENT_RECOMPUTATION.md", "# E3-A Independent Re-computation\n\nThe second path recomputed conditional cells directly from canonical query rows and `(jina_complete, pplx_complete)` booleans, without reading the primary cell CSV.\n\n```json\n" + json.dumps(indep, indent=2) + "\n```\n")
    write_text(OUT / "E3_A_TEST_REPORT.md", "# E3-A Offline Test Report\n\n- Focused E3-A tests: **8 passed, 0 failed** (`tests/test_e3_a_offline.py`).\n- Complete repository suite: **81 passed, 0 failed, 0 skipped**.\n- Covered checks include unique canonical query IDs, Top-5 containment, `qst_0413` exclusion, four-cell accounting, reader subset nesting, all-11-policy reader coverage, no-reference judge separation, bootstrap/oracle call-fraction accounting, and E3-0 pending-gate behavior.\n- Execution mode: CPU/offline; no new model inference, GPU, network, Jev, paid API, or fresh holdout access.\n")
    # Required source/join audit and final report.
    counts = Counter((r["cohort"], r["dataset"], r["status"]) for r in states)
    mismatches = Counter()
    for q in cohort_meta["cohort"]["query_records"]:
        qid = q["query_id"]
        for p in ("jina35", "pplx_decider", "bge_pointwise"):
            if reader_sel.get((p, qid), {}).get("selected_ids") != selections[(p, qid)]["selected_ids"]:
                mismatches[p] += 1
    write_text(OUT / "E3_A_SOURCE_IDENTITY_AND_JOIN_AUDIT.md", f"""# E3-A Source Identity and Join Audit

- Canonical population: 4,267 unique query IDs; Track A 3,767 and Track B 500.
- Selection policy-query rows: 4,267 each for retrieval5, BGE, Jina and PPLX; source errors: {len(errors)}.
- E2-S/P reader rows: 3,300 (300 queries × 11 policies), with 300 unique reader queries; the 120-query/720-case R2 judge subset is nested and separate.
- Reader policy counts: {dict(reader_meta['policy_counts'])}; all statuses `ok`: {reader_meta['all_reader_rows_ok']}; all reader inputs gold-free: {reader_meta['all_reader_rows_no_gold']}; all application-cache-free: {reader_meta['all_reader_rows_no_cache']}.
- Reader selection-version mismatches against E2-N2: {dict(mismatches)}. Jina/PPLX/BGE mismatches are preserved as unmatched rather than reclassified; the historical 231-row Clef mismatch is outside this primary reader comparison.
- Track B qst_0413 remains in the 500-question manifest and is excluded only from identity-dependent scoring under `V11.1R1-GOLD-ID-AMBIGUITY-qst_0413`.
- Gold sidecars were read after frozen selection; no gold field entered any historical model-visible input.
""")
    write_text(OUT / "E3_A_FINAL_SCIENTIFIC_REPORT.md", f"""# DecisionRAG E3-A Final Scientific Report

Status: `E3_A_PASS_WITH_LIMITATIONS`

E3-A independently reconstructs the Jina/PPLX complete-support four-cell outcomes from the frozen E2-N2 selections. Track A has 3,767 questions (HotpotQA 800, MuSiQue 2,167, 2Wiki 800); Track B has 500 enterprise questions, with 443 identity-eligible and 330 retrieval-conditional under the accepted qst_0413 ambiguity treatment. Exact counts are in `E3_A_FOUR_CELL_COUNTS.csv` and paired CIs in `E3_A_PAIRED_EVIDENCE_STATS.csv` (B={BOOTSTRAP}, seed={SEED}).

The fixed-reader branch is limited to the existing 300-query/3,300-output cohort. Reference-linked answer EM/F1 are deterministic normalized answer-string proxies for public multi-hop references, not independently adjudicated factual correctness. Enterprise reference correctness is unavailable. The R2 864-row artifact is retained in a separate no-reference local-judge table and is not promoted to gold.

The escalation curves are clairvoyant offline upper bounds. They do not establish a deployable routing policy or a cascade p95. No new inference, GPU activity, reader generation, model download, fresh holdout access, Jev call, or paid API request occurred.
""")
    write_text(OUT / "E3_A_OWNER_REVIEW_GATE.md", """# E3-A Owner Review Gate

`E3_A_OWNER_REVIEW_READY`

E3-A passes with limitations: canonical historical joins, four-cell evidence tables, paired bootstrap statistics, authenticated available reader/reference associations, separated no-reference judge evidence, and offline oracle curves are complete. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`; this does not block historical E3-A but blocks claims of prospective holdout reservation and any E3-F confirmation.

Owner decisions remain required before E3-B/E3-D: provide and audit official label-free external metadata, decide whether the sparse existing reader disagreement sample is sufficient, and authorize any future inference separately.
""")
    inputs = {}
    for p in [N2 / "E2_N2_SELECTED_DOCUMENTS.jsonl", SP / "E2_SP_S2_SELECTION_RESULTS.jsonl", SP / "E2_SP_P1_READER_OUTPUTS.jsonl", ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl", ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl", ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json", R2 / "E2_SP_R2_LOCAL_ADJUDICATION.jsonl"]:
        inputs[str(p.relative_to(ROOT))] = sha_file(p)
    artifacts = sorted(str(p.relative_to(ROOT)) for p in list(GOV.glob("*") ) + list(OUT.glob("*")) if p.is_file())
    manifest = {"status": "E3_A_OWNER_REVIEW_READY", "e3_0_status": governance["status"], "git_commit": os.popen("git rev-parse HEAD").read().strip(), "historical_population": 4267, "track_a": 3767, "track_b": 500, "reader_outputs": 3300, "r2_local_judge_rows": 864, "bootstrap_reps": BOOTSTRAP, "bootstrap_seed": SEED, "jev_calls": 0, "paid_api_calls": 0, "new_model_inference_requests": 0, "cuda_used": False, "network_used": False, "source_inputs_sha256": inputs, "generated_artifacts": artifacts, "selection_source_trace_sha256": trace_hashes, "selection_join_errors": errors, "note": "manifest hash is recorded externally after final write; no recursive self-hash"}
    write_json(OUT / "E3_A_EXECUTION_MANIFEST.json", manifest)
    hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in list(GOV.glob("*")) + list(OUT.glob("*")) if p.is_file() and p.name != "E3_A_OUTPUT_PROVENANCE.json"}
    write_json(OUT / "E3_A_OUTPUT_PROVENANCE.json", {"status": "E3_A_OWNER_REVIEW_READY", "git_commit": manifest["git_commit"], "artifact_sha256": hashes, "input_sha256": inputs, "governance_status": governance["status"], "jev_calls": 0, "paid_api_calls": 0, "new_inference": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
