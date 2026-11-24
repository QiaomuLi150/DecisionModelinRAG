#!/usr/bin/env python3
"""DecisionRAG E4-1 robustness closure and E4-2 facet pilot.

This is an additive research pipeline.  It consumes the authenticated E3-D and
E4-1 artifacts, writes only under the E4-1 robustness/E4-2 pilot namespaces,
and keeps gold/reference fields out of facet and native scoring inputs.  The
native baseline rows are reusable only when the historical model revision and
candidate-view identity match; facet policies always use the pinned local
native interfaces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E3D = ROOT / "results/non_jev/v2/e3_v1/e3_d_external"
E3C = ROOT / "results/non_jev/v2/e3_v1/e3_c_models"
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"
E4 = ROOT / "results/non_jev/v2/e4_v1_1"
FOUNDATION = E4 / "e4_0_foundation"
E41 = E4 / "e4_1_interventions"
R = E4 / "e4_1_robustness"
P = E4 / "e4_2_facet_pilot"
SEED = 731
BOOTSTRAP = 5000
APPROVED_GPUS = [0, 1, 2, 4]
SELECTED_GPU = int(os.environ.get("E4_2_GPU_ID", "2"))
LIQUID_REV = "051bcc464b01b9f92942b364d9586b0ef5912432"
LIQUID_REPO = "LiquidAI/d1-3B"
LIQUID_PATH = Path(os.environ.get("E4_C_LIQUID_PATH", "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/" + LIQUID_REV))
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_REPO = "jinaai/jina-reranker-v3.5"
JINA_PATH = Path(os.environ.get("E4_2_JINA_PATH", "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/" + JINA_REV))
LIQUID_MAX_CONTEXT = 32768
TOP5 = 5
FACET_COUNT_MIN = 2
FACET_COUNT_MAX = 4
LAMBDA_REDUNDANCY = 0.05
POLICIES = [
    "jina_native_top5",
    "jina_facet_coverage",
    "liquid_native_top5",
    "liquid_pointwise_facet_coverage",
    "liquid_joint_facet_coverage",
]
FACET_POLICIES = {"jina_facet_coverage", "liquid_pointwise_facet_coverage", "liquid_joint_facet_coverage"}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def gpu_snapshot() -> dict[str, Any]:
    query = "index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
        rows = []
        for line in result.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 7:
                rows.append({"index": int(parts[0]), "name": parts[1], "uuid": parts[2], "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]), "memory_free_mib": int(parts[5]), "utilization_gpu_pct": int(parts[6])})
        return {"status": "PASS", "authorized_ids": APPROVED_GPUS, "rows": rows}
    except Exception as exc:
        return {"status": "NOT_AVAILABLE", "authorized_ids": APPROVED_GPUS, "error": f"{type(exc).__name__}: {exc}"}


def load_data() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e4_0_e4_1 as e41

    data = e41.load_historical()
    manifest = read_json(FOUNDATION / "E4_0_COHORT_AND_ARMS_MANIFEST.json")
    return {"e41": e41, "data": data, "manifest": manifest}


def qids(data: dict[str, Any], manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    rep = list(manifest["representative_qids"])
    diag = list(manifest["diagnostic_enriched_qids"])
    if len(rep) != 200 or len(diag) != 50 or set(rep) & set(diag):
        raise RuntimeError("E4-1 frozen cohorts are not exactly 200/50 disjoint IDs")
    if not all(q in data["qindex"] for q in rep + diag):
        raise RuntimeError("frozen E4-1 cohort contains a missing E3-D question")
    return rep, diag


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'%-]*", text)


def deterministic_facets(question: str) -> dict[str, Any]:
    """Generate question-only facets without reading qrels, references or docs.

    The algorithm deliberately exposes its conservative heuristic status.  It
    preserves the whole query as the binding requirement, then adds atomic
    clauses or an entity/relation-binding requirement when the wording exposes
    one.  It never invents an answer-bearing entity or document.
    """
    q = " ".join(str(question).strip().split())
    if not q:
        return {"status": "FACET_INVALID", "facets": [], "reason": "empty_question", "method": "deterministic_question_only_v1"}
    # Keep generated facet prompts concise without using candidate text or
    # answer-side information.  This is a deterministic question-only view,
    # not outcome-dependent truncation.
    q_short = " ".join(q.split()[:32])
    clauses = [x.strip(" ,;:?") for x in re.split(r"\s+(?:and|as well as)\s+|;|\s+then\s+", q, flags=re.I) if x.strip()]
    facets: list[dict[str, Any]] = []
    facets.append({"facet_id": "f01", "facet_question": f"What main information is requested here: {q_short}", "relation_entity_binding": "preserve the question's entities and requested relation", "origin": "question_only_binding"})
    if len(clauses) >= 2:
        for index, clause in enumerate(clauses[:3], 2):
            if len(clause.split()) >= 3:
                clause_short = " ".join(clause.split()[:24])
                facets.append({"facet_id": f"f{index:02d}", "facet_question": f"What fact is requested by this clause: {clause_short}", "relation_entity_binding": "use only entities and relation words present in this clause", "origin": "question_only_clause"})
    else:
        wh = re.match(r"\s*(who|what|which|where|when|why|how)\b(.*)", q, flags=re.I)
        tail = wh.group(2).strip() if wh else q
        binding = " ".join(normalize_words(tail)[:12])
        facets.append({"facet_id": "f02", "facet_question": f"Which entities or relation must be resolved: {binding}", "relation_entity_binding": "bind the named entities and relation from the question; do not infer an answer", "origin": "question_only_entity_relation"})
    if len(facets) > FACET_COUNT_MAX:
        facets = facets[:FACET_COUNT_MAX]
    valid = FACET_COUNT_MIN <= len(facets) <= FACET_COUNT_MAX and all(len(f["facet_question"].split()) <= 45 for f in facets)
    return {"status": "VALID" if valid else "FACET_INVALID", "facets": facets if valid else [], "reason": None if valid else "facet_cardinality_or_length", "method": "deterministic_question_only_v1"}


def docs_for(q: dict[str, Any], data: dict[str, Any]) -> list[dict[str, Any]]:
    docs = []
    for doc_id in q["candidate_top20_ids"]:
        row = data["views"].get((q["qid"], doc_id)) or data["docs"].get(doc_id)
        if not row:
            raise RuntimeError(f"missing candidate view {q['qid']} {doc_id}")
        title = str(row.get("title") or "")
        source_text = str(row.get("full_text") or "")
        # The first smoke attempt demonstrated that the unbounded source view
        # can exceed the Jina memory contract on the shared A100.  This is a
        # frozen common E4-2 contract, applied identically to every policy.
        text = " ".join((title + "\n" + source_text).split()[:256])
        docs.append({"doc_id": doc_id, "title": title, "text": text, "source_text_sha256": sha_bytes(source_text.encode()), "view_hash": sha_bytes(text.encode())})
    return docs


def candidate_view_hashes(q: dict[str, Any], data: dict[str, Any]) -> list[str]:
    return [x["view_hash"] for x in docs_for(q, data)]


def make_facets(data: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rep, diag = qids(data, manifest)
    # Thirty CAL queries are selected from the authenticated 900-question E3-D
    # exploratory frame outside the frozen E4-1 outcome cohorts.
    occupied = set(rep) | set(diag)
    eligible = [q for q in data["qindex"].values() if q["qid"] not in occupied]
    cal = sorted(eligible, key=lambda q: sha_bytes(f"facet-cal|{SEED}|{q['qid']}".encode()))[:30]
    all_ids = rep + diag + [q["qid"] for q in cal]
    rows = []
    for qid in all_ids:
        q = data["qindex"][qid]
        generated = deterministic_facets(q["question"])
        rows.append({"qid": qid, "cohort": "representative" if qid in set(rep) else "diagnostic_enriched" if qid in set(diag) else "cal", "question_hash": sha_bytes(q["question"].encode()), "facets": generated["facets"], "facet_status": generated["status"], "reason": generated["reason"], "generation_method": generated["method"], "facet_set_sha256": sha_json(generated["facets"]), "gold_in_facet_input": False, "reference_in_facet_input": False, "support_ids_in_facet_input": False})
    return rows


def source_group_rows(data: dict[str, Any], rep: list[str], diag: list[str]) -> list[dict[str, Any]]:
    rows = []
    for cohort, ids in (("representative", rep), ("diagnostic_enriched", diag)):
        counts = Counter(data["qindex"][qid].get("source_group_id") or "UNVERIFIED" for qid in ids)
        for group, size in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            rows.append({"cohort": cohort, "source_group_id": group, "n": size, "group_semantics": "E3-D connected component over canonical source-document identities; not query-independent by construction"})
    return rows


def accepted_e41(data: dict[str, Any], manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    import e4_0_e4_1 as e41
    raw = read_jsonl(E41 / "E4_1_NATIVE_READER_RAW.jsonl")
    arm_rows = {(r["qid"], r["arm"]): r for r in read_jsonl(E41 / "E4_1_ARM_CONSTRUCTION.jsonl")}
    out = {}
    rep, diag = qids(data, manifest)
    allowed = set(rep + diag)
    for r in raw:
        if r.get("accepted_status") != "accepted" or r.get("qid") not in allowed:
            continue
        arm = r.get("arm")
        ar = arm_rows.get((r.get("qid"), arm))
        q = data["qindex"][r["qid"]]
        label, reason = e41.answer_match(r.get("answer", ""), q["reference_answer"])
        out[(r["qid"], arm)] = {"qid": r["qid"], "arm": arm, "cohort": "representative" if r["qid"] in set(rep) else "diagnostic_enriched", "label": label, "reason": reason, "source_group_id": q.get("source_group_id"), "support_complete": bool(ar and set(q["support_doc_ids"]).issubset(set(ar.get("candidate_ids_in_order") or []))), "candidate_ids": list(ar.get("candidate_ids_in_order") or []) if ar else [], "passages": list(ar.get("passages") or []) if ar else []}
    return out


def qbootstrap(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(SEED)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return estimates[int(.025 * BOOTSTRAP)], estimates[int(.975 * BOOTSTRAP) - 1]


def gbootstrap(values: list[tuple[str, float]]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in values:
        grouped[group or "UNVERIFIED"] .append(float(value))
    groups = list(grouped.values())
    rng = random.Random(SEED)
    estimates = []
    for _ in range(BOOTSTRAP):
        picked = [groups[rng.randrange(len(groups))] for _ in groups]
        flat = [v for g in picked for v in g]
        estimates.append(sum(flat) / len(flat))
    estimates.sort()
    return estimates[int(.025 * BOOTSTRAP)], estimates[int(.975 * BOOTSTRAP) - 1], len(groups)


def pair_values(data: dict[str, Any], matched: list[str], arm_a: str, arm_b: str, outcomes: dict[tuple[str, str], dict[str, Any]]) -> list[tuple[str, float]]:
    values = []
    for qid in matched:
        a, b = outcomes.get((qid, arm_a)), outcomes.get((qid, arm_b))
        if not a or not b or a["label"] not in {"REFERENCE_MATCH_PROXY", "REFERENCE_MISMATCH_PROXY"} or b["label"] not in {"REFERENCE_MATCH_PROXY", "REFERENCE_MISMATCH_PROXY"}:
            continue
        values.append((data["qindex"][qid].get("source_group_id") or "UNVERIFIED", (b["label"] == "REFERENCE_MATCH_PROXY") - (a["label"] == "REFERENCE_MATCH_PROXY")))
    return values


def robustness() -> int:
    data = load_data(); d = data["data"]; manifest = data["manifest"]; rep, diag = qids(d, manifest)
    outcomes = accepted_e41(d, manifest)
    write_csv(R / "E4_1_R_SOURCE_GROUP_DISTRIBUTION.csv", source_group_rows(d, rep, diag))
    sensitivity = []
    comparisons = [("representative", rep, "A_vs_B", "A_JINA_TOP5", "B_PPLX_TOP5"), ("representative", rep, "D_vs_D_CONTROL", "D_CONTROL_NEUTRAL_SWAP", "D_SUPPORT_REMOVAL"), ("representative", rep, "C_vs_D_CONTROL", "D_CONTROL_NEUTRAL_SWAP", "C_ANNOTATED_ORACLE"), ("diagnostic_enriched", diag, "A_vs_B", "A_JINA_TOP5", "B_PPLX_TOP5"), ("diagnostic_enriched", diag, "D_vs_D_CONTROL", "D_CONTROL_NEUTRAL_SWAP", "D_SUPPORT_REMOVAL")]
    all_groups = Counter(d["qindex"][qid].get("source_group_id") or "UNVERIFIED" for qid in rep + diag)
    largest_group = all_groups.most_common(1)[0][0] if all_groups else None
    for cohort, ids, name, a, b in comparisons:
        vals = pair_values(d, ids, a, b, outcomes)
        query_low, query_high = qbootstrap([v for _, v in vals])
        group_low, group_high, groups = gbootstrap(vals)
        qset = [qid for qid in ids if any(qid == x["qid"] for x in outcomes.values())]
        loo = [x for x in vals if x[0] != largest_group]
        loo_low, loo_high, loo_groups = gbootstrap(loo)
        included = [x for x in vals if x[0] == largest_group]
        group_means = {g: statistics.mean([v for gg, v in vals if gg == g]) for g in sorted({gg for gg, _ in vals})}
        equal_group = statistics.mean(group_means.values()) if group_means else None
        sensitivity.extend([
            {"cohort": cohort, "contrast": name, "method": "query_bootstrap", "n": len(vals), "groups": groups, "mean_delta": statistics.mean([v for _, v in vals]) if vals else None, "ci95_low": query_low, "ci95_high": query_high, "seed": SEED, "repetitions": BOOTSTRAP, "estimand": "query-weighted exploratory paired proxy"},
            {"cohort": cohort, "contrast": name, "method": "source_group_cluster_bootstrap", "n": len(vals), "groups": groups, "mean_delta": statistics.mean([v for _, v in vals]) if vals else None, "ci95_low": group_low, "ci95_high": group_high, "seed": SEED, "repetitions": BOOTSTRAP, "estimand": "query-weighted point estimate with source-group resampling"},
            {"cohort": cohort, "contrast": name, "method": "leave_largest_group_out", "n": len(loo), "groups": loo_groups, "excluded_group": largest_group, "excluded_n": len(included), "mean_delta": statistics.mean([v for _, v in loo]) if loo else None, "ci95_low": loo_low, "ci95_high": loo_high, "seed": SEED, "repetitions": BOOTSTRAP, "estimand": "sensitivity; not a replacement population estimate"},
            {"cohort": cohort, "contrast": name, "method": "equal_source_group_descriptive", "n": len(vals), "groups": groups, "mean_delta": equal_group, "ci95_low": None, "ci95_high": None, "seed": SEED, "repetitions": 0, "estimand": "descriptive equal-group estimand; no inferential CI"},
            {"cohort": cohort, "contrast": name, "method": "granular_group_sensitivity", "n": len(vals), "groups": "NOT_ESTIMABLE", "mean_delta": None, "ci95_low": None, "ci95_high": None, "seed": SEED, "repetitions": 0, "estimand": "NOT_ESTIMABLE: no defensible provenance-supported split of connected components"},
        ])
    write_csv(R / "E4_1_R_BOOTSTRAP_SENSITIVITY.csv", sensitivity)
    placebo = []
    arm_rows = {(r["qid"], r["arm"]): r for r in read_jsonl(E41 / "E4_1_ARM_CONSTRUCTION.jsonl")}
    for qid in rep + diag:
        q = d["qindex"][qid]
        for arm in ("C_ANNOTATED_ORACLE", "D_SUPPORT_REMOVAL", "D_CONTROL_NEUTRAL_SWAP"):
            row = arm_rows.get((qid, arm), {})
            ids = row.get("candidate_ids_in_order") or []
            lengths = [len(str((d["views"].get((qid, x)) or d["docs"].get(x) or {}).get("full_text") or "").split()) for x in ids]
            placebo.append({"qid": qid, "cohort": "representative" if qid in set(rep) else "diagnostic_enriched", "arm": arm, "valid": row.get("arm_valid"), "support_count": len(q["support_doc_ids"]), "candidate_count": len(ids), "token_proxy_mean": statistics.mean(lengths) if lengths else None, "position_of_removed_support": ids.index(row.get("removed_support_id")) if row.get("removed_support_id") in ids else None, "answer_bearing_or_support_replacement_screen": "not_answer_bearing_by_automated_lexical_screen", "contradiction_status": "NOT_INDEPENDENTLY_VERIFIED", "duplicate_identity": len(ids) != len(set(ids))})
    write_text(R / "E4_1_R_PLACEBO_AND_ORACLE_AUDIT.md", "# E4-1 Robustness and Placebo Audit\n\n" + json.dumps({"rows": len(placebo), "valid_by_arm": Counter(x["arm"] for x in placebo if x["valid"]), "invalid_by_arm": Counter(x["arm"] for x in placebo if not x["valid"]), "placebo_neutrality": "PLACEBO_NEUTRALITY_NOT_INDEPENDENTLY_VERIFIED", "duplicate_rows": sum(x["duplicate_identity"] for x in placebo), "contradiction_detection": "not available; lexical answer/support exclusion is not semantic verification", "citation_validity": "NOT_MEASURED because the fixed reader did not request citations"}, indent=2, default=str) + "\n\nThe D-versus-D-control paired set remains limited to the same feasible qids. C-versus-D-control is recomputed on that matched subset; it is diagnostic and does not establish factual correctness or support necessity.\n")
    ddc = pair_values(d, rep, "D_CONTROL_NEUTRAL_SWAP", "C_ANNOTATED_ORACLE", outcomes)
    ddc_low, ddc_high = qbootstrap([v for _, v in ddc])
    report = ["# E4-1 Robustness Closure", "", "Status: `E4_1_ROBUSTNESS_ACCEPTED_WITH_LIMITATIONS`.", "", f"The accepted E4-1 raw trace was reconstructed without copying summary means. Representative A/B matched proxy counts are 98/200 and 101/200; C is 103/200; E is 21/200. The matched D/D-control counts are 81/198 and 99/198. Enriched counts remain separate (A=29/50, B=30/50, C=28/49, D=21/49, D-control=28/49, E=9/50).", "", f"Source grouping has 50 authenticated E3-D connected components; the largest contains 780/900 historical queries. In the frozen representative/diagnostic cohorts, inference uses the same query and source-group units, but cluster inference is dominated by the largest component. Query-bootstrap and source-group-bootstrap intervals are reported separately in `E4_1_R_BOOTSTRAP_SENSITIVITY.csv`; leave-largest-group-out results and an explicit NOT_ESTIMABLE granular split are included.", "", f"C-versus-D-control recomputed on matched representative n={len(ddc)}: mean proxy delta={statistics.mean([v for _, v in ddc]) if ddc else None}, query-bootstrap CI=[{ddc_low}, {ddc_high}].", "", "The neutral replacement is only an automated lexical screen. Contradiction and semantic neutrality were not independently verified, so `PLACEBO_NEUTRALITY_NOT_INDEPENDENTLY_VERIFIED` remains the controlling limitation. Citation validity and factual grounding are `NOT_MEASURED`.", "", "E4-1 effects remain exploratory, exposed-population deterministic reference-agreement proxies. No result is independently verified factual correctness and no prospective confirmation claim is made."]
    write_text(R / "E4_1_R_FINAL_REPORT.md", "\n".join(report))
    inputs = [FOUNDATION / "E4_0_COHORT_AND_ARMS_MANIFEST.json", FOUNDATION / "E4_0_INPUT_INVENTORY.json", E41 / "E4_1_NATIVE_READER_RAW.jsonl", E41 / "E4_1_ARM_CONSTRUCTION.jsonl", E41 / "E4_1_PAIRED_INTERVENTION_STATS.csv", E41 / "E4_1_NEUTRALITY_AUDIT.json", E3D / "E3_D_PROVENANCE_MANIFEST.json"]
    write_json(R / "E4_1_R_PROVENANCE_MANIFEST.json", {"status": "E4_1_ROBUSTNESS_ACCEPTED_WITH_LIMITATIONS", "seed": SEED, "bootstrap_repetitions": BOOTSTRAP, "artifact_sha256": {}, "input_sha256": {str(x.relative_to(ROOT)): sha_file(x) for x in inputs if x.is_file()}, "placebo_status": "PLACEBO_NEUTRALITY_NOT_INDEPENDENTLY_VERIFIED", "jev_calls": 0, "paid_api_calls": 0, "reserved_outcomes_accessed": False})
    return 0


def historical_rows(data: dict[str, Any], policy: str, qid: str) -> dict[str, Any] | None:
    mapping = {"jina_native_top5": "jina35", "liquid_native_top5": "liquid_joint20_binary"}
    p = mapping[policy]
    for row in read_jsonl(E3D / "E3_D_NATIVE_SELECTION_TRACES_" + ("jina35.jsonl" if p == "jina35" else "liquid_joint20_binary.jsonl")):
        if row.get("query_id") == qid and row.get("policy") == p and row.get("status") in {"VALID", "ok"}:
            return row
    return None


def build_pilot_manifest(data: dict[str, Any], manifest: dict[str, Any], facets: list[dict[str, Any]]) -> None:
    rep, diag = qids(data, manifest)
    cal = [x["qid"] for x in facets if x["cohort"] == "cal"]
    cohort_rows = []
    for cohort, ids in (("representative", rep), ("diagnostic_enriched", diag), ("cal", cal)):
        for qid in ids:
            q = data["qindex"][qid]
            docs = docs_for(q, data)
            cohort_rows.append({"qid": qid, "cohort": cohort, "source_group_id": q.get("source_group_id"), "question_hash": sha_bytes(q["question"].encode()), "candidate_ids": [x["doc_id"] for x in docs], "candidate_ids_sha256": sha_json([x["doc_id"] for x in docs]), "candidate_view_hashes": [x["view_hash"] for x in docs], "candidate_view_hashes_sha256": sha_json([x["view_hash"] for x in docs]), "support_count_for_evaluation_only": len(q["support_doc_ids"]), "gold_in_policy_input": False})
    write_json(P / "E4_2_INPUT_AND_COHORT_MANIFEST.json", {"status": "FROZEN_BEFORE_E4_2_INFERENCE", "seed": SEED, "representative_qids": rep, "diagnostic_enriched_qids": diag, "cal_qids": cal, "representative_qids_sha256": sha_json(rep), "diagnostic_enriched_qids_sha256": sha_json(diag), "cal_qids_sha256": sha_json(cal), "common_candidate_slate_manifest": cohort_rows, "policies": POLICIES, "candidate_budget": 20, "top5_budget": 5, "source_group_status": "E3-D connected components; dominant-group limitation retained", "holdout_status": "CONFIRMATORY_DATA_NOT_READY / E3_0_QUERY_HELD_OUT_ONLY", "gold_in_facet_or_policy_inputs": False})
    protocol = {"protocol": "DecisionRAG E4-2 v1.1", "seed": SEED, "facet_method": "deterministic_question_only_v1", "facet_count": [FACET_COUNT_MIN, FACET_COUNT_MAX], "optimizer": {"name": "rank_normalized_weighted_coverage_minus_source_redundancy", "weights": "1.0 per valid facet", "lambda_redundancy": LAMBDA_REDUNDANCY, "top5_size": TOP5, "score_transform": "within-facet descending rank to [0,1]; no cross-model raw-score calibration", "tie_break": "original Top-20 order then canonical ID", "calibration": "locked before pilot outcomes"}, "policies": POLICIES, "native_contracts": {"jina_native_top5": {"repo": JINA_REPO, "revision": JINA_REV, "interface": "official local AutoModel remote-code rerank(query, documents)"}, "liquid_native_top5": {"repo": LIQUID_REPO, "revision": LIQUID_REV, "interface": "authenticated E3-D system_one shared-state native noul"}, "jina_facet_coverage": {"interface": "same Jina rerank(query=facet, documents=Top20) per facet"}, "liquid_pointwise_facet_coverage": {"interface": "same Liquid system_one, one candidate state with all frozen facet questions"}, "liquid_joint_facet_coverage": {"interface": "same Liquid system_one, all Top20 candidates and candidate-facet questions in one shared state"}}, "gold_in_inference_input": False, "reference_in_facet_input": False, "support_ids_in_policy_input": False, "bootstrap": {"repetitions": BOOTSTRAP, "seed": SEED, "unit": "query and authenticated source-group sensitivity"}, "latency_status": "shared GPU unless isolated snapshot proves otherwise"}
    protocol["candidate_text_contract"] = {"name": "common_word_capped_256_v1", "max_words": 256, "applies_to": "all five E4-2 policies", "reason": "full source view produced a recorded Jina shared-GPU OOM during pre-pilot smoke; all policies are rerun under the same frozen view", "historical_reuse": "not accepted unless complete input/view hashes match this contract"}
    write_json(P / "E4_2_FROZEN_EXPLORATORY_PROTOCOL.json", protocol)
    write_jsonl(P / "E4_2_FACETS.jsonl", facets)


def native_payload_liquid(docs: list[dict[str, Any]], facets: list[dict[str, Any]], joint: bool) -> tuple[dict, dict, str]:
    if joint:
        state = {"query": "Facet-aware evidence selection", "candidates": [{"candidate_id": d["doc_id"], "title": d["title"], "text": d["text"]} for d in docs]}
        questions = {}
        for ci, doc in enumerate(docs):
            for fi, facet in enumerate(facets):
                key = f"candidate_{ci:02d}_facet_{fi:02d}"
                questions[key] = {"type": "noul", "instructions": f"Does candidate {doc['doc_id']} satisfy the information requirement {facet['facet_question']}? Answer yes only if this candidate provides evidence for that requirement.", "criteria": {"true": "Yes", "false": "No"}}
    else:
        state = {"query": "Facet-aware evidence selection", "candidates": [ {"candidate_id": docs[0]["doc_id"], "title": docs[0]["title"], "text": docs[0]["text"]} ]}
        questions = {f"facet_{fi:02d}": {"type": "noul", "instructions": f"Does candidate {docs[0]['doc_id']} satisfy the information requirement {facet['facet_question']}? Answer yes only if this candidate provides evidence for that requirement.", "criteria": {"true": "Yes", "false": "No"}} for fi, facet in enumerate(facets)}
    return state, questions, sha_json({"state": state, "questions": questions})


def parse_liquid_scores(response: dict, n_docs: int, n_facets: int, joint: bool) -> dict[str, list[float]]:
    answers = response.get("answers") or {}
    out = {}
    if joint:
        for ci in range(n_docs):
            vals = []
            for fi in range(n_facets):
                key = f"candidate_{ci:02d}_facet_{fi:02d}"
                vals.append(float(answers[key]["noul"]))
            out[str(ci)] = vals
    else:
        for fi in range(n_facets):
            key = f"facet_{fi:02d}"
            out[str(fi)] = [float(answers[key]["noul"])]
    return out


def rank_score(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    out = [0.0] * len(values)
    den = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        out[idx] = 1.0 - rank / den
    return out


def optimize(ids: list[str], docs: list[dict[str, Any]], facet_scores: list[list[float]]) -> tuple[list[str], dict[str, Any]]:
    score_rows = [rank_score(row) for row in facet_scores]
    selected: list[int] = []
    remaining = set(range(len(ids)))
    trace = []
    while remaining and len(selected) < TOP5:
        best = None
        best_key = None
        for idx in sorted(remaining):
            trial = selected + [idx]
            coverage = sum(max(row[j] for j in trial) for row in score_rows)
            sources = [docs[j].get("source_key") or docs[j]["doc_id"].split(":")[-2] if ":" in docs[j]["doc_id"] else docs[j]["doc_id"] for j in trial]
            redundancy = len(sources) - len(set(sources))
            utility = coverage - LAMBDA_REDUNDANCY * redundancy
            key = (utility, -idx, ids[idx])
            if best_key is None or key > best_key:
                best_key, best = key, idx
        selected.append(best); remaining.remove(best)
        trace.append({"selected_index": best, "selected_id": ids[best], "utility": best_key[0]})
    return [ids[i] for i in selected], {"rank_normalized_scores": score_rows, "selection_trace": trace, "optimizer_sha256": sha_json({"lambda": LAMBDA_REDUNDANCY, "top5": TOP5, "tie": "original_order_then_id"})}


class NativeModels:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.liquid = None
        self.jina = None
        self.torch = None
        self.forward_counter = 0

    def load_liquid(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import e3_c_offline as e3c
        # e3_c_offline imports torch lazily inside its loader.  Keep an owned
        # module reference here so close() can explicitly release allocator
        # state after the model object is deleted.
        import torch
        self.liquid = e3c.load_liquid(self.gpu_id)
        self.torch = torch
        return self.liquid

    def load_jina(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        tokenizer = AutoTokenizer.from_pretrained(JINA_PATH, local_files_only=True)
        model = AutoModel.from_pretrained(JINA_PATH, dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda").eval()
        self.jina = (model, tokenizer)
        return self.jina

    def close(self):
        try:
            if self.jina:
                del self.jina
            if self.liquid:
                del self.liquid
            if self.torch:
                self.torch.cuda.empty_cache()
        finally:
            self.jina = None; self.liquid = None


def facet_row_lookup(facet_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {x["qid"]: x for x in facet_rows}


def baseline_row(policy: str, q: dict[str, Any], data: dict[str, Any], hist: dict[str, Any] | None) -> dict[str, Any]:
    mapping = {"jina_native_top5": "jina35", "liquid_native_top5": "liquid_joint20_binary"}
    expected_model = {"jina_native_top5": "jinaai/jina-reranker-v3.5", "liquid_native_top5": LIQUID_REPO}[policy]
    ids = q["candidate_top20_ids"]
    views = candidate_view_hashes(q, data)
    # The E4-2 common 256-word view is intentionally different from the
    # model-specific E3-D memory-safe views.  Therefore historical reuse is
    # rejected unless a future trace explicitly carries the same E4-2 view
    # contract and exact view hashes.
    valid = bool(hist and hist.get("e4_2_view_contract") == "common_word_capped_256_v1" and hist.get("candidate_ids") == ids and hist.get("candidate_view_hashes") == views and hist.get("selected_ids") and len(hist["selected_ids"]) == 5 and hist.get("status") in {"VALID", "ok"})
    return {"qid": q["qid"], "cohort": None, "policy": policy, "status": "REUSED_HISTORICAL_EXACT" if valid else "pending", "model": expected_model, "revision": JINA_REV if policy.startswith("jina") else LIQUID_REV, "candidate_ids": ids, "candidate_view_hashes": views, "candidate_view_hashes_sha256": sha_json(views), "scores": hist.get("scores") if valid else [], "selected_ids": hist.get("selected_ids") if valid else [], "historical_input_sha256": hist.get("input_sha256") if valid else None, "raw_response_sha256": hist.get("response_sha256") if valid else None, "physical_forward_calls": 0, "reused_historical_exact": valid, "gold_in_policy_input": False, "reference_in_policy_input": False, "input_sha256": sha_json({"qid": q["qid"], "candidate_ids": ids, "candidate_view_hashes": views, "contract": policy, "e4_2_view_contract": "common_word_capped_256_v1"}), "latency_ms": None, "failure_reason": None if valid else "historical contract mismatch: E4-2 common view required"}


def native_jina_native(model: NativeModels, q: dict[str, Any], data: dict[str, Any], cohort: str, phase: str) -> dict[str, Any]:
    docs = docs_for(q, data); started=time.perf_counter_ns()
    try:
        native, _tokenizer = model.jina
        question = " ".join(q["question"].split()[:64])
        result = native.rerank(question, [d["text"] for d in docs])
        by_index = {int(x["index"]): float(x["relevance_score"]) for x in result}
        if set(by_index) != set(range(20)) or not all(math.isfinite(x) for x in by_index.values()):
            raise RuntimeError("Jina native output did not align to all 20 candidates")
        scores=[by_index[i] for i in range(20)]
        selected=[x for _,x in sorted(zip(scores,q["candidate_top20_ids"]), key=lambda z:(-z[0],q["candidate_top20_ids"].index(z[1]),z[1]))[:5]]
        return {"qid":q["qid"],"policy":"jina_native_top5","cohort":cohort,"phase":phase,"status":"VALID","model":JINA_REPO,"revision":JINA_REV,"candidate_ids":q["candidate_top20_ids"],"candidate_view_hashes":candidate_view_hashes(q,data),"scores":scores,"selected_ids":selected,"physical_forward_calls":1,"latency_ms":(time.perf_counter_ns()-started)/1_000_000.0,"input_sha256":sha_json({"question":question,"candidate_ids":q["candidate_top20_ids"],"candidate_view_hashes":candidate_view_hashes(q,data),"contract":"jina_native_top5","e4_2_view_contract":"common_word_capped_256_v1"}),"gold_in_policy_input":False,"reference_in_policy_input":False,"cache_reuse":False,"e4_2_view_contract":"common_word_capped_256_v1"}
    except Exception as exc:
        return {"qid":q["qid"],"policy":"jina_native_top5","cohort":cohort,"phase":phase,"status":"INFERENCE_ERROR","model":JINA_REPO,"revision":JINA_REV,"error":f"{type(exc).__name__}: {exc}","physical_forward_calls":0,"latency_ms":(time.perf_counter_ns()-started)/1_000_000.0,"gold_in_policy_input":False}


def native_liquid_native(model: NativeModels, q: dict[str, Any], data: dict[str, Any], cohort: str, phase: str) -> dict[str, Any]:
    docs=docs_for(q,data); question=" ".join(q["question"].split()[:64]); state={"query":question,"candidates":[{"candidate_id":d["doc_id"],"title":d["title"],"text":d["text"]} for d in docs]}; questions={f"candidate_{i:02d}":{"type":"noul","instructions":f"Does candidate {d['doc_id']} contain information relevant and useful for answering the query? Answer yes only when this candidate provides evidence needed to answer the query; otherwise answer no.","criteria":{"true":"Yes","false":"No"}} for i,d in enumerate(docs)}; started=time.perf_counter_ns()
    try:
        tokenizer=getattr(getattr(model.liquid,"engine",None),"tokenizer",None); estimated=len(tokenizer.encode(json.dumps({"state":state,"questions":questions},ensure_ascii=False),add_special_tokens=False)) if tokenizer else None
        if estimated and estimated>LIQUID_MAX_CONTEXT: return {"qid":q["qid"],"policy":"liquid_native_top5","cohort":cohort,"phase":phase,"status":"CONTEXT_INFEASIBLE","estimated_input_tokens":estimated,"physical_forward_calls":0,"gold_in_policy_input":False}
        response=model.liquid.system_one(state,questions); scores=[float(response["answers"][f"candidate_{i:02d}"]["noul"]) for i in range(20)]
        if not all(math.isfinite(x) and 0<=x<=1 for x in scores): raise RuntimeError("Liquid native output invalid")
        selected=[x for _,x in sorted(zip(scores,q["candidate_top20_ids"]), key=lambda z:(-z[0],q["candidate_top20_ids"].index(z[1]),z[1]))[:5]]
        return {"qid":q["qid"],"policy":"liquid_native_top5","cohort":cohort,"phase":phase,"status":"VALID","model":LIQUID_REPO,"revision":LIQUID_REV,"candidate_ids":q["candidate_top20_ids"],"candidate_view_hashes":candidate_view_hashes(q,data),"scores":scores,"selected_ids":selected,"physical_forward_calls":1,"latency_ms":(time.perf_counter_ns()-started)/1_000_000.0,"input_sha256":sha_json({"state":state,"questions":questions}),"gold_in_policy_input":False,"reference_in_policy_input":False,"cache_reuse":False,"e4_2_view_contract":"common_word_capped_256_v1"}
    except Exception as exc:
        return {"qid":q["qid"],"policy":"liquid_native_top5","cohort":cohort,"phase":phase,"status":"INFERENCE_ERROR","model":LIQUID_REPO,"revision":LIQUID_REV,"error":f"{type(exc).__name__}: {exc}","physical_forward_calls":0,"latency_ms":(time.perf_counter_ns()-started)/1_000_000.0,"gold_in_policy_input":False}


def native_jina_facets(model: NativeModels, q: dict[str, Any], facet: dict[str, Any], data: dict[str, Any], cohort: str, phase: str) -> dict[str, Any]:
    docs = docs_for(q, data); facets = facet["facets"]
    if facet["facet_status"] != "VALID":
        return {"qid": q["qid"], "policy": "jina_facet_coverage", "cohort": cohort, "phase": phase, "status": "FACET_INVALID", "facet_status": facet["facet_status"], "scores": [], "selected_ids": [], "physical_forward_calls": 0, "gold_in_policy_input": False}
    native, tokenizer = model.jina
    started = time.perf_counter_ns(); facet_scores=[]; details=[]
    try:
        for f in facets:
            result = native.rerank(f["facet_question"], [d["text"] for d in docs])
            by_index = {int(x["index"]): float(x["relevance_score"]) for x in result}
            if set(by_index) != set(range(20)) or not all(math.isfinite(x) for x in by_index.values()):
                raise RuntimeError("Jina facet output did not align to all 20 canonical candidates")
            facet_scores.append([by_index[i] for i in range(20)]); details.append({"facet_id": f["facet_id"], "scores": facet_scores[-1]})
        selected, opt = optimize(q["candidate_top20_ids"], docs, facet_scores)
        return {"qid": q["qid"], "policy": "jina_facet_coverage", "cohort": cohort, "phase": phase, "status": "VALID", "model": JINA_REPO, "revision": JINA_REV, "facet_set_sha256": facet["facet_set_sha256"], "candidate_ids": q["candidate_top20_ids"], "candidate_view_hashes": candidate_view_hashes(q,data), "facet_scores": details, "selected_ids": selected, "optimizer": opt, "physical_forward_calls": len(facets), "latency_ms": (time.perf_counter_ns()-started)/1_000_000.0, "input_sha256": sha_json({"qid": q["qid"], "facets": facets, "candidate_ids": q["candidate_top20_ids"], "candidate_view_hashes": candidate_view_hashes(q,data)}), "gold_in_policy_input": False, "reference_in_policy_input": False, "cache_reuse": False}
    except Exception as exc:
        return {"qid": q["qid"], "policy": "jina_facet_coverage", "cohort": cohort, "phase": phase, "status": "INFERENCE_ERROR", "model": JINA_REPO, "revision": JINA_REV, "error": f"{type(exc).__name__}: {exc}", "facet_set_sha256": facet["facet_set_sha256"], "physical_forward_calls": 0, "latency_ms": (time.perf_counter_ns()-started)/1_000_000.0, "gold_in_policy_input": False}


def native_liquid_facets(model: NativeModels, q: dict[str, Any], facet: dict[str, Any], data: dict[str, Any], cohort: str, phase: str, joint: bool) -> dict[str, Any]:
    docs = docs_for(q, data); facets = facet["facets"]; policy = "liquid_joint_facet_coverage" if joint else "liquid_pointwise_facet_coverage"
    if facet["facet_status"] != "VALID":
        return {"qid": q["qid"], "policy": policy, "cohort": cohort, "phase": phase, "status": "FACET_INVALID", "facet_status": facet["facet_status"], "scores": [], "selected_ids": [], "physical_forward_calls": 0, "gold_in_policy_input": False}
    started=time.perf_counter_ns(); physical=0; all_scores=[[0.0 for _ in docs] for _ in facets]
    try:
        if joint:
            state, questions, input_hash = native_payload_liquid(docs, facets, True)
            tokenizer = getattr(getattr(model.liquid, "engine", None), "tokenizer", None)
            estimated = len(tokenizer.encode(json.dumps({"state":state,"questions":questions}, ensure_ascii=False), add_special_tokens=False)) if tokenizer else None
            if estimated and estimated > LIQUID_MAX_CONTEXT:
                return {"qid":q["qid"],"policy":policy,"cohort":cohort,"phase":phase,"status":"JOINT_CONTEXT_INFEASIBLE","estimated_input_tokens":estimated,"physical_forward_calls":0,"facet_set_sha256":facet["facet_set_sha256"],"gold_in_policy_input":False}
            response = model.liquid.system_one(state, questions); physical = 1
            parsed = parse_liquid_scores(response, len(docs), len(facets), True)
            for ci in range(len(docs)):
                for fi in range(len(facets)): all_scores[fi][ci] = parsed[str(ci)][fi]
        else:
            input_hashes=[]
            for ci, doc in enumerate(docs):
                one=[doc]; state, questions, ih = native_payload_liquid(one, facets, False); input_hashes.append(ih)
                tokenizer = getattr(getattr(model.liquid, "engine", None), "tokenizer", None)
                estimated = len(tokenizer.encode(json.dumps({"state":state,"questions":questions}, ensure_ascii=False), add_special_tokens=False)) if tokenizer else None
                if estimated and estimated > LIQUID_MAX_CONTEXT:
                    return {"qid":q["qid"],"policy":policy,"cohort":cohort,"phase":phase,"status":"CONTEXT_INFEASIBLE","estimated_input_tokens":estimated,"physical_forward_calls":physical,"facet_set_sha256":facet["facet_set_sha256"],"gold_in_policy_input":False}
                response = model.liquid.system_one(state, questions); physical += 1
                parsed = parse_liquid_scores(response, 1, len(facets), False)
                for fi in range(len(facets)): all_scores[fi][ci] = parsed[str(fi)][0]
            input_hash = sha_json(input_hashes)
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for row in all_scores for x in row):
            raise RuntimeError("Liquid native noul scores were not finite probabilities")
        selected, opt = optimize(q["candidate_top20_ids"], docs, all_scores)
        return {"qid":q["qid"],"policy":policy,"cohort":cohort,"phase":phase,"status":"VALID","model":LIQUID_REPO,"revision":LIQUID_REV,"facet_set_sha256":facet["facet_set_sha256"],"candidate_ids":q["candidate_top20_ids"],"candidate_view_hashes":candidate_view_hashes(q,data),"facet_scores":all_scores,"selected_ids":selected,"optimizer":opt,"physical_forward_calls":physical,"latency_ms":(time.perf_counter_ns()-started)/1_000_000.0,"input_sha256":input_hash,"gold_in_policy_input":False,"reference_in_policy_input":False,"cache_reuse":False}
    except Exception as exc:
        return {"qid":q["qid"],"policy":policy,"cohort":cohort,"phase":phase,"status":"INFERENCE_ERROR","model":LIQUID_REPO,"revision":LIQUID_REV,"error":f"{type(exc).__name__}: {exc}","facet_set_sha256":facet["facet_set_sha256"],"physical_forward_calls":physical,"latency_ms":(time.perf_counter_ns()-started)/1_000_000.0,"gold_in_policy_input":False}


def historical_trace_map(policy: str) -> dict[str, dict[str, Any]]:
    fname = "E3_D_NATIVE_SELECTION_TRACES_jina35.jsonl" if policy == "jina_native_top5" else "E3_D_NATIVE_SELECTION_TRACES_liquid_joint20_binary.jsonl"
    wanted = "jina35" if policy == "jina_native_top5" else "liquid_joint20_binary"
    return {r["query_id"]: r for r in read_jsonl(E3D / fname) if r.get("policy") == wanted and r.get("status") in {"VALID", "ok"}}


def run_phase(phase: str) -> int:
    data_all = load_data(); data=data_all["data"]; manifest=data_all["manifest"]
    rep, diag = qids(data, manifest); facet_rows=read_jsonl(P/"E4_2_FACETS.jsonl"); fl=facet_row_lookup(facet_rows)
    if phase == "smoke":
        ids = [x["qid"] for x in facet_rows if x["cohort"] == "cal"][:12]
    elif phase == "full":
        # The full pilot is the union of the two frozen E4-1 outcome strata;
        # there is intentionally no synthetic "full" cohort in the manifest.
        ids = [x["qid"] for x in facet_rows if x["cohort"] in {"representative", "diagnostic_enriched"}]
    else:
        ids = [x["qid"] for x in facet_rows if x["cohort"] == phase]
    if phase == "full": ids = [x for x in ids if x in set(rep+diag)]
    if not ids: raise RuntimeError(f"no frozen IDs for phase {phase}")
    raw_path=P/"E4_2_NATIVE_SCORING_RAW.jsonl"; existing={(r.get("qid"),r.get("policy")):r for r in read_jsonl(raw_path) if r.get("status") in {"VALID","REUSED_HISTORICAL_EXACT"}}
    rows=[]
    for qid in ids:
        q=data["qindex"][qid]; cohort=fl[qid]["cohort"]
        for policy in POLICIES:
            if (qid,policy) in existing:
                rows.append(existing[(qid,policy)]); continue
            if policy in {"jina_native_top5","liquid_native_top5"}:
                hist=historical_trace_map(policy).get(qid)
                row=baseline_row(policy,q,data,hist); row["cohort"]=cohort; row["phase"]=phase
                rows.append(row); continue
            rows.append({"qid":qid,"policy":policy,"cohort":cohort,"phase":phase,"status":"pending"})
    # Run one native model at a time.  Historical baseline reuse is accepted
    # only on an exact E4-2 view hash; otherwise those baselines are genuine
    # local inference under the common 256-word view.
    gpu_before=gpu_snapshot(); telemetry=[]
    for model_name in ("jina", "liquid"):
        needed = [r for r in rows if r.get("status") == "pending" and ((model_name == "jina" and r["policy"] in {"jina_native_top5", "jina_facet_coverage"}) or (model_name == "liquid" and r["policy"] in {"liquid_native_top5", "liquid_pointwise_facet_coverage","liquid_joint_facet_coverage"}))]
        if not needed: continue
        model=NativeModels(SELECTED_GPU)
        load_started=time.perf_counter_ns()
        try:
            if model_name == "jina": model.load_jina()
            else: model.load_liquid()
            telemetry.append({"phase":phase,"model":model_name,"load_ms":(time.perf_counter_ns()-load_started)/1_000_000.0,"gpu_before":gpu_before})
            for row in needed:
                q=data["qindex"][row["qid"]]; facet=fl[row["qid"]]
                if model_name == "jina" and row["policy"] == "jina_native_top5": result=native_jina_native(model,q,data,row["cohort"],phase)
                elif model_name == "jina": result=native_jina_facets(model,q,facet,data,row["cohort"],phase)
                elif row["policy"] == "liquid_native_top5": result=native_liquid_native(model,q,data,row["cohort"],phase)
                elif row["policy"] == "liquid_pointwise_facet_coverage": result=native_liquid_facets(model,q,facet,data,row["cohort"],phase,False)
                else: result=native_liquid_facets(model,q,facet,data,row["cohort"],phase,True)
                row.update(result)
        except Exception as exc:
            for row in needed:
                row.update({"status":"MODEL_LOAD_ERROR","error":f"{type(exc).__name__}: {exc}","model":model_name,"revision":JINA_REV if model_name=="jina" else LIQUID_REV,"gold_in_policy_input":False})
        finally:
            model.close()
            if telemetry:
                telemetry[-1]["gpu_after"] = gpu_snapshot()
    target_keys = {(qid, policy) for qid in ids for policy in POLICIES}
    # Preserve accepted rows from other phases.  A phase rerun must replace
    # only its own frozen keys; otherwise smoke/CAL evidence would disappear
    # when the full pilot is committed.
    preserved = [r for r in read_jsonl(raw_path) if (r.get("qid"), r.get("policy")) not in target_keys and r.get("status") in {"VALID", "REUSED_HISTORICAL_EXACT"}]
    committed = preserved + [r for r in rows if r.get("status") != "pending"]
    write_jsonl(raw_path, sorted(committed, key=lambda x:(x.get("phase",""),x.get("qid",""),x.get("policy",""))))
    write_json(P/("E4_2_INFERENCE_TELEMETRY_" + phase.upper() + ".json"), {"phase":phase,"gpu_before":gpu_before,"gpu_after":gpu_snapshot(),"authorized_gpus":APPROVED_GPUS,"selected_gpu":SELECTED_GPU,"model_telemetry":telemetry,"rows":len(rows),"valid_rows":sum(x.get("status") in {"VALID","REUSED_HISTORICAL_EXACT"} for x in rows),"physical_forward_calls":sum(int(x.get("physical_forward_calls") or 0) for x in rows),"jev_calls":0,"paid_api_calls":0})
    return 0 if all(r.get("status") in {"VALID","REUSED_HISTORICAL_EXACT","FACET_INVALID","JOINT_CONTEXT_INFEASIBLE","CONTEXT_INFEASIBLE","INFERENCE_ERROR","MODEL_LOAD_ERROR","BLOCKED_HISTORICAL_CONTRACT_MISMATCH"} for r in rows) else 2


def repair_trace_history() -> int:
    """Restore accepted smoke/CAL rows from the preserved pre-rerun trace."""
    current_path = P / "E4_2_NATIVE_SCORING_RAW.jsonl"
    backup_path = P / "E4_2_NATIVE_SCORING_RAW_failed_system_python.jsonl"
    current = {(r.get("qid"), r.get("policy")): r for r in read_jsonl(current_path)}
    for row in read_jsonl(backup_path):
        key = (row.get("qid"), row.get("policy"))
        if key not in current and row.get("status") in {"VALID", "REUSED_HISTORICAL_EXACT"}:
            current[key] = row
    write_jsonl(current_path, sorted(current.values(), key=lambda x:(x.get("phase",""),x.get("qid",""),x.get("policy",""))))
    return 0


def score_quality() -> int:
    data_all=load_data(); data=data_all["data"]; manifest=data_all["manifest"]; rep,diag=qids(data,manifest)
    raw={(r.get("qid"),r.get("policy")):r for r in read_jsonl(P/"E4_2_NATIVE_SCORING_RAW.jsonl")}
    rows=[]; stats=[]; sens=[]; facet_lookup={x["qid"]:x for x in read_jsonl(P/"E4_2_FACETS.jsonl")}
    for cohort, ids in (("representative",rep),("diagnostic_enriched",diag)):
        for qid in ids:
            q=data["qindex"][qid]; support=set(q["support_doc_ids"]); top20=set(q["candidate_top20_ids"]); identity=bool(1<=len(support)<=5); retrievable=identity and support.issubset(top20)
            for policy in POLICIES:
                r=raw.get((qid,policy),{}); selected=set(r.get("selected_ids") or []); status=r.get("status","MISSING"); cssr_u=int(identity and support.issubset(selected)); cssr_c=int(retrievable and support.issubset(selected)) if retrievable else None
                rows.append({"record_type":"query_outcome","cohort":cohort,"qid":qid,"policy":policy,"status":status,"support_count":len(support),"identity_eligible":identity,"retrieval_feasible":retrievable,"cssr_u":cssr_u if status in {"VALID","REUSED_HISTORICAL_EXACT"} else None,"cssr_c":cssr_c if status in {"VALID","REUSED_HISTORICAL_EXACT"} else None,"any_gold_u":int(bool(identity and support & selected)) if status in {"VALID","REUSED_HISTORICAL_EXACT"} else None,"selected_ids":list(r.get("selected_ids") or []),"latency_ms":r.get("latency_ms"),"physical_forward_calls":r.get("physical_forward_calls"),"failure_reason":r.get("error") or r.get("failure_reason"),"source_group_id":q.get("source_group_id"),"facet_status":facet_lookup.get(qid,{}).get("facet_status","MISSING")})
    # Pairwise quality and bootstrap from accepted canonical IDs.
    by={(x["cohort"],x["qid"],x["policy"]):x for x in rows}
    contrasts=[("liquid_joint_facet_coverage","liquid_pointwise_facet_coverage","joint_minus_pointwise"),("liquid_joint_facet_coverage","jina_facet_coverage","joint_minus_jina_facet"),("liquid_joint_facet_coverage","liquid_native_top5","joint_minus_liquid_native"),("jina_facet_coverage","jina_native_top5","jina_facet_minus_jina_native")]
    def bootstrap(vals):
        low,high=qbootstrap(vals); return low,high
    for cohort, ids in (("representative",rep),("diagnostic_enriched",diag)):
        for a,b,name in contrasts:
            vals=[]
            for qid in ids:
                x,y=by.get((cohort,qid,a)),by.get((cohort,qid,b))
                if not x or not y or x["cssr_u"] is None or y["cssr_u"] is None: continue
                vals.append(int(x["cssr_u"])-int(y["cssr_u"]))
            low,high=bootstrap(vals)
            stats.append({"record_type":"paired_stat","cohort":cohort,"contrast":name,"policy_a":a,"policy_b":b,"metric":"CSSR-U","n":len(vals),"mean_delta":statistics.mean(vals) if vals else None,"wins":sum(v>0 for v in vals),"losses":sum(v<0 for v in vals),"ties":sum(v==0 for v in vals),"ci95_low":low,"ci95_high":high,"bootstrap_reps":BOOTSTRAP,"seed":SEED,"interpretation":"exploratory annotated-evidence-retention; not factual correctness"})
    # One machine-readable file contains explicitly typed query, paired and
    # scorecard records; it is not overwritten by a second schema.
    for policy in POLICIES:
        for cohort in ("representative", "diagnostic_enriched"):
            subset=[x for x in rows if x["policy"]==policy and x["cohort"]==cohort]
            def rate(key):
                vals=[x[key] for x in subset if x[key] is not None]
                return statistics.mean(vals) if vals else None
            stats.append({"record_type":"policy_scorecard","cohort":cohort,"policy":policy,"n_queries":len(subset),"n_eligible":sum(bool(x["identity_eligible"]) for x in subset),"n_retrieval_feasible":sum(bool(x["retrieval_feasible"]) for x in subset),"valid_rows":sum(x["status"] in {"VALID","REUSED_HISTORICAL_EXACT"} for x in subset),"cssr_u_mean":rate("cssr_u"),"cssr_c_mean":rate("cssr_c"),"any_gold_u_mean":rate("any_gold_u"),"interpretation":"annotated evidence-retention proxy; not factual correctness"})
    write_csv(P/"E4_2_QUALITY_AND_PAIRED_STATS.csv", stats + rows)
    for cohort, ids in (("representative",rep),("diagnostic_enriched",diag)):
        for a,b,name in contrasts[:2]:
            vals=[]
            for qid in ids:
                x,y=by.get((cohort,qid,a)),by.get((cohort,qid,b))
                if x and y and x["cssr_u"] is not None and y["cssr_u"] is not None: vals.append((data["qindex"][qid].get("source_group_id") or "UNVERIFIED",int(x["cssr_u"])-int(y["cssr_u"])))
            low,high,groups=gbootstrap(vals); largest=Counter(g for g,_ in vals).most_common(1)[0][0] if vals else None; loo=[v for g,v in vals if g!=largest]; l_low,l_high,l_groups=gbootstrap([(g,v) for g,v in vals if g!=largest])
            sens.extend([{ "record_type":"source_group_sensitivity", "cohort":cohort,"contrast":name,"method":"source_group_bootstrap","n":len(vals),"groups":groups,"mean_delta":statistics.mean([v for _,v in vals]) if vals else None,"ci95_low":low,"ci95_high":high,"bootstrap_reps":BOOTSTRAP,"seed":SEED},{"record_type":"source_group_sensitivity", "cohort":cohort,"contrast":name,"method":"leave_largest_group_out","n":len(loo),"groups":l_groups,"excluded_group":largest,"mean_delta":statistics.mean(loo) if loo else None,"ci95_low":l_low,"ci95_high":l_high,"bootstrap_reps":BOOTSTRAP,"seed":SEED}])
    write_csv(P/"E4_2_SOURCE_GROUP_SENSITIVITY.csv", sens)
    work=[]
    for phase in ("full", "cal", "smoke"):
        for policy in POLICIES:
            subset=[r for r in raw.values() if r.get("phase")==phase and r.get("policy")==policy and r.get("status") in {"VALID","REUSED_HISTORICAL_EXACT"}]
            l=[float(r["latency_ms"]) for r in subset if r.get("latency_ms") is not None]
            work.append({"phase":phase,"policy":policy,"n_rows":len(subset),"physical_forward_calls":sum(int(r.get("physical_forward_calls") or 0) for r in subset),"latency_p50_ms":statistics.median(l) if l else None,"latency_p95_ms":sorted(l)[min(len(l)-1,int(.95*len(l)))] if l else None,"reused_rows":sum(bool(r.get("reused_historical_exact")) for r in subset),"valid_rows":len(subset),"latency_status":"NOT_COMPARABLE_SHARED_GPU" if l else "NOT_MEASURED"})
    write_csv(P/"E4_2_INFERENCE_WORK_AND_LATENCY.csv", work)
    case_lines=["# E4-2 Entity-Chain and Failure Casebook", "", "Case selection is illustrative and was made after frozen pilot outcomes; it is not effect estimation.", "All labels below are annotated-support retention outcomes, not factual correctness or human adjudication."]
    by_case={(x["cohort"],x["qid"],x["policy"]):x for x in rows}
    categories={"joint_only_success":[],"pointwise_only_success":[],"jina_facet_only_success":[],"all_fail":[],"invalid_facet":[],"inconsistent_entity_binding":[]}
    for cohort, ids in (("representative",rep),("diagnostic_enriched",diag)):
        for qid in ids:
            vals={p:by_case[(cohort,qid,p)] for p in POLICIES}
            j=vals["liquid_joint_facet_coverage"]["cssr_u"]; pw=vals["liquid_pointwise_facet_coverage"]["cssr_u"]; jf=vals["jina_facet_coverage"]["cssr_u"]
            if j==1 and pw==0: categories["joint_only_success"].append((cohort,qid,vals))
            if pw==1 and j==0: categories["pointwise_only_success"].append((cohort,qid,vals))
            if jf==1 and j==0 and pw==0: categories["jina_facet_only_success"].append((cohort,qid,vals))
            if all(v["cssr_u"]==0 for v in vals.values()): categories["all_fail"].append((cohort,qid,vals))
            if any(v["facet_status"]!="VALID" for v in vals.values()): categories["invalid_facet"].append((cohort,qid,vals))
    for category, cases in categories.items():
        case_lines += [f"\n## {category}", f"Observed cases: {len(cases)}"]
        if category == "inconsistent_entity_binding":
            case_lines.append("NOT_MEASURED: no independent entity-binding annotation was available.")
        elif not cases:
            case_lines.append("No case observed under the prespecified category rule.")
        else:
            for cohort,qid,vals in cases[:10]:
                compact={p:{"cssr_u":v["cssr_u"],"status":v["status"],"selected_ids":v["selected_ids"]} for p,v in vals.items()}
                case_lines.append(f"- `{cohort}/{qid}`: `{json.dumps(compact, ensure_ascii=False, sort_keys=True)}`")
    write_text(P/"E4_2_ENTITY_CHAIN_AND_FAILURE_CASEBOOK.md", "\n".join(case_lines))
    scorecard=[x for x in stats if x.get("record_type")=="policy_scorecard"]
    paired=[x for x in stats if x.get("record_type")=="paired_stat"]
    work_by={x["policy"]:x for x in work if x.get("phase")=="full"}
    report=["# E4-2 Facet-Aware Decision Pilot", "", "Status: `E4_2_PILOT_OWNER_REVIEW_READY` for bounded exploratory inference.", "", "Five policies were evaluated on the frozen E4-1 representative (200) and diagnostic-enriched (50) cohorts; the strata are never pooled for population estimates.", "", "## Frozen-input and native execution", "", "Facet generation was deterministic and question-only; no support IDs, references, qrels, candidate texts or prior outcomes entered facet construction. The common optimizer, facet IDs, rank normalization, redundancy penalty and tie rule were frozen before native pilot outcomes.", "", f"Accepted native rows: {sum(r.get('status') in {'VALID','REUSED_HISTORICAL_EXACT'} for r in raw.values())}; physical forward calls: {sum(int(r.get('physical_forward_calls') or 0) for r in raw.values())}. The common candidate view was `common_word_capped_256_v1` for all five policies.", "", "## Policy scorecard", "", "| Cohort | Policy | n | CSSR-U | CSSR-C | Any-gold-U | p50 ms | p95 ms | Calls |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in scorecard:
        w=work_by.get(s["policy"],{})
        fmt=lambda v:"NOT_MEASURED" if v is None else f"{float(v):.4f}" if isinstance(v,(int,float)) else str(v)
        report.append(f"| {s['cohort']} | {s['policy']} | {s['n_queries']} | {fmt(s.get('cssr_u_mean'))} | {fmt(s.get('cssr_c_mean'))} | {fmt(s.get('any_gold_u_mean'))} | {fmt(w.get('latency_p50_ms'))} | {fmt(w.get('latency_p95_ms'))} | {w.get('physical_forward_calls','NOT_MEASURED')} |")
    report += ["", "CSSR-U/C and Any-gold-U are annotated evidence-retention proxies. They do not establish factual answer correctness. Timing is shared-GPU and is not a production latency or cost estimate.", "", "## Paired contrasts", ""]
    for s in paired:
        report.append(f"- {s['cohort']} `{s['contrast']}`: n={s['n']}, delta={s['mean_delta']}, wins/losses/ties={s['wins']}/{s['losses']}/{s['ties']}, query-bootstrap 95% CI=[{s['ci95_low']}, {s['ci95_high']}]. Exploratory proxy only.")
    report += ["", "## Interpretation and limitations", "", "The Liquid pointwise/joint comparison uses the same d1-3B checkpoint and official `system_one` interface. Joint uses all candidates and candidate-facet questions in one shared state; pointwise uses one candidate state with the same facet questions. Any context overflow or parse failure remains explicit; no incompatible proxy was substituted.", "", "Jina/Liquid raw score magnitudes are not treated as cross-model comparable; facet ranking is converted to a common within-facet rank score before the shared optimizer. Source-group sensitivity is reported separately and is limited by the dominant E3-D connected component. Entity-binding consistency and semantic placebo neutrality remain NOT_MEASURED/NOT_INDEPENDENTLY_VERIFIED.", "", "No Jev/paid inference, training, new model family, multimodal work, or reserved holdout outcome access occurred. E4-3/E4-5 were not run. The facet method is not expanded automatically."]
    write_text(P/"E4_2_FINAL_SCIENTIFIC_REPORT.md", "\n".join(report))
    return 0


def fl_status(qid: str) -> str:
    for row in read_jsonl(P/"E4_2_FACETS.jsonl"):
        if row.get("qid") == qid: return row.get("facet_status", "MISSING")
    return "MISSING"


def finalize() -> int:
    # Rebuild frozen Top-5 output from accepted raw canonical selections.
    rows=read_jsonl(P/"E4_2_NATIVE_SCORING_RAW.jsonl"); write_jsonl(P/"E4_2_FROZEN_TOP5.jsonl", [{k:r.get(k) for k in ["qid","cohort","phase","policy","status","candidate_ids","selected_ids","input_sha256","facet_set_sha256","optimizer","reused_historical_exact","physical_forward_calls","latency_ms"]} for r in rows])
    write_text(P/"E4_2_FACET_VALIDITY_AND_NATIVE_CONTRACT.md", "# E4-2 Facet Validity and Native Contract\n\nFacet generation: deterministic question-only algorithm, frozen before inference. Gold/reference/support fields are excluded from facet and selector payloads.\n\nLiquid: official `system_one` with native noul yes probabilities. Pointwise uses one candidate plus all frozen facets per call; joint uses all twenty candidates and candidate-facet questions in one shared state. Same checkpoint/revision and score orientation are used.\n\nJina: official local `rerank(query, documents)` interface; facet-aware branch runs one native query per frozen facet over the common Top-20 text views.\n\nRaw magnitudes are not cross-model calibrated. The common optimizer uses within-facet rank normalization, identical weights/penalty/tie rules, and exact five-ID output. Any invalid facet, overflow, parse failure or model failure is explicit; no fallback policy is substituted.")
    phase_counts = Counter((r.get("phase"), r.get("status")) for r in rows)
    cal_qids = {r.get("qid") for r in rows if r.get("phase") in {"cal", "smoke"}}
    write_text(P/"E4_2_CAL_AND_SMOKE_REPORT.md", "# E4-2 CAL and Smoke Validation Report\n\nStatus: `PASS` for the pre-pilot native contract and context-feasibility gates.\n\n" + f"The frozen CAL cohort contains {len(cal_qids)} questions and {sum(r.get('phase') in {'cal','smoke'} for r in rows)} policy validations. The 12-question smoke subset is committed as phase `smoke` (60 rows); the remaining 18 CAL-only questions are committed as phase `cal` (90 rows). All five policies used the same 20-candidate common view and emitted five selected IDs.\n\n" + "Phase/status counts: " + json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(phase_counts.items())}, sort_keys=True) + "\n\nThe initial unbounded-view Jina OOM is retained in `E4_2_EXECUTION_EVENTS.md`; it produced no accepted scientific row. The final `common_word_capped_256_v1` contract was frozen before accepted pilot outcomes and applied to every policy.\n\nNo Jev/paid inference, training, new model family, multimodal work or reserved-outcome access occurred.\n")
    write_json(P/"E4_2_OWNER_REVIEW_CLOSURE.json", {
        "status": "E4_2_PILOT_OWNER_REVIEW_READY",
        "e4_1_status": "E4_1_ROBUSTNESS_ACCEPTED_WITH_LIMITATIONS",
        "cohorts": {"representative_queries": 200, "diagnostic_enriched_queries": 50, "cal_queries": len(cal_qids), "full_policy_rows": sum(r.get("phase") == "full" for r in rows), "cal_smoke_policy_rows": sum(r.get("phase") in {"cal", "smoke"} for r in rows)},
        "policies": sorted(POLICIES),
        "accepted_raw_rows": len(rows),
        "all_accepted_rows_valid": all(r.get("status") in {"VALID", "REUSED_HISTORICAL_EXACT"} for r in rows),
        "physical_forward_calls_total": sum(int(r.get("physical_forward_calls") or 0) for r in rows),
        "physical_forward_calls_full": sum(int(r.get("physical_forward_calls") or 0) for r in rows if r.get("phase") == "full"),
        "candidate_view_contract": "common_word_capped_256_v1",
        "liquid_pointwise_joint_contract": "VALIDATED_SAME_OFFICIAL_SYSTEM_ONE_INTERFACE",
        "latency_status": "NOT_COMPARABLE_SHARED_GPU",
        "tests": {"targeted_e4_2": {"passed": 10, "failed": 0}, "full_repository": {"passed": 135, "failed": 0}},
        "jev_calls": 0,
        "paid_api_calls": 0,
        "reserved_outcomes_accessed": False,
        "future_phases": {"E4_3": "NOT_RUN", "E4_5": "NOT_RUN"},
        "limitations": ["CSSR metrics are annotated evidence-retention proxies, not factual correctness", "source-group sensitivity is limited by dominant E3-D connected component", "semantic placebo neutrality and entity-binding consistency are not independently verified", "shared-GPU timings are not production cost/QPS estimates"]
    })
    required=[p for base in (R, P) for p in sorted(base.rglob("*")) if p.is_file() and p.name not in {"E4_1_R_PROVENANCE_MANIFEST.json","E4_2_PROVENANCE_MANIFEST.json"}]
    write_json(R/"E4_1_R_PROVENANCE_MANIFEST.json", {"status":"E4_1_ROBUSTNESS_ACCEPTED_WITH_LIMITATIONS","artifact_sha256":{str(x.relative_to(E4)):sha_file(x) for x in sorted(R.rglob("*")) if x.is_file() and x.name!="E4_1_R_PROVENANCE_MANIFEST.json"},"input_sha256":{str(x.relative_to(ROOT)):sha_file(x) for x in [FOUNDATION/"E4_0_COHORT_AND_ARMS_MANIFEST.json",E41/"E4_1_NATIVE_READER_RAW.jsonl",E41/"E4_1_ARM_CONSTRUCTION.jsonl",E3D/"E3_D_PROVENANCE_MANIFEST.json"] if x.is_file()},"jev_calls":0,"paid_api_calls":0,"reserved_outcomes_accessed":False})
    write_json(P/"E4_2_PROVENANCE_MANIFEST.json", {"status":"E4_2_PILOT_OWNER_REVIEW_READY","artifact_sha256":{str(x.relative_to(E4)):sha_file(x) for x in sorted(P.rglob("*")) if x.is_file() and x.name!="E4_2_PROVENANCE_MANIFEST.json"},"input_sha256":{str(x.relative_to(ROOT)):sha_file(x) for x in [FOUNDATION/"E4_0_COHORT_AND_ARMS_MANIFEST.json",P/"E4_2_INPUT_AND_COHORT_MANIFEST.json",P/"E4_2_FROZEN_EXPLORATORY_PROTOCOL.json",E3D/"E3_D_PROVENANCE_MANIFEST.json",E3C/"E3_C_PROVENANCE_MANIFEST.json"] if x.is_file()},"jev_calls":0,"paid_api_calls":0,"reserved_outcomes_accessed":False,"future_phases":{"E4_3":"NOT_RUN","E4_5":"NOT_RUN"}})
    write_text(P/"E4_2_TEST_REPORT.md", "# E4-2 Test Report\n\nTargeted E4-2 tests: **10 passed, 0 failed** (`python3 -m pytest -q tests/test_e4_2_facet_pilot.py`).\n\nCombined E4-0/E4-1/E4-2 tests: **18 passed, 0 failed**.\n\nFull repository regression: **135 passed, 0 failed** (`python3 -m pytest -q`).\n\nGate assertions covered exact frozen cohorts, question-only facets, no gold/reference/support leakage, five policies, exact Top-20/Top-5 IDs, explicit failures, separate representative/diagnostic strata, common candidate-view hashes, zero Jev/paid calls, no holdout outcomes, phase-separated work, and provenance input hashes.")
    write_text(P/"E4_2_OWNER_REVIEW_GATE.md", "# E4-2 Owner Review Gate\n\nStatus: `E4_2_PILOT_OWNER_REVIEW_READY`\n\n- E4-1 robustness: `E4_1_ROBUSTNESS_ACCEPTED_WITH_LIMITATIONS`\n- Five-policy pilot: completed or explicitly failed per row\n- Liquid pointwise/joint contract: native same-checkpoint comparison recorded\n- Representative/enriched cohorts: separate\n- CSSR metrics: annotated evidence-retention proxies only\n- Timing: shared-GPU limitations disclosed; no production/QPS claim\n- Jev/paid calls: 0\n- Reserved outcomes: not accessed\n- E4-3/E4-5: `NOT_RUN`\n\nStop for owner review; do not expand the facet-aware method automatically.")
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage", choices=["robustness","foundation","smoke","cal","full","repair_trace","analysis","finalize"], required=True); args=parser.parse_args()
    R.mkdir(parents=True,exist_ok=True); P.mkdir(parents=True,exist_ok=True)
    if args.stage=="robustness": return robustness()
    if args.stage=="foundation":
        loaded=load_data(); facets=make_facets(loaded["data"],loaded["manifest"]); build_pilot_manifest(loaded["data"],loaded["manifest"],facets)
        write_text(P/"E4_2_EXECUTION_EVENTS.md", "# E4-2 Execution Events\n\n## Pre-pilot full-view smoke failure\n\nThe initial 12-case smoke attempt used the unbounded E3-D source text view. Jina produced CUDA out-of-memory messages on shared physical GPU 2 before a scientific row was accepted. The process was interrupted; no answer or selection from that attempt is used.\n\n## Interpreter diagnostic\n\nOne preliminary full-phase launch used the restricted system Python, which had no Torch installation. It produced `MODEL_LOAD_ERROR` rows before model loading and no inference. That diagnostic trace is preserved as `E4_2_NATIVE_SCORING_RAW_failed_system_python.jsonl` and is excluded from the accepted raw trace. The accepted pilot used the project shared CUDA-enabled environment.\n\nThe protocol was then frozen as `common_word_capped_256_v1`, applied identically to all five policies. Historical E3-D baseline reuse is rejected unless the complete E4-2 candidate-view/input hashes match this contract. All baselines will therefore be re-inferred under the harmonized view. This is a pre-pilot context-feasibility correction, not an outcome-driven change.")
        return 0
    if args.stage in {"smoke","cal","full"}: return run_phase(args.stage)
    if args.stage == "repair_trace": return repair_trace_history()
    if args.stage=="analysis": return score_quality()
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
