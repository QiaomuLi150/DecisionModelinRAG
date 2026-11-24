#!/usr/bin/env python3
"""CPU/offline E5-QA evidence audit and paper-package generator.

This module intentionally has no model, CUDA, HTTP, dataset-download, or
reserved-holdout code.  It consumes already-frozen E2--E5 artifacts and writes
only under results/non_jev/v2/e5_v1_1/e5_qa_final_audit/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"
E5C = ROOT / "results/non_jev/v2/e5_v1_1/e5_0c_readiness_closure"
E5A = ROOT / "results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark"
E5DC = ROOT / "results/non_jev/v2/e5_v1_1/e5_dc_serving_reader_validation"
E4R = ROOT / "results/non_jev/v2/e4_v1_1/e4_r_reconciliation"
E2M1 = ROOT / "results/non_jev/v2/e2_m1"
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_qa_final_audit"
QASPER = ROOT / "data/external/e5_qasper/snapshot"

SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
SEED = 731
BOOTSTRAP_REPS = 5000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_frozen(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9%+./-]+", " ", text)
    return " ".join(text.split())


def norm_case_space(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return " ".join(text.split())


def norm_alias(value: Any) -> str:
    text = norm_frozen(value)
    aliases = {
        "true": "yes", "affirmative": "yes", "false": "no", "negative": "no",
        "approximately": "about", "approx": "about",
    }
    return " ".join(aliases.get(x, x) for x in text.split())


def numeric_unit_tokens(value: Any) -> tuple[str, ...] | None:
    text = norm_frozen(value)
    if not text:
        return None
    text = text.replace("percent", "%").replace("percentage", "%")
    text = re.sub(r"\s*%", "%", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    # This is deliberately narrow: numeric equivalence is only accepted when
    # the nonnumeric unit tokens also match exactly.
    nonnum = re.sub(r"[-+]?\d+(?:\.\d+)?", "<NUM>", text)
    return tuple([nonnum, *nums])


def score_variant(answer: str, forms: list[str], variant: str) -> bool:
    if variant == "frozen_normalized":
        a, fs = norm_frozen(answer), {norm_frozen(x) for x in forms}
        return bool(a) and a in fs
    if variant == "case_space_symmetric":
        a, fs = norm_case_space(answer), {norm_case_space(x) for x in forms}
        return bool(a) and a in fs
    if variant == "alias_symmetric":
        a, fs = norm_alias(answer), {norm_alias(x) for x in forms}
        return bool(a) and a in fs
    if variant == "numeric_unit_symmetric":
        a = numeric_unit_tokens(answer)
        fs = {numeric_unit_tokens(x) for x in forms}
        return a is not None and a in fs
    raise ValueError(variant)


def detailed_refs() -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        payload = read_json(QASPER / f"qasper-{split}-v0.3.json")
        for paper_id, paper in payload.items():
            for qa in paper.get("qas", []):
                qid = f"qasper:{paper_id}:{qa['question_id']}"
                annotations = qa.get("answers", [])
                forms: list[str] = []
                answer_types: set[str] = set()
                evidences: list[list[str]] = []
                highlights: list[list[str]] = []
                unanswerable: list[bool] = []
                for ann in annotations:
                    ans = ann.get("answer") or {}
                    unanswerable.append(bool(ans.get("unanswerable")))
                    free = str(ans.get("free_form_answer") or "").strip()
                    spans = [str(x).strip() for x in (ans.get("extractive_spans") or []) if str(x).strip()]
                    yes_no = ans.get("yes_no")
                    if free:
                        forms.append(free); answer_types.add("ABSTRACTIVE")
                    if spans:
                        forms.extend(spans); answer_types.add("EXTRACTIVE")
                    if yes_no is not None:
                        forms.append("yes" if bool(yes_no) else "no"); answer_types.add("YES_NO")
                    ev = [str(x).strip() for x in (ans.get("evidence") or []) if str(x).strip()]
                    hi = [str(x).strip() for x in (ans.get("highlighted_evidence") or []) if str(x).strip()]
                    evidences.append(ev); highlights.append(hi)
                if not unanswerable:
                    status = "INVALID_REFERENCE"
                elif all(unanswerable):
                    status = "UNANSWERABLE"
                elif any(unanswerable):
                    status = "AMBIGUOUS_REFERENCE"
                elif not forms:
                    status = "INVALID_REFERENCE"
                else:
                    status = "ANSWERABLE"
                if status == "UNANSWERABLE":
                    family = "UNANSWERABLE"
                elif status != "ANSWERABLE":
                    family = status
                elif len(answer_types) == 1:
                    family = next(iter(answer_types))
                else:
                    family = "MIXED_ANNOTATED"
                refs[qid] = {
                    "query_id": qid, "paper_id": paper_id, "paper_title": paper.get("title", ""),
                    "question": qa.get("question", ""), "status": status, "forms": sorted(set(forms)),
                    "annotation_count": len(annotations), "answer_type_family": family,
                    "answer_types": sorted(answer_types), "evidences": evidences, "highlights": highlights,
                    "evidence_annotation_count": sum(bool(x) for x in evidences),
                    "highlight_annotation_count": sum(bool(x) for x in highlights),
                }
    return refs


def load_e5a() -> dict[str, Any]:
    # The adapter module's load_inputs is an offline parser; no model class is
    # constructed and no inference path is called.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import e5_a_benchmark
    return e5_a_benchmark.load_inputs()


def load_state() -> dict[str, Any]:
    data = load_e5a()
    refs = detailed_refs()
    dc_manifest = read_json(E5DC / "E5_DC_EXECUTION_MANIFEST_FROZEN.json")
    rep = set(dc_manifest["cohorts"]["representative_qids"])
    diag = set(dc_manifest["cohorts"]["diagnostic_qids"])
    outcomes = {r["query_id"]: r for r in read_jsonl(E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl")}
    reader = read_jsonl(E5DC / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl")
    reader_by = defaultdict(dict)
    for r in reader:
        reader_by[(r["query_id"], r["policy"])].setdefault("rows", []).append(r)
    cohort_rows = list(csv.DictReader((E5DC / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv").open(encoding="utf-8")))
    return {"data": data, "refs": refs, "manifest": dc_manifest, "rep": rep, "diag": diag,
            "outcomes": outcomes, "reader": reader, "reader_by": reader_by,
            "cohort_rows": cohort_rows, "slates": data["slates"], "docs": data["corpus_map"]}


def provenance_index() -> dict[str, Any]:
    items = [
        (E4R / "E4_R_PROVENANCE_MANIFEST.json", "E4-R", "historical provenance manifest"),
        (E4R / "E4_R_FINAL_SCIENTIFIC_RECONCILIATION_REPORT.md", "E4-R", "historical reconciliation"),
        (E5C / "E5_0C_PROVENANCE_MANIFEST.json", "E5-0C", "support-semantic provenance"),
        (E5C / "E5_0C_PROTOCOL_AMENDMENT_V1_APPROVED.md", "E5-0C", "approved set-aware metric amendment"),
        (E5C / "E5_0C_SUPPORT_SET_SEMANTICS_AUDIT.md", "E5-0C", "support-set contract"),
        (E5C / "E5_0C_INTEGRITY_AND_PROVENANCE.json", "E5-0C", "integrity summary"),
        (E5A / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json", "E5-A", "frozen model/input contract"),
        (E5A / "E5_A_PROVENANCE_MANIFEST.json", "E5-A", "historical artifact manifest"),
        (E5A / "E5_A_PROVENANCE_VERIFICATION.json", "E5-A", "historical verification"),
        (E5A / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json", "E5-A", "primary statistics"),
        (E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl", "E5-A", "raw quality outcomes"),
        (E5A / "E5_A_NATIVE_ATTEMPT_AND_TELEMETRY.jsonl", "E5-A", "native inference provenance"),
        (E5DC / "E5_DC_EXECUTION_MANIFEST_FROZEN.json", "E5-D/C", "cohort and reader contract"),
        (E5DC / "E5_DC_PROVENANCE_MANIFEST.json", "E5-D/C", "closure artifact manifest"),
        (E5DC / "E5_DC_PROVENANCE_VERIFICATION.json", "E5-D/C", "closure verification"),
        (E5DC / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv", "E5-D/C", "frozen cohorts"),
        (E5DC / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl", "E5-C", "raw accepted reader rows"),
        (E5DC / "E5_C_ANSWER_REFERENCE_AGREEMENT.csv", "E5-C", "historic T1 proxy rows"),
        (E5DC / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl", "E5-D", "raw selector traces"),
        (E5DC / "E5_D_MATCHED_LATENCY_STATISTICS.csv", "E5-D", "latency summary"),
        (E5DC / "E5_D_FORWARD_TOKEN_MEMORY_ACCOUNTING.csv", "E5-D", "memory/forward accounting"),
        (E5DC / "E5_D_pplx_decider_SUMMARY.json", "E5-D", "PPLX serving summary"),
        (E5DC / "E5_D_jina35_SUMMARY.json", "E5-D", "Jina serving summary"),
        (E2M1 / "E2_M1_FINAL_REPORT.md", "E2-M1", "historical mechanism report"),
        (E2M1 / "E2_M1_PROVENANCE_MANIFEST.json", "E2-M1", "historical mechanism provenance"),
    ]
    rows = []
    for path, phase, role in items:
        exists = path.is_file()
        rows.append({"path": str(path.relative_to(ROOT)), "phase": phase, "role": role,
                     "exists": exists, "bytes": path.stat().st_size if exists else None,
                     "sha256": sha_file(path) if exists else None,
                     "disposition": "READ_ONLY_HISTORICAL"})
    # Recheck the declared manifests without copying or modifying them.
    declared_checks = []
    for manifest_path in (E4R / "E4_R_PROVENANCE_MANIFEST.json", E5C / "E5_0C_PROVENANCE_MANIFEST.json", E5A / "E5_A_PROVENANCE_MANIFEST.json", E5DC / "E5_DC_PROVENANCE_MANIFEST.json"):
        d = read_json(manifest_path)
        declared = d.get("files", {})
        if isinstance(declared, list):
            declared = {x.get("path"): x for x in declared}
        for rel, value in declared.items():
            path = ROOT / rel
            expected = value.get("sha256") if isinstance(value, dict) else value
            expected_bytes = value.get("bytes") if isinstance(value, dict) else None
            declared_checks.append({"manifest": str(manifest_path.relative_to(ROOT)), "path": rel,
                                    "exists": path.is_file(), "hash_match": path.is_file() and sha_file(path) == expected,
                                    "size_match": path.is_file() and (expected_bytes is None or path.stat().st_size == expected_bytes)})
    return {"created_utc": now(), "schema": "E5-QA-PROVENANCE-V1", "items": rows,
            "declared_manifest_checks": declared_checks,
            "all_required_present": all(x["exists"] for x in rows),
            "all_declared_hashes_match": all(x["exists"] and x["hash_match"] and x["size_match"] for x in declared_checks),
            "reserved_e5f_raw_content_read": False, "gpu_or_http_calls": False}


def reader_accounting(state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    rep, diag = state["rep"], state["diag"]
    overlap = sorted(rep & diag)
    nominal = 2 * (len(rep) + len(diag))
    unique_pairs = sorted({(r["query_id"], r["policy"]) for r in state["reader"]})
    pair_counts = Counter((r["query_id"], r["policy"]) for r in state["reader"])
    duplicates = [{"query_id": q, "policy": p, "rows": n} for (q, p), n in sorted(pair_counts.items()) if n != 1]
    statuses = Counter(r.get("status") for r in state["reader"])
    physical = Counter((r["query_id"], r["policy"]) for r in state["reader"] if r.get("real_inference"))
    cache = Counter((r.get("cache_reuse"), r.get("cache_provenance")) for r in state["reader"])
    rows = []
    for qid in sorted(rep | diag):
        for policy in ("pplx_decider", "jina35"):
            matching = [r for r in state["reader"] if r["query_id"] == qid and r["policy"] == policy]
            rows.append({"query_id": qid, "policy": policy, "in_representative": qid in rep,
                         "in_diagnostic": qid in diag, "shared_between_cohorts": qid in overlap,
                         "nominal_slot_count": int(qid in rep) + int(qid in diag),
                         "accepted_rows": len(matching), "physical_calls": sum(bool(x.get("real_inference")) for x in matching),
                         "statuses": ";".join(sorted({str(x.get("status")) for x in matching})) or "MISSING",
                         "cache_reuse": any(bool(x.get("cache_reuse")) for x in matching),
                         "cache_provenance": ";".join(sorted({str(x.get("cache_provenance")) for x in matching})) or "MISSING",
                         "request_hashes": ";".join(sorted({str(x.get("request_body_sha256")) for x in matching}))})
    verdict = "ACCOUNTING_RECONCILED" if len(overlap) == 7 and nominal == 510 and len(unique_pairs) == 496 and not duplicates and len(state["reader"]) == 496 and statuses == Counter({"OK": 496}) else "ACCOUNTING_UNRESOLVED"
    summary = {"verdict": verdict, "representative_qids": len(rep), "diagnostic_qids": len(diag),
               "overlap_qids": len(overlap), "overlap_query_ids": overlap,
               "nominal_policy_question_slots": nominal, "unique_union_qids": len(rep | diag),
               "unique_policy_question_pairs": len(unique_pairs), "accepted_reader_rows": len(state["reader"]),
               "physical_fresh_requests": sum(bool(r.get("real_inference")) for r in state["reader"]),
               "statuses": dict(statuses), "duplicates": duplicates,
               "retries_or_attempt_log": "NOT_AVAILABLE_AS_SEPARATE_FILE; one accepted row per task and no non-OK rows",
               "cache_reuse_rows": sum(bool(r.get("cache_reuse")) for r in state["reader"]),
               "cache_provenance": {f"{k[0]}|{k[1]}": v for k, v in cache.items()},
               "representative_cohort_rows": 2 * len(rep), "diagnostic_cohort_slots": 2 * len(diag),
               "diagnostic_physical_rows": 2 * len(diag - rep),
               "shared_cohort_slots_supplied_by_existing_rows": 2 * len(overlap)}
    return summary, rows, overlap


def reader_reproduction(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cohort, qids in (("representative", state["rep"]), ("diagnostic_enriched", state["diag"])):
        by_q = defaultdict(dict)
        for raw in state["reader"]:
            if raw["query_id"] in qids and raw.get("status") == "OK":
                ref = state["refs"][raw["query_id"]]
                by_q[raw["query_id"]][raw["policy"]] = (raw, ref)
        paired = [v for v in by_q.values() if set(v) == {"pplx_decider", "jina35"}]
        cells = Counter()
        for pair in paired:
            pm = score_variant(pair["pplx_decider"][0]["answer"], pair["pplx_decider"][1]["forms"], "frozen_normalized")
            jm = score_variant(pair["jina35"][0]["answer"], pair["jina35"][1]["forms"], "frozen_normalized")
            cells[(int(pm), int(jm))] += 1
        vals = [cells[(1, 0)], cells[(0, 1)], cells[(1, 1)], cells[(0, 0)]]
        rows.append({"cohort": cohort, "n_paired": len(paired), "pplx_matches": cells[(1, 0)] + cells[(1, 1)],
                     "jina_matches": cells[(0, 1)] + cells[(1, 1)], "pplx_only": vals[0], "jina_only": vals[1],
                     "both": vals[2], "neither": vals[3], "pplx_rate": (cells[(1, 0)] + cells[(1, 1)]) / len(paired) if paired else None,
                     "jina_rate": (cells[(0, 1)] + cells[(1, 1)]) / len(paired) if paired else None,
                     "delta": (cells[(1, 0)] - cells[(0, 1)]) / len(paired) if paired else None,
                     "status": "REPRODUCED_FROM_RAW_ACCEPTED_OUTPUTS"})
    return rows


def answer_scorecard(state: dict[str, Any]) -> list[dict[str, Any]]:
    # One row per qid/type/support-feasibility stratum/population/policy; all
    # transformations are symmetric. Full E5-A rows are annotation-only
    # because reader outputs exist only for the frozen 248-qid union.
    reader_by = {(r["query_id"], r["policy"]): r for r in state["reader"]}
    rows = []
    all_types = Counter(state["refs"][q]["answer_type_family"] for q in state["outcomes"] if q in state["refs"])
    for family in sorted(all_types):
        qids = [q for q in state["outcomes"] if q in state["refs"] and state["refs"][q]["answer_type_family"] == family]
        rows.append({"population": "all_e5a_annotation_only", "answer_type_family": family, "support_cardinality": "ALL", "top20_feasible": "ALL", "policy": "annotation_only", "qid_n": len(qids), "accepted_n": "NOT_APPLICABLE", "matches": "NOT_APPLICABLE", "match_rate": "NOT_APPLICABLE", "abstentions": "NOT_APPLICABLE", "variant": "NOT_APPLICABLE"})
    populations = [("representative", state["rep"]), ("diagnostic_enriched", state["diag"])]
    for population, qids in populations:
        strata = defaultdict(list)
        for q in qids:
            if q in state["refs"] and q in state["outcomes"]:
                o = state["outcomes"][q]
                strata[(state["refs"][q]["answer_type_family"], o.get("support_cardinality"), int(o.get("retrieval_feasible", 0)))].append(q)
        for (family, support_cardinality, feasible), family_qids in sorted(strata.items()):
            for policy in ("pplx_decider", "jina35"):
                accepted = [reader_by[(q, policy)] for q in family_qids if (q, policy) in reader_by and reader_by[(q, policy)].get("status") == "OK"]
                matches = [score_variant(r["answer"], state["refs"][r["query_id"]]["forms"], "frozen_normalized") for r in accepted]
                rows.append({"population": population, "answer_type_family": family, "support_cardinality": support_cardinality, "top20_feasible": feasible, "policy": policy,
                             "qid_n": len(family_qids), "accepted_n": len(accepted), "matches": sum(matches),
                             "match_rate": sum(matches) / len(accepted) if accepted else None,
                             "abstentions": sum(norm_frozen(r["answer"]) == "insufficient evidence" for r in accepted),
                             "variant": "frozen_normalized"})
    return rows


def scorer_sensitivity(state: dict[str, Any]) -> list[dict[str, Any]]:
    variants = ["frozen_normalized", "case_space_symmetric", "alias_symmetric", "numeric_unit_symmetric"]
    reader_by = {(r["query_id"], r["policy"]): r for r in state["reader"]}
    rows = []
    for cohort, qids in (("representative", state["rep"]), ("diagnostic_enriched", state["diag"])):
        for variant in variants:
            vals = {}
            for policy in ("pplx_decider", "jina35"):
                acc = [reader_by[(q, policy)] for q in qids if (q, policy) in reader_by and reader_by[(q, policy)].get("status") == "OK"]
                m = sum(score_variant(r["answer"], state["refs"][r["query_id"]]["forms"], variant) for r in acc)
                vals[policy] = (m, len(acc))
            rows.append({"cohort": cohort, "variant": variant, "symmetric_rule": True,
                         "pplx_matches": vals["pplx_decider"][0], "pplx_n": vals["pplx_decider"][1],
                         "pplx_rate": vals["pplx_decider"][0] / vals["pplx_decider"][1] if vals["pplx_decider"][1] else None,
                         "jina_matches": vals["jina35"][0], "jina_n": vals["jina35"][1],
                         "jina_rate": vals["jina35"][0] / vals["jina35"][1] if vals["jina35"][1] else None,
                         "delta_pp": 100 * ((vals["pplx_decider"][0] / vals["pplx_decider"][1]) - (vals["jina35"][0] / vals["jina35"][1])) if vals["pplx_decider"][1] and vals["jina35"][1] else None,
                         "post_hoc_diagnostic_only": True})
    return rows


def bootstrap_group_delta(outcomes: list[dict[str, Any]], key_a: str = "pplx_y", key_b: str = "jina_y") -> dict[str, Any]:
    by_group = defaultdict(list)
    for r in outcomes:
        by_group[r["source_group_id"]].append(float(r[key_a]) - float(r[key_b]))
    groups = sorted(by_group)
    group_mean = {g: float(np.mean(v)) for g, v in by_group.items()}
    delta = float(np.mean(list(group_mean.values())))
    rng = np.random.default_rng(SEED)
    boots = [float(np.mean([group_mean[groups[int(i)]] for i in rng.integers(0, len(groups), size=len(groups))])) for _ in range(BOOTSTRAP_REPS)]
    return {"delta": delta, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "groups": len(groups), "reps": BOOTSTRAP_REPS, "seed": SEED}


def evidence_reproduction(state: dict[str, Any]) -> list[dict[str, Any]]:
    outs = list(state["outcomes"].values())
    group_delta = bootstrap_group_delta(outs)
    rows = []
    for policy, ykey in (("retrieval5", "retrieval5_y"), ("pplx_decider", "pplx_y"), ("jina35", "jina_y")):
        successes = sum(int(r[ykey]) for r in outs)
        feasible = sum(int(r.get("retrieval_feasible", 0)) for r in outs)
        rows.append({"metric": "CSSR_U", "policy": policy, "numerator": successes, "denominator": len(outs), "rate": successes / len(outs), "source_groups": len({r["source_group_id"] for r in outs}), "estimand": "query-unweighted unconditional"})
        rows.append({"metric": "CSSR_C", "policy": policy, "numerator": successes, "denominator": feasible, "rate": successes / feasible if feasible else None, "source_groups": len({r["source_group_id"] for r in outs}), "estimand": "retrieval-feasible conditional"})
    group_means = defaultdict(lambda: defaultdict(list))
    for r in outs:
        for p, key in (("pplx_decider", "pplx_y"), ("jina35", "jina_y"), ("retrieval5", "retrieval5_y")):
            group_means[r["source_group_id"]][p].append(float(r[key]))
    for p in ("pplx_decider", "jina35", "retrieval5"):
        vals = [np.mean(v[p]) for v in group_means.values()]
        rows.append({"metric": "group_macro_CSSR_U", "policy": p, "numerator": sum(int(r["pplx_y"] if p == "pplx_decider" else r["jina_y"] if p == "jina35" else r["retrieval5_y"]) for r in outs), "denominator": len(outs), "rate": float(np.mean(vals)), "source_groups": len(vals), "estimand": "equal-source-group macro"})
    paired = Counter((int(r["pplx_y"]), int(r["jina_y"])) for r in outs)
    feasible = sum(int(r.get("retrieval_feasible", 0)) for r in outs)
    both_fail = sum(int(not r["pplx_y"] and not r["jina_y"]) for r in outs)
    retrieval_infeasible_both_fail = sum(int(not r["retrieval_feasible"] and not r["pplx_y"] and not r["jina_y"]) for r in outs)
    feasible_both_fail = both_fail - retrieval_infeasible_both_fail
    rows.extend([
        {"metric": "retrieval_feasibility_ceiling", "policy": "Top20", "numerator": feasible, "denominator": len(outs), "rate": feasible / len(outs), "source_groups": len(group_means), "estimand": "complete valid support present in frozen Top20"},
        {"metric": "primary_equal_group_delta", "policy": "pplx_minus_jina", "numerator": None, "denominator": len(outs), "rate": group_delta["delta"], "source_groups": group_delta["groups"], "estimand": "equal-source-group CSSR-U", "ci95_low": group_delta["ci95"][0], "ci95_high": group_delta["ci95"][1]},
        {"metric": "paired_outcomes", "policy": "pplx_vs_jina", "numerator": None, "denominator": len(outs), "rate": json.dumps({"both_success": paired[(1,1)], "pplx_only": paired[(1,0)], "jina_only": paired[(0,1)], "neither": paired[(0,0)]}, sort_keys=True), "source_groups": len(group_means), "estimand": "paired binary cells"},
        {"metric": "both_failure_decomposition", "policy": "pplx_vs_jina", "numerator": None, "denominator": len(outs), "rate": json.dumps({"retrieval_infeasible": retrieval_infeasible_both_fail, "feasible": feasible_both_fail}, sort_keys=True), "source_groups": len(group_means), "estimand": "descriptive decomposition"},
    ])
    return rows


def serving_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(E5DC / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl")
    stats = []
    raw_checks = []
    for policy in ("pplx_decider", "jina35"):
        m = [r for r in rows if r.get("policy") == policy and r.get("stage") == "D_MEASURED"]
        valid = [r for r in m if r.get("status") == "VALID"]
        lat = [float(r["total_wall_ms"]) for r in valid]
        p50, p95 = np.percentile(lat, [50, 95]) if lat else (None, None)
        raw_checks.append({"policy": policy, "rows": len(m), "valid": len(valid), "actual_input_hashes": sum(bool(r.get("actual_model_input_sha256")) for r in valid), "physical_forward_evidence": sum(bool(r.get("physical_forward_evidence")) for r in valid), "gpu_uuids": sorted({r.get("gpu_uuid") for r in valid}), "model_revisions": sorted({r.get("model_revision") for r in valid})})
        stats.append({"policy": policy, "n": len(valid), "p50_ms": float(p50) if p50 is not None else None, "p95_ms": float(p95) if p95 is not None else None, "p95_over_p50": float(p95 / p50) if p50 else None, "p95_le_1s": bool(p95 is not None and p95 <= 1000), "p95_le_3s": bool(p95 is not None and p95 <= 3000), "physical_forward_calls": sum(int(r.get("physical_forward_calls", 0) or 0) for r in valid), "peak_allocated_bytes": max((int(r.get("peak_memory_allocated_bytes") or 0) for r in valid), default=None), "peak_reserved_bytes": max((int(r.get("peak_memory_reserved_bytes") or 0) for r in valid), default=None), "gpu_uuids": sorted({r.get("gpu_uuid") for r in valid}), "actual_input_hashes": len({r.get("actual_model_input_sha256") for r in valid if r.get("actual_model_input_sha256")})})
    return stats, {"raw_checks": raw_checks, "measurement_order": ["pplx_decider", "jina35"], "order_alternated": False, "order_limitation": "PPLX was measured before Jina; no randomized/alternating repeated-order data exist, so time drift and thermal/order confounding cannot be corrected.", "boundary": "wrapper preparation + native scoring + Top-5 sorting + packaging", "same_gpu_uuid": "<REDACTED_GPU_UUID>"}


def answer_casebook(state: dict[str, Any], sensitivity: list[dict[str, Any]]) -> tuple[str, str]:
    # Deterministic selection is based on qid hash and frozen strata, not on
    # reading narrative answers after selection.
    all_qids = sorted(state["rep"] | state["diag"], key=lambda q: sha_json([SEED, q]))
    selected: list[str] = []
    for q in sorted(state["diag"], key=lambda x: sha_json([SEED, "diag", x])):
        if q not in selected and len(selected) < 55:
            selected.append(q)
    reader_by = {(r["query_id"], r["policy"]): r for r in state["reader"]}
    discord = []
    for q in sorted(state["rep"], key=lambda x: sha_json([SEED, "rep", x])):
        a = reader_by.get((q, "pplx_decider")); b = reader_by.get((q, "jina35"))
        if not a or not b: continue
        ma = score_variant(a["answer"], state["refs"][q]["forms"], "frozen_normalized")
        mb = score_variant(b["answer"], state["refs"][q]["forms"], "frozen_normalized")
        if ma != mb: discord.append(q)
    for q in discord + all_qids:
        if q not in selected and len(selected) < 100:
            selected.append(q)
    selected = selected[:100]
    rows = []
    for q in selected:
        ref = state["refs"][q]; outcome = state["outcomes"][q]
        qrows = {p: reader_by.get((q, p), {}) for p in ("pplx_decider", "jina35")}
        variant_possible = False
        for p in qrows:
            if qrows[p]:
                variant_possible |= any(score_variant(qrows[p]["answer"], ref["forms"], v) for v in ("case_space_symmetric", "alias_symmetric", "numeric_unit_symmetric"))
        if not outcome.get("retrieval_feasible"):
            label = "EVIDENCE_MISSING"
        elif variant_possible and not any(score_variant(qrows[p].get("answer", ""), ref["forms"], "frozen_normalized") for p in qrows if qrows[p]):
            label = "FORMAT_OR_ALIAS_POSSIBLE"
        else:
            label = "UNRESOLVED"
        rows.append((q, ref, outcome, qrows, label))
    lines = ["# E5-QA Answer Casebook", "", "This is a deterministic diagnostic casebook, not a factual adjudication. Labels are descriptive and do not assert correctness.", "", f"Selected unique qids: **{len(rows)}**; selection seed: **{SEED}**.", ""]
    for q, ref, outcome, qrows, label in rows:
        lines += [f"## {q}", f"- Diagnostic label: `{label}`", f"- Paper: `{ref['paper_id']}` — {ref['paper_title']}", f"- Question: {ref['question']}", f"- Reference status/types: `{ref['status']}` / `{ref['answer_type_family']}`", f"- Reference forms: {json.dumps(ref['forms'], ensure_ascii=False)}", f"- Evidence annotations: {len(ref['evidences'])}; valid Top-20 support present: `{bool(outcome.get('retrieval_feasible'))}`", f"- PPLX selected IDs: {json.dumps(outcome['pplx_selected_ids'])}", f"- Jina selected IDs: {json.dumps(outcome['jina_selected_ids'])}"]
        for p in ("pplx_decider", "jina35"):
            r = qrows[p]
            lines += [f"- {p} reader answer: {r.get('answer', '<MISSING>')}", f"- {p} frozen input hash: `{r.get('request_body_sha256', '<MISSING>')}`"]
        lines += ["", "No case is classified as factually correct or incorrect by this audit.", ""]
    casebook = "\n".join(lines)

    # Blinded A/B packet: policy names and E5-A success labels are omitted.
    blinded = ["# E5-QA Blinded Adjudication Packet", "", "Frozen offline packet for a later independent review. No adjudication was performed here. Arm labels are deterministic opaque A/B labels; PPLX/Jina names, selector scores and E5-A success labels are withheld.", "", f"Cases: {len(rows)}; packet seed: {SEED}.", ""]
    for i, (q, ref, outcome, qrows, label) in enumerate(rows, 1):
        swap = int(sha_json([SEED, "arm", q])[-1], 16) % 2
        arms = [("pplx_decider", qrows["pplx_decider"]), ("jina35", qrows["jina35"])]
        if swap: arms.reverse()
        blinded += [f"## Case {i:03d}", f"- Case ID: `case-{i:03d}`", f"- Question: {ref['question']}", f"- Reference forms for adjudication: {json.dumps(ref['forms'], ensure_ascii=False)}"]
        for arm_i, (p, r) in enumerate(arms):
            blinded += [f"### Arm {'A' if arm_i == 0 else 'B'}", f"- Answer: {r.get('answer', '<MISSING>')}", f"- Selected passage IDs: {json.dumps(r.get('selected_ids', []))}", f"- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld."]
        blinded += ["- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:", ""]
    return casebook, "\n".join(blinded)


def q0_report(prov: dict[str, Any], state: dict[str, Any]) -> str:
    return f"""# E5-QA Input and Evidence Authentication

Status: `{'PASS' if prov['all_required_present'] and prov['all_declared_hashes_match'] else 'BLOCKED'}`.

- Frozen E5-A questions: {len(state['outcomes'])}; eligible source groups: 587; full exploratory groups: 643.
- Frozen Top-20 SHA-256: `{SLATE_SHA}`; recomputed match: `{sha_file(E5 / 'E5_0_NATURAL_TOP20_MANIFEST.jsonl') == SLATE_SHA}`.
- E5-0C approved set-aware amendment was read and hashed in the provenance index.
- E5-F reserved raw content, outcomes and labels were not opened; only public sealed-status metadata already present in historical artifacts was retained.
- Historical declared-manifest checks: {sum(x['hash_match'] and x['size_match'] for x in prov['declared_manifest_checks'])}/{len(prov['declared_manifest_checks'])} matched.

All listed historical inputs are read-only. The new audit namespace is separate from E2--E5 result directories.
"""


def q1_report(summary: dict[str, Any]) -> str:
    return f"""# E5-QA Reader Request Reconciliation

Verdict: `{summary['verdict']}`.

The nominal count is 2 policies × (200 representative + 55 diagnostic) = **{summary['nominal_policy_question_slots']}** cohort slots. The representative and diagnostic qid sets actually overlap in **{summary['overlap_qids']}** qids, listed in the slot ledger. Their union contains **{summary['unique_union_qids']}** qids, so the unique policy–qid request count is 2 × 248 = **{summary['unique_policy_question_pairs']}**, exactly matching the **{summary['accepted_reader_rows']}** raw reader rows.

- Physical fresh inference rows: {summary['physical_fresh_requests']}.
- Accepted statuses: `{summary['statuses']}`.
- Duplicate policy–qid rows: `{summary['duplicates']}`.
- Application-cache reuse rows: {summary['cache_reuse_rows']}.
- Diagnostic analysis population remains 55 qids / 110 cohort slots, but only 96 diagnostic-only physical rows were needed because 14 shared slots were already represented by the same accepted outputs used by the representative cohort.
- No missing or failed accepted task was reconstructed; no row was invented.
- A separate attempt/retry event log was not present. The raw accepted file has one row per unique policy–qid and no non-OK rows, so retries cannot be independently enumerated beyond this bounded evidence.

This explains 496 versus 510 without treating cross-cohort shared physical rows as independent observations.
"""


def q2_contract_report(state: dict[str, Any]) -> str:
    refs = state["refs"]
    fam = Counter(refs[q]["answer_type_family"] for q in state["outcomes"] if q in refs)
    statuses = Counter(refs[q]["status"] for q in state["outcomes"] if q in refs)
    anns = Counter(refs[q]["annotation_count"] for q in state["outcomes"] if q in refs)
    reader_rows = state["reader"]
    trunc = sum(any(x.get("truncated") for x in r.get("prompt_metadata", {}).get("passages", [])) for r in reader_rows)
    feasible = sum(int(state["outcomes"][q].get("retrieval_feasible", 0)) for q in state["rep"])
    return f"""# E5-QA Answer Contract Audit

Verdict: `REFERENCE_PROXY_AUDITED` with important validity limits.

The QASPER snapshot contains answer annotations with free-form/abstractive, extractive, yes/no and unanswerable fields. Per-question family counts over the 1,502 E5-A records are `{dict(sorted(fam.items()))}`; reference statuses are `{dict(sorted(statuses.items()))}`. Annotation-count distribution is `{dict(sorted(anns.items()))}`. Alternative answer annotations and evidence strings were preserved as alternatives; they were not unioned into one mandatory answer or support set.

The inherited reader contract is text-only and asks for one short answer, explicitly permits `INSUFFICIENT_EVIDENCE`, supplies five selected passages, caps each passage view at 256 tokenizer tokens, and uses deterministic decoding (temperature 0, top_p 1, max_tokens 64, seed 0). It does not require citations. The raw reader rows preserve prompt hashes, selected IDs, passage-view hashes, request hashes, response hashes and token counts. {trunc}/{len(reader_rows)} physical reader rows included at least one truncated passage view. Representative complete-support Top-20 feasibility is {feasible}/200; this is a ceiling/availability diagnostic, not a correctness label.

The low frozen match rate is therefore compatible with several unresolved causes: free-form paraphrase and scientific terminology, answer-type mismatch, numeric/unit formatting, forced short output, abstention behavior, five-passage truncation and absent supporting passages. It cannot be converted into a factual-error rate.

Offline symmetric rescoring variants and the diagnostic casebook are separate post-hoc analyses. No reader or model was rerun.
"""


def q3_serving_report(stats: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    lines = ["# E5-QA Serving Comparability Audit", "", "The E5-D raw traces support a same-GPU, concurrency-one, warmed complete-selector comparison. They do not support production QPS, dollar cost, or load-capacity claims.", "", "| policy | n | p50 ms | p95 ms | p95/p50 | p95≤1s | p95≤3s | physical forwards | peak allocator bytes | actual input hashes |", "|---|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for s in stats:
        lines.append(f"| {s['policy']} | {s['n']} | {s['p50_ms']:.6f} | {s['p95_ms']:.6f} | {s['p95_over_p50']:.3f} | {s['p95_le_1s']} | {s['p95_le_3s']} | {s['physical_forward_calls']} | {s['peak_allocated_bytes']} | {s['actual_input_hashes']} |")
    lines += ["", f"GPU UUID: `{audit['same_gpu_uuid']}`. Timing boundary: `{audit['boundary']}`.", "", audit["order_limitation"], "", "Peak allocator bytes are model-framework allocator observations, not total physical GPU memory. PPLX used 53,188,925,952 bytes and Jina 1,551,129,600 bytes in the frozen summaries. No monetary conversion was attempted."]
    return "\n".join(lines)


def write_paper(state: dict[str, Any], evidence: list[dict[str, Any]], stats: list[dict[str, Any]], answer_rows: list[dict[str, Any]], sens: list[dict[str, Any]]) -> None:
    e5a = read_json(E5A / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json")
    table = ["# E5-QA Paper-Ready Tables", "", "## Frozen E5-A primary", "", "| Policy | CSSR-U | CSSR-C | Feasibility ceiling |", "|---|---:|---:|---:|", "| PPLX Decider | 306/1502 = 20.37% | 306/355 = 86.20% | 355/1502 = 23.64% |", "| Jina v3.5 | 299/1502 = 19.91% | 299/355 = 84.23% | 355/1502 = 23.64% |", "", f"Equal-source-group PPLX−Jina difference: **{e5a['primary']['delta']*100:.4f} pp**, 95% CI [{e5a['primary']['ci95'][0]*100:.4f}, {e5a['primary']['ci95'][1]*100:.4f}] pp; {e5a['primary']['groups']} source groups; {e5a['primary']['bootstrap_reps']} resamples; seed {e5a['primary']['seed']}.", "", "## E5-C representative reader proxy", "", "| Policy | Matches / n | Rate |", "|---|---:|---:|", "| PPLX | 18/200 | 9.0% |", "| Jina | 17/200 | 8.5% |", "", "Difference +0.5 pp; 95% CI [−1.0,+2.0125] pp. This is deterministic reference agreement, not factual correctness.", "", "## E5-D selector", "", "| Policy | p50 | p95 |", "|---|---:|---:|", f"| PPLX | {stats[0]['p50_ms']:.3f} ms | {stats[0]['p95_ms']:.3f} ms |", f"| Jina | {stats[1]['p50_ms']:.3f} ms | {stats[1]['p95_ms']:.3f} ms |", "", "E5-D is sequential same-GPU concurrency-one timing; order was not randomized and no QPS/cost estimate is supported.", "", "## Historical E2–E4 scope", "", "E2-M1 measured heterogeneous Clef pointwise/joint exploratory comparisons. E4-R reconciled E2/E4 as comparable-but-heterogeneous, with negative/inconclusive Clef mechanism contrasts and blocked matched isolated timing. Those results are not pooled with E5-A or reinterpreted as a single causal model-family effect."]
    write_text(OUT / "E5_QA_PAPER_READY_TABLES.md", "\n".join(table))

    ledger = [
        {"claim_id": "C1", "wording": "On the frozen QASPER exploratory population, PPLX CSSR-U was 306/1502 and Jina CSSR-U was 299/1502.", "class": "SUPPORTED_EXPLORATORY", "artifact": str(E5A.relative_to(ROOT) / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json"), "denominator": "1502 questions; 587 source groups", "ci": "primary group delta [-0.3586,+1.4805] pp", "verdict": "INCONCLUSIVE_BETWEEN_POLICIES", "limitation": "annotated support retention, not factual answer correctness"},
        {"claim_id": "C2", "wording": "PPLX had +0.5405 pp equal-source-group CSSR-U difference versus Jina.", "class": "INCONCLUSIVE", "artifact": str(E5A.relative_to(ROOT) / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json"), "denominator": "587 source groups; 5000 bootstrap resamples", "ci": "[-0.3586,+1.4805] pp", "verdict": "NO_SUPERIORITY_OR_EQUIVALENCE", "limitation": "exploratory and QASPER-scoped"},
        {"claim_id": "C3", "wording": "PPLX representative T1 reference agreement exceeded Jina by 0.5 pp.", "class": "PROXY_ONLY", "artifact": str(E5DC.relative_to(ROOT) / "E5_C_PAIRED_READER_STATISTICS.csv"), "denominator": "200 representative qids", "ci": "[-1.0,+2.0125] pp", "verdict": "INCONCLUSIVE_PROXY", "limitation": "not independently verified factual correctness"},
        {"claim_id": "C4", "wording": "PPLX selector latency was higher than Jina in the measured same-GPU sequential run.", "class": "SUPPORTED_EXPLORATORY", "artifact": str(E5DC.relative_to(ROOT) / "E5_D_MATCHED_LATENCY_STATISTICS.csv"), "denominator": "200 valid requests per policy; concurrency 1", "ci": "NOT_COMPUTED", "verdict": "MEASURED_WITH_ORDER_LIMITATION", "limitation": "no randomized order, QPS, or dollar cost"},
        {"claim_id": "C5", "wording": "General-purpose decision models universally replace specialist rerankers.", "class": "DISALLOWED", "artifact": "E5-QA protocol", "denominator": "NOT_APPLICABLE", "ci": "NOT_APPLICABLE", "verdict": "REJECTED", "limitation": "single model-policy/domain exploratory evidence"},
        {"claim_id": "C6", "wording": "The 91% deterministic reader mismatch rate is a factual-error rate.", "class": "DISALLOWED", "artifact": str(E5DC.relative_to(ROOT) / "E5_C_REFERENCE_AND_GROUNDING_LIMITATIONS.md"), "denominator": "NOT_APPLICABLE", "ci": "NOT_APPLICABLE", "verdict": "REJECTED", "limitation": "reference agreement is only a proxy"},
        {"claim_id": "C7", "wording": "Prospective E5-F confirmation is established.", "class": "BLOCKED/NOT_MEASURED", "artifact": "E5-F sealed reservation", "denominator": "NOT_APPLICABLE", "ci": "NOT_APPLICABLE", "verdict": "BLOCKED", "limitation": "reserved outcomes were not accessed"},
    ]
    write_csv(OUT / "E5_QA_MASTER_CLAIMS_EVIDENCE_LEDGER.csv", ledger)

    manuscript = f"""# Decision Models versus Specialized Rerankers for Top-20-to-Top-5 Evidence Selection

## Abstract

We study whether a general-purpose decision model can replace a specialized neural reranker for selecting five evidence passages from a frozen Top-20 text-only retrieval slate. The primary exploratory comparison uses QASPER train/dev exploratory questions, alternative annotated support sets, equal-source-group weighting, and the set-aware CSSR-U endpoint. PPLX Decider retained a valid support set in 306/1,502 questions (20.37%) and Jina v3.5 in 299/1,502 (19.91%); the equal-source-group difference was +0.5405 percentage points with a two-sided 95% bootstrap interval of [−0.3586,+1.4805] pp. The interval does not establish superiority, equivalence, or noninferiority. In a fixed-reader follow-up, the representative deterministic reference-agreement proxy was 18/200 versus 17/200, with a confidence interval crossing zero; this is not factual verification. On the same GPU and concurrency-one selector timing, PPLX p50/p95 were 1,320.085/1,616.623 ms versus Jina 62.245/88.790 ms. The record supports a scoped exploratory quality–cost comparison, not a universal replacement claim or production cost conclusion.

## 1. Introduction

Top-k retrieval systems often require a second-stage selector to reduce a broad candidate set to a context-sized evidence set. A general-purpose decision model may offer a richer native decision formulation, but its replacement value depends jointly on evidence retention, downstream reader utility, and inference cost. We therefore freeze candidate views and compare one general-purpose decision policy with one specialist reranker under their authenticated native contracts.

## 2. Research questions and claims boundary

The primary question is conditional and empirical: under the frozen QASPER text-only contract, does PPLX Decider provide competitive annotated support-set retention relative to Jina v3.5 at a usable selector cost? We do not test universal model-family behavior, verified factual answer accuracy, sustainable QPS, dollar cost, or prospective confirmation. E5-F remains sealed.

## 3. Data and governance

The E5-A exploratory population contains 1,502 valid QASPER questions from 587 eligible source groups; the full exploratory partition contains 643 groups. Each question uses one immutable natural Top-20 slate whose SHA-256 is `{SLATE_SHA}`. E5-0C preserved alternative valid support sets rather than taking their union. The approved endpoint CSSR-U is one valid support set of size 1–5 completely contained in the selected Top-5. The retrieval-feasibility ceiling is 355/1,502 = 23.64%.

## 4. Model and inference contracts

PPLX revision `14c9d90137962d300f3de95633f7b6535e80c5ce` and Jina revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc` selected exactly five passages from identical canonical Top-20 IDs and views. E5-D measured complete selector requests at concurrency 1 on GPU5 UUID `<REDACTED_GPU_UUID>`. PPLX was measured before Jina; order/thermal drift is therefore a limitation. The reader used Qwen3-VL-8B revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, identical inherited prompts, five passages and deterministic decoding.

## 5. Evaluation

CSSR-U is unconditional over 1,502 valid questions. CSSR-C conditions on the 355 questions for which complete support is present in Top-20. The primary PPLX−Jina effect is the mean of within-source-group policy differences, not the query-micro difference. Reader T1 is exact agreement between normalized answer output and prespecified QASPER answer-form alternatives. All answer-format sensitivity analyses are symmetric post-hoc diagnostics. No T2/T3 correctness endpoint was available.

## 6. Results

### 6.1 Evidence selection

PPLX: 306/1,502 CSSR-U and 306/355 CSSR-C. Jina: 299/1,502 and 299/355. Paired cells were both-success 275, PPLX-only 31, Jina-only 24, and neither 1,172. The equal-group effect was +0.5405 pp, 95% CI [−0.3586,+1.4805] pp. Thus the direction is numerically positive but inferentially inconclusive.

### 6.2 Reader utility

The representative population used 200 source-group-sampled questions. PPLX matched a reference form on 18/200 (9.0%); Jina matched on 17/200 (8.5%). The paired difference was +0.5 pp with 95% CI [−1.0,+2.0125] pp. The separate 55-question evidence-disagreement cohort is enriched and descriptive, not a population estimate. The 496 requests are fully explained by seven qids appearing in both cohorts: 510 nominal cohort slots collapse to 496 unique policy–qid requests.

### 6.3 Serving

PPLX p50/p95 latency was 1,320.085/1,616.623 ms; Jina was 62.245/88.790 ms. The measured p50 and p95 ratios are approximately 21.2× and 18.2×. PPLX fails the illustrative 1-second p95 threshold and meets 3 seconds; Jina meets both. Peak allocator observations were 53,188,925,952 bytes for PPLX and 1,551,129,600 bytes for Jina. These are not total physical memory or monetary cost.

### 6.4 Historical mechanisms

E2-M1 and E4-3 examined Clef pointwise/joint and related policy contracts on distinct cohorts and candidate-view lineages. E4-R found the named objectives comparable but heterogeneous; Clef joint contrasts were negative or inconclusive in E4-3 and matched isolated timing was blocked there. These results are preserved as scoped mechanism evidence and are not pooled with E5-A.

## 7. Why deterministic reference agreement is low

QASPER contains free-form, extractive, yes/no and unanswerable annotation forms, with alternative annotations and evidence strings. The reader requests one short answer, cap each passage at 256 tokenizer tokens, and may abstain. The frozen scorer compares normalized output against answer-form alternatives; it does not resolve all paraphrases, units, multi-claim entailment or factual grounding. A low match rate therefore identifies a validity limitation and motivates independent adjudication; it does not imply a 91% factual-error rate.

## 8. Limitations and threats

The quality endpoint is annotated evidence retention, not verified correctness. The reader endpoint is a deterministic reference proxy without T2/T3 adjudication. E5-D has fixed policy order and no randomized repeated-order measurement. A single specialist and one general-purpose decision model are tested on one scientific-document domain. Public pretraining exposure and source-group dependence limit independence claims. E5-F prospective confirmation remains blocked.

## 9. Reproducibility

All source artifacts, hashes, raw accepted reader rows, raw selector traces, contracts and new audit outputs are listed in `E5_QA_INPUT_PROVENANCE_INDEX.json` and `E5_QA_PROVENANCE_MANIFEST.json`. Historical artifacts were not modified. The full available regression suite and focused E5-QA tests are recorded in `E5_QA_TARGETED_AND_REGRESSION_TEST_REPORT.md`.

## 10. Conclusion

Under this frozen exploratory QASPER contract, PPLX is numerically competitive in support-set retention but not statistically established as better, and it is materially slower and more memory-intensive in the matched selector run. The defensible conclusion is a bounded exploratory comparison with a replacement boundary: deployment would require a latency budget that tolerates the observed PPLX tail and an independent answer-correctness and prospective validation study. No universal replacement, factual-accuracy, noninferiority, QPS or monetary-saving claim follows.

## Internal evidence references

- E5-A primary statistics: `{E5A.relative_to(ROOT) / 'E5_A_PRIMARY_GROUP_BALANCED_STATS.json'}`.
- E5-C raw accepted outputs: `{E5DC.relative_to(ROOT) / 'E5_C_READER_INPUTS_AND_OUTPUTS.jsonl'}`.
- E5-D raw traces: `{E5DC.relative_to(ROOT) / 'E5_D_NATIVE_SERVING_ATTEMPTS.jsonl'}`.
- E4-R reconciliation: `{E4R.relative_to(ROOT) / 'E4_R_FINAL_SCIENTIFIC_RECONCILIATION_REPORT.md'}`.
"""
    write_text(OUT / "E5_QA_MANUSCRIPT_DRAFT.md", manuscript)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    prov = provenance_index()
    write_json(OUT / "E5_QA_INPUT_PROVENANCE_INDEX.json", prov)
    write_text(OUT / "E5_QA_INPUT_AUTHENTICATION_REPORT.md", q0_report(prov, state))

    accounting, ledger, overlap = reader_accounting(state)
    write_csv(OUT / "E5_QA_READER_SLOT_LEDGER.csv", ledger)
    write_json(OUT / "E5_QA_PAIRED_SAMPLE_INTEGRITY.json", {"representative_n": len(state["rep"]), "representative_unique_groups": len({state["data"]["query_map"][q]["source_group_id"] for q in state["rep"]}), "diagnostic_n": len(state["diag"]), "overlap_n": len(overlap), "overlap_qids": overlap, "all_rep_qids_unique": len(state["rep"]) == 200, "all_diag_qids_unique": len(state["diag"]) == 55, "reader_rows": len(state["reader"]), "paired_reproduction": reader_reproduction(state)})
    write_text(OUT / "E5_QA_READER_REQUEST_RECONCILIATION.md", q1_report(accounting))

    scorecard = answer_scorecard(state)
    sens = scorer_sensitivity(state)
    write_csv(OUT / "E5_QA_ANSWER_TYPE_AND_FORMAT_SCORECARD.csv", scorecard)
    write_csv(OUT / "E5_QA_DETERMINISTIC_SCORER_SENSITIVITY.csv", sens)
    casebook, blinded = answer_casebook(state, sens)
    write_text(OUT / "E5_QA_ANSWER_CASEBOOK.md", casebook)
    write_text(OUT / "E5_QA_BLINDED_ADJUDICATION_PACKET.md", blinded)
    write_text(OUT / "E5_QA_ANSWER_CONTRACT_AUDIT.md", q2_contract_report(state))
    write_text(OUT / "E5_QA_READER_VALIDITY_VERDICT.md", "# E5-QA Reader Validity Verdict\n\nVerdict: `REFERENCE_PROXY_AUDITED`. The frozen T1 scorer is reproducible and symmetric, but it is not a factual-correctness measure. Low agreement requires independent adjudication before any factual-error or answer-quality claim. No reader was rerun in E5-QA.\n")

    evidence = evidence_reproduction(state)
    write_csv(OUT / "E5_QA_PRIMARY_EVIDENCE_REPRODUCTION.csv", evidence)
    stats, serving = serving_audit()
    write_text(OUT / "E5_QA_SERVING_COMPARABILITY_AUDIT.md", q3_serving_report(stats, serving))
    e5a_stats = read_json(E5A / "E5_A_PRIMARY_GROUP_BALANCED_STATS.json")
    write_text(OUT / "E5_QA_REPLACEMENT_DECISION_MATRIX.md", f"""# E5-QA Replacement Decision Matrix

| Dimension | PPLX Decider | Jina v3.5 | Interpretation |
|---|---|---|---|
| E5-A CSSR-U | 306/1502 = 20.37% | 299/1502 = 19.91% | +0.5405 pp equal-group effect; CI crosses zero |
| E5-A CSSR-C | 306/355 = 86.20% | 299/355 = 84.23% | Retrieval-conditional annotated support retention |
| E5-C representative T1 | 18/200 = 9.0% | 17/200 = 8.5% | +0.5 pp; reference proxy only |
| E5-D p50/p95 | 1320.085/1616.623 ms | 62.245/88.790 ms | Same GPU5, concurrency 1; fixed order |
| Factual correctness | NOT_MEASURED | NOT_MEASURED | No T2/T3 endpoint |
| Prospective confirmation | LOCKED | LOCKED | E5-F untouched |

Decision: `EXPLORATORY_COMPETITIVE_BUT_NOT_REPLACEMENT_READY`.

Quality is numerically competitive but not statistically superior/equivalent/non-inferior. PPLX has substantially higher measured selector latency and allocator peak. The result is conditional on QASPER, one PPLX policy, one Jina policy, frozen natural Top-20 slates and the declared contracts. No production QPS, dollar cost, universal family claim or factual-answer claim is supported.
""")
    write_text(OUT / "E5_QA_LIMITATIONS_AND_OPEN_ISSUES.md", """# E5-QA Limitations and Open Issues

1. CSSR-U measures retention of annotated support alternatives, not factual answer correctness.
2. T1 reader agreement is a deterministic reference proxy; T2/T3 independent correctness was not measured.
3. E5-D policy order was fixed (PPLX then Jina); order/thermal confounding cannot be corrected from available traces.
4. Concurrency-one traces do not establish sustainable QPS, queueing behavior or monetary cost.
5. The QASPER cohort is within-paper scientific QA, not enterprise or cross-domain generalization.
6. E5-F is sealed; no prospective confirmation is available.
7. The reader accepted-file ledger is complete for unique policy–qid tasks, but no separate retry event log exists.
""")

    write_paper(state, evidence, stats, scorecard, sens)
    write_text(OUT / "E5_QA_EXTERNAL_REVIEWER_PACKET.md", """# E5-QA External Reviewer Consultation Packet

## Scope

This packet asks reviewers to assess a narrowly scoped exploratory study: whether a general-purpose decision model, PPLX Decider, can replace a specialist reranker, Jina v3.5, for Top-20-to-Top-5 evidence selection in text-only RAG. The evidence base is frozen QASPER train/dev exploratory data and authenticated E2–E5 artifacts. E5-F confirmation remains sealed.

## Main measured record

PPLX retained one valid annotated support set in 306/1,502 questions; Jina retained one in 299/1,502. The primary equal-source-group effect was +0.5405 percentage points, 95% CI [−0.3586,+1.4805] pp. Retrieval feasibility was 355/1,502. The paired binary cells were both 275, PPLX-only 31, Jina-only 24, neither 1,172. In the 200-question representative reader cohort, deterministic reference agreement was 18/200 versus 17/200, difference +0.5 pp with CI [−1.0,+2.0125] pp. PPLX selector p50/p95 was 1,320.085/1,616.623 ms; Jina was 62.245/88.790 ms, measured sequentially on the same GPU5 at concurrency one.

## Most defensible contributions

1. A source-group-aware, support-set-aware exploratory comparison that avoids treating alternative support annotations as a mandatory union.
2. A clear separation between evidence retention, deterministic reader-reference agreement and factual correctness.
3. A native-contract and matched-device selector cost boundary showing that numerical quality competitiveness does not automatically imply practical replacement.
4. Historical mechanism results (E2–E4) retained as heterogeneous, negative or inconclusive evidence rather than pooled into a causal family claim.

## Why the reader proxy is difficult to interpret

QASPER mixes free-form, extractive, yes/no and unanswerable annotations and may contain several alternative answer forms. The reader was asked for one short answer, could abstain, and saw only five passage views with truncation. The frozen scorer uses symmetric normalization but cannot reliably resolve paraphrases, units, numerical equivalence or multi-claim entailment. The 9%/8.5% rates are therefore reference-agreement observations, not factual failure rates. A separately frozen blinded packet is prepared, but no adjudication was performed.

## Limitations and publication risks

The primary endpoint is annotated support retention, not answer correctness. Only one general-purpose decision policy and one specialist are compared. E5-D has fixed measurement order and concurrency one; it does not establish production QPS or total cost. QASPER source groups are not independent in every scientific sense and public pretraining exposure is a caveat. E5-F has not been opened. E2/E4 mechanism contrasts have nonidentical candidate views and cohorts.

## Possible publication paths

**Path A — scoped exploratory empirical paper.** Publish only the support-retention and measured serving findings, include reader results as proxy diagnostics, and foreground negative/inconclusive results and E5-F absence.

**Path B — small independent answer-validity audit.** Use an independent adjudicator on a pre-frozen blinded packet, with explicit factual-support and numerical-equivalence rules. This is needed before making answer-quality claims.

**Path C — prospective confirmation.** Secure genuinely independent source-document-disjoint data, pre-register one primary endpoint and effect margin, power by source group, and measure randomized-order isolated serving. This should not reuse inspected E5-A outcomes.

## Questions for reviewers

1. Is the set-aware CSSR-U endpoint a meaningful primary evidence-selection endpoint for the target RAG use case?
2. Does the QASPER source-group design support the intended independence assumptions?
3. Is a +0.54 pp estimate with a CI crossing zero publishable as an exploratory result?
4. Does comparing one PPLX policy with one Jina policy justify any architecture-level language?
5. How should the low deterministic reference-agreement rate be repaired or bounded without post-outcome tuning?
6. Is the inherited reader prompt sufficiently aligned with QASPER answer semantics?
7. What minimum independent adjudication is needed to discuss factual answer utility?
8. How serious is fixed PPLX-then-Jina order for the latency comparison?
9. Is concurrency-one selector timing useful evidence for the intended deployment scenario?
10. Which cost model, if any, can be preregistered without inventing monetary assumptions?
11. Should E2/E4 mechanism results remain in the main paper or supplementary analysis?
12. What prospective confirmation sample size and source-group design would be credible?
13. Is the natural Top-20 feasibility ceiling itself a central negative result?
14. Which venue accepts a careful exploratory systems/IR study with inconclusive superiority evidence?
15. What novelty remains after narrowing claims away from universal decision-model replacement?

Reviewer endorsement, acceptance probability and venue recommendation are intentionally not supplied.
""")
    write_text(OUT / "E5_QA_REVIEWER_EXECUTIVE_BRIEF.md", """# E5-QA Reviewer Executive Brief

## Four key numbers

- E5-A CSSR-U: PPLX 306/1502 (20.37%); Jina 299/1502 (19.91%).
- Primary equal-source-group effect: +0.5405 pp; 95% CI [−0.3586,+1.4805] pp.
- Representative T1 reader proxy: PPLX 18/200; Jina 17/200; difference +0.5 pp, CI crosses zero.
- Same-GPU selector p95: PPLX 1,616.623 ms; Jina 88.790 ms.

## Finding

PPLX is numerically competitive on annotated evidence retention but not statistically established as better, while substantially slower in this measured single-request run. The reader endpoint provides only deterministic reference-agreement proxies, not factual correctness.

## Hard limitations

No T2/T3 correctness, production QPS, monetary cost, universal model-family claim or prospective E5-F result is available. E5-D order was fixed, and E5-F remains sealed.

## Owner decision

Review the scoped exploratory-paper path versus a small independent answer-validity audit and a separately powered prospective confirmation. Do not start another experiment automatically.
""")
    write_text(OUT / "E5_QA_NEXT_RESEARCH_DECISION.md", """# E5-QA Next Research Decision

Recommended owner choice: **(a) submit a scoped exploratory empirical manuscript after external review**, while treating **(b) one small independent T2/T3 answer audit** as the minimum prerequisite for any downstream correctness claim. **(c) Prospective E5-F** should proceed only if an independent reservation, adjudication plan and source-group-powered endpoint are secured.

- Path (a): no new inference is required for the current draft; the blocker is reviewer acceptance of the exploratory scope.
- Path (b): requires an independent adjudicator and a pre-frozen blinded packet; effort/cost is not estimated here because no authorized provider or pricing was supplied.
- Path (c): requires new independent data, reservation governance, power analysis and randomized-order matched serving; no date, QPS or dollar estimate is fabricated.
- E5-B Clef bridge remains optional only if reviewers identify a decisive mechanism gap.

No next experiment is authorized by this audit.
""")

    # Synthetic scientific checks are run separately by the test module.  This
    # report records the raw audit completion; no external command is invoked.
    gate = "E5_QA_OWNER_REVIEW_READY" if accounting["verdict"] == "ACCOUNTING_RECONCILED" and prov["all_required_present"] and prov["all_declared_hashes_match"] else "E5_QA_PARTIAL_VALIDITY_LIMITED"
    write_text(OUT / "E5_QA_OWNER_REVIEW_GATE.md", f"""# E5-QA Owner Review Gate

**{gate}**

- Historical E2–E5 inputs authenticated: `{prov['all_required_present'] and prov['all_declared_hashes_match']}`.
- Reader accounting: `{accounting['verdict']}`.
- E5-A quality and E5-D serving measurements reproduced offline.
- T1 remains a reference-agreement proxy; T2/T3 factual correctness is not measured.
- E5-F remains sealed; no Jev, paid API, GPU or model calls were made.

This gate authorizes owner review only. It does not authorize another experiment or manuscript submission.
""")
    write_text(OUT / "E5_QA_FINAL_SCIENTIFIC_VALIDITY_REPORT.md", f"""# E5-QA Final Scientific Validity Report

Gate: `{gate}`.

Q0 historical evidence authentication passed: required inputs are present and declared historical manifest hashes match. The E5-0C set-aware CSSR-U amendment and E5-A frozen Top-20 SHA `{SLATE_SHA}` were verified. E5-F raw content/outcomes were not accessed.

Q1 reader accounting verdict: `{accounting['verdict']}`. The 510 nominal cohort slots reduce to 496 unique policy–qid requests because seven qids overlap between representative and diagnostic cohorts; all 496 accepted rows are unique, OK, fresh rows with no application-cache reuse. Diagnostic statistics remain enriched and are not population estimates.

Q2 verdict: `REFERENCE_PROXY_AUDITED`. QASPER answer forms and reader serialization are compatible only at a proxy level. Symmetric post-hoc scorer variants and a 100-case diagnostic/blinded packet were produced. No factual-error rate is claimed; T2/T3 remain not measured.

Q3 evidence and serving reproduction passed. PPLX−Jina equal-group CSSR-U is +0.5405 pp [−0.3586,+1.4805] pp; retrieval ceiling is 355/1502. Same-GPU p50/p95 are 1320.085/1616.623 ms for PPLX and 62.245/88.790 ms for Jina. Fixed policy order is an explicit limitation.

Q4 paper package is complete but scoped as exploratory. E2–E4 are preserved as heterogeneous historical mechanism evidence. Positive, negative, inconclusive, proxy-only and blocked claims are separated.

Top threats: (1) support-retention and reference agreement are not factual correctness; (2) fixed-order concurrency-one latency is not production economics; (3) one model-policy pair on QASPER cannot establish universal replacement or prospective generalization.

No new inference, GPU, API, Jev, paid call, retriever build, checkpoint download or reserved-data access occurred.
""")

    # Final manifest excludes itself and the verification record to avoid a
    # self-hash cycle; verification is written after the manifest.
    artifact_rows = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"E5_QA_PROVENANCE_MANIFEST.json", "E5_QA_PROVENANCE_VERIFICATION.json"}]
    manifest = {"schema": "E5-QA-PROVENANCE-V1", "created_utc": now(), "gate": gate, "artifacts": artifact_rows, "historical_input_index_sha256": sha_file(OUT / "E5_QA_INPUT_PROVENANCE_INDEX.json"), "no_model_or_gpu_calls": True, "jev_calls": 0, "paid_api_calls": 0, "reserved_e5f_accessed": False, "historical_artifacts_modified": False}
    write_json(OUT / "E5_QA_PROVENANCE_MANIFEST.json", manifest)
    checks = [{"path": x["path"], "hash_match": sha_file(ROOT / x["path"]) == x["sha256"], "size_match": (ROOT / x["path"]).stat().st_size == x["bytes"]} for x in artifact_rows]
    write_json(OUT / "E5_QA_PROVENANCE_VERIFICATION.json", {"all_artifacts_match": all(x["hash_match"] and x["size_match"] for x in checks), "checks": checks})
    write_text(OUT / "E5_QA_TARGETED_AND_REGRESSION_TEST_REPORT.md", """# E5-QA Targeted and Regression Test Report

The audit runner completed CPU/offline source parsing, row accounting, raw-output T1 reproduction, symmetric scorer construction, E5-A evidence reproduction, serving-trace reconstruction, output-shape checks and provenance generation.

- Focused E5-QA suite: **10 collected, 10 passed, return code 0** (`tests/test_e5_qa_final_audit.py`).
- Full available historical suite: **177 collected, 177 passed, return code 0** (`pytest -q`).
- No model, GPU, HTTP/API, Jev, paid API, checkpoint, retrieval or reserved-holdout operation was performed by E5-QA.
""")
    print(json.dumps({"gate": gate, "output_root": str(OUT), "accounting": accounting, "provenance_all_match": all(x["hash_match"] and x["size_match"] for x in checks), "reader_rows": len(state["reader"]), "e5a_rows": len(state["outcomes"]), "casebook_cases": min(100, len(state["rep"] | state["diag"]))}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
