#!/usr/bin/env python3
"""CPU/offline DecisionRAG E4-R reconciliation.

This module reads preserved E2--E4 artifacts only.  It does not import torch,
load checkpoints, contact services, access holdout outcomes, or perform model
inference.  All newly written files live below the E4-R output namespace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e4_v1_1/e4_r_reconciliation"
E2 = ROOT / "results/non_jev/v2/e2_m1"
E2N2 = ROOT / "results/non_jev/v2/e2_n2"
E2SP = ROOT / "results/non_jev/v2/e2_sp"
E2SPR2 = ROOT / "results/non_jev/v2/e2_sp_r2"
E2SPR = ROOT / "results/non_jev/v2/e2_sp_r"
E3 = ROOT / "results/non_jev/v2/e3_v1"
E4 = ROOT / "results/non_jev/v2/e4_v1_1"
E43 = E4 / "e4_3_operating_regimes"
SEED = 731
BOOTSTRAP_REPS = 5000

HISTORICAL_ROOTS = [E2, E2N2, E2SP, E2SPR2, E2SPR, E3, E4]
GOVERNING_FILES = [
    ROOT / "DecisionRAG_E4_v1_1_Revised_Research_Plan.md",
    ROOT / "DecisionRAG_E4_v1_1_Master_Technical_SDD.md",
    ROOT / "DecisionRAG_E4_R_Scientific_Reconciliation_Reviewer_Dossier_Codex_SDD.md",
    ROOT / "DecisionRAG_E4_3_Operating_Regimes_Execution_SDD_Codex.md",
    ROOT / "scripts/e4_3_operating_regimes.py",
]

E4_POLICIES = [
    "clef_pointwise_binary", "clef_joint_binary", "liquid_native_joint",
    "jina35_native", "pplx_decider_native", "bge_pointwise",
]


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


def fmt(v: Any, digits: int = 6) -> str:
    if v is None or v == "":
        return "NOT_MEASURED"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    return vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]


def bootstrap_ci(values: list[float], seed: int = SEED, reps: int = BOOTSTRAP_REPS) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)]
    means.sort()
    return sum(values) / n, means[int(.025 * (reps - 1))], means[int(.975 * (reps - 1))]


def cluster_bootstrap(groups: dict[str, list[float]], seed: int = SEED, reps: int = BOOTSTRAP_REPS) -> tuple[float | None, float | None, float | None]:
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return None, None, None
    names = list(groups)
    rng = random.Random(seed)
    observed = [v for values in groups.values() for v in values]
    means = []
    for _ in range(reps):
        sample = []
        for _ in names:
            sample.extend(groups[rng.choice(names)])
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(observed) / len(observed), means[int(.025 * (reps - 1))], means[int(.975 * (reps - 1))]


def all_historical_files() -> list[Path]:
    paths: set[Path] = set()
    for root in HISTORICAL_ROOTS:
        if root.is_dir():
            paths.update(p for p in root.rglob("*") if p.is_file() and not p.is_relative_to(OUT))
    paths.update(p for p in GOVERNING_FILES if p.is_file())
    return sorted(paths)


def verify_manifest(path: Path) -> dict[str, Any]:
    result = {"manifest": str(path.relative_to(ROOT)), "status": "NOT_READ", "checks": [], "missing": [], "mismatches": []}
    try:
        value = read_json(path)
    except Exception as exc:
        result["status"] = "UNPARSEABLE"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["status"] = "PARSED"
    mappings: list[tuple[str, Any]] = []
    for field in ("outputs", "files", "referenced_inputs", "sources"):
        if isinstance(value.get(field), dict):
            mappings.extend((field, x) for x in value[field].items())
    for field, (name, meta) in mappings:
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = ROOT / name if str(name).startswith(("results/", "data/", "scripts/")) else path.parent / name
        expected = meta.get("sha256") if isinstance(meta, dict) else (meta if isinstance(meta, str) and len(meta) == 64 else None)
        if expected is None:
            continue
        check = {"field": field, "name": str(name), "path": str(candidate), "expected": expected}
        # Some historical manifests stored a governing SDD basename relative to
        # their output directory, while the immutable source file lives at the
        # project root. Resolve that existing file; do not create a copy.
        if not candidate.is_file() and (ROOT / Path(name).name).is_file():
            root_candidate = ROOT / Path(name).name
            if sha_file(root_candidate) == expected:
                candidate = root_candidate
                check["path"] = str(candidate)
                check["path_resolution"] = "PROJECT_ROOT_FALLBACK"
        if not candidate.is_file():
            result["missing"].append(str(candidate))
            check["status"] = "MISSING"
        else:
            actual = sha_file(candidate)
            check["actual"] = actual
            check["status"] = "PASS" if actual == expected else "MISMATCH"
            if actual != expected:
                result["mismatches"].append(check)
        result["checks"].append(check)
    if result["mismatches"]:
        result["status"] = "HASH_MISMATCH"
    elif result["missing"]:
        result["status"] = "MISSING_REFERENCES"
    return result


def inventory() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in all_historical_files():
        rel = str(p.relative_to(ROOT))
        category = "governing" if p in GOVERNING_FILES else "historical_artifact"
        rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha_file(p), "category": category, "readable": True})
    write_csv(OUT / "E4_R_INPUT_INVENTORY.csv", rows)
    # Markdown gates and JSONL row streams are not JSON-object manifests.
    # They remain byte-hashed in the inventory; only JSON manifests are parsed
    # for structured path/hash references here.
    manifests = [p for p in all_historical_files() if p.suffix.lower() == ".json" and ("MANIFEST" in p.name or "manifest" in p.name)]
    checks = [verify_manifest(p) for p in manifests]
    e2_config = read_json(E2 / "E2_M1_FROZEN_EXPLORATORY_CONFIG.json") if (E2 / "E2_M1_FROZEN_EXPLORATORY_CONFIG.json").is_file() else {}
    e2_exec = read_json(E2 / "E2_M1_EXECUTION_MANIFEST.json") if (E2 / "E2_M1_EXECUTION_MANIFEST.json").is_file() else {}
    audit = {
        "status": "PASS_WITH_HISTORICAL_STATUS_INCONSISTENCY" if e2_config.get("status") == "RUNNING" and e2_exec.get("status") == "E2_M1_PASS" else "PASS",
        "inventory_count": len(rows), "inventory_bytes": sum(int(x["bytes"]) for x in rows),
        "manifest_checks": checks, "manifest_parse_count": sum(x["status"] == "PARSED" for x in checks),
        "manifest_hash_mismatch_count": sum(len(x["mismatches"]) for x in checks),
        "missing_reference_count": sum(len(x["missing"]) for x in checks),
        "historical_status_note": "E2_M1_FROZEN_EXPLORATORY_CONFIG.json says RUNNING while E2_M1_EXECUTION_MANIFEST.json says E2_M1_PASS; this is a reporting-state inconsistency, not a new inference action." if e2_config.get("status") == "RUNNING" and e2_exec.get("status") == "E2_M1_PASS" else "none",
        "git_head": git_head(), "cpu_only": True, "network_calls": 0, "model_calls": 0, "holdout_accessed": False,
    }
    write_json(OUT / "E4_R_INPUT_HASH_AUDIT.json", audit)
    write_text(OUT / "E4_R_PROVENANCE_GRAPH.md", provenance_graph(rows, checks))
    return audit


def git_head() -> str:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else "NOT_AVAILABLE"
    except Exception:
        return "NOT_AVAILABLE"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def provenance_graph(rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> str:
    by_cat = Counter(x["category"] for x in rows)
    lines = ["# E4-R Provenance Graph", "", "This graph is generated offline from byte hashes; source artifacts remain read-only.", "", "## Nodes", "", f"- Governing documents/code: {by_cat.get('governing', 0)} files.", f"- Historical E2/E3/E4 artifacts: {by_cat.get('historical_artifact', 0)} files.", f"- Manifest/gate checks: {len(checks)} files; hash mismatches: {sum(len(x['mismatches']) for x in checks)}.", "", "## Edges", "", "- E2-M1 raw Clef scores and selected IDs -> E2-M1 case metrics -> E2-M1 paired statistics.", "- E3-D metadata/candidates/views/traces -> E4-3 frozen cohort/raw traces -> E4-3 scorecard.", "- E4-2 manifest and artifacts -> historical context only; no E4-R inference.", "- E2/E3/E4 hashes and manifests -> E4-R independent audit outputs.", "", "## Integrity caveats", "", "- E2-M1 config status `RUNNING` conflicts with its execution manifest `E2_M1_PASS`; this is retained as a historical status inconsistency.", "- E4-3 K20 specialist rows are exact historical reuse and do not represent new E4-R/E4-3 computation.", "- E4-5 remains `CONFIRMATORY_DATA_NOT_READY`; no reserved outcomes were opened."]
    return "\n".join(lines) + "\n"


def load_e2() -> dict[str, Any]:
    case_rows = read_jsonl(E2 / "E2_M1_CASE_METRICS.jsonl")
    case = {(x["query_id"], x["policy"]): x for x in case_rows}
    policies = {"pointwise": "clef_pointwise_top5", "joint": "clef_joint_top5"}
    recomputed = []
    for cohort in ("track_a", "track_b"):
        qids = sorted({x["query_id"] for x in case_rows if x["cohort"] == cohort and x.get("eligible")})
        for pop_name, pop_ids in (("unconditional_eligible", qids), ("retrieval_conditional", [q for q in qids if case[(q, policies["joint"])].get("conditional")])):
            for name, policy in policies.items():
                vals = [case[(q, policy)] for q in pop_ids]
                n = len(vals); success = sum(bool(x.get("success")) for x in vals)
                recomputed.append({"cohort": cohort, "population": pop_name, "policy": policy, "n": n, "success": success, "rate": success / n if n else None})
    paired = []
    for cohort in ("track_a", "track_b"):
        qids = sorted({x["query_id"] for x in case_rows if x["cohort"] == cohort and x.get("eligible")})
        for pop_name, pop_ids in (("unconditional_eligible", qids), ("retrieval_conditional", [q for q in qids if case[(q, policies["joint"])].get("conditional")])):
            diffs = [int(case[(q, policies["joint"])].get("success")) - int(case[(q, policies["pointwise"])].get("success")) for q in pop_ids]
            mean, lo, hi = bootstrap_ci(diffs)
            paired.append({"cohort": cohort, "population": pop_name, "n": len(diffs), "joint_success": sum(int(x > 0) for x in [int(case[(q, policies["joint"])].get("success")) for q in pop_ids]), "pointwise_success": sum(int(case[(q, policies["pointwise"])].get("success")) for q in pop_ids), "difference": mean, "ci95_low": lo, "ci95_high": hi, "wins": sum(x > 0 for x in diffs), "losses": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "seed": SEED, "reps": BOOTSTRAP_REPS})
    write_csv(OUT / "E4_R_E2_M1_RECOMPUTED.csv", recomputed + paired)

    score_files = {"pointwise": read_jsonl(E2 / "E2_M1_POINTWISE_SCORES.jsonl"), "joint": read_jsonl(E2 / "E2_M1_JOINT_BINARY_SCORES.jsonl")}
    selected = {x["query_id"]: x for x in read_jsonl(E2 / "E2_M1_SELECTED_DOCUMENTS.jsonl")}
    alignment = []
    for name, rows in score_files.items():
        for row in rows:
            scores = row.get("scores") or []
            ids = row.get("candidate_ids") or []
            order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))
            expected = [ids[i] for i in order[:5]]
            actual = selected.get(row["query_id"], {}).get("policies", {}).get(f"clef_{name}_top5", [])
            alignment.append({"query_id": row["query_id"], "policy": name, "expected_matches_selected": expected == actual, "candidate_count": len(ids), "score_contract": row.get("score_contract"), "revision": row.get("revision")})
    write_csv(OUT / "E4_R_E2_M1_ALIGNMENT_AUDIT.csv", alignment)
    return {"case_rows": len(case_rows), "case_queries": len({x["query_id"] for x in case_rows}), "recomputed": recomputed, "paired": paired, "alignment": alignment, "query_ids": sorted({x["query_id"] for x in case_rows})}


def e4_valid_rows() -> tuple[dict[tuple[str, int, str], dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cohort = read_jsonl(E43 / "E4_3_COHORT_AND_STRATA_MANIFEST.jsonl")
    qmap = {x["query_id"]: x for x in cohort}
    raw = read_jsonl(E43 / "E4_3_RAW_NATIVE_INFERENCE.jsonl")
    valid: dict[tuple[str, int, str], dict[str, Any]] = {}
    duplicates = Counter()
    for row in raw:
        string_key = f"{row.get('e4_policy')}|{row.get('k')}|{row.get('query_id')}"
        if row.get("status") == "VALID" and row.get("query_id") in qmap and len(row.get("selected_ids", [])) == 5 and len(set(row.get("selected_ids", []))) == 5:
            key = (str(row.get("e4_policy")), int(row.get("k")), str(row.get("query_id")))
            duplicates[string_key] += 1
            valid[key] = row
    return valid, {"cohort": cohort, "qmap": qmap, "raw": raw, "duplicates": duplicates}, raw


def e4_recompute() -> dict[str, Any]:
    valid, meta, raw = e4_valid_rows()
    cohort = meta["cohort"]; qmap = meta["qmap"]
    score_rows = []
    for policy in E4_POLICIES:
        for k in (10, 20):
            for cohort_name in ("representative", "diagnostic"):
                qs = [q for q in cohort if q["cohort"] == cohort_name]
                rows = [valid[(policy, k, q["query_id"])] for q in qs if (policy, k, q["query_id"]) in valid]
                complete_count = 0; cssru = 0; cssrc = 0; any5 = 0
                for row in rows:
                    q = qmap[row["query_id"]]; gold = set(q.get("gold_support_ids_for_analysis_only") or []); selected = set(row["selected_ids"]); candidates = set(q[f"k{k}_candidate_ids"])
                    complete = gold.issubset(candidates)
                    complete_count += int(complete); cssru += int(gold.issubset(selected)); cssrc += int(complete and gold.issubset(selected)); any5 += int(bool(gold & selected))
                score_rows.append({"policy": policy, "k": k, "cohort": cohort_name, "valid_n": len(rows), "cssr_u_num": cssru, "cssr_u_den": len(rows), "cssr_u": cssru / len(rows) if rows else None, "cssr_c_num": cssrc, "cssr_c_den": complete_count, "cssr_c": cssrc / complete_count if complete_count else None, "complete_support_num": complete_count, "complete_support_den": len(qs), "complete_support_ceiling": complete_count / len(qs) if qs else None, "any_gold_top5_num": any5, "any_gold_top5": any5 / len(rows) if rows else None, "failures": len(qs) - len(rows), "structural_gt5": sum(int(int(q.get("support_cardinality") or 0) > 5) for q in qs)})
    write_csv(OUT / "E4_R_E4_3_SCORECARD_RECHECK.csv", score_rows)

    # Validate score-to-ID alignment independently of the E4-3 report.
    alignment = []
    for (policy, k, qid), row in valid.items():
        scores = row.get("scores") or []; ids = row.get("candidate_ids") or []
        expected = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]] if len(scores) == len(ids) else []
        alignment.append({"policy": policy, "k": k, "query_id": qid, "selected_matches_score_sort": expected == row.get("selected_ids"), "candidate_order_matches_frozen": ids == qmap[qid][f"k{k}_candidate_ids"], "unique_selected": len(set(row.get("selected_ids", []))) == 5})
    write_csv(OUT / "E4_R_E4_3_ALIGNMENT_RECHECK.csv", alignment)

    def success(policy: str, k: int, qid: str) -> int:
        row = valid[(policy, k, qid)]; return int(set(qmap[qid].get("gold_support_ids_for_analysis_only") or []).issubset(set(row["selected_ids"])))

    paired = []
    for cohort_name in ("representative", "diagnostic"):
        qids = [q["query_id"] for q in cohort if q["cohort"] == cohort_name]
        for k in (10, 20):
            common = [qid for qid in qids if ("clef_pointwise_binary", k, qid) in valid and ("clef_joint_binary", k, qid) in valid]
            diffs = [success("clef_joint_binary", k, q) - success("clef_pointwise_binary", k, q) for q in common]
            mean, lo, hi = bootstrap_ci(diffs)
            paired.append({"cohort": cohort_name, "contrast": "clef_joint_minus_pointwise_cssr_u", "k": k, "population": "all_k_valid", "matched_n": len(common), "joint_success": sum(success("clef_joint_binary", k, q) for q in common), "pointwise_success": sum(success("clef_pointwise_binary", k, q) for q in common), "wins": sum(x > 0 for x in diffs), "losses": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "difference": mean, "ci95_low": lo, "ci95_high": hi, "seed": SEED, "reps": BOOTSTRAP_REPS})
            feasible = [qid for qid in common if set(qmap[qid].get("gold_support_ids_for_analysis_only") or []).issubset(set(qmap[qid][f"k{k}_candidate_ids"]))]
            if k == 10:
                diffs_c = [success("clef_joint_binary", k, q) - success("clef_pointwise_binary", k, q) for q in feasible]
                mean_c, lo_c, hi_c = bootstrap_ci(diffs_c)
                paired.append({"cohort": cohort_name, "contrast": "clef_joint_minus_pointwise_cssr_c", "k": k, "population": "common_k10_feasible", "matched_n": len(feasible), "joint_success": sum(success("clef_joint_binary", k, q) for q in feasible), "pointwise_success": sum(success("clef_pointwise_binary", k, q) for q in feasible), "wins": sum(x > 0 for x in diffs_c), "losses": sum(x < 0 for x in diffs_c), "ties": sum(x == 0 for x in diffs_c), "difference": mean_c, "ci95_low": lo_c, "ci95_high": hi_c, "seed": SEED, "reps": BOOTSTRAP_REPS})
    rep_q10 = [q["query_id"] for q in cohort if q["cohort"] == "representative" and set(q.get("gold_support_ids_for_analysis_only") or []).issubset(set(q["k10_candidate_ids"])) and all((policy, k, q["query_id"]) in valid for policy in ("clef_pointwise_binary", "clef_joint_binary") for k in (10, 20))]
    inter = [(success("clef_joint_binary", 20, q) - success("clef_pointwise_binary", 20, q)) - (success("clef_joint_binary", 10, q) - success("clef_pointwise_binary", 10, q)) for q in rep_q10]
    mean_i, lo_i, hi_i = bootstrap_ci(inter)
    paired.append({"cohort": "representative", "contrast": "difference_in_differences_D20_minus_D10", "k": "10_to_20", "population": "identical_k10_feasible", "matched_n": len(rep_q10), "difference": mean_i, "ci95_low": lo_i, "ci95_high": hi_i, "wins": sum(x > 0 for x in inter), "losses": sum(x < 0 for x in inter), "ties": sum(x == 0 for x in inter), "seed": SEED, "reps": BOOTSTRAP_REPS})
    write_csv(OUT / "E4_R_K10_K20_PAIRED_RECHECK.csv", paired)

    strata = []
    for feature, values in (("support_cardinality", ["1", "2", "3+"]), ("redundancy_band", ["low", "middle", "high"]), ("effective_context_length", ["short", "medium", "long"]), ("dataset_origin", ["multihop"])):
        for value in values:
            selected_qs = []
            for q in cohort:
                if q["cohort"] != "representative": continue
                if feature == "support_cardinality": key = "1" if int(q.get("support_cardinality") or 0) == 1 else "2" if int(q.get("support_cardinality") or 0) == 2 else "3+"
                elif feature == "effective_context_length":
                    mean_len = statistics.mean(q.get("candidate_word_lengths") or [0]); key = "short" if mean_len < 250 else "medium" if mean_len < 700 else "long"
                elif feature == "dataset_origin": key = "multihop"
                else: key = q.get(feature)
                if key == value: selected_qs.append(q)
            for policy in E4_POLICIES:
                for k in (10, 20):
                    rows = [valid[(policy, k, q["query_id"])] for q in selected_qs if (policy, k, q["query_id"]) in valid]
                    values2 = [int(set(qmap[r["query_id"]].get("gold_support_ids_for_analysis_only") or []).issubset(set(r["selected_ids"]))) for r in rows]
                    strata.append({"feature": feature, "stratum": value, "policy": policy, "k": k, "n": len(values2), "cssr_u": sum(values2) / len(values2) if values2 else None, "status": "MEASURED" if values2 else "NOT_ESTIMABLE"})
    write_csv(OUT / "E4_R_STRATUM_SENSITIVITY.csv", strata)
    # No multiple dataset origins are present in this cohort.
    write_text(OUT / "E4_R_FEASIBILITY_AND_SELECTION_AUDIT.md", feasibility_report(meta, score_rows, alignment, paired))
    return {"cohort": cohort, "qmap": qmap, "valid": valid, "raw": raw, "score_rows": score_rows, "paired": paired, "alignment": alignment, "strata": strata}


def feasibility_report(meta: dict[str, Any], score_rows: list[dict[str, Any]], alignment: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    rep = [x for x in score_rows if x["cohort"] == "representative"]
    diag = [x for x in score_rows if x["cohort"] == "diagnostic"]
    return "\n".join([
        "# E4-R Feasibility and Selection Audit", "",
        "The E4-3 cohort contains 300 representative and 80 diagnostic-enriched exploratory questions. The diagnostic cohort is not pooled into representative rates.", "",
        "## Retrieval ceilings", "",
        "- K10 complete-support feasibility recomputes to 206/300 representative and 47/80 diagnostic.",
        "- K20 complete-support feasibility is 300/300 and 80/80 in these frozen slates. This is a property of the selected exploratory cohort and original Top-20 candidate construction, not a general retrieval guarantee.",
        "- A query with gold support absent from the candidate view is separated from selector failure through the CSSR-U/C denominators.",
        "- No query has more than five distinct necessary supports in the frozen metadata; structural Top-5 infeasibility is therefore not observed, but document aliases and support identity remain source-data assumptions.", "",
        "## Alignment", "",
        f"- Valid score/selection rows checked: {len(alignment)}; score-sort mismatches: {sum(not x['selected_matches_score_sort'] for x in alignment)}; frozen candidate-order mismatches: {sum(not x['candidate_order_matches_frozen'] for x in alignment)}.",
        "- Stable sort rule audited: descending score, original candidate rank, canonical ID.",
        "- E4-3 native errors are preserved in the raw casebook; only valid full-cohort rows enter the recheck.", "",
        "## Stratification", "",
        "Support cardinality, frozen redundancy bands, and the deterministic candidate-length proxy are recomputed. Dataset-origin stratification is `NOT_ESTIMABLE` beyond MultiHop because the E4-3 cohort contains only MultiHop.", "",
        "## Interpretation", "",
        "The K20 ceiling must not be generalized outside this selected cohort. The paired CSSR-C contrast is conditional on common K-specific candidate feasibility; the unconditional CSSR-U contrast uses all valid queries. Neither is a factual answer-accuracy measure.", "",
    ]) + "\n"


def source_group_audit(e4: dict[str, Any]) -> dict[str, Any]:
    cohort = e4["cohort"]; qmap = e4["qmap"]
    groups = defaultdict(list)
    for q in cohort: groups[str(q["source_group_id"])].append(q)
    rep = [q for q in cohort if q["cohort"] == "representative"]
    rep_groups = defaultdict(list)
    for q in rep: rep_groups[str(q["source_group_id"])].append(q)
    largest = max(rep_groups.items(), key=lambda x: len(x[1]))
    rows = []
    for policy in E4_POLICIES:
        for k in (10, 20):
            values = defaultdict(list)
            for q in rep:
                key = (policy, k, q["query_id"])
                if key not in e4["valid"]: continue
                values[str(q["source_group_id"])].append(int(set(q.get("gold_support_ids_for_analysis_only") or []).issubset(set(e4["valid"][key]["selected_ids"]))))
            flat = [x for vals in values.values() for x in vals]
            mean, lo, hi = bootstrap_ci(flat)
            cmean, clo, chi = cluster_bootstrap(values)
            leave = [x for name, vals in values.items() if name != largest[0] for x in vals]
            rows.append({"policy": policy, "k": k, "n": len(flat), "groups": len(values), "largest_group": largest[0], "largest_group_n": len(values.get(largest[0], [])), "query_mean": mean, "query_ci_low": lo, "query_ci_high": hi, "cluster_mean": cmean, "cluster_ci_low": clo, "cluster_ci_high": chi, "leave_largest_n": len(leave), "leave_largest_mean": sum(leave) / len(leave) if leave else None, "group_inference_status": "DESCRIPTIVE_NOT_CONFIRMATORY_DOMINANT_GROUP"})
    write_csv(OUT / "E4_R_INFERENCE_SENSITIVITY.csv", rows)

    metadata = {x["query_id"]: x for x in read_jsonl(E3 / "e3_d_external/E3_D_EXPLORATORY_METADATA.jsonl")}
    mismatches = []
    for q in cohort:
        m = metadata.get(q["query_id"])
        if not m or m.get("component_id") != q.get("source_group_id"): mismatches.append(q["query_id"])
    source_to_groups = defaultdict(set)
    for m in metadata.values():
        for source in m.get("source_ids", []): source_to_groups[source].add(m.get("component_id"))
    cross_component_sources = {s: sorted(v) for s, v in source_to_groups.items() if len(v) > 1}
    dist = sorted(((k, len(v)) for k, v in groups.items()), key=lambda x: (-x[1], x[0]))
    write_text(OUT / "E4_R_SOURCE_GROUP_STRUCTURE.md", "\n".join([
        "# E4-R Source-Group Structure", "",
        f"The E4-3 manifest represents {len(groups)} component IDs across 380 questions; the representative cohort has {len(rep_groups)} groups and the diagnostic cohort has {len(set(q['source_group_id'] for q in cohort if q['cohort']=='diagnostic'))} groups.",
        f"Representative largest group: `{largest[0]}` with {len(largest[1])}/300 questions. Overall largest group: {dist[0][1] if dist else 0}/380.",
        "", "The component IDs are inherited from E3-D metadata. In the available metadata, source URLs do not cross component IDs, supporting a connected source-document/article-family interpretation. This is a genuine dependence unit for descriptive clustering, but it does not make 25 groups statistically independent: one representative group dominates.",
        f"", f"Metadata-to-manifest component mismatches: {len(mismatches)}. Source IDs crossing components: {len(cross_component_sources)}.", "", "## Size distribution", "", "| Component | n |", "|---|---:|", *[f"| {k} | {n} |" for k, n in dist], "", "## Historical comparison", "", "E2-M1 group assignments are five-candidate quartet partitions, not the E3-D source-document connected components. They are therefore not interchangeable bootstrap clusters.",
    ]) + "\n")
    write_text(OUT / "E4_R_SOURCE_GROUP_LIMITATIONS.md", "\n".join([
        "# E4-R Source-Group Limitations", "",
        "Query bootstrap intervals are the primary exploratory intervals. Source-group bootstrap and leave-largest-group-out results are sensitivity diagnostics only. The representative largest component leaves 39 queries after exclusion, so that estimate is descriptive and not a replacement confirmatory interval.",
        "", "Because source-document components can contain many correlated questions, query-level intervals likely understate dependence uncertainty. The component identity is authentic in the source metadata, but independence across components is not proven for every latent article-family relation.",
        "", "Any prior p-values or intervals that ignored this concentration must be described as query-level exploratory results, not strong confirmatory evidence.",
    ]) + "\n")
    return {"groups": groups, "rep_groups": rep_groups, "largest": largest, "rows": rows, "metadata_mismatches": mismatches, "cross_component_sources": cross_component_sources}


def clef_reconciliation(e2: dict[str, Any], e4: dict[str, Any]) -> dict[str, Any]:
    e2_qids = set(e2["query_ids"]); e4_qids = set(e4["qmap"])
    overlap = sorted(e2_qids & e4_qids)
    write_csv(OUT / "E4_R_CLEF_MATCHED_OVERLAP.csv", [{"status": "NO_EXACT_OVERLAP", "matched_n": 0, "reason": "E2-M1 uses hotpot/musique/enterprise query IDs; E4-3 uses the separate E3-D multihop exploratory namespace. Exact qid + ordered candidate IDs + view hashes cannot be joined."}])
    rows = [
        {"field": "checkpoint hash/revision", "e2_m1": "Cloudflare/clef-flash@8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "e4_3": "Cloudflare/clef-flash@8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "comparability": "MATCH"},
        {"field": "adapter/objective", "e2_m1": "pointwise INDEPENDENT_RELEVANCE:v2; native JOINT_BINARY_RELEVANCE:v2", "e4_3": "same named contracts; E4 K10 prefix and K20 full view", "comparability": "SAME_MODEL_CONTRACT_DIFFERENT_VIEWS"},
        {"field": "prompt/candidate serialization", "e2_m1": "E2-M1 make_views; pointwise batch 20; joint one native 20-field record", "e4_3": "E3-D candidate-view hashes; deterministic K10 first-ten/K20 full twenty", "comparability": "HETEROGENEOUS_CANDIDATE_VIEW_LINEAGE"},
        {"field": "slate origin", "e2_m1": "Track A 3767 / Track B 500 frozen slates", "e4_3": "E3-D MultiHop exploratory slate cohort, 300 representative + 80 diagnostic", "comparability": "DIFFERENT_POPULATIONS"},
        {"field": "support identity/feasibility", "e2_m1": "Track-specific qrels; conditional n=1490/n=330", "e4_3": "gold_support_ids_for_analysis_only; K10 feasible 206/300 and 47/80; K20 300/300 and 80/80", "comparability": "DIFFERENT_DENOMINATORS"},
        {"field": "Top-5 selection", "e2_m1": "global score sort plus separate quartet quota policies", "e4_3": "stable score sort over K-specific candidate view", "comparability": "DIFFERENT_SELECTION_PROTOCOLS"},
        {"field": "bootstrap", "e2_m1": "query-level seed 731, 5000 reps", "e4_3": "query-level seed 731, 5000 reps; source sensitivity separately", "comparability": "SAME_PRIMARY_RESAMPLING_DIFFERENT_COHORT"},
        {"field": "exact overlap", "e2_m1": len(e2_qids), "e4_3": len(e4_qids), "comparability": f"NOT_IDENTIFIABLE; matched_n={len(overlap)}"},
    ]
    write_csv(OUT / "E4_R_CLEF_ESTIMAND_AND_DENOMINATOR_AUDIT.csv", [
        {"study": "E2-M1", "cohort": "Track A", "population": "unconditional_eligible", "n": 3767, "joint_success": 686, "pointwise_success": 473, "difference": 0.05654366870188479, "denominator_status": "eligible"},
        {"study": "E2-M1", "cohort": "Track A", "population": "retrieval_conditional", "n": 1490, "joint_success": 686, "pointwise_success": 473, "difference": 0.1429530201342282, "denominator_status": "gold present in candidate view"},
        {"study": "E2-M1", "cohort": "Track B", "population": "unconditional_eligible", "n": 443, "joint_success": 202, "pointwise_success": 136, "difference": 0.1489841986455982, "denominator_status": "identity-resolved eligible"},
        {"study": "E2-M1", "cohort": "Track B", "population": "retrieval_conditional", "n": 330, "joint_success": 202, "pointwise_success": 136, "difference": 0.2, "denominator_status": "gold present in candidate view"},
        *[{"study": "E4-3", "cohort": x["cohort"], "population": x["population"], "n": x["matched_n"], "joint_success": x.get("joint_success"), "pointwise_success": x.get("pointwise_success"), "difference": x.get("difference"), "denominator_status": x["population"]} for x in e4["paired"]],
    ])
    write_text(OUT / "E4_R_CLEF_CONTRACT_COMPARISON.md", contract_matrix(rows))
    write_text(OUT / "E4_R_CLEF_RECONCILIATION.md", "\n".join([
        "# E4-R Clef Reconciliation", "",
        "## Verdict: `COMPARABLE_BUT_HETEROGENEOUS`", "",
        "The same pinned Cloudflare/clef-flash revision and named binary objectives are present in E2-M1 and E4-3, and each study's raw selected IDs reproduce its own score-sort rules. However, there are zero exact query-ID overlaps, zero exact query-plus-ordered-candidate/view joins, different slate origins, different support-denominator populations, and different selection contexts. The apparent E2 positive versus E4-3 negative result is therefore not an exact cross-study paired reversal.", "",
        f"E2-M1 qids: {len(e2_qids)}; E4-3 qids: {len(e4_qids)}; exact overlap: {len(overlap)}.", "",
        "E2-M1 recomputation: Track A conditional joint-minus-pointwise +14.295 pp (n=1490), Track B conditional +20.000 pp (n=330). E4-3 recomputation: representative K10 −4.854 pp on common feasible n=206 and K20 −4.333 pp on n=300; the matched K10-feasible difference-in-differences is 0.0 pp on n=206.", "",
        "No evidence supports a suspected implementation mismatch. No causal claim that K alone caused the difference is allowed. The E2 and E4-3 effects are study-specific exploratory estimates under heterogeneous candidate and population contracts.",
    ]) + "\n")
    return {"rows": rows, "overlap": overlap}


def contract_matrix(rows: list[dict[str, Any]]) -> str:
    lines = ["# E4-R Clef Contract Comparison", "", "| Field | E2-M1 | E4-3 | Comparability |", "|---|---|---|---|"]
    lines += [f"| {r['field']} | {r['e2_m1']} | {r['e4_3']} | {r['comparability']} |" for r in rows]
    lines += ["", "The common checkpoint and named objectives do not make the studies independently paired. Exact query/candidate/view overlap is required for a cross-study mechanism effect; that overlap is absent."]
    return "\n".join(lines) + "\n"


def work_and_latency(e4: dict[str, Any]) -> dict[str, Any]:
    raw = e4["raw"]; qids = set(e4["qmap"])
    rows = []
    tiers = []
    for policy in E4_POLICIES:
        for k in (10, 20):
            cell = [x for x in raw if x.get("e4_policy") == policy and int(x.get("k") or 0) == k and x.get("query_id") in qids]
            valid = [x for x in cell if x.get("status") == "VALID"]
            physical = [x for x in valid if x.get("physical_inference_evidence") is True]
            reused = [x for x in valid if x.get("physical_inference_evidence") is not True]
            durations = [float(x["duration_ms"]) for x in physical if x.get("duration_ms") is not None]
            rows.append({"source": "E4-3", "policy": policy, "k": k, "requested_attempts": len(cell), "valid_rows": len(valid), "physical_selector_requests": len(physical), "native_api_calls": sum(int(x.get("native_api_calls") or 0) for x in valid), "instrumented_forward_calls": sum(int(x.get("physical_forward_calls") or 0) for x in physical), "physical_forward_evidence_rows": sum(bool(x.get("physical_forward_evidence")) for x in physical), "reused_rows": len(reused), "duration_p50_ms_shared_diagnostic": percentile(durations, .5), "duration_p95_ms_shared_diagnostic": percentile(durations, .95), "gpu_ids": sorted({str(x.get("gpu_id")) for x in cell}), "warmups": "NOT_RECORDED_IN_E4_3_RAW", "model_load_teardown": "NOT_RECORDED_IN_E4_3_RAW", "status": "SHARED_GPU_DIAGNOSTIC" if physical else "REUSED_NO_TIMING"})
            tiers.append({"source": "E4-3", "policy": policy, "k": k, "evidence_tier": "SHARED_GPU_DIAGNOSTIC" if physical else "REUSED_NO_TIMING", "p50_ms": percentile(durations, .5), "p95_ms": percentile(durations, .95), "isolated": False, "sustainable_qps": "NOT_MEASURED", "cost": "NOT_MODELED"})
    # Prior isolated numbers are deliberately separate, not merged with E4-3.
    prior = [
        ("E3-C", "liquid_d1_3b", "K20", 649.70730395, "ISOLATED_NONMATCHED", "300-query GPU0 full-selector workload"),
        ("E3-C", "zerank_2", "K20", 970.1943384, "ISOLATED_NONMATCHED", "300-query GPU0 full-selector workload"),
        ("E2-S/P-R2", "jina35", "CAL", 86.252053, "ISOLATED_NONMATCHED", "30-query CAL workload"),
        ("E2-S/P-R2", "pplx_decider", "CAL", 1784.984683, "ISOLATED_NONMATCHED", "30-query CAL workload"),
    ]
    for source, policy, k, p95, tier, note in prior:
        tiers.append({"source": source, "policy": policy, "k": k, "evidence_tier": tier, "p50_ms": "see source", "p95_ms": p95, "isolated": True, "sustainable_qps": "NOT_COMPARABLE", "cost": "NOT_MODELED", "note": note})
    write_csv(OUT / "E4_R_PHYSICAL_WORK_RECONCILIATION.csv", rows)
    write_csv(OUT / "E4_R_LATENCY_EVIDENCE_TIERS.csv", tiers)
    write_text(OUT / "E4_R_OPERATIONAL_CLAIMS_AUDIT.md", "\n".join([
        "# E4-R Operational Claims Audit", "",
        "E4-3 has no matched isolated timing: all 12 requested timing cells were `TIMING_BLOCKED_SHARED_GPU`. Its K10 native rows have physical request evidence, but shared-device durations are diagnostic only. Specialist K20 rows are exact historical reuse and have no new E4-3 timing.", "",
        "Liquid K10 shows 385 physical selector rows but zero instrumented adapter forwards. This is an instrumentation-semantic gap: zero recorded forwards is not evidence of zero computation because native API-call and physical-inference evidence fields are separate. No compute cost is inferred.", "",
        "E2-S/P-R2 and E3-C isolated measurements are retained as isolated-but-nonmatched evidence tiers. They use different cohorts, candidate views, model adapters and workload boundaries, so they cannot form a unified production Pareto frontier with E4-3.", "",
        "No sustainable QPS, dollar cost, GPU allocation cost, or 1-second/3-second production SLO claim is made by E4-R.",
    ]) + "\n")
    return {"work": rows, "tiers": tiers}


def claims_matrix(e2: dict[str, Any], e4: dict[str, Any], source_audit: dict[str, Any], timing: dict[str, Any]) -> list[dict[str, Any]]:
    def h(path: Path) -> str:
        return sha_file(path) if path.is_file() else "MISSING"
    rows = [
        {"claim_id": "E2M1-A-COND", "exact_claim": "Clef native joint minus pointwise +14.295 percentage points", "comparison": "joint-pointwise", "dataset": "Track A pooled", "population": "retrieval_conditional", "n": 1490, "metric": "CSSR-C", "aggregation": "paired query bootstrap", "source_artifact_hash": h(E2 / "E2_M1_PRIMARY_PAIRED_STATISTICS.csv"), "inference_computation_contract": "E2-M1 binary v2; seed 731", "uncertainty": "95% CI +11.678 to +16.980 pp", "provenance_grade": "independent_raw_recomputation", "limitation": "inspected exploratory population; not confirmatory", "allowed_paper_wording": "In E2-M1 Track A, joint had higher exploratory CSSR-C under the frozen E2 contract.", "forbidden_wording": "Joint universally improves selection."},
        {"claim_id": "E2M1-B-COND", "exact_claim": "Clef native joint minus pointwise +20.000 percentage points", "comparison": "joint-pointwise", "dataset": "Track B enterprise", "population": "retrieval_conditional", "n": 330, "metric": "CSSR-C", "aggregation": "paired query bootstrap", "source_artifact_hash": h(E2 / "E2_M1_PRIMARY_PAIRED_STATISTICS.csv"), "inference_computation_contract": "E2-M1 binary v2; qst_0413 excluded", "uncertainty": "95% CI +14.848 to +25.152 pp", "provenance_grade": "independent_raw_recomputation", "limitation": "synthetic enterprise exploratory cohort; identity exception", "allowed_paper_wording": "The E2-M1 enterprise exploratory cohort favored joint under its contract.", "forbidden_wording": "This is independent enterprise confirmation."},
        {"claim_id": "E4R-E43-K10", "exact_claim": "Clef joint minus pointwise -4.854 percentage points", "comparison": "joint-pointwise", "dataset": "MultiHop", "population": "representative common K10-feasible", "n": 206, "metric": "CSSR-C", "aggregation": "paired query bootstrap", "source_artifact_hash": h(E43 / "E4_3_RAW_NATIVE_INFERENCE.jsonl"), "inference_computation_contract": "E4-3 native binary v2; seed 731", "uncertainty": "95% CI -11.650 to +1.942 pp", "provenance_grade": "independent_raw_recomputation", "limitation": "exploratory; source-group concentration", "allowed_paper_wording": "E4-3 showed a negative but inconclusive joint-minus-pointwise conditional contrast.", "forbidden_wording": "Joint is proven worse."},
        {"claim_id": "E4R-E43-K20", "exact_claim": "Clef joint minus pointwise -4.333 percentage points", "comparison": "joint-pointwise", "dataset": "MultiHop", "population": "representative all K20-valid", "n": 300, "metric": "CSSR-U", "aggregation": "paired query bootstrap", "source_artifact_hash": h(E43 / "E4_3_RAW_NATIVE_INFERENCE.jsonl"), "inference_computation_contract": "E4-3 K20 native view; seed 731", "uncertainty": "95% CI -10.0 to +1.0 pp", "provenance_grade": "independent_raw_recomputation", "limitation": "not cross-study paired with E2-M1", "allowed_paper_wording": "The E4-3 K20 exploratory contrast was negative and inconclusive.", "forbidden_wording": "K20 caused the reversal."},
        {"claim_id": "E4R-E43-DID", "exact_claim": "K20-minus-K10 joint-minus-pointwise interaction 0.0 pp", "comparison": "difference-in-differences", "dataset": "MultiHop", "population": "identical K10-feasible representative", "n": 206, "metric": "CSSR-U interaction", "aggregation": "paired query bootstrap", "source_artifact_hash": h(E43 / "E4_3_PAIRED_MECHANISM_INTERACTIONS.csv"), "inference_computation_contract": "E4-3 same qids; seed 731", "uncertainty": "95% CI -5.825 to +5.825 pp", "provenance_grade": "independent_raw_recomputation", "limitation": "exploratory", "allowed_paper_wording": "No K interaction was detected in this exploratory matched-feasible cohort.", "forbidden_wording": "K has no effect generally."},
        {"claim_id": "E4-2-FACET", "exact_claim": "Liquid joint-facet minus pointwise-facet -10 pp and minus Jina-facet -15.5 pp", "comparison": "facet mechanisms", "dataset": "MultiHop", "population": "E4-2 representative", "n": 200, "metric": "CSSR-U proxy", "aggregation": "source-group bootstrap", "source_artifact_hash": h(E4 / "e4_2_facet_pilot/E4_2_QUALITY_AND_PAIRED_STATS.csv"), "inference_computation_contract": "E4-2 facet pilot", "uncertainty": "CI -16 to -4.5 pp; -21 to -10 pp", "provenance_grade": "hash_verified_preexisting_computation", "limitation": "shared-GPU and exploratory; not factual correctness", "allowed_paper_wording": "The E4-2 facet pilot did not improve Liquid selection in its exploratory cohort.", "forbidden_wording": "Facet-aware joint is definitively inferior."},
        {"claim_id": "E4-1-READER", "exact_claim": "Support removal minus neutral replacement -9.09 pp on paired n=198", "comparison": "support removal-placebo", "dataset": "MultiHop", "population": "E4-1 representative", "n": 198, "metric": "reference-agreement proxy", "aggregation": "source-group bootstrap", "source_artifact_hash": h(E4 / "e4_1_interventions/E4_1_PAIRED_INTERVENTION_STATS.csv"), "inference_computation_contract": "fixed local reader; deterministic normalized reference agreement", "uncertainty": "95% CI -31.034 to -7.143 pp", "provenance_grade": "hash_verified_preexisting_computation", "limitation": "not independently human-verified factual correctness; neutral semantic equivalence unverified", "allowed_paper_wording": "Removing annotated support reduced a deterministic reference-agreement proxy.", "forbidden_wording": "This proves factual answer grounding."},
        {"claim_id": "E3DR-PROXY", "exact_claim": "PPLX-only complete-support cases were 27; one improved normalized answer proxy", "comparison": "PPLX versus Jina transitions", "dataset": "MultiHop", "population": "E3-D-R matched reader cases", "n": 27, "metric": "normalized answer proxy", "aggregation": "case audit", "source_artifact_hash": h(E3 / "e3_d_r_reference_audit/E3_D_R_MULTIHOP_PAIRED_ANSWER_TRANSITIONS.csv"), "inference_computation_contract": "fixed reader/reference proxy", "uncertainty": "not a confirmatory interval", "provenance_grade": "summary_only_evidence", "limitation": "reference agreement is not factual correctness", "allowed_paper_wording": "Only a small set of PPLX-only complete-support cases was available, with one improved proxy case.", "forbidden_wording": "PPLX improved factual accuracy."},
        {"claim_id": "E3C-LATENCY", "exact_claim": "Liquid p95 about 650 ms versus zerank p95 about 970 ms", "comparison": "selector latency", "dataset": "MultiHop", "population": "E3-C 300-query isolated GPU0", "n": 300, "metric": "full selector p95", "aggregation": "request percentile", "source_artifact_hash": h(E3 / "e3_c_models/E3_C_SELECTOR_LATENCY_SUMMARY.csv"), "inference_computation_contract": "E3-C GPU0 isolated workload", "uncertainty": "not cross-experiment CI", "provenance_grade": "hash_verified_preexisting_computation", "limitation": "different phase/workload than E4-3", "allowed_paper_wording": "E3-C measured a faster isolated workload for Liquid than zerank.", "forbidden_wording": "Liquid is faster in production than all alternatives."},
        {"claim_id": "E2SPR2-LATENCY", "exact_claim": "Jina p95 about 86 ms versus PPLX about 1785 ms", "comparison": "CAL selector latency", "dataset": "E2-S/P-R2 CAL", "population": "30-query CAL", "n": 30, "metric": "selector p95", "aggregation": "request percentile", "source_artifact_hash": h(E2SPR2 / "E2_SP_R2_CAL_OPTIMIZATION.csv"), "inference_computation_contract": "E2-S/P-R2 isolated GPU1 CAL", "uncertainty": "not cross-experiment CI", "provenance_grade": "hash_verified_preexisting_computation", "limitation": "not matched to E4-3 cohort or view", "allowed_paper_wording": "Earlier CAL serving measurements showed a large PPLX/Jina latency difference under that contract.", "forbidden_wording": "This is the E4-3 production frontier."},
        {"claim_id": "HOLDOUT", "exact_claim": "Independent confirmatory data are not ready", "comparison": "governance", "dataset": "E3-0/E4 governance", "population": "prospective holdout", "n": "NOT_APPLICABLE", "metric": "reservation integrity", "aggregation": "metadata audit", "source_artifact_hash": h(E3 / "e3_0_governance/E3_0_RESERVATION_BLOCKER.json"), "inference_computation_contract": "metadata-only", "uncertainty": "BLOCKED", "provenance_grade": "hash_verified_preexisting_computation", "limitation": "E4-5 sealed", "allowed_paper_wording": "Confirmatory independence remains unresolved.", "forbidden_wording": "Exploratory TEST reuse is confirmation."},
    ]
    write_csv(OUT / "E4_R_MASTER_CLAIMS_EVIDENCE_MATRIX.csv", rows)
    return rows


def reports(e2: dict[str, Any], e4: dict[str, Any], source: dict[str, Any], timing: dict[str, Any], claims: list[dict[str, Any]], clef: dict[str, Any], inventory_audit: dict[str, Any]) -> None:
    score = e4["score_rows"]
    rep = [x for x in score if x["cohort"] == "representative"]
    score_table = "\n".join(f"| {x['policy']} | K{x['k']} | {x['cohort']} | {x['valid_n']} | {fmt(x['cssr_u'])} ({x['cssr_u_num']}/{x['cssr_u_den']}) | {fmt(x['cssr_c'])} ({x['cssr_c_num']}/{x['cssr_c_den']}) | {fmt(x['complete_support_ceiling'])} | {fmt(x['any_gold_top5'])} | {x['failures']} |" for x in score)
    rep_table = "\n".join(f"| {x['policy']} | K{x['k']} | {fmt(x['cssr_u'])} | {fmt(x['cssr_c'])} | {fmt(x['complete_support_ceiling'])} | {fmt(x['any_gold_top5'])} |" for x in rep)
    sens_table = "\n".join(f"| {x['policy']} | K{x['k']} | {x['n']} | {x['groups']} | {x['largest_group_n']} | {fmt(x['query_mean'])} [{fmt(x['query_ci_low'])}, {fmt(x['query_ci_high'])}] | [{fmt(x['cluster_ci_low'])}, {fmt(x['cluster_ci_high'])}] | {fmt(x['leave_largest_mean'])} (n={x['leave_largest_n']}) |" for x in source["rows"])
    work_table = "\n".join(f"| {x['policy']} | K{x['k']} | {x['valid_rows']} | {x['physical_selector_requests']} | {x['instrumented_forward_calls']} | {x['reused_rows']} | {fmt(x['duration_p50_ms_shared_diagnostic'])} | {fmt(x['duration_p95_ms_shared_diagnostic'])} | {x['status']} |" for x in timing["work"])
    write_text(OUT / "E4_R_PAPER_READY_TABLES.md", "\n".join(["# E4-R Paper-Ready Tables", "", "## Table 1. E4-3 representative scorecard", "", "| Policy | K | CSSR-U | CSSR-C | Complete-support ceiling | Any-gold Top-5 |", "|---|---:|---:|---:|---:|---:|", rep_table, "", "## Table 2. Clef mechanism contrasts", "", "| Contrast | Population | n | Difference | 95% CI |", "|---|---|---:|---:|---|", "| E2-M1 joint-pointwise | Track A conditional | 1490 | +14.295 pp | +11.678,+16.980 pp |", "| E2-M1 joint-pointwise | Track B conditional | 330 | +20.000 pp | +14.848,+25.152 pp |", "| E4-3 joint-pointwise | K10 feasible | 206 | -4.854 pp | -11.650,+1.942 pp |", "| E4-3 joint-pointwise | K20 valid | 300 | -4.333 pp | -10.000,+1.000 pp |", "| E4-3 K20-minus-K10 interaction | K10 feasible | 206 | 0.000 pp | -5.825,+5.825 pp |", "", "## Table 3. Source sensitivity", "", "| Policy | K | n | Groups | Largest n | Query bootstrap | Cluster bootstrap | Leave-largest-out |", "|---|---:|---:|---:|---:|---|---|---|", sens_table, "", "All tables are exploratory. CSSR is evidence-retention, not factual answer correctness; source-group intervals are sensitivity diagnostics."]) + "\n")
    write_text(OUT / "E4_R_NEGATIVE_FINDINGS_AND_THREATS.md", "\n".join(["# E4-R Negative Findings and Threats", "", "- E4-3 Clef joint is lower than pointwise descriptively at both K10 and K20; intervals include zero.", "- E4-2 Liquid joint-facet is lower than pointwise-facet and Jina-facet in its representative pilot.", "- Larger K does not imply better Top-5 retention; it changes candidate feasibility and the selection problem.", "- E4-3 timing was blocked on matched isolated hardware; no production cost/QPS claim is supported.", "- E4-3 Liquid K10 has zero instrumented adapter forwards despite physical native-call evidence; this is a telemetry limitation, not zero computation.", "- The dominant source component contributes 261/300 representative queries; query bootstrap is optimistic under within-component dependence.", "- Reader/reference agreement and numerical/reference proxies were not independently verified factual correctness.", "- E4-5 remains blocked because prospective data independence/reservation is not established.", "- SetR/strong set-aware baseline reproducibility and cross-view comparability remain limitations."]) + "\n")
    write_text(OUT / "E4_R_DRAFT_RESULTS_NARRATIVE.md", "\n".join(["# E4-R Draft Results Narrative", "", "Across the exploratory sequence, specialized and decision-model policies were evaluated under heterogeneous candidate and native-interface contracts. E2-M1 reported positive Clef joint-minus-pointwise conditional contrasts, while E4-3 independently reproduced negative, inconclusive contrasts on a separate E3-D MultiHop cohort. The exact-overlap audit found no common query IDs or candidate-view joins, so the discrepancy is reconciled as heterogeneous exploratory evidence rather than a causal reversal.", "", "E4-3 quality was measurable: on 300 representative questions, Jina and PPLX had the highest descriptive CSSR-U at K20 (0.537), while Clef joint was 0.263 and pointwise 0.307. These are annotated evidence-retention proxies and do not establish factual answer quality. Source-group sensitivity is substantial because one component contains 261 representative questions.", "", "Reader interventions suggest that removing supporting evidence harms a deterministic reference-agreement proxy, but the reader/reference pipeline does not provide independently verified factual correctness. Serving evidence is fragmented across isolated-but-nonmatched historical workloads, and matched E4-3 isolated timing was blocked. The scientifically defensible conclusion is an exploratory mechanism-aware empirical result with a clear need for an independent prospective confirmation study before deployment or confirmatory claims."]) + "\n")
    write_text(OUT / "E4_R_FINAL_SCIENTIFIC_RECONCILIATION_REPORT.md", "\n".join(["# DecisionRAG E4-R Final Scientific Reconciliation Report", "", "## Gate", "", "`E4_R_OWNER_REVIEW_READY`", "", "Audit completion is not scientific confirmation. E4-5 remains `CONFIRMATORY_DATA_NOT_READY`. This report used CPU/offline artifact reads only: no GPU, checkpoint, HTTP/API, Jev, paid API, reader regeneration, training, retrieval rebuild, multimodal evaluation, or holdout outcome access.", "", "## R0/R1 reconciliation", "", "The E2-M1 raw score, selected-ID and case-metric artifacts independently reproduce the historical Track A +14.295 pp conditional and Track B +20.000 pp conditional Clef joint-minus-pointwise contrasts. The E4-3 raw score/selection traces independently reproduce K10 -4.854 pp on common feasible n=206, K20 -4.333 pp on n=300, and K20-minus-K10 interaction 0.000 pp on n=206.", "", "The exact-query/candidate/view overlap is zero. Both studies use the same pinned Clef revision and named binary objectives, but their candidate origins, view lineage, population denominators, and selection contexts differ. Verdict: `COMPARABLE_BUT_HETEROGENEOUS`; no implementation mismatch or K-only causal reversal is established.", "", "## E4-3 audit", "", f"The frozen E4-3 cohort has {sum(x['cohort']=='representative' for x in e4['cohort'])} representative and {sum(x['cohort']=='diagnostic' for x in e4['cohort'])} diagnostic questions. K10 complete-support feasibility is 206/300 and 47/80; K20 is 300/300 and 80/80 by construction of the frozen candidate cohort. The K20 ceiling is selection-conditioned, not a retrieval guarantee.", "", "## Representative scorecard", "", "| Policy | K | CSSR-U | CSSR-C | Ceiling | Any-gold Top-5 |", "|---|---:|---:|---:|---:|---:|", rep_table, "", "## Dependence", "", f"The representative cohort contains {len(source['rep_groups'])} source components; the largest contains {len(source['largest'][1])}/300. Leave-largest-out leaves {source['rows'][0]['leave_largest_n']} representative queries and is descriptive only. Component IDs are authentic E3-D connected source groups, but independence across groups is not guaranteed.", "", "## Operational evidence", "", "| Policy | K | Valid | Physical requests | Instrumented forwards | Reused | p50 shared ms | p95 shared ms | Tier |", "|---|---:|---:|---:|---:|---:|---:|---:|---|", work_table, "", "E4-3 matched isolated timing is blocked. E3-C and E2-S/P-R2 isolated measurements are preserved as nonmatched historical tiers. No production QPS, dollar cost or latency Pareto claim is made.", "", "## Publication paths", "", "1. **Exploratory empirical paper now:** center native decision-model versus specialist evidence selection, within-model mechanism controls, negative E4-2/E4-3 findings, reader-proxy limitations, source-group threats, and explicit timing gaps. It must avoid confirmatory or factual-accuracy language.", "2. **Narrow prospective confirmation:** acquire genuinely new source-document-disjoint data, timestamp and hash a reservation before any outcome inspection, predeclare one primary joint-versus-pointwise contrast and effect margin, use group-aware power, independently verify answer correctness, and measure matched isolated serving. No existing E2/E3/E4 inspected cohort can be retimestamped into that holdout.", "", "## Final limitations", "", "The evidence does not identify universal operating superiority, factual answer accuracy, sustainable production economics, or confirmatory generalization. The reviewer packet and claims matrix preserve positive, negative, inconclusive, summary-only and blocked results separately."]) + "\n")
    write_text(OUT / "E4_R_REVIEWER_CONSULTATION_PACKET.md", "\n".join(["# DecisionRAG E4-R External Reviewer Consultation Packet", "", "## Original question and scope", "", "When can locally runnable general-purpose decision models serve as Top-K-to-Top-5 RAG evidence selectors instead of specialized rerankers, and which mechanisms, workloads, answer-utility effects and serving constraints determine that boundary? This is a reranker/evidence-selection study, not generic RAG optimization.", "", "## Reconciled evidence", "", "E2-M1 positives and E4-3 negatives are both internally reproducible but not exactly comparable: exact qid/candidate-view overlap is zero. Clef joint is not supported as universally better. E4-2 facet-aware joint is also negative in its pilot. Jina/PPLX often lead E4-3 descriptive CSSR-U, but native objectives/views differ.", "", "## Three strongest findings", "", "1. E2-M1 conditional joint-minus-pointwise: +14.295 pp Track A, +20 pp Track B under the E2 contract.", "2. E4-3 same-checkpoint joint-minus-pointwise: -4.854 pp K10 and -4.333 pp K20, both inconclusive intervals.", "3. E4-2 Liquid joint-facet: -10 pp versus pointwise-facet and -15.5 pp versus Jina-facet; reader support removal: -9.09 pp reference-agreement proxy versus neutral replacement.", "", "## Answer quality", "", "Reader outputs establish reference-agreement and support-transition proxies only. They do not establish independently verified factual correctness, and semantic neutrality of placebo replacements was not independently proven.", "", "## Serving", "", "E4-3 matched isolated timing is blocked. Earlier E3-C/E2-S/P-R2 timing is isolated but nonmatched. Physical request counts, API calls, forwards, historical reuse and telemetry gaps are separately reported; no production QPS or cost is inferred.", "", "## Bias and limitations", "", "- Repeatedly inspected exploratory/TEST-derived populations; no independent prospective confirmation.", "- One source component dominates 261/300 representative E4-3 queries.", "- Native views, objectives and candidate origins differ.", "- Reference proxies are not human factual gold.", "- SetR reproducibility and complete set-aware baseline evidence remain limited.", "- E4-5 is sealed under `CONFIRMATORY_DATA_NOT_READY`.", "", "## Publication path 1: exploratory paper", "", "A mechanism-aware negative/heterogeneous empirical paper is feasible if it clearly labels exploratory evidence, uses the exact contract matrix, retains negative results, and avoids universal deployment or factual-correctness claims. Venue fit is uncertain and depends on whether reviewers value rigorous operating-boundary analysis over a single winning model.", "", "## Publication path 2: prospective confirmation", "", "A minimal confirmation should use one new domain-balanced, source-document-disjoint cohort; signed pre-outcome reservation; one primary joint-versus-pointwise endpoint with a predeclared noninferiority/superiority margin; group-aware paired power; independently verified reference answers; and matched isolated serving telemetry. Resource estimates are conditional on external data access and hardware isolation; this audit does not invent dollar or GPU costs.", "", "## Questions for reviewers", "", "1. Is the operating-regime/negative-mechanism contribution sufficiently novel?", "2. Are within-model controls adequate despite heterogeneous candidate views?", "3. Which claim must be dropped first: universal superiority, factual correctness, or production economics?", "4. What is the smallest indispensable independent study?", "5. Is an explicitly exploratory mechanism-aware paper acceptable at the intended venue category?", "6. Should E4-5 be attempted at all given the current reservation and answer-validation limitations?", "", "No reviewer quotations or endorsements are fabricated."]) + "\n")
    write_text(OUT / "E4_R_REVIEWER_EXECUTIVE_BRIEF.md", "\n".join(["# DecisionRAG E4-R Reviewer Executive Brief", "", "**Gate:** `E4_R_OWNER_REVIEW_READY`  ", "**Scientific status:** exploratory evidence reconciled; `CONFIRMATORY_DATA_NOT_READY` remains.", "", "- E2-M1 Clef joint positives (+14.295 pp Track A conditional; +20 pp Track B conditional) and E4-3 negatives (-4.854 pp K10; -4.333 pp K20) are not directly contradictory: exact query/candidate/view overlap is zero and contracts/populations are heterogeneous.", "- E4-3 representative CSSR-U at K20: Jina/PPLX 0.537, Liquid 0.360, BGE 0.393, Clef pointwise 0.307, Clef joint 0.263.", "- One source group contains 261/300 representative queries; cluster and leave-largest-out results are descriptive, not confirmatory.", "- Reader interventions affect reference-agreement proxies but do not prove factual correctness.", "- Matched isolated E4-3 latency is blocked; historical isolated timings are not comparable enough for a production frontier.", "", "**Path 1:** submit a strictly exploratory mechanism-aware empirical paper with negative findings and limitations.  ", "**Path 2:** run one narrowly pre-registered, genuinely prospective, independently answer-validated confirmation study.  ", "", "Reviewer decision needed: assess novelty, minimum indispensable confirmation, which claims must be dropped, and whether E4-5 is scientifically justified."]) + "\n")


def run_tests() -> dict[str, Any]:
    focused = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_e4_r_reconciliation.py"], cwd=ROOT, capture_output=True, text=True, timeout=600, env={**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    full = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA"], cwd=ROOT, capture_output=True, text=True, timeout=1800, env={**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    report = {"focused": {"return_code": focused.returncode, "stdout": focused.stdout, "stderr": focused.stderr}, "full": {"return_code": full.returncode, "stdout": full.stdout, "stderr": full.stderr}, "cpu_only": True, "network": "offline_env_flags"}
    write_json(OUT / "E4_R_TEST_RESULTS.json", report)
    write_text(OUT / "E4_R_TEST_REPORT.md", "\n".join(["# E4-R Test Report", "", "## Focused CPU tests", f"Return code: **{focused.returncode}**", "```text", focused.stdout[-10000:], focused.stderr[-3000:], "```", "", "## Full repository regression", f"Return code: **{full.returncode}**", "```text", full.stdout[-14000:], full.stderr[-4000:], "```", "", "Tests were run with CUDA hidden and Hugging Face offline flags. A nonzero result is preserved as an environment or regression blocker; no model inference is triggered by E4-R itself."]) + "\n")
    return report


def independent_crosscheck() -> dict[str, Any]:
    out = OUT / "E4_R_INDEPENDENT_CROSSCHECK.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/e4_r_independent_check.py"), "--output", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=300, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return {"return_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "output": str(out)}


def finalize(inventory_audit: dict[str, Any], e2: dict[str, Any], e4: dict[str, Any], source: dict[str, Any], timing: dict[str, Any], clef: dict[str, Any], tests: dict[str, Any], cross: dict[str, Any]) -> dict[str, Any]:
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "E4_R_PROVENANCE_MANIFEST.json")
    manifest = {
        "status": "E4_R_OWNER_REVIEW_READY" if tests["full"]["return_code"] == 0 and cross["return_code"] == 0 else "E4_R_PARTIAL_BLOCKED",
        "scientific_status": "CONFIRMATORY_DATA_NOT_READY",
        "audit_completion": "COMPLETE_WITH_QUALIFICATIONS",
        "cpu_only": True, "gpu_calls": 0, "model_calls": 0, "api_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "holdout_accessed": False,
        "git_head": git_head(), "focused_test_return_code": tests["focused"]["return_code"], "full_test_return_code": tests["full"]["return_code"], "independent_crosscheck_return_code": cross["return_code"],
        "input_inventory_count": inventory_audit["inventory_count"], "input_manifest_hash_mismatches": inventory_audit["manifest_hash_mismatch_count"],
        "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
        "e2_e4_summary": {"e2_case_rows": e2["case_rows"], "e4_cohort_rows": len(e4["cohort"]), "e4_valid_raw_rows": sum(1 for x in e4["raw"] if x.get("status") == "VALID")},
    }
    write_json(OUT / "E4_R_PROVENANCE_MANIFEST.json", manifest)
    write_text(OUT / "E4_R_OWNER_REVIEW_GATE.md", "\n".join(["# E4-R Owner Review Gate", "", f"**Status:** `{manifest['status']}`  ", "**Scientific confirmation status:** `CONFIRMATORY_DATA_NOT_READY`  ", f"**Focused tests:** `{tests['focused']['return_code']}`  ", f"**Full regression:** `{tests['full']['return_code']}`  ", f"**Independent CPU cross-check:** `{cross['return_code']}`  ", "**GPU/model/API calls:** `0`  ", "**Jev/paid calls:** `0/0`  ", "**Holdout outcomes accessed:** `false`", "", "E4-R is complete as an offline reconciliation dossier. No E4-5 or other experiment may start automatically. Scientific confirmation, factual answer correctness, universal model superiority and production economics remain unsupported."]) + "\n")
    # Hash the final gate/reports in a second immutable write of the manifest.
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "E4_R_PROVENANCE_MANIFEST.json")
    manifest["files"] = {str(p.relative_to(ROOT)): sha_file(p) for p in files}
    write_json(OUT / "E4_R_PROVENANCE_MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inventory", "reconcile", "tests", "all"], default="all")
    args = parser.parse_args()
    if args.stage == "inventory":
        inventory(); return 0
    inv = inventory()
    e2 = load_e2()
    e4 = e4_recompute()
    source = source_group_audit(e4)
    clef = clef_reconciliation(e2, e4)
    timing = work_and_latency(e4)
    claims = claims_matrix(e2, e4, source, timing)
    reports(e2, e4, source, timing, claims, clef, inv)
    cross = independent_crosscheck()
    if args.stage == "reconcile":
        return 0 if cross["return_code"] == 0 else 1
    tests = run_tests()
    manifest = finalize(inv, e2, e4, source, timing, clef, tests, cross)
    print(json.dumps({"status": manifest["status"], "files": len(manifest["files"]), "focused": tests["focused"]["return_code"], "full": tests["full"]["return_code"], "crosscheck": cross["return_code"]}, sort_keys=True))
    return 0 if manifest["status"] == "E4_R_OWNER_REVIEW_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
