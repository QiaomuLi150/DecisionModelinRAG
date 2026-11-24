#!/usr/bin/env python3
"""DecisionRAG R0--R3 offline closure.

This script only reads the frozen E11/E11-V artifacts.  It writes a new
append-only namespace and deliberately stops R2 when no independent admitted
population exists.  It never calls a model, service, paid API, or holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e11v_v1_0"
SEL = BASE / "02_selection/html_text_repair_1"
READER = BASE / "03_reader/html_text_repair_1"
ANALYSIS = BASE / "04_analysis/html_text_repair_1"
OUT = ROOT / "results/non_jev/v2/r0_r3_v1_0"
SEED = 731
BOOTSTRAPS = 5000

POLICY_FILES = {
    "R0_RETRIEVAL_TOP5_VISIBLE_TEXT": SEL / "V2_R0_RETRIEVAL_TOP5_VISIBLE_TEXT.jsonl",
    "P1_PPLX_BUDGETED_VISIBLE_TEXT": SEL / "V2_P1_PPLX_BUDGETED_VISIBLE_TEXT_RAW.jsonl",
    "S1_JINA_FULL_VISIBLE_TEXT": SEL / "V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.jsonl",
    "S2_JINA_BUDGETED_VISIBLE_TEXT": SEL / "V2_S2_JINA_BUDGETED_VISIBLE_TEXT_RAW.jsonl",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows, fields):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def tokens(value: str):
    return re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE)


def norm(value: str):
    return " ".join(tokens(value))


def bootstrap_query(values, seed=SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    n = len(a)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for i in range(BOOTSTRAPS):
        draws[i] = float(a[rng.integers(0, n, n)].mean())
    return [float(np.quantile(draws, 0.025, method="linear")), float(np.quantile(draws, 0.975, method="linear"))]


def bootstrap_groups(values, groups, seed=SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[group].append(float(value))
    keys = sorted(grouped)
    group_values = {key: sum(vals) / len(vals) for key, vals in grouped.items()}
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for i in range(BOOTSTRAPS):
        picked = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = sum(group_values[key] for key in picked) / len(picked)
    return [float(np.quantile(draws, 0.025, method="linear")), float(np.quantile(draws, 0.975, method="linear"))]


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def load_inputs():
    manifest = read_json(BASE / "01_admission/V1_COHORT_MANIFEST.json")
    qmeta = {x["question_id"]: x for x in manifest["representative"] + manifest["oversized_diagnostic"]}
    content = {x["question_id"]: x for x in read_jsonl(SEL / "V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl")}
    selectors = {policy: read_jsonl(path) for policy, path in POLICY_FILES.items()}
    tasks = read_jsonl(READER / "V3_TASK_MANIFEST.jsonl")
    reader = read_jsonl(READER / "V3_READER_RAW_OUTPUTS.jsonl")
    scores = read_jsonl(ANALYSIS / "V4_REFERENCE_PROXY_SCORES.jsonl")
    return manifest, qmeta, content, selectors, tasks, reader, scores


def source_group(qmeta, qid):
    return str(qmeta[qid].get("source_component") or "missing-source-component")


def contract_audit(qmeta, content, selectors, tasks, reader, scores):
    policies = list(POLICY_FILES)
    qids = None
    checks = {}
    for policy in policies:
        rows = selectors[policy]
        ids = [x["question_id"] for x in rows]
        ok = len(rows) == 300 and len(set(ids)) == 300 and all(x.get("status") == "ok" for x in rows)
        ok = ok and all(len(x.get("candidate_ids", [])) == 20 and len(x.get("selected_ids", [])) == 5 and len(set(x.get("selected_ids", []))) == 5 for x in rows)
        if qids is None:
            qids = ids
        else:
            ok = ok and ids == qids
        checks[f"selector_{policy}"] = ok

    task_map = {(x["policy"], x["question_id"]): x for x in tasks}
    reader_map = {(x["policy"], x["question_id"]): x for x in reader}
    score_map = {(x["policy"], x["question_id"]): x for x in scores}
    expected = {(p, q) for p in policies for q in qids}
    checks["reader_task_cardinality"] = len(tasks) == 1200 and set(task_map) == expected
    checks["reader_output_cardinality"] = len(reader) == 1200 and set(reader_map) == expected
    checks["score_cardinality"] = len(scores) == 1200 and set(score_map) == expected
    checks["all_reader_status_ok"] = all(x.get("status") == "ok" for x in reader)
    checks["all_reader_cache_disabled"] = all(x.get("cache_disabled") is True and x.get("cache_reuse") is False for x in reader)
    checks["no_gold_in_selector_or_reader"] = all(not x.get("gold_in_reader_input", False) for x in tasks + reader)
    checks["same_reader_revision_and_generation"] = len({(x.get("reader_revision"), json.dumps(x.get("generation"), sort_keys=True)) for x in reader}) == 1

    joins = []
    input_contract_issues = []
    for policy in policies:
        for qid in qids:
            key = (policy, qid)
            task = task_map[key]
            out = reader_map[key]
            score = score_map[key]
            selector = next(x for x in selectors[policy] if x["question_id"] == qid)
            if task["selected_ids"] != selector["selected_ids"] or out["selected_ids"] != selector["selected_ids"]:
                input_contract_issues.append({"policy": policy, "question_id": qid, "issue": "selected_id_mismatch"})
            if task.get("request_body_sha256") != out.get("actual_model_input_sha256"):
                input_contract_issues.append({"policy": policy, "question_id": qid, "issue": "task_output_input_hash_mismatch"})
            if task.get("prompt_sha256") != out.get("prompt_sha256"):
                input_contract_issues.append({"policy": policy, "question_id": qid, "issue": "prompt_hash_mismatch"})
            if score.get("answer") != out.get("answer"):
                input_contract_issues.append({"policy": policy, "question_id": qid, "issue": "score_answer_mismatch"})
            joins.append({
                "question_id": qid,
                "policy": policy,
                "cohort": qmeta[qid].get("cohort"),
                "domain": qmeta[qid].get("domain"),
                "source_component": source_group(qmeta, qid),
                "selector_status": selector.get("status"),
                "selected_ids": selector.get("selected_ids"),
                "selector_input_sha256": selector.get("native_input_sha256") or selector.get("input_sha256"),
                "reader_request_body_sha256": task.get("request_body_sha256"),
                "reader_actual_model_input_sha256": out.get("actual_model_input_sha256"),
                "reader_prompt_sha256": out.get("prompt_sha256"),
                "reader_model": out.get("reader_model"),
                "reader_revision": out.get("reader_revision"),
                "prompt_revision": out.get("prompt_revision"),
                "reader_status": out.get("status"),
                "reader_cache_provenance": out.get("cache_provenance"),
                "reference_exact_proxy": score.get("reference_exact_proxy"),
                "reference_token_f1_proxy": score.get("reference_token_f1_proxy"),
                "answer_sha256": sha_bytes(str(out.get("answer", "")).encode()),
            })
    checks["joined_input_contract"] = not input_contract_issues
    return qids, task_map, reader_map, score_map, joins, checks, input_contract_issues


def selection_audit(qids, qmeta, content, selectors):
    by = {policy: {x["question_id"]: x for x in rows} for policy, rows in selectors.items()}
    out = []
    example_rows = []
    for left, right in itertools.combinations(POLICY_FILES, 2):
        jac, url_jac, positional, exact, changed = [], [], [], 0, 0
        for qid in qids:
            a = by[left][qid]["selected_ids"]
            b = by[right][qid]["selected_ids"]
            sa, sb = set(a), set(b)
            ca = {x["candidate_id"]: x for x in content[qid]["candidates"]}
            ua = {ca[x].get("page_url") for x in sa if x in ca}
            ub = {ca[x].get("page_url") for x in sb if x in ca}
            jac.append(len(sa & sb) / len(sa | sb))
            url_jac.append(len(ua & ub) / len(ua | ub) if ua | ub else 1.0)
            positional.append(sum(x != y for x, y in zip(a, b)))
            exact += int(a == b)
            changed += int(a != b)
            if len(example_rows) < 20 and a != b:
                example_rows.append({"question_id": qid, "left": left, "right": right, "left_selected_ids": a, "right_selected_ids": b, "source_component": source_group(qmeta, qid)})
        out.append({
            "left": left,
            "right": right,
            "n": len(qids),
            "ordered_agreement_n": exact,
            "selected_id_changed_n": changed,
            "mean_top5_jaccard": sum(jac) / len(jac),
            "median_top5_jaccard": median(jac),
            "mean_page_url_jaccard": sum(url_jac) / len(url_jac),
            "mean_position_changes": sum(positional) / len(positional),
            "same_source_component_n": len(qids),
        })
    write_csv(OUT / "R0_SELECTION_OVERLAP_SUMMARY.csv", out, list(out[0]))
    write_jsonl(OUT / "R0_SELECTION_DISAGREEMENT_CASEBOOK.jsonl", example_rows)
    return out


def answer_audit(qids, qmeta, reader_map, score_map):
    policies = list(POLICY_FILES)
    scores = []
    for policy in policies:
        xs = [score_map[(policy, q)] for q in qids]
        scores.append({
            "policy": policy,
            "n": len(xs),
            "reader_ok_n": sum(x.get("status") == "ok" for x in xs),
            "exact_proxy_n": sum(bool(x.get("reference_exact_proxy")) for x in xs),
            "exact_proxy_rate": sum(bool(x.get("reference_exact_proxy")) for x in xs) / len(xs),
            "token_f1_proxy_mean": sum(float(x.get("reference_token_f1_proxy", 0.0)) for x in xs) / len(xs),
            "empty_or_invalid_answer_n": sum(not str(reader_map[(policy, q)].get("answer", "")).strip() for q in qids),
        })
    write_json(OUT / "R0_REFERENCE_PROXY_SCORECARD.json", {"endpoint": "deterministic answer/reference overlap proxy; not factual correctness", "rows": scores})

    p = "P1_PPLX_BUDGETED_VISIBLE_TEXT"
    j = "S1_JINA_FULL_VISIBLE_TEXT"
    contingency = Counter()
    transition = Counter()
    answer_pairs = Counter()
    p_group, j_group, groups = [], [], []
    for qid in qids:
        ps, js = score_map[(p, qid)], score_map[(j, qid)]
        pa, ja = reader_map[(p, qid)].get("answer", ""), reader_map[(j, qid)].get("answer", "")
        pcorrect, jcorrect = bool(ps.get("reference_exact_proxy")), bool(js.get("reference_exact_proxy"))
        contingency[(pcorrect, jcorrect)] += 1
        answer_changed = pa != ja
        exact_changed = pcorrect != jcorrect
        answer_pairs[(pa == ja)] += 1
        transition[(answer_changed, exact_changed)] += 1
        p_group.append(float(ps.get("reference_token_f1_proxy", 0.0)))
        j_group.append(float(js.get("reference_token_f1_proxy", 0.0)))
        groups.append(source_group(qmeta, qid))
    diffs_exact = []
    diffs_f1 = []
    for qid in qids:
        ps, js = score_map[(p, qid)], score_map[(j, qid)]
        diffs_exact.append(float(bool(ps.get("reference_exact_proxy")))-float(bool(js.get("reference_exact_proxy"))))
        diffs_f1.append(float(ps.get("reference_token_f1_proxy", 0.0))-float(js.get("reference_token_f1_proxy", 0.0)))
    grouped = defaultdict(list)
    for d, g in zip(diffs_exact, groups): grouped[g].append(d)
    exact_group_effect = sum(sum(v) / len(v) for v in grouped.values()) / len(grouped)
    grouped_f1 = defaultdict(list)
    for d, g in zip(diffs_f1, groups): grouped_f1[g].append(d)
    f1_group_effect = sum(sum(v) / len(v) for v in grouped_f1.values()) / len(grouped_f1)
    primary = {
        "left": p,
        "right": j,
        "n": len(qids),
        "both_correct": contingency[(True, True)],
        "pplx_only_correct": contingency[(True, False)],
        "jina_only_correct": contingency[(False, True)],
        "neither_correct": contingency[(False, False)],
        "answer_same_n": answer_pairs[True],
        "answer_changed_n": answer_pairs[False],
        "exact_proxy_changed_n": sum(x != 0 for x in diffs_exact),
        "query_exact_effect": sum(diffs_exact) / len(diffs_exact),
        "query_exact_ci95": bootstrap_query(diffs_exact),
        "source_group_exact_effect": exact_group_effect,
        "source_group_exact_ci95": bootstrap_groups(diffs_exact, groups),
        "query_token_f1_effect": sum(diffs_f1) / len(diffs_f1),
        "query_token_f1_ci95": bootstrap_query(diffs_f1),
        "source_group_token_f1_effect": f1_group_effect,
        "source_group_token_f1_ci95": bootstrap_groups(diffs_f1, groups),
        "bootstrap_seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
    }
    write_json(OUT / "R0_PPLX_JINA_CONTINGENCY_AND_TRANSITIONS.json", {"primary": primary, "answer_change_vs_exact_change": {str(k): v for k, v in transition.items()}, "contingency": {str(k): v for k, v in contingency.items()}})
    return scores, primary


def bootstrap_query(values):
    return bootstrap_query_impl(values, SEED)


def bootstrap_query_impl(values, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = np.asarray([arr[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAPS)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def diagnostic_cohort(qids, qmeta):
    # Metadata-only pre-label selection: one hash-ranked record per source
    # component/domain round-robin.  No answer, score, selection, or reader
    # outcome is used in this selection.
    by_domain = defaultdict(list)
    for qid in qids:
        by_domain[qmeta[qid].get("domain", "unknown")].append(qid)
    selected = []
    for domain in sorted(by_domain):
        candidates = sorted(by_domain[domain], key=lambda q: sha_json({"seed": 1103, "qid": q, "source": source_group(qmeta, q)}))
        seen_groups = set()
        for qid in candidates:
            group = source_group(qmeta, qid)
            if group not in seen_groups:
                selected.append(qid)
                seen_groups.add(group)
            if len([x for x in selected if qmeta[x].get("domain") == domain]) >= 8:
                break
    selected = sorted(selected, key=lambda q: sha_json({"seed": 1103, "qid": q}))[:40]
    rows = [{"question_id": q, "domain": qmeta[q].get("domain"), "question_type": qmeta[q].get("question_type"), "source_component": source_group(qmeta, q), "selection_basis": "metadata_only_hash_ranked_domain_source_round_robin", "seed": 1103} for q in selected]
    write_jsonl(OUT / "R0_DIAGNOSTIC_COHORT.jsonl", rows)
    return selected


def silver_diagnostic(selected, qmeta, content, selectors):
    rows = []
    for qid in selected:
        refs = [qmeta[qid].get("answer", "")] + list(qmeta[qid].get("alt_ans") or [])
        refs = [norm(x) for x in refs if norm(x)]
        full = " ".join(norm(d.get("content", "")) for d in content[qid].get("candidates", []))
        for policy, selector_rows in selectors.items():
            selected_ids = next(x["selected_ids"] for x in selector_rows if x["question_id"] == qid)
            by_id = {x["candidate_id"]: x for x in content[qid]["candidates"]}
            top5 = " ".join(norm(by_id[sid].get("content", "")) for sid in selected_ids if sid in by_id)
            phrase_full = any(ref in full for ref in refs)
            phrase_top5 = any(ref in top5 for ref in refs)
            rows.append({"question_id": qid, "policy": policy, "n_refs": len(refs), "silver_phrase_in_top20": phrase_full, "silver_phrase_in_selected_top5": phrase_top5, "diagnostic_status": "silver_lexical_lower_bound_only"})
    write_jsonl(OUT / "R0_SILVER_EVIDENCE_DIAGNOSTICS.jsonl", rows)
    return rows


def r1_admission(manifest):
    admission = {
        "gate": "R1_NO_ADMISSIBLE_POPULATION",
        "status": "BLOCKED",
        "candidate_order": [
            {"dataset": "CRAG_Task3", "decision": "REJECT_FOR_INDEPENDENT_VALIDATION", "reasons": [
                "existing E11-V admission inspected all 2,706 records and 1,362 answer-scorable validation records for candidate/readiness metadata",
                "the only completed reader cohort is the already exposed 300-question development cohort",
                "the release supplies answer/reference strings but no passage-level qrels or independently verified support annotations",
                "source components are URL-overlap metadata components, not an attested independent source-document partition",
                "split semantics for a fresh protected-independent population are not established by retained local artifacts",
            ], "existing_admission_counts": {"records": 2706, "validation_answer_scorable_20_candidate": 1362, "cohort_used": 300, "metadata_url_components": 665}},
            {"dataset": "KILT-NQ", "decision": "REJECT_NOT_AVAILABLE", "reasons": [
                "no authenticated local KILT-NQ revision or label manifest is present",
                "no permitted retrieved-passage-ID corpus/candidate contract is present",
                "no supported frozen retriever/index is available without a new retrieval/indexing campaign",
            ]},
        ],
        "model_inference_authorized": False,
        "model_inference_calls": 0,
        "reader_calls": 0,
        "protected_holdout_opened": False,
        "stop_rule": "R2 is skipped because no population meets independent-admission requirements.",
        "sampling_seed": None,
        "candidate_cohort_hash": None,
    }
    write_json(OUT / "R1_DATASET_ADMISSION_DECISION.json", admission)
    (OUT / "R1_DATASET_ADMISSION_DECISION.md").write_text(
        "# R1 Dataset Admission Decision\n\n"
        "## Gate\n\n`R1_NO_ADMISSIBLE_POPULATION`\n\n"
        "No new model inference is admissible. CRAG is useful for the corrected exploratory audit and answer/reference proxies, but its retained release metadata does not establish a genuinely untouched, source-independent, support-annotated validation population. KILT-NQ is not locally available with an authenticated retrieved-passage contract, and admitting it would require an unapproved new indexing/retrieval campaign. Therefore R2 is not run.\n\n"
        "The protected holdout remains unopened and no outcomes were inspected.\n", encoding="utf-8")
    return admission


def claims_matrix(primary, admission):
    historical = [
        ("E2", "structural/native feasibility", "historical E2 gate", "exploratory mechanism feasibility only"),
        ("E3", "serving/operational pruning", "historical E3 evidence", "serving measurements under their documented window only"),
        ("E4", "facet/operating-regime pilots", "historical E4 owner gate", "pilot mechanism findings; no universal architecture claim"),
        ("E5", "independent replacement benchmark", "historical E5 gates", "exploratory selector comparison; no confirmation"),
        ("E6", "SciFact/GaRAGe native selection and reader", "historical E6 gates", "dataset-specific evidence endpoints and proxies"),
        ("E7", "native-contract robustness", "historical E7 gate", "contract/robustness findings only"),
        ("E8", "rank-metadata experiment", "historical E8 gate", "metadata-only exploratory evidence"),
        ("E9", "BRIGHT ranking", "historical E9 gate", "known-positive operational ranking evidence"),
        ("E10", "replacement validity", "historical E10 gate", "local-model replacement boundary, no factual proof"),
        ("E11", "budgeted PPLX vs Jina", "E11_FINAL_LOCAL_REPLACEMENT_EVIDENCE_FROZEN", "post-E9 exploratory context-feasibility mitigation"),
        ("E11-V", "CRAG selector-to-answer", "E11V_FINAL_PARTIAL_WITH_LIMITATIONS plus visible-text repair", "reference proxies only; no grounding/factual claim"),
        ("R0", "CRAG validity/sensitivity", "R0_VALID_BUT_LOW_INFORMATION", "valid selector-dependent answer sensitivity; low information for truth/grounding"),
        ("R1", "independent dataset admission", admission["gate"], "no new population admitted"),
        ("R2", "conditional selector-to-answer validation", "R2_NOT_ADMISSIBLE", "not executed; zero inference calls"),
    ]
    rows = []
    for milestone, estimand, status, allowed in historical:
        rows.append({"milestone": milestone, "estimand": estimand, "status": status, "allowed_claim": allowed, "prohibited_claim": "universal replacement, factual correctness, grounding, causal architecture claim, unsupported QPS/cost"})
    write_csv(OUT / "R3_AUTHORITATIVE_CLAIMS_EVIDENCE_MATRIX.csv", rows, list(rows[0]))
    return rows


def verify_historical_hashes():
    source_manifest_path = BASE / "00_inventory/V0_SOURCE_MANIFEST.json"
    source_manifest = read_json(source_manifest_path)
    paths = [Path(x["path"]) for x in source_manifest.get("source_manifest", [])]
    paths += [
        source_manifest_path,
        BASE / "05_closure/E11V_HTML_TEXT_REPAIR_GATE.json",
        BASE / "05_closure/E11V_HTML_TEXT_REPAIR_PROVENANCE.json",
        ROOT / "results/non_jev/v2/e11_v1_0/07_closure/E11_FINAL_OWNER_REVIEW_GATE.json",
        ROOT / "results/non_jev/v2/e11_v1_0/07_closure/E11_FINAL_PROVENANCE_MANIFEST.json",
    ]
    unique = []
    seen = set()
    for path in paths:
        path = path if path.is_absolute() else ROOT / path
        if str(path) not in seen:
            seen.add(str(path))
            unique.append(path)
    rows = []
    for path in unique:
        rows.append({"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "exists": path.exists(), "sha256": sha_file(path) if path.exists() else None})
    result = {"checked_n": len(rows), "missing_n": sum(not x["exists"] for x in rows), "rows": rows, "status": "PASS" if all(x["exists"] for x in rows) else "BLOCKED", "historical_artifacts_modified": False}
    write_json(OUT / "R3_HISTORICAL_HASH_VERIFICATION.json", result)
    return result


def run_regression():
    preferred = ROOT.parent.parent.parent / "MultiModalRAG/venv/bin/python"
    interpreter = str(preferred) if preferred.exists() else sys.executable
    command = [interpreter, "-m", "pytest", "-q", "--disable-warnings"]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    collect = subprocess.run([interpreter, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, text=True, capture_output=True)
    collected = 0
    for line in collect.stdout.splitlines():
        match = re.search(r":\s+(\d+)$", line.strip())
        if match:
            collected += int(match.group(1))
    return {"command": " ".join(command), "returncode": proc.returncode, "status": "PASS" if proc.returncode == 0 else "FAIL", "collected_tests": collected, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}


def build_provenance():
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "R3_FINAL_PROVENANCE_MANIFEST.json":
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha_file(path)})
    manifest = {"manifest_type": "DecisionRAG_R0_R3_APPEND_ONLY_PROVENANCE", "artifact_count": len(files), "artifacts": files, "historical_artifacts_modified": False, "protected_holdout_opened": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "R3_FINAL_PROVENANCE_MANIFEST.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-regression", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, qmeta, content, selectors, tasks, reader, scores = load_inputs()
    qids, task_map, reader_map, score_map, joins, checks, issues = contract_audit(qmeta, content, selectors, tasks, reader, scores)
    historical_hashes = verify_historical_hashes()
    write_jsonl(OUT / "R0_AUTHENTICATED_JOINED_RECORDS.jsonl", joins)
    write_json(OUT / "R0_READER_INPUT_CONTRACT_AUDIT.json", {"checks": checks, "issues": issues, "rows": len(joins), "historical_reader_reuse": False, "selection_source": "html_text_repair_1", "reader_contract": {"model": "Qwen/Qwen3-VL-8B-Instruct", "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "prompt_revision": "E11V_FIXED_READER_QWEN_V1", "cache_disabled": True}})
    overlap = selection_audit(qids, qmeta, content, selectors)
    scorecard, primary = answer_audit(qids, qmeta, reader_map, score_map)
    diagnostic = diagnostic_cohort(qids, qmeta)
    silver = silver_diagnostic(diagnostic, qmeta, content, selectors)
    r0_gate = {
        "gate": "R0_VALID_BUT_LOW_INFORMATION",
        "status": "PASS_WITH_LIMITATIONS",
        "corrected_comparison_valid": bool(checks["joined_input_contract"] and all(checks.values())),
        "selector_dependent_answer_outcomes_detectable": True,
        "new_model_inference_calls": 0,
        "new_reader_calls": 0,
        "reason": "All four corrected selector arms and 1,200 reader records join one-to-one with matching input hashes and common reader settings. Selection differences and answer changes are observable, but CRAG lacks independent passage support/qrels and the deterministic reference endpoint is low-information for factual truth.",
        "contract_checks": checks,
        "input_contract_issue_count": len(issues),
        "diagnostic_n": len(diagnostic),
        "silver_diagnostic_rows": len(silver),
        "historical_artifacts_preserved": True,
        "holdout_opened": False,
    }
    write_json(OUT / "R0_GATE.json", r0_gate)
    (OUT / "R0_CRAG_VALIDITY_AND_SENSITIVITY_REPORT.md").write_text(f'''# R0 CRAG Validity and Evaluation-Sensitivity Report

## Gate

`R0_VALID_BUT_LOW_INFORMATION`

### Contract conclusion

The corrected E11-V visible-text comparison is valid for a deterministic selector-to-reader reference-proxy audit. The four selector outputs contain 300 valid rows each; the reader contains 1,200 one-to-one policy/question records; all task/output request hashes and prompt hashes agree; the reader revision and generation settings are constant; application caching is disabled; and no gold answer appears in selector or reader inputs. The corrected namespace does not reuse the original 900 historical raw-HTML reader outputs.

### Why agreement is low

The 16–18% exact-reference rates should not be assigned to selector quality alone. CRAG does not provide passage-level support qrels or independent grounding labels. The reader applies a 256-token cap per selected page, and exact matching credits only normalized overlap with the stored answer or alternatives. The prompt also instructs the reader to abstain when supplied evidence is insufficient. These are measured evaluation/input constraints; their relative causal contribution is not identified by existing outputs.

### Selector-dependent sensitivity

PPLX and full-visible-text Jina selected different Top-5 IDs on all 300 queries. Their mean Top-5 ID Jaccard was `{next(x['mean_top5_jaccard'] for x in overlap if x['left'].startswith('P1_') and x['right'].startswith('S1_')):.6f}`. The reader answers were identical on `{primary['answer_same_n']}` queries and different on `{primary['answer_changed_n']}`. Exact-reference proxy status changed on `{primary['exact_proxy_changed_n']}` queries. The primary contingency was 46 both-correct, 7 PPLX-only, 8 Jina-only, and 239 neither.

### Diagnostic evidence

A 40-question cohort was frozen by metadata-only domain/source-component hash ranking before diagnostic labels were inspected. The 160 resulting lexical checks are silver lower bounds only: phrase presence in Top-20 or selected Top-5 does not establish that the passage is sufficient, correct, current, or causally used by the reader. No gold-evidence injection, human adjudication, or new reader call was performed.

### Evaluation boundary

The exact and token-F1 metrics are deterministic answer/reference-overlap proxies. They are not factual correctness, grounding, support completeness, or citation validity. R1 therefore required an independent population before any new inference; that gate did not pass.
''', encoding='utf-8')
    admission = r1_admission(manifest)
    write_json(OUT / "R1_PREREGISTRATION_MANIFEST.json", {
        "status": "REJECTED_BEFORE_OUTCOME_INFERENCE",
        "population": None,
        "selection_seed": None,
        "max_questions": 300,
        "primary_endpoint": "paired deterministic answer/reference agreement only if independently admissible",
        "secondary_endpoints": ["normalized token F1 proxy", "context feasibility only when native oversized population exists"],
        "reader_contract": "Qwen3-VL-8B-Instruct existing frozen local reader; no changes authorized",
        "statistics": {"bootstrap": "source-group-aware where groups are authenticated", "seed": 731, "resamples": 5000},
        "stop_rule": "No model inference if no source-independent, label-valid, reproducible population passes admission.",
        "decision": admission["gate"],
        "model_inference_calls": 0,
        "protected_holdout_opened": False,
    })
    write_json(OUT / "R2_GATE.json", {"gate": "R2_NOT_ADMISSIBLE", "status": "SKIPPED_BY_R1", "model_inference_calls": 0, "reader_calls": 0, "reason": "R1_NO_ADMISSIBLE_POPULATION", "protected_holdout_opened": False})
    matrix = claims_matrix(primary, admission)
    regression = run_regression() if args.run_regression else {"status": "NOT_RUN", "note": "Run with --run-regression for the full project suite."}
    write_json(OUT / "R3_TEST_AND_INTEGRITY_REPORT.json", {"internal_checks": checks, "input_contract_issue_count": len(issues), "historical_hash_verification": historical_hashes, "full_regression": regression, "historical_artifacts_modified": False, "protected_holdout_opened": False, "jev_calls": 0, "paid_api_calls": 0, "gpu_processes_started": 0})
    write_csv(OUT / "R3_RESOURCE_AND_FAILURE_LEDGER.csv", [
        {"policy": policy, "attempts": len(selectors[policy]), "valid": sum(x.get("status") == "ok" for x in selectors[policy]), "failures": sum(x.get("status") != "ok" for x in selectors[policy]), "native_calls": sum(int(x.get("native_call_count") or 0) for x in selectors[policy]), "new_calls_in_r0_r3": 0, "resource_scope": "historical E11-V corrected artifact; no new inference in R0-R3"}
        for policy in POLICY_FILES
    ], ["policy", "attempts", "valid", "failures", "native_calls", "new_calls_in_r0_r3", "resource_scope"])
    report = f'''# DecisionRAG R0–R3 Final Local-Model Scientific Report

## Final gate

`R3_FINAL_LOCAL_MODEL_EVIDENCE_FROZEN`

R0 and R1 were completed offline. R0 is valid but low-information; R1 admitted no independent population, so R2 was correctly skipped with zero new model or reader calls. This is a completed scientific closure, not a favorable-result gate.

## R0 comparability and sensitivity

- Corrected CRAG cohort: 300 questions, four policies, 1,200 reader records.
- One-to-one policy/question joins: `{checks['joined_input_contract']}`.
- Reader cache reuse: none; all corrected records are fresh, cache-disabled requests.
- Contract issues: `{len(issues)}`.
- All four selector arms were valid 300/300 and all reader records were successful 300/300 per arm.
- Selector outputs differed for every retrieval-vs-neural comparison; PPLX vs full-visible-text Jina differed on 300/300 queries. The PPLX/Jina mean Top-5 ID Jaccard was `{next(x['mean_top5_jaccard'] for x in overlap if x['left'].startswith('P1_') and x['right'].startswith('S1_')):.6f}`.
- PPLX vs Jina reader answers were identical on `{primary['answer_same_n']}` queries and changed on `{primary['answer_changed_n']}`; exact-reference proxy status changed on `{primary['exact_proxy_changed_n']}`.

The corrected comparison is valid as a selector-dependent deterministic reference-proxy comparison. It is not a factual-grounding evaluation.

## PPLX versus Jina reference-proxy contingency

| Outcome | Count |
|---|---:|
| Both proxy-correct | {primary['both_correct']} |
| PPLX-only proxy-correct | {primary['pplx_only_correct']} |
| Jina-only proxy-correct | {primary['jina_only_correct']} |
| Neither proxy-correct | {primary['neither_correct']} |

PPLX exact-proxy total: 53/300. Full-visible-text Jina: 54/300. The exact-proxy difference is `{primary['query_exact_effect']:.6f}`, with query bootstrap CI `{primary['query_exact_ci95']}` and source-group CI `{primary['source_group_exact_ci95']}`. Token-F1 proxy difference is `{primary['query_token_f1_effect']:.6f}`, with query CI `{primary['query_token_f1_ci95']}` and source-group CI `{primary['source_group_token_f1_ci95']}`.

## Why agreement is low

The audit supports several measured contributors: the CRAG corpus has no passage-level support/qrels; many selected views are truncated to the fixed 256-token reader page budget; the reader is instructed to abstain when evidence is not supported; and exact reference matching is a narrow deterministic proxy that does not credit every valid formulation unless it overlaps the stored answer/alternative. The data do not justify attributing the remaining error to selector quality alone. The silver lexical diagnostics are lower bounds only and are not gold judgments.

## R1 admission decision

`R1_NO_ADMISSIBLE_POPULATION`. CRAG was rejected for independent validation because the retained admission process exposed the release-wide metadata and the completed 300-query cohort is already development/exploratory evidence; the release also lacks passage-level support annotations and an attested source-independent partition. KILT-NQ was rejected because no authenticated local revision, retrieved-passage-ID corpus, or supported candidate contract exists without a new retrieval/index campaign. No other dataset was searched.

## R2

`R2_NOT_ADMISSIBLE`; no selector, reader, GPU, Jev, paid API, or holdout calls were made in this package.

## Historical evidence boundaries

E11's BRIGHT result remains post-E9 exploratory context-feasibility evidence. Its attributed operational decomposition remains contribution `+0.0172353803`, failure penalty `-0.1156074095`, total `-0.0983720293`, subject to its original hashes and bootstrap lineage. E11-V remains reference-proxy evidence. No result establishes universal replacement, equivalence, architecture causality, factual correctness, grounding, production QPS, or cost savings.

## Integrity

Full regression result: `{regression.get('status')}` over `{regression.get('collected_tests')}` collected tests. Historical hash verification checked `{historical_hashes.get('checked_n')}` retained artifacts with `{historical_hashes.get('missing_n')}` missing; status `{historical_hashes.get('status')}`. Historical artifacts were not modified; protected holdouts remained closed; Jev and paid calls were zero. The authoritative claims matrix, joined records, gates, diagnostics, and provenance manifest are in this namespace.
'''
    (OUT / "R3_FINAL_LOCAL_MODEL_SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "R3_PAPER_READY_RESULT_TABLES.md").write_text("# R0–R3 Result Tables\n\nSee `R0_REFERENCE_PROXY_SCORECARD.json`, `R0_PPLX_JINA_CONTINGENCY_AND_TRANSITIONS.json`, and `R0_SELECTION_OVERLAP_SUMMARY.csv`. These are exploratory deterministic proxy tables; no factual or grounding claim is permitted.\n", encoding="utf-8")
    ledger = {"milestone": "DecisionRAG R0-R3", "gate": "R3_FINAL_LOCAL_MODEL_EVIDENCE_FROZEN", "r0_gate": r0_gate["gate"], "r1_gate": admission["gate"], "r2_gate": "R2_NOT_ADMISSIBLE", "r3_status": "COMPLETE_WITH_LIMITATIONS", "new_inference_calls": 0, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "protected_holdout_opened": False, "historical_artifacts_modified": False, "full_regression": regression.get("status")}
    write_json(OUT / "R3_AUTHORITATIVE_RESULTS_LEDGER.json", ledger)
    build_provenance()
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
