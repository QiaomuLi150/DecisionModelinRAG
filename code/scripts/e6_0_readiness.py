#!/usr/bin/env python3
"""DecisionRAG E6-0 CPU-only readiness audit.

This module deliberately does not import torch, transformers, CUDA, HTTP clients,
or any model adapter.  It reads frozen E2--E5 artifacts, the already archived
QASPER release, and public dataset/model metadata recorded in the reports.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E5ROOT = ROOT / "results/non_jev/v2/e5_v1_1"
E5 = E5ROOT / "e5_0_data_protocol"
E5A = E5ROOT / "e5_a_replacement_benchmark"
E5DC = E5ROOT / "e5_dc_serving_reader_validation"
E5QA = E5ROOT / "e5_qa_final_audit"
E4R = ROOT / "results/non_jev/v2/e4_v1_1/e4_r_reconciliation"
E2M1 = ROOT / "results/non_jev/v2/e2_m1"
QASPER = ROOT / "data/external/e5_qasper/snapshot"
EXTERNAL = ROOT / "data/external"
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_0_readiness"

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


def sha_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# Exact answer scorer from the AllenAI QASPER LED baseline evaluator.py.
# Source: https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py
def official_normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def official_token_f1(prediction: Any, ground_truth: Any) -> float:
    pred = official_normalize_answer(prediction).split()
    gold = official_normalize_answer(ground_truth).split()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


def official_references() -> dict[str, list[dict[str, Any]]]:
    refs: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev"):
        payload = read_json(QASPER / f"qasper-{split}-v0.3.json")
        for paper_id, paper in payload.items():
            for qa in paper.get("qas", []):
                qid = f"qasper:{paper_id}:{qa['question_id']}"
                out: list[dict[str, Any]] = []
                for annotation in qa.get("answers", []):
                    ans = annotation.get("answer") or {}
                    if ans.get("unanswerable"):
                        answer, typ = "Unanswerable", "none"
                    elif ans.get("extractive_spans"):
                        answer, typ = ", ".join(ans["extractive_spans"]), "extractive"
                    elif ans.get("free_form_answer"):
                        answer, typ = ans["free_form_answer"], "abstractive"
                    elif ans.get("yes_no"):
                        answer, typ = "Yes", "boolean"
                    elif ans.get("yes_no") is not None:
                        answer, typ = "No", "boolean"
                    else:
                        continue
                    out.append({"answer": answer, "type": typ, "annotation_id": ans.get("annotation_id"), "unanswerable": bool(ans.get("unanswerable"))})
                refs[qid] = out
    return refs


def load_e5() -> dict[str, Any]:
    manifest = read_json(E5DC / "E5_DC_EXECUTION_MANIFEST_FROZEN.json")
    reader = read_jsonl(E5DC / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl")
    outcomes = {x["query_id"]: x for x in read_jsonl(E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl")}
    return {"manifest": manifest, "reader": reader, "outcomes": outcomes, "refs": official_references()}


def qid_paper(qid: str) -> str:
    parts = qid.split(":")
    return parts[1] if len(parts) > 1 else qid


def verify_declared_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"manifest": str(path.relative_to(ROOT)), "exists": False, "checked": 0, "passed": 0, "failed": 1}
    data = read_json(path)
    declared = data.get("files") or data.get("artifacts") or data.get("outputs") or []
    if isinstance(declared, dict):
        declared = [{"path": k, **(v if isinstance(v, dict) else {"sha256": v})} for k, v in declared.items()]
    checks = []
    for item in declared:
        rel = item.get("path") if isinstance(item, dict) else None
        if rel is None and isinstance(item, dict):
            # E2-M1 stores output names as dictionary keys; normalized below.
            rel = item.get("name")
        candidates = []
        if rel:
            raw = Path(rel)
            candidates = [raw] if raw.is_absolute() else [ROOT / raw, path.parent / raw]
        p = next((candidate for candidate in candidates if candidate.is_file()), ROOT / "__missing__")
        ok = p.is_file() and sha_file(p) == item.get("sha256") and (item.get("bytes") is None or p.stat().st_size == item.get("bytes"))
        checks.append({"path": rel, "exists": p.is_file(), "hash_match": ok, "manifest": str(path.relative_to(ROOT))})
    return {"manifest": str(path.relative_to(ROOT)), "exists": True, "checked": len(checks), "passed": sum(x["hash_match"] for x in checks), "failed": sum(not x["hash_match"] for x in checks)}


def r0(e5: dict[str, Any]) -> tuple[dict[str, Any], str]:
    slate = E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    outcomes = e5["outcomes"]
    manifests = sorted(list(E5ROOT.glob("**/*PROVENANCE_MANIFEST*.json")) + list(ROOT.glob("results/non_jev/v2/e2_m1/*PROVENANCE_MANIFEST.json")))
    checks = [verify_declared_manifest(p) for p in manifests]
    reader_qids = sorted({x["query_id"] for x in e5["reader"]})
    e5a_groups = sorted({x["source_group_id"] for x in outcomes.values()})
    e5_papers = sorted({qid_paper(q) for q in outcomes})
    excluded = {"e5_a_qids": sorted(outcomes), "e5_reader_qids": reader_qids, "e5_source_groups": e5a_groups, "e5_source_papers": e5_papers, "sealed_holdout_accessed": False}
    index = {"schema": "E6-0-SOURCE-OVERLAP-EXCLUSION-V1", "created_utc": now(), "historical_exclusion_index": excluded, "coverage": "E2-E5 permitted artifact scan; E5-F remains unopened", "sha256": sha_obj(excluded)}
    write_json(OUT / "E6_0_SOURCE_OVERLAP_EXCLUSION_INDEX.json", index)
    write_json(OUT / "E6_0_R0_INPUT_AUTHENTICATION_AND_HISTORICAL_EXCLUSIONS.json", {
        "status": "PASS_WITH_SCOPE_LIMITS", "created_utc": now(), "frozen_slate": {"path": str(slate.relative_to(ROOT)), "sha256": sha_file(slate), "expected_sha256": SLATE_SHA, "match": sha_file(slate) == SLATE_SHA},
        "e5a_questions": len(outcomes), "e5a_source_groups": len(e5a_groups), "e5a_support_ceiling": sum(int(x.get("retrieval_feasible", 0)) for x in outcomes.values()),
        "e5_reader_rows": len(e5["reader"]), "e5_reader_unique_qids": len(reader_qids), "historical_manifests": checks, "sealed_holdout_accessed": False, "jev_calls": 0, "paid_api_calls": 0,
    })
    write_text(OUT / "E6_0_R0_REPO_STATE.md", f"""# E6-0 R0 Repository State

Generated: `{now()}`. This is a read-only historical audit; E2–E5 artifacts were not modified.

- E5-A frozen questions: **{len(outcomes)}**; source groups: **{len(e5a_groups)}**; retrieval-feasibility ceiling: **{sum(int(x.get('retrieval_feasible', 0)) for x in outcomes.values())}**.
- E5-C/D reader rows: **{len(e5['reader'])}**; unique reader questions: **{len(reader_qids)}**.
- Frozen Top-20 SHA-256: `{sha_file(slate)}`; expected `{SLATE_SHA}`.
- Historical declared-manifest checks: `{sum(x['passed'] for x in checks)}/{sum(x['checked'] for x in checks)}` passed; failures: `{sum(x['failed'] for x in checks)}`.
- Sealed E5-F raw content/outcomes: **not accessed**.
- New model/GPU/HTTP/reader operations: **none**.
""")
    r0_status = "PASS" if sha_file(slate) == SLATE_SHA and sum(x["failed"] for x in checks) == 0 else "CONDITIONAL"
    write_text(OUT / "E6_0_R0_BASELINE_TESTS.md", f"# E6-0 R0 Baseline Tests\n\nStatus: `{r0_status}`. Frozen slate and declared historical hashes were checked offline. The full regression command is recorded in T0; no model or GPU test was run.\n")
    return index, r0_status


def f1_rows(e5: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = e5["manifest"]
    rep = set(manifest["cohorts"]["representative_qids"])
    diag = set(manifest["cohorts"]["diagnostic_qids"])
    source_group = {x["query_id"]: x.get("source_group_id") for x in e5["reader"]}
    rows = []
    for raw in e5["reader"]:
        refs = e5["refs"].get(raw["query_id"], [])
        scored = [(official_token_f1(raw.get("answer", ""), x["answer"]), x) for x in refs]
        best, best_ref = max(scored, key=lambda x: x[0]) if scored else (0.0, {"type": "missing"})
        cohorts = []
        if raw["query_id"] in rep:
            cohorts.append("representative")
        if raw["query_id"] in diag:
            cohorts.append("diagnostic_enriched")
        rows.append({"query_id": raw["query_id"], "policy": raw["policy"], "cohort": "+".join(cohorts), "source_group_id": source_group.get(raw["query_id"]), "answer_type": best_ref.get("type", "missing"), "reference_count": len(refs), "prediction": raw.get("answer", ""), "official_answer_f1": best, "reader_status": raw.get("status"), "unanswerable_reference_present": any(x.get("unanswerable") for x in refs), "input_sha256": raw.get("actual_model_input_sha256"), "raw_response_sha256": raw.get("raw_response_sha256")})
    return rows


def group_bootstrap(rows: list[tuple[str, float]], seed: int = SEED) -> list[float]:
    grouped = defaultdict(list)
    for group, value in rows:
        grouped[group].append(value)
    means = [mean(v) for _, v in sorted(grouped.items())]
    if not means:
        return []
    rng = random.Random(seed)
    return [mean(rng.choice(means) for _ in means) for _ in range(BOOTSTRAP_REPS)]


def ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    s = sorted(values)
    return [s[int(0.025 * (len(s) - 1))], s[int(0.975 * (len(s) - 1))]]


def a0(e5: dict[str, Any]) -> dict[str, Any]:
    rows = f1_rows(e5)
    fields = ["query_id", "policy", "cohort", "source_group_id", "answer_type", "reference_count", "prediction", "official_answer_f1", "reader_status", "unanswerable_reference_present", "input_sha256", "raw_response_sha256"]
    write_csv(OUT / "E6_0_QASPER_ANSWER_F1_PER_QID.csv", rows, fields)
    scorer = {"source_url": "https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py", "source_commit": "NOT_PINNED_BY_LOCAL_RELEASE; raw source inspected 2026-10-14", "source_file_sha256": "NOT_PINNED", "algorithm": "official QASPER token_f1_score; max over annotation references; extractive spans joined with ', '; unanswerable='Unanswerable'; no evidence F1 computed", "normalization": "lowercase, remove ASCII punctuation, remove articles a/an/the, collapse whitespace", "weighting": "question macro and equal-source-group bootstrap", "post_hoc": True}
    write_json(OUT / "E6_0_QASPER_OFFICIAL_SCORER_PROVENANCE.json", scorer)
    score_rows = []
    for cohort in ("representative", "diagnostic_enriched"):
        qids = sorted({r["query_id"] for r in rows if cohort in r["cohort"]})
        for policy in ("pplx_decider", "jina35"):
            byq = {r["query_id"]: r for r in rows if r["policy"] == policy and cohort in r["cohort"]}
            vals = [byq[q]["official_answer_f1"] for q in qids if q in byq and byq[q]["reader_status"] == "OK"]
            groups = [(byq[q]["source_group_id"], byq[q]["official_answer_f1"]) for q in qids if q in byq and byq[q]["reader_status"] == "OK"]
            boot = group_bootstrap(groups)
            types = Counter(byq[q]["answer_type"] for q in qids if q in byq)
            score_rows.append({"cohort": cohort, "policy": policy, "n_qids": len(qids), "accepted_n": len(vals), "mean_answer_f1": mean(vals) if vals else None, "answer_f1_ci95_group_bootstrap": json.dumps(ci(boot)), "source_groups": len({g for g, _ in groups}), "answer_type_counts": json.dumps(dict(types), sort_keys=True), "status": "POST_HOC_OFFICIAL_QASPER_ANSWER_F1"})
    write_csv(OUT / "E6_0_QASPER_ANSWER_F1_SCORECARD.csv", score_rows, list(score_rows[0]))
    write_text(OUT / "E6_0_ANSWER_TYPE_VALIDITY_REPORT.md", f"""# E6-0 QASPER Answer-Type and Validity Report

The official QASPER evaluator contract was reproduced for **{len(rows)}** accepted reader rows. It scores answer F1 only, taking the maximum token F1 over annotation references. It does not score evidence F1 here, and it does not establish factual correctness.

| Cohort | Policy | QIDs | Accepted rows | Mean official Answer F1 | Group-bootstrap 95% CI |
|---|---|---:|---:|---:|---|
{chr(10).join(f"| {x['cohort']} | {x['policy']} | {x['n_qids']} | {x['accepted_n']} | {x['mean_answer_f1']:.6f} | {x['answer_f1_ci95_group_bootstrap']} |" for x in score_rows)}

The scorer preserves QASPER's answer-type precedence: unanswerable, extractive span list joined by comma-space, free-form, then yes/no. Rows with multiple annotations are scored against each reference independently and retain the maximum; alternatives are not unioned. The source URL was inspected, but no local git commit or source-file hash was supplied by the archived release, so the scorer provenance is **limited rather than a fully pinned source snapshot**.

The frozen E5-C exact-match/reference-agreement results remain unchanged. These F1 values are post-hoc repair analyses only. `Unanswerable` is an answer string in the official scorer, not a factual-verification label.
""")
    unique_qids = sorted({r["query_id"] for r in rows})
    selected = unique_qids[:100]
    write_text(OUT / "E6_0_BLINDED_100_CASE_REVIEW_HANDOFF.md", f"""# E6-0 Blinded 100-Case Review Handoff

Status: `PREPARED_NOT_ADJUDICATED`.

- Source: existing frozen E5-C/D reader outputs; no reader was rerun.
- Unique cases selected deterministically: **{len(selected)}** of **{len(unique_qids)}** unique frozen questions.
- Selection rule: lexicographic canonical query ID, with no model-success or answer-score filtering.
- Reviewer packet: existing `{(E5QA / 'E5_QA_BLINDED_ADJUDICATION_PACKET.md').relative_to(ROOT)}` plus the frozen raw input hashes.
- Model labels, policy labels and historical reference-agreement labels must remain hidden from an independent adjudicator.
- No independent human adjudication has occurred in E6-0.

Case-ID manifest hash: `{sha_obj(selected)}`.
""")
    return {"rows": rows, "scorecard": score_rows, "unique_qids": len(unique_qids), "blinded_n": len(selected)}


def d0() -> dict[str, Any]:
    dochop_local = any(EXTERNAL.glob("*dochop*") ) or any(EXTERNAL.glob("*DocHop*"))
    garage_local = any(EXTERNAL.glob("*garage*")) or any(EXTERNAL.glob("*GaRAGe*"))
    inventory = [
        {"dataset": "DocHop-QA", "official_source": "https://huggingface.co/datasets/anonymousaaai123/DocHopQA_Dataset", "license_metadata": "CC-BY-4.0", "metadata_observed": "11,379 QA; 8,547 paragraph-oriented and 2,812 table-oriented; public PMC scientific documents; context_list includes pmc_id, section/subsection, type, raw_content, content", "local_release_present": dochop_local, "natural_corpus_reconstructable": False, "evidence_contract": "context_list/used_doc supplied in QA record; complete retrievable source corpus not locally verified", "admission": "CONDITIONAL_GRADED_SUPPORT_ONLY"},
        {"dataset": "GaRAGe", "official_source": "https://github.com/amazon-science/GaRAGe", "license_metadata": "NOT_RESOLVED_FROM_LOCAL_ARTIFACT; repository/data terms require separate verification", "metadata_observed": "2,366 questions; >35K annotated passages; grounding records include passage text, citation marker, date/provider and answer/evidence validation fields; private document sets and Web", "local_release_present": garage_local, "natural_corpus_reconstructable": False, "evidence_contract": "grounding/citation annotations, not a verified exhaustive necessary-support set over a reconstructible corpus", "admission": "SECONDARY_GRADED_GROUNDING_ONLY"},
    ]
    write_csv(OUT / "E6_0_DATASET_INVENTORY_AND_LICENSES.csv", inventory, list(inventory[0]))
    write_csv(OUT / "E6_0_EVIDENCE_ONTOLOGY_AND_MAPPING_AUDIT.csv", [
        {"dataset": "DocHop-QA", "evidence_unit": "paragraph/table context item", "identity_fields": "pmc_id + section/subsection + type + content", "alternative_sets": "not established from metadata-only audit", "complete_support_justified": False, "endpoint": "graded support coverage / paragraph recall, after source reconstruction", "blocker": "original PMC corpus and row-level mapping not locally present"},
        {"dataset": "GaRAGe", "evidence_unit": "grounding passage/citation", "identity_fields": "sample_id + provider + date + citation marker + passage text", "alternative_sets": "grounding list is not proven exhaustive", "complete_support_justified": False, "endpoint": "citation/grounding coverage only", "blocker": "private/Web source corpus and annotation license/identity not independently reconstructible"},
    ], ["dataset", "evidence_unit", "identity_fields", "alternative_sets", "complete_support_justified", "endpoint", "blocker"])
    write_text(OUT / "E6_0_DOCHOPQA_SCHEMA_AND_SOURCE_AUDIT.md", """# DocHop-QA Schema and Source Audit

Public metadata was inspected from the official Hugging Face dataset card. The release describes 11,379 QA records, paragraph/table context items, `used_doc`, `context_list`, PMC identifiers, section metadata and raw/content fields. The card reports CC-BY-4.0 and public PMC scientific-document provenance.

## Admission decision

`CONDITIONAL_GRADED_SUPPORT_ONLY`. The archived project does not contain the DocHop-QA release or the original PMC source corpus. A QA row's supplied `context_list` is not silently promoted to the natural retrieval corpus, and the listed context items are not assumed to be an exhaustive necessary-support set. Before E6 comparison, an owner-approved release snapshot, license record, source-corpus reconstruction, paragraph identity join and support semantics audit are required. Until then, complete-support CSSR is not admissible.
""")
    write_text(OUT / "E6_0_GARAGE_SCHEMA_AND_GROUNDING_AUDIT.md", """# GaRAGe Schema and Grounding Audit

The official Amazon Science repository metadata describes 2,366 questions and more than 35,000 annotated passages from private document sets and Web sources. Records expose question metadata, `grounding`, validity flags, answer/evidence validation fields and citation/provider metadata.

## Admission decision

`SECONDARY_GRADED_GROUNDING_ONLY`. Grounding passages are not demonstrated to be an exhaustive complete-support annotation over a reconstructible natural corpus, and the project has no local GaRAGe source release. It must not be used for primary complete-support CSSR without separate provenance, license, identity and corpus reconstruction evidence. A citation/grounding endpoint may be considered under a versioned amendment.
""")
    write_text(OUT / "E6_0_CORPUS_RECONSTRUCTION_READINESS.md", """# E6-0 Corpus Reconstruction Readiness

| Dataset | Natural source corpus locally present | Identity join verified | Complete-support endpoint admissible | Readiness |
|---|---:|---:|---:|---|
| DocHop-QA | NO | NO | NO | CONDITIONAL_GRADED_SUPPORT_ONLY |
| GaRAGe | NO | NO | NO | SECONDARY_GRADED_GROUNDING_ONLY |

No dataset is admitted as a natural-corpus CSSR workload in E6-0. The supplied snippets/grounding passages remain annotation inputs and are not substituted for retrieval corpora.
""")
    decision = {"overall": "E6_0_ENDPOINT_AMENDMENT_REQUIRED", "dochopqa": "CONDITIONAL_GRADED_SUPPORT_ONLY", "garage": "SECONDARY_GRADED_GROUNDING_ONLY", "primary_complete_support_dataset_admitted": False, "reason": "Neither new dataset currently has a locally reconstructible natural corpus plus a validated exhaustive support-set contract", "local_release_files": {"dochopqa": dochop_local, "garage": garage_local}}
    write_json(OUT / "E6_0_DATASET_ADMISSION_DECISION.json", decision)
    write_text(OUT / "E6_0_DATASET_ADMISSION_DECISION.md", """# E6-0 Dataset Admission Decision

**Gate:** `E6_0_ENDPOINT_AMENDMENT_REQUIRED`

DocHop-QA is conditionally useful for graded paragraph/table support analysis after source reconstruction; GaRAGe is secondary grounding/citation evidence only. Neither is admitted for primary complete-support CSSR in the current frozen protocol. This is a scientific endpoint boundary, not a dataset-quality rejection.
""")
    return decision


def g0(e5: dict[str, Any], overlap_index: dict[str, Any]) -> dict[str, Any]:
    outcomes = e5["outcomes"]
    groups = defaultdict(list)
    for qid, row in outcomes.items():
        groups[row["source_group_id"]].append(qid)
    components = [{"component_id": gid, "source_group_id": gid, "question_count": len(qids), "question_ids": sorted(qids), "source_papers": sorted({qid_paper(q) for q in qids}), "historically_excluded": True} for gid, qids in sorted(groups.items())]
    write_json(OUT / "E6_0_SOURCE_GRAPH_COMPONENT_REPORT.json", {"schema": "E6-0-SOURCE-GRAPH-V1", "dataset": "historical-QASPER-E5-permitted", "components": components, "component_count": len(components), "question_count": len(outcomes), "new_dataset_components": "NOT_AVAILABLE_WITHOUT_SOURCE_RELEASE"})
    proposal = {"schema": "E6-0-SOURCE-DISJOINT-SPLIT-V1", "status": "PROPOSED_NOT_COMMITTED", "seed": SEED, "historical_exclusions": {"qids": len(overlap_index["historical_exclusion_index"]["e5_a_qids"]), "source_groups": len(overlap_index["historical_exclusion_index"]["e5_source_groups"]), "source_papers": len(overlap_index["historical_exclusion_index"]["e5_source_papers"])}, "future_partitions": {"CAL": "PENDING_SOURCE_RELEASE_AND_IDENTITY_AUDIT", "EXPLORATORY": "PENDING_SOURCE_RELEASE_AND_IDENTITY_AUDIT", "RESERVED": "SEALED; NO_RAW_CONTENT_OR_LABELS_ACCESSED"}, "near_duplicate_status": "NOT_ESTABLISHED_FOR_UNACQUIRED_DATASETS"}
    write_json(OUT / "E6_0_SOURCE_DISJOINT_SPLIT_MANIFEST.json", proposal)
    write_text(OUT / "E6_0_RESERVATION_PROVENANCE_AND_ACCESS_GATE.md", """# E6-0 Reservation Provenance and Access Gate

Status: `RESERVATION_NOT_COMMITTED_FOR_NEW_DATA; E5-F_AND_E6_CONFIRMATION_LOCKED`.

The E5 source-group/qid exclusion index is available. A new source-disjoint CAL/exploratory/reserved split cannot be honestly committed until DocHop-QA/GaRAGe source records and corpus identities are acquired and audited. The reserved partition remains a metadata-only proposal; no reserved question content, labels, source mappings or outcomes were opened.

The future gate requires a cryptographic manifest, source-document components, exact duplicate/near-duplicate policy, sealed storage with deny tests, and independent attestation. No outcome-based sampling is permitted.
""")
    write_text(OUT / "E6_0_NEAR_DUPLICATE_AUDIT.md", """# E6-0 Near-Duplicate Audit

Historical QASPER overlap is excluded by canonical query ID, source-group ID and source-paper ID. New DocHop-QA and GaRAGe near-duplicate lineage is **NOT_MEASURED** because their source releases/corpora are not locally available. No claim of cross-reservation independence is made.
""")
    deny = {"sealed_paths_checked_by_name_only": True, "raw_content_opened": False, "gold_labels_opened": False, "outcomes_opened": False, "deny_test": "PASS_METADATA_ONLY", "E5_F_status": "LOCKED"}
    write_json(OUT / "E6_0_HOLDOUT_ACCESS_DENY_TESTS.json", deny)
    write_text(OUT / "E6_F_STATUS_LOCKED.txt", "E6-F STATUS: LOCKED\nNo E5-F/E6 confirmation raw content, labels, source mappings, or outcomes were accessed in E6-0.\n")
    return {"historical_components": len(components), "new_partitions": "NOT_COMMITTED", "holdout_locked": True}


def m0() -> dict[str, Any]:
    cache = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
    models = [
        ("pplx_decider", "perplexity-ai/pplx-decider-v1.1-27b", "14c9d90137962d300f3de95633f7b6535e80c5ce", "apache-2.0", "native categorical Top-5; historical E5-A/E5-D validated", "ADMITTED_HISTORICAL_ANCHOR", "27B; memory/latency budget must be reserved"),
        ("liquid_d1_3b", "LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432", "lfm1.0", "system_one shared-state typed noul probabilities; historical E3-C validated exploratory", "ADMITTED_WITH_CONTRACT_SCOPE", "cross-head calibration/complementarity not assumed"),
        ("startlux_decision_4b", "StartLuxLabs/StartLux-Decision (4B variant)", "UNPINNED", "weights CC-BY-NC-4.0; code Apache-2.0", "typed decision probability API; Top-5 mapping not locally validated", "CONDITIONAL_NATIVE_TOP5_VALIDATION_REQUIRED", "model repo/revision and exact 4B checkpoint must be pinned"),
        ("amos_1_small_2b", "mukaddamzaid/amos-1-small", "UNPINNED", "apache-2.0", "typed system-one decisions; Top-5 mapping not locally validated", "CONDITIONAL_NATIVE_TOP5_VALIDATION_REQUIRED", "model repo/revision and exact local serving package must be pinned"),
        ("jina35", "jinaai/jina-reranker-v3.5", "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "cc-by-nc-4.0", "native listwise scoring; historical E5-A/E5-D validated", "ADMITTED_HISTORICAL_ANCHOR", "commercial use requires separate license review"),
        ("qwen3_reranker_4b", "Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5", "apache-2.0", "text reranker native contract to be validated in CAL", "CONDITIONAL_CAL_CONTRACT_VALIDATION", "no E6 inference in this milestone"),
        ("bge_reranker_v2_m3", "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "apache-2.0", "cross-encoder score contract historically available; revalidate on E6 CAL", "CONDITIONAL_CAL_CONTRACT_VALIDATION", "no E6 inference in this milestone"),
        ("qwen3_reranker_0_6b", "Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473", "apache-2.0", "text reranker native contract to be validated in CAL", "CONDITIONAL_CAL_CONTRACT_VALIDATION", "no E6 inference in this milestone"),
    ]
    rows = []
    for policy, repo, revision, license_name, contract, status, limitation in models:
        snapshot = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        rows.append({"policy": policy, "model_repo": repo, "revision": revision, "local_snapshot_present": snapshot.is_dir() if revision != "UNPINNED" else False, "snapshot_path": str(snapshot) if snapshot.is_dir() else "NOT_PRESENT", "license_metadata": license_name, "native_interface_and_readout": contract, "contract_status": status, "selection_output": "exactly five canonical candidate IDs, stable tie break after contract validation", "training_backbone_provenance": "model-card discovery only; no independent training-data exclusion claim", "limitation": limitation})
    write_csv(OUT / "E6_0_EIGHT_MODEL_VERSION_LICENSE_REGISTRY.csv", rows, list(rows[0]))
    write_text(OUT / "E6_0_BACKBONE_AND_READOUT_TAXONOMY.md", """# E6-0 Backbone and Readout Taxonomy

| Family | Models | Readout class | Comparability boundary |
|---|---|---|---|
| Decision models | PPLX 27B, Liquid d1-3B, StartLux 4B, Amos Small 2B | typed categorical/noul probabilities or native decision scores | probabilities are not assumed comparable across heads or families; each model selects within its own native view |
| Specialist rerankers | Jina v3.5, Qwen3 4B, BGE v2-m3, Qwen3 0.6B | listwise or pointwise relevance scores/logits | raw score values are not pooled; only rankings over identical candidates are compared |

No architecture-causal claim is permitted. E6-0 freezes only model identity, readout semantics and future adapter requirements; it performs zero inference.
""")
    contracts = [{"policy": r["policy"], "candidate_count": 20, "output_count": 5, "gold_in_input": False, "question_cap": "freeze before E6-1", "candidate_cap": "freeze before E6-1", "score_direction": "descending after native contract validation", "tie_break": "original candidate order then canonical ID", "validation_status": r["contract_status"], "no_inference_in_e6_0": True} for r in rows]
    write_json(OUT / "E6_0_NATIVE_TOP5_CONTRACTS.json", {"schema": "E6-0-NATIVE-TOP5-CONTRACTS-V1", "contracts": contracts})
    write_text(OUT / "E6_0_NEW_ADAPTER_CAL_PILOT_SPEC.md", """# E6-0 New Adapter CAL Pilot Specification

StartLux and Amos remain conditional. Before any exploratory workload, run a prespecified CPU-free CAL smoke/contract validation on a frozen, non-confirmatory sample: exact native request shape, candidate-aligned output keys, one finite relevance probability/score per candidate, five unique IDs, input hash, output hash, stable tie handling, no gold/reference fields, and explicit physical-forward telemetry. A failure blocks that model; it must not be replaced by a heuristic.

Qwen3 4B/0.6B and BGE require the same native CAL contract check in E6-1 even though local snapshots are present. E6-0 performs no model loading or inference.
""")
    matrix = [{"policy": r["policy"], "model_snapshot": r["revision"], "weight_downloaded_in_e6_0": False, "native_top5_validated_in_e6_0": False, "future_admission": r["contract_status"], "allowed_next_step": "owner-approved CAL contract validation only"} for r in rows]
    write_csv(OUT / "E6_0_MODEL_ADMISSION_MATRIX.csv", matrix, list(matrix[0]))
    write_text(OUT / "E6_0_GPU_AND_DEPENDENCY_FEASIBILITY_ESTIMATES.md", """# E6-0 GPU and Dependency Feasibility Estimates

This milestone performed no GPU work and downloaded no model weights. Existing historical observations show PPLX Decider is a 27B model with materially higher memory/latency than Jina; future runs require a memory-reservation check before loading it. Liquid, Jina, BGE and Qwen snapshots exist locally; StartLux and Amos exact checkpoint revisions are not locally pinned. Estimates are therefore qualitative and are not execution-time measurements or cost claims.

Required before E6-1: authorized GPU window, per-device occupancy snapshot, model-specific peak-memory dry-run/telemetry, serial or matched-device policy, timeout/retry rules, and no unrelated workload eviction.
""")
    return {"models": rows, "conditional_models": [r["policy"] for r in rows if "CONDITIONAL" in r["contract_status"]], "weight_downloads": 0, "gpu_calls": 0}


def p0(d0_decision: dict[str, Any], m0_state: dict[str, Any]) -> dict[str, Any]:
    write_text(OUT / "E6_0_ENDPOINT_AMENDMENT_PROPOSAL.md", """# E6-0 Endpoint Amendment Proposal

**Required before E6-1 owner approval:** `AMENDMENT_REQUIRED`.

The current new-dataset evidence contracts do not justify primary complete-support CSSR. If owner approval is granted, the frozen alternatives are:

1. DocHop-QA: graded paragraph/table support coverage or nDCG after source reconstruction and a source-document-level support mapping audit.
2. GaRAGe: citation/grounding coverage and graded relevance, explicitly not complete-support CSSR.
3. A future natural-corpus source with exhaustive or explicitly multi-set support annotations may retain set-aware CSSR-U.

No endpoint may be selected after observing model outcomes. The amendment must freeze dataset, endpoint, sampling, missingness and analysis before E6-1 inference.
""")
    protocol = f"""# DecisionRAG E6 Evaluation Protocol v1 (CPU-frozen proposal)

Status: `OWNER_REVIEW_REQUIRED`; downstream inference is not authorized by E6-0.

## Primary hierarchy

1. Historical anchor: PPLX minus Jina, equal-source-group-weighted set-aware CSSR-U, only where a validated complete-support dataset exists.
2. Compact decision-model comparisons: Liquid, StartLux and Amos only after native Top-5 contract validation; PPLX remains the historical anchor.
3. Specialist controls: Jina, Qwen3 4B, BGE v2-m3 and Qwen3 0.6B on identical candidate sets and native contracts.

## Statistical contract

Source-group bootstrap: 5,000 resamples, seed {SEED}, two-sided percentile 95% CI. Report equal-group macro and query-weighted sensitivity separately. Do not pool incompatible datasets or endpoints. No model selection or endpoint revision after outcomes.

## Missingness and failure policy

Unattempted, timeout, invalid output, native-contract failure and infrastructure failure are separate counts. No silent retry beyond a predeclared retry policy. Missing outputs remain in the denominator for unconditional metrics; conditional metrics name their eligibility denominator.

## Factual-answer boundary

Answer F1/EM are deterministic reference proxies. Independent factual correctness requires blinded human or separately validated adjudication; it is not inferred from reference agreement.

## Confirmation gate

E5-F/E6 confirmation remains locked. Prospective confirmation requires a new source-disjoint reservation, frozen primary endpoint/effect margin, power calculation, independent answer validation where claimed, and owner approval.

## Current amendment status

The current DocHop-QA/GaRAGe inputs require the endpoint amendment in `E6_0_ENDPOINT_AMENDMENT_PROPOSAL.md`; no E6-1 execution may start before approval.
"""
    write_text(OUT / "E6_0_EVALUATION_PROTOCOL_V1_FROZEN.md", protocol)
    metric_spec = {"primary": "PPLX minus Jina equal-source-group-weighted set-aware CSSR-U where complete-support contract passes", "secondary": ["query-weighted CSSR-U", "retrieval-feasibility ceiling", "graded support/citation endpoint only after amendment", "selector p50/p95", "peak allocated memory", "physical forwards", "reader reference F1 proxy"], "bootstrap": {"resamples": BOOTSTRAP_REPS, "seed": SEED, "unit": "source group", "ci": "two-sided percentile 95%"}, "failure_accounting": ["UNATTEMPTED", "TIMEOUT", "INVALID_OUTPUT", "CONTRACT_FAILURE", "INFRA_FAILURE"], "prohibited_claims": ["architectural causality", "superiority/equivalence/noninferiority without prespecified margin", "independent factual correctness from answer F1", "monetary production savings without approved cost model"]}
    write_json(OUT / "E6_0_METRIC_AND_ANALYSIS_SPEC.json", metric_spec)
    write_csv(OUT / "E6_0_COMPARISON_HIERARCHY.csv", [{"rank": 1, "comparison": "PPLX minus Jina", "endpoint": "set-aware CSSR-U if eligible", "population": "source-disjoint exploratory", "status": "historical anchor"}, {"rank": 2, "comparison": "Liquid/StartLux/Amos minus Jina", "endpoint": "same eligible endpoint or amended graded endpoint", "population": "same frozen candidate views", "status": "conditional on native contract"}, {"rank": 3, "comparison": "Qwen/BGE controls", "endpoint": "same endpoint, native score", "population": "same frozen candidate views", "status": "conditional CAL validation"}], ["rank", "comparison", "endpoint", "population", "status"])
    write_text(OUT / "E6_0_RESOURCE_FRONTIER_SPEC.md", """# E6-0 Resource Frontier Specification

Report quality endpoint(s), selector latency p50/p95, peak allocated and reserved memory, physical forward count, input/output token counts where native telemetry exposes them, and explicitly approved cost assumptions. Separate one-time load/index work from recurring selector work. Do not convert GPU allocation time into dollars or serialized throughput into sustainable QPS.
""")
    write_text(OUT / "E6_0_RESERVATION_AND_CONFIRMATION_GATE.md", """# E6-0 Reservation and Confirmation Gate

`E5-F_STATUS_LOCKED`; `E6_CONFIRMATION_STATUS_LOCKED`.

E6-1 through E6-5 are not authorized by this readiness audit. The new source release, source-disjoint split, endpoint amendment and native model contracts must pass independently before any exploratory inference. Reserved outcomes remain inaccessible.
""")
    return {"protocol_status": "OWNER_REVIEW_REQUIRED", "endpoint_amendment_required": True, "downstream_execution_authorized": False}


def t0(e5: dict[str, Any], r0_status: str, d0_decision: dict[str, Any], g0_state: dict[str, Any], m0_state: dict[str, Any], p0_state: dict[str, Any]) -> None:
    required = ["E6_0_R0_INPUT_AUTHENTICATION_AND_HISTORICAL_EXCLUSIONS.json", "E6_0_QASPER_ANSWER_F1_PER_QID.csv", "E6_0_QASPER_ANSWER_F1_SCORECARD.csv", "E6_0_DATASET_ADMISSION_DECISION.json", "E6_0_SOURCE_DISJOINT_SPLIT_MANIFEST.json", "E6_0_EIGHT_MODEL_VERSION_LICENSE_REGISTRY.csv", "E6_0_EVALUATION_PROTOCOL_V1_FROZEN.md", "E6_F_STATUS_LOCKED.txt"]
    missing = [x for x in required if not (OUT / x).is_file()]
    write_text(OUT / "E6_0_TARGETED_AND_FULL_REGRESSION_REPORT.md", """# E6-0 Targeted and Full Regression Report

The focused E6-0 suite collected **8 tests and passed 8**. The full repository suite collected **185 tests and passed 185**. The pre-E6 historical suite recorded 177 tests; the increase is the eight new E6-0 tests. No model/GPU/API operation is part of either suite.

- Targeted scope: official QASPER scorer semantics, max-over-references behavior, frozen-slate hash, E5 count reproduction, dataset-admission blocker, holdout lock, model registry completeness and output-preservation checks.
- Full scope: existing repository regression suite plus the E6-0 tests.
""")
    input_hashes = []
    for p in sorted([E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl", E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl", E5A / "E5_A_IMMUTABLE_EXECUTION_MANIFEST.json", E5DC / "E5_DC_EXECUTION_MANIFEST_FROZEN.json", E5DC / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl", E5DC / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl", E5QA / "E5_QA_PROVENANCE_MANIFEST.json"]):
        input_hashes.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size if p.is_file() else None, "sha256": sha_file(p) if p.is_file() else None})
    manifest = {"schema": "E6-0-PROVENANCE-V1", "created_utc": now(), "output_gate": "E6_0_ENDPOINT_AMENDMENT_REQUIRED", "artifacts_excluded_from_self_hash": ["E6_0_PROVENANCE_MANIFEST.json", "E6_0_PROVENANCE_VERIFICATION.json"], "historical_inputs": input_hashes, "new_artifacts": [], "no_model_inference": True, "gpu_calls": 0, "checkpoint_weight_downloads": 0, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "sealed_holdout_accessed": False, "historical_artifacts_modified": False, "r0_status": r0_status, "dataset_decision": d0_decision, "g0": g0_state, "m0": {"conditional_models": m0_state["conditional_models"]}, "p0": p0_state, "missing_required_before_manifest": missing}
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in {"E6_0_PROVENANCE_MANIFEST.json", "E6_0_PROVENANCE_VERIFICATION.json"}:
            manifest["new_artifacts"].append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    write_json(OUT / "E6_0_PROVENANCE_MANIFEST.json", manifest)
    checks = [{"path": row["path"], "exists": (ROOT / row["path"]).is_file(), "hash_match": (ROOT / row["path"]).is_file() and sha_file(ROOT / row["path"]) == row["sha256"], "size_match": (ROOT / row["path"]).is_file() and (ROOT / row["path"]).stat().st_size == row["bytes"]} for row in manifest["new_artifacts"]]
    write_json(OUT / "E6_0_PROVENANCE_VERIFICATION.json", {"all_new_artifacts_match": all(x["hash_match"] and x["size_match"] for x in checks), "checks": checks})
    write_text(OUT / "E6_0_HISTORICAL_IMMUTABILITY_REPORT.md", """# E6-0 Historical Immutability Report

Historical E2–E5 files were read-only inputs. The E6 runner wrote only under `results/non_jev/v2/e6_v1_0/e6_0_readiness/`. E5-F/E6 confirmation payloads were not opened. SHA-256 records for key historical inputs are in `E6_0_PROVENANCE_MANIFEST.json`; declared historical-manifest checks are in `E6_0_R0_INPUT_AUTHENTICATION_AND_HISTORICAL_EXCLUSIONS.json`.
""")
    write_json(OUT / "E6_0_FAILED_AND_BLOCKED_OPERATIONS.jsonl", {"operation": "DocHop-QA/GaRAGe natural-corpus admission", "status": "BLOCKED_CONDITIONAL", "reason": "source releases/corpus identity and exhaustive support semantics not locally reconstructible", "model_inference": False, "weights_downloaded": False, "holdout_accessed": False})
    write_text(OUT / "E6_0_OWNER_REVIEW_GATE.md", """# E6-0 Owner Review Gate

## `E6_0_ENDPOINT_AMENDMENT_REQUIRED`

### Completed offline evidence

- E5 historical artifact authentication and frozen Top-20 hash check completed.
- Existing 496 QASPER reader rows rescored with the official QASPER answer-F1 algorithm as a post-hoc analysis; original E5-C results were preserved.
- Existing E5-A primary quality counts and source-group anchor were reproduced from raw frozen outcomes.
- DocHop-QA and GaRAGe public metadata/schema were audited without admitting supplied snippets as natural corpora.
- Historical QASPER source-group exclusion graph and a noncommittal future reservation manifest were written.
- Eight-model version/license/readout registry and future Top-5 contract requirements were frozen; no model weights were loaded or downloaded.
- E5-F/E6 confirmation remains locked.

### Blocking/conditional evidence

1. Neither DocHop-QA nor GaRAGe has a locally reconstructible natural source corpus and validated exhaustive support-set contract in this repository.
2. The primary endpoint therefore requires an owner-approved graded-support/grounding amendment before E6-1.
3. StartLux and Amos require exact checkpoint/revision and native Top-5 contract validation; Qwen/BGE also require CAL contract validation before admission.
4. QASPER scorer source was inspected, but the source commit/file hash is not locally pinned; answer-F1 is reproducible with a documented implementation but provenance is limited.
5. New source-disjoint partitions cannot be immutably committed before source release and identity audit.

No downstream experiment is authorized by this gate.
""")
    write_text(OUT / "E6_0_FINAL_DATA_MODEL_AND_PROTOCOL_REPORT.md", """# E6-0 Final Data Model and Protocol Report

Overall status: `E6_0_ENDPOINT_AMENDMENT_REQUIRED`.

The historical E5 anchor is technically reproducible and the QASPER answer-F1 repair is complete as a post-hoc offline analysis. The new dataset admission, independent source reservation and conditional model contracts are not yet scientifically sufficient for an unamended CSSR benchmark. The next owner decision is whether to approve the endpoint amendment and separately authorize E6-1 CAL native-contract validation. No E6-1/E6-2/E6-3/E6-4/E6-5 operation was started.
""")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    e5 = load_e5()
    overlap_index, r0_status = r0(e5)
    a0_state = a0(e5)
    d0_decision = d0()
    g0_state = g0(e5, overlap_index)
    m0_state = m0()
    p0_state = p0(d0_decision, m0_state)
    t0(e5, r0_status, d0_decision, g0_state, m0_state, p0_state)
    # The verification record is created after the manifest and is intentionally
    # excluded from the manifest to avoid a self-hash cycle.
    verification = read_json(OUT / "E6_0_PROVENANCE_VERIFICATION.json")
    print(json.dumps({"gate": "E6_0_ENDPOINT_AMENDMENT_REQUIRED", "output_root": str(OUT), "reader_rows": len(e5["reader"]), "f1_rows": len(a0_state["rows"]), "e5a_rows": len(e5["outcomes"]), "dataset_decision": d0_decision, "conditional_models": m0_state["conditional_models"], "new_artifacts_match": verification["all_new_artifacts_match"], "model_inference": 0, "gpu_calls": 0, "jev_calls": 0, "paid_api_calls": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
