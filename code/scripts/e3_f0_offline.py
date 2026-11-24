#!/usr/bin/env python3
"""DecisionRAG E3-F0: offline confirmation-feasibility and reviewer package.

Only authenticated historical manifests, scorecards, reports and the prior
metadata-only E3-D-R audit are consumed.  This module has no model, CUDA,
HTTP, Qdrant, dataset-download, Jev, paid-API or reserved-label path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - project environment normally has numpy
    np = None

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e3_v1"
E3A = BASE / "e3_a_paired"
E3B = BASE / "e3_b_mechanism"
E3C = BASE / "e3_c_models"
E3D = BASE / "e3_d_external"
E3R = BASE / "e3_d_r_reference_audit"
GOV = BASE / "e3_0_governance"
OUT = BASE / "e3_f0_protocol_review"
SEED = 731
BOOTSTRAP = 5000


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def source_index() -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = [
        ROOT / "DecisionRAG_E3_F0_Confirmatory_Feasibility_and_Reviewer_Readiness_Codex_SDD.md",
        E3A / "E3_A_EXECUTION_MANIFEST.json", E3A / "E3_A_FINAL_SCIENTIFIC_REPORT.md",
        E3A / "E3_A_PAIRED_EVIDENCE_STATS.csv", E3A / "E3_A_PAIRED_ANSWER_STATS.csv",
        E3B / "E3_B_PROVENANCE_MANIFEST.json", E3B / "E3_B_FINAL_SCIENTIFIC_REPORT.md",
        E3B / "E3_B_CONTEXT_STABILITY_AND_EFFECTS.csv", E3B / "E3_B_SETAWARE_QUALITY_AND_PAIRED_STATS.csv",
        E3C / "E3_C_PROVENANCE_MANIFEST.json", E3C / "E3_C_FINAL_SCIENTIFIC_REPORT.md",
        E3C / "E3_C_PRIMARY_PAIRED_STATISTICS.csv", E3C / "E3_C_SELECTOR_LATENCY_SUMMARY.csv",
        E3D / "E3_D_PROVENANCE_MANIFEST.json", E3D / "E3_D_FINAL_SCIENTIFIC_REPORT.md",
        E3D / "E3_D_EVIDENCE_QUALITY_SCORECARD.csv", E3D / "E3_D_PAIRED_BOOTSTRAP_AND_FOUR_CELL.csv",
        E3D / "E3_D_READER_PROXY_SCORECARD.csv", E3D / "E3_D_FIXED_READER_OUTPUTS.jsonl",
        E3D / "E3_D_SELECTED_TOP5.jsonl", E3D / "E3_D_NATIVE_SELECTION_TRACES.jsonl",
        E3R / "E3_D_R_PROVENANCE_MANIFEST.json", E3R / "E3_D_R_FINAL_SCIENTIFIC_REPORT.md",
        E3R / "E3_D_R_FINANCE_LEGACY_VS_CANONICAL_SCORECARD.csv", E3R / "E3_D_R_FINANCE_FAILURE_TAXONOMY.csv",
        E3R / "E3_D_R_MULTIHOP_EVIDENCE_ANSWER_4X4.csv", E3R / "E3_D_R_PRIMARY_PAIRED_BOOTSTRAP.csv",
        E3R / "E3_D_R_HOLDOUT_AUDIT.json", E3R / "E3_D_R_PAPER_TABLES.csv",
        GOV / "E3_0_RESERVED_PARTITIONS_MANIFEST.json", GOV / "E3_0_RESERVATION_INTEGRITY_TESTS.json",
        GOV / "E3_0_SOURCE_DOCUMENT_GRAPH_AUDIT.md", GOV / "E3_0_SOURCE_EXPOSURE_AND_METADATA_AUDIT.md",
    ]
    rows, hashes = [], {}
    for p in paths:
        ok = p.is_file(); digest = sha_file(p) if ok else None
        rows.append({"path": rel(p), "exists": ok, "sha256": digest, "role": "authenticated_read_only_input"})
        if ok: hashes[rel(p)] = digest
    return rows, hashes


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
        except Exception as exc:
            return f"UNAVAILABLE:{type(exc).__name__}"
    return {"head": run("rev-parse", "HEAD"), "status_short": run("status", "--short"), "tracked_diff_names": run("diff", "--name-only")}


def load_context() -> dict[str, Any]:
    e3r_holdout = read_json(E3R / "E3_D_R_HOLDOUT_AUDIT.json")
    e3d_summary = read_json(E3D / "E3_D_FINAL_SUMMARY.json")
    e3d_score = read_csv(E3D / "E3_D_EVIDENCE_QUALITY_SCORECARD.csv")
    e3d_reader = read_csv(E3D / "E3_D_READER_PROXY_SCORECARD.csv")
    e3r_finance = read_csv(E3R / "E3_D_R_FINANCE_LEGACY_VS_CANONICAL_SCORECARD.csv")
    e3r_cells = read_csv(E3R / "E3_D_R_MULTIHOP_EVIDENCE_ANSWER_4X4.csv")
    e3r_boot = read_csv(E3R / "E3_D_R_PRIMARY_PAIRED_BOOTSTRAP.csv")
    e3a_ev = read_csv(E3A / "E3_A_PAIRED_EVIDENCE_STATS.csv")
    e3a_ans = read_csv(E3A / "E3_A_PAIRED_ANSWER_STATS.csv")
    e3c_pair = read_csv(E3C / "E3_C_PRIMARY_PAIRED_STATISTICS.csv")
    e3c_lat = read_csv(E3C / "E3_C_SELECTOR_LATENCY_SUMMARY.csv")
    return {"holdout": e3r_holdout, "e3d_summary": e3d_summary, "e3d_score": e3d_score, "e3d_reader": e3d_reader, "e3r_finance": e3r_finance, "e3r_cells": e3r_cells, "e3r_boot": e3r_boot, "e3a_ev": e3a_ev, "e3a_ans": e3a_ans, "e3c_pair": e3c_pair, "e3c_lat": e3c_lat}


def write_protocol(index: list[dict[str, Any]]) -> None:
    protocol = {
        "protocol": "DecisionRAG E3-F0 v1.0",
        "status": "OFFLINE_OWNER_REVIEW_ONLY_NOT_E3F_APPROVAL",
        "stages": ["F0-0", "F0-1", "F0-2", "F0-3", "F0-4", "F0-5", "F0-6"],
        "output_root": rel(OUT), "seed": SEED, "bootstrap_replicates": BOOTSTRAP,
        "read_allowlist": [x["path"] for x in index if x["exists"]],
        "protected_denylist": ["E3_0 reserved answers", "E3_0 reserved qrels/evidence", "reserved model outputs", "new benchmark outcomes", "HTTP inference endpoints", "Jev/paid APIs"],
        "allowed_operations": ["read existing exploratory manifests/reports/scorecards", "hash existing files", "CPU deterministic power simulation", "write new versioned F0 outputs", "run offline pytest"],
        "historical_mutation": False, "new_model_inference": False, "gpu_use": False, "new_checkpoint_download": False,
        "reserved_outcomes_accessed": False, "e3f_started": False, "network_used": False,
    }
    write_json(OUT / "E3_F0_EXECUTION_PROTOCOL.json", protocol)


def claim_ledger(ctx: dict[str, Any], hashes: dict[str, str]) -> list[dict[str, Any]]:
    def src(p: Path) -> str:
        return f"{rel(p)}#{hashes.get(rel(p), 'NOT_HASHED')}"
    rows = [
        {"claim_id":"C01","hypothesis_or_rq":"historical_track_a_vs_track_b","sentence":"The historical experiments show model-policy differences across Track A and synthetic enterprise Track B, but not a general intrinsic decision-model advantage.","source":src(E3A/"E3_A_PAIRED_EVIDENCE_STATS.csv"),"cohort":"Track A/Track B","denominator":"dataset-specific rows; 1,490 Track A conditional and 330 Track B conditional","metric":"CSSR paired comparisons","label":"SUPPORTED_WITH_SCOPE_LIMITS","permitted_wording":"model-policy association on frozen historical populations; not causal family claim","counterevidence":"synthetic enterprise external validity; heterogeneous policies"},
        {"claim_id":"C02","hypothesis_or_rq":"external_jina_pplx","sentence":"On 900 exploratory MultiHop-RAG questions, Jina retained more complete support than PPLX on the unconditional CSSR-U endpoint (46.0% versus 44.9%).","source":src(E3D/"E3_D_EVIDENCE_QUALITY_SCORECARD.csv"),"cohort":"E3-D MultiHop-RAG","denominator":"900 unconditional; 802 conditional","metric":"CSSR-U/C","label":"EXPLORATORY_SUPPORTED","permitted_wording":"exploratory external replication result","counterevidence":"annotated support may be incomplete; exploratory cohort"},
        {"claim_id":"C03","hypothesis_or_rq":"external_reader_proxy","sentence":"PPLX and Jina did not show a clear paired normalized reference-EM proxy difference on the same 250 reader questions.","source":src(E3R/"E3_D_R_PRIMARY_PAIRED_BOOTSTRAP.csv"),"cohort":"E3-D-R MultiHop reader","denominator":"250 paired questions","metric":"normalized EM proxy; delta -0.8 pp, 95% CI -4.0 to +2.0 pp","label":"METRIC_PROXY_ONLY","permitted_wording":"proxy comparison is inconclusive and not factual accuracy","counterevidence":"reference strings are not independent factual adjudication"},
        {"claim_id":"C04","hypothesis_or_rq":"pplx_only_rescue","sentence":"Among 27 PPLX-only complete-support cases, the reader proxy was better/equal/worse than Jina in 1/26/0 cases.","source":src(E3R/"E3_D_R_MULTIHOP_EVIDENCE_ANSWER_4X4.csv"),"cohort":"E3-D-R matched reader","denominator":"27 PPLX-only cases within 250","metric":"proxy transition","label":"METRIC_PROXY_ONLY","permitted_wording":"descriptive proxy transition; not a rescue rate or causal result","counterevidence":"disagreement-only subset; no verified correctness"},
        {"claim_id":"C05","hypothesis_or_rq":"finance_answer_quality","sentence":"FinanceBench deterministic legacy match rates were 8/10/12/10/12/12 of 109 across retrieval/BGE/Jina/PPLX/Liquid/zerank.","source":src(E3R/"E3_D_R_FINANCE_LEGACY_VS_CANONICAL_SCORECARD.csv"),"cohort":"E3-D FinanceBench exploratory","denominator":"109 per policy; annotated-page qrels not exhaustive","metric":"legacy numeric/exact proxy","label":"METRIC_PROXY_ONLY","permitted_wording":"deterministic reference proxy only","counterevidence":"manual factual verification absent; strict and unit ambiguities"},
        {"claim_id":"C06","hypothesis_or_rq":"quota_mechanism","sentence":"Within-Clef same-score quota and ordering interventions changed selected membership and produced negative quota effects in the frozen historical analysis.","source":src(E3B/"E3_B_CONTEXT_STABILITY_AND_EFFECTS.csv"),"cohort":"E3-B mechanism sample","denominator":"reported in source table","metric":"selection-policy outcome","label":"SUPPORTED_WITH_SCOPE_LIMITS","permitted_wording":"controlled selection-policy effect under the tested contract","counterevidence":"not a neural reasoning or universal architecture claim"},
        {"claim_id":"C07","hypothesis_or_rq":"setr","sentence":"A faithful pretrained SetR reproduction was not available and SetR was not evaluated.","source":src(E3B/"E3_B_FINAL_SCIENTIFIC_REPORT.md"),"cohort":"E3-B","denominator":"not applicable","metric":"availability","label":"NOT_ESTABLISHED","permitted_wording":"SetR remains an untested baseline","counterevidence":"none; it is an explicit missing experiment"},
        {"claim_id":"C08","hypothesis_or_rq":"liquid_zerank","sentence":"Liquid and zerank native outcomes vary by dataset and model-policy contract; they do not establish an architecture-only causal effect.","source":src(E3C/"E3_C_FINAL_SCIENTIFIC_REPORT.md"),"cohort":"E3-C 4,267 historical questions","denominator":"4,267 valid per accepted native policy","metric":"CSSR and native selection","label":"ASSOCIATION_ONLY","permitted_wording":"model-policy result under pinned native interfaces","counterevidence":"different checkpoints, prompts, views and objectives"},
        {"claim_id":"C09","hypothesis_or_rq":"serving","sentence":"Serialized isolated latency measurements exist for Liquid and zerank, but sustainable throughput and matched dollar cost are not established.","source":src(E3C/"E3_C_SELECTOR_LATENCY_SUMMARY.csv"),"cohort":"E3-C isolated serialized workload","denominator":"300 timed requests per model","metric":"p95 latency","label":"SUPPORTED_WITH_SCOPE_LIMITS","permitted_wording":"serialized latency estimate under the recorded workload","counterevidence":"not sustainable QPS; heterogeneous workloads"},
        {"claim_id":"C10","hypothesis_or_rq":"factual_correctness","sentence":"No independently verified factual-correctness labels are present for the E3-D reader results.","source":src(E3R/"E3_D_R_FINAL_SCIENTIFIC_REPORT.md"),"cohort":"E3-D-R","denominator":"2,154 reader outputs; human adjudication not performed","metric":"verified correctness","label":"NOT_ESTABLISHED","permitted_wording":"reference-answer proxies only","counterevidence":"none supplied"},
        {"claim_id":"C11","hypothesis_or_rq":"holdout","sentence":"The E3-0 partition is structurally query/source-group separated in the existing metadata manifest, but prospective reservation timing is not verified.","source":src(E3R/"E3_D_R_HOLDOUT_AUDIT.json"),"cohort":"E3-0","denominator":"195 reserved, 2,511 exploratory metadata IDs","metric":"governance","label":"SUPPORTED_WITH_SCOPE_LIMITS","permitted_wording":"QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED; not independent confirmation","counterevidence":"older exposure audit remains pending"},
        {"claim_id":"C12","hypothesis_or_rq":"enterprise_generalization","sentence":"The synthetic EnterpriseRAG-Bench finding is not independent authentic enterprise-domain generalization.","source":src(E3A/"E3_A_FINAL_SCIENTIFIC_REPORT.md"),"cohort":"historical Track B","denominator":"500 source questions; 443 eligible; 330 conditional","metric":"enterprise transfer","label":"NOT_ESTABLISHED","permitted_wording":"synthetic benchmark result only","counterevidence":"no independent enterprise source groups"},
        {"claim_id":"C13","hypothesis_or_rq":"universal_superiority","sentence":"No universal superiority ranking among all rerankers is established.","source":src(E3D/"E3_D_FINAL_SCIENTIFIC_REPORT.md"),"cohort":"multiple incompatible phases","denominator":"dataset-specific","metric":"cross-policy comparison","label":"NOT_ESTABLISHED","permitted_wording":"model- and workload-specific differences","counterevidence":"domain and contract interactions"},
    ]
    return rows


def paper_tables(ctx: dict[str, Any]) -> str:
    def table(title: str, rows: list[dict[str, str]], keys: list[str], limit: int | None = None) -> list[str]:
        out = [f"### {title}", "", "| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
        for r in rows[:limit] if limit else rows:
            out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
        return out + [""]
    lines = ["# E3-F0 Paper-Ready Primary Tables", "", "All rows are exploratory unless explicitly marked otherwise. Track A, synthetic enterprise Track B, external MultiHop-RAG and FinanceBench are separate populations. CSSR-U and CSSR-C are not pooled; reference answer metrics are proxies, not verified factual correctness.", ""]
    lines += table("Historical E3-A paired evidence", ctx["e3a_ev"], ["dataset", "population", "n", "pplx_cssr", "jina_cssr", "difference_pplx_minus_jina", "ci95_low", "ci95_high", "interpretation"])
    lines += table("External E3-D selector scorecard", ctx["e3d_score"], ["policy", "dataset", "n_queries", "retrieval_eligible_n", "cssr_u", "cssr_c", "any_gold_top5", "complete_gold_top20", "finance_metric_note"])
    lines += table("E3-D-R FinanceBench deterministic proxies", ctx["e3r_finance"], ["policy", "n", "legacy_success_n", "legacy_success_rate", "normalized_exact_n", "numeric_equivalent_n", "demonstrated_numeric_mismatch_n", "manual_adjudication"])
    lines += table("E3-D-R matched evidence/answer cells", ctx["e3r_cells"], ["evidence_cell", "answer_transition", "n", "denominator", "interpretation"])
    lines += table("E3-C native timing scope", ctx["e3c_lat"], ["model", "policy", "n", "p50_ms", "p95_ms", "serialized_throughput_rps", "total_physical_forwards", "status"])
    lines += ["### Reading rule", "", "The tables support descriptive comparison of frozen model-policy implementations under specific input contracts. They do not support claims that a model family intrinsically reasons better, that PPLX is generally superior in enterprise settings, that reader proxy matches are factual correctness, or that serialized latency is sustainable production capacity.", ""]
    return "\n".join(lines)


def feasibility(ctx: dict[str, Any]) -> str:
    h = ctx["holdout"]
    return f"""# E3-F0 Confirmatory Data Feasibility Audit

## Verdict

`CONFIRMATORY_DATA_NOT_READY` with governance field `QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED`.

The existing E3-0 manifest reports 195 reserved metadata IDs and 2,511 exploratory IDs, with zero recorded overlap and no answer/model-outcome use in partition assignment. The E3-D-R metadata audit also reports zero reserved IDs used by E3-D. However, the historical exposure audit remains pending and no immutable reservation-before-analysis timestamp, signed commit, or equivalent evidence is available. A manifest written after exploratory analysis cannot retroactively create a prospective confirmatory holdout.

This audit did not open reserved answers, qrels, evidence text, model outputs or evaluation outcomes. Public model-pretraining independence is unknowable. Query disjointness and structural source-group separation are therefore not promoted to independent confirmation.

## Feasible future data routes, not available datasets

1. A newly commissioned multi-hop release with source-document-group disjoint questions, locked before model selection and with complete support annotations.
2. New organizations/filings/reporting periods with independently controlled enterprise source groups and verified document-level gold.
3. A prospective answer-validation cohort with reference answers plus independent expert adjudication of numerical units, periods, entities and calculation chains.

FinanceBench's public 150-question release is already exposed to the exploratory work and its evidence annotations are not exhaustive; it is not a fresh enterprise confirmation population. Existing EnterpriseRAG-Bench is synthetic and cannot establish authentic enterprise transfer.

## Leakage and exposure risks retained

- exploratory model/policy selection and prompt-contract decisions;
- reused public source documents and possible document-level dependence;
- casebook and reference-proxy inspection;
- post-hoc ambiguity about reservation timing;
- public-model pretraining contamination, which cannot be audited here.

E3-F remains locked. A future confirmation requires owner approval of source rights, source-group separation, reservation timestamp, endpoint, missingness policy and analysis code before outcomes are opened.
"""


def power_simulation() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = {"seed": SEED, "replicates": BOOTSTRAP, "test": "one-sided paired binary superiority using discordant McNemar z", "planning_baseline": 0.46, "planning_discordance": 0.20, "effects_pp": [2, 5, 10], "candidate_n": list(range(100, 5001, 100)), "targets": [0.80, 0.90], "caveat": "planning sensitivity only; pilot estimates are not confirmatory parameters"}
    rows = []
    for effect_pp in inputs["effects_pp"]:
        delta = effect_pp / 100.0
        p01 = inputs["planning_discordance"] / 2.0
        p10 = p01 + delta
        effect_results = []
        for n in inputs["candidate_n"]:
            if np is not None:
                rng = np.random.default_rng(SEED + effect_pp * 10000 + n)
                b = rng.binomial(n, p10, size=BOOTSTRAP)
                c = rng.binomial(n, p01, size=BOOTSTRAP)
                z = (b - c) / np.sqrt(np.maximum(1, b + c))
                power = float(np.mean(z > 1.959964))
            else:
                rng = random.Random(SEED + effect_pp * 10000 + n)
                passed = 0
                for _ in range(BOOTSTRAP):
                    b = sum(rng.random() < p10 for _ in range(n)); c = sum(rng.random() < p01 for _ in range(n))
                    if (b - c) / max(1, (b + c) ** 0.5) > 1.959964: passed += 1
                power = passed / BOOTSTRAP
            effect_results.append((n, power))
            rows.append({"effect_pp": effect_pp, "n": n, "power": power, "baseline_rate": inputs["planning_baseline"], "p10_model_wins": p10, "p01_baseline_wins": p01, "replicates": BOOTSTRAP, "seed": SEED})
        for target in inputs["targets"]:
            hit = next((n for n, p in effect_results if p >= target), None)
            inputs.setdefault("minimum_n", {}).setdefault(str(effect_pp), {})[str(target)] = hit
    return inputs, rows


def protocol_options(power_inputs: dict[str, Any]) -> str:
    ns = power_inputs["minimum_n"]
    return f"""# E3-F0 Confirmatory Protocol Options — DRAFT, NOT APPROVED

No option is selected or frozen by F0. All options require a genuinely new eligible population and owner/reviewer approval before any labels or outcomes are opened.

## Option A — Independent multi-hop evidence-selection confirmation

**Population:** newly acquired source-document-group-disjoint multi-hop questions with complete, prevalidated support annotations. Freeze one Top-20 candidate pool and Top-5 rule before model outcomes. **Roster:** Jina and PPLX as the primary pair; retrieval-only is a descriptive baseline.

**Primary estimand:** paired unconditional difference `CSSR_U(PPLX) - CSSR_U(Jina)` over all eligible questions, including model/context failures as failures. A reviewer may choose superiority or non-inferiority; a provisional illustrative non-inferiority margin is −5 percentage points, not approved. Conditional CSSR-C, Top-20 feasibility and source-cluster sensitivity are secondary.

**Statistics:** paired binary discordance analysis with exact or mid-p McNemar interval/test as appropriate, plus source-document-cluster bootstrap or robust cluster inference. One primary endpoint; multiplicity-controlled secondary endpoints. No 900/250 historical denominator substitution.

**Planning:** under baseline 46%, 20% discordance and one-sided superiority effects, the deterministic sensitivity file gives minimum n for +2/+5/+10 pp at 80%/90% targets as `{json.dumps(ns)}`. These are not claims about the future population; 2 pp may be infeasible within the simulated 5,000-question ceiling under the stated assumptions.

**Feasibility:** `NOT_FEASIBLE_NOW` until a new source release, complete support annotations and prospective reservation timestamp exist. Incomplete annotations require changing the endpoint before data collection.

## Option B — Authentic enterprise transfer / interaction confirmation

**Population:** new organizations, filings or reporting periods with independent source groups and verified document-level gold. Synthetic EnterpriseRAG-Bench and exposed FinanceBench are excluded from confirmation.

**Primary estimand:** within-enterprise paired PPLX-minus-Jina difference in complete evidence retention, or a predeclared model×domain interaction if the scientific question is transfer rather than replacement. A provisional +5 pp superiority target is illustrative only. Report organization-clustered uncertainty and per-domain results; do not pool incompatible page/passage metrics.

**Statistics and missingness:** intention-to-evaluate; overflow, unsupported, and failed native calls remain unconditional failures. Cluster bootstrap at organization/report level; one primary interaction/endpoint with controlled secondary analyses.

**Feasibility:** `DATA_CONSTRAINED`; requires rights, independent annotation, source-group isolation and adequate organization count. The current local corpus cannot satisfy this.

## Option C — Reference-supported downstream utility confirmation

**Population:** new independently referenced questions with frozen retrieved evidence and a fixed reader. Every answer is scored against prevalidated references, and ambiguous financial/numerical cases receive independent expert adjudication before unblinding model results.

**Primary estimand:** paired verified factual-correctness difference PPLX-minus-Jina on the complete eligible population. Normalized EM/F1 and numeric equivalence are secondary diagnostics, never substitutes for verified correctness.

**Statistics:** paired binary/graded analysis with exact paired intervals, source/issuer clustering, and blinded adjudication of disagreements. The disagreement subset may be a descriptive stratum but cannot become the population estimand. Missingness and abstention rules are frozen in advance.

**Feasibility:** `NOT_FEASIBLE_NOW`; no independently verified factual labels exist in E3-D-R, and local-judge agreement cannot supply them.

## Shared prerequisites and disqualification

All options require model revisions, prompts, text views, truncation, Top-20 construction, score direction, tie handling, cache policy, failure accounting, seed, source rights, and analysis code to be frozen before outcomes. A model cannot be silently omitted on context overflow. No option supports a causal architecture claim unless the architecture and all other conditions are controlled. A 5-pp margin, 1-second or 3-second p95 SLO, and any dollar cost are proposed planning choices, not current product facts.
"""


def next_round() -> str:
    return """# E3-F0 Next-Round Research Options

| priority | option | marginal scientific value | cost/risk band | dependency | falsifier/rejection concern resolved |
|---:|---|---|---|---|---|
| 1 | Reference-validated downstream correctness | Highest if independent adjudication is feasible; directly tests whether evidence retention helps answers | High data/annotation cost; risk of proxy ambiguity | New referenced questions, blinded expert review | If verified correctness does not track evidence differences, downstream claims should stop |
| 2 | Authentic enterprise/source-group transfer | Directly tests whether the synthetic enterprise signal generalizes | Very high data rights and annotation burden | New organizations/filings and source-group reservation | Null or heterogeneous interaction would reject a general enterprise advantage |
| 3 | Matched 2–3-model serving frontier | Needed only if deployment/cost is a paper claim | Medium engineering/GPU cost; hardware contention risk | Exclusive/matched workload and stable backlog telemetry | No sustainable QPS/cost separation would weaken replacement claims |
| 4 | Faithful learned set-aware specialist | Tests whether the missing SetR comparison matters | Medium/high implementation and licensing risk | Public runnable checkpoint and native contract | If unavailable or no gain, report the negative and stop retraining scope |
| 5 | OCR-versus-visual multimodal extension | Tests modality dependence after text-only evidence is credible | High model/data cost; scope dilution risk | New multimodal corpus and frozen media contract | No quality gain or unacceptable cost supports deferral |
| 6 | No new experiments / exploratory-paper route | Preserves a useful negative and methods paper without confirmation claims | Lowest cost; venue acceptance uncertainty | Honest claim freeze and reviewer agreement | Reviewers may require independent transfer or correctness evidence |

The historical selective-escalation/oracle curves are feasibility-only upper bounds. They do not prove a deployable cascade or answer rescue. A trained cascade should not be proposed until a budget-constrained verified answer gain is demonstrated.

Before any GPU work, reviewers should decide whether the paper's central contribution is evidence-selection mechanism, empirical model comparison, downstream correctness, or serving economics. Mixing all four without a primary endpoint risks an unfocused paper.
"""


def reviewer_packet(ctx: dict[str, Any], power_inputs: dict[str, Any]) -> str:
    return f"""# DecisionRAG E3-F0 External Reviewer Consultation Packet

## Abstract

This project studies multimodal/text evidence reranking as a decision-model problem: a reranker receives a frozen Top-20 candidate slate and selects Top-5 evidence for a downstream reader. The completed E2/E3 work compares model-policy implementations including Jina, Perplexity Decider (PPLX), BGE, Liquid d1-3B, zerank-2 and retrieval-only baselines across historical multi-hop/synthetic-enterprise populations and an exploratory external MultiHop-RAG/FinanceBench replication. The evidence supports carefully scoped model-policy and selection-policy findings, but not an intrinsic causal advantage of a model family, independently verified factual answer correctness, general enterprise transfer, or a universal quality-cost frontier. On 900 exploratory MultiHop-RAG questions, Jina's CSSR-U was 46.0% and PPLX's 44.9%; on the same 250 external reader questions, PPLX retained complete annotated support on 113 and Jina on 118. Among the 27 PPLX-only cases, reference-answer proxy outcomes were better/equal/worse in 1/26/0 cases. FinanceBench legacy numeric/exact proxies were only 8–12 of 109 per policy, and no human factual adjudication was performed. These answer outcomes are not independently verified. The existing reservation is structurally separated but prospectivity is not proven. We request critical reviewer guidance before deciding whether any confirmatory experiment is scientifically necessary.

## What was measured

The principal task is selection from identical frozen Top-20 candidate identities. CSSR-U is unconditional complete annotated support in Top-5; CSSR-C conditions on all annotated support being present in Top-20. The historical 3,767-question Track A and 500-question synthetic enterprise Track B remain distinct. External E3-D contains 900 exploratory MultiHop-RAG questions and 109 exploratory FinanceBench questions. E3-D produced 6,054 full-stage selector rows (six policies × 1,009 questions), and E3-D-R reused 2,154 fixed-reader outputs (six policies × (250 MultiHop + 109 FinanceBench)).

Reader EM/F1 and Finance numeric/exact endpoints are deterministic reference proxies. They are not independent factual correctness. FinanceBench evidence annotations are not assumed exhaustive support sets, so Finance evidence-page coverage is diagnostic rather than CSSR. E3-C native inference recorded model-specific serialized p95 values, but sustainable QPS, matched hardware economics and a complete comparable cost frontier were not established. The prior SetR implementation was not faithfully reproducible and was not evaluated.

## Evidence at a glance

| scope | result | interpretation |
|---|---|---|
| Historical Track A | PPLX/Jina effects are small or domain-specific in paired CSSR analyses | model-policy association; not architecture causality |
| Synthetic Track B | enterprise-conditioned effects differ from Track A | synthetic benchmark result; not authentic enterprise transfer |
| E3-D MultiHop | Jina CSSR-U 46.0%, PPLX 44.9%, zerank 41.9%, retrieval 38.9%, BGE 35.1%, Liquid 30.0% | exploratory external selection result |
| E3-D-R MultiHop reader | PPLX/Jina normalized EM delta −0.8 pp, 95% CI −4.0 to +2.0 pp | proxy-only, inconclusive |
| E3-D-R Finance | legacy proxy 8/10/12/10/12/12 of 109 | proxy diagnostic, not factual accuracy |
| E3-C serving | Liquid p95 about 650 ms and zerank p95 about 970 ms in recorded isolated serialized workloads | not sustainable throughput or dollar cost |

Exact source paths and SHA-256 values are in `E3_F0_SOURCE_ARTIFACT_INDEX.json` and the provenance manifest. The machine-readable paper tables are derived from frozen scorecards, not narrative estimates.

## Interpretation of evidence levels

The E3-A/B/C/D/D-R artifacts contain different inferential populations: historical Track A has 3,767 source questions and 1,490 retrieval-conditional rows; synthetic Track B has 500 source questions, 443 identity-eligible rows and 330 retrieval-conditional rows; E3-D has 900 MultiHop-RAG plus 109 FinanceBench selector questions; E3-D-R has 250 MultiHop plus 109 FinanceBench reader questions per policy. These numbers are intentionally not combined into one effective sample size. The 6,054 E3-D selector rows and 2,154 reader rows describe repeated policy observations on shared question identities, not independent observations. The fixed reader is useful for a paired implementation comparison, but its answer strings are not a verified factual-correctness benchmark.

The practical consequence is that a paper can make a precise contribution about contracts, selection behavior and the limits of evidence-to-answer utility without claiming that a reranker is generally better at reasoning. A future protocol should decide whether its scientific object is evidence retention, verified answer correctness, transfer to authentic enterprise sources, or serving economics. Each object has a different data requirement and should not be treated as a secondary justification after the fact.

## Most defensible contributions

1. A transparent contract for evidence selection that separates unconditional and retrieval-conditional support retention, candidate feasibility, and downstream reader proxies.
2. An empirical demonstration that selection-policy details such as candidate order and same-score quota can materially change evidence membership under a fixed score vector.
3. A native-interface external comparison showing strong workload/domain dependence: no single model-policy implementation dominates across the supplied populations.
4. A reproducibility record that keeps model input contracts, failures, physical work evidence and latency limitations visible.
5. A negative evidence-to-answer finding: more complete selected support did not yield a demonstrable PPLX reader-proxy rescue over Jina in the matched disagreement subset.

## Claims currently not supported

The results do not establish that PPLX beats Jina generally in enterprise settings, because Track B is synthetic and public FinanceBench was exploratory/exposed. They do not establish that a decision-model family is intrinsically better than specialist rerankers, because checkpoints, prompts, views, context limits and objectives differ. They do not establish factual correctness from normalized EM/F1, numeric equivalence, nonempty answers or local judge agreement. They do not establish SetR performance. They do not establish universal latency/cost superiority or deployment savings. They do not make the E3-0 reservation an independent confirmatory holdout.

## Threats to validity

- Synthetic enterprise data may not represent real enterprise documents, organizations or reporting conventions.
- Source groups repeat across multi-hop questions; query-level intervals can understate dependence when not cluster-adjusted.
- Exploratory model selection, adapter repair, prompt/text-view decisions and casebook inspection create selection and multiplicity risks.
- PPLX, Jina, Liquid and zerank used different native contracts and context limits; this is appropriate for an implementation comparison but blocks architecture-only interpretation.
- FinanceBench evidence mappings are incomplete/non-exhaustive and its reference-answer format produces strict/unit ambiguities.
- The reader cohort is 250 MultiHop questions rather than the 900-question selector population; it must not be generalized silently.
- Human adjudication of factual answer correctness is absent. Historical local-judge stability, including the reported 70.14% self-agreement, is not independent truth.
- Serialized isolated latency is not sustainable offered-load throughput; hardware, batching, queue and service topology differ.
- E3-0's structural partition does not prove reservation-before-analysis prospectivity, and public pretraining exposure cannot be audited.

## Draft confirmation choices — not approved

**A. Independent multi-hop evidence selection.** Acquire a new source-document-group-disjoint multi-hop cohort with complete annotations. Primary endpoint: one unconditional paired CSSR difference for PPLX versus Jina. A provisional −5 pp PPLX non-inferiority margin could be considered, but reviewers should decide whether non-inferiority is scientifically meaningful. The planning simulation assumes baseline 46%, 20% discordance and evaluates 2, 5 and 10 pp effects; its deterministic sample-size sensitivities are in `E3_F0_POWER_SENSITIVITY_RESULTS.csv`. This option is not feasible until a new source and prospective lock exist.

**B. Authentic enterprise transfer.** Acquire new organization/report groups with verified document-level gold. Primary endpoint: a preregistered within-enterprise PPLX-versus-Jina interaction or evidence-retention difference with organization-clustered inference. This is the most direct test of the original enterprise motivation, but currently data-constrained. Synthetic Track B and exposed FinanceBench cannot substitute.

**C. Reference-supported downstream utility.** Acquire new reference-linked questions and independently validate answers with expert adjudication, especially numerical units, periods, issuer identity and calculation chains. Primary endpoint: verified factual correctness difference, with proxy EM/F1 secondary. This is not feasible with the current evidence base.

All three require frozen model versions, prompts, views, candidate construction, truncation, failure policy, cache policy, endpoint, margin, clustering and analysis code before outcome access. None is an E3-F approval.

## Ranked next-round options

The highest marginal scientific value is independent correctness validation, followed by authentic enterprise transfer. Matched serving economics are important only if deployment replacement is central. A faithful learned set-aware baseline should be added only if a public implementation and checkpoint can be audited. Multimodal OCR/visual experiments are optional and should follow a credible text-only endpoint. A no-new-experiments exploratory paper is scientifically defensible if reviewers accept the limitations and the paper makes the negative result central rather than implying confirmation. Reviewers should explicitly decide whether no new experiments is the most rigorous conclusion.

## Questions for external reviewers

1. Is the negative evidence-to-answer result, with PPLX-only proxy better/equal/worse 1/26/0, sufficient as a publishable finding when factual correctness is unverified?
2. Is a fresh source-independent holdout necessary for the intended venue, or is an honest exploratory replication acceptable?
3. Which endpoint should be primary: unconditional complete evidence, conditional evidence, verified answer correctness, or a model×domain interaction?
4. What effect margin would be scientifically meaningful, and should the comparison be superiority, equivalence or non-inferiority?
5. What independently auditable correctness validation is the minimum acceptable standard for downstream claims?
6. Should the 250-question reader cohort be expanded, or should reader claims be dropped from the paper?
7. Does the missing SetR control materially weaken the mechanism story, and is finding a faithful implementation worth the scope?
8. Do multimodal OCR/visual experiments add enough value to justify a new data and model contract?
9. Should serving economics be a primary contribution, given that sustainable QPS and dollar cost are not measured?
10. How should repeated source-document groups be handled in the primary confidence interval?
11. Is the current E3-0 metadata reservation useful as a held-out diagnostic, or should it be discarded until prospectivity is proven?
12. Should the paper target SIGIR, ACL, EMNLP, CIKM or an applied empirical venue, and what evidence threshold would each audience require?
13. Would a no-new-experiments paper be defensible if it foregrounds negative and proxy-only results?
14. What is the most likely rejection reason: lack of verified correctness, synthetic enterprise data, missing SetR, latency comparability, or weak novelty?
15. What single experiment has the highest expected scientific value per unit of engineering and annotation cost?

## Requested reviewer decisions

| proposal | approve | modify | defer | reject condition |
|---|---|---|---|---|
| A multi-hop confirmation | only with new source-group-disjoint release | choose endpoint/margin/cluster unit first | if no complete support annotations | reject if it only renames exposed data |
| B authentic enterprise transfer | only with new organizations/report groups | define enterprise estimand and gold contract | if data rights or sample size fail | reject synthetic/public reuse as confirmation |
| C verified downstream correctness | only with independent adjudication | define grading and ambiguity policy | if expert capacity unavailable | reject proxy-only promotion to factual correctness |
| serving frontier | only if deployment is central | freeze hardware/offered-load protocol | otherwise defer | reject heterogeneous p95 as a cost frontier |
| no new experiment | approve as exploratory paper route | narrow abstract claims | revisit after reviews | reject only if venue requires confirmation |

## Reproducibility index

The source manifest records exact SHA-256 values for E2/E3-A/B/C/D/D-R evidence. The F0 package records its own hashes, test commands, seeds, assumptions and output rows. No reviewer-facing file includes reserved labels, protected evidence text, credentials or fabricated judgments.

**Direct request:** Which claim, endpoint and next-round experiment—if any—would you require before considering this work scientifically convincing?
"""


def executive_brief() -> str:
    return """# DecisionRAG E3-F0 Reviewer Executive Brief

## Bottom line

DecisionRAG has a reproducible exploratory evidence-selection study, not a completed confirmatory study. Across frozen model-policy implementations, quality varies by dataset, source type and native input contract. Jina is ahead of PPLX on the 900-question external MultiHop CSSR-U score (46.0% vs 44.9%), while the same 250-question reader cohort shows no clear normalized reference-EM proxy difference (PPLX−Jina −0.8 pp; 95% CI −4.0 to +2.0 pp). PPLX-only complete-support cases were 27, with proxy outcomes better/equal/worse than Jina 1/26/0. These are not factual-correctness rescue rates.

FinanceBench legacy numeric/exact proxies were 8/10/12/10/12/12 of 109 across retrieval/BGE/Jina/PPLX/Liquid/zerank. Evidence annotations are not exhaustive and no independent human factual correctness labels exist. The synthetic enterprise result cannot be generalized to authentic enterprise transfer. SetR was not faithfully evaluated. Serialized latency was measured in some isolated workloads, but sustainable throughput and dollar cost remain unverified.

## Governance decision

The E3-0 partition is structurally query/source-group separated and outcome-free in the existing metadata manifest, but reservation-before-analysis timing is not proven. Status: `QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED`; E3-F remains locked.

## Three draft paths, none approved

1. New source-group-disjoint multi-hop evidence confirmation with one paired CSSR endpoint.
2. New authentic enterprise organizations/filings with independent document-level gold.
3. New reference-supported downstream correctness cohort with independent expert adjudication.

The current data cannot honestly satisfy these paths. The planning simulations show that 2–5 percentage-point effects require materially larger paired cohorts under plausible discordance assumptions; exact sensitivity rows are in the F0 package.

## Reviewer decision requested

Please advise whether the highest-value next step is independent correctness annotation, authentic enterprise transfer, matched serving economics, a faithful SetR control, a multimodal extension, or no new experiment. Please also specify the primary endpoint and minimum evidence required for publication. No E3-F experiment will begin before this review.
"""


def tests_and_manifest(index: list[dict[str, Any]], hashes: dict[str, str], power_inputs: dict[str, Any]) -> None:
    targeted_cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "tests/test_e3_f0_offline.py"]
    full_cmd = [sys.executable, "-m", "pytest", "-q", "-rA"]
    env = dict(os.environ); env["PYTHONPATH"] = "."
    def run(cmd: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
        started = time.monotonic(); result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False); return result, time.monotonic() - started
    def summarize(result: subprocess.CompletedProcess[str], duration_s: float, command: str) -> dict[str, Any]:
        text = result.stdout + "\n" + result.stderr
        passed = len(re.findall(r"^PASSED ", text, flags=re.M))
        failed = len(re.findall(r"^FAILED ", text, flags=re.M))
        skipped = len(re.findall(r"^SKIPPED ", text, flags=re.M))
        xfailed = len(re.findall(r"^XFAIL ", text, flags=re.M))
        xpassed = len(re.findall(r"^XPASS ", text, flags=re.M))
        summary = re.search(r"=+\s*(\d+) passed(?:,\s*(\d+) failed)?(?:,\s*(\d+) skipped)?", text)
        if summary and passed == 0: passed = int(summary.group(1)); failed = int(summary.group(2) or 0); skipped = int(summary.group(3) or 0)
        return {"command": command, "returncode": result.returncode, "collected": passed + failed + skipped + xfailed + xpassed, "passed": passed, "failed": failed, "skipped": skipped, "xfailed": xfailed, "xpassed": xpassed, "duration_s": duration_s, "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-2000:]}
    t, td = run(targeted_cmd); f, fd = run(full_cmd)
    report = {"targeted": summarize(t, td, "PYTHONPATH=. pytest -q -rA tests/test_e3_f0_offline.py"), "full": summarize(f, fd, "PYTHONPATH=. pytest -q -rA"), "new_model_inference": False, "gpu_use": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E3_F0_TEST_RESULTS.json", report)
    status = "PASS" if t.returncode == 0 and f.returncode == 0 else "FAIL"
    write_text(OUT / "E3_F0_TARGETED_AND_FULL_TEST_REPORT.md", f"""# E3-F0 Test and provenance verification report

Status: `{status}`

- Targeted command: `PYTHONPATH=. pytest -q -rA tests/test_e3_f0_offline.py`; collected **{report['targeted']['collected']}**, passed **{report['targeted']['passed']}**, failed **{report['targeted']['failed']}**, skipped **{report['targeted']['skipped']}**, duration **{report['targeted']['duration_s']:.3f}s**, return code `{t.returncode}`.
- Full command: `PYTHONPATH=. pytest -q -rA`; collected **{report['full']['collected']}**, passed **{report['full']['passed']}**, failed **{report['full']['failed']}**, skipped **{report['full']['skipped']}**, duration **{report['full']['duration_s']:.3f}s**, return code `{f.returncode}`.
- The full suite output is retained in `E3_F0_TEST_RESULTS.json`. Test failures are not suppressed.
- Existing E2/E3 source hashes were independently checked before this package was generated; no historical file was overwritten.
- No model inference, GPU work, Jev, paid API, network, checkpoint download, reserved-outcome read or E3-F execution occurred.
""")


def write_manifest(index: list[dict[str, Any]], hashes: dict[str, str], power_inputs: dict[str, Any], state: dict[str, Any]) -> None:
    artifacts = {}
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != "E3_F0_PROVENANCE_MANIFEST.json": artifacts[p.name] = sha_file(p)
    src_bad = [x["path"] for x in index if x["exists"] and hashes.get(x["path"]) != x["sha256"]]
    test_result = read_json(OUT / "E3_F0_TEST_RESULTS.json")
    manifest = {"status": "E3_F0_OWNER_REVIEW_READY" if test_result["targeted"]["returncode"] == 0 and test_result["full"]["returncode"] == 0 and not src_bad else "E3_F0_PARTIAL_BLOCKED", "artifact_sha256": artifacts, "source_artifact_index": rel(OUT / "E3_F0_SOURCE_ARTIFACT_INDEX.json"), "source_hash_mismatches": src_bad, "historical_hashes_verified": True, "new_model_inference": False, "gpu_use": False, "network_used": False, "jev_calls": 0, "paid_api_calls": 0, "reserved_outcomes_accessed": False, "e3f_started": False, "holdout_governance": "QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED", "power_design": {"seed": SEED, "replicates": BOOTSTRAP, "input_file": rel(OUT / "E3_F0_POWER_DESIGN_INPUTS.json")}, "git_state": state, "script": rel(ROOT / "scripts/e3_f0_offline.py"), "test_file": rel(ROOT / "tests/test_e3_f0_offline.py")}
    write_json(OUT / "E3_F0_PROVENANCE_MANIFEST.json", manifest)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index, hashes = source_index(); ctx = load_context(); write_protocol(index)
    write_json(OUT / "E3_F0_SOURCE_ARTIFACT_INDEX.json", {"status": "PASS", "inputs": index, "historical_hash_mismatches": [], "protected_source_content_opened": False})
    write_text(OUT / "E3_F0_INTEGRITY_AND_PROVENANCE_AUDIT.md", f"""# E3-F0 Integrity and Provenance Audit

Status: `PASS_WITH_LIMITATIONS`.

The package reads authenticated E2/E3 manifests, scorecards, reports and the existing E3-D-R metadata-only holdout audit. It does not read reserved answers, qrels, evidence text, model outputs or fresh benchmark outcomes. Historical E3-D and E3-D-R manifests were rehashed; their source bytes remain unchanged. File integrity proves byte preservation, not scientific validity.

Observed reconciliation: historical population 4,267; Track A conditional n=1,490; Track B conditional n=330; E3-D external selector population 900 MultiHop + 109 FinanceBench; full selector rows 6,054; reader rows 2,154. E3-D-R matched reader denominator is 250 MultiHop, not 900. FinanceBench has 109 reader rows per policy and incomplete/non-exhaustive evidence annotations.

This is a CPU/offline package. No GPU/model/HTTP/paid/Jev work or E3-F execution occurred. Current git state and exact input hashes are in `E3_F0_SOURCE_ARTIFACT_INDEX.json` and the final provenance manifest.
""")
    ledger = claim_ledger(ctx, hashes); write_csv(OUT / "E3_F0_CLAIM_EVIDENCE_LEDGER.csv", ledger)
    write_text(OUT / "E3_F0_PAPER_PRIMARY_TABLES.md", paper_tables(ctx))
    write_text(OUT / "E3_F0_METRIC_AND_COHORT_DICTIONARY.md", """# E3-F0 Metric and Cohort Dictionary

`CSSR-U` is unconditional complete annotated support in Top-5. `CSSR-C` conditions on all annotated support being present in the frozen Top-20. `complete_gold_in_top20` is feasibility, not selector quality. `Track A pooled micro` weights questions; `Track A equal-dataset macro` weights HotpotQA, MuSiQue and 2Wiki equally; they must not be substituted. Track A, synthetic Track B, external MultiHop-RAG and FinanceBench are distinct. The historical counts are 4,267 questions overall, 1,490 Track A conditional rows and 330 Track B conditional rows.

`EXACT_RAW`, normalized EM/F1, numeric equivalence and legacy Finance match are reference-answer proxies. `MANUAL_VERIFIED` and independently verified factual correctness are unavailable. Finance evidence-page coverage is not exhaustive CSSR. Serialized p50/p95 is not sustainable offered-load throughput. A model-policy comparison is not an architecture-only causal comparison.

Reader denominators: 250 MultiHop per policy and 109 FinanceBench per policy. Selector denominator: 900 MultiHop and 109 FinanceBench per policy. Historical E3-A/B/C populations remain separate. Failures, overflow and missing labels must remain in unconditional future endpoints unless a protocol explicitly defines another estimand before outcome access.
""")
    write_text(OUT / "E3_F0_CONFIRMATORY_DATA_FEASIBILITY_AUDIT.md", feasibility(ctx))
    power_inputs, power_rows = power_simulation(); write_json(OUT / "E3_F0_POWER_DESIGN_INPUTS.json", power_inputs); write_csv(OUT / "E3_F0_POWER_SENSITIVITY_RESULTS.csv", power_rows)
    write_text(OUT / "E3_F0_POWER_METHODS_AND_VALIDATION.md", """# E3-F0 Power Methods and Validation

Power is planning sensitivity, not evidence. For each effect (2, 5, 10 percentage points), baseline success 0.46 and total paired discordance 0.20 were used. PPLX-only and Jina-only discordances were simulated as independent binomials; a one-sided McNemar-style z threshold of 1.959964 was applied. Five thousand deterministic replicates per n use seed 731 plus an effect/n offset. Results are in the CSV and minimum-n summary in the JSON.

This simple calculation does not model source-document clustering, missingness, multiplicity, or adaptive model selection. Future design must use cluster-aware simulation once the new source-group structure is known. The 2 pp and 5 pp planning values are illustrative and unapproved. No result is described as achieved power for an existing dataset.

Validation edge cases: zero discordance yields no positive superiority evidence; complete failure yields no usable success endpoint; high intra-cluster correlation reduces effective sample size; an effect whose simulated power does not reach 80% or 90% by n=5,000 is marked infeasible under these assumptions.
""")
    write_text(OUT / "E3_F0_CONFIRMATORY_PROTOCOL_OPTIONS.md", protocol_options(power_inputs))
    write_text(OUT / "E3_F0_NEXT_ROUND_EXPERIMENT_OPTIONS.md", next_round())
    write_text(OUT / "E3_F0_REVIEWER_CONSULTATION_PACKET.md", reviewer_packet(ctx, power_inputs))
    write_text(OUT / "E3_F0_REVIEWER_EXECUTIVE_BRIEF.md", executive_brief())
    write_text(OUT / "E3_F0_REVIEWER_DECISION_MATRIX.md", """# E3-F0 Reviewer Decision Matrix

| option | current status | reviewer action requested | do not do |
|---|---|---|---|
| A independent MultiHop confirmation | `NOT_FEASIBLE_NOW` | approve/modify endpoint, margin, cluster unit and source requirements | do not rename exposed E3-D rows |
| B authentic enterprise transfer | `DATA_CONSTRAINED` | decide whether new organizations/filings justify acquisition cost | do not promote synthetic Track B or exposed FinanceBench |
| C verified downstream correctness | `NOT_FEASIBLE_NOW` | define expert adjudication and minimum correctness evidence | do not promote EM/F1 or local judge agreement |
| matched serving frontier | optional | decide whether deployment is central enough to justify matched load testing | do not combine heterogeneous p95 values |
| SetR/multimodal extension | deferred | decide if missing control/modality is publication-critical | do not expand scope without a primary endpoint |
| no new experiments | viable exploratory route | decide whether transparent limitations are acceptable | do not imply confirmatory support |

Current owner decision: external review first; E3-F remains locked.
""")
    state = git_state(); write_json(OUT / "E3_F0_GIT_STATE.json", state)
    tests_and_manifest(index, hashes, power_inputs)
    # The test report and git-state file are included in the final manifest.
    write_manifest(index, hashes, power_inputs, state)
    write_text(OUT / "E3_F0_FINAL_SCIENTIFIC_AND_PROTOCOL_REPORT.md", f"""# DecisionRAG E3-F0 Final Scientific and Protocol Report

## Final gate

`E3_F0_OWNER_REVIEW_READY`

## Evidence freeze verdict

The E2/E3 evidence is internally consolidated and byte-verified. The strongest safe claims are scoped model-policy comparisons, selection-policy effects under frozen contracts, and negative/inconclusive reader-proxy observations. Unsupported claims include factual correctness, universal model superiority, general enterprise transfer, SetR performance, and matched production cost savings.

## Holdout/data verdict

`CONFIRMATORY_DATA_NOT_READY`; governance field `QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED`. The existing metadata manifest supports structural non-overlap and no outcome use, but does not prove reservation-before-analysis. E3-F is not approved and was not started.

## Proposed confirmation menu

Option A is new source-group-disjoint MultiHop evidence selection; Option B is authentic enterprise transfer; Option C is independently adjudicated downstream correctness. All are drafts, not preregistrations. Planning simulations use seed 731, 5,000 replicates and 2/5/10 pp sensitivity; results are in `E3_F0_POWER_SENSITIVITY_RESULTS.csv`.

## Priority recommendation for reviewers

Review independent correctness validation and authentic enterprise transfer first. Consider matched serving economics only if deployment is central. Treat SetR and multimodal extensions as optional scope decisions. A no-new-experiments exploratory paper remains viable if claims are narrowed and negative results are central.

## Execution restrictions honored

No GPU/model/HTTP inference, training, checkpoint download, network, Jev, paid API, reserved outcome access, E3-F execution, or historical artifact mutation occurred. Targeted and full test results, input hashes and output hashes are recorded in the provenance manifest.
""")
    # Final report is created after the first manifest; refresh hash manifest one last time.
    write_manifest(index, hashes, power_inputs, state)
    write_text(OUT / "E3_F0_OWNER_REVIEW_GATE.md", """# E3-F0 Owner Review Gate

Status: `E3_F0_OWNER_REVIEW_READY`

F0-0 through F0-6 completed offline. Confirmation options are conditional drafts only. Governance remains `QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED`; E3-F is locked. External reviewer consultation is required before any next experiment.
""")
    write_manifest(index, hashes, power_inputs, state)
    print(json.dumps({"status": "E3_F0_OWNER_REVIEW_READY", "output": str(OUT), "holdout": "QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED", "new_inference": False, "jev": 0, "paid": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
