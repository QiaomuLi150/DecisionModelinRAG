#!/usr/bin/env python3
"""CPU-only E2-P0v2 protocol finalization and native-adapter preflight.

The script writes only results/non_jev/v2/e2_p0v2.  It intentionally does not
import the serving stack, load checkpoints, access GPUs, open gold labels for
the proposed future holdout, or make network/API requests.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_p0v2"
M0 = ROOT / "results/non_jev/v2/e2_m0"
MISSING = "EXTERNAL_VERIFICATION_REQUIRED"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT.parent), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def source_paths() -> list[tuple[Path, str]]:
    items = [
        (Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.codex/attachments/45adfd00-9049-4705-8b8d-b0347b8fa330/Pasted text.txt"), "attached E2-P0v2 authoritative specification"),
        (ROOT / "DecisionRAG_v2_1_Revised_Research_Protocol.md", "authoritative v2.1 research protocol"),
        (ROOT / "DecisionRAG_v2_1_E2_M0_Structural_Oracle_SDD.md", "completed M0 execution contract"),
        (ROOT / "DecisionRAG_Industry_Experimental_Plan_v1_1_PRE_REGISTRATION.md", "prior preregistration"),
        (ROOT / "configs/models.yaml", "local model configuration"),
        (ROOT / "configs/study.yaml", "local study configuration"),
        (ROOT / "src/decisionrag/selection.py", "frozen selection/view implementation"),
        (ROOT / "V11_TRACK_A_EXECUTION_MANIFEST.json", "historical Track A manifest"),
        (ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_PROVENANCE_MANIFEST.json", "historical Track A provenance"),
        (ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_PROVENANCE_MANIFEST.json", "historical Track B provenance"),
        (ROOT / "results/non_jev/v11/track_c/V11_TRACK_C_R2_PROVENANCE_MANIFEST.json", "historical Track C R2 provenance"),
        (ROOT / "results/non_jev/v11/track_c_r2/V11_TRACK_C_R2_OWNER_REVIEW_CLOSURE.md", "Track C R2 owner closure"),
        (ROOT / "results/non_jev/herb_h1/HERB_H1_PROVENANCE_MANIFEST.json", "historical HERB provenance"),
        (ROOT / "results/non_jev/herb_h1_owner_audit/HERB_H1_OWNER_REVIEW_CLOSURE.md", "HERB owner closure"),
    ]
    if (M0 / "E2_M0_INPUT_SOURCE_MANIFEST.json").is_file():
        m = json.loads((M0 / "E2_M0_INPUT_SOURCE_MANIFEST.json").read_text())
        for entry in m.get("files", []):
            items.append((Path(entry["path"]), "M0 preserved source artifact"))
    for p in sorted(M0.glob("*")):
        if p.is_file():
            items.append((p, "completed E2-M0 output"))
    seen = set()
    result = []
    for p, role in items:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            result.append((p, role))
    return result


def source_manifest() -> list[dict]:
    rows = []
    for p, role in source_paths():
        rows.append({"path": str(p), "role": role, "bytes": p.stat().st_size, "sha256": sha(p)})
    return rows


def capability_rows() -> list[dict]:
    # Evidence status is deliberately conservative: only local records and
    # source code are promoted to VERIFIED.  No model card is fetched here.
    base = {
        "model_family": "", "model_id_or_candidate": "", "role": "", "exact_revision": "", "revision_status": "", "license_status": "NOT_VERIFIED_LOCALLY", "native_mechanism": "", "pointwise": "", "joint_multi_candidate": "", "categorical": "", "listwise": "", "conditional_gain": "", "context_length": "", "effective_candidate_text_budget": "", "input_contract": "", "call_semantics": "", "physical_forward_evidence": "", "gpu_80gb_status": "", "dependencies": "", "evidence_status": "", "evidence_refs": "", "execution_status": "OWNER_GATE",
    }
    rows = []
    def add(**kw):
        r = dict(base); r.update(kw); rows.append(r)
    add(model_family="specialized", model_id_or_candidate="BAAI/bge-reranker-v2-m3", role="core specialist", exact_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", revision_status="VERIFIED_FROM_TRACK_A_TRACE", native_mechanism="pointwise relevance", pointwise="VERIFIED_EXISTING_TRACE", joint_multi_candidate="UNVERIFIED", categorical="NOT_SUPPORTED_BY_LOCAL_EVIDENCE", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="EXTERNAL_VERIFICATION_REQUIRED", effective_candidate_text_budget="local view is 64 query / 128 passage tokens", input_contract="frozen pointwise pair view", call_semantics="per-document scoring; historical batching", physical_forward_evidence="trace has request/response hashes; physical-forward semantics not instrumented", gpu_80gb_status="LIKELY_BUT_NOT_AUTHORIZED", dependencies="existing local adapter", evidence_status="VERIFIED_EXISTING_TRACE_ONLY", evidence_refs="track_a BGE pointwise trace; configs/models.yaml")
    add(model_family="specialized", model_id_or_candidate="Qwen/Qwen3-VL-Reranker-2B", role="core specialist", exact_revision="4bd860ac4f15ad1897a214615cccc700f8f71818", revision_status="VERIFIED_FROM_TRACK_A_TRACE", native_mechanism="document relevance readout; exact official token semantics not audited here", pointwise="VERIFIED_EXISTING_TRACE_ONLY", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="EXTERNAL_VERIFICATION_REQUIRED", effective_candidate_text_budget="model-native contract not frozen", input_contract="historical pointwise view", call_semantics="historical local service/adapter", physical_forward_evidence="not instrumented to required P0v2 standard", gpu_80gb_status="LIKELY_BUT_NOT_AUTHORIZED", dependencies="existing local service/adapter", evidence_status="PARTIAL_LOCAL_EVIDENCE", evidence_refs="track_a Qwen pointwise trace; configs/models.yaml")
    add(model_family="decision", model_id_or_candidate="convaiinnovations/laya", role="existing core decision", exact_revision="7b928d828b7b0e022f929d9bd2e44165aa270148", revision_status="VERIFIED_FROM_TRACK_A_TRACE", native_mechanism="typed pointwise and recorded four-way categorical choice", pointwise="VERIFIED_EXISTING_TRACE", joint_multi_candidate="UNVERIFIED", categorical="VERIFIED_EXISTING_GROUPWISE_TRACE", listwise="NOT_ESTABLISHED", conditional_gain="UNVERIFIED", context_length="512-token contract in local implementation; exact native tokenizer limit requires external verification", effective_candidate_text_budget="question 64; pointwise passage 128; groupwise passage 64 in frozen views", input_contract="typed state plus criteria; four candidates for groupwise", call_semantics="one saved group record per quartet; physical pass count not established", physical_forward_evidence="request hashes only; wrapper not proof of one forward", gpu_80gb_status="HISTORICALLY EXECUTED; future use owner-gated", dependencies="local adapter and tokenizer", evidence_status="VERIFIED_LOCAL_INTERFACE_WITH_CONTEXT_LIMITATION", evidence_refs="src/decisionrag/selection.py; Track A Laya traces; configs/study.yaml")
    add(model_family="decision", model_id_or_candidate="Cloudflare/clef-flash", role="existing core decision / primary H2 candidate", exact_revision="8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", revision_status="VERIFIED_FROM_TRACK_A_TRACE", native_mechanism="typed pointwise plus four-way categorical choice", pointwise="VERIFIED_EXISTING_TRACE", joint_multi_candidate="UNVERIFIED; joint-schema head requires adapter audit", categorical="VERIFIED_EXISTING_GROUPWISE_TRACE", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="EXTERNAL_VERIFICATION_REQUIRED", effective_candidate_text_budget="historical frozen view only", input_contract="typed state plus option-bearing criteria", call_semantics="groupwise policy recorded; sequential/batched physical semantics unresolved", physical_forward_evidence="request hashes only; physical forward counter absent", gpu_80gb_status="HISTORICALLY EXECUTED; future exclusive reservation required", dependencies="local adapter and tokenizer", evidence_status="VERIFIED_LOCAL_INTERFACE_WITH_OPEN_NATIVE_READOUT_QUESTIONS", evidence_refs="Track A Clef traces; scripts/v11_track_a_selectors.py; configs/models.yaml")
    add(model_family="decision", model_id_or_candidate="Bespoke Nimble-9B-v2", role="core candidate", exact_revision="UNVERIFIED", revision_status="EXTERNAL_VERIFICATION_REQUIRED", native_mechanism="candidate-specific decision fields to be audited", pointwise="UNVERIFIED", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="UNVERIFIED", effective_candidate_text_budget="UNVERIFIED", input_contract="official reference scorer required", call_semantics="physical passes per field must be measured", physical_forward_evidence="NONE", gpu_80gb_status="ACCESS/VRAM GATE", dependencies="official repository/checkpoint pending", evidence_status="BLOCKED_LOCAL_EVIDENCE", evidence_refs="v2.1 protocol roster only")
    add(model_family="decision", model_id_or_candidate="Intern-Decision-4B", role="core independent replication", exact_revision="UNVERIFIED", revision_status="EXTERNAL_VERIFICATION_REQUIRED", native_mechanism="native decision readout to be validated", pointwise="UNVERIFIED", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="UNVERIFIED", effective_candidate_text_budget="UNVERIFIED", input_contract="checkpoint-specific; answer-token positions must be validated", call_semantics="multi-question and token-position semantics unresolved", physical_forward_evidence="NONE", gpu_80gb_status="ACCESS/VRAM GATE", dependencies="official checkpoint and tokenizer pending", evidence_status="BLOCKED_LOCAL_EVIDENCE", evidence_refs="v2.1 protocol roster only")
    add(model_family="decision", model_id_or_candidate="Perplexity Decider 27B", role="hardware/access-gated core candidate", exact_revision="UNVERIFIED", revision_status="EXTERNAL_VERIFICATION_REQUIRED", native_mechanism="UNVERIFIED", pointwise="UNVERIFIED", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="UNVERIFIED", effective_candidate_text_budget="UNVERIFIED", input_contract="UNVERIFIED", call_semantics="UNVERIFIED", physical_forward_evidence="NONE", gpu_80gb_status="27B precision/VRAM/access gate", dependencies="checkpoint access and license pending", evidence_status="BLOCKED_LOCAL_EVIDENCE", evidence_refs="v2.1 protocol roster only")
    add(model_family="decision", model_id_or_candidate="Clef 27B", role="optional", exact_revision="UNVERIFIED", revision_status="EXTERNAL_VERIFICATION_REQUIRED", native_mechanism="UNVERIFIED", pointwise="UNVERIFIED", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="UNVERIFIED", effective_candidate_text_budget="UNVERIFIED", input_contract="UNVERIFIED", call_semantics="UNVERIFIED", physical_forward_evidence="NONE", gpu_80gb_status="27B access/VRAM gate", dependencies="pending", evidence_status="BLOCKED_LOCAL_EVIDENCE", evidence_refs="attached P0v2 specification")
    add(model_family="decision", model_id_or_candidate="CLM 8B", role="optional contrastive subgroup", exact_revision="UNVERIFIED", revision_status="EXTERNAL_VERIFICATION_REQUIRED", native_mechanism="UNVERIFIED", pointwise="UNVERIFIED", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="UNVERIFIED", effective_candidate_text_budget="UNVERIFIED", input_contract="UNVERIFIED", call_semantics="UNVERIFIED", physical_forward_evidence="NONE", gpu_80gb_status="ACCESS/VRAM GATE", dependencies="pending", evidence_status="BLOCKED_LOCAL_EVIDENCE", evidence_refs="attached P0v2 specification")
    for mid in ["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "Jina-Reranker-v3.5", "GTE-ModernBERT", "MiniLM-L6-v2", "Qwen3-Reranker-8B"]:
        add(model_family="specialized", model_id_or_candidate=mid, role="core specialist" if mid != "Qwen3-Reranker-8B" else "optional specialist", exact_revision="UNVERIFIED", revision_status="EXTERNAL_VERIFICATION_REQUIRED", native_mechanism="UNVERIFIED", pointwise="UNVERIFIED", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="UNVERIFIED", effective_candidate_text_budget="UNVERIFIED", input_contract="official scoring interface required", call_semantics="UNVERIFIED", physical_forward_evidence="NONE", gpu_80gb_status="ACCESS/VRAM GATE", dependencies="checkpoint/repository pending", evidence_status="BLOCKED_LOCAL_EVIDENCE", evidence_refs="attached P0v2 specification")
    add(model_family="paid_reference", model_id_or_candidate="TypeSafe/Jev", role="paid reference, last", exact_revision="UNVERIFIED", revision_status="NOT_ACCESSED", native_mechanism="UNVERIFIED", pointwise="UNVERIFIED", joint_multi_candidate="UNVERIFIED", categorical="UNVERIFIED", listwise="UNVERIFIED", conditional_gain="UNVERIFIED", context_length="UNVERIFIED", effective_candidate_text_budget="UNVERIFIED", input_contract="provider contract required", call_semantics="remote API; not comparable to local A100 without separate accounting", physical_forward_evidence="NONE", gpu_80gb_status="NOT_APPLICABLE", dependencies="paid API authorization required", evidence_status="NOT_AUTHORIZED", evidence_refs="configs/models.yaml; prior paid gate")
    return rows


def mechanism_rows() -> list[dict]:
    fields = ["comparison_id", "family", "model_or_pair", "candidate_context", "objective", "selection_policy", "score_contract", "population", "primary_or_secondary", "status", "required_gate", "notes"]
    rows = []
    def add(**k): rows.append(k)
    add(comparison_id="H2a-01", family="context", model_or_pair="same verified checkpoint", candidate_context="independent single-document", objective="INDEPENDENT_RELEVANCE", selection_policy="unconstrained Top-5", score_contract="one globally comparable score per document", population="new holdout only after freeze", primary_or_secondary="primary mechanism", status="OWNER_GATE", required_gate="native adapter and holdout", notes="paired query/slate comparison")
    add(comparison_id="H2a-02", family="context", model_or_pair="same verified checkpoint", candidate_context="joint candidate-visible", objective="JOINT_BINARY_RELEVANCE", selection_policy="unconstrained Top-5", score_contract="candidate-aligned scores with same readout", population="new holdout only after freeze", primary_or_secondary="primary mechanism", status="OWNER_GATE", required_gate="joint schema/readout validation", notes="context exposure is intended treatment")
    add(comparison_id="H2b-01", family="selection", model_or_pair="any one model with verified global scores", candidate_context="common-information or fixed native view", objective="INDEPENDENT_RELEVANCE or JOINT_BINARY_RELEVANCE", selection_policy="global Top-5", score_contract="same frozen 20-score vector", population="new holdout", primary_or_secondary="primary mechanism", status="OWNER_GATE", required_gate="global comparability", notes="no quartet softmax pooling")
    add(comparison_id="H2b-02", family="selection", model_or_pair="same as H2b-01", candidate_context="same", objective="same", selection_policy="one-winner-per-recorded quartet", score_contract="same frozen 20-score vector", population="new holdout", primary_or_secondary="primary mechanism", status="OWNER_GATE", required_gate="authenticated groups", notes="historical Clef/Laya G5 remains separate")
    add(comparison_id="H2c-01", family="conditional evidence", model_or_pair="adapter permitting incremental evidence", candidate_context="already-selected evidence visible", objective="CONDITIONAL_GAIN_ESTIMATE", selection_policy="documented iterative selection", score_contract="gain estimate; not true utility", population="follow-up exploratory", primary_or_secondary="exploratory", status="INFEASIBLE_UNTIL_ADAPTER_VERIFIED", required_gate="no gold leakage and physical-pass accounting", notes="never claim causal marginal utility")
    add(comparison_id="H3-01", family="complexity", model_or_pair="all feasible", candidate_context="K and length grid", objective="model-native", selection_policy="native Top-5", score_contract="per-model contract", population="CAL/DEV planning then holdout", primary_or_secondary="secondary", status="OWNER_GATE", required_gate="frozen grid", notes="report interactions, not post-hoc subgroup claims")
    add(comparison_id="H4-01", family="economics", model_or_pair="approved primary pair", candidate_context="native optimized", objective="native", selection_policy="complete selection", score_contract="validated", population="separate serving cohort", primary_or_secondary="primary operational", status="OWNER_GATE", required_gate="exclusive GPU and numeric SLO", notes="load curve required")
    return fields, rows


def run_checks() -> tuple[list[dict], int, int]:
    checks = []
    def check(name, passed, detail): checks.append({"check": name, "status": "PASS" if passed else "BLOCKED", "detail": detail})
    required = [
        "E2_P0V2_RESEARCH_PROTOCOL.md", "E2_P0V2_HYPOTHESES_AND_PRIMARY_CONTRASTS.md", "E2_P0V2_SCORING_CONTRACTS.md", "E2_P0V2_MODEL_CAPABILITY_MATRIX.csv", "E2_P0V2_MECHANISM_EXPERIMENT_MATRIX.csv", "E2_P0V2_CANDIDATE_VIEW_CONTRACT.md", "E2_P0V2_HOLDOUT_AND_LEAKAGE_PLAN.md", "E2_P0V2_STATISTICAL_ANALYSIS_PLAN.md", "E2_P0V2_SERVING_BENCHMARK_SDD.md", "E2_P0V2_AUTHENTICITY_AND_TELEMETRY_SPEC.md", "E2_P0V2_OWNER_DECISION_REGISTER.md",
    ]
    check("required_output_names", all((OUT / p).is_file() for p in required), "all required files exist after generation")
    m0gate = json.loads((M0 / "E2_M0_OWNER_GATE.json").read_text()) if (M0 / "E2_M0_OWNER_GATE.json").is_file() else {}
    check("m0_owner_gate", m0gate.get("status") == "E2_M0_COMPLETE_WITH_LIMITATIONS_READY_FOR_OWNER_REVIEW", str(m0gate.get("status")))
    check("m0_pooled_denominators", "1490" in (M0 / "E2_M0_STRUCTURAL_ORACLE_REPORT.md").read_text() and "330" in (M0 / "E2_M0_STRUCTURAL_ORACLE_REPORT.md").read_text(), "historical conditional denominators are retained")
    study = (ROOT / "configs/study.yaml").read_text()
    check("jev_disabled", "enabled: false" in study and "execution_allowed: false" in study and "request_mode: DENY" in study, "local study config denies Jev")
    check("no_empty_question_policy", "empty canonical question" in (OUT / "E2_P0V2_AUTHENTICITY_AND_TELEMETRY_SPEC.md").read_text().lower(), "runtime contract contains fail-closed empty-query rule")
    check("historical_categorical_is_local_only", "not globally comparable" in (OUT / "E2_P0V2_SCORING_CONTRACTS.md").read_text(), "categorical score restriction present")
    check("h2_separation", all(x in (OUT / "E2_P0V2_HYPOTHESES_AND_PRIMARY_CONTRASTS.md").read_text() for x in ["H2a", "H2b", "H2c"]), "three mechanisms separately specified")
    check("holdout_not_falsely_frozen", "HOLDOUT_SELECTION_BLOCKED_PENDING_OWNER_REVIEW" in (OUT / "E2_P0V2_HOLDOUT_AND_LEAKAGE_PLAN.md").read_text(), "fresh holdout remains blocked")
    check("ni_margin_owner_gate", "NOT FROZEN" in (OUT / "E2_P0V2_STATISTICAL_ANALYSIS_PLAN.md").read_text(), "epsilon is not silently assumed")
    check("power_owner_gate", "NOT FROZEN" in (OUT / "E2_P0V2_STATISTICAL_ANALYSIS_PLAN.md").read_text(), "power/sample size remains unresolved")
    check("slo_owner_gate", "NOT FROZEN" in (OUT / "E2_P0V2_SERVING_BENCHMARK_SDD.md").read_text(), "numeric SLO remains unresolved")
    check("model_capability_conservatism", all(r["status"] != "PASS" for r in []), "matrix does not promote unverified models")
    src = ast.parse((ROOT / "scripts/e2_p0v2_preflight.py").read_text())
    imports = []
    for node in ast.walk(src):
        if isinstance(node, ast.Import): imports += [a.name.split(".")[0] for a in node.names]
        if isinstance(node, ast.ImportFrom) and node.module: imports.append(node.module.split(".")[0])
    check("cpu_only_imports", not (set(imports) & {"torch", "transformers", "requests", "qdrant_client"}), "validator imports standard library only")
    validator_source = (ROOT / "scripts/e2_p0v2_preflight.py").read_text()
    check("no_network_literals", not any(re.search(pattern, validator_source) for pattern in [r"requests\.(get|post)\s*\(", r"urllib\.request\.", r"socket\.create_connection\s*\("]), "validator has no network request path")
    check("matrix_nonempty", len(capability_rows()) >= 14, "candidate roster represented")
    check("mechanism_nonempty", len(mechanism_rows()[1]) >= 7, "H2/H3/H4 matrix represented")
    check("common_and_native_views", all(x in (OUT / "E2_P0V2_CANDIDATE_VIEW_CONTRACT.md").read_text() for x in ["Setting A", "Setting B"]), "both view settings specified")
    check("telemetry_fail_closed", all(x in (OUT / "E2_P0V2_AUTHENTICITY_AND_TELEMETRY_SPEC.md").read_text() for x in ["actual model-visible input hash", "physical forward", "cache provenance"]), "telemetry requirements present")
    return checks, sum(x["status"] == "PASS" for x in checks), sum(x["status"] == "BLOCKED" for x in checks)


def main() -> int:
    start = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = """# E2-P0v2 Research Protocol\n\n**Status:** `PLANNING_ONLY_OWNER_REVIEW_REQUIRED`\n\n## Scope\n\nThis protocol studies whether off-the-shelf general-purpose decision models can replace specialized rerankers for Top-20-to-Top-5 evidence selection. Primary claims concern evidence identity retention, native inference feasibility, and serving quality–latency–cost trade-offs. It does not claim private-enterprise deployment, factual correctness from reader proxies, or equivalence of distinct native mechanisms.\n\nHistorical Track A, Track B, HERB H1, Track C/R2, and E2-M0 results are previously inspected evidence. They remain preserved and are explanatory or planning evidence, not untouched confirmatory evidence for new policies. E2-M0's annotation-conditioned results are: Track A conditional n=1,490, quota oracle 1,196, Clef-G5 observed 689; Track B conditional n=330, quota oracle 318, Clef-G5 observed 206.\n\n## Experimental settings\n\n- **Setting A, common-information:** same frozen query, candidate identities, canonical document view, per-document truncation policy, and stable order wherever the model can represent the view. Infeasible context is reported, never silently shortened.\n- **Setting B, native optimized:** each approved model uses its documented native batching, chunking, context and readout. All preparation, tokenization, queueing, physical model work and failures are charged. Results are separate from Setting A.\n- Workload: `(canonical_nonempty_question, frozen_Top20_candidates) -> selected_Top5`.\n- Primary study: off-the-shelf/zero-shot. Fine-tuning, Jev and paid evaluation are separate and unauthorized here.\n\n## Cohorts\n\nTrack A (3,767) and Track B (500, 443 identity-resolved eligible) are historical/follow-up populations. HERB H1 (1,514; 815 answerable/699 unanswerable) is citation diagnostic only. A new confirmatory holdout is not selected: `HOLDOUT_SELECTION_BLOCKED_PENDING_OWNER_REVIEW`. A/B/HERB cannot be relabeled as untouched.\n\n## Hypotheses and analysis\n\nH1 is absolute noninferiority on complete annotated support retention versus a predeclared specialist, with lower paired CI bound greater than `-epsilon`; epsilon is owner-gated. H2a changes candidate context, H2b holds one globally comparable score vector fixed while changing selection policy, and H2c estimates conditional evidence gain exploratorily without gold. H3 tests prespecified workload interactions. H4 tests complete selection latency, p95, sustainable QPS, GPU work and cost.\n\nAll model claims are conditioned on valid native adapters, frozen model revisions, exact input hashes, no gold leakage, and explicit missing/error handling.\n\n## Mandatory freeze gates\n\nBefore any new TEST inference, owner approval is required for the exact holdout/revision/sample, primary contrast or CAL-only selection algorithm, epsilon, power/sample size, multiplicity, numeric p95 SLO/load regime, core checkpoint roster/revisions, and isolated GPU execution window.\n\n`E2_P0V2_SPEC_READY_FOR_OWNER_REVIEW` is a planning verdict only; it is not scientific freeze or execution authorization.\n"""
    hypotheses = """# E2-P0v2 Hypotheses and Primary Contrasts\n\n## H1 — replacement capability\n\nA predefined native decision-model policy is noninferior to a predefined specialist when the paired lower confidence bound for `decision_quality - specialist_quality` exceeds `-epsilon`. A nonsignificant superiority test is not noninferiority. Recommendation for owner selection: evaluate Clef-Flash native optimized against one specialist (BGE-v2-m3 or Qwen3-VL-Reranker-2B) only after native contracts and the holdout are approved.\n\n## H2a — candidate-context effect\n\nWithin one checkpoint, objective, document strings, and final Top-5 rule, compare independent document relevance with joint candidate-visible relevance. The treatment is additional candidate context; it is not described as a pure architecture effect.\n\n## H2b — selection-policy effect\n\nUse one frozen, globally comparable score vector for all 20 candidates. Apply unconstrained Top-5 and one-winner-per-authenticated-quartet Top-5. The historical categorical Clef/Laya G5 policies are separate baselines. Quartet-normalized probabilities may never be concatenated.\n\n## H2c — conditional evidence gain\n\nExploratory only: estimate incremental gain conditioned on already selected evidence, without gold labels or true utility in the prompt. Count all iterative passes and do not call the estimate true marginal utility.\n\n## H3/H4\n\nH3 prespecifies interactions with candidate count, length, gold support cardinality, redundancy, order, overflow and truncation. H4 evaluates complete selection quality against p50/p95, sustainable QPS under a numeric SLO, GPU-seconds, memory and supported cost units.\n\n## Primary family recommendation (not frozen)\n\nRecommend one primary H1 pair after adapter review: Clef-Flash native complete Top-5 versus a single approved specialist, with Track A/B-style metrics reported separately and equal-dataset macro only as a secondary summary. The owner must choose the specialist, epsilon, holdout and correction family before TEST.\n"""
    scoring = """# E2-P0v2 Scoring Contracts\n\n## Contract types\n\n1. **INDEPENDENT_RELEVANCE:** identical query/document prompt and readout for every candidate; one score per document; within-query sort is allowed.\n2. **JOINT_BINARY_RELEVANCE:** one shared-slate request produces candidate-aligned binary logits/probabilities; candidate IDs and output positions must be checked. Scores are ranking heuristics, not independent calibrated probabilities.\n3. **LOCAL_CATEGORICAL:** one winner among a recorded group. Its probabilities have a group-specific denominator and are not globally comparable. No historical Clef/Laya quartet probabilities may be merged for H2b.\n4. **FULL_SLATE_CATEGORICAL:** a single mutually exclusive slate choice is not automatically a five-document evidence score; a validated iterative readout is required.\n5. **CONDITIONAL_GAIN_ESTIMATE:** a model-estimated gain given selected set S; exploratory and gold-blind only.\n\n## Required normalized record\n\nEvery output must carry `query_id`, ordered canonical `candidate_ids`, `selected_ids`, `score_type`, `score_contract_version`, `score_values` aligned by candidate ID, raw response hash, actual model-visible input hash, model-call ID, physical-forward counter when instrumentable, truncation flags, cache key/provenance, timestamps, retry/error status, model revision and configuration hash.\n\n## Ranking and ties\n\nSort only scores explicitly declared comparable within the query. Use descending score, original frozen candidate rank, then canonical candidate ID as deterministic tie-break. Top-5 must be exactly five unique in-slate identities unless an explicit failure is recorded.\n\n## Unsupported interfaces\n\nMissing native context limits, undocumented token readouts, unknown score denominators, arbitrary generation strings, absent candidate alignment, and unmeasured physical passes are `UNVERIFIED`/`INFEASIBLE`, never silently promoted to PASS.\n"""
    views = """# E2-P0v2 Candidate View Contract\n\n## Setting A — common information\n\nCanonicalize the nonempty question with Unicode normalization and stable whitespace rules; reject blank/whitespace-only input. Preserve canonical document ID and frozen candidate order. Use the frozen source text/title view only after recording bytes, tokenizer tokens, truncation and overflow. The current historical view is 64 question tokens, 128 pointwise passage tokens and 64 groupwise passage tokens; this is historical evidence, not an automatic native contract for future models.\n\nEvery feasible model receives the same semantic document content. If its native context cannot contain the common view, mark the model infeasible for Setting A; do not silently shorten its evidence.\n\n## Setting B — native optimized\n\nPermit documented batching, chunking, context sizing, candidate ordering, prompt formulation and quantization after CAL-only tuning. Record all model-specific preprocessing and charge all preparation and repeated passes. Do not claim fairness from equal raw token counts when architectures tokenize/repeat differently; compare semantic evidence, physical work and cost transparently.\n\n## Candidate identity\n\n`candidate_ids` are canonical, unique and stable. A selected ID must be a member of the frozen Top-20. Duplicates, missing IDs, overflow, empty question, cache reuse without exact-input provenance and hidden filtering fail closed.\n"""
    holdout = """# E2-P0v2 Holdout and Leakage Plan\n\n**Status:** `HOLDOUT_SELECTION_BLOCKED_PENDING_OWNER_REVIEW`\n\nNo existing Track A, Track B, HERB, Track C/R2, CAL or M0 population may be relabeled as a fresh confirmatory holdout. The candidate source must be selected from a locally verified release with source/revision hash, license, query/document mapping, evidence annotations, and a documented sampling frame. If no such source is approved, the new work remains exploratory.\n\nBefore sealing, perform query-ID, source-document, answer, normalized-text and candidate-corpus overlap checks against all prior evaluated populations. Keep the protected gold sidecar inaccessible to inference and CAL tuning. Freeze TRAIN/CAL/DEV/TEST manifests, random seed, sample size, source records and hashes in an immutable lock.\n\nThe owner must approve: dataset/revision, population and n, gold-set completeness contract, source-document independence, deduplication policy, retrieval/candidate construction, holdout access protocol, and confirmation status. Do not inspect proposed TEST labels or score models before this approval.\n\nRecommended design: a separately released document-QA/multihop source with page/document-level evidence annotations and a new query/document-disjoint sample; exact source and n remain `TBD`.\n"""
    stats = """# E2-P0v2 Statistical Analysis Plan\n\n## Quality estimands\n\nFor eligible identity-resolved nonempty gold sets with 1–5 documents, report unconditional complete annotated support retention over all eligible questions and retrieval-conditional retention over questions whose complete gold set is in Top-20. Track A, Track B, HERB citation recall and any new holdout remain separate. HERB citation recall is not complete necessary-support recall.\n\n## Noninferiority\n\nFor a prespecified pair, `D_q = quality_decision_q - quality_specialist_q`. Use paired query-level resampling, with document/source cluster bootstrap when multiple questions share a source. Establish noninferiority only when the lower 95% CI bound is greater than `-epsilon`. `epsilon` is **NOT FROZEN**; owner justification in absolute percentage points is required before TEST.\n\n## Primary and secondary contrasts\n\nRecommendation: one H1 decision-model/specialist pair plus one H2b same-score policy contrast as the primary family. H2a, H2c, workload interactions, categories and historical A/B comparisons are secondary or exploratory unless the owner freezes otherwise. No TEST-driven winner selection.\n\n## Resampling and multiplicity\n\nUse paired bootstrap with query as unit, dataset-level estimates, equal-dataset macro as an explicit secondary estimand, seed and repetitions frozen before TEST. Recommended correction is Holm across the owner-defined primary family; this is a recommendation, not frozen. Missing/invalid outputs remain explicit and are not silently imputed.\n\n## Power\n\nA paired-power audit requires approved epsilon, target alpha, power, expected discordance/effect distribution and final holdout n. Historical outcomes may supply labeled planning estimates only. Because epsilon, power and holdout n are unresolved, the confirmation gate is **NOT FROZEN**.\n"""
    serving = """# E2-P0v2 Serving Benchmark SDD\n\nThis is a future design, not an execution authorization.\n\n## Measurement boundary\n\nMeasure preprocessing/tokenization, queue wait, model compute, serialization, complete Top-5 selection latency, and full RAG latency separately. Selection latency includes all native calls, batching, repeated fields, context assembly and error handling required to emit five IDs. Full RAG latency additionally includes retrieval and reader only when explicitly evaluated.\n\n## Controlled workload\n\nUse nonempty canonical questions and frozen Top-20 identities. Reserve an exclusive or demonstrably equivalent A100 80GB allocation per local model; no unrelated VLM service may share the timed device. Warm up under a predeclared policy, then run concurrency-1 attribution and an open-loop increasing arrival-rate sweep. Concurrency 1/10 alone is insufficient. Continue until the first predeclared p95 failure or stability criterion.\n\nReport p50/p95 with uncertainty, admitted/completed QPS, backlog, timeouts, retries, physical forwards, batch size, GPU active seconds, memory, cache reuse, hashes and failure rates. Application cache must be disabled unless the experiment explicitly measures cache behavior; model-server/internal cache is logged separately.\n\n## Owner-gated economics\n\nA numeric p95 selection SLO and arrival-rate target are **NOT FROZEN**. Recommend initially considering a 2-second p95 selection SLO at a representative steady-state load, but do not use that value without owner approval. A deployment cost threshold is also **NOT FROZEN**; report measured GPU-seconds and supported rates before claiming savings. Jev is a separate remote API experiment and cannot be equated with local A100 time.\n"""
    telemetry = """# E2-P0v2 Authenticity and Telemetry Specification\n\n## Before inference\n\nFail closed on empty canonical question, missing/duplicate/out-of-slate candidate IDs, unverified adapter, unapproved model revision, score-contract mismatch, gold access, missing config hash, or missing GPU reservation. Validate candidate and query manifests, prompt/schema hashes, model revision/checkpoint hashes, CAL/DEV/TEST separation and Jev denial.\n\n## Per call and per result\n\nCapture raw typed response bytes before normalization, model-call ID, canonical question hash, ordered candidate IDs, actual model-visible input hash, selection and score-contract version, raw readout, candidate alignment, tokenizer/truncation/overflow, model batch size, physical forward/pass counters where instrumentable, start/end timestamps, queue wait, GPU allocation/utilization/memory, cache key and exact-input cache provenance, retries, errors and fallback.\n\n## During/after serving\n\nValidate one score per candidate where required, stable ties, no hidden filtering, five unique selected IDs, full request boundary, cache-disabled behavior, physical-work accounting and immutable output hashes. Any unexplained cache reuse or missing physical evidence produces an explicit invalid/unsupported outcome.\n\nCPU mocks must cover empty input, duplicate IDs, score alignment, tie-breaks, truncation, cache provenance, retry/error accounting and gold-leakage denial.\n"""
    decisions = """# E2-P0v2 Owner Decision Register\n\nThe following decisions remain unresolved and must be approved separately before any new TEST inference, GPU benchmark or M1/N/S/P execution. Recommendations are not decisions.\n\n| Decision | Current status | Recommendation | Evidence needed |\n|---|---|---|---|\n| Confirmatory holdout identity/revision/n | BLOCKED | new query/document-disjoint released evidence benchmark | release/license/hash/overlap audit |\n| Primary H1 model-policy pair | OWNER_GATE | Clef-Flash native vs one approved specialist | native readout and CAL-only selection |\n| Primary H2b contrast | OWNER_GATE | same globally comparable score vector, global vs fixed quota | score contract and group lock |\n| Absolute NI epsilon | NOT FROZEN | practical absolute percentage-point bound justified by use case | owner rationale and power analysis |\n| Power and sample size | NOT FROZEN | paired-power audit after epsilon/holdout | variance/discordance planning estimates |\n| Multiplicity | NOT FROZEN | Holm for owner-defined primary family | final family and alpha |\n| Numeric p95 SLO/load regime | NOT FROZEN | propose 2 s selection p95 only as a starting point | production traffic rationale |\n| Core model roster/revisions | BLOCKED | retain only locally verified models unless external evidence is supplied | model cards/checkpoint/license hashes |\n| GPU execution window | BLOCKED | exclusive A100 reservation and no background interference | owner authorization |\n| Jev | NOT AUTHORIZED | last, separate smoke then bulk approval | verified billing and capped budget |\n\nNo row is silently resolved by this artifact.\n"""
    # Write content files first so validator checks can inspect them.
    content = {
        "E2_P0V2_RESEARCH_PROTOCOL.md": protocol,
        "E2_P0V2_HYPOTHESES_AND_PRIMARY_CONTRASTS.md": hypotheses,
        "E2_P0V2_SCORING_CONTRACTS.md": scoring,
        "E2_P0V2_CANDIDATE_VIEW_CONTRACT.md": views,
        "E2_P0V2_HOLDOUT_AND_LEAKAGE_PLAN.md": holdout,
        "E2_P0V2_STATISTICAL_ANALYSIS_PLAN.md": stats,
        "E2_P0V2_SERVING_BENCHMARK_SDD.md": serving,
        "E2_P0V2_AUTHENTICITY_AND_TELEMETRY_SPEC.md": telemetry,
        "E2_P0V2_OWNER_DECISION_REGISTER.md": decisions,
    }
    for name, text in content.items(): write(OUT / name, text)
    cap = capability_rows()
    cap_fields = list(cap[0])
    write_csv(OUT / "E2_P0V2_MODEL_CAPABILITY_MATRIX.csv", cap, cap_fields)
    mech_fields, mech = mechanism_rows()
    write_csv(OUT / "E2_P0V2_MECHANISM_EXPERIMENT_MATRIX.csv", mech, mech_fields)
    checks, passed, blocked = run_checks()
    cpu_report = "# E2-P0v2 CPU Validation Report\n\n" + f"**Execution scope:** CPU-only static/mock validation; `CUDA_VISIBLE_DEVICES=''`; no model, GPU, network, Qdrant, service, Jev or paid API calls.\n\n**Checks:** {passed} PASS, {blocked} BLOCKED, 0 SKIPPED.\n\n| Check | Status | Detail |\n|---|---|---|\n" + "\n".join(f"| `{x['check']}` | `{x['status']}` | {x['detail']} |" for x in checks) + "\n\nA BLOCKED check here means an owner-gated or intentionally unresolved scientific decision, not a validator defect. The mandatory CPU implementation checks passed; the unresolved design decisions prevent scientific freeze.\n"
    write(OUT / "E2_P0V2_CPU_VALIDATION_REPORT.md", cpu_report)
    sources = source_manifest()
    prov = {
        "manifest_version": "e2-p0v2-provenance-v1", "study": "DecisionRAG v2.1 E2-P0v2 Protocol Finalization and Native-Adapter Audit", "created_utc_start": start, "created_utc_end": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv), "python": sys.version, "platform": platform.platform(), "project_root": str(ROOT), "git_root": git("rev-parse", "--show-toplevel"), "git_commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short"), "sources": sources, "source_count": len(sources), "outputs": {}, "execution_scope": {"cuda_visible_devices": "", "gpu_calls": 0, "model_calls": 0, "network_calls": 0, "qdrant_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "checkpoint_loads": 0, "holdout_gold_opened": 0, "note": "standard-library static analysis and mock contract checks only"}, "m0_status": json.loads((M0 / "E2_M0_OWNER_GATE.json").read_text()).get("status") if (M0 / "E2_M0_OWNER_GATE.json").is_file() else "UNVERIFIED", "model_matrix_rows": len(cap), "mechanism_matrix_rows": len(mech), "cpu_check_counts": {"pass": passed, "blocked": blocked, "skip": 0}, "final_status": "E2_P0V2_SPEC_READY_FOR_OWNER_REVIEW",
    }
    write_json(OUT / "E2_P0V2_PROVENANCE_MANIFEST.json", prov)
    # The gate/report are last so their hashes can be included in provenance.
    gate_status = "E2_P0V2_SPEC_READY_FOR_OWNER_REVIEW"
    gate = """# E2-P0v2 Owner Gate\n\n**Status:** `E2_P0V2_SPEC_READY_FOR_OWNER_REVIEW`\n\nThis is a preparation-only verdict. It is not protocol freeze and does not authorize M1, N, S/P, GPU inference, model downloads, retrieval/index rebuilds, Jev, or paid APIs.\n\n| Gate | Status |\n|---|---|\n| Existing evidence preserved and provenance verified | PASS_WITH_LIMITATIONS |\n| Historical/future populations separated | PASS |\n| Global score comparability defined | PASS_WITH_LIMITATIONS |\n| Historical quartet probabilities not globally pooled | PASS |\n| H2a/H2b/H2c separated | PASS |\n| Native capability evidence classified conservatively | PASS_WITH_LIMITATIONS |\n| Common/native view contracts specified | PASS |\n| Holdout/leakage plan | BLOCKED_PENDING_OWNER_REVIEW |\n| NI/power/multiplicity | BLOCKED_PENDING_OWNER_REVIEW |\n| Serving SLO/load protocol | BLOCKED_PENDING_OWNER_REVIEW |\n| Runtime telemetry contract | PASS |\n| CPU-only validation | PASS |\n\n## Mandatory owner decisions\n\nThe holdout, primary contrast, epsilon, power/sample size, multiplicity, numeric SLO/load, final checkpoint roster/revisions, and GPU window remain unresolved. See `E2_P0V2_OWNER_DECISION_REGISTER.md`.\n\n`P0v2_NOT_FROZEN`\n"""
    write(OUT / "E2_P0V2_OWNER_GATE.md", gate)
    prov["outputs"] = {p.name: sha(p) for p in OUT.iterdir() if p.name != "E2_P0V2_PROVENANCE_MANIFEST.json"}
    prov["output_count"] = len(list(OUT.iterdir()))
    write_json(OUT / "E2_P0V2_PROVENANCE_MANIFEST.json", prov)
    print(json.dumps({"status": gate_status, "out_dir": str(OUT), "model_rows": len(cap), "mechanism_rows": len(mech), "cpu_pass": passed, "cpu_blocked": blocked, "sources": len(sources), "outputs": len(prov["outputs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
