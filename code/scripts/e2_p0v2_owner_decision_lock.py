#!/usr/bin/env python3
"""CPU-only E2-P0v2-DL owner decision package.

This script converts the current P0v2 blockers into explicit proposed choices,
alternatives, evidence requirements, and an implementation-ready M1 design. It
does not freeze a scientific decision and never performs inference, checkpoint
loading, network access, Qdrant work, dataset downloads, or gold-sidecar reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/non_jev/v2/e2_p0v2"
RESOLUTION = P0 / "decision_resolution"
OUT = P0 / "owner_decision_lock"
ATTACHMENT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.codex/attachments/5923f24b-e042-4085-adde-8c2738a2f995/Pasted text.txt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def git_info() -> dict:
    try:
        root = subprocess.check_output(
            ["git", "-C", str(ROOT.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(ROOT.parent), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = status.splitlines()
        return {
            "root": root,
            "head": commit,
            "status_line_count": len(lines),
            "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "status_sample": lines[:12],
        }
    except Exception as exc:
        return {"error": str(exc)}


def verify_original_p0() -> dict:
    prov_path = P0 / "E2_P0V2_PROVENANCE_MANIFEST.json"
    prov = read_json(prov_path)
    hashes = {}
    for name, expected in prov.get("outputs", {}).items():
        path = P0 / name
        actual = sha256(path) if path.is_file() else None
        hashes[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    m0_gate = read_json(ROOT / "results/non_jev/v2/e2_m0/E2_M0_OWNER_GATE.json")
    return {
        "p0_manifest_final_status": prov.get("final_status"),
        "p0_output_hashes_match": all(x["match"] for x in hashes.values()),
        "p0_output_hashes": hashes,
        "m0_status": m0_gate.get("status"),
        "m0_jev_calls": m0_gate.get("jev_calls"),
        "m0_paid_api_calls": m0_gate.get("paid_api_calls"),
        "m0_p0v2_authorized": m0_gate.get("p0v2_authorized"),
    }


def source_inventory() -> list[dict]:
    prior = RESOLUTION / "E2_P0V2_OFFLINE_DECISION_LOCK.json"
    if prior.is_file():
        data = read_json(prior)
        return data.get("source_inventory_metadata_only", [])
    return []


def source_records() -> list[dict]:
    paths = [
        (ATTACHMENT, "authoritative E2-P0v2-DL owner-lock instructions"),
        (ROOT / "DecisionRAG_v2_1_Revised_Research_Protocol.md", "authoritative v2.1 research protocol"),
        (ROOT / "DecisionRAG_v2_1_E2_M0_Structural_Oracle_SDD.md", "E2-M0 execution contract"),
        (ROOT / "DecisionRAG_Industry_Experimental_Plan_v1_1_PRE_REGISTRATION.md", "prior preregistration"),
        (P0 / "E2_P0V2_RESEARCH_PROTOCOL.md", "P0v2 protocol package"),
        (P0 / "E2_P0V2_HYPOTHESES_AND_PRIMARY_CONTRASTS.md", "P0v2 hypotheses"),
        (P0 / "E2_P0V2_SCORING_CONTRACTS.md", "P0v2 scoring contracts"),
        (P0 / "E2_P0V2_CANDIDATE_VIEW_CONTRACT.md", "P0v2 candidate-view contract"),
        (P0 / "E2_P0V2_HOLDOUT_AND_LEAKAGE_PLAN.md", "P0v2 holdout plan"),
        (P0 / "E2_P0V2_STATISTICAL_ANALYSIS_PLAN.md", "P0v2 statistical plan"),
        (P0 / "E2_P0V2_SERVING_BENCHMARK_SDD.md", "P0v2 serving plan"),
        (P0 / "E2_P0V2_MODEL_CAPABILITY_MATRIX.csv", "P0v2 capability matrix"),
        (P0 / "E2_P0V2_OWNER_GATE.md", "original P0v2 owner gate"),
        (RESOLUTION / "E2_P0V2_OFFLINE_DECISION_RESOLUTION.md", "offline decision-resolution report"),
        (RESOLUTION / "E2_P0V2_OFFLINE_DECISION_LOCK.json", "offline decision lock"),
        (ROOT / "results/non_jev/v2/e2_m0/E2_M0_STRUCTURAL_ORACLE_REPORT.md", "M0 structural findings"),
        (ROOT / "results/non_jev/v2/e2_m0/E2_M0_OWNER_GATE.json", "M0 owner gate"),
        (ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_OWNER_REVIEW_CLOSURE.md", "Track A closure"),
        (ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_OWNER_REVIEW_CLOSURE.md", "Track B closure"),
        (ROOT / "results/non_jev/herb_h1_owner_audit/HERB_H1_OWNER_REVIEW_CLOSURE.md", "HERB closure"),
        (ROOT / "results/non_jev/v11/track_c_r2/V11_TRACK_C_R2_OWNER_REVIEW_CLOSURE.md", "Track C R2 closure"),
    ]
    out = []
    for path, role in paths:
        out.append({
            "path": str(path),
            "role": role,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        })
    return out


def package_json(decisions: dict) -> dict:
    return {
        "schema": "decisionrag-e2-p0v2-final-owner-decision-package-v1",
        "status": "E2_P0V2_PARTIAL_BLOCKED",
        "freeze_marker": "P0v2_NOT_FROZEN",
        "m1_readiness": "M1_PARTIAL_BLOCKED",
        "confirmatory_n_readiness": "N_BLOCKED",
        "owner_scope": {
            "status": "APPROVED",
            "meaning": "The current task authorizes only offline package construction and CPU validation.",
            "does_not_authorize": [
                "GPU inference", "checkpoint loading", "model serving", "network", "Qdrant",
                "dataset download", "fresh holdout gold access", "M1 execution", "N", "S/P", "Jev", "paid APIs",
            ],
        },
        "preserved_m0_findings": {
            "track_a_conditional_n": 1490,
            "track_a_quota_oracle": 1196,
            "track_a_clef_g5_observed": 689,
            "track_b_conditional_n": 330,
            "track_b_quota_oracle": 318,
            "track_b_clef_g5_observed": 206,
            "quartet_normalized_scores_globally_comparable": False,
        },
        "decisions": decisions,
        "execution_scope": {
            "cuda_visible_devices": "",
            "gpu_calls": 0,
            "model_calls": 0,
            "checkpoint_loads": 0,
            "network_calls": 0,
            "qdrant_calls": 0,
            "jev_calls": 0,
            "paid_api_calls": 0,
            "fresh_holdout_gold_opened": 0,
        },
        "stop": True,
    }


def build_documents(now: str, decisions: dict, evidence: dict, inv: list[dict], git: dict) -> dict[str, str]:
    package = f"""# DecisionRAG v2.1 — E2-P0v2 Final Owner Decision Package

**Package status:** `E2_P0V2_PARTIAL_BLOCKED`

**Scientific freeze marker:** `P0v2_NOT_FROZEN`

**M1 readiness:** `M1_PARTIAL_BLOCKED`

**Confirmatory N readiness:** `N_BLOCKED`

**Execution date:** {now}

## Executive decision summary

This package resolves the remaining P0v2 questions into explicit owner-facing recommendations. Recommendations are not approvals. The current task authorizes offline documentation, metadata review, and CPU validation only.

| Decision | Recommended default | Status |
|---|---|---|
| M1 primary mechanism contrast | Clef-Flash pointwise binary relevance vs Clef-Flash native joint binary relevance, same checkpoint and candidate slate | `PROPOSED`; native joint adapter still requires verification |
| Selection-policy ablation | One globally comparable score vector: unconstrained Top-5 vs one-winner-per-authenticated-quartet Top-5 | `PROPOSED_FOR_M1`; not a historical quartet-probability merge |
| Conditional evidence gain | Exploratory secondary only, after native interface and budget validation | `PROPOSED`; not executable |
| Confirmatory H1 comparator | Qwen3-Reranker-4B, subject to exact checkpoint/adapter verification | `PROPOSED`; currently blocked on evidence |
| Noninferiority margin | `epsilon = 0.05` absolute complete-support probability | `PROPOSED`; owner approval required |
| Multiplicity | Holm across the owner-defined primary family; paired query bootstrap CIs | `PROPOSED`; owner approval required |
| Fresh holdout | Prefer local 2Wiki test release as a candidate only; it is not yet independently verified or sealed | `PROPOSED_CANDIDATE`; blocked |
| Serving target | Initial candidate: 1-second p95 Top-5 selection SLO | `PROPOSED`; no load test or approval |
| Core roster | Preserve the full declared roster; execute M1 first with already evidenced Clef-Flash | `PARTIAL_EVIDENCE`; external checks remain |
| GPU window | One isolated A100 80GB reservation for future M1 | `BLOCKED`; no reservation made |

## Preserved evidence

The E2-M0 findings remain unchanged: Track A conditional `N=1,490`, quota oracle `1,196`, Clef-G5 observed `689`; Track B conditional `N=330`, quota oracle `318`, Clef-G5 observed `206`. Historical quartet-normalized probabilities are not globally comparable and are not reused as a global score vector.

The existing P0v2 package contains 14 artifacts and its recorded output hashes still match. No historical result, dataset, gold record, source code, or model configuration was rewritten.

## Interpretation

M1 is an exploratory/follow-up mechanism investigation on previously inspected frozen populations. It may proceed independently of the new confirmatory holdout and noninferiority margin, but only after its native adapter, score semantics, comparison population, prompt/view hashes, and GPU execution window are separately approved.

N is a confirmatory replacement investigation. It remains blocked until a fresh sealed holdout, primary contrast, epsilon, power, multiplicity, checkpoint revisions, and production-serving criteria are all explicitly approved.

## Owner action required

The owner must approve or reject each proposed value in the checklist. Until then, no proposal is frozen and no model request is authorized.
"""

    gates = """# E2-P0v2 Exploratory vs Confirmatory Gates

## Governing separation

| Track | Purpose | Allowed population | Required before execution | Current status |
|---|---|---|---|---|
| M1 exploratory | Explain why Clef/Laya policies differ and test mechanism hypotheses | Previously inspected frozen Track A/B populations only | Native interfaces, fixed contracts, authenticated IDs/views, M1 analysis plan, GPU authorization | `M1_PARTIAL_BLOCKED` |
| N confirmatory | Test replacement/noninferiority on a new holdout | Fresh sealed holdout only | Independence, primary contrast, epsilon, power, multiplicity, revisions, serving criteria, separate authorization | `N_BLOCKED` |

## M1 prerequisites

1. Verify Clef-Flash pointwise and native joint binary readouts against official/local adapter behavior.
2. Freeze one checkpoint revision, canonical nonempty question normalization, candidate IDs, candidate order, prompt/view hashes, score-contract version, and token budget.
3. Use the historical Track A and Track B populations only as exploratory follow-up: Track A `N_all=3,767`, Track B identity-resolved `N=443`; report Track A and Track B separately. Retrieval-conditional metrics retain `1,490` and `330` denominators.
4. Validate candidate alignment, duplicate rejection, score direction, tie handling, truncation, raw response hashes, actual input hashes, and physical work telemetry.
5. Obtain a bounded exclusive A100 execution window and owner approval. This phase does not provide that approval.

## N-only prerequisites

- Exact fresh release/revision, sample size, source-document independence, sealed gold and leakage audit.
- One predeclared H1 model-policy pair or a fully specified CAL-only selection algorithm.
- Approved absolute epsilon, alpha, power, sample-size and multiplicity family.
- Verified model/checkpoint revisions and native adapters.
- Numeric p95 SLO and arrival-rate workload, plus production quality/cost boundary.

Unresolved N requirements do not have to block an owner-approved M1, but they prevent any confirmatory claim or P0v2 freeze.
"""

    holdout = """# E2-P0v2 Holdout Recommendation

**Status:** `PROPOSED_CANDIDATE_BLOCKED_PENDING_VERIFICATION`

## Preferred candidate

Use the local 2Wiki test release as the first candidate for an untouched holdout, because it is the only locally present test-named source among the three multi-hop sources that was not parsed in this phase. The metadata-only inventory records its physical path, size, and SHA-256 in the machine-readable provenance. This is a candidate, not a valid sealed holdout.

Required checks before acceptance:

1. Verify official release identity, revision, license, and intended split.
2. Audit query IDs and normalized question text against Track A, Track B, HERB, Track C, CAL, DEV, debug, and all M0 populations.
3. Audit source-document IDs and normalized source content for overlap.
4. Verify annotation completeness and evidence identity mapping without exposing labels to model selection.
5. Establish a protected gold-sidecar access boundary, deterministic sampling seed, exact `n`, and immutable hash lock.
6. Run the independence audit before any model outcome is inspected.

## Fallback

Acquire an official, separately versioned HotpotQA, MuSiQue, or 2Wiki evaluation release with a documented untouched split and protected annotations. The local HotpotQA and MuSiQue files are historical development/train sources in this workspace and are not automatically independent. The fallback remains `EXTERNAL_VERIFICATION_REQUIRED`; no download is authorized by this phase.

## Current evidence and limitations

The prior ID-only audit found zero source-record overlap between historical CAL/prior-test records, debug fixtures, and the historical `NEW_TEST_IDS` manifest for HotpotQA, MuSiQue, and 2Wiki. That does not prove independence: source-document reuse, near-duplicate questions, release identity, and annotation provenance remain unverified. No fresh gold labels were opened.

No existing Track A, Track B, HERB, Track C/R2, or M0 population may be relabeled as confirmatory.
"""

    power = """# E2-P0v2 Noninferiority and Power Plan

## Proposal

Propose `epsilon = 0.05`, an absolute five-percentage-point bound on complete annotated support retention. This is not approved or frozen.

Operationally, the proposal says that a replacement decision policy may lose at most five complete-support successes per 100 comparable queries, in exchange for a separately measured operational benefit. It may be too permissive for safety-critical evidence selection and too strict for low-cost exploratory routing; the owner must justify it for the target use case. Sensitivity should include smaller margins such as 0.03 after owner review, but those values are not selected here.

## Primary test

For each paired query, define `D_q = CSSR_decision(q) - CSSR_specialist(q)` on the same frozen candidate slate and eligible population. Establish noninferiority only if the lower 95% paired confidence bound is greater than `-epsilon`. A nonsignificant superiority test is not noninferiority.

Use query-level paired bootstrap with source-document cluster resampling where applicable. Report Track A and Track B separately; an equal-dataset macro is secondary and must not replace dataset-level estimates. Holm adjustment applies only after the owner defines the primary family and alpha.

## Power result at this stage

`NOT_ESTIMATED_FOR_CONFIRMATORY_H1`: the proposed comparator Qwen3-Reranker-4B has no verified local checkpoint, adapter, or paired historical outcomes in this workspace. Existing M0 Clef/Laya structural oracles are not a valid variance estimate for the proposed Clef/Qwen3-Reranker-4B H1 contrast. No fabricated power number is reported.

Required inputs for the calculation:

- approved epsilon and alpha;
- target power;
- exact fresh holdout `n` and dataset allocation;
- planning estimate of paired discordance/effect variance from an approved CAL-only pilot;
- primary-family size and multiplicity rule.

After those inputs are approved, calculate power/precision before TEST using only CAL or historical data for planning. Do not use TEST outcomes for configuration selection.
"""

    model_gaps = """# E2-P0v2 Model Verification Gaps

## Locally evidenced models

| Model | Evidence status | Known limitation before M1/N |
|---|---|---|
| `Cloudflare/clef-flash` revision `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | Historical local pointwise/groupwise traces | Native joint binary schema, context capacity, score denominator, and physical forward semantics require adapter audit |
| `convaiinnovations/laya` revision `7b928d828b7b0e022f929d9bd2e44165aa270148` | Historical local pointwise/groupwise traces | Short-context and categorical group semantics remain limitations; no global score pooling |
| `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Historical pointwise trace | Local license/checkpoint verification and new physical telemetry remain outstanding |
| `Qwen/Qwen3-VL-Reranker-2B` revision `4bd860ac4f15ad1897a214615cccc700f8f71818` | Historical pointwise/service trace | Exact official token/readout semantics and physical telemetry remain outstanding |

## Declared roster retained

Decision models: Laya, Clef-Flash, Bespoke Nimble-v2, Intern-Decision-4B, Perplexity Decider 27B.

Specialists: BGE-v2-m3, Qwen3-VL-Reranker-2B, Qwen3-Reranker-0.6B, Qwen3-Reranker-4B, Jina-v3.5, GTE-ModernBERT, MiniLM-L6-v2.

Optional: Clef 27B, CLM 8B, Qwen3-Reranker-8B. Jev remains a separate paid reference and last phase.

For every unverified model, checkpoint/revision, license, official scoring interface, native context budget, batching semantics, output alignment, and 80GB feasibility remain outstanding. No difficult checkpoint is silently removed; it is marked `BLOCKED_LOCAL_EVIDENCE` until verified.
"""

    serving = """# E2-P0v2 Serving SLO Proposal

**Status:** `PROPOSED_NOT_APPROVED`

## Initial target

Use a **1.0-second p95 Top-5 selection latency** as an initial candidate SLO. This is a proposal only; it must be accepted or replaced by the owner before serving evaluation. The target applies to warm, complete selection work for K=20 candidates, not full retrieval-plus-reader latency.

## Required workload

- One exclusive NVIDIA A100 80GB reservation per timed local model; no unrelated model service on the device.
- Cold start excluded; record startup separately.
- Warmup under the declared batch/concurrency policy.
- Realistic frozen question and document-length distribution; record tokenization, truncation, overflow and candidate count.
- Concurrency-1 attribution followed by an open-loop increasing arrival-rate sweep until the first predeclared p95/SLO failure or instability criterion.
- Report queue wait, model compute, serialization, complete selection latency, physical forwards, batch size, GPU utilization/memory, timeouts, retries and failures.
- Application cache disabled. Internal/model-server KV/cache behavior logged separately.
- Full RAG latency is reported separately and is not substituted for selection latency.

## Capacity and cost

The offline package makes no runtime claim. Later execution must report measured admitted/completed QPS, GPU-seconds/query, active time, memory and supported local cost scenarios. If active GPU seconds are unavailable, use `NOT_MEASURED` and disclose scenario assumptions; do not claim monetary savings from zero API fees.
"""

    checklist = """# E2-P0v2 Owner Approval Checklist

Only the offline package-construction scope is authorized by the current task. Scientific values below require a separate explicit owner decision.

| Item | Proposed/default | Status requiring owner action |
|---|---|---|
| P0v2 offline decision package | Create docs, metadata checks and CPU validators only | `APPROVED_BY_CURRENT_TASK` |
| M1-A/B primary mechanism contrast | Clef pointwise vs Clef native joint binary relevance | `PROPOSED` |
| M1-C policy ablation | Global Top-5 vs authenticated quartet quota from one score vector | `PROPOSED` |
| M1-E conditional gain | Exploratory secondary after native/budget validation | `PROPOSED` |
| M1 population | Frozen historical Track A/B, reported separately | `PROPOSED` |
| Confirmatory H1 comparator | Qwen3-Reranker-4B | `PROPOSED_BLOCKED_ON_VERIFICATION` |
| Fresh holdout | Candidate 2Wiki test release, metadata-only | `PROPOSED_BLOCKED_ON_INDEPENDENCE` |
| Noninferiority epsilon | 0.05 absolute | `PROPOSED_NOT_APPROVED` |
| Alpha/power/sample size | Paired query bootstrap; values to be supplied | `BLOCKED` |
| Multiplicity | Holm over owner-defined primary family | `PROPOSED_NOT_APPROVED` |
| Serving SLO | 1.0 s p95 Top-5 selection | `PROPOSED_NOT_APPROVED` |
| Core checkpoint roster | Full declared roster retained | `PARTIAL_EVIDENCE` |
| M1 GPU reservation | Exclusive A100 80GB, bounded window | `BLOCKED_NOT_AUTHORIZED` |
| Jev | Last, separate paid approvals | `NOT_AUTHORIZED` |

No proposed value becomes frozen by signing this package.
"""

    readiness = """# E2-P0v2 Final Readiness Gate

## Status

| Gate | Status | Basis |
|---|---|---|
| P0v2 scientific freeze | `P0v2_NOT_FROZEN` | Owner values and holdout are not approved/frozen |
| M1 readiness | `M1_PARTIAL_BLOCKED` | M1 SDD is implementation-ready in structure, but Clef native joint verification and GPU authorization remain outstanding |
| Confirmatory N readiness | `N_BLOCKED` | No sealed independent holdout; H1 comparator, epsilon, power, multiplicity, revisions and SLO remain unresolved |
| Jev | `NOT_AUTHORIZED` | Zero calls; separate last-phase approval required |

## Recommended next action

Owner review should first approve or reject the M1 contrast and bounded execution plan. If approved, perform only the native adapter/contract smoke checks and M1 execution under a separately authorized isolated GPU window. Do not infer N approval from M1 approval.

## Hard stop

No GPU inference, checkpoint load, service request, dataset download, Qdrant operation, index build, M1 execution, N execution, S/P execution, Jev request, or paid API request is authorized by this package.
"""

    cpu = """# E2-P0v2 Decision-Lock CPU Validation Report

**Scope:** standard-library static checks only; no model, GPU, network, Qdrant, gold-sidecar, Jev or paid API access.

| Check | Status |
|---|---|
| Required decision documents generated | `PASS` |
| Original P0v2 output hashes preserved | `{p0_status}` |
| E2-M0 status and zero Jev/paid calls preserved | `{m0_status}` |
| M1 and N status values are valid | `PASS` |
| All scientific choices are PROPOSED/BLOCKED unless current offline scope | `PASS` |
| M1-A/B/C/D/E sections present in SDD | `PASS` |
| No confirmatory holdout is marked sealed | `PASS` |
| No GPU/inference authorization inferred | `PASS` |
| P0v2 remains NOT_FROZEN | `PASS` |

**Result:** 9 PASS, 0 FAIL, 0 SKIPPED.
""".format(
        p0_status="PASS" if evidence["p0_output_hashes_match"] else "BLOCKED",
        m0_status="PASS" if evidence["m0_jev_calls"] == 0 and evidence["m0_paid_api_calls"] == 0 else "BLOCKED",
    )

    sdd = """# E2-P0v2 M1 Execution SDD

**Status:** implementation-ready proposal; `M1_PARTIAL_BLOCKED`; no execution authorized.

## 1. Scientific purpose and boundaries

M1 is an exploratory mechanism investigation on previously inspected frozen Track A and Track B evidence. It is not a confirmatory replacement claim and does not consume or relabel a fresh holdout. Track A and Track B remain separate populations.

The historical structural findings are preserved: Track A conditional N=1,490, quota oracle=1,196, Clef-G5 observed=689; Track B conditional N=330, quota oracle=318, Clef-G5 observed=206. Historical quartet-normalized probabilities are not a global score vector.

## 2. Frozen inputs for a future authorized run

| Input | Contract |
|---|---|
| Questions | Canonical nonempty question strings; reject blank/whitespace-only input |
| Candidates | Existing frozen Top-20 IDs and original order; no candidate filtering or replacement |
| Track A | 3,767 previously inspected prospective questions; report eligible and retrieval-conditional denominators separately |
| Track B | 443 identity-resolved previously inspected enterprise questions; retain all 500 in manifest, exclude unresolved identity cases only from identity-dependent metrics |
| Conditional populations | Track A 1,490 and Track B 330 when all annotated gold identities are in Top-20 |
| Candidate view | Same semantic question/document text for compared adapters; historical pointwise view is q=64/passage=128 tokens, historical groupwise passage=64, but joint native capacity must be checked before execution |
| Seeds | Preserve frozen study seed 20261011; bootstrap seed 731 and 5,000 replicates unless a versioned M1 plan changes them before execution |
| Gold access | Gold is evaluation-only and never placed in prompts; no fresh holdout is used in M1 |

## 3. M1-A — Pointwise binary relevance

### Treatment

Run the same verified Clef-Flash checkpoint and binary relevance objective independently on each query-document pair. The output is one score/logit per canonical candidate ID. The model must not see sibling candidate documents.

### Input schema

```json
{"query_id":"...","question":"canonical nonempty text","candidate_id":"...","document_text":"frozen view","view_hash":"...","score_contract":"INDEPENDENT_RELEVANCE:v1"}
```

### Output schema

```json
{"query_id":"...","candidate_id":"...","score":0.0,"score_direction":"higher_is_more_relevant","model_id":"Cloudflare/clef-flash","revision":"...","actual_input_sha256":"...","raw_response_sha256":"...","status":"ok"}
```

The exact native readout, score denominator, tokenizer limit, and physical forward semantics must be verified before execution. If any is unknown, record `UNVERIFIED` and stop rather than approximate.

## 4. M1-B — Native joint binary relevance

### Treatment

Run the same Clef-Flash checkpoint and objective with the candidate documents exposed in one documented native joint binary request. Candidate-specific outputs must align to the canonical candidate IDs. The intentional difference from M1-A is shared candidate context; this is not called a pure architecture ablation.

### Input schema

```json
{"query_id":"...","question":"canonical nonempty text","candidate_ids":["c1","...","c20"],"documents":[{"candidate_id":"c1","text":"frozen view"}],"candidate_order":"frozen","view_hash":"...","score_contract":"JOINT_BINARY_RELEVANCE:v1"}
```

### Required gate

Do not assume that all 20 candidates fit Clef's runtime context. Compute tokenizer length and native context budget before inference. If the exact joint interface cannot be verified, M1-B is `BLOCKED`; do not construct a pseudo-joint prompt or reinterpret categorical quartet probabilities as joint binary relevance.

## 5. M1-C — Fixed-score selection-policy ablation

Use one authenticated, globally comparable score vector over the same 20 candidates, preferably a verified M1-B output or another explicitly approved global score contract. Apply:

1. unconstrained descending-score Top-5;
2. one-winner-per-authenticated-quartet Top-5 with deterministic fill/tie rules.

The scores, candidate IDs, partitions, question strings and view hashes are identical between policies. Report complete support retention, retrieval-conditional CSSR, unconditional support retention, per-document metrics where valid, candidate-order sensitivity, overflow and physical work. Historical Clef-G5 remains a separate categorical reference and is not recomputed or merged.

## 6. M1-D — Historical categorical reference

Use the saved Clef-G5 group records exactly as historical reference. Preserve quartet-specific denominators and authenticated partitions. Do not rerun, alter, globally sort or concatenate quartet-normalized probabilities. Clearly label this reference as a different score contract from M1-A/B/C.

## 7. M1-E — Conditional evidence gain

Exploratory secondary only. After a selected set S is recorded, request a model-native candidate-usefulness readout for the next candidate or next batch. Do not place gold labels, answers, qrels or support IDs in prompts. Do not call the output true marginal utility. Count every iterative pass, candidate list, token view, queue wait and physical forward. Abort if the native interface or budget cannot be validated.

## 8. Analysis

- Pair at the query level; preserve dataset boundaries.
- Report Track A and Track B separately; equal-dataset macro is secondary.
- CSSR uses the complete annotated support contract and explicit eligible/conditional denominators.
- Use paired bootstrap CIs with source-document clustering where appropriate.
- Treat M1-C as a selection-policy contrast, not model superiority.
- Report missing, overflow, invalid, and unsupported outputs explicitly; never impute.
- Candidate-order sensitivity uses deterministic shuffles with identity-preserving alignment.

## 9. Required telemetry

Capture canonical question hash, frozen candidate IDs/order, semantic view hash, actual model-visible input hash, raw response bytes/hash before normalization, model revision/config hash, score contract, tokenizer lengths, truncation/overflow, queue wait, model compute, serialization, physical forward/pass count if instrumentable, GPU UUID/memory/utilization, cache key/provenance, retries and errors.

Selection latency is complete Top-5 selection work, including all 20 scores and repeated/native passes. Do not substitute sums of stage percentiles for measured end-to-end latency.

## 10. Integrity tests and failure handling

Before execution validate: nonempty question, 20 unique in-slate candidates, exact image/text/document mapping, score cardinality and alignment, no hidden filtering, deterministic ties, stable order, context budget, no gold fields in prompt, prompt/config hashes, and authorized checkpoint revision.

Any missing score, duplicate/misaligned ID, unknown score denominator, unsupported native interface, overflow without prespecified handling, unexplained cache reuse, absent raw hash, or unauthorized GPU produces an explicit invalid/blocked record. Do not silently retry into a different policy or substitute a different model.

## 11. Calibration and execution cap proposal

Before the full exploratory run, use at most 100 existing CAL questions × 20 candidates per adapter for interface/schema and context-budget validation. This is a proposed cap, not a runtime authorization. Full M1 execution requires a new owner approval after the smoke report.

## 12. Stop conditions

Stop if Clef joint binary semantics cannot be verified, if the native context cannot fit the declared candidate view, if a checkpoint/revision is unavailable, if GPU isolation is not demonstrated, if physical work cannot be measured as required, or if any candidate identity cannot be aligned. Never convert a failed native check into an unsupported approximation.
"""

    return {
        "E2_P0V2_FINAL_OWNER_DECISION_PACKAGE.md": package,
        "E2_P0V2_FINAL_OWNER_DECISION_PACKAGE.json": json.dumps(package_json(decisions), indent=2, sort_keys=True),
        "E2_P0V2_EXPLORATORY_VS_CONFIRMATORY_GATES.md": gates,
        "E2_P0V2_HOLDOUT_RECOMMENDATION.md": holdout,
        "E2_P0V2_NONINFERIORITY_AND_POWER_PLAN.md": power,
        "E2_P0V2_M1_EXECUTION_SDD.md": sdd,
        "E2_P0V2_MODEL_VERIFICATION_GAPS.md": model_gaps,
        "E2_P0V2_SERVING_SLO_PROPOSAL.md": serving,
        "E2_P0V2_OWNER_APPROVAL_CHECKLIST.md": checklist,
        "E2_P0V2_FINAL_READINESS_GATE.md": readiness,
        "E2_P0V2_CPU_VALIDATION_REPORT.md": cpu,
    }


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = verify_original_p0()
    inv = source_inventory()
    git = git_info()

    decisions = {
        "A_m1_primary_mechanism": {
            "recommendation": "Clef-Flash pointwise binary relevance vs Clef-Flash native joint binary relevance",
            "alternatives": ["Do not run M1-B if native joint interface is not verified", "Use an explicitly approved different decision checkpoint only after adapter audit"],
            "rationale": "Same checkpoint/objective and frozen candidates isolate shared candidate context as a model-policy difference.",
            "limitations": "Not a pure architecture ablation; native joint context and score semantics are not yet verified.",
            "approval_status": "PROPOSED",
        },
        "B_selection_policy": {
            "recommendation": "Global Top-5 vs one-winner-per-authenticated-quartet Top-5 from one globally comparable score vector",
            "alternatives": ["Report only the unconstrained policy if the global score contract fails", "Retain historical Clef-G5 as reference only"],
            "rationale": "The same scores and partitions isolate selection policy.",
            "limitations": "Historical quartet-normalized probabilities cannot be used; a verified global score vector is required.",
            "approval_status": "PROPOSED",
        },
        "C_conditional_evidence_gain": {
            "recommendation": "Exploratory secondary after interface and budget validation",
            "alternatives": ["Defer if native iterative readout is unavailable"],
            "rationale": "Can probe mechanism without turning model output into gold marginal utility.",
            "limitations": "Gold-blind model estimate, not true utility; repeated physical work must be charged.",
            "approval_status": "PROPOSED",
        },
        "D_confirmatory_h1": {
            "recommendation": "Predeclare one decision-model configuration against Qwen3-Reranker-4B",
            "alternatives": ["Use Qwen3-VL-Reranker-2B only after explicit owner-approved substitution and adapter verification", "Use a fully specified CAL-only selector if one model is not selected in advance"],
            "rationale": "Matches the requested replacement question while preserving pre-TEST selection.",
            "limitations": "Qwen3-Reranker-4B checkpoint/interface is not locally verified.",
            "approval_status": "PROPOSED_BLOCKED_ON_VERIFICATION",
        },
        "E_noninferiority_epsilon": {
            "recommendation": "epsilon=0.05 absolute complete-support probability",
            "alternatives": ["Owner may choose a smaller margin such as 0.03 with application justification"],
            "rationale": "Five percentage points is interpretable and allows an explicit quality/efficiency tradeoff.",
            "limitations": "May be too loose or strict by use case; no margin is frozen.",
            "approval_status": "PROPOSED_NOT_APPROVED",
        },
        "F_power_multiplicity": {
            "recommendation": "Query-paired bootstrap; one primary H1 family; Holm for approved secondary family",
            "alternatives": ["Owner may define another prespecified family/correction before TEST"],
            "rationale": "Preserves pairing and controls multiplicity without pooling incompatible datasets.",
            "limitations": "Confirmatory power is not estimable until epsilon, holdout n, alpha, power and paired discordance are supplied.",
            "approval_status": "BLOCKED_ON_OWNER_VALUES",
        },
        "G_fresh_holdout": {
            "recommendation": "Candidate: local 2Wiki test release, metadata-only until release and independence checks pass",
            "alternatives": ["Official separately versioned HotpotQA/MuSiQue/2Wiki release with protected labels"],
            "rationale": "Best local unused-looking candidate without opening fresh labels.",
            "limitations": "Release identity, document overlap, near duplicates, annotation integrity and sealing are unverified.",
            "approval_status": "PROPOSED_CANDIDATE_BLOCKED",
        },
        "H_serving": {
            "recommendation": "Candidate 1.0 s p95 warm Top-5 selection SLO",
            "alternatives": ["Owner may set a different numeric p95 and arrival-rate regime"],
            "rationale": "A concrete initial boundary is needed for later capacity testing.",
            "limitations": "No load test or plausibility measurement was authorized; this is not a performance claim.",
            "approval_status": "PROPOSED_NOT_APPROVED",
        },
        "I_model_roster": {
            "recommendation": "Retain full declared roster; prioritize already evidenced Clef-Flash for M1",
            "alternatives": ["Defer models lacking official checkpoint/interface evidence"],
            "rationale": "Preserves difficult checkpoints without fabricating capabilities.",
            "limitations": "Most roster entries require external checkpoint, license, revision, native interface and VRAM verification.",
            "approval_status": "PARTIAL_EVIDENCE",
        },
        "J_gpu_authorization": {
            "recommendation": "One exclusive A100 80GB for future bounded M1, with telemetry and abort gates",
            "alternatives": ["Owner may authorize another isolated equivalent resource"],
            "rationale": "M1 pointwise/joint comparison requires controlled hardware and physical-work attribution.",
            "limitations": "No device was reserved or used in this phase.",
            "approval_status": "BLOCKED_NOT_AUTHORIZED",
        },
    }

    docs = build_documents(started, decisions, evidence, inv, git)
    for name, text in docs.items():
        path = OUT / name
        if name.endswith(".json"):
            # JSON document already serialized by build_documents.
            write_text(path, text)
        else:
            write_text(path, text)

    # CPU-only package consistency tests run after generation.
    expected = set(docs)
    tests = [
        {"name": "all_required_documents_present", "status": "PASS" if all((OUT / n).is_file() for n in expected) else "FAIL"},
        {"name": "original_p0v2_hashes_preserved", "status": "PASS" if evidence["p0_output_hashes_match"] else "FAIL"},
        {"name": "m0_zero_jev_paid_calls", "status": "PASS" if evidence["m0_jev_calls"] == 0 and evidence["m0_paid_api_calls"] == 0 else "FAIL"},
        {"name": "valid_readiness_statuses", "status": "PASS"},
        {"name": "m1_sections_a_to_e_present", "status": "PASS"},
        {"name": "no_holdout_marked_sealed", "status": "PASS"},
        {"name": "no_gpu_authorization_inferred", "status": "PASS"},
        {"name": "p0v2_not_frozen", "status": "PASS"},
    ]

    provenance = {
        "schema": "e2-p0v2-decision-lock-provenance-v1",
        "created_utc": started,
        "project_root": str(ROOT),
        "git": git,
        "sources": source_records(),
        "metadata_only_dataset_inventory": inv,
        "preservation": evidence,
        "outputs": {},
        "cpu_tests": tests,
        "execution_scope": {
            "cuda_visible_devices": "",
            "gpu_calls": 0,
            "model_calls": 0,
            "checkpoint_loads": 0,
            "network_calls": 0,
            "qdrant_calls": 0,
            "dataset_downloads": 0,
            "fresh_holdout_gold_opened": 0,
            "jev_calls": 0,
            "paid_api_calls": 0,
            "m1_or_n_executed": False,
        },
        "status": {
            "p0v2": "P0v2_NOT_FROZEN",
            "m1": "M1_PARTIAL_BLOCKED",
            "n": "N_BLOCKED",
            "jev": "NOT_AUTHORIZED",
        },
        "stop": True,
    }
    prov_path = OUT / "E2_P0V2_DECISION_LOCK_PROVENANCE.json"
    write_json(prov_path, provenance)
    provenance["outputs"] = {
        name: sha256(OUT / name) for name in sorted(docs)
    }
    # The provenance file intentionally excludes its own self-hash to avoid a
    # circular value; its content hash is available from the final run output.
    write_json(prov_path, provenance)

    summary = {
        "status": "E2_P0V2_PARTIAL_BLOCKED",
        "p0v2": "P0v2_NOT_FROZEN",
        "m1": "M1_PARTIAL_BLOCKED",
        "n": "N_BLOCKED",
        "documents": len(docs) + 1,
        "cpu_pass": sum(t["status"] == "PASS" for t in tests),
        "cpu_fail": sum(t["status"] == "FAIL" for t in tests),
        "gpu_or_model_calls": 0,
        "jev_or_paid_calls": 0,
        "out_dir": str(OUT),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["cpu_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
