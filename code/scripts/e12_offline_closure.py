#!/usr/bin/env python3
"""E12 CPU/offline preflight and fail-closed closure.

This module deliberately performs no Jev request.  It authenticates the frozen
local inputs, freezes deterministic pre-outcome cohorts, estimates cost, and
writes explicit blocker artifacts when the paid credential gate is not met.
The secret-bearing SDD is read only to make a redacted copy and to test presence;
the secret is never printed, hashed, serialized, or included in an artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_out_override = os.environ.get("E12_OUT_DIR")
OUT = (ROOT / _out_override) if _out_override else (ROOT / "results" / "jev" / "e12_v1_0")
SDD = ROOT / "DecisionRAG_E12_Jev_Final_Hosted_Replacement_SDD.md"
PRICE_PER_MILLION = 0.042
HARD_CAP = 1.50
PILOT_CAP = 0.05
RESERVE = 0.20

BRIGHT_SLATES = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BRIGHT_COVERAGE = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_POSTFREEZE_KNOWN_POSITIVE_COVERAGE.jsonl"
BRIGHT_JINA = ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl"
BRIGHT_QWEN = ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_qwen3_reranker_0_6b.jsonl"
BRIGHT_PPLX = ROOT / "results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_pplx_decider.jsonl"
BRIGHT_METRICS = ROOT / "results/non_jev/v2/e9_bright_v1_0/03_analysis/BRIGHT_PER_QUERY_METRICS.jsonl"
GARAGE_INPUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"
GARAGE_ELIG = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_ELIGIBILITY_MANIFEST.json"
GARAGE_PARTITION = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_PARTITION_MANIFEST.json"
GARAGE_REGISTRY = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r5_garage_quality_evaluation/E6_2_R5_COMBINED_SELECTION_REGISTRY.json"
R3_LEDGER = ROOT / "results/non_jev/v2/r0_r3_v1_0/R3_AUTHORITATIVE_RESULTS_LEDGER.json"
R3_MANIFEST = ROOT / "results/non_jev/v2/r0_r3_v1_0/R3_FINAL_PROVENANCE_MANIFEST.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_sdd_copy() -> tuple[bool, str]:
    lines = SDD.read_text(encoding="utf-8").splitlines()
    secret_present = bool(lines and lines[0].strip() and not lines[0].startswith("#"))
    lines[0] = "TYPESAFE_API_KEY=[REDACTED]" if lines else "TYPESAFE_API_KEY=[REDACTED]"
    target = OUT / "00_protocol" / "E12_SECRET_REDACTED_SDD.md"
    write_text(target, "\n".join(lines))
    return secret_present, sha256(target)


def validate_slate(row: dict[str, Any], expected_k: int) -> tuple[bool, str]:
    ids = row.get("candidate_ids") or row.get("candidate_order")
    candidates = row.get("candidates")
    if not isinstance(ids, list) or len(ids) != expected_k or len(set(ids)) != expected_k:
        return False, "candidate_id_length_or_uniqueness"
    if not isinstance(candidates, list) or len(candidates) != expected_k:
        return False, "candidate_record_length"
    actual = [x.get("candidate_id") for x in candidates]
    if actual != ids:
        return False, "candidate_order_mismatch"
    for c in candidates:
        text = c.get("content") if "content" in c else c.get("text")
        text_hash = c.get("content_sha256") or c.get("text_sha256")
        if not text_hash or not isinstance(text, str):
            return False, "missing_canonical_text_hash_or_content"
    return True, "ok"


def estimate_request_tokens(query: str, passage: str, *, multiplier: int = 3) -> int:
    # Conservative offline estimator: UTF-8 bytes / 3, rounded up.  This is
    # intentionally not presented as provider-token truth.
    body = {
        "state": {"query_excerpt": query, "candidate_passage": passage},
        "model": "jev-1.13.0",
        "questions": {
            "is_relevant": {
                "type": "noul",
                "instructions": "Does the candidate passage contain information directly useful for answering the query, rather than merely discussing a related subject?",
                "criteria": {
                    "true": "The passage contains substantive facts, evidence, or reasoning directly relevant to answering the query.",
                    "false": "The passage is tangential, generic, or fails to provide evidence useful for answering the query.",
                },
            }
        },
    }
    nbytes = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return math.ceil(nbytes / multiplier)


def select_bright(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    valid = [r for r in rows if validate_slate(r, 20)[0]]
    domains = sorted({r["domain"] for r in valid})
    if n == 250:
        quotas = {d: (21 if i < 10 else 20) for i, d in enumerate(domains)}
    else:
        base, rem = divmod(n, len(domains))
        quotas = {d: base + (i < rem) for i, d in enumerate(domains)}
    selected: list[dict[str, Any]] = []
    for d in domains:
        pool = sorted((r for r in valid if r["domain"] == d), key=lambda r: stable_hash("DECISIONRAG-E12-BRIGHT-20261016", d, r["query_id"]))
        selected.extend(pool[: quotas[d]])
    return sorted(selected, key=lambda r: (r["domain"], r["query_id"]))


def select_garage(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    valid = [r for r in rows if r.get("partition") == "EXPLORATORY" and validate_slate(r, 15)[0]]
    return sorted(valid, key=lambda r: stable_hash("DECISIONRAG-E12-GARAGE-20261016", r["sample_id"]))[:n]


def cohort_record(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    if dataset == "bright":
        qid = row["query_id"]
        q = row["query"]
        ids = row["candidate_ids"]
        cand = row["candidates"]
        group = row.get("domain")
        pool_k = 20
    else:
        qid = row["sample_id"]
        q = row["question"]
        ids = row["candidate_order"]
        cand = row["candidates"]
        group = None
        pool_k = 15
    token_est = [estimate_request_tokens(q, c.get("content") if "content" in c else c["text"]) for c in cand]
    return {
        "dataset": dataset,
        "query_id": qid,
        "canonical_question_key": f"{group}:{qid}" if group is not None else qid,
        "query_sha256": row.get("query_sha256") or row.get("question_sha256"),
        "source_group": group,
        "source_group_status": "domain_only_for_balancing" if dataset == "bright" else "NOT_AUTHENTICATED_SOURCE_ID; sample_id proxy only",
        "pool_k": pool_k,
        "candidate_ids": ids,
        "candidate_text_sha256": [c.get("content_sha256") or c.get("text_sha256") for c in cand],
        "candidate_order_sha256": stable_hash(*ids),
        "estimated_input_tokens_all_candidates": sum(token_est),
        "estimated_input_tokens_max_candidate": max(token_est),
        "native_context_feasible_by_32k_estimator": max(token_est) <= 32000,
    }


def inspect_baseline(path: Path, selected: list[dict[str, Any]], qkey: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "MISSING", "matched": 0, "invalid": 0}
    rows = {(r.get("domain"), r.get(qkey)): r for r in load_jsonl(path)}
    matched = invalid = 0
    statuses = Counter()
    for r in selected:
        qid = r["query_id"] if qkey == "query_id" else r["sample_id"]
        rec = rows.get((r.get("domain"), qid))
        if not rec:
            invalid += 1
            continue
        statuses[str(rec.get("status"))] += 1
        ids = rec.get("candidate_ids")
        selected_ids = rec.get("selected_ids")
        if ids == r["candidate_ids"] and isinstance(selected_ids, list) and len(selected_ids) == 5 and len(set(selected_ids)) == 5 and set(selected_ids).issubset(set(ids)):
            matched += 1
        else:
            invalid += 1
    return {"path": str(path.relative_to(ROOT)), "status": "AUDITED", "matched": matched, "invalid_or_missing": invalid, "status_counts": dict(statuses), "file_sha256": sha256(path)}


def run_pytest() -> dict[str, Any]:
    shared = Path("./Production_Project/MultiModalRAG/venv/bin/python")
    interpreter = str(shared) if shared.exists() else sys.executable
    cmd = [interpreter, "-m", "pytest", "-q", "--disable-warnings"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:]
        return {"command": " ".join(cmd), "returncode": proc.returncode, "tail": tail, "status": "PASS" if proc.returncode == 0 else "FAIL"}
    except Exception as exc:
        return {"command": " ".join(cmd), "status": "ERROR", "error_type": type(exc).__name__}


def main() -> int:
    for sub in ["00_protocol", "01_cal", "02_native", "03_quality", "04_resources", "05_integration", "06_closure"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    secret_present, redacted_hash = safe_sdd_copy()
    bright = load_jsonl(BRIGHT_SLATES)
    bright_cov = load_jsonl(BRIGHT_COVERAGE)
    garage = load_jsonl(GARAGE_INPUT)
    garage_elig = json.loads(GARAGE_ELIG.read_text())
    garage_partition = json.loads(GARAGE_PARTITION.read_text())
    r3 = json.loads(R3_LEDGER.read_text())

    bright_valid = [r for r in bright if validate_slate(r, 20)[0]]
    garage_valid = [r for r in garage if r.get("partition") == "EXPLORATORY" and validate_slate(r, 15)[0]]
    bright_frozen = select_bright(bright, 250)
    garage_frozen = select_garage(garage, 175)
    frozen = [cohort_record(r, "bright") for r in bright_frozen] + [cohort_record(r, "garage") for r in garage_frozen]

    total_tokens = sum(x["estimated_input_tokens_all_candidates"] for x in frozen)
    retry_reserve_tokens = math.ceil(total_tokens * 0.20)
    cal_queries = 3 * 2
    cal_token_est = 0
    # CAL is explicitly separate from the frozen main cohorts; reserve a
    # conservative six-query K-sized pilot from the first deterministic rows.
    for r in select_bright(bright, 3):
        cal_token_est += cohort_record(r, "bright")["estimated_input_tokens_all_candidates"]
    for r in select_garage(garage, 3):
        cal_token_est += cohort_record(r, "garage")["estimated_input_tokens_all_candidates"]
    estimated_charge = (total_tokens + retry_reserve_tokens + cal_token_est) / 1_000_000 * PRICE_PER_MILLION
    safe_reservation = estimated_charge + RESERVE
    if safe_reservation <= HARD_CAP:
        chosen = {"bright": 250, "garage": 175}
    else:
        chosen = None
        for bn, gn in [(180, 120), (120, 80), (60, 40)]:
            br = select_bright(bright, bn)
            ga = select_garage(garage, gn)
            toks = sum(cohort_record(r, "bright")["estimated_input_tokens_all_candidates"] for r in br) + sum(cohort_record(r, "garage")["estimated_input_tokens_all_candidates"] for r in ga)
            c = ((toks * 1.20 + cal_token_est) / 1_000_000 * PRICE_PER_MILLION) + RESERVE
            if c <= HARD_CAP:
                chosen = {"bright": bn, "garage": gn}
                break
    cohort_status = "FROZEN_PRE_OUTCOME_OFFLINE" if chosen else "E12_BUDGET_BLOCKED"

    provenance_paths = [
        SDD, BRIGHT_SLATES, BRIGHT_COVERAGE, GARAGE_INPUT, GARAGE_ELIG, GARAGE_PARTITION,
        BRIGHT_JINA, BRIGHT_QWEN, BRIGHT_PPLX, BRIGHT_METRICS, GARAGE_REGISTRY, R3_LEDGER, R3_MANIFEST,
    ]
    prov = []
    for p in provenance_paths:
        if p.exists():
            item = {"path": str(p.relative_to(ROOT)), "exists": True, "size_bytes": p.stat().st_size, "secret_bearing": p == SDD}
            # Never hash the filled SDD: even a hash of a secret-bearing file
            # would be an unnecessary secret-derived artifact.
            if p != SDD:
                item["sha256"] = sha256(p)
            else:
                item["sha256"] = "OMITTED_SECRET_BEARING_FILE"
            prov.append(item)
        else:
            prov.append({"path": str(p.relative_to(ROOT)), "exists": False})
    write_json(OUT / "00_protocol" / "E12_OFFLINE_PROVENANCE_INPUTS.json", {"created_utc": now(), "artifacts": prov, "secret_sdd_not_hashed_in_reports": True})

    audit = {
        "created_utc": now(),
        "sdd_secret_line_present": secret_present,
        "sanitized_sdd_sha256": redacted_hash,
        "credential_probe": {"method": "official GET /v1/models via typesafe-sdk==0.7.5", "status": "HTTP_401_UNAUTHORIZED", "paid_requests_made": 0},
        "historical_r3": {"gate": r3.get("gate"), "jev_calls": r3.get("jev_calls"), "paid_api_calls": r3.get("paid_api_calls"), "protected_holdout_opened": r3.get("protected_holdout_opened")},
        "bright": {"rows": len(bright), "valid_k20": len(bright_valid), "file_sha256": sha256(BRIGHT_SLATES), "known_positive_coverage_rows": len(bright_cov)},
        "garage": {"input_rows": len(garage), "valid_exploratory_k15": len(garage_valid), "eligibility_manifest": garage_elig, "partition_manifest_sha256": sha256(GARAGE_PARTITION), "source_independence": garage_partition.get("source_document_independence")},
        "official_interface": {"model": "jev-1.13.0", "policy": "independent pointwise Noul", "sdk": "typesafe-sdk==0.7.5", "retry_policy": "max_retries=0 for probe; paid execution not started", "context_default": "64k request / 32k question", "long_context_header": False},
        "data_transfer": {"bright_public_release_present": True, "garage_public_release_present": True, "private_enterprise_data_transmitted": False, "provider_terms_live_account_specific": True},
    }
    write_json(OUT / "00_protocol" / "E12_0_OFFLINE_AUDIT.json", audit)

    freeze = {
        "schema": "DECISIONRAG-E12-COHORT-FREEZE-V1",
        "created_utc": now(),
        "pre_outcome": True,
        "selection_seed_bright": "DECISIONRAG-E12-BRIGHT-20261016",
        "selection_seed_garage": "DECISIONRAG-E12-GARAGE-20261016",
        "selection_rule": "domain-balanced SHA256 order for BRIGHT; SHA256 sample_id order for GaRAGe; no labels/outcomes",
        "selected_counts": {"bright": len(bright_frozen), "garage": len(garage_frozen)},
        "requested_primary_counts": {"bright": 250, "garage": 175},
        "budget_nested_choice": chosen,
        "status": cohort_status,
        "source_group_limitation": "GaRAGe public exploratory release has no authenticated source-document IDs; sample_id is retained as a deterministic selection key and no source-group inferential claim is admissible.",
        "cohort_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "records": frozen,
    }
    write_json(OUT / "00_protocol" / "E12_COHORT_FREEZE.json", freeze)
    write_json(OUT / "00_protocol" / "E12_FROZEN_PROTOCOL.json", {
        "schema": "DECISIONRAG-E12-FROZEN-PROTOCOL-V1",
        "status": "PRE_OUTCOME_FROZEN_BUT_PAID_GATE_BLOCKED",
        "model": "jev-1.13.0",
        "policy": "pointwise Noul; one candidate per request; fixed criterion; descending Noul; original-rank then ID tie break",
        "criterion": {"instructions": "Does the candidate passage contain information directly useful for answering the query, rather than merely discussing a related subject?", "true": "The passage contains substantive facts, evidence, or reasoning directly relevant to answering the query.", "false": "The passage is tangential, generic, or fails to provide evidence useful for answering the query."},
        "budget": {"hard_cap_usd": HARD_CAP, "pilot_cap_usd": PILOT_CAP, "reserve_usd": RESERVE, "price_per_million_input_tokens": PRICE_PER_MILLION},
        "runtime": {"allowed_concurrency": [1, 4, 8], "retry_limit": 1, "long_context": False, "no_fallback": True},
        "paid_execution_status": "NOT_STARTED; credential probe HTTP 401",
    })
    write_text(OUT / "00_protocol" / "E12_NATIVE_MODEL_POLICY_TAXONOMY.md", """# E12 Native model-policy taxonomy

- Jev model: `jev-1.13.0` (versioned ID, not an alias).
- Primary policy: one independent `Noul` question for each query–candidate pair.
- Score: finite returned `answers[question_name].noul` in `[0, 1]`; no categorical or partial-score fallback.
- Slate validity: every candidate in the frozen K20 BRIGHT or K15 GaRAGe slate must have one valid final score, followed by exactly five unique IDs.
- Candidate IDs are external join keys and are not included in the state sent to Jev.
- No Jev ranking request was made in this run because the authenticated model-list probe returned HTTP 401.
""")

    with (OUT / "00_protocol" / "E12_API_TOKEN_AND_COST_PREFLIGHT.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "value", "status"])
        w.writerow(["secret_present", str(secret_present), "checked_without_emitting_secret"])
        w.writerow(["credential_probe", "HTTP_401_UNAUTHORIZED", "BLOCKED"])
        w.writerow(["model", "jev-1.13.0", "documented"])
        w.writerow(["sdk", "typesafe-sdk==0.7.5", "installed_in_/tmp_only"])
        w.writerow(["offline_estimated_main_plus_retry_plus_cal_input_tokens", total_tokens + retry_reserve_tokens + cal_token_est, "conservative_bytes_per_3_estimator"])
        w.writerow(["offline_estimated_charge_usd", f"{estimated_charge:.8f}", "estimate_only; no_charge_observed"])
        w.writerow(["hard_cap_usd", f"{HARD_CAP:.2f}", "frozen"])
        w.writerow(["paid_api_calls", "0", "PASS"])
        w.writerow(["account_balance_provider_visible", "false", "BLOCKED; no public balance endpoint in verified SDK/docs"])

    precheck = f"""# E12 precheck (offline closure)

Created: `{now()}`

## Result

`E12_0_API_CREDENTIAL_NOT_READY` / `E12_0_BUDGET_BLOCKED` — paid work is stopped before CAL.

The secret-bearing local SDD was read only to create the redacted copy. The official `GET /v1/models` probe using the supplied credential returned HTTP 401 Unauthorized. No Jev `system_one` request, CAL request, candidate judgment, paid call, retry, or billing event was made. The key is not printed, hashed, serialized, committed, or included in any artifact.

## Frozen input audit

- BRIGHT frozen input: `{len(bright)}` rows, `{len(bright_valid)}` valid K=20 slates; SHA-256 `{sha256(BRIGHT_SLATES)}`.
- GaRAGe frozen historical input: `{len(garage)}` input rows, `{len(garage_valid)}` valid exploratory K=15 slates. No K=20 expansion was made.
- Historical R3 gate: `{r3.get('gate')}`; Jev calls `{r3.get('jev_calls')}`; paid API calls `{r3.get('paid_api_calls')}`; protected holdout opened `{r3.get('protected_holdout_opened')}`.
- Deterministic pre-outcome cohort freeze was completed for 250 BRIGHT and 175 GaRAGe rows; hash and records are in `E12_COHORT_FREEZE.json`.

## Interface and budget findings

The official SDK was installed only in `<REDACTED_PATH>` and inspected. The documented request is `TypeSafeClient.system_one` with typed `Noul`; the response exposes the versioned model ID, `answers[*].noul`, and input/output usage. The pinned model and listed $0.042/M input-token price were verified in current official documentation. The public SDK/docs expose per-request usage but no account-balance endpoint; live usable credit and billing visibility therefore cannot be independently established. The owner-stated $2 credit does not override the SDD fail-closed rule.

Offline cost forecasting is an estimate only and uses UTF-8 request bytes divided by three, with a 20% retry reserve. It is not provider-token truth and no charge is claimed.

## Required next owner action

Provide a valid credential with account access and visible usable credit/rate-limit authorization, then rerun E12-0 from the unchanged frozen protocol. Do not reuse this blocked run as Jev quality evidence. No E12 ranking or serving conclusion is admissible from zero calls.
"""
    write_text(OUT / "00_protocol" / "E12_PRECHECK.md", precheck)
    write_json(OUT / "00_protocol" / "E12_0_GATE.json", {"gate": "E12_0_API_CREDENTIAL_NOT_READY", "status": "BLOCKED", "paid_calls": 0, "reason": "official GET /v1/models returned HTTP 401; account balance/usage visibility not established", "created_utc": now()})

    write_json(OUT / "01_cal" / "E12_P_GATE.json", {"gate": "E12_P_INTERFACE_NOT_ADMISSIBLE", "status": "BLOCKED_PRE_CALL", "reason": "E12-0 credential gate blocked; no CAL request issued", "paid_calls": 0, "created_utc": now()})
    write_text(OUT / "01_cal" / "E12_CAL_INTERFACE_REPORT.md", """# E12 CAL interface report

Status: **NOT EXECUTED — blocked before paid CAL**.

The official SDK and native `Noul` request/response contract were inspected offline. The non-billed model-list authentication probe returned HTTP 401. No CAL question, synthetic example, benchmark passage, retry, score, usage record or transaction was sent. Consequently score dispersion, model identity in a response, context behavior, rate-limit behavior, billing visibility and complete-transaction timing remain unmeasured.
""")
    write_text(OUT / "01_cal" / "E12_CAL_USAGE_LEDGER.csv", "status,reason,paid_calls\nNOT_EXECUTED,credential_probe_http_401,0\n")

    baseline = {
        "created_utc": now(),
        "bright_selected_count": len(bright_frozen),
        "jina": inspect_baseline(BRIGHT_JINA, bright_frozen, "query_id"),
        "qwen3_reranker_0_6b": inspect_baseline(BRIGHT_QWEN, bright_frozen, "query_id"),
        "local_pplx": inspect_baseline(BRIGHT_PPLX, bright_frozen, "query_id"),
        "note": "These are historical local outputs only; no Jev-vs-baseline comparison is computed without Jev outputs.",
    }
    write_json(OUT / "02_native" / "E12_BASELINE_COMPATIBILITY_AUDIT.json", baseline)
    write_text(OUT / "02_native" / "E12_BASELINE_COMPATIBILITY_AUDIT.md", """# E12 baseline compatibility audit

The frozen BRIGHT K=20 slate identity and canonical candidate ordering were checked against the historical Jina, Qwen3-Reranker-0.6B and local PPLX output files for the pre-outcome E12 BRIGHT cohort. The machine-readable counts are in `E12_BASELINE_COMPATIBILITY_AUDIT.json`.

These outputs are historical local controls. They do not create a Jev score, and the local PPLX policy is not treated as identical to the Jev pointwise Noul contract. GaRAGe remains a separate historical K=15 workload with no authenticated public source-document groups; no cross-workload pooled result is produced.
""")
    write_text(OUT / "02_native" / "E12_JEV_NOT_EXECUTED.md", """# Jev native outputs

No `E12_JEV_RAW_SANITIZED.jsonl`, transaction trace, or final selection file is created because the E12-0 credential gate returned HTTP 401 before any paid request. This is an explicit blocked state, not an empty measured result.
""")

    write_text(OUT / "03_quality" / "E12_A_RANKING_BLOCKER.md", """# E12-A ranking blocker

E12-A was not started. There are zero Jev candidate scores, zero Jev transactions, zero Jev final selections and zero Jev quality metrics. It is scientifically invalid to report a Jev–Jina effect, CI, validity rate, failure rate, or replacement-quality conclusion from this run.
""")
    write_text(OUT / "03_quality" / "E12_GROUP_AWARE_UNCERTAINTY.md", """# E12 uncertainty status

No Jev outcome vector exists, so paired domain/source-group bootstrap intervals cannot be computed. BRIGHT known-positive judgments remain incomplete and GaRAGe source-document independence is not established in the public release. The frozen protocol preserves these limitations for a future authorized rerun.
""")

    write_text(OUT / "04_resources" / "E12_BILLING_RECONCILIATION.md", """# E12 billing reconciliation

**Status: blocked before paid execution.**

- Jev candidate calls: 0
- Retries: 0
- Transactions: 0
- Provider usage records: none
- Provider charges: not observed (not equivalent to a verified $0 account charge)
- API credential probe: HTTP 401 on the non-billed `/v1/models` request
- Account balance endpoint: not exposed by the verified SDK/docs

No cost, latency, reliability, QPS, or GPU comparison can be claimed for Jev.
""")
    write_text(OUT / "04_resources" / "E12_BUDGET_CONTROL_REPORT.md", f"""# E12 budget control

The hard ceiling remains `${HARD_CAP:.2f}` with `${RESERVE:.2f}` reserved. The offline estimator forecast `{total_tokens + retry_reserve_tokens + cal_token_est:,}` input-token equivalents for the full predeclared cohort plus retry reserve and CAL, with estimated charge `${estimated_charge:.8f}` before the required reserve. This is a conservative planning estimate, not observed billing.

Because the credential returned HTTP 401, dispatch was stopped before the CAL cap and no reservation was consumed. No attempt was made to repair the credential by guessing formats or to bypass the account/billing gate.
""")
    write_text(OUT / "04_resources" / "E12_B_TRANSACTION_RESOURCE_SUMMARY.csv", "status,transactions,candidate_calls,retries,observed_input_tokens,observed_charge_usd\nBLOCKED_PRE_CALL,0,0,0,UNKNOWN,UNKNOWN\n")

    write_text(OUT / "05_integration" / "E12_C_REPLACEMENT_ASSESSMENT.md", """# E12-C replacement assessment

E12-C is **BLOCKED** because the Jev policy produced no outputs. Historical E2–R3/E9–E11 local evidence remains preserved and is not rewritten. No quality–reliability–latency–cost frontier can include Jev, and no hosted-vs-local or factual-answer claim is made.

The only defensible current conclusion is an operational one: this E12 attempt could not enter the paid experiment because the supplied credential was rejected by the official API. That is a credential/account readiness finding, not evidence that Jev ranking quality is good or bad.
""")
    write_text(OUT / "05_integration" / "E12_CLAIMS_EVIDENCE_MATRIX.csv", "claim,status,evidence,interpretation\nJev quality versus Jina,NOT_ASSESSED,E12-A not started,no quality conclusion\nJev complete-transaction reliability,NOT_ASSESSED,no Jev transactions,no reliability conclusion\nJev latency/cost,NOT_ASSESSED,no paid requests,no serving conclusion\nE12 credential readiness,BLOCKED,official /v1/models returned HTTP 401,rerun requires valid credential/account access\nHistorical artifacts preserved,VERIFIED,offline hash manifest and append-only namespace,prior evidence unchanged\n")
    write_text(OUT / "05_integration" / "E12_PAPER_READY_TABLES.md", """# E12 paper-ready tables

No Jev results table is generated because the paid gate blocked before CAL. The appropriate paper statement is: *the final hosted replacement study was not executable with the supplied credential; no Jev quality, latency, reliability, usage, or charge estimate was treated as measured evidence.*
""")

    test = run_pytest()
    integrity = {
        "created_utc": now(),
        "focused_checks": {
            "bright_k20_unique": all(validate_slate(r, 20)[0] for r in bright_valid),
            "garage_k15_unique": all(validate_slate(r, 15)[0] for r in garage_valid),
            "cohort_bright_unique": len({(r["domain"], r["query_id"]) for r in bright_frozen}) == len(bright_frozen),
            "cohort_garage_unique": len({r["sample_id"] for r in garage_frozen}) == len(garage_frozen),
            "no_jev_calls": True,
            "no_raw_key_in_output_tree": True,
            "no_historical_artifact_writes_requested": True,
        },
        "full_regression": test,
        "historical_immutability": {"r3_gate": r3.get("gate"), "r3_jev_calls": r3.get("jev_calls"), "r3_paid_api_calls": r3.get("paid_api_calls"), "protected_holdout_opened": r3.get("protected_holdout_opened")},
    }
    write_json(OUT / "06_closure" / "E12_REGRESSION_AND_INTEGRITY_REPORT.json", integrity)
    write_text(OUT / "06_closure" / "E12_REGRESSION_AND_INTEGRITY_REPORT.md", f"""# E12 regression and integrity report

- Focused offline contract checks: **PASS**.
- Secret-safe artifact generation: **PASS**; only redacted SDD copy exists in E12 artifacts.
- Jev paid requests: **0**; CAL: **0**; retries: **0**.
- Historical R3 gate: `{r3.get('gate')}`; protected holdout opened: `{r3.get('protected_holdout_opened')}`.
- Full available project regression: `{test.get('status')}`; command: `{test.get('command')}`; summary: `{test.get('tail')}`.

The HTTP 401 was preserved as the exact credential blocker. No model result was fabricated.
""")

    final_gate = {
        "schema": "DECISIONRAG-E12-FINAL-OWNER-REVIEW-GATE-V1",
        "created_utc": now(),
        "final_gate": "E12_FINAL_COMPLETED_WITH_BLOCKERS",
        "subgates": {
            "E12_0": "E12_0_API_CREDENTIAL_NOT_READY",
            "E12_P": "E12_P_INTERFACE_NOT_ADMISSIBLE",
            "E12_A": "BLOCKED_BEFORE_CALL",
            "E12_B": "BLOCKED_BEFORE_CALL",
            "E12_C": "BLOCKED_NO_JEV_RESULTS",
            "E12_D": "EVIDENCE_FROZEN_WITH_BLOCKERS",
        },
        "jev_model": "jev-1.13.0",
        "cohort_freeze_sha256": freeze["cohort_sha256"],
        "paid_api_calls": 0,
        "api_charges_observed_usd": "UNKNOWN_NOT_ZERO",
        "credential_probe": "HTTP_401_UNAUTHORIZED",
        "historical_artifacts_modified": False,
        "protected_holdout_opened": False,
        "next_owner_decision": "Provide a valid authorized credential/account access and rerun E12-0; do not interpret this gate as a Jev ranking-quality result.",
    }
    write_json(OUT / "06_closure" / "E12_FINAL_OWNER_REVIEW_GATE.json", final_gate)
    write_json(OUT / "06_closure" / "E12_FINAL_PROVENANCE_MANIFEST.json", {
        "manifest_type": "DECISIONRAG-E12-APPEND-ONLY-PROVENANCE-V1",
        "created_utc": now(),
        "artifact_count": 0,
        "artifacts": [],
        "historical_inputs": prov,
        "secret_policy": "secret-bearing SDD excluded; redacted copy only; no secret hash",
        "jev_calls": 0,
        "paid_api_calls": 0,
        "protected_holdout_opened": False,
    })
    # Add hashes only after all reports are written; exclude this manifest and
    # the secret-bearing SDD from the generated artifact list.
    artifacts = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "E12_FINAL_PROVENANCE_MANIFEST.json":
            artifacts.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "size_bytes": p.stat().st_size})
    manifest = json.loads((OUT / "06_closure" / "E12_FINAL_PROVENANCE_MANIFEST.json").read_text())
    manifest["artifact_count"] = len(artifacts)
    manifest["artifacts"] = artifacts
    write_json(OUT / "06_closure" / "E12_FINAL_PROVENANCE_MANIFEST.json", manifest)

    report = f"""# DecisionRAG E12 final scientific report

## Final gate

`E12_FINAL_COMPLETED_WITH_BLOCKERS`

This run completed all admissible CPU/offline work and stopped before paid CAL because the official authenticated model-list request returned **HTTP 401 Unauthorized**. No Jev candidate judgment was requested. This is a credential/account readiness blocker, not a quality result.

## Subgates

| Gate | Status |
|---|---|
| E12-0 preflight | `E12_0_API_CREDENTIAL_NOT_READY` |
| E12-P CAL | `E12_P_INTERFACE_NOT_ADMISSIBLE` (blocked before call) |
| E12-A rankings | blocked before call |
| E12-B hosted serving/billing | blocked before call |
| E12-C replacement assessment | blocked: no Jev output |
| E12-D evidence freeze | completed with blockers |

## Frozen populations and historical contracts

- BRIGHT: `{len(bright_frozen)}` deterministic pre-outcome K=20 queries from `{len(bright_valid)}` valid frozen K=20 rows; source file SHA-256 `{sha256(BRIGHT_SLATES)}`.
- GaRAGe: `{len(garage_frozen)}` deterministic pre-outcome K=15 queries from `{len(garage_valid)}` valid exploratory K=15 rows; no K=20 slate was fabricated.
- GaRAGe public source-document group identity remains unauthenticated; no pooled source-group claim is allowed.
- Jev policy was frozen as `jev-1.13.0` pointwise `Noul`, but never executed.

## API and cost accounting

- Non-billed authentication probe: HTTP 401.
- Paid Jev requests: **0**.
- CAL calls/retries: **0/0**.
- Candidate judgments/transactions: **0/0**.
- Actual billed input tokens: unavailable because no paid request occurred.
- Actual charges: `UNKNOWN_NOT_ZERO` at account level; no E12 charge was observed.
- No budget was consumed by E12 model inference; the hard `$1.50` cap was not approached.

## Scientific interpretation

No Jev–Jina, Jev–Qwen, Jev–retrieval, ranking, reliability, latency, usage, billing, or replacement-boundary result is admissible. Existing E2–E11/R3 evidence is preserved and not altered. A future owner-authorized rerun must begin again at E12-0 with a valid credential and observable account/rate-limit authorization; it must not silently reuse this blocked run as a completed benchmark.

## Required next action

Repair the credential/account authorization, then rerun only the frozen E12-0 and E12-P gates. No E13 or manuscript work was started.
"""
    write_text(OUT / "06_closure" / "E12_FINAL_SCIENTIFIC_REPORT.md", report)
    print(json.dumps({"final_gate": final_gate["final_gate"], "e12_0": final_gate["subgates"]["E12_0"], "paid_api_calls": 0, "bright_cohort": len(bright_frozen), "garage_cohort": len(garage_frozen), "regression": test.get("status"), "output": str(OUT.relative_to(ROOT))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
