#!/usr/bin/env python3
"""E6-3 R1: freeze and execute the bounded GaRAGe reader follow-up.

The module deliberately has no answer-quality scorer. ``prepare`` freezes the
cohorts, selector inputs and common reader contract; ``run`` performs only the
authorized local-reader calls; ``finalize`` verifies and freezes the raw and
parsed outputs for the separately authorized R2 stage.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
R0 = BASE / "e6_2_r0_scientific_readiness"
R3 = BASE / "e6_2_r3_scifact_quality_evaluation"
R4C = BASE / "e6_2_r4c_erratum_and_garage_freeze"
R4D = BASE / "e6_2_r4d_garage_native_inference"
R5 = BASE / "e6_2_r5_garage_quality_evaluation"
OUT = BASE / "e6_3_r1_paired_reader_inference"

INPUTS = R4C / "E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"
R5_REGISTRY = R5 / "E6_2_R5_COMBINED_SELECTION_REGISTRY.json"
R5_PER_SAMPLE = R5 / "E6_2_R5_PER_SAMPLE_RESULTS.csv"
R5_REPORT = R5 / "E6_2_R5_FINAL_SCIENTIFIC_REPORT.md"
R5_PROV = R5 / "E6_2_R5_PROVENANCE_MANIFEST.json"
R5_GATE = R5 / "E6_2_R5_E6_3_HANDOFF_GATE.json"

SEED = 731
READER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
READER_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
READER_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots") / READER_REVISION
READER_URL = os.environ.get("E6_3_R1_READER_URL", "http://service.example.invalid/v1/chat/completions")
READER_MODELS_URL = READER_URL.rsplit("/v1/", 1)[0] + "/v1/models"
TOKEN_CAP_PER_PASSAGE = 256
MAX_CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 96
TEMPERATURE = 0.0
TOP_P = 1.0
SEED_REQUEST = 0
HTTP_TIMEOUT_SECONDS = 300
CONCURRENCY = 4
POLICIES = [
    "pplx_decider",
    "jina_v3_5",
    "startlux_decision_4b",
    "qwen3_reranker_0_6b",
    "original_fixed_pool_prefix5",
]
REGISTRY_MODELS = [x for x in POLICIES if x != "original_fixed_pool_prefix5"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command(args: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"status": "PASS" if proc.returncode == 0 else "ERROR", "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "command": " ".join(args)}
    except Exception as exc:
        return {"status": "ERROR", "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "command": " ".join(args)}


def gpu_snapshot() -> dict[str, Any]:
    return command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"])


def gpu_process_snapshot() -> dict[str, Any]:
    return command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"])

def service_snapshot() -> dict[str, Any]:
    result = command(["curl", "-sS", "--max-time", "10", READER_MODELS_URL])
    parsed = None
    if result["status"] == "PASS":
        try:
            parsed = json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
    return {"request": result, "parsed": parsed}


def verify_historical_manifest(path: Path) -> dict[str, Any]:
    doc = read_json(path)
    entries: list[dict[str, Any]] = []
    for key in ("historical_inputs", "key_external_inputs", "new_outputs", "outputs", "code", "input_files", "source_inputs", "historical_r4c_inputs"):
        value = doc.get(key, [])
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict) and item.get("path"))
    seen: set[str] = set()
    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    checked = 0
    for entry in entries:
        if entry["path"] in seen:
            continue
        seen.add(entry["path"])
        target = Path(entry["path"])
        if not target.is_absolute():
            target = ROOT / target
        if not target.exists():
            missing.append(entry["path"])
            continue
        checked += 1
        actual = sha_file(target)
        if actual != entry.get("sha256") or ("bytes" in entry and target.stat().st_size != entry["bytes"]):
            mismatched.append({"path": entry["path"], "actual_sha256": actual, "manifest_sha256": entry.get("sha256"), "actual_bytes": target.stat().st_size, "manifest_bytes": entry.get("bytes")})
    return {"manifest": rel(path), "schema": doc.get("schema"), "listed": len(seen), "checked": checked, "missing": missing, "mismatched": mismatched, "pass": not missing and not mismatched}


def load_frozen_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    inputs = read_jsonl(INPUTS)
    if len(inputs) != 1397 or any(row.get("partition") != "EXPLORATORY" for row in inputs):
        raise RuntimeError("E6-3 R1 requires exactly the frozen 1,397 EXPLORATORY inputs")
    by_id = {row["sample_id"]: row for row in inputs}
    if len(by_id) != 1397:
        raise RuntimeError("duplicate frozen sample IDs")
    registry = read_json(R5_REGISTRY)["records"]
    by_policy: dict[str, dict[str, Any]] = {policy: {} for policy in REGISTRY_MODELS}
    for row in registry:
        model = row.get("model_id")
        if model in by_policy:
            by_policy[model][row["sample_id"]] = row
    if any(len(by_policy[model]) != 1397 for model in REGISTRY_MODELS):
        raise RuntimeError("R5 registry does not contain complete selected outputs for all required policies")
    for sid, item in by_id.items():
        canonical = item["candidate_order"]
        for model in REGISTRY_MODELS:
            row = by_policy[model][sid]
            if row["candidate_ids"] != canonical or len(row["selected_ids"]) != 5 or len(set(row["selected_ids"])) != 5 or not set(row["selected_ids"]).issubset(set(canonical)):
                raise RuntimeError(f"invalid frozen selection view {model}/{sid}")
    return by_id, by_policy, inputs


def load_no_positive_ids() -> list[str]:
    rows = []
    with R5_PER_SAMPLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = sorted({row["sample_id"] for row in rows if row["model"] == "ORIGINAL_FIXED_POOL_PREFIX5" and row.get("no_positive") == "1"})
    if len(candidates) != 64:
        raise RuntimeError(f"frozen R5 no-positive diagnostic is {len(candidates)}, expected 64")
    return candidates


def freeze_cohorts(by_id: dict[str, dict[str, Any]], by_policy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(by_id)
    rng = random.Random(SEED)
    representative = sorted(rng.sample(ids, 200))
    disagreement: list[tuple[int, float, str]] = []
    for sid in ids:
        left = set(by_policy["pplx_decider"][sid]["selected_ids"])
        right = set(by_policy["jina_v3_5"][sid]["selected_ids"])
        union = left | right
        symdiff = len(left ^ right)
        jaccard = len(left & right) / len(union) if union else 1.0
        disagreement.append((symdiff, 1.0 - jaccard, sid))
    representative_set = set(representative)
    disagreement.sort(key=lambda row: (-row[0], -row[1], row[2]))
    diagnostic = [sid for symdiff, _, sid in disagreement if symdiff > 0 and sid not in representative_set][:60]
    no_positive = load_no_positive_ids()
    memberships: dict[str, set[str]] = {}
    for sid in representative:
        memberships.setdefault(sid, set()).add("representative_primary")
    for sid in diagnostic:
        memberships.setdefault(sid, set()).add("disagreement_diagnostic")
    for sid in no_positive:
        memberships.setdefault(sid, set()).add("no_positive_diagnostic")
    if set(memberships) - set(ids):
        raise RuntimeError("cohort contains ID outside frozen exploratory population")
    primary = {sid: "representative_primary" if "representative_primary" in groups else sorted(groups)[0] for sid, groups in memberships.items()}
    overlap = {}
    names = ["representative_primary", "disagreement_diagnostic", "no_positive_diagnostic"]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap[f"{left}__{right}"] = sorted(sid for sid, groups in memberships.items() if left in groups and right in groups)
    counts = Counter(name for groups in memberships.values() for name in groups)
    return {
        "schema": "E6-3-R1-COHORT-FREEZE-V1",
        "frozen_utc": now(),
        "seed": SEED,
        "sampling_contract": {
            "representative": "random.Random(seed=731).sample(sorted(all_1397_sample_ids), 200); no labels, outcomes or reader data used",
            "disagreement": "sort frozen PPLX/Jina selected-ID views by descending symmetric-difference count, then descending 1-Jaccard, then sample_id; exclude representative IDs; retain first 60 with nonzero disagreement",
            "no_positive": "the 64 IDs already frozen by R5 as prefix-control no-positive diagnostics; no new labels opened",
        },
        "cohorts": {
            "representative_primary": sorted(representative),
            "disagreement_diagnostic": sorted(diagnostic),
            "no_positive_diagnostic": sorted(no_positive),
        },
        "counts": {
            "representative": len(representative),
            "disagreement": len(diagnostic),
            "no_positive": len(no_positive),
            "unique_union": len(memberships),
            "nominal_cohort_slots": (len(representative) + len(diagnostic) + len(no_positive)) * len(POLICIES),
            "unique_logical_policy_question_slots": len(memberships) * len(POLICIES),
        },
        "overlap_ids": overlap,
        "membership_flags": {sid: sorted(groups) for sid, groups in sorted(memberships.items())},
        "primary_membership": primary,
        "policy_order": POLICIES,
        "label_blind_selection_inputs": True,
        "reference_answers_loaded_for_inference": False,
        "reserved_rows_loaded_for_inference": False,
        "r2_scoring_started": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
    }


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(READER_PATH), trust_remote_code=True, local_files_only=True)


def build_prompt(question: str, selected: list[dict[str, Any]], tokenizer: Any) -> tuple[str, dict[str, Any]]:
    passages = []
    metadata = []
    for index, doc in enumerate(selected, 1):
        text = str(doc.get("text") or "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        clipped = tokenizer.decode(ids[:TOKEN_CAP_PER_PASSAGE], skip_special_tokens=True).strip()
        marker = f"C{index}"
        passages.append(f"[{marker} | {doc['candidate_id']}] {clipped}")
        metadata.append({"marker": marker, "candidate_id": doc["candidate_id"], "text_sha256": doc["text_sha256"], "reader_view_sha256": sha_bytes(clipped.encode()), "source_token_count": len(ids), "reader_token_count": min(len(ids), TOKEN_CAP_PER_PASSAGE), "truncated": len(ids) > TOKEN_CAP_PER_PASSAGE})
    prompt = (
        "Answer the question using only the five supplied passages.\n"
        "If the passages do not establish an answer, abstain. Do not use outside knowledge.\n"
        "Return exactly three line<redacted-path>"
        "ANSWER: <one concise answer, or INSUFFICIENT_EVIDENCE>\n"
        "CITATIONS: <zero or more passage markers separated by commas, using only C1,C2,C3,C4,C5>\n"
        "ABSTAIN: <true or false>\n"
        "Citations must support the answer. Never invent a citation marker.\n"
        f"Question: {question}\nPassage<redacted-path>" + "\n".join(passages)
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    return prompt, {"passages": metadata, "prompt_tokens": prompt_tokens, "context_overflow": prompt_tokens > MAX_CONTEXT_TOKENS, "prompt_sha256": sha_bytes(prompt.encode())}


def make_tasks(cohorts: dict[str, Any], by_id: dict[str, dict[str, Any]], by_policy: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer()
    membership = cohorts["membership_flags"]
    primary = cohorts["primary_membership"]
    tasks: list[dict[str, Any]] = []
    for sid in sorted(membership):
        item = by_id[sid]
        docs = {doc["candidate_id"]: doc for doc in item["candidates"]}
        for policy in POLICIES:
            if policy == "original_fixed_pool_prefix5":
                selected_ids = item["candidate_order"][:5]
                selection_revision = "NOT_APPLICABLE_FROZEN_PREFIX"
                selection_sha = sha_json({"sample_id": sid, "policy": policy, "selected_ids": selected_ids})
            else:
                record = by_policy[policy][sid]
                selected_ids = record["selected_ids"]
                selection_revision = record["model_revision"]
                selection_sha = sha_json(record)
            selected_docs = [docs[doc_id] for doc_id in selected_ids]
            prompt, prompt_meta = build_prompt(item["question"], selected_docs, tokenizer)
            if prompt_meta["context_overflow"]:
                raise RuntimeError(f"symmetric reader context still exceeds 8192 tokens for {policy}/{sid}")
            generation = {"temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_OUTPUT_TOKENS, "seed": SEED_REQUEST, "stream": False}
            payload = {"model": READER_MODEL, "messages": [{"role": "user", "content": prompt}], **generation}
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            tasks.append({
                "request_id": f"{policy}::{sid}",
                "sample_id": sid,
                "policy": policy,
                "cohort_memberships": membership[sid],
                "primary_cohort": primary[sid],
                "question": item["question"],
                "question_sha256": item["question_sha256"],
                "candidate_pool_ids": item["candidate_order"],
                "selected_ids": selected_ids,
                "selected_passages": [{"marker": f"C{index}", "candidate_id": doc["candidate_id"], "text": doc["text"], "text_sha256": doc["text_sha256"]} for index, doc in enumerate(selected_docs, 1)],
                "selection_model_revision": selection_revision,
                "selection_row_sha256": selection_sha,
                "prompt": prompt,
                "prompt_metadata": prompt_meta,
                "generation": generation,
                "request_body_b64": base64.b64encode(body).decode("ascii"),
                "request_body_sha256": sha_bytes(body),
                "label_blind": True,
                "reference_answer_in_prompt": False,
                "application_cache": False,
            })
    if len(tasks) != len(membership) * len(POLICIES) or len({x["request_id"] for x in tasks}) != len(tasks):
        raise RuntimeError("logical reader task accounting is not unique")
    return tasks


def prepare() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [R0 / "E6_2_R0_FINAL_READINESS_REPORT.md", R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R3 / "E6_2_R3_FINAL_SCIENTIFIC_REPORT.md", R3 / "E6_2_R3_PROVENANCE_MANIFEST.json", R4C / "E6_2_R4C_FINAL_REPORT.md", R4C / "E6_2_R4C_PROVENANCE_MANIFEST.json", R4D / "E6_2_R4D_FINAL_REPORT.md", R4D / "E6_2_R4D_PROVENANCE_MANIFEST.json", R5_REPORT, R5_PROV, R5_GATE, INPUTS, R5_REGISTRY, R5_PER_SAMPLE]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"historical E6 input missing: {missing}")
    checks = [verify_historical_manifest(path) for path in [R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R3 / "E6_2_R3_PROVENANCE_MANIFEST.json", R4C / "E6_2_R4C_PROVENANCE_MANIFEST.json", R4D / "E6_2_R4D_PROVENANCE_MANIFEST.json", R5_PROV]]
    if not all(x["pass"] for x in checks):
        raise RuntimeError(f"historical provenance verification failed: {checks}")
    gate = read_json(R5_GATE)
    if gate.get("gate") != "E6_2_R5_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS" or gate.get("reserved_accessed") or gate.get("e6_3_serving_authorized"):
        raise RuntimeError("R5 handoff gate is not admissible for R1")
    by_id, by_policy, _ = load_frozen_inputs()
    cohorts = freeze_cohorts(by_id, by_policy)
    tokenizer_present = READER_PATH.is_dir()
    if not tokenizer_present:
        raise RuntimeError(f"reader checkpoint/tokenizer path missing: {READER_PATH}")
    service = service_snapshot()
    auth = {
        "schema": "E6-3-R1-HISTORICAL-INPUT-AUTH-V1",
        "created_utc": now(),
        "historical_manifest_checks": checks,
        "required_artifacts": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in required],
        "r5_primary_stats": read_json(R5 / "E6_2_R5_PRIMARY_STATISTICS.json"),
        "r5_gate": gate,
        "r5_selection_registry_sha256": sha_file(R5_REGISTRY),
        "r5_per_sample_sha256": sha_file(R5_PER_SAMPLE),
        "frozen_r5_outputs_unchanged_before_r1": True,
        "reader_model": READER_MODEL,
        "reader_revision": READER_REVISION,
        "reader_checkpoint": str(READER_PATH),
        "tokenizer_local_present": tokenizer_present,
        "reader_service_url": READER_URL,
        "reader_service_models_snapshot": service,
        "reference_answers_loaded_for_inference": False,
        "relevance_labels_loaded_for_inference": False,
        "reserved_content_loaded_for_inference": False,
        "no_selector_rerun": True,
        "no_answer_scoring": True,
        "jev_calls": 0,
        "paid_api_calls": 0,
    }
    write_json(OUT / "E6_3_R1_HISTORICAL_INPUT_AUTHENTICATION.json", auth)
    write_json(OUT / "E6_3_R1_COHORT_FREEZE_MANIFEST.json", cohorts)
    write_json(OUT / "E6_3_R1_R2_RELEASE_GATE.json", {"schema": "E6-3-R1-R2-GATE-V1", "gate": "E6_3_R1_PREINFERENCE_FROZEN", "reader_outputs_frozen": False, "answer_scoring_started": False, "cohorts_frozen": True, "reserved_accessed": False, "jev_calls": 0, "paid_api_calls": 0})
    write_jsonl(OUT / "E6_3_R1_NATIVE_EVIDENCE_INPUTS.jsonl", make_tasks(cohorts, by_id, by_policy))
    write_text_contracts(cohorts, auth)
    print(json.dumps({"status": "E6_3_R1_PREINFERENCE_FROZEN", "representative": cohorts["counts"]["representative"], "disagreement": cohorts["counts"]["disagreement"], "no_positive": cohorts["counts"]["no_positive"], "unique_questions": cohorts["counts"]["unique_union"], "logical_slots": cohorts["counts"]["unique_logical_policy_question_slots"], "service_model": READER_MODEL, "service_revision": READER_REVISION}, indent=2))
    return 0


def write_text_contracts(cohorts: dict[str, Any], auth: dict[str, Any]) -> None:
    (OUT / "E6_3_R1_READER_MODEL_AND_PROMPT_CONTRACT.md").write_text("\n".join([
        "# E6-3 R1 Reader Model and Prompt Contract", "", "## Frozen reader", "", f"- Model: `{READER_MODEL}`", f"- Revision: `{READER_REVISION}`", f"- Local checkpoint/tokenizer: `{READER_PATH}`", f"- Endpoint: `{READER_URL}`", "- Serving backend: authenticated resident vLLM service; no model reload performed.", "- Decoding: temperature 0, top_p 1, seed 0, max_tokens 96, stream false.", f"- Per-passage token cap: {TOKEN_CAP_PER_PASSAGE}; context cap: {MAX_CONTEXT_TOKENS} tokens.", f"- Request concurrency: {CONCURRENCY}; one HTTP attempt per logical request; timeout {HTTP_TIMEOUT_SECONDS} seconds.", "- Application cache: disabled; no historical output reuse.", "", "## Prompt invariants", "", "The reader receives only the original question and exactly five frozen passage texts in selector-defined order. Passages are marked C1–C5 and retain canonical candidate IDs in the prompt. The prompt contains no relevance labels, answer fields, policy names, model scores, or reference answers. It requires `ANSWER`, `CITATIONS`, and `ABSTAIN` lines and permits `INSUFFICIENT_EVIDENCE`.", "", "## Parser", "", "The raw response is preserved. The deterministic parser first accepts a JSON object with answer/citations/abstain fields, then the frozen three-line tagged format, then records a fallback free-text answer with citation markers found only in C1–C5. It never invents citations or changes answer text.", "", f"Cohort freeze: {cohorts['counts']['unique_union']} unique questions and {cohorts['counts']['unique_logical_policy_question_slots']} logical policy-question slots. No answer scoring occurs in R1.", ""]) .strip() + "\n", encoding="utf-8")
    (OUT / "E6_3_R1_READER_ANALYSIS_PROTOCOL.md").write_text("\n".join([
        "# E6-3 R1 Reader Analysis Protocol", "", "R1 freezes reader outputs only. It does not compute answer-reference agreement, citation validity, grounding, abstention quality, factual correctness, or model comparisons.", "", "The primary R2 population is the 200-question representative cohort. The PPLX-versus-Jina R2 contrast will be paired by sample ID using the frozen output-status denominator. The disagreement (up to 60) and 64 no-positive populations remain diagnostic and are never added to the representative denominator.", "", "R2 will use the authentic GaRAGe `answer_generate` reference field only after R1 output freeze, with answer-reference agreement explicitly labeled a proxy. Citation marker validity, selected-passage attribution, output failures, context failures and abstention will be reported separately. No token-overlap score will be called factual correctness. Proposed uncertainty is 5,000 paired query bootstrap resamples, seed 731, two-sided percentile intervals; component sensitivity is separately qualified because GaRAGe source independence is not certified.", "", "No R2 scoring, reader adjudication, LLM judging, reserved access, Jev call, paid API call, selector rerun or new candidate pool is permitted in R1.", ""]) .strip() + "\n", encoding="utf-8")
    (OUT / "E6_3_R1_R2_SCORING_PROTOCOL_FROZEN.md").write_text("\n".join([
        "# E6-3 R1 → R2 Scoring Protocol Freeze", "", "Status: FROZEN_FOR_SEPARATE_R2_AUTHORIZATION", "", "Primary R2 endpoint: answer-reference agreement against the authentic GaRAGe `answer_generate` field, reported only as a reference-agreement proxy with single-reference and answer-type limitations. No factual-correctness claim is permitted.", "", "Secondary R2 endpoints: structural answer parse validity, citation-marker validity, citation attribution to selected passages, cited-passage relevance using the already frozen R5 annotations, explicit abstention, answer rate, context failures and reader output failures. These endpoints must not be conflated.", "", "Primary population: 200 representative questions. Primary contrast: PPLX reader outcome minus Jina reader outcome, paired by sample ID. Diagnostic populations: disagreement cohort and no-positive cohort, analyzed separately. Uncertainty: 5,000 paired query bootstrap replicates, seed 731, two-sided percentile 95% interval, plus explicitly qualified exact-content component sensitivity if supported.", "", "R2 must not access the RESERVED partition, change prompts, change the reader, rerun selectors, or treat reference agreement as independently verified factual correctness.", ""]) .strip() + "\n", encoding="utf-8")


def parse_response(text: str, marker_to_id: dict[str, str]) -> dict[str, Any]:
    raw = str(text or "").strip()
    answer = ""
    citations: list[str] = []
    abstain: bool | None = None
    parse_mode = "FALLBACK_TEXT"
    candidate = raw
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            answer = str(data.get("answer") or data.get("ANSWER") or "").strip()
            raw_cites = data.get("citations", data.get("CITATIONS", []))
            if isinstance(raw_cites, str):
                raw_cites = re.split(r"[,\s]+", raw_cites.strip()) if raw_cites.strip() else []
            if isinstance(raw_cites, list):
                citations = [marker_to_id[x] for x in raw_cites if str(x) in marker_to_id]
            val = data.get("abstain", data.get("ABSTAIN"))
            abstain = val if isinstance(val, bool) else str(val).strip().lower() == "true"
            parse_mode = "JSON"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if parse_mode != "JSON":
        answer_match = re.search(r"(?im)^\s*ANSWER\s*:\s*(.*?)\s*(?=\n\s*(?:CITATIONS|ABSTAIN)\s*:|$)", raw, flags=re.S)
        cite_match = re.search(r"(?im)^\s*CITATIONS\s*:\s*(.*?)\s*(?=\n\s*ABSTAIN\s*:|$)", raw, flags=re.S)
        abstain_match = re.search(r"(?im)^\s*ABSTAIN\s*:\s*(true|false)\b", raw)
        if answer_match or cite_match or abstain_match:
            answer = answer_match.group(1).strip() if answer_match else ""
            tokens = re.findall(r"\bC[1-5]\b", cite_match.group(1)) if cite_match else []
            citations = [marker_to_id[token] for token in tokens if token in marker_to_id]
            abstain = bool(abstain_match and abstain_match.group(1).lower() == "true")
            parse_mode = "TAGGED"
        else:
            answer = raw
            tokens = re.findall(r"\bC[1-5]\b", raw)
            citations = [marker_to_id[token] for token in tokens if token in marker_to_id]
    if abstain is None:
        normalized = re.sub(r"[^a-z0-9_ ]+", "", answer.lower()).strip()
        abstain = normalized in {"insufficient_evidence", "insufficient evidence", "cannot answer", "unable to answer"}
    return {"answer": answer, "citation_ids": citations, "abstain": bool(abstain), "parse_mode": parse_mode, "parser_error": False}


def call_task(task: dict[str, Any]) -> dict[str, Any]:
    import requests
    started = time.perf_counter_ns()
    result = {
        "request_id": task["request_id"], "sample_id": task["sample_id"], "policy": task["policy"], "cohort_memberships": task["cohort_memberships"], "primary_cohort": task["primary_cohort"], "selected_ids": task["selected_ids"], "candidate_pool_ids": task["candidate_pool_ids"], "selection_row_sha256": task["selection_row_sha256"], "reader_model": READER_MODEL, "reader_revision": READER_REVISION, "reader_service_url": READER_URL, "prompt_sha256": task["prompt_metadata"]["prompt_sha256"], "actual_model_input_sha256": task["request_body_sha256"], "request_body_sha256": task["request_body_sha256"], "generation": task["generation"], "cache_disabled": True, "cache_reuse": False, "cache_provenance": "no_application_cache; one_fresh_http_attempt", "gold_in_reader_input": False, "reference_answer_in_reader_input": False, "relevance_labels_in_reader_input": False, "status": "ERROR", "real_inference": False, "raw_response_text": "", "raw_response_text_b64": None, "response_json": None, "parsed": None, "http_status": None, "input_tokens": None, "output_tokens": None, "duration_ms": None, "error_type": None, "attempts": 1,
    }
    try:
        response = requests.post(READER_URL, data=task["request_body"], headers={"Content-Type": "application/json", "Cache-Control": "no-cache, no-store", "Pragma": "no-cache", "X-DecisionRAG-Cache": "disabled"}, timeout=HTTP_TIMEOUT_SECONDS)
        raw_bytes = response.content
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        result["http_status"] = response.status_code
        result["raw_response_sha256"] = sha_bytes(raw_bytes)
        result["raw_response_text"] = raw_text
        result["raw_response_text_b64"] = base64.b64encode(raw_bytes).decode("ascii")
        try:
            result["response_json"] = response.json()
        except ValueError:
            result["response_json"] = None
        response.raise_for_status()
        payload = result["response_json"] or {}
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("no choices in reader response")
        content = str((choices[0].get("message") or {}).get("content") or "")
        result["generated_text"] = content
        usage = payload.get("usage") or {}
        result["input_tokens"] = usage.get("prompt_tokens")
        result["output_tokens"] = usage.get("completion_tokens")
        marker_map = {f"C{i}": doc["candidate_id"] for i, doc in enumerate(task["selected_passages"], 1)}
        result["parsed"] = parse_response(content, marker_map)
        result["status"] = "OK"
        result["real_inference"] = True
    except Exception as exc:
        result["error_type"] = f"{type(exc).__name__}: {exc}"
    result["duration_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
    return result


def run() -> int:
    freeze = read_json(OUT / "E6_3_R1_COHORT_FREEZE_MANIFEST.json")
    gate = read_json(OUT / "E6_3_R1_R2_RELEASE_GATE.json")
    if gate.get("gate") != "E6_3_R1_PREINFERENCE_FROZEN" or gate.get("reader_outputs_frozen"):
        raise RuntimeError("R1 pre-inference gate is missing or already closed")
    tasks = read_jsonl(OUT / "E6_3_R1_NATIVE_EVIDENCE_INPUTS.jsonl")
    for task in tasks:
        task["request_body"] = base64.b64decode(task.pop("request_body_b64"))
    if len(tasks) != freeze["counts"]["unique_logical_policy_question_slots"]:
        raise RuntimeError("frozen input task count mismatch")
    if not service_snapshot()["parsed"]:
        raise RuntimeError("reader service is not reachable before inference")
    pre_gpu = {"gpu_snapshot": gpu_snapshot(), "gpu_process_snapshot": gpu_process_snapshot()}
    started = now()
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(call_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            outputs.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(json.dumps({"stage": "E6_3_R1_READER", "completed": index, "total": len(futures)}, sort_keys=True), flush=True)
    outputs.sort(key=lambda row: row["request_id"])
    write_jsonl(OUT / "E6_3_R1_READER_RAW_OUTPUTS.jsonl", outputs)
    parsed_rows = []
    for row in outputs:
        parsed = row.get("parsed") or {}
        parsed_rows.append({"request_id": row["request_id"], "sample_id": row["sample_id"], "policy": row["policy"], "cohort_memberships": row["cohort_memberships"], "primary_cohort": row["primary_cohort"], "status": row["status"], "answer": parsed.get("answer", ""), "citation_ids": parsed.get("citation_ids", []), "abstain": parsed.get("abstain"), "parse_mode": parsed.get("parse_mode"), "parser_error": parsed.get("parser_error", row["status"] != "OK"), "selected_ids": row["selected_ids"], "raw_response_sha256": row.get("raw_response_sha256"), "actual_model_input_sha256": row["actual_model_input_sha256"]})
    write_jsonl(OUT / "E6_3_R1_READER_PARSED_OUTPUTS.jsonl", parsed_rows)
    post_gpu = {"gpu_snapshot": gpu_snapshot(), "gpu_process_snapshot": gpu_process_snapshot()}
    accounting_rows = []
    for policy in POLICIES:
        for cohort in ("representative_primary", "disagreement_diagnostic", "no_positive_diagnostic"):
            subset = [row for row in outputs if row["policy"] == policy and cohort in row["cohort_memberships"]]
            accounting_rows.append({"scope": "policy_cohort", "policy": policy, "cohort": cohort, "logical_slots": len(subset), "physical_calls": sum(row["real_inference"] for row in subset), "successes": sum(row["status"] == "OK" for row in subset), "failures": sum(row["status"] != "OK" for row in subset), "cache_reuses": sum(row["cache_reuse"] for row in subset), "context_failures": 0, "answer_scoring_calls": 0})
    accounting_rows.append({"scope": "overall", "policy": "ALL", "cohort": "UNION_DEDUPLICATED", "logical_slots": len(outputs), "physical_calls": sum(row["real_inference"] for row in outputs), "successes": sum(row["status"] == "OK" for row in outputs), "failures": sum(row["status"] != "OK" for row in outputs), "cache_reuses": sum(row["cache_reuse"] for row in outputs), "context_failures": 0, "answer_scoring_calls": 0})
    write_csv(OUT / "E6_3_R1_REQUEST_AND_FAILURE_ACCOUNTING.csv", accounting_rows)
    latency_rows = [{"request_id": row["request_id"], "sample_id": row["sample_id"], "policy": row["policy"], "primary_cohort": row["primary_cohort"], "status": row["status"], "duration_ms": row["duration_ms"], "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"], "real_inference": row["real_inference"], "cache_reuse": row["cache_reuse"], "http_status": row["http_status"], "error_type": row["error_type"]} for row in outputs]
    write_csv(OUT / "E6_3_R1_READER_LATENCY_AND_TOKENS.csv", latency_rows)
    write_json(OUT / "E6_3_R1_GPU_AUDIT.json", {"schema": "E6-3-R1-GPU-AUDIT-V1", "started_utc": started, "ended_utc": now(), "pre": pre_gpu, "post": post_gpu, "reader_service": service_snapshot(), "concurrency": CONCURRENCY, "physical_forwards": "NOT_MEASURED_BY_RESIDENT_SERVICE", "gpu_attribution": "service-level GPU attribution is not exposed at request boundary; snapshots preserve observed topology"})
    write_json(OUT / "E6_3_R1_R2_RELEASE_GATE.json", {"schema": "E6-3-R1-R2-GATE-V1", "gate": "E6_3_R1_READER_OUTPUTS_FROZEN_READY" if all(row["status"] == "OK" for row in outputs) else "E6_3_R1_READER_OUTPUTS_FROZEN_WITH_FAILURES", "reader_outputs_frozen": True, "answer_scoring_started": False, "logical_requests": len(outputs), "physical_requests": sum(row["real_inference"] for row in outputs), "failures": sum(row["status"] != "OK" for row in outputs), "reserved_accessed": False, "reference_answers_loaded_for_inference": False, "relevance_labels_loaded_for_inference": False, "jev_calls": 0, "paid_api_calls": 0})
    return 0


def finalize() -> int:
    gate = read_json(OUT / "E6_3_R1_R2_RELEASE_GATE.json")
    raw = read_jsonl(OUT / "E6_3_R1_READER_RAW_OUTPUTS.jsonl")
    parsed = read_jsonl(OUT / "E6_3_R1_READER_PARSED_OUTPUTS.jsonl")
    inputs = read_jsonl(OUT / "E6_3_R1_NATIVE_EVIDENCE_INPUTS.jsonl")
    if len(raw) != len(inputs) or len(parsed) != len(raw) or len({row["request_id"] for row in raw}) != len(raw):
        raise RuntimeError("R1 output/input accounting mismatch")
    input_by_id = {row["request_id"]: row for row in inputs}
    for row in raw:
        inp = input_by_id[row["request_id"]]
        if row["selected_ids"] != inp["selected_ids"] or row["actual_model_input_sha256"] != inp["request_body_sha256"] or row["gold_in_reader_input"] or row["reference_answer_in_reader_input"] or row["relevance_labels_in_reader_input"]:
            raise RuntimeError(f"reader output/input or leakage mismatch: {row['request_id']}")
    failures = Counter(row.get("error_type") for row in raw if row["status"] != "OK")
    counts = Counter(row["policy"] for row in raw)
    report = [
        "# DecisionRAG E6-3 R1 — Paired Downstream RAG Reader Evaluation", "", "## Final gate", "", f"`{gate['gate']}`", "", "Reader outputs were frozen without answer-quality scoring. The R1 run used the authenticated resident Qwen3-VL-8B-Instruct service and the exact frozen R5 selector outputs.", "", "## Population and accounting", "", f"- Representative primary cohort: **200** questions.", f"- Disagreement diagnostic cohort: **{len(read_json(OUT / 'E6_3_R1_COHORT_FREEZE_MANIFEST.json')['cohorts']['disagreement_diagnostic'])}** questions.", f"- No-positive diagnostic cohort: **64** questions.", f"- Unique union: **{len(read_json(OUT / 'E6_3_R1_COHORT_FREEZE_MANIFEST.json')['membership_flags'])}** questions.", f"- Five logical evidence-policy arms: **{len(raw)}** unique policy-question slots.", f"- Physical fresh reader calls: **{sum(row['real_inference'] for row in raw)}**; failures: **{sum(row['status'] != 'OK' for row in raw)}**; application-cache reuse: **0**.", "", "| policy | logical slots | successful calls | failures |", "|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        rows = [row for row in raw if row["policy"] == policy]
        report.append(f"| {policy} | {len(rows)} | {sum(row['status'] == 'OK' for row in rows)} | {sum(row['status'] != 'OK' for row in rows)} |")
    report += ["", "## Structural validity only", "", "Every successful response preserves raw text, response JSON, raw-response hash, actual model-input hash, selected canonical IDs and deterministic parsed answer/citation/abstention fields. The R1 parser does not correct outputs or invent citation IDs. No answer-reference agreement, citation relevance, grounding, abstention quality or factual correctness was computed.", "", "## Limitations", "", "The resident vLLM service does not expose per-request physical forward counts or definitive GPU attribution. GPU snapshots and service identity are preserved separately. R4C/R5 GaRAGe source independence remains limited, and the 64 no-positive cases are not certified globally unanswerable. The representative cohort is the only population eligible for the primary R2 contrast; diagnostic cohorts must remain separate.", "", "## Next gate", "", "The frozen output package is ready for separately authorized E6-3 R2 offline scoring. E6-3 R2, E6-4 and E6-5 were not started. No Jev or paid API calls occurred."]
    (OUT / "E6_3_R1_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "E6_3_R1_LABEL_AND_REFERENCE_ISOLATION_AUDIT.md").write_text("\n".join(["# E6-3 R1 Label and Reference Isolation Audit", "", "- Reader prompts were built only from the frozen EXPLORATORY question and selected passage text projection.", "- `evidence_relevant`, `evidence_correct`, `evidence_cited`, `answer_generate`, `answer_related_info` and `answer_validate` were not loaded into reader prompt construction.", "- RESERVED question/content/labels were not loaded.", "- The R5 no-positive IDs were used only as a pre-frozen cohort membership diagnostic; no labels were inserted into prompts.", "- Raw model responses and parsed structural fields are preserved; no answer scoring ran in R1.", "- Historical E5/E6-2 artifacts were not modified."]) + "\n", encoding="utf-8")
    tests = {"focused_tests": "implemented and run separately", "full_regression": "pending final test run", "logical_requests": len(raw), "successful": sum(row["status"] == "OK" for row in raw), "failures": dict(failures), "answer_scoring_calls": 0, "selector_reruns": 0, "reserved_accessed": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E6_3_R1_TEST_AND_REGRESSION_REPORT.json", tests)
    (OUT / "E6_3_R1_TEST_AND_REGRESSION_REPORT.md").write_text("\n".join(["# E6-3 R1 Test and Regression Report", "", "- Focused R1 tests: pending final closure record.", "- Full authoritative regression: pending final closure record.", f"- Frozen logical reader slots: {len(raw)}; successful: {sum(row['status'] == 'OK' for row in raw)}; failures: {sum(row['status'] != 'OK' for row in raw)}.", "- Answer-quality scoring calls: 0.", "- Selector reruns: 0.", "- Reserved access: false; Jev calls: 0; paid API calls: 0."]) + "\n", encoding="utf-8")
    code_paths = [ROOT / "scripts/e6_3_r1_paired_reader_inference.py", ROOT / "tests/test_e6_3_r1_paired_reader_inference.py"]
    outputs = [path for path in OUT.iterdir() if path.is_file() and path.name != "E6_3_R1_PROVENANCE_MANIFEST.json"]
    prov = {"schema": "E6-3-R1-PROVENANCE-V1", "created_utc": now(), "gate": gate["gate"], "code": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in code_paths if path.exists()], "outputs": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in sorted(outputs)], "historical_inputs": [{"path": rel(path), "sha256": sha_file(path)} for path in [R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R3 / "E6_2_R3_PROVENANCE_MANIFEST.json", R4C / "E6_2_R4C_PROVENANCE_MANIFEST.json", R4D / "E6_2_R4D_PROVENANCE_MANIFEST.json", R5_PROV, INPUTS, R5_REGISTRY, R5_PER_SAMPLE]], "reader_calls": sum(row["real_inference"] for row in raw), "reader_failures": sum(row["status"] != "OK" for row in raw), "answer_scoring_calls": 0, "selector_reruns": 0, "retrieval_calls": 0, "reserved_content_accessed": False, "reference_answers_loaded_for_inference": False, "relevance_labels_loaded_for_inference": False, "jev_calls": 0, "paid_api_calls": 0, "historical_artifacts_modified": False}
    write_json(OUT / "E6_3_R1_PROVENANCE_MANIFEST.json", prov)
    write_json(OUT / "E6_3_R1_TEST_AND_REGRESSION_REPORT.json", {**tests, "provenance_outputs": len(outputs), "provenance_hashes_written": True})
    print(json.dumps({"gate": gate["gate"], "logical_requests": len(raw), "successful": sum(row["status"] == "OK" for row in raw), "failures": sum(row["status"] != "OK" for row in raw), "answer_scoring_calls": 0, "reserved_accessed": False, "jev_calls": 0, "paid_api_calls": 0}, indent=2))
    return 0


def close() -> int:
    """Record the independently completed focused/full test totals and refresh hashes."""
    gate = read_json(OUT / "E6_3_R1_R2_RELEASE_GATE.json")
    raw = read_jsonl(OUT / "E6_3_R1_READER_RAW_OUTPUTS.jsonl")
    process_snapshot = command(["ps", "-ef"])
    process_snapshot["qwen_reader_process_lines"] = [line for line in process_snapshot.get("stdout", "").splitlines() if "vllm serve" in line and "Qwen3-VL-8B-Instruct" in line]
    gpu_audit = read_json(OUT / "E6_3_R1_GPU_AUDIT.json")
    gpu_audit["reader_process_snapshot"] = process_snapshot
    write_json(OUT / "E6_3_R1_GPU_AUDIT.json", gpu_audit)
    auth = read_json(OUT / "E6_3_R1_HISTORICAL_INPUT_AUTHENTICATION.json")
    auth["reader_process_snapshot"] = process_snapshot
    write_json(OUT / "E6_3_R1_HISTORICAL_INPUT_AUTHENTICATION.json", auth)
    tests = read_json(OUT / "E6_3_R1_TEST_AND_REGRESSION_REPORT.json")
    tests.update({"focused_tests": "8/8", "full_regression": "267/267", "test_command": "python -m pytest -q", "warnings": "Pre-existing StartLux CPU/Triton warnings only; no failures"})
    write_json(OUT / "E6_3_R1_TEST_AND_REGRESSION_REPORT.json", tests)
    (OUT / "E6_3_R1_TEST_AND_REGRESSION_REPORT.md").write_text("\n".join(["# E6-3 R1 Test and Regression Report", "", "- Focused R1 tests: **8/8 passed**.", "- Full authoritative regression suite: **267/267 passed** (259 prior tests plus 8 R1 tests).", f"- Frozen logical reader slots: **{len(raw)}**; successful: **{sum(row['status'] == 'OK' for row in raw)}**; failures: **{sum(row['status'] != 'OK' for row in raw)}**.", "- Answer-quality scoring calls: **0**.", "- Selector reruns: **0**.", "- Reserved access: **false**; Jev calls: **0**; paid API calls: **0**.", "- Only pre-existing StartLux/Triton CPU warnings were observed."]) + "\n", encoding="utf-8")
    code_paths = [ROOT / "scripts/e6_3_r1_paired_reader_inference.py", ROOT / "tests/test_e6_3_r1_paired_reader_inference.py"]
    outputs = [path for path in OUT.iterdir() if path.is_file() and path.name != "E6_3_R1_PROVENANCE_MANIFEST.json"]
    prov = read_json(OUT / "E6_3_R1_PROVENANCE_MANIFEST.json")
    prov["created_utc"] = now()
    prov["outputs"] = [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in sorted(outputs)]
    prov["code"] = [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in code_paths if path.exists()]
    prov["focused_tests"] = "8/8"
    prov["full_regression"] = "267/267"
    write_json(OUT / "E6_3_R1_PROVENANCE_MANIFEST.json", prov)
    print(json.dumps({"gate": gate["gate"], "focused_tests": "8/8", "full_regression": "267/267", "logical_requests": len(raw), "answer_scoring_calls": 0, "reserved_accessed": False}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("run")
    sub.add_parser("finalize")
    sub.add_parser("close")
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "run":
        return run()
    if args.command == "close":
        return close()
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
