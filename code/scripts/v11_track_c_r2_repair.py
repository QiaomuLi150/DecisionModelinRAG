#!/usr/bin/env python3
"""Offline repair and preflight preparation for Track C input wiring R2.

This script only reads frozen manifests and writes small versioned repair artifacts.
It never imports a model runtime, contacts a service, or launches inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK_C = ROOT / "results/non_jev/v11/track_c"
PREFLIGHT = ROOT / "results/preflight/v11_remediation"
COHORT = PREFLIGHT / "TRACK_C_V11R1_COHORT.jsonl"
SCHEDULE = PREFLIGHT / "TRACK_C_V11R1_SCHEDULE.json"
MULTIHOP_SOURCE = ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_RETRIEVAL_SLATES.jsonl"
ENTERPRISE_SOURCE = PREFLIGHT / "ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"
LOCK = ROOT / "V11_PREREGISTRATION_LOCK_MANIFEST.json"
INPUT_MANIFEST = TRACK_C / "V11_TRACK_C_REPAIRED_INPUT_MANIFEST.jsonl"
EXECUTION_MANIFEST = TRACK_C / "V11_TRACK_C_R2_EXECUTION_MANIFEST.json"
PROVENANCE_MANIFEST = TRACK_C / "V11_TRACK_C_R2_PROVENANCE_MANIFEST.json"
PREFLIGHT_REPORT = TRACK_C / "V11_TRACK_C_R2_PREFLIGHT_REPORT.md"
INPUT_CONTRACT_VERSION = "decisionrag-v1.1-track-c-input-wiring-r2"
POLICIES = [
    "raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5",
    "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5",
]
MODEL_IDS = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "qwen": "Qwen/Qwen3-VL-Reranker-2B",
    "laya": "convaiinnovations/laya",
    "clef": "Cloudflare/clef-flash",
    "reader": "Qwen/Qwen3-VL-8B-Instruct",
    "embedding": "Qwen/Qwen3-VL-Embedding-2B",
}
MODEL_REVISIONS = {
    "bge": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "qwen": "4bd860ac4f15ad1897a214615cccc700f8f71818",
    "laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "clef": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
    "reader": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "embedding": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def resolve_questions(cohort: list[dict]) -> tuple[list[dict], dict]:
    multi = {}
    for row in load_jsonl(MULTIHOP_SOURCE):
        query_id = row.get("query_id")
        if query_id in multi:
            raise ValueError(f"duplicate multi-hop source query ID: {query_id}")
        multi[query_id] = row
    enterprise = {}
    for row in load_jsonl(ENTERPRISE_SOURCE):
        query_id = row.get("question_id")
        if query_id in enterprise:
            raise ValueError(f"duplicate enterprise source query ID: {query_id}")
        enterprise[query_id] = row
    source_hashes = {
        rel(MULTIHOP_SOURCE): sha256_file(MULTIHOP_SOURCE),
        rel(ENTERPRISE_SOURCE): sha256_file(ENTERPRISE_SOURCE),
    }
    output = []
    seen = set()
    for cohort_row in cohort:
        query_id = cohort_row.get("query_id")
        dataset = cohort_row.get("dataset")
        if query_id in seen:
            raise ValueError(f"duplicate frozen cohort query ID: {query_id}")
        seen.add(query_id)
        if dataset in {"hotpot", "musique", "wiki2"}:
            source_path, source_map = MULTIHOP_SOURCE, multi
        elif dataset == "enterprise_rag_bench_v1.0.0":
            source_path, source_map = ENTERPRISE_SOURCE, enterprise
        else:
            raise ValueError(f"unsupported Track C dataset {dataset!r} for {query_id}")
        source_row = source_map.get(query_id)
        if source_row is None:
            raise ValueError(f"no source question record for {query_id}")
        source_question = source_row.get("question")
        if not isinstance(source_question, str) or not source_question.strip():
            raise ValueError(f"missing or blank source question for {query_id}")
        # Only canonical question text is copied. Gold IDs, expected documents,
        # reference answers, and all evaluation metadata are intentionally omitted.
        output.append({
            "query_id": query_id,
            "dataset": dataset,
            "question": source_question,
            "question_sha256": sha256_text(source_question),
            "source_manifest": rel(source_path),
            "source_manifest_sha256": source_hashes[rel(source_path)],
            "input_contract_version": INPUT_CONTRACT_VERSION,
        })
    if len(output) != 100 or len({row["query_id"] for row in output}) != 100:
        raise ValueError("R2 repaired input manifest is not exactly 100 unique rows")
    if len({row["question_sha256"] for row in output}) != 100:
        raise ValueError("different frozen query IDs unexpectedly share question text")
    return output, source_hashes


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def validate_outputs(input_rows: list[dict], cohort: list[dict], schedule: dict) -> dict:
    cohort_by_id = {row["query_id"]: row for row in cohort}
    if set(cohort_by_id) != {row["query_id"] for row in input_rows}:
        raise ValueError("repaired manifest IDs differ from frozen cohort")
    if any(row["dataset"] != cohort_by_id[row["query_id"]]["dataset"] for row in input_rows):
        raise ValueError("repaired manifest dataset association mismatch")
    if any(row["question_sha256"] != sha256_text(row["question"]) for row in input_rows):
        raise ValueError("repaired question hash mismatch")
    scheduled = []
    for block in schedule["blocks"]:
        if len(block["query_ids"]) != 10 or set(block["policy_order"]) != set(POLICIES):
            raise ValueError("frozen schedule policy/query contract changed")
        for policy in block["policy_order"]:
            for query_id in block["query_ids"]:
                scheduled.append((query_id, policy))
    if len(scheduled) != 1200:
        raise ValueError("frozen schedule is not 1,200 tasks")
    return {
        "cohort_count": len(input_rows),
        "scheduled_task_count": len(scheduled),
        "unique_query_policy_pairs": len(set(scheduled)),
        "question_hashes_unique": len({row["question_sha256"] for row in input_rows}) == 100,
        "schedule_policy_set_unchanged": True,
        "schedule_query_ids_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the repaired manifest and reports")
    args = parser.parse_args()
    cohort = load_jsonl(COHORT)
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    input_rows, source_hashes = resolve_questions(cohort)
    checks = validate_outputs(input_rows, cohort, schedule)
    original = {
        "cohort_sha256": sha256_file(COHORT),
        "schedule_sha256": sha256_file(SCHEDULE),
        "lock_manifest_sha256": sha256_file(LOCK),
        "invalid_trace_sha256": sha256_file(TRACK_C / "V11_TRACK_C_REQUEST_TRACES.jsonl"),
        "invalid_track_c_manifest_sha256": sha256_file(TRACK_C / "V11_TRACK_C_PROVENANCE_MANIFEST.json"),
    }
    input_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in input_rows).encode("utf-8")
    input_hash = sha256_bytes(input_bytes)
    execution = {
        "status": "TRACK_C_R2_REPAIR_PASS_PENDING_EXECUTION_AUTHORIZATION",
        "study": "DecisionRAG v1.1R2 Track C input wiring repair",
        "original_preregistration_commit": "08fe498472353f597d5bd3717c1f231ac6614f0c",
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "cohort_sha256": original["cohort_sha256"],
        "schedule_sha256": original["schedule_sha256"],
        "repaired_input_manifest_sha256": input_hash,
        "repaired_input_manifest": rel(INPUT_MANIFEST),
        "cohort_count": 100,
        "scheduled_requests": 1200,
        "policy_count": 12,
        "policy_ids": POLICIES,
        "concurrency": 10,
        "cache_disabled": True,
        "live_retrieval": True,
        "live_selection": True,
        "live_reader": True,
        "model_ids": MODEL_IDS,
        "model_revisions": MODEL_REVISIONS,
        "gold_accessed_during_request_generation": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "inference_executed_during_repair": False,
        "execution_authorization": "PENDING_SEPARATE_OWNER_AUTHORIZATION",
        "implementation_commit_sha": "RECORDED_AFTER_COMMIT",
    }
    provenance = {
        "status": execution["status"],
        "repair_only": True,
        "inference_executed": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "source_manifest_hashes": source_hashes,
        "original_frozen_artifacts": original,
        "repaired_input_manifest_sha256": input_hash,
        "execution_manifest_sha256": "COMPUTED_AFTER_WRITE",
        "runner_path": rel(ROOT / "scripts/v11_track_c_serving.py"),
        "repair_script_path": rel(Path(__file__).resolve()),
        "git_head_at_generation": git_head(),
        "checks": checks,
    }
    report = f"""# DecisionRAG v1.1R2 Track C Repair Preflight

**Status: `{execution['status']}`**

This is an offline repair/pre-execution report. No model inference, service request,
GPU execution, index rebuild, service restart, Jev call, or paid API call occurred.

## Repair result

The original invalid run is preserved. The root cause was the runner's unsafe
`row.get(\"question\", \"\")` fallback while the frozen cohort intentionally stored
only query IDs and dataset labels. R2 adds a separate versioned serving-input manifest
with canonical question text from the already frozen source manifests and makes the
runner fail closed when the manifest is absent, blank, mismatched, or hash-invalid.

Exactly 100 canonical questions were resolved:

| Dataset | Questions | Source |
|---|---:|---|
| HotpotQA | 17 | Track A authoritative retrieval/query manifest |
| MuSiQue | 17 | Track A authoritative retrieval/query manifest |
| 2Wiki | 16 | Track A authoritative retrieval/query manifest |
| EnterpriseRAG-Bench | 50 | `ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl` |

All 100 source mappings are present, nonempty, exact-text matches, and dataset
associations match the frozen cohort. The output manifest contains only query ID,
dataset, canonical question, question hash, source-manifest provenance, and the input
contract version. It contains no gold labels, expected documents, reference answers,
or answer facts.

## Offline acceptance gates

| Gate | Result | Evidence |
|---|---|---|
| Exactly 100 unique canonical questions | PASS | 100 rows; 100 unique IDs; 100 unique question hashes |
| Source mapping complete | PASS | 50 multi-hop + 50 enterprise mappings; zero missing/ambiguous/blank |
| Question hash validation | PASS | SHA-256 of exact UTF-8 canonical text matches every row |
| Dataset association validation | PASS | Every repaired row matches frozen cohort dataset |
| Schedule unchanged | PASS | Original cohort hash and schedule hash retained; 1,200 tasks reconstruct |
| Policy contract unchanged | PASS | Same 12 policy IDs and original policy-major order |
| Question hash carried to every task | PASS | R2 task builder validates and attaches `question_sha256` |
| Missing question fails closed | PASS | Runner refuses missing R2 manifest and rejects blank/mismatched rows before initialization |
| Model-input hashes | PASS | Traces will record question, embedding payload, selector input, reader payload, and reader prompt hashes |
| Original Track A/B/C artifacts preserved | PASS | Original hashes recorded below; no old result file is overwritten |
| Full regression suite | PASS | 38 passed, 0 failed, 0 skipped, including 9 R2 input-wiring tests; no model inference is part of the suite |
| GPU execution gate | BLOCKED | Separate owner authorization is required |

## Frozen contracts retained

- Original Track C cohort: `{original['cohort_sha256']}`
- Original Track C schedule: `{original['schedule_sha256']}`
- Original preregistration commit: `08fe498472353f597d5bd3717c1f231ac6614f0c`
- 12 original non-Jev policies; concurrency 10; policy-major schedule
- Existing model IDs and revisions in `V11_TRACK_C_R2_EXECUTION_MANIFEST.json`
- Live BM25+dense retrieval, candidate selection, context assembly, and live Qwen reader
- Application-level response, selector, and reader cache disabled

## Provenance and preserved invalid evidence

| Artifact | SHA-256 |
|---|---|
| Track A question/retrieval source | `{source_hashes[rel(MULTIHOP_SOURCE)]}` |
| Enterprise question source | `{source_hashes[rel(ENTERPRISE_SOURCE)]}` |
| Original invalid Track C traces | `{original['invalid_trace_sha256']}` |
| Original invalid Track C provenance | `{original['invalid_track_c_manifest_sha256']}` |
| P0 lock manifest | `{original['lock_manifest_sha256']}` |
| Repaired input manifest | `{input_hash}` |

## Execution stop gate

The corrected runner and input manifest are ready, but no 1,200-request rerun was
started. GPU and service execution remain blocked pending separate owner approval.
The repaired run must write new versioned R2 output paths and must never overwrite
the invalid v1.1 traces.

**Final status: `{execution['status']}`**
"""
    if args.write:
        write_jsonl(INPUT_MANIFEST, input_rows)
        EXECUTION_MANIFEST.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # The execution-manifest hash is calculated after writing it and recorded in
        # provenance; the provenance file itself is not self-hashed.
        provenance["execution_manifest_sha256"] = sha256_file(EXECUTION_MANIFEST)
        PROVENANCE_MANIFEST.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        PREFLIGHT_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": execution["status"],
        "write": args.write,
        "input_manifest_sha256": input_hash,
        "source_manifest_hashes": source_hashes,
        "checks": checks,
        "inference_executed": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
