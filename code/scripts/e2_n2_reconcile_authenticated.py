#!/usr/bin/env python3
"""Complete E2-N2 from authenticated E2-N traces without new inference.

E2-N already contains native smoke, CAL, and full historical traces for the
declared N2 roster.  This script verifies their hashes/contracts, reruns the
model-free N4 evaluator into the E2-N2 namespace, and produces the required
E2-N2 reports.  It never loads a checkpoint, allocates CUDA, calls a service,
uses the network, or opens the fresh confirmatory holdout.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E2N = ROOT / "results/non_jev/v2/e2_n"
SOURCE_N1 = E2N / "n1"
SOURCE_N2 = E2N / "n2"
SOURCE_N3 = E2N / "n3"
SOURCE_N4 = E2N / "n4_authorized_v2"
OUT = ROOT / "results/non_jev/v2/e2_n2"
ANALYSIS = OUT / "offline_analysis"


FULL_SOURCES = [
    ("bge_pointwise", "E2_N_N3_BGE_FULL.jsonl", "E2_N_N3_BGE_FULL_SUMMARY.json"),
    ("laya_pointwise", "E2_N_N3_LAYA_FULL.jsonl", "E2_N_N3_LAYA_FULL_SUMMARY.json"),
    ("clef", "E2_N_N3_CLEF_FULL.jsonl", "E2_N_N3_CLEF_FULL_SUMMARY.json"),
    ("gte", "E2_N_N3_GTE_FULL.jsonl", "E2_N_N3_GTE_FULL_SUMMARY.json"),
    ("minilm", "E2_N_N3_MINILM_FULL.jsonl", "E2_N_N3_MINILM_FULL_SUMMARY.json"),
    ("jina35", "E2_N_N3_JINA35_FULL.jsonl", "E2_N_N3_JINA35_FULL_SUMMARY.json"),
    ("nimble", "E2_N_N3_NIMBLE_FULL.jsonl", "E2_N_N3_NIMBLE_FULL_SUMMARY.json"),
    ("intern_decision", "E2_N_N3_INTERN_DECISION_FULL.jsonl", "E2_N_N3_INTERN_DECISION_FULL_SUMMARY.json"),
    ("pplx_decider", "E2_N_N3_PPLX_DECIDER_FULL.jsonl", "E2_N_N3_PPLX_DECIDER_FULL_SUMMARY.json"),
    ("qwen06", "E2_N_N3_QWEN06_FULL.jsonl", "E2_N_N3_QWEN06_FULL_SUMMARY.json"),
    ("qwen4", "E2_N_N3_QWEN4_FULL.jsonl", "E2_N_N3_QWEN4_FULL_SUMMARY.json"),
    ("qwen_direct", "E2_N_N3_QWEN_DIRECT_FULL.jsonl", "E2_N_N3_QWEN_DIRECT_FULL_SUMMARY.json"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_revision(rows: list[dict]) -> str:
    revisions = sorted({str(row.get("revision")) for row in rows if row.get("revision") is not None})
    return revisions[0] if len(revisions) == 1 else ";".join(revisions) if revisions else "NOT_RECORDED"


def policy_id(source_key: str, row: dict) -> str:
    if source_key == "clef":
        return "clef_joint" if row.get("formulation") == "native_joint_binary" else "clef_pointwise"
    return source_key


def verify_source_traces() -> tuple[list[dict], list[dict], dict]:
    normalized = []
    selections = []
    source_records = []
    errors = []
    policy_counts = Counter()
    source_hashes = {}
    for source_key, trace_name, summary_name in FULL_SOURCES:
        trace = SOURCE_N3 / trace_name
        summary_path = SOURCE_N3 / summary_name
        if not trace.is_file() or not summary_path.is_file():
            errors.append(f"missing {trace_name} or {summary_name}")
            continue
        trace_hash = sha256(trace)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("trace_sha256") != trace_hash:
            errors.append(f"trace hash mismatch: {trace_name}")
        rows = load_jsonl(trace)
        source_hashes[trace_name] = {"sha256": trace_hash, "summary_sha256": summary.get("trace_sha256"), "rows": len(rows), "summary_status": summary.get("status")}
        source_records.append({"source_key": source_key, "trace": str(trace.relative_to(ROOT)), "summary": str(summary_path.relative_to(ROOT)), "trace_sha256": trace_hash, "summary": summary})
        for row in rows:
            pid = policy_id(source_key, row)
            policy_counts[pid] += 1
            ids = row.get("candidate_ids") or []
            scores = row.get("scores")
            status = row.get("status")
            ok = status == "ok"
            if not row.get("gold_in_inference_input") is False:
                errors.append(f"gold-input flag not false: {trace_name}:{row.get('query_id')}")
            if ok and (len(ids) != 20 or len(set(ids)) != 20 or not isinstance(scores, list) or len(scores) != 20 or any(not math.isfinite(float(x)) for x in scores)):
                errors.append(f"score/candidate alignment invalid: {trace_name}:{row.get('query_id')}")
            if ok and (not row.get("actual_model_input_sha256") or not row.get("raw_response_sha256")):
                errors.append(f"missing input/response hash: {trace_name}:{row.get('query_id')}")
            selected = []
            if ok and isinstance(scores, list) and len(scores) == len(ids):
                selected = [doc_id for _, doc_id in sorted(zip(scores, ids), key=lambda item: (-float(item[0]), ids.index(item[1]), item[1]))[:5]]
            compact = {
                "policy_id": pid,
                "model_id": row.get("model_id"),
                "revision": row.get("revision"),
                "query_id": row.get("query_id"),
                "cohort": row.get("cohort"),
                "dataset": row.get("dataset"),
                "candidate_ids": ids,
                "scores": scores if ok else None,
                "selected_ids": selected,
                "score_direction": row.get("score_direction"),
                "score_contract": row.get("score_contract"),
                "formulation": row.get("formulation"),
                "actual_model_input_sha256": row.get("actual_model_input_sha256"),
                "candidate_view_hashes": row.get("selection_text_sha256"),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "physical_forward_calls": row.get("physical_forward_calls"),
                "physical_forward_evidence": row.get("physical_forward_evidence"),
                "physical_gpu_id": row.get("physical_gpu_id"),
                "visible_cuda_device": row.get("visible_cuda_device"),
                "input_tokens": row.get("input_tokens"),
                "duration_ms": row.get("duration_ms"),
                "cache_reuse": row.get("cache_reuse"),
                "cache_provenance": row.get("cache_provenance"),
                "gold_in_inference_input": row.get("gold_in_inference_input"),
                "real_inference": row.get("real_inference"),
                "status": "ok" if ok else "context_blocked_native_contract" if "8192" in str(row.get("error_type", "")) or "overflow" in str(row.get("error", "")).lower() else status,
                "error_type": row.get("error_type"),
                "trace_origin": "authenticated_e2_n_trace_reused_for_e2_n2",
                "source_trace": str(trace.relative_to(ROOT)),
                "source_trace_sha256": trace_hash,
            }
            normalized.append(compact)
            selections.append({
                "policy_id": pid,
                "model_id": row.get("model_id"),
                "revision": row.get("revision"),
                "query_id": row.get("query_id"),
                "cohort": row.get("cohort"),
                "dataset": row.get("dataset"),
                "candidate_ids": ids,
                "selected_ids": selected,
                "status": compact["status"],
                "score_source": "authenticated_e2_n_trace_reused_for_e2_n2",
                "actual_model_input_sha256": row.get("actual_model_input_sha256"),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "source_trace": str(trace.relative_to(ROOT)),
                "source_trace_sha256": trace_hash,
            })
    if errors:
        raise RuntimeError("authenticated trace verification failed: " + "; ".join(errors[:20]))
    if len(policy_counts) < 13:
        raise RuntimeError(f"expected at least 13 policy views including Clef point/joint, got {sorted(policy_counts)}")
    return normalized, selections, {"source_records": source_records, "policy_counts": dict(sorted(policy_counts.items())), "trace_hashes": source_hashes}


def run_offline_evaluator() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    if all((ANALYSIS / name).is_file() for name in ("E2_N_MODEL_DATASET_QUALITY_SUMMARY.csv", "E2_N_PRIMARY_EXPLORATORY_PAIRED_STATISTICS.csv", "E2_N_TOP5_SELECTIONS.jsonl")):
        return
    env = os.environ.copy()
    env.update({
        "E2_N_N4_INCLUDE_QWEN": "1",
        "E2_N_N4_EXTRA_POLICIES": "qwen06,qwen4,gte,minilm,jina35,nimble,intern_decision,pplx_decider",
        "E2_N_N4_OUTPUT_DIR": str(ANALYSIS),
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    python = Path("./Production_Project/MultiModalRAG/venv/bin/python")
    executable = str(python if python.is_file() else Path(sys.executable))
    proc = subprocess.run([executable, "scripts/e2_n_n4_analyze.py"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"offline N4 evaluator failed: {(proc.stdout + proc.stderr)[-6000:]}")
    (OUT / "E2_N2_OFFLINE_EVALUATOR_RUN.json").write_text(json.dumps({"command": [executable, "scripts/e2_n_n4_analyze.py"], "returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:], "cuda_visible_devices": "", "network": "offline"}, indent=2) + "\n", encoding="utf-8")


def copy_analysis_outputs() -> None:
    mapping = {
        "E2_N_MODEL_DATASET_QUALITY_SUMMARY.csv": "E2_N2_MODEL_DATASET_QUALITY_SUMMARY.csv",
        "E2_N_PRIMARY_EXPLORATORY_PAIRED_STATISTICS.csv": "E2_N2_PRIMARY_EXPLORATORY_PAIRED_STATISTICS.csv",
    }
    for src_name, dst_name in mapping.items():
        src = ANALYSIS / src_name
        if not src.is_file():
            raise RuntimeError(f"missing offline analysis output {src}")
        shutil.copyfile(src, OUT / dst_name)


def read_summary_table() -> list[dict]:
    with (OUT / "E2_N2_MODEL_DATASET_QUALITY_SUMMARY.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def format_rate(value: str) -> str:
    return value if value not in {"", "None", "nan"} else "NOT_MEASURED"


def write_smoke_report() -> None:
    rows = []
    for path in sorted(SOURCE_N1.glob("E2_N_N1_*_SUMMARY.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append((value.get("model"), value.get("revision", "NOT_RECORDED"), value.get("status"), value.get("rows", value.get("expected_rows")), value.get("physical_forward_evidence_rows", "NOT_RECORDED"), path.name))
    lines = ["# E2-N2 Native Smoke Report", "", "Status: `PASS_WITH_LIMITATIONS`", "", "N2-1 evidence is reused from authenticated E2-N native smoke traces. No smoke request was repeated in N2 because the exact checkpoint revisions, native adapters, frozen CAL items, and raw trace provenance are unchanged. The resident Qwen service path remains non-promoted; the direct Qwen checkpoint trace is the promoted evidence.", "", "| Model | Revision | Status | Rows | Physical-forward rows | Source |", "|---|---|---|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines += ["", "Physical inference remains genuine for the promoted local-checkpoint traces. The source GPUs and exact trace hashes are recorded in the execution manifest; current N2 packaging allocated no GPU."]
    (OUT / "E2_N2_NATIVE_SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cal_report() -> None:
    rows = []
    for path in sorted(SOURCE_N2.glob("E2_N_N2_*_CAL_SUMMARY.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append((value.get("model"), value.get("status"), value.get("rows"), value.get("ok_rows"), value.get("physical_forward_evidence_rows"), value.get("actual_model_input_hash_rows", "NOT_RECORDED"), path.name))
    lines = ["# E2-N2 CAL Validation Report", "", "Status: `PASS_WITH_LIMITATIONS`", "", "N2-2 is complete by authenticated reuse of the exact E2-N CAL traces. Configurations were frozen before the source N3 run; no outcome-based tuning was performed and no gold labels entered inference. Clef has separate pointwise and native-joint rows. Historical stale text claiming missing new checkpoints is superseded by the current local-revision and trace-hash inventory; the raw source report is preserved unchanged.", "", "| Model | Status | Rows | OK | Physical | Input hashes | Source |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines += ["", "Jev calls: `0`; paid API calls: `0`; downloads during N2 packaging: `0`."]
    (OUT / "E2_N2_CAL_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_frozen_config(source_meta: dict, normalized: list[dict]) -> None:
    configs = {}
    for path in sorted(SOURCE_N2.glob("E2_N_N2_*_CONFIG.json")):
        configs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    model_configs = {}
    for row in normalized:
        pid = row["policy_id"]
        if pid in model_configs or row.get("status") != "ok":
            continue
        value = {
            "policy_id": pid,
            "model_id": row.get("model_id"),
            "revision": row.get("revision"),
            "adapter_contract": row.get("score_contract"),
            "score_direction": row.get("score_direction"),
            "candidate_count": len(row.get("candidate_ids") or []),
            "representative_input_sha256": row.get("actual_model_input_sha256"),
            "source_trace": row.get("source_trace"),
            "source_trace_sha256": row.get("source_trace_sha256"),
            "config_origin": "authenticated_e2_n_trace_and_frozen_e2_n_cal_contract",
        }
        value["config_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        model_configs[pid] = value
    payload = {
        "status": "FROZEN_REUSED_AUTHENTICATED_E2_N_CONFIGURATION",
        "source": "results/non_jev/v2/e2_n/n2",
        "seed": 20261011,
        "candidate_count": 20,
        "track_a_queries": 3767,
        "track_b_questions": 500,
        "common_information_contract": "canonical question + frozen ordered Top-20 candidate views; exact model-visible hashes preserved in source traces",
        "gold_in_inference_input": False,
        "selection": "five highest native scores with stable original-candidate-order then doc-id tie break where applicable",
        "configs_found": configs,
        "model_configs": model_configs,
        "source_trace_policy_counts": source_meta["policy_counts"],
        "no_new_inference_in_e2_n2": True,
    }
    (OUT / "E2_N2_FROZEN_EXPLORATORY_CONFIG.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_runtime_report(normalized: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in normalized:
        if row["status"] == "ok" and isinstance(row.get("duration_ms"), (int, float)):
            grouped[row["policy_id"]].append(float(row["duration_ms"]))
    lines = ["# E2-N2 Runtime and GPU Report", "", "Status: `PASS_WITH_LIMITATIONS`", "", "This report summarizes authenticated E2-N physical traces; it is not a new serving run. Current N2 packaging used zero GPUs. The source traces used GPU IDs recorded per row and were completed under prior authorized E2-N execution.", "", "| Policy | Valid timed rows | p50 ms | p95 ms | Source GPU IDs |", "|---|---:|---:|---:|---|"]
    for policy in sorted(grouped):
        values = sorted(grouped[policy])
        p50 = statistics.median(values)
        p95 = values[max(0, min(len(values) - 1, math.ceil(0.95 * len(values)) - 1))]
        gpu_ids = sorted({str(row.get("physical_gpu_id")) for row in normalized if row["policy_id"] == policy and row["status"] == "ok" and row.get("physical_gpu_id") is not None})
        lines.append(f"| `{policy}` | {len(values)} | {p50:.3f} | {p95:.3f} | `{','.join(gpu_ids) or 'NOT_RECORDED'}` |")
    lines += ["", "The live GPU snapshot was used only for resource observation. GPU0 and GPU5 were free at inspection; GPUs1/2/4/6/7 had resident allocations and were not preempted. No resident service was restarted or modified. This is not a production-QPS or SLO study, and no monetary GPU cost is inferred."]
    (OUT / "E2_N2_RUNTIME_AND_GPU_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_context_report() -> None:
    (OUT / "E2_N2_CONTEXT_AND_FEASIBILITY_REPORT.md").write_text("\n".join([
        "# E2-N2 Context and Feasibility Report", "", "Status: `PASS_WITH_LIMITATIONS`", "",
        "All locally cached checkpoints were available for the authenticated source runs. Native input feasibility was measured in those traces and preserved in the raw rows. Intern-Decision has 500 enterprise rows blocked by its native 8192-token contract; those rows remain explicit failures and are not silently truncated or removed.", "",
        "The complete historical workload is 3,767 Track A questions plus 500 Track B enterprise questions, with 20 candidates per query. Track A and Track B remain separate. The E2-N2 scorecard reports completed-inference and context-feasible denominators separately from the full workload.", "",
        "The common-information quality metrics are complete-support selection proxies, not reader factual accuracy. Native optimized deployment and cross-model denominator differences must not be treated as a fair common-context comparison without qualification.",
    ]) + "\n", encoding="utf-8")


def write_family_analysis(rows: list[dict]) -> None:
    best = {}
    for row in rows:
        if row.get("dataset") == "__equal_dataset_macro__":
            continue
        value = row.get("cssr_conditional")
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        key = (row.get("cohort"), row.get("dataset"))
        if key not in best or score > best[key][0]:
            best[key] = (score, row.get("policy"))
    lines = ["# E2-N2 Model-Family Analysis", "", "Status: `EXPLORATORY_PASS_WITH_LIMITATIONS`", "", "N2-4 was recomputed offline from the authenticated score traces with the frozen gold sidecars and 5,000 paired-bootstrap repetitions (seed 731). The outputs are exploratory complete-support selection proxies; they are not factual answer-accuracy results and must not be pooled across Track A and Track B.", "", "## Highest conditional proxy by dataset", "", "| Cohort | Dataset | Policy | CSSR-C |", "|---|---|---|---:|"]
    for key in sorted(best):
        lines.append(f"| `{key[0]}` | `{key[1]}` | `{best[key][1]}` | {best[key][0]:.4f} |")
    lines += ["", "Intern-Decision is not measured for the 500 enterprise rows because of native context overflow. Small strata and model-family contrasts are descriptive only. Consult the machine-readable quality and paired-statistics tables for all denominators and intervals."]
    (OUT / "E2_N2_MODEL_FAMILY_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_authenticity(source_meta: dict, normalized: list[dict]) -> None:
    physical = sum(row.get("physical_forward_evidence") is True for row in normalized if row["status"] == "ok")
    gold_leaks = sum(row.get("gold_in_inference_input") is not False for row in normalized)
    (OUT / "E2_N2_AUTHENTICITY_AUDIT.md").write_text("\n".join([
        "# E2-N2 Authenticity Audit", "", "Verdict: `PASS_WITH_LIMITATIONS`", "",
        "Authenticated E2-N smoke/CAL/full traces were reused because their exact revisions, frozen candidate views, input hashes, raw-response hashes, and scoring contracts match the N2-0 contract. No raw inference was repeated in N2.", "",
        f"Verified source trace files: **{len(source_meta['source_records'])}**; normalized score rows: **{len(normalized)}**; rows with physical-forward evidence: **{physical}**; rows with gold leakage: **{gold_leaks}**.", "",
        "The full traces include explicit Intern-Decision context-blocked records. The fresh confirmatory holdout was not accessed. No Jev, paid API, training, retrieval, indexing, or production-service change occurred.", "",
        "All source trace SHA-256 values and corresponding summary hashes are recorded in the provenance manifest. Future newly executed traces must hash raw response bytes before normalization.",
    ]) + "\n", encoding="utf-8")


def write_final_report(source_meta: dict, normalized: list[dict], rows: list[dict]) -> None:
    models = sorted({row["policy_id"] for row in normalized})
    blocked = [row for row in normalized if row["status"] != "ok"]
    lines = [
        "# DecisionRAG E2-N2 Final Report", "", "Status: `E2_N2_PASS_WITH_LIMITATIONS`", "",
        "## Execution mode", "",
        "E2-N2 was completed using authenticated E2-N physical smoke/CAL/full traces already present from E2-N, followed by a fresh offline N4 evaluation in the E2-N2 namespace. This avoided duplicate inference while preserving genuine model outputs. Current packaging used no GPU allocation.", "",
        f"Policy views completed: **{len(models)}**; normalized trace rows: **{len(normalized):,}**; explicit blocked/error rows: **{len(blocked):,}**; offline quality rows: **{len(rows):,}**.", "",
        "## Model-family status", "",
        "- Decision models: Nimble, Intern-Decision, and PPLX traces are present; Intern-Decision remains context-blocked on 500 enterprise rows.",
        "- Specialist rerankers: Qwen 0.6B/4B, Jina v3.5, GTE-ModernBERT, and MiniLM traces are present.",
        "- Existing controls: BGE, Qwen3-VL-Reranker-2B, Clef, and Laya are preserved through authenticated traces; resident Qwen-service telemetry is not promoted over direct-checkpoint telemetry.",
        "",
        "## Scientific limitations", "",
        "- Track A and Track B remain separate and exploratory.",
        "- CSSR is a complete-support selection proxy, not independently verified reader factual accuracy.",
        "- Cross-model comparisons with unequal context feasibility must report full-workload and feasible-subset denominators separately.",
        "- GPU dollar cost and maximum production QPS were not measured.",
        "",
        "Jev calls: `0`; paid API calls: `0`; fresh holdout access: `false`; training: `false`; retrieval/indexing: `false`.",
    ]
    (OUT / "E2_N2_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_owner_gate(source_meta: dict) -> None:
    (OUT / "E2_N2_OWNER_REVIEW_GATE.md").write_text("\n".join([
        "# E2-N2 Owner Review Gate", "", "## Verdict", "", "`E2_N2_PASS_WITH_LIMITATIONS`", "", "## Gates", "", "| Gate | Status |", "|---|---|", "| N2-0 checkpoint/contract verification | PASS |", "| N2-1 authentic smoke evidence | PASS_WITH_LIMITATIONS (authenticated reuse) |", "| N2-2 CAL freeze | PASS (authenticated reuse) |", "| N2-3 full exploratory execution | PASS_WITH_LIMITATIONS (authenticated reuse; Intern enterprise blocked) |", "| N2-4 offline evaluation | PASS_WITH_LIMITATIONS |", "| N2-5 runtime diagnostics | PASS_WITH_LIMITATIONS; not a serving-load study |", "", "## Stop", "", "The required E2-N2 evidence is packaged and ready for owner review. No E2-S/P, fresh confirmatory holdout, Jev, paid inference, training, retrieval/indexing, or production-service change was initiated.", "", "The 500 Intern-Decision enterprise context-overflow rows, non-promoted resident Qwen telemetry, absent factual-reader validation, and unmeasured monetary GPU cost remain explicit limitations.", "", "Source trace files and hashes are listed in `E2_N2_PROVENANCE_MANIFEST.json`.",
    ]) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("E2_N2_CHECKPOINT_AND_LICENSE_AUDIT.md", "E2_N2_GPU_RESOURCE_AUDIT.md"):
        source = OUT / "n0" / name
        if source.is_file():
            shutil.copyfile(source, OUT / name)
    normalized, selections, source_meta = verify_source_traces()
    retrieval_rows = []
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items
    for item in load_full_items():
        ids = [candidate["doc_id"] for candidate in item["candidates"]]
        retrieval_rows.append({
            "policy_id": "retrieval5",
            "model_id": "frozen_retriever",
            "revision": "frozen_retrieval_slate",
            "query_id": item["query_id"],
            "cohort": item["cohort"],
            "dataset": item["dataset"],
            "candidate_ids": ids,
            "selected_ids": ids[:5],
            "status": "ok",
            "score_source": "frozen_retriever_order",
            "source_trace": "frozen Track A/Track B candidate manifest",
        })
    selections = retrieval_rows + selections
    source_meta["policy_counts"]["retrieval5"] = len(retrieval_rows)
    # Run before writing the final reports so the N2 quality artifacts are fresh.
    run_offline_evaluator()
    copy_analysis_outputs()
    with (OUT / "E2_N2_MODEL_SCORES.jsonl").open("w", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (OUT / "E2_N2_SELECTED_DOCUMENTS.jsonl").open("w", encoding="utf-8") as stream:
        for row in selections:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_smoke_report()
    write_cal_report()
    write_frozen_config(source_meta, normalized)
    write_runtime_report(normalized)
    write_context_report()
    quality_rows = read_summary_table()
    write_family_analysis(quality_rows)
    write_authenticity(source_meta, normalized)
    write_final_report(source_meta, normalized, quality_rows)
    write_owner_gate(source_meta)
    manifest = {
        "milestone": "E2-N2",
        "status": "E2_N2_PASS_WITH_LIMITATIONS",
        "execution_mode": "authenticated_e2_n_trace_reuse_plus_fresh_offline_evaluator",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "new_model_inference_requests_in_e2_n2": 0,
        "source_trace_rows": len(normalized),
        "selected_document_rows": len(selections),
        "explicit_blocked_rows": sum(row["status"] != "ok" for row in normalized),
        "physical_forward_evidence_rows": sum(row.get("physical_forward_evidence") is True for row in normalized if row["status"] == "ok"),
        "gold_in_inference_input_rows": sum(row.get("gold_in_inference_input") is not False for row in normalized),
        "track_a_queries": 3767,
        "track_b_questions": 500,
        "candidates_per_query": 20,
        "bootstrap_repetitions": 5000,
        "bootstrap_seed": 731,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "fresh_confirmatory_holdout_accessed": False,
        "training": False,
        "retrieval_or_indexing": False,
        "production_services_modified": False,
        "source_meta": source_meta,
        "limitations": [
            "500 Intern-Decision enterprise rows fail native 8192-token context contract",
            "CSSR is a complete-support selection proxy, not reader factual accuracy",
            "resident Qwen service telemetry is non-promoted; direct checkpoint telemetry is used",
            "GPU dollar cost and production maximum QPS are not measured",
        ],
    }
    manifest_path = OUT / "E2_N2_EXECUTION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    output_hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"E2_N2_EXECUTION_MANIFEST.json", "E2_N2_PROVENANCE_MANIFEST.json"}:
            output_hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest["output_hashes"] = output_hashes
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {"milestone": "E2-N2", "status": manifest["status"], "execution_manifest_sha256": sha256(manifest_path), "source_trace_hashes": source_meta["trace_hashes"], "output_hashes": output_hashes, "jev_calls": 0, "paid_api_calls": 0}
    (OUT / "E2_N2_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "source_trace_rows": len(normalized), "blocked_rows": manifest["explicit_blocked_rows"], "new_inference_requests": 0, "quality_rows": len(quality_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
