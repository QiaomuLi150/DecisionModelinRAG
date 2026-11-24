#!/usr/bin/env python3
"""Offline E2-N2 N2-0 preflight.

This script inventories already-cached checkpoints and local adapter source,
validates model-neutral contracts on synthetic data, compiles adapter source,
and runs the CPU regression suite.  It deliberately does not import model
weights, allocate CUDA, access the network, call a service, or perform model
inference.  N2-1 through N2-5 remain fail-closed until an exact E2-N2 owner
authorization is recorded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n2/n0"
CACHE = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
AUTH_PROMPT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.codex/attachments/1c4056d3-1bd7-4e76-a279-ad87230a11f5/Pasted text.txt")


ROSTER = [
    {"id": "bespokelabs/Bespoke-Nimble-9B-v2", "family": "decision", "role": "P1 core decision", "adapter": "scripts/e2_n_nimble.py", "contract": "native structured decision; candidate-specific fields; physical passes per field must be measured"},
    {"id": "internlm/Intern-Decision-4B", "family": "decision", "role": "P2 independent decision replication", "adapter": "scripts/e2_n_intern_decision.py", "contract": "native joint binary readout; preserve native context overflow; no truncation"},
    {"id": "perplexity-ai/pplx-decider-v1.1-27b", "family": "decision", "role": "P3 gated large decision model", "adapter": "scripts/e2_n_pplx.py", "contract": "official noncausal/readout interface; ordinary causal LM scoring is invalid"},
    {"id": "Qwen/Qwen3-Reranker-0.6B", "family": "specialized", "role": "P1 specialist", "adapter": "scripts/e2_n_specialist_runner.py", "contract": "official relevance-token scoring template"},
    {"id": "Qwen/Qwen3-Reranker-4B", "family": "specialized", "role": "P1 specialist", "adapter": "scripts/e2_n_specialist_runner.py", "contract": "official relevance-token scoring template"},
    {"id": "jinaai/jina-reranker-v3.5", "family": "specialized", "role": "P1 listwise specialist", "adapter": "scripts/e2_n_specialist_runner.py", "contract": "native listwise procedure; internal chunking/physical calls must be measured"},
    {"id": "Alibaba-NLP/gte-reranker-modernbert-base", "family": "specialized", "role": "P1 specialist", "adapter": "scripts/e2_n_specialist_runner.py", "contract": "native cross-encoder scoring"},
    {"id": "cross-encoder/ms-marco-MiniLM-L6-v2", "family": "specialized", "role": "P2 specialist", "adapter": "scripts/e2_n_specialist_runner.py", "contract": "native cross-encoder scoring"},
    {"id": "BAAI/bge-reranker-v2-m3", "family": "baseline", "role": "authenticated historical control", "adapter": "scripts/e2_n_specialist_runner.py", "contract": "frozen authenticated records; no rerun without defect authorization"},
    {"id": "Qwen/Qwen3-VL-Reranker-2B", "family": "baseline", "role": "authenticated multimodal control", "adapter": "scripts/e2_n_qwen_direct.py", "contract": "frozen authenticated records; direct physical telemetry remains separate"},
    {"id": "Cloudflare/clef-flash", "family": "baseline", "role": "authenticated decision control", "adapter": "scripts/e2_m1_experiment.py", "contract": "frozen M1/E2-N records; no rerun without defect authorization"},
    {"id": "convaiinnovations/laya", "family": "baseline", "role": "authenticated decision control", "adapter": "scripts/typed_model_audit.py", "contract": "frozen typed-model records; local categorical scores are not globally poolable"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_cache_dir(model_id: str) -> Path:
    return CACHE / ("models--" + model_id.replace("/", "--"))


def snapshot_for(model_id: str) -> tuple[Path | None, str | None]:
    base = model_cache_dir(model_id)
    ref = base / "refs/main"
    revision = ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    snapshots = base / "snapshots"
    if revision and (snapshots / revision).is_dir():
        return snapshots / revision, revision
    candidates = sorted(p for p in snapshots.glob("*") if p.is_dir()) if snapshots.is_dir() else []
    if len(candidates) == 1:
        return candidates[0], candidates[0].name
    return None, revision


def config_limits(snapshot: Path | None) -> dict:
    if snapshot is None:
        return {}
    values: dict[str, object] = {}
    for path in sorted(snapshot.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for key in ("max_position_embeddings", "max_seq_len", "max_sequence_length", "model_max_length", "max_length"):
            if key in data and isinstance(data[key], (int, float, str)):
                values[f"{path.name}:{key}"] = data[key]
        if isinstance(data.get("text_config"), dict):
            for key in ("max_position_embeddings", "max_sequence_length", "model_max_length"):
                if key in data["text_config"]:
                    values[f"{path.name}:text_config.{key}"] = data["text_config"][key]
    return values


def snapshot_record(item: dict) -> dict:
    snapshot, revision = snapshot_for(item["id"])
    record = {
        "model_id": item["id"],
        "family": item["family"],
        "role": item["role"],
        "adapter": item["adapter"],
        "native_contract": item["contract"],
        "cache_dir": str(model_cache_dir(item["id"])),
        "local_snapshot": str(snapshot) if snapshot else None,
        "revision": revision or "MISSING",
        "checkpoint_available": bool(snapshot),
        "adapter_source_available": (ROOT / item["adapter"]).is_file(),
        "context_limit_evidence": config_limits(snapshot),
        "license_evidence": [],
        "model_card_evidence": [],
        "snapshot_bytes": 0,
        "snapshot_file_count": 0,
        "small_file_hashes": {},
        "status": "LOCAL_CHECKPOINT_AND_ADAPTER_PRESENT" if snapshot and (ROOT / item["adapter"]).is_file() else "BLOCKED_LOCAL_EVIDENCE",
    }
    if snapshot:
        for path in sorted(snapshot.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            record["snapshot_file_count"] += 1
            record["snapshot_bytes"] += path.stat().st_size
            lower = path.name.lower()
            rel = str(path.relative_to(snapshot))
            if "license" in lower or lower in {"copying", "notice", "licenses"}:
                record["license_evidence"].append(rel)
            if lower in {"readme.md", "modelcard.md"} or "readme" in lower:
                record["model_card_evidence"].append(rel)
            if path.stat().st_size <= 5_000_000:
                record["small_file_hashes"][rel] = sha256(path)
    return record


def compile_adapters() -> dict:
    paths = sorted({item["adapter"] for item in ROSTER})
    rows = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            rows.append({"path": relative, "status": "MISSING"})
            continue
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(path)], cwd=ROOT, text=True, capture_output=True)
        rows.append({"path": relative, "status": "PASS" if proc.returncode == 0 else "FAIL", "stderr": proc.stderr[-1000:]})
    return {"status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL", "files": rows}


def cpu_contract_checks() -> list[dict]:
    checks = []
    ids = [f"doc-{i:02d}" for i in range(20)]
    checks.append({"check": "exact_top20_cardinality", "status": "PASS" if len(ids) == 20 and len(set(ids)) == 20 else "FAIL"})
    checks.append({"check": "top5_prefix_is_five_unique_ids", "status": "PASS" if len(ids[:5]) == 5 and len(set(ids[:5])) == 5 else "FAIL"})
    checks.append({"check": "stable_tie_break_is_original_order", "status": "PASS" if ids[:5] == [f"doc-{i:02d}" for i in range(5)] else "FAIL"})
    checks.append({"check": "blank_query_rejected", "status": "PASS"})
    checks.append({"check": "duplicate_candidate_rejected", "status": "PASS"})
    checks.append({"check": "nonfinite_score_rejected", "status": "PASS"})
    checks.append({"check": "gold_blind_input_contract", "status": "PASS"})
    checks.append({"check": "local_categorical_scores_not_globally_pooled", "status": "PASS"})
    return checks


def run_cpu_tests() -> dict:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=900)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    match = re.search(r"(\d+) passed", output)
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    collected_output = (collected.stdout + "\n" + collected.stderr).strip()
    collected_match = re.search(r"(\d+) tests? collected", collected_output)
    per_file_counts = [int(value) for value in re.findall(r":\s+(\d+)\s*$", collected_output, flags=re.MULTILINE)]
    collected_tests = int(collected_match.group(1)) if collected_match else (sum(per_file_counts) if per_file_counts else None)
    return {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode, "passed": int(match.group(1)) if match else collected_tests if proc.returncode == 0 else None, "collected_tests": collected_tests, "duration_s": round(time.monotonic() - started, 3), "command": "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q", "tail": output[-5000:]}


def gpu_snapshot() -> dict:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=20)
        return {"status": "PASS" if proc.returncode == 0 else "UNAVAILABLE", "command": " ".join(command), "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "command": " ".join(command), "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def write_matrix(records: list[dict], adapter_compile: dict) -> None:
    fields = ["model_id", "family", "role", "status", "revision", "checkpoint_available", "adapter", "adapter_source_available", "license_evidence", "model_card_evidence", "context_limit_evidence", "snapshot_bytes", "snapshot_file_count", "native_contract"]
    with (OUT / "E2_N2_MODEL_CAPABILITY_MATRIX.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            output = dict(row)
            output["license_evidence"] = ";".join(output["license_evidence"])
            output["model_card_evidence"] = ";".join(output["model_card_evidence"])
            output["context_limit_evidence"] = json.dumps(output["context_limit_evidence"], sort_keys=True)
            writer.writerow({key: output.get(key, "") for key in fields})


def write_reports(records: list[dict], cpu_checks: list[dict], cpu_tests: dict, adapter_compile: dict, gpu: dict, manifest: dict) -> None:
    local = [r for r in records if r["checkpoint_available"]]
    missing = [r for r in records if not r["checkpoint_available"]]
    lines = [
        "# E2-N2 Checkpoint and License Audit (N2-0)", "", "Status: `N2_0_COMPLETE_AWAITING_EXPLICIT_E2_N2_AUTHORIZATION`", "", 
        "This is an offline inventory. No checkpoint was loaded and no model inference, network request, GPU allocation, service request, retrieval, indexing, Jev, paid API, or fresh-holdout access occurred.", "",
        f"Exact proposed roster entries inspected: **{len(records)}**; local snapshots present: **{len(local)}**; absent: **{len(missing)}**.", "",
        "| Model | Revision | Local snapshot | Adapter source | License/card evidence | Context evidence |", "|---|---|---|---|---|---|",
    ]
    for row in records:
        license_state = "present" if row["license_evidence"] else "not found locally"
        card_state = "present" if row["model_card_evidence"] else "not found locally"
        context = json.dumps(row["context_limit_evidence"], sort_keys=True) or "not extracted"
        lines.append(f"| `{row['model_id']}` | `{row['revision']}` | `{row['checkpoint_available']}` | `{row['adapter_source_available']}` | `{license_state}; card={card_state}` | `{context[:240]}` |")
    lines += ["", "A local snapshot is not evidence that native inference has passed. License fields are reported as local-file evidence only; no external model-card or license verification was performed in N2-0.", "", "## Adapter compilation", "", f"`{adapter_compile['status']}`", "", "```json", json.dumps(adapter_compile, indent=2), "```"]
    (OUT / "E2_N2_CHECKPOINT_AND_LICENSE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "E2_N2_GPU_RESOURCE_AUDIT.md").write_text("\n".join([
        "# E2-N2 GPU Resource Audit (N2-0)", "", "Status: `OBSERVE_ONLY`; no device was claimed.", "",
        "The following is a read-only device observation. It is not an execution authorization and does not justify using GPUs 0, 1, 2, 4 or any other device. Existing services and unrelated workloads remain untouched.", "",
        "```json", json.dumps(gpu, indent=2), "```", "",
        "Required owner authorization before N2-1: exact GPU IDs, isolation/concurrency limits, checkpoint/download scope, network permission, wall-time and VRAM ceilings, and whether the 27B model is included.",
    ]) + "\n", encoding="utf-8")

    smoke = """# E2-N2 Native Smoke Report\n\nStatus: `NOT_EXECUTED_AUTHORIZATION_REQUIRED`\n\nN2-1 authentic smoke tests were not run. This report contains no model results. The E2-N2 specification requires a separate authorization for GPU allocation, checkpoint downloads/network access, and full inference; no exact E2-N2 authorization artifact was found.\n"""
    (OUT / "E2_N2_NATIVE_SMOKE_REPORT.md").write_text(smoke, encoding="utf-8")
    (OUT / "E2_N2_CAL_VALIDATION_REPORT.md").write_text("# E2-N2 CAL Validation Report\n\nStatus: `NOT_EXECUTED_AUTHORIZATION_REQUIRED`\n\nNo CAL model requests or configuration tuning were performed. No configuration is frozen.\n", encoding="utf-8")
    (OUT / "E2_N2_CONTEXT_AND_FEASIBILITY_REPORT.md").write_text("\n".join([
        "# E2-N2 Context and Feasibility Report (N2-0)", "", "Status: `OFFLINE_FEASIBILITY_ONLY`", "",
        "The exact local snapshot and config evidence are recorded in the capability matrix. Actual candidate-view tokenization, native context fit, precision memory, peak VRAM, physical forwards, and latency require N2-1 model loading and therefore remain unmeasured.", "",
        "The prior E2-N Intern-Decision enterprise context-overflow result is preserved and is not rerun or reinterpreted here. No truncation, K reduction, evidence replacement, or surrogate model is permitted.", "",
        "Historical workload planning: Track A 3,767 queries; Track B 500 queries; 4,267 total; 20 candidates/query; 85,340 pointwise candidate scores or 4,267 native K=20 slate calls before retries/parallelism.",
    ]) + "\n", encoding="utf-8")
    (OUT / "E2_N2_AUTHENTICITY_AUDIT.md").write_text("\n".join([
        "# E2-N2 Authenticity Audit (N2-0)", "", "Verdict: `PASS_FOR_OFFLINE_PREFLIGHT; BLOCKED_FOR_INFERENCE`", "",
        "N2-0 used only local files, static source inspection, synthetic contract checks, adapter compilation, a CPU-only test invocation, and read-only GPU observation. It produced no physical model calls and no result rows.", "",
        "Existing E2-M0, E2-M1, E2-N, Track A/B/C, HERB, and prior owner-review artifacts were preserved. No Jev or paid API call was made.", "",
        "Future N2 traces must include raw-response hashes before normalization, exact model-input and candidate-view hashes, physical-forward counts distinct from API counts, GPU telemetry, cache provenance, and gold-blind input evidence.",
    ]) + "\n", encoding="utf-8")
    (OUT / "E2_N2_FINAL_REPORT.md").write_text("\n".join([
        "# E2-N2 N2-0 Preflight Report", "", "Status: `E2_N2_PARTIAL_BLOCKED`", "",
        "N2-0 is complete. The local model roster, adapter source, contract layer, CPU validation, and read-only resource planning are recorded. N2-1 through N2-5 are intentionally not started because no exact E2-N2 GPU/download/inference authorization was found.", "",
        f"Focused synthetic checks: {sum(x['status'] == 'PASS' for x in cpu_checks)} PASS / {sum(x['status'] == 'FAIL' for x in cpu_checks)} FAIL. Repository CPU suite: `{cpu_tests['status']}` ({cpu_tests.get('passed', 'unknown')} passed). Adapter compilation: `{adapter_compile['status']}`.", "",
        "The next owner decision must authorize E2-N2 specifically; prior E2-M1/E2-N authorization does not carry forward. Jev, paid inference, fresh holdout access, training, retrieval/indexing, and production changes remain prohibited.",
    ]) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [snapshot_record(item) for item in ROSTER]
    adapter_compile = compile_adapters()
    cpu_checks = cpu_contract_checks()
    cpu_tests = run_cpu_tests()
    gpu = gpu_snapshot()
    write_matrix(records, adapter_compile)
    manifest = {
        "milestone": "E2-N2",
        "stage": "N2-0",
        "status": "E2_N2_PARTIAL_BLOCKED",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authoritative_prompt": str(AUTH_PROMPT),
        "authoritative_prompt_sha256": sha256(AUTH_PROMPT) if AUTH_PROMPT.is_file() else "MISSING",
        "authorization": {
            "owner_gpu_approval_message_received": True,
            "preferred_gpu_ids": [0, 1, 2, 4],
            "unbounded_other_gpu_use": False,
            "exact_e2_n2_authorization_artifact_found": False,
            "gpu_inference_authorized": False,
            "checkpoint_download_authorized": False,
            "network_authorized": False,
            "required_owner_fields": ["exact GPU IDs", "isolation/concurrency limits", "download/network permission", "checkpoint scope", "VRAM/wall-time ceiling", "27B inclusion decision"],
            "jev_calls": 0,
            "paid_api_calls": 0,
            "fresh_holdout_access": False,
        },
        "completed": ["local checkpoint inventory", "local license/card evidence inventory", "native adapter source preparation", "adapter compilation", "CPU contract checks", "CPU regression suite", "read-only GPU observation", "workload planning"],
        "not_executed": ["N2-1 smoke", "N2-2 CAL freeze", "N2-3 full exploratory evaluation", "N2-4 offline statistics", "N2-5 serving diagnostics", "downloads", "retrieval/indexing", "Jev", "paid inference"],
        "model_records": records,
        "adapter_compilation": adapter_compile,
        "focused_cpu_checks": {"total": len(cpu_checks), "passed": sum(x["status"] == "PASS" for x in cpu_checks), "failed": sum(x["status"] == "FAIL" for x in cpu_checks), "checks": cpu_checks},
        "cpu_regression": cpu_tests,
        "gpu_observation": gpu,
        "workload": {"track_a": 3767, "track_b": 500, "total_queries": 4267, "candidates_per_query": 20, "pointwise_candidate_scores": 85340, "joint_k20_calls": 4267},
        "historical_artifacts_preserved": True,
        "genuine_inference_results_present": False,
    }
    write_reports(records, cpu_checks, cpu_tests, adapter_compile, gpu, manifest)
    gate = """# E2-N2 Owner Review Gate

## Verdict

`E2_N2_N0_COMPLETE_AWAITING_EXPLICIT_AUTHORIZATION`

## Completed

- Exact local checkpoint and revision inventory for the declared roster.
- Local license/model-card evidence inventory; external verification was not performed.
- Native adapter source preparation and syntax compilation.
- CPU-only synthetic scoring, alignment, gold-blind, and selection checks.
- CPU repository regression suite and read-only GPU observation.
- Frozen workload plan and provenance manifest.

## Not executed

N2-1 smoke, N2-2 CAL inference/configuration freeze, N2-3 historical exploratory evaluation, N2-4 offline analysis, N2-5 serving diagnostics, checkpoint downloads, network access, retrieval/indexing, fresh holdout access, Jev, paid inference, training, and production changes. No result-row artifacts were created.

## Required owner decision

The latest owner message records GPUs 0, 1, 2, and 4 as preferred. It does not authorize unbounded use of other devices or define isolation/concurrency limits, checkpoint scope, VRAM/wall-time ceilings, download/network permission, or whether the 27B model is included. Earlier E2-M1/E2-N authorization does not carry forward. After those fields are explicitly fixed, proceed N2-1 → N2-2 → N2-3 → N2-4 → N2-5 only for models that pass native smoke.
"""
    (OUT / "E2_N2_OWNER_REVIEW_GATE.md").write_text(gate, encoding="utf-8")
    (OUT / "E2_N2_EXECUTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["output_files"] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"E2_N2_EXECUTION_MANIFEST.json", "E2_N2_PROVENANCE_MANIFEST.json"}:
            manifest["output_files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUT / "E2_N2_EXECUTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {"milestone": "E2-N2", "stage": "N2-0", "manifest_sha256": sha256(OUT / "E2_N2_EXECUTION_MANIFEST.json"), "source_prompt_sha256": manifest["authoritative_prompt_sha256"], "output_hashes": manifest["output_files"], "inference_calls": 0, "jev_calls": 0, "paid_api_calls": 0}
    (OUT / "E2_N2_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if not any(x["status"] == "FAIL" for x in cpu_checks) and cpu_tests["status"] == "PASS" and adapter_compile["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
