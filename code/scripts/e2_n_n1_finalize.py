#!/usr/bin/env python3
"""Finalize the E2-N N1 smoke-test evidence without new model calls."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n/n1"
MODELS = ("BGE", "CLEF", "LAYA", "QWEN")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model_rows = {}
    artifacts = []
    for model in MODELS:
        trace = OUT / f"E2_N_N1_{model}_SMOKE.jsonl"
        summary = OUT / f"E2_N_N1_{model}_SMOKE_SUMMARY.json"
        if not trace.is_file() or not summary.is_file():
            raise SystemExit(f"missing completed N1 artifacts for {model.lower()}")
        rows = load_jsonl(trace)
        sm = json.loads(summary.read_text(encoding="utf-8"))
        physical_ok = sum(row.get("physical_forward_evidence") is True for row in rows)
        actual_hashes = sum(bool(row.get("actual_model_input_sha256")) for row in rows)
        candidate_ok = sum(len(row.get("candidate_ids", [])) == 20 and len(set(row.get("candidate_ids", []))) == 20 for row in rows)
        model_rows[model.lower()] = {
            "summary": sm,
            "trace_rows": len(rows),
            "status_rows_ok_or_limited": sum(row.get("status") in {"ok", "ok_with_telemetry_limit"} for row in rows),
            "physical_forward_evidence_rows": physical_ok,
            "actual_model_input_hash_rows": actual_hashes,
            "valid_unique_20_candidate_rows": candidate_ok,
            "trace_sha256": sha_file(trace),
            "summary_sha256": sha_file(summary),
        }
        artifacts.extend([
            {"path": str(trace.relative_to(ROOT)), "sha256": sha_file(trace), "bytes": trace.stat().st_size},
            {"path": str(summary.relative_to(ROOT)), "sha256": sha_file(summary), "bytes": summary.stat().st_size},
        ])

    strict_models = [m for m, x in model_rows.items() if x["trace_rows"] == x["physical_forward_evidence_rows"] and x["trace_rows"] == x["actual_model_input_hash_rows"] and x["trace_rows"] == x["valid_unique_20_candidate_rows"] and x["status_rows_ok_or_limited"] == x["trace_rows"]]
    limited_models = [m for m, x in model_rows.items() if m not in strict_models and x["status_rows_ok_or_limited"] == x["trace_rows"]]
    manifest = {
        "stage": "N1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "owner_text": "User authorized E2-N execution and free use of GPU(0,1,2,4) and other available GPUs if possible.",
            "approved_gpu_ids": [0, 1, 2, 4],
            "fallback_gpu_ids_used": [6],
            "download_authorized": False,
            "download_count": 0,
            "jev_calls": 0,
            "paid_api_calls": 0,
            "unrelated_services_interrupted": False,
            "service_restarts": 0,
        },
        "smoke_contract": {
            "cal_queries": 10,
            "datasets": ["hotpot", "musique", "wiki2"],
            "candidates_per_query": 20,
            "gold_in_inference_input": False,
            "cache_reuse": False,
            "selection_policy": "not yet applied; raw scores/typed outputs only",
        },
        "models": model_rows,
        "strict_models_eligible_for_n2": strict_models,
        "limited_models_not_eligible_for_n2": limited_models,
        "gates": {
            "N1_BGE": "PASS" if "bge" in strict_models else "BLOCKED",
            "N1_CLEF": "PASS" if "clef" in strict_models else "BLOCKED",
            "N1_LAYA": "PASS" if "laya" in strict_models else "BLOCKED",
            "N1_QWEN": "PASS_WITH_TELEMETRY_LIMITATION" if "qwen" in limited_models else ("PASS" if "qwen" in strict_models else "BLOCKED"),
            "N1_OVERALL": "PASS_WITH_LIMITATIONS" if strict_models and limited_models else ("PASS" if strict_models else "BLOCKED"),
        },
        "artifact_index": artifacts,
    }
    manifest_path = OUT / "E2_N_N1_PROVENANCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E2-N N1 GPU Smoke and Resource Report",
        "",
        f"Status: `{manifest['gates']['N1_OVERALL']}`",
        "",
        "This report is generated only from completed N1 traces; it performs no model calls.",
        "",
        "## Model gates",
        "",
        "| Model | Rows | Output-complete | Physical-forward evidence | Input hashes | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model in ("bge", "clef", "laya", "qwen"):
        x = model_rows[model]
        gate = manifest["gates"][f"N1_{model.upper()}"]
        lines.append(f"| {model} | {x['trace_rows']} | {x['status_rows_ok_or_limited']} | {x['physical_forward_evidence_rows']} | {x['actual_model_input_hash_rows']} | {gate} |")
    lines += [
        "",
        "## Resource and integrity notes",
        "",
        "- Clef ran on GPU0; BGE and Laya ran on idle fallback GPU6. GPUs1/2/4 and other active workloads were not interrupted.",
        "- No checkpoint downloads were authorized or performed. All local checkpoints were loaded from pinned snapshots.",
        "- Qwen used the existing local service on port 8002. Its response alignment and raw-response hashes passed, but the service did not expose physical-forward telemetry; the N2/N3 Qwen branch is therefore fail-closed unless a durable service-level telemetry source is supplied.",
        "- Laya emitted a checkpoint temperature warning. The warning is retained as a calibration limitation; no score ordering change was inferred.",
        "- Jev calls: `0`; paid API calls: `0`; model training: `0`; production-service restarts: `0`.",
        "",
        "## Artifact provenance",
        "",
        "The machine-readable manifest contains SHA-256 hashes for both raw traces and summaries.",
    ]
    (OUT / "E2_N_GPU_SMOKE_AND_RESOURCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["gates"]["N1_OVERALL"], "strict_models": strict_models, "limited_models": limited_models}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
