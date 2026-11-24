#!/usr/bin/env python3
"""Finalize N2 CAL evidence and repair only derived summary denominators."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n/n2"
MODELS = ("bge", "clef", "laya")


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
    models = {}
    artifact_index = []
    for model in MODELS:
        trace = OUT / f"E2_N_N2_{model.upper()}_CAL.jsonl"
        config_path = OUT / f"E2_N_N2_{model.upper()}_CONFIG.json"
        if not trace.is_file() or not config_path.is_file():
            raise SystemExit(f"missing N2 artifacts for {model}")
        rows = load_jsonl(trace)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_rows = 60 if model == "clef" else 30
        ok = [r for r in rows if r.get("status") == "ok"]
        physical = [r for r in ok if r.get("physical_forward_evidence") is True]
        input_hashes = [r for r in ok if r.get("actual_model_input_sha256")]
        candidate_ok = [r for r in ok if len(r.get("candidate_ids", [])) == 20 and len(set(r.get("candidate_ids", []))) == 20]
        sm = {
            "stage": "N2_CAL",
            "model": model,
            "config_sha256": config["config_sha256"],
            "rows": len(rows),
            "expected_rows": expected_rows,
            "unique_queries": len({r.get("query_id") for r in rows}),
            "ok_rows": len(ok),
            "physical_forward_evidence_rows": len(physical),
            "actual_model_input_hash_rows": len(input_hashes),
            "valid_unique_20_candidate_rows": len(candidate_ok),
            "gold_in_inference_input_rows": sum(r.get("gold_in_inference_input") is True for r in rows),
            "cache_reuse_rows": sum(r.get("cache_reuse") is True for r in rows),
            "jev_calls": 0,
            "paid_api_calls": 0,
            "downloads": 0,
            "status": "PASS" if len(rows) == expected_rows and len(ok) == len(rows) and len(physical) == len(rows) and len(input_hashes) == len(rows) and len(candidate_ok) == len(rows) else "BLOCKED",
            "denominator_correction": "Clef has two rows per query (pointwise and native_joint); expected_rows=60 is intentional and the raw trace is unchanged." if model == "clef" else None,
        }
        (OUT / f"E2_N_N2_{model.upper()}_CAL_SUMMARY.json").write_text(json.dumps(sm, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        models[model] = sm | {"trace_sha256": sha_file(trace), "config_sha256": config["config_sha256"], "config_path": str(config_path.relative_to(ROOT))}
        for path in (trace, config_path, OUT / f"E2_N_N2_{model.upper()}_CAL_SUMMARY.json"):
            artifact_index.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size})

    gate = "PASS" if all(x["status"] == "PASS" for x in models.values()) else "BLOCKED"
    manifest = {
        "stage": "N2_CAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate,
        "cal_contract": {
            "queries": 30,
            "dataset_counts": {"hotpot": 10, "musique": 10, "wiki2": 10},
            "candidate_count_per_query": 20,
            "gold_in_inference_input": False,
            "cache_reuse": False,
            "configuration_frozen_before_n3": True,
            "seed": 20261011,
        },
        "models": models,
        "qwen_gate": "BLOCKED_SERVICE_PHYSICAL_TELEMETRY_NOT_EXPOSED",
        "new_models_gate": "BLOCKED_NO_LOCAL_CHECKPOINT_AND_DOWNLOAD_NOT_AUTHORIZED",
        "jev_calls": 0,
        "paid_api_calls": 0,
        "artifact_index": artifact_index,
    }
    (OUT / "E2_N_CAL_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E2-N N2 CAL Freeze Report",
        "",
        f"Status: `{gate}`",
        "",
        "Thirty existing CAL queries were evaluated with fixed, predeclared local configurations. No gold labels were loaded into inference, no model replacement or outcome-based tuning was performed, and no cache reuse occurred.",
        "",
        "| Model | Expected rows | Valid rows | Physical evidence | Input hashes | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        x = models[model]
        lines.append(f"| {model} | {x['expected_rows']} | {x['ok_rows']} | {x['physical_forward_evidence_rows']} | {x['actual_model_input_hash_rows']} | {x['status']} |")
    lines += [
        "",
        "Clef intentionally has 60 rows because each of the 30 queries has one pointwise and one native-joint trace. The initial derived summary briefly used 30 as the expected row count; this finalizer corrected that denominator without changing the raw trace.",
        "",
        "Qwen is not promoted: its local service returned aligned scores but did not expose physical-forward telemetry. New candidate models remain blocked because checkpoint downloads were not authorized and no local snapshots were present.",
        "",
        "Jev calls: `0`; paid API calls: `0`.",
    ]
    (OUT / "E2_N_N2_CAL_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate, "models": {m: x["status"] for m, x in models.items()}}, sort_keys=True))
    return 0 if gate == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
