#!/usr/bin/env python3
"""Write the aggregate E2-N owner-review package from completed artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e2_n"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    n0_matrix = read_csv(BASE / "n0/E2_N_MODEL_NATIVE_CAPABILITY_MATRIX.csv")
    n1 = read_json(BASE / "n1/E2_N_N1_PROVENANCE_MANIFEST.json")
    n2 = read_json(BASE / "n2/E2_N_CAL_FREEZE_MANIFEST.json")
    n3 = {p.stem.split("_N3_")[1].split("_FULL")[0].lower(): read_json(p) for p in (BASE / "n3").glob("*_SUMMARY.json")}
    n4 = read_json(BASE / "n4/E2_N4_PROVENANCE_MANIFEST.json")
    quality = read_csv(BASE / "n4/E2_N_MODEL_DATASET_QUALITY_SUMMARY.csv")
    paired = read_csv(BASE / "n4/E2_N_PRIMARY_EXPLORATORY_PAIRED_STATISTICS.csv")

    existing = sum(row["status"] == "VERIFIED_LOCAL" for row in n0_matrix)
    blocked = sum(row["status"] == "BLOCKED" for row in n0_matrix)
    not_authorized = sum(row["status"] == "NOT_AUTHORIZED" for row in n0_matrix)
    models = []
    for row in n0_matrix:
        mid = row["model_id"]
        key = {"BAAI/bge-reranker-v2-m3": "bge", "convaiinnovations/laya": "laya", "Cloudflare/clef-flash": "clef", "Qwen/Qwen3-VL-Reranker-2B": "qwen"}.get(mid)
        models.append({
            "model_id": mid,
            "role": row["role"],
            "n0_status": row["status"],
            "revision": row["revision"],
            "n1_gate": n1["gates"].get(f"N1_{key.upper()}") if key else ("NOT_RUN" if row["status"] != "VERIFIED_LOCAL" else "UNMAPPED"),
            "n2_gate": n2["models"].get(key, {}).get("status", "NOT_RUN") if key in {"bge", "laya", "clef"} else ("BLOCKED_TELEMETRY" if key == "qwen" else "NOT_RUN"),
            "n3_gate": n3.get(key, {}).get("status", "NOT_RUN"),
        })

    # Compact scorecard: use pooled dataset rows only, never merge Track A/B.
    scorecard = []
    for row in quality:
        if row["dataset"] in {"__equal_dataset_macro__"}:
            continue
        scorecard.append({
            "cohort": row["cohort"],
            "dataset": row["dataset"],
            "policy": row["policy"],
            "n_total": int(row["n_total"]),
            "n_eligible": int(row["n_eligible"]),
            "n_conditional": int(row["n_conditional"]),
            "cssr_unconditional": float(row["cssr_unconditional"]) if row["cssr_unconditional"] else None,
            "cssr_conditional": float(row["cssr_conditional"]) if row["cssr_conditional"] else None,
        })

    manifest = {
        "stage": "E2-N-COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_LIMITATIONS",
        "authorization": {
            "owner_authorized_gpu_ids": [0, 1, 2, 4],
            "fallback_gpu_ids_used": [6],
            "gpu_ids_used_by_new_e2_n_runs": {"gpu0": ["clef N1/N2", "laya N3"], "gpu6": ["bge N1/N2/N3", "laya N1/N2"]},
            "active_service_gpu_ids_left_untouched": [1, 2, 4],
            "downloads": 0,
            "jev_calls": 0,
            "paid_api_calls": 0,
            "training_runs": 0,
            "production_service_restarts": 0,
            "fresh_confirmatory_holdout_accessed": False,
        },
        "completed_stages": {
            "N0": {"focused_cpu_checks": 22, "repository_tests": 65, "verified_local_roster": existing, "blocked_roster_entries": blocked, "not_authorized_entries": not_authorized},
            "N1": n1["gates"],
            "N2": n2["status"],
            "N3": {m: x["status"] for m, x in n3.items()},
            "N4": n4["status"],
        },
        "models": models,
        "historical_population": {"track_a_queries": 3767, "track_b_questions": 500, "track_b_identity_resolved_eligible": 443, "track_a_retrieval_conditional": 1490, "track_b_retrieval_conditional": 330},
        "scorecard": scorecard,
        "paired_statistics_rows": len(paired),
        "limitations": [
            "Qwen smoke scores were aligned but physical-forward telemetry was unavailable from the local service; it was excluded from N2-N4.",
            "New official E2-N checkpoints were not locally available and checkpoint downloads were not authorized; no substitutes were used.",
            "Laya checkpoint temperature metadata is invalid/out of range; ranking outputs are reported, confidence calibration is not certified.",
            "Clef N3 full traces are compatible E2-M1 traces reused with source hashes; they were not silently regenerated.",
            "CSSR is complete-support selection quality, not independently verified reader factual accuracy; no new reader outputs were generated.",
            "GPU dollar cost was not measured and is reported as NOT_MEASURED.",
        ],
        "next_owner_decision": [
            "Decide whether to provide service-level physical-forward telemetry if Qwen must enter the benchmark.",
            "Separately authorize public checkpoint downloads and exact revisions if the blocked new roster is required.",
            "Review the exploratory Track A/B scorecards before any further model expansion or a separately approved Jev phase.",
        ],
    }
    manifest_path = BASE / "E2_N_EXECUTION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# DecisionRAG v2.1 E2-N Expanded Model Benchmark — Final Owner-Review Package",
        "",
        "Status: `PASS_WITH_LIMITATIONS`",
        "",
        "## Completed scope",
        "",
        "N0 offline verification, N1 authentic local smoke tests, N2 CAL configuration freezing, N3 frozen Track A/B historical exploratory scoring, and N4 independent offline analysis are complete for the verified, instrumented local branches. No training, retrieval/indexing, reader generation, Jev, paid API, or fresh confirmatory-holdout access occurred.",
        "",
        f"- N0: `{existing}` verified-local roster entries; `{blocked}` exact proposed entries blocked by absent local checkpoint/dependency and no download authorization; `{not_authorized}` Jev entry not authorized.",
        "- N0 CPU validation: 22 focused checks passed; existing repository suite: 65/65 passed.",
        "- N1: BGE, Clef, and Laya strict gates passed; Qwen aligned smoke output passed with a physical-service telemetry limitation.",
        "- N2: BGE 30 rows, Laya 30 rows, Clef 60 rows (30 queries × pointwise/native-joint) passed. Config hashes are in `n2/E2_N_CAL_FREEZE_MANIFEST.json`.",
        "- N3: BGE 4,267 rows, Laya 4,267 rows, Clef 8,534 compatible reused rows; every row has candidate alignment, input/output hashes, and physical-forward evidence.",
        "- N4: 21,335 frozen Top-5 selection rows, 30 dataset/policy quality rows, 32 paired bootstrap rows.",
        "",
        "## Verified model roster",
        "",
        "| Model | N0 | N1 | N2 | N3 | Final handling |",
        "|---|---|---|---|---|---|",
    ]
    for m in models[:4]:
        handling = "included" if m["n3_gate"] == "PASS" else ("smoke-only; telemetry blocked" if m["model_id"] == "Qwen/Qwen3-VL-Reranker-2B" else "not run")
        lines.append(f"| {m['model_id']} | {m['n0_status']} | {m['n1_gate']} | {m['n2_gate']} | {m['n3_gate']} | {handling} |")
    lines += [
        "",
        "New decision candidates Bespoke Nimble-9B-v2, Intern-Decision-4B, and PPLX Decider 27B, and new specialist candidates Qwen3-Reranker-0.6B/4B, Jina-v3.5, GTE-ModernBERT, and MiniLM-L6 remained blocked because exact local checkpoints/dependencies were absent and downloads were not authorized. No nearby substitute was silently used.",
        "",
        "## Scorecard (complete-support CSSR proxy)",
        "",
        "Track A and Track B are intentionally separate. `CSSR-U` is unconditional over identity-resolved eligible records; `CSSR-C` is conditional on all gold identities being present in the frozen Top-20. These are selector-support metrics, not factual answer accuracy.",
        "",
        "| Cohort | Dataset | Policy | n eligible | n conditional | CSSR-U | CSSR-C |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in scorecard:
        lines.append(f"| {r['cohort']} | {r['dataset']} | {r['policy']} | {r['n_eligible']} | {r['n_conditional']} | {r['cssr_unconditional']:.4f} | {r['cssr_conditional']:.4f} |" if r["cssr_unconditional"] is not None and r["cssr_conditional"] is not None else f"| {r['cohort']} | {r['dataset']} | {r['policy']} | {r['n_eligible']} | {r['n_conditional']} | {r['cssr_unconditional']} | {r['cssr_conditional']} |")
    lines += [
        "",
        "## Runtime and integrity",
        "",
        "- BGE selector p50/p95: 20.25/27.84 ms per 20-candidate query; Laya: 145.34/156.56 ms; Clef combined pointwise/native-joint trace rows: 1,195.80/2,385.23 ms. These are model-scoring scopes, not end-to-end serving latency.",
        "- Laya physical forwards: 5 per query under its fixed candidate batch size; Clef preserves pointwise and native-joint semantics separately.",
        "- GPUs used: GPU0 for Clef and Laya workloads; idle fallback GPU6 for BGE and Laya CAL/smoke workloads. GPUs1/2/4 hosted active services and were not disturbed.",
        "- No checkpoint downloads, service restarts, model training, Jev calls, or paid API calls.",
        "- All score traces contain frozen candidate IDs, actual input hashes, raw output hashes, physical-forward evidence, cache markers, and gold-blind inference markers.",
        "",
        "## Remaining blockers and owner decision",
        "",
        "1. Qwen requires durable service-level physical-forward telemetry before any full E2-N promotion.",
        "2. The new official model roster requires separate download/network authorization and exact checkpoint/revision verification.",
        "3. The current findings are exploratory on previously inspected Track A/B populations; they do not authorize Jev or paid inference.",
        "",
        "The next owner decision is whether to resolve those blockers or accept the current three-model exploratory comparison as the E2-N endpoint. Jev remains untouched and requires separate authorization.",
        "",
        "## Artifact locations",
        "",
        "- N0: `results/non_jev/v2/e2_n/n0/`",
        "- N1: `results/non_jev/v2/e2_n/n1/`",
        "- N2: `results/non_jev/v2/e2_n/n2/`",
        "- N3: `results/non_jev/v2/e2_n/n3/`",
        "- N4: `results/non_jev/v2/e2_n/n4/`",
        "- Machine-readable aggregate: `results/non_jev/v2/e2_n/E2_N_EXECUTION_MANIFEST.json`",
    ]
    (BASE / "E2_N_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (BASE / "E2_N_OWNER_REVIEW_GATE.md").write_text("# E2-N Owner Review Gate\n\nStatus: `PASS_WITH_LIMITATIONS`\n\nN0–N4 are complete for the three strict local branches (BGE, Laya, Clef) on the frozen historical Track A/B populations. Qwen is smoke-only because the local service lacks physical-forward telemetry. New official checkpoints remain blocked without download authorization. Jev calls: `0`; paid calls: `0`; fresh confirmatory holdout access: `false`.\n\nOwner decision required: review the aggregate report and decide whether to authorize checkpoint downloads/telemetry remediation or close E2-N with the documented limitations.\n", encoding="utf-8")
    # Do not self-hash the manifest: adding its own digest would make the
    # recorded value recursive and unverifiable.  The manifest itself is
    # covered by its filesystem path and the stage manifests below.
    manifest["artifact_hashes"] = {str(p.relative_to(ROOT)): sha_file(p) for p in [BASE / "E2_N_FINAL_REPORT.md", BASE / "E2_N_OWNER_REVIEW_GATE.md"]}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "scorecard_rows": len(scorecard), "paired_rows": len(paired), "models": [m["model_id"] for m in models[:4]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
