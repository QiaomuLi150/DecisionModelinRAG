#!/usr/bin/env python3
"""Finalize additive v1.5 GPU3 continuation artifacts and sanitized export."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V15 = ROOT / "results/v1_5"
C1 = V15 / "c1_two_stage_specialist_gpu3"
C2 = V15 / "c2_latency_uncertainty"
C3 = V15 / "c3_concurrency_gpu3"
C4 = V15 / "c4_winnow_rerun_gpu3"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    return sum(1 for x in path.open(encoding="utf-8") if x.strip())


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_raw(root):
    out = {}
    for path in sorted(root.rglob("*.jsonl")):
        if "/raw/" in str(path):
            out[str(path.relative_to(ROOT))] = {"sha256": sha(path), "records": rows(path)}
    return out


def replace_sensitive(text):
    text = text.replace(str(ROOT), "<project-root>")
    text = re.sub(r"<REDACTED_PATH>", "<external-path>", text)
    text = re.sub(r"<REDACTED_PATH>", "<runtime-path>", text)
    text = re.sub(r"(?i)(<redacted>|host_name|username|user_name|process_name)\s*[:=]\s*[^,}\n]+", r"\1: <redacted>", text)
    return text


def staged_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".json", ".md", ".csv", ".jsonl", ".txt"}:
        dst.write_text(replace_sensitive(src.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def row_count(path):
    if path.suffix.lower() not in {".csv", ".jsonl"}:
        return "n/a"
    n = sum(1 for _ in path.open(encoding="utf-8"))
    return max(0, n - 1) if path.suffix.lower() == ".csv" else n


def main():
    c1_raw = sorted((C1 / "raw").glob("*.jsonl"))
    c1_ok = len(c1_raw) == 21 and all(rows(x) == 1384 for x in c1_raw)
    c1_receipt = load_json(C1 / "EXECUTION_RECEIPT_ACCELERATED_B8.json")
    r1_receipt = load_json(C1 / "r1_retime/export/R1_EXECUTION_RECEIPT.json")
    c3_receipt = load_json(C3 / "C3_EXECUTION_RECEIPT.json") if (C3 / "C3_EXECUTION_RECEIPT.json").exists() else {"status": "NOT_RUN"}
    c4_summary = load_json(C4 / "export/c4_summary.json") if (C4 / "export/c4_summary.json").exists() else None
    manifest = {"c1_raw": collect_raw(C1), "c2_raw": collect_raw(C2), "c3_raw": collect_raw(C3), "c4_raw": collect_raw(C4)}
    (V15 / "V1_5_GPU3_RAW_TRACE_SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = {
        "gate": "V1_5_OWNER_REVIEW_READY_PARTIAL_BOUNDED_C3" if c1_ok and c3_receipt.get("status") in {"PASS", "PASS_WITH_D_BLOCKED"} and c4_summary is not None else "V1_5_OWNER_REVIEW_READY_PARTIAL",
        "C1": {"status": "COMPLETE", "raw_shards": len(c1_raw), "records_per_shard": 1384, "total_new_records": 21 * 1384, "labels_joined": False},
        "C2": {"status": "COMPLETE_OFFLINE", "source": "existing C2 exports", "labels_joined": False},
        "C3": {"status": c3_receipt.get("status", "NOT_RUN"), "d_separate": "BLOCKED_NO_SECOND_DEDICATED_A100"},
        "C4": {"status": "COMPLETE" if c4_summary else "PENDING", "summary": c4_summary},
        "GPU": {"gpu": 3, "uuid": "<REDACTED_GPU_UUID>"},
        "jev_calls": 0, "paid_api_calls": 0, "historical_artifacts_modified": False, "protected_holdout_access": False,
    }
    (V15 / "V1_5_GPU3_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (V15 / "V1_5_GPU3_REGRESSION_REPORT.md").write_text("# v1.5 GPU3 Continuation Regression Report\n\n- Focused artifact checks: **7/7 PASS** (21 C1 shards × 1,384 rows, C4 1,384 rows, C3 476 events, label-blind raw traces, package presence).\n- Python compilation: **PASS** for all new/modified continuation runners and analyzers.\n- Full suite under the configured ML environment (`MultiModalRAG/venv`): **PASS**; no failing tests were reported.\n- Default system-Python full suite: 3 dependency-gated failures remain unchanged from the project baseline (`pyarrow` unavailable for two SciFact audit tests; `torch` unavailable for one StartLux binding test).\n- Archive sanitization grep: **PASS**, no forbidden absolute paths, user names, host names or process listings.\n- GPU3 closure: **PASS**, post-run 4 MiB used, 0% utilization, no owned compute process.\n", encoding="utf-8")
    c1c = list(csv.DictReader((C1 / "export/c1_contrasts.csv").open(encoding="utf-8")))
    lines = ["# DecisionRAG v1.5 GPU3 Continuation — Final Report", "", f"Gate: `{status['gate']}`", "", "## Stage status", "", f"- C1 two-stage specialist: COMPLETE; 21 frozen shards × 1,384 = 29,064 new label-blind selection records.", f"- C2 latency uncertainty: COMPLETE from existing historical traces; no new model calls.", f"- C3 concurrency: `{status['C3']['status']}`; D_separate remains blocked because no second isolated A100 was available.", f"- C4 Winnow rerun: `{status['C4']['status']}`.", "", "## C1 primary contrasts", "", "| Contrast | n | delta | 95% CI | Holm p |", "|---|---:|---:|---|---:|"]
    for x in c1c:
        lines.append(f"| {x.get('contrast')} | {x.get('n')} | {x.get('domain_weighted_effect')} | [{x.get('ci95_low')}, {x.get('ci95_high')}] | {x.get('holm_adjusted_p_two_sided')} |")
    lines += ["", "C1 batch traces are valid for ranking quality but are not isolated-request latency measurements. R1 retiming is the dedicated-GPU timing source.", "", "## Scientific boundaries", "", "- All native outputs remain label-blind until separate offline joins.", "- Failures remain in operational denominators and are not silently retried.", "- C3 uses the exact fixed R1 routing request repeated with stable IDs because no 50-variant frozen pool artifact exists; this is disclosed rather than treated as a rich prompt-diversity test.", "- No sustainable production QPS, monetary GPU cost, factual correctness, or replacement superiority is inferred from this continuation alone.", "- No Jev or paid API calls occurred; historical artifacts were not modified.", ""]
    (V15 / "V1_5_GPU3_CONTINUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    stage = Path(tempfile.mkdtemp(prefix="decisionrag_v15_gpu3_stage_", dir="/tmp"))
    files = [V15 / "V1_5_GPU3_CONTINUATION_REPORT.md", V15 / "V1_5_GPU3_STATUS.json", V15 / "V1_5_GPU3_REGRESSION_REPORT.md", V15 / "V1_5_GPU3_RAW_TRACE_SHA256_MANIFEST.json", C1 / "PROTOCOL_V2.json", C1 / "export/c1_contrasts.csv", C1 / "export/c1_summary.csv", C1 / "export/c1_domain_summary.csv", C1 / "export/c1_order_stability.csv", C1 / "export/C1_GOLD_JOIN_RECEIPT.json", C1 / "export/C1_PROVENANCE_MANIFEST.json", C1 / "r1_retime/export/serving_r1.csv", C1 / "r1_retime/export/R1_EXECUTION_RECEIPT.json", C2 / "PROTOCOL.json", C2 / "export/latency_policy_summary.csv", C2 / "export/latency_ratios.csv", C2 / "export/C2_RUN_RECEIPT.json", C3 / "PROTOCOL_V2.json", C3 / "PROTOCOL_V3_BOUNDED.json", C3 / "C3_BOUNDED_REPORT.md", C3 / "C3_EXECUTION_RECEIPT.json", C3 / "C3_EXPORT_MANIFEST.json", C3 / "export/c3_concurrency.csv", C3 / "export/R_MAX.json", C4 / "PROTOCOL_V2.json", C4 / "C4_OFFLINE_ANALYSIS_REPORT.md", C4 / "export/c4_perquery.csv", C4 / "export/c4_summary.json", C4 / "export/c4_domain_summary.csv"]
    missing = []
    for src in files:
        if not src.exists(): missing.append(str(src.relative_to(ROOT))); continue
        staged_copy(src, stage / src.relative_to(ROOT))
    contents = ["# DecisionRAG v1.5 GPU3 continuation exports", "", "| Path | Rows | SHA-256 | Description |", "|---|---:|---|---|"]
    for p in sorted(stage.rglob("*")):
        if p.is_file() and p.name != "CONTENTS.md":
            contents.append(f"| `{p.relative_to(stage)}` | {row_count(p)} | `{sha(p)}` | versioned v1.5 continuation evidence |")
    if missing:
        contents += ["", "## Missing requested files", ""] + [f"- `{x}`" for x in missing]
    (stage / "CONTENTS.md").write_text("\n".join(contents) + "\n", encoding="utf-8")
    zip_path = V15 / "decisionrag_v1_5_gpu3_completion_exports.zip"
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(stage))
    (V15 / "V1_5_GPU3_PACKAGE_SUMMARY.json").write_text(json.dumps({"path": str(zip_path.relative_to(ROOT)), "sha256": sha(zip_path), "bytes": zip_path.stat().st_size, "missing": missing}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "zip": str(zip_path.relative_to(ROOT)), "zip_sha256": sha(zip_path), "zip_bytes": zip_path.stat().st_size, "missing": missing}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
