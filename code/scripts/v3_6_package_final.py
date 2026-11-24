#!/usr/bin/env python3
"""Build the append-only v3.6 Stage-3 evidence package."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential"
STAGE = OUT / ".export_stage_v2"
ZIP = OUT / "decisionrag_v3_6_essential_exports_v2.zip"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def row_count(path):
    if path.suffix.lower() not in {".csv", ".jsonl"}:
        return ""
    with path.open(encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1) if path.suffix.lower() == ".csv" else n


def sanitize(text):
    text = re.sub(r"GPU-[0-9a-fA-F-]{20,}", "<redacted>", text)
    text = re.sub(r"(?:/mnt|/home|/root)/[^\s\"']+", "<redacted-path>", text)
    text = re.sub(r"\b(?:<REDACTED_USER>|<REDACTED_USER>|<redacted>|nvidia-smi)\b", "<redacted>", text, flags=re.I)
    return text


def copy_file(src, rel):
    dst = STAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".tex", ".txt"}:
        dst.write_text(sanitize(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def collect_files():
    files = []
    fixed = [
        (OUT / "STAGE3_COMPLETION_REPORT.md", "STAGE3_COMPLETION_REPORT.md"),
        (OUT / "STATUS_STAGE3_FINAL.md", "STATUS_STAGE3_FINAL.md"),
        (OUT / "DEVIATIONS_STAGE3_FINAL.md", "DEVIATIONS_STAGE3_FINAL.md"),
        (OUT / "TEST_AND_INTEGRITY_REPORT_STAGE3.md", "TEST_AND_INTEGRITY_REPORT_STAGE3.md"),
        (OUT / "TEST_AND_INTEGRITY_REPORT_STAGE3.json", "TEST_AND_INTEGRITY_REPORT_STAGE3.json"),
        (OUT / "GATE_SUMMARY_STAGE3.json", "GATE_SUMMARY_STAGE3.json"),
        (OUT / "stage0_preflight/INPUT_MANIFEST.json", "stage0/INPUT_MANIFEST.json"),
        (OUT / "stage0_preflight/cohort_manifest.csv", "stage0/cohort_manifest.csv"),
        (OUT / "stage0_preflight/budget_tokens.csv", "stage0/budget_tokens.csv"),
        (OUT / "stage0_preflight/PROTOCOL.json", "stage0/PROTOCOL.json"),
        (OUT / "stage2_t2/SOLO_PROTOCOL.json", "t2/SOLO_PROTOCOL.json"),
        (OUT / "stage3_analysis/STAGE3_COMPLETION_SUMMARY.json", "stage3/STAGE3_COMPLETION_SUMMARY.json"),
        (OUT / "stage3_analysis/T2_RAW_TRACE_MANIFEST.json", "stage3/T2_RAW_TRACE_MANIFEST.json"),
        (OUT / "stage3_analysis/T2_TIMING_SUMMARY.md", "stage3/T2_TIMING_SUMMARY.md"),
        (OUT / "stage3_analysis/T2_FRONTIER_STATISTICS.md", "stage3/T2_FRONTIER_STATISTICS.md"),
        (OUT / "stage3_analysis/SOLO_INTERFERENCE_REPORT.md", "stage3/SOLO_INTERFERENCE_REPORT.md"),
    ]
    files.extend(fixed)
    for p in sorted((OUT / "stage2_t2").glob("T2_*/PROTOCOL.json")):
        files.append((p, Path("t2") / p.parent.name / "PROTOCOL.json"))
    for p in sorted((OUT / "stage1_quality_v2/export/quality").glob("*")):
        if p.is_file(): files.append((p, Path("quality") / p.name))
    for p in sorted((OUT / "stage2_serving/export").glob("*")):
        if p.is_file(): files.append((p, Path("t1_serving") / p.name))
    for p in sorted((OUT / "stage3_analysis/export").rglob("*")):
        if p.is_file(): files.append((p, Path("stage3") / "export" / p.relative_to(OUT / "stage3_analysis/export")))
    return [(Path(src), Path(rel)) for src, rel in files if Path(src).exists()]


def raw_manifest():
    specs = []
    for p in sorted((OUT / "stage2_serving").glob("raw/*/serving_perrequest_raw.jsonl")):
        specs.append(p)
    for p in sorted((OUT / "stage2_serving_gpu6").glob("raw/*/serving_perrequest_raw.jsonl")):
        specs.append(p)
    for p in sorted((OUT / "stage2_serving_gpu3_r3").glob("raw/*/serving_perrequest_raw.jsonl")):
        specs.append(p)
    for p in sorted((OUT / "stage2_t2").glob("T2_*/raw/*/serving_perrequest_raw.jsonl")):
        specs.append(p)
    out = {}
    for p in specs:
        rel = str(p.relative_to(ROOT))
        out[rel] = {"sha256": sha(p), "bytes": p.stat().st_size, "record_count": sum(1 for line in p.open(encoding="utf-8") if line.strip())}
    return out


def main():
    if STAGE.exists():
        raise RuntimeError("append-only staging directory already exists: " + str(STAGE))
    STAGE.mkdir(parents=True)
    files = collect_files()
    for src, rel in files:
        copy_file(src, rel)
    protocol = {
        "experiment": "DecisionRAG v3.6 Essential Verification",
        "stage3_status": "PASS_T2_FRONTIER_AND_CPU_STATS",
        "overall_status": "PARTIAL_T1_REPLICATION_LIMIT",
        "quality_population": "BRIGHT frozen 1,384 queries / 12 domains",
        "timing_population": "C120 x 3 attempts; T2 A/B/C plus B6.2 solo",
        "slate_sha256": "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2",
        "models": {"jina_v3_5_revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "pplx_decider_revision": "14c9d90137962d300f3de95633f7b6535e80c5ce"},
        "bootstrap": {"quality_draws": 40000, "timing_draws": 10000, "frontier_probability_draws": 10000, "seed": 731},
        "replication": {"t1_complete_primary_sessions": 2, "t1_third_session": "Jina-only", "t2_complete_sessions": 3, "solo_complete": True},
        "restrictions": {"new_retrieval": False, "protected_holdout_access": False, "paid_api_calls": 0, "raw_traces_in_export": False},
    }
    (STAGE / "PROTOCOL_STAGE3.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = raw_manifest()
    (STAGE / "RAW_TRACE_SHA256_MANIFEST_STAGE3.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entries = []
    for p in sorted(STAGE.rglob("*")):
        if not p.is_file() or p.name == "CONTENTS.md": continue
        rel = p.relative_to(STAGE).as_posix()
        entries.append({"path": rel, "rows": row_count(p), "sha256": sha(p), "description": "v3.6 Stage-3 derived/protocol artifact"})
    lines = ["# DecisionRAG v3.6 Stage-3 export contents", "", "Raw serving traces are excluded. Their hashes and record counts are in `RAW_TRACE_SHA256_MANIFEST_STAGE3.json`.", "", "| Path | Rows | SHA-256 | Description |", "|---|---:|---|---|"]
    lines.extend(f"| {x['path']} | {x['rows']} | {x['sha256']} | {x['description']} |" for x in entries)
    (STAGE / "CONTENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = []
    for p in STAGE.rglob("*"):
        if p.is_file() and re.search(r"<REDACTED_PATH><REDACTED_USER>|<REDACTED_USER>|<redacted>|nvidia-smi|GPU-[0-9a-fA-F-]{20,}", p.read_text(encoding="utf-8", errors="ignore"), re.I):
            bad.append(p.relative_to(STAGE).as_posix())
    if bad: raise RuntimeError("EXPORT_HYGIENE_FAILURE:" + ",".join(bad))
    if ZIP.exists(): raise RuntimeError("append-only ZIP already exists: " + str(ZIP))
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        content = z.read("CONTENTS.md").decode("utf-8")
    print(json.dumps({"status": "PACKAGE_DONE", "zip": str(ZIP.relative_to(ROOT)), "bytes": ZIP.stat().st_size, "sha256": sha(ZIP), "files": len(names), "grep_hygiene": "PASS", "contents_lines": len(content.splitlines())}, indent=2))


if __name__ == "__main__": main()
