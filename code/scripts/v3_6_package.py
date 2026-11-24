#!/usr/bin/env python3
"""Build the path-free, derived-only v3.6 export package."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential"
STAGE = ROOT / "results/v3_6_essential/.export_stage"
ZIP = OUT / "decisionrag_v3_6_essential_exports.zip"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    try:
        with p.open(encoding="utf-8") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return None


def sanitize(s):
    s = re.sub(r"GPU-[0-9a-fA-F-]{20,}", "<redacted>", s)
    s = re.sub(r"(?:/mnt|/home|/root)/[^\s\"']+", "<redacted-path>", s)
    s = re.sub(r"\b(?:<REDACTED_USER>|<REDACTED_USER>|nvidia-smi|<redacted>)\b", "<redacted>", s, flags=re.I)
    return s


def copy_sanitized(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".tex", ".txt"}:
        dst.write_text(sanitize(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    # Derived reports and schemas only; raw traces are represented by hashes.
    files = [
        (OUT / "REPORT.md", Path("REPORT.md")),
        (OUT / "DEVIATIONS.md", Path("DEVIATIONS.md")),
        (OUT / "TEST_AND_INTEGRITY_REPORT.md", Path("TEST_AND_INTEGRITY_REPORT.md")),
        (OUT / "STATUS.md", Path("STATUS.md")),
        (OUT / "GATE_SUMMARY.json", Path("GATE_SUMMARY.json")),
        (OUT / "stage0_preflight/INPUT_MANIFEST.json", Path("stage0/INPUT_MANIFEST.json")),
        (OUT / "stage0_preflight/cohort_manifest.csv", Path("stage0/cohort_manifest.csv")),
        (OUT / "stage0_preflight/budget_tokens.csv", Path("stage0/budget_tokens.csv")),
        (OUT / "stage0_preflight/PROTOCOL.json", Path("stage0/PROTOCOL.json")),
    ]
    for p in sorted((OUT / "stage1_quality_v2/export/quality").glob("*")):
        if p.is_file():
            files.append((p, Path("quality") / p.name))
    for p in sorted((OUT / "stage2_serving/export").glob("*")):
        if p.is_file():
            files.append((p, Path("serving") / p.name))
    for p in sorted((OUT / "stage3_analysis/export/frontier").glob("*")):
        if p.is_file():
            files.append((p, Path("frontier") / p.name))
    for src, rel in files:
        if src.exists():
            copy_sanitized(src, STAGE / rel)

    protocol = {
        "experiment": "DecisionRAG v3.6 Essential Verification",
        "status": "PARTIAL",
        "population": "BRIGHT frozen BM25 Top-20, 1384 queries, 12 domains",
        "slate_sha256": "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2",
        "models": {"jina_v3_5_revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "pplx_decider_revision": "14c9d90137962d300f3de95633f7b6535e80c5ce"},
        "budgets": {"single": [1024, 2048, 3072, 4096, 6144, 8192], "two_stage": [2048, 4096, 8192], "rho": 128, "k": 20, "window_k": 10},
        "bootstrap": {"quality_draws": 40000, "serving_draws": 10000, "seed": 731},
        "gpu_slots": ["A", "B", "C"],
        "serving": {"cohort": "C360", "attempts": 3, "concurrency": 1, "timed_boundary": "view preparation, native calls, parsing, ID mapping, Top-5 packaging", "labels_joined": False},
        "replication": {"complete_primary_sessions": 2, "third_session": "Jina-only", "required_complete_primary_sessions": 3},
        "restrictions": {"new_retrieval": False, "protected_holdout_access": False, "paid_api_calls": 0, "raw_traces_in_export": False},
    }
    (STAGE / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw_specs = [
        (OUT / "stage2_serving/raw/jina/serving_perrequest_raw.jsonl", "raw/T1_A_jina.jsonl"),
        (OUT / "stage2_serving/raw/jina_j2_repair/serving_perrequest_raw.jsonl", "raw/T1_A_j2_repair.jsonl"),
        (OUT / "stage2_serving/raw/pplx/serving_perrequest_raw.jsonl", "raw/T1_A_pplx.jsonl"),
        (OUT / "stage2_serving_gpu6/raw/jina/serving_perrequest_raw.jsonl", "raw/T1_B_jina.jsonl"),
        (OUT / "stage2_serving_gpu6/raw/jina_j2_repair/serving_perrequest_raw.jsonl", "raw/T1_B_j2_repair.jsonl"),
        (OUT / "stage2_serving_gpu6/raw/pplx/serving_perrequest_raw.jsonl", "raw/T1_B_pplx.jsonl"),
        (OUT / "stage2_serving_gpu3_r3/raw/jina/serving_perrequest_raw.jsonl", "raw/T1_C_jina.jsonl"),
    ]
    manifest = {}
    for src, rel in raw_specs:
        if src.exists():
            manifest[rel] = {"sha256": sha(src), "bytes": src.stat().st_size, "record_count": sum(1 for line in src.open(encoding="utf-8") if line.strip())}
    (STAGE / "RAW_TRACE_SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    content = []
    for p in sorted(STAGE.rglob("*")):
        if not p.is_file() or p.name == "CONTENTS.md":
            continue
        rel = p.relative_to(STAGE).as_posix()
        content.append({"path": rel, "rows": rows(p) if p.suffix.lower() in {".csv", ".jsonl"} else "", "sha256": sha(p), "description": "derived v3.6 artifact"})
    lines = ["# DecisionRAG v3.6 derived export contents", "", "Raw traces are excluded; their hashes and record counts are in `RAW_TRACE_SHA256_MANIFEST.json`.", "", "| Path | Rows | SHA-256 | Description |", "|---|---:|---|---|"]
    lines.extend(f"| {x['path']} | {x['rows']} | {x['sha256']} | {x['description']} |" for x in content)
    (STAGE / "CONTENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bad = []
    for p in STAGE.rglob("*"):
        if p.is_file() and re.search(r"<REDACTED_PATH><REDACTED_USER>|<REDACTED_USER>|<redacted>|nvidia-smi|GPU-[0-9a-fA-F-]{20,}", p.read_text(encoding="utf-8", errors="ignore"), re.I):
            bad.append(p.relative_to(STAGE).as_posix())
    if bad:
        raise RuntimeError("EXPORT_HYGIENE_FAILURE:" + ",".join(bad))
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(ZIP) as z:
        file_count = len(z.namelist())
    print(json.dumps({"status": "PACKAGE_DONE", "zip": str(ZIP.relative_to(ROOT)), "bytes": ZIP.stat().st_size, "sha256": sha(ZIP), "files": file_count}))


if __name__ == "__main__":
    main()
