#!/usr/bin/env python3
"""Build the completed, sanitized append-only v1.6 export package."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_6"
ZIP = OUT / "decisionrag_v1_6_continuation_exports_v2.zip"

FILES = {
    "README.md": OUT / "README.md",
    "d1_perwindow_jina/PROTOCOL.json": OUT / "d1_perwindow_jina/PROTOCOL.json",
    "d1_perwindow_jina/D1_PERWINDOW_REPORT.md": OUT / "d1_perwindow_jina/D1_PERWINDOW_REPORT.md",
    "d1_perwindow_jina/D1_INPUT_MANIFEST_LOCK.json": OUT / "d1_perwindow_jina/D1_INPUT_MANIFEST_LOCK.json",
    "d1_perwindow_jina/D1_RAW_TRACE_SHA256_MANIFEST.json": OUT / "d1_perwindow_jina/D1_RAW_TRACE_SHA256_MANIFEST.json",
    "d1_perwindow_jina/D1_DETERMINISM_RECHECK.json": OUT / "d1_perwindow_jina/D1_DETERMINISM_RECHECK.json",
    "d1_perwindow_jina/D1_OFFLINE_SUMMARY.json": OUT / "d1_perwindow_jina/D1_OFFLINE_SUMMARY.json",
    "d1_perwindow_jina/export/d1_perquery.csv": OUT / "d1_perwindow_jina/export/d1_perquery.csv",
    "d1_perwindow_jina/export/d1_domain_summary.csv": OUT / "d1_perwindow_jina/export/d1_domain_summary.csv",
    "d1_perwindow_jina/export/d1_contrasts.csv": OUT / "d1_perwindow_jina/export/d1_contrasts.csv",
    "d2_matched_timing/PROTOCOL.json": OUT / "d2_matched_timing/PROTOCOL.json",
    "d2_matched_timing/D2_RESOURCE_GATE.json": OUT / "d2_matched_timing/D2_RESOURCE_GATE.json",
    "d2_matched_timing/PROTOCOL_V2.json": OUT / "d2_matched_timing/PROTOCOL_V2.json",
    "d2_matched_timing/D2_RESOURCE_GATE_V2.json": OUT / "d2_matched_timing/D2_RESOURCE_GATE_V2.json",
    "d2_matched_timing/D2_SESSION_METADATA_V2.json": OUT / "d2_matched_timing/D2_SESSION_METADATA_V2.json",
    "d2_matched_timing/D2_LATENCY_STATS_V2.json": OUT / "d2_matched_timing/D2_LATENCY_STATS_V2.json",
    "d2_matched_timing/D2_RAW_TRACE_SHA256_MANIFEST_V2.json": OUT / "d2_matched_timing/D2_RAW_TRACE_SHA256_MANIFEST_V2.json",
    "d2_matched_timing/D2_COMPLETION_V2.md": OUT / "d2_matched_timing/D2_COMPLETION_V2.md",
    "d2_matched_timing/D2_MATCHED_TIMING_REPORT.md": OUT / "d2_matched_timing/D2_MATCHED_TIMING_REPORT.md",
    "d2_matched_timing/export/d2_latency_perrequest.csv": OUT / "d2_matched_timing/export/d2_latency_perrequest.csv",
    "d2_matched_timing/export/d2_latency_ratios.csv": OUT / "d2_matched_timing/export/d2_latency_ratios.csv",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def scrub_json(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if "process" in k.lower():
                out[k] = "<redacted>"
            else:
                out[k] = scrub_json(v)
        return out
    if isinstance(value, list):
        return [scrub_json(x) for x in value]
    return value


def sanitize(text: str, source: Path) -> str:
    # Absolute paths are never needed in the shareable package.
    text = re.sub(r"/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", "<redacted>", text)
    text = re.sub(r"\b(?:<REDACTED_USER>|<REDACTED_USER>)\b", "<redacted>", text, flags=re.I)
    text = re.sub(r"GPU-[0-9A-Fa-f-]{8,}", "<redacted>", text)
    text = re.sub(r"(?i)<redacted>", "<redacted>", text)
    return text


def copy_sanitized(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".json":
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
            dst.write_text(sanitize(json.dumps(scrub_json(raw), ensure_ascii=False, indent=2, sort_keys=True) + "\n", src), encoding="utf-8")
            return
        except Exception:
            pass
    dst.write_text(sanitize(src.read_text(encoding="utf-8", errors="replace"), src), encoding="utf-8")


def row_count(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    if path.suffix.lower() == ".jsonl":
        return sum(1 for line in path.open(encoding="utf-8") if line.strip())
    return "n/a"


def main() -> None:
    missing = [rel for rel, src in FILES.items() if not src.exists()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    stage = Path(tempfile.mkdtemp(prefix="decisionrag_v1_6_stage_v2_", dir="/tmp"))
    entries = []
    for rel, src in FILES.items():
        dst = stage / rel
        copy_sanitized(src, dst)
        entries.append({"path": rel, "rows": row_count(dst), "sha256": sha(dst), "description": rel.rsplit("/", 1)[-1]})
    contents = [
        "# DecisionRAG v1.6 continuation export contents",
        "",
        "This append-only v2 package contains sanitized exports and receipts only; unsanitized raw traces are excluded.",
        "Raw-trace hashes and record counts are retained in the D1 and D2 raw-trace manifest files.",
        "",
        "| path | rows | SHA-256 | description |",
        "|---|---:|---|---|",
    ]
    contents.extend(f"| `{e['path']}` | {e['rows']} | `{e['sha256']}` | {e['description']} |" for e in entries)
    contents += ["", "## Requested files not present", "", "- No required v1.6 D1/D2 export is missing.", "- Raw native traces are intentionally not included; only their hashes and counts are included."]
    (stage / "CONTENTS.md").write_text("\n".join(contents) + "\n", encoding="utf-8")
    bad = []
    for p in stage.rglob("*"):
        if p.is_file():
            t = p.read_text(encoding="utf-8", errors="replace")
            if re.search(r"<REDACTED_PATH><REDACTED_USER>|<REDACTED_USER>|<redacted>|GPU-[0-9A-Fa-f-]{8,}", t, flags=re.I):
                bad.append(str(p.relative_to(stage)))
    if bad:
        raise SystemExit(f"export hygiene failure: {bad}")
    if ZIP.exists():
        raise SystemExit(f"refusing to overwrite existing package: {ZIP}")
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    print(json.dumps({"zip": str(ZIP.relative_to(ROOT)), "size_bytes": ZIP.stat().st_size, "sha256": sha(ZIP), "staging": str(stage), "files": len(entries) + 1}, indent=2))
    print("\n".join(contents))


if __name__ == "__main__":
    main()
