#!/usr/bin/env python3
"""Build the sanitized, shareable v1.5 evidence ZIP."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V15 = ROOT / "results/v1_5"
ZIP_PATH = V15 / "decisionrag_v1_5_control_exports.zip"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_count(path: Path) -> int | str:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            return sum(1 for _ in csv.DictReader(fh))
    if path.suffix.lower() == ".jsonl":
        return sum(1 for line in path.open() if line.strip())
    return "N/A"


def description(rel: str) -> str:
    name = Path(rel).name
    if name == "V1_5_FINAL_REPORT.md":
        return "final v1.5 scientific report"
    if name == "V1_5_STATUS.json":
        return "machine-readable stage gate"
    if name == "V1_5_PROVENANCE_MANIFEST.json":
        return "v1.5 artifact and historical-source hash manifest"
    if name == "V1_5_REGRESSION_REPORT.md":
        return "focused and repository regression report"
    if name == "README.md":
        return "artifact guide"
    if name == "PROTOCOL.json":
        return "frozen experiment protocol"
    if name == "BLOCKER.json":
        return "explicit stage blocker record"
    if name == "STATUS.json":
        return "stage export status"
    if name == "C2_RAW_TRACE_MANIFEST.json":
        return "historical raw-trace hashes and counts"
    if name == "C2_INPUT_INTEGRITY.json":
        return "C2 input and frozen-slate integrity record"
    if name == "C2_RUN_RECEIPT.json":
        return "C2 execution receipt"
    if name == "latency_perrequest.csv":
        return "normalized request-level latency records"
    if name == "fallback_cost.csv":
        return "CPU-only frozen-slate fallback measurements"
    if name == "latency_policy_summary.csv":
        return "policy latency and failure summary"
    if name == "latency_ratios.csv":
        return "query-cluster bootstrap p95 ratios"
    return "v1.5 evidence artifact"


def sanitize(text: str) -> str:
    # The selected files should already be relative and metadata-free. These
    # replacements are a fail-safe for the shareable copy only.
    text = re.sub(r"/(?:mnt|home|root)/[^\s\"'`,}]+", "[absolute-path-redacted]", text)
    text = re.sub(r"(?im)^.*(?:" + "host" + "name" + r"|username|nvidia-smi|process_listing).*\n", "", text)
    return text


def main() -> None:
    selected: list[Path] = []
    for path in sorted(V15.rglob("*")):
        if not path.is_file():
            continue
        if path == ZIP_PATH:
            continue
        rel = path.relative_to(V15)
        # The package contains reports, protocols, blockers, status files and
        # export folders. It does not copy historical raw traces or scripts.
        if rel.parts[0] in {"c1_two_stage_specialist", "c2_latency_uncertainty", "c3_concurrency", "c4_winnow_rerun"} or rel.name in {"README.md", "V1_5_FINAL_REPORT.md", "V1_5_PROVENANCE_MANIFEST.json", "V1_5_REGRESSION_REPORT.md", "V1_5_STATUS.json"}:
            selected.append(path)

    with tempfile.TemporaryDirectory(prefix="decisionrag_v15_package_") as tmp:
        stage = Path(tmp)
        for source in selected:
            rel = source.relative_to(V15)
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() in {".json", ".jsonl", ".csv", ".md"}:
                target.write_text(sanitize(source.read_text()))
            else:
                shutil.copy2(source, target)

        forbidden = re.compile(r"<REDACTED_PATH><REDACTED_USER>|<REDACTED_USER>|<redacted>|nvidia-smi")
        hits = []
        for path in stage.rglob("*"):
            if path.is_file() and forbidden.search(path.read_text(errors="replace")):
                hits.append(path.relative_to(stage).as_posix())
        if hits:
            raise RuntimeError("sanitization grep found matches: " + ", ".join(hits))

        rows = []
        for path in sorted(stage.rglob("*")):
            if not path.is_file() or path.name == "CONTENTS.md":
                continue
            rel = path.relative_to(stage).as_posix()
            rows.append({
                "path": rel,
                "records": record_count(path),
                "sha256": digest(path),
                "description": description(rel),
            })
        contents = [
            "# DecisionRAG v1.5 Control-Experiment Exports",
            "",
            "This archive contains sanitized v1.5 protocols, stage status/blocker records, CPU-only C2 exports, and reports. Historical raw traces are not copied; their relative paths, hashes and record counts are recorded in the C2 manifests.",
            "",
            "| Path | Records | SHA-256 | Description |",
            "|---|---:|---|---|",
        ]
        contents.extend(f"| `{r['path']}` | {r['records']} | `{r['sha256']}` | {r['description']} |" for r in rows)
        contents += [
            "",
            "C1, C3 and C4 are blocked at the dedicated-A100 preflight gate; their status files contain zero model calls. C2 is complete offline. No absolute paths, user names, machine names or GPU process listings are included.",
        ]
        (stage / "CONTENTS.md").write_text("\n".join(contents) + "\n")

        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())

    package = {
        "path": ZIP_PATH.relative_to(ROOT).as_posix(),
        "bytes": ZIP_PATH.stat().st_size,
        "sha256": digest(ZIP_PATH),
        "grep_check": "PASS",
        "file_count": len(rows) + 1,
        "contents": "CONTENTS.md",
    }
    (V15 / "V1_5_PACKAGE_SUMMARY.json").write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    print(json.dumps(package, sort_keys=True))


if __name__ == "__main__":
    main()
