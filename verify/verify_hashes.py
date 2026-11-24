#!/usr/bin/env python3
"""Verify release-file checksums and the hashes/counts in the trace manifest."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_records(path: Path) -> int | None:
    name = path.name.lower()
    if name.endswith(".csv"):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return max(0, sum(1 for _ in csv.reader(stream)) - 1)
    if name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    if name.endswith(".jsonl"):
        with path.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    return None


def main() -> int:
    failures: list[str] = []
    checksum_file = ROOT / "CHECKSUMS.sha256"
    if not checksum_file.is_file():
        print("FAIL CHECKSUMS.sha256 missing")
        return 1
    expected: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed checksum line: {line[:80]}")
            continue
        if rel in expected:
            failures.append(f"duplicate checksum entry: {rel}")
        expected[rel] = digest

    generated_dirs = {".git", ".pytest_cache", "__pycache__"}
    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != checksum_file
        and not generated_dirs.intersection(path.relative_to(ROOT).parts)
    }
    missing = sorted(actual_files - expected.keys())
    stale = sorted(expected.keys() - actual_files)
    if missing:
        failures.append("unlisted release files: " + ", ".join(missing))
    if stale:
        failures.append("checksum entries without files: " + ", ".join(stale))
    for rel, digest in sorted(expected.items()):
        path = ROOT / rel
        if not path.is_file():
            continue
        actual = sha256(path)
        if actual != digest:
            failures.append(f"checksum mismatch: {rel}")
    print(f"CHECKSUM_FILES {len(expected)}")

    crosswalk_path = ROOT / "data/SOURCE_CROSSWALK.csv"
    if crosswalk_path.exists():
        with crosswalk_path.open(newline="", encoding="utf-8-sig") as stream:
            crosswalk = list(csv.DictReader(stream))
        checked_targets: dict[str, str] = {}
        for row in crosswalk:
            source_hash = row.get("source_sha256", "")
            release_hash = row.get("release_sha256", "")
            if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash.lower()):
                failures.append(f"invalid source SHA-256 in crosswalk: {row.get('source_relative_path', '')}")
            rel = row.get("release_relative_path", "")
            if len(release_hash) != 64 or any(c not in "0123456789abcdef" for c in release_hash.lower()):
                failures.append(f"invalid release SHA-256 in crosswalk: {rel}")
                continue
            checked_targets.setdefault(rel, release_hash)
            if checked_targets[rel] != release_hash:
                failures.append(f"inconsistent release hashes for crosswalk target: {rel}")
        for rel, digest in sorted(checked_targets.items()):
            path = ROOT / rel
            if not path.is_file():
                failures.append(f"crosswalk target missing: {rel}")
            elif sha256(path) != digest:
                failures.append(f"crosswalk release hash mismatch: {rel}")
        print(f"SOURCE_CROSSWALK_ROWS {len(crosswalk)} TARGETS {len(checked_targets)}")

    manifest_path = ROOT / "raw_traces/TRACE_MANIFEST.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            rel = row.get("release_relative_path", "")
            path = ROOT / rel
            if not path.is_file():
                failures.append(f"trace projection missing: {rel}")
                continue
            if sha256(path) != row.get("release_sha256"):
                failures.append(f"trace projection hash mismatch: {rel}")
            count = count_records(path)
            recorded = row.get("release_record_count", "")
            if count is not None and recorded and count != int(recorded):
                failures.append(f"trace projection record count mismatch: {rel}: {count} != {recorded}")
        print(f"TRACE_MANIFEST_ROWS {len(rows)}")
        print("SOURCE_TRACE_HASHES are recorded; source payloads are intentionally not redistributed")

    protocol_hash_file = ROOT / "protocols/cqadupstack_protocol_original_sha256.txt"
    protocol = ROOT / "protocols/cqadupstack_frozen_protocol.md"
    cohort = ROOT / "protocols/cqadupstack_latency_cohort_ids.jsonl"
    if protocol_hash_file.exists() and protocol.exists():
        recorded = protocol_hash_file.read_text(encoding="utf-8").split()[0]
        actual = sha256(protocol)
        if recorded != actual:
            failures.append("frozen CQADupStack protocol original hash mismatch")
        else:
            print(f"FROZEN_PROTOCOL_SHA256 PASS {actual}")
    if cohort.exists():
        actual = sha256(cohort)
        required = "5d5f34e3b3a9cfd1b547744e1e4417c992e97b86a81edd94f8f175645652961c"
        if actual != required:
            failures.append("frozen CQADupStack latency cohort hash mismatch")
        else:
            print(f"LATENCY_COHORT_SHA256 PASS {actual}")

    if failures:
        for item in failures:
            print("FAIL", item)
        return 1
    print("HASH_VERIFICATION PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
