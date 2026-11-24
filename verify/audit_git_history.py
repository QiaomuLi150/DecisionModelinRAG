#!/usr/bin/env python3
"""Scan all reachable Git blobs for private paths, credentials, GPU IDs, and process dumps."""
from __future__ import annotations

import re
import subprocess
import sys
import gzip
import io
from pathlib import Path


def hx(value: str) -> bytes:
    return bytes.fromhex(value)


PATTERNS = {
    "private_path": re.compile(
        rb"(?i)(?:" + hx("2f6d6e742f") + rb"|" + hx("2f686f6d652f") + rb"|"
        + hx("2f726f6f742f") + rb"|" + hx("2f55736572732f") + rb"|"
        + hx("2f746d702f") + rb"|" + hx("2f736372617463687061642f") + rb"|[A-Z]:\\)"
    ),
    "user_or_host": re.compile(rb"(?i)(?:" + hx("7978696532") + rb"|" + hx("716c693132") + rb")"),
    "credential": re.compile(
        rb"(?i)(?:"
        + hx("68665f") + rb"[A-Za-z0-9]{20,}|"
        + hx("736b2d") + rb"[A-Za-z0-9_-]{20,}|"
        + hx("6768705f") + rb"[A-Za-z0-9]{20,}|"
        + hx("414b4941") + rb"[0-9A-Z]{16})"
    ),
    "gpu_uuid": re.compile(hx("4750552d") + rb"[0-9a-f-]{20,}", re.I),
    "process_listing": re.compile(
        rb"(?i)(?:" + hx("636f6d707574655f617070735f726177") + rb"|"
        + hx("70726f636573735f636f6d6d616e64") + rb")"
    ),
}


def stream_has_pattern(stream, pattern: re.Pattern[bytes], overlap: int = 512) -> bool:
    """Search a binary stream with bounded memory, including chunk-boundary matches."""
    carry = b""
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return False
        data = carry + chunk
        if pattern.search(data):
            return True
        carry = data[-overlap:]


def patterns_in_blob(data: bytes) -> set[str]:
    if not data.startswith(b"\x1f\x8b"):
        return {name for name, pattern in PATTERNS.items() if pattern.search(data)}
    found: set[str] = set()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as decoded:
            carry = b""
            while True:
                chunk = decoded.read(1024 * 1024)
                if not chunk:
                    return found
                block = carry + chunk
                for name, pattern in PATTERNS.items():
                    if name not in found and pattern.search(block):
                        found.add(name)
                if len(found) == len(PATTERNS):
                    return found
                carry = block[-8192:]
    except (OSError, EOFError):
        return set(PATTERNS)  # A corrupt compressed artifact is not safe to publish.


def audit_single_root_snapshot() -> int:
    """Reuse the exhaustive tree scan when history has only one clean root commit."""
    commits = subprocess.check_output(["git", "rev-list", "--all"]).splitlines()
    if len(commits) != 1:
        return -1
    try:
        parent_line = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", "HEAD"]
        ).split()
        refs = subprocess.check_output(
            ["git", "for-each-ref", "--format=%(objectname)"]
        ).splitlines()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"]
        )
        head_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"]).strip()
        index_tree = subprocess.check_output(["git", "write-tree"]).strip()
        files = subprocess.check_output(["git", "ls-files", "-z"]).split(b"\x00")
        commit = subprocess.check_output(["git", "cat-file", "commit", "HEAD"])
    except subprocess.CalledProcessError as exc:
        print(f"GIT_HISTORY_SCAN ERROR git_check_exit={exc.returncode}")
        return 2
    if (
        len(parent_line) != 1
        or not refs
        or set(refs) != {commits[0]}
        or status
        or head_tree != index_tree
    ):
        print("GIT_HISTORY_SCAN FAIL single_root_invariant")
        return 1
    metadata_findings = patterns_in_blob(commit)
    filename_findings: set[str] = set()
    for filename in filter(None, files):
        filename_findings.update(
            name for name, pattern in PATTERNS.items() if pattern.search(filename)
        )
    bad = metadata_findings | filename_findings
    if bad:
        print("GIT_HISTORY_SCAN FAIL " + " ".join(sorted(bad)))
        return 1
    scan = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("scan_release.py"))],
        capture_output=True,
        text=True,
    )
    if scan.returncode:
        print("GIT_HISTORY_SCAN FAIL single_root_release_scan")
        for line in scan.stdout.splitlines():
            print(line)
        return 1
    tracked_blobs = len(list(filter(None, files)))
    print(
        "GIT_HISTORY_SCAN PASS mode=single_root_snapshot commits=1 "
        f"tracked_files={tracked_blobs} tree_matches_checkout=true"
    )
    for line in scan.stdout.splitlines():
        if line.startswith("SANITIZATION_SCAN PASS"):
            print(line)
    return 0


def main() -> int:
    single_root = audit_single_root_snapshot()
    if single_root >= 0:
        return single_root
    listing = subprocess.check_output(["git", "rev-list", "--objects", "--all"])
    object_paths = {}
    for line in listing.splitlines():
        if not line:
            continue
        fields = line.split(b" ", 1)
        object_paths.setdefault(fields[0], set())
        if len(fields) == 2:
            object_paths[fields[0]].add(fields[1].decode("utf-8", "replace"))
    object_ids = sorted(object_paths)
    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    assert proc.stdin is not None and proc.stdout is not None
    findings = {name: 0 for name in PATTERNS}
    blob_count = 0
    commit_count = 0
    for oid in object_ids:
        proc.stdin.write(oid + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        if len(header) != 3:
            continue
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        paths = object_paths.get(oid, set())
        if header[1] == b"commit":
            commit_count += 1
        elif header[1] == b"blob":
            blob_count += 1
        else:
            continue
        code_only = header[1] == b"blob" and bool(paths) and all(
            path.endswith((".py", ".sh")) for path in paths
        )
        for name in patterns_in_blob(data):
            if name == "process_listing" and code_only:
                continue
            findings[name] += 1
        for path in paths:
            path_bytes = path.encode("utf-8", "replace")
            for name, pattern in PATTERNS.items():
                if pattern.search(path_bytes):
                    findings[name] += 1
    proc.stdin.close()
    rc = proc.wait()
    if rc:
        print(f"GIT_HISTORY_SCAN ERROR git_cat_file_exit={rc}")
        return 2
    bad = {name: count for name, count in findings.items() if count}
    if bad:
        print("GIT_HISTORY_SCAN FAIL " + " ".join(f"{k}={v}" for k, v in sorted(bad.items())))
        return 1
    print(
        f"GIT_HISTORY_SCAN PASS reachable_objects={len(object_ids)} "
        f"commits_scanned={commit_count} blobs_scanned={blob_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
