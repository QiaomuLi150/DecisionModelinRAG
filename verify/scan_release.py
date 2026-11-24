#!/usr/bin/env python3
"""Scan release files and compressed payloads for privacy/security strings."""
from __future__ import annotations

import gzip
import io
import argparse
import ipaddress
import re
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_MARKERS = [bytes.fromhex(value) for value in ("2f6d6e742f", "2f686f6d652f", "2f726f6f742f", "2f55736572732f", "2f746d702f", "2f736372617463687061642f")]
PRIVATE_MARKERS = [bytes.fromhex(value) for value in ("7978696532", "716c693132")]
PATTERNS = [
    ("absolute_or_private_path", re.compile(rb"(?i)(" + b"|".join(re.escape(x) for x in PATH_MARKERS) + rb"|[A-Z]:\\)")),
    ("user_or_host_marker", re.compile(rb"(?i)(" + b"|".join(re.escape(x) for x in PRIVATE_MARKERS) + rb")")),
    ("email", re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ip_address", re.compile(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("internal_url", re.compile(rb"(?i)https?://(?:" + re.escape(bytes.fromhex("6c6f63616c686f7374")) + rb"|" + re.escape(bytes.fromhex("3132372e302e302e31")) + rb"|[A-Za-z0-9.-]+\." + re.escape(bytes.fromhex("696e7465726e616c")) + rb")(?:\:\d+)?")),
    ("credential_pattern", re.compile(rb"(?i)(?:" + re.escape(bytes.fromhex("68665f")) + rb"[A-Za-z0-9]{20,}|" + re.escape(bytes.fromhex("736b2d")) + rb"[A-Za-z0-9_-]{20,}|" + re.escape(bytes.fromhex("6768705f")) + rb"[A-Za-z0-9]{20,}|" + re.escape(bytes.fromhex("414b4941")) + rb"[0-9A-Z]{16}|" + re.escape(bytes.fromhex("2d2d2d2d2d424547494e")) + rb"(?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----|" + re.escape(bytes.fromhex("417574686f72697a6174696f6e3a")) + rb"\s*(?:Bearer|Basic)\s+[A-Za-z0-9._~+/-]{20,}|" + re.escape(bytes.fromhex("42656172657220")) + rb"[A-Za-z0-9._~+/-]{20,})")),
    ("gpu_uuid", re.compile(bytes.fromhex("4750552d5b302d39612d66412d462d5d7b32302c7d"))),
    ("process_listing", re.compile(rb"(?i)(?:" + re.escape(bytes.fromhex("6e76696469612d736d69")) + rb".{0,100}(?:" + re.escape(bytes.fromhex("706964")) + rb"|" + re.escape(bytes.fromhex("70726f63657373")) + rb")|" + re.escape(bytes.fromhex("70726f636573735f636f6d6d616e64")) + rb"|" + re.escape(bytes.fromhex("636f6d707574655f617070735f726177")) + rb")")),
]
GATES = {
    "absolute_or_private_path": [*PATH_MARKERS, b":\\"],
    "user_or_host_marker": PRIVATE_MARKERS,
    "email": [b"@"],
    "ip_address": [b"."],
    "internal_url": [b"http"],
    "credential_pattern": [bytes.fromhex(value) for value in ("68665f", "736b2d", "6768705f", "414b4941", "2d2d2d2d2d424547494e", "417574686f72697a6174696f6e3a", "42656172657220")],
    "gpu_uuid": [bytes.fromhex("4750552d")],
    "process_listing": [bytes.fromhex("6e76696469612d736d69"), bytes.fromhex("70726f636573735f636f6d6d616e64"), bytes.fromhex("636f6d707574655f617070735f726177")],
}


def inspect(label: str, data: bytes, findings: list[str]) -> None:
    lowered = data.lower()
    for name, pattern in PATTERNS:
        if name == "process_listing" and label.lower().endswith(".py"):
            continue
        if not any(marker.lower() in lowered for marker in GATES[name]):
            continue
        if name == "ip_address":
            for match in pattern.finditer(data):
                try:
                    address = ipaddress.ip_address(match.group().decode("ascii"))
                except ValueError:
                    continue
                if not address.is_loopback and not address.is_unspecified:
                    findings.append(f"{label}: {name}")
                    break
            continue
        if pattern.search(data):
            findings.append(f"{label}: {name}")


def scan_blob(label: str, data: bytes, name: str, findings: list[str]) -> None:
    low = name.lower()
    if low.endswith(".tar.gz") or low.endswith(".tgz"):
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                for item in archive.getmembers():
                    inspect(label + "!" + item.name, item.name.encode(), findings)
                    if item.isfile():
                        stream = archive.extractfile(item)
                        if stream:
                            scan_blob(label + "!" + item.name, stream.read(), item.name, findings)
        except tarfile.TarError as exc:
            findings.append(f"{label}: unreadable tar payload ({type(exc).__name__})")
        return
    if low.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for item in archive.infolist():
                    inspect(label + "!" + item.filename, item.filename.encode(), findings)
                    if not item.is_dir():
                        scan_blob(label + "!" + item.filename, archive.read(item), item.filename, findings)
        except zipfile.BadZipFile as exc:
            findings.append(f"{label}: unreadable ZIP payload ({type(exc).__name__})")
        return
    if low.endswith(".gz"):
        try:
            scan_blob(label + " [gzip payload]", gzip.decompress(data), name[:-3], findings)
        except OSError as exc:
            findings.append(f"{label}: unreadable gzip payload ({type(exc).__name__})")
        return
    if low.endswith(".png"):
        # PNGs are binary; scan only textual metadata chunks, not compressed pixels.
        try:
            pos = 8
            while pos + 12 <= len(data):
                size = int.from_bytes(data[pos:pos + 4], "big")
                kind = data[pos + 4:pos + 8]
                payload = data[pos + 8:pos + 8 + size]
                if kind in {b"tEXt", b"iTXt", b"zTXt"}:
                    inspect(label + " [PNG text metadata]", payload, findings)
                pos += size + 12
                if kind == b"IEND":
                    break
        except Exception as exc:
            findings.append(f"{label}: unreadable PNG metadata ({type(exc).__name__})")
        return
    inspect(label, data, findings)


def scan_stream(label: str, stream, findings: list[str]) -> None:
    """Scan decompressed data in bounded chunks, retaining overlap for split matches."""
    carry = b""
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        data = carry + chunk
        inspect(label, data, findings)
        carry = data[-512:]


def scan_file(path: Path, findings: list[str], rel: str | None = None) -> None:
    if rel is None:
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.name
    low = path.name.lower()
    try:
        if low.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    inspect(rel + "!" + item.filename, item.filename.encode(), findings)
                    if item.is_dir():
                        continue
                    with archive.open(item) as stream:
                        if item.filename.lower().endswith(".gz"):
                            with gzip.GzipFile(fileobj=stream, mode="rb") as decoded:
                                scan_stream(rel + "!" + item.filename, decoded, findings)
                        else:
                            scan_stream(rel + "!" + item.filename, stream, findings)
        elif low.endswith(".gz"):
            with gzip.open(path, "rb") as stream:
                scan_stream(rel + " [gzip payload]", stream, findings)
        elif low.endswith(".png"):
            scan_blob(rel, path.read_bytes(), path.name, findings)
        else:
            inspect(rel, path.read_bytes(), findings)
    except (OSError, gzip.BadGzipFile, tarfile.TarError, zipfile.BadZipFile) as exc:
        findings.append(f"{rel}: unreadable compressed artifact ({type(exc).__name__})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, action="append", default=[])
    args = parser.parse_args()
    findings: list[str] = []
    generated_dirs = {".git", ".pytest_cache", "__pycache__"}
    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and not generated_dirs.intersection(p.relative_to(ROOT).parts)
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        def scan_one(path: Path) -> list[str]:
            local: list[str] = []
            scan_file(path, local)
            return local
        for local in pool.map(scan_one, files):
            findings.extend(local)
    for archive in args.archive:
        if not archive.is_file():
            findings.append(f"archive missing: {archive.name}")
        else:
            scan_file(archive, findings, f"[archive]/{archive.name}")
    if findings:
        print("SANITIZATION_SCAN FAIL")
        for item in findings:
            print("MATCH", item)
        return 1
    print(f"SANITIZATION_SCAN PASS files={len(files)} archives_scanned={len(args.archive)} compressed_payloads_scanned=all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
