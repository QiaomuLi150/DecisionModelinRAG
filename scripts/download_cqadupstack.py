#!/usr/bin/env python3
"""Fetch the official BEIR CQADupStack archive and verify its frozen SHA-256.

This is an opt-in data download; it is not run by the verification suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protocols/cqadupstack_download_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/external/cqadupstack"))
    parser.add_argument("--archive", type=Path, help="reuse an already downloaded official ZIP")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archive_info = manifest["archive"]
    archive = args.archive or (args.output / "cqadupstack.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.archive:
        request = urllib.request.Request(archive_info["official_url"], headers={"User-Agent": "DecisionRAG-data-verifier/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
    actual = digest(archive)
    if actual != archive_info["sha256"]:
        raise SystemExit(f"archive SHA-256 mismatch: {actual}")
    extraction = args.output / "release"
    extraction.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise SystemExit(f"unsafe archive member: {member.filename!r}")
            target = extraction.joinpath(*name.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
    verified = 0
    for forum, files in manifest["forums"].items():
        for kind, info in files.items():
            rel = Path(info["path"])
            matches = list(extraction.rglob(rel.name))
            matches = [p for p in matches if p.as_posix().endswith(rel.as_posix())]
            if len(matches) != 1:
                raise SystemExit(f"expected one extracted file for {forum}/{kind}: found {len(matches)}")
            if digest(matches[0]) != info["sha256"]:
                raise SystemExit(f"file SHA-256 mismatch: {forum}/{kind}/{rel.name}")
            verified += 1
    print(f"CQADUPSTACK_DOWNLOAD_VERIFIED archive_sha256={actual} files={verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
