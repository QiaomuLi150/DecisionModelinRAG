#!/usr/bin/env python3
"""Fetch the official BRIGHT dataset at a caller-pinned Hugging Face revision.

The exact historical dataset revision and canonical shard checksums were not
preserved in the experiment workspace. This helper verifies the supplied
revision and checks that released candidate IDs exist; it cannot certify that
the fetched revision is byte-identical to the historical benchmark snapshot.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protocols/bright_download_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="immutable Hugging Face dataset commit hash")
    parser.add_argument("--task", required=True, help="BRIGHT task/config, e.g. biology or economics")
    parser.add_argument("--subset", choices=("examples", "documents", "long_documents"), required=True)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install the optional `datasets` package before running this download helper.")
    if len(args.revision) < 40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        raise SystemExit("--revision must be an immutable hexadecimal commit hash")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candidate_file = ROOT / manifest["released_candidate_ids"]["path"]
    digest = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    if digest != manifest["released_candidate_ids"]["sha256"]:
        raise SystemExit("released candidate-ID artifact checksum mismatch")
    data = load_dataset("xlangai/BRIGHT", args.subset, revision=args.revision)[args.task]
    ids = {str(row.get("id")) for row in data if row.get("id") is not None}
    if not ids:
        raise SystemExit("the selected official split contained no id field")
    print(f"BRIGHT_DATASET_LOADED repo=xlangai/BRIGHT revision={args.revision} subset={args.subset} task={args.task} rows={len(data)} ids={len(ids)}")
    if args.subset == "documents":
        candidate_ids: set[str] = set()
        with gzip.open(ROOT / "data/candidates/bright_top20_ids.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    candidate_ids.update(map(str, json.loads(line).get("candidate_ids", [])))
        missing = sorted(candidate_ids - ids)
        print(f"RELEASED_CANDIDATE_ID_COVERAGE found={len(candidate_ids)-len(missing)} expected={len(candidate_ids)}")
        if missing:
            print(f"ID_COVERAGE_INCOMPLETE count={len(missing)}; compare all BRIGHT task shards before scoring")
    print("LIMITATION: historical canonical shard SHA-256 values are absent; this is not a byte-identity reproduction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
