#!/usr/bin/env python3
"""Opt-in download of an explicitly pinned model snapshot outside this repo.

No model files are downloaded by the verification or packaging commands.
Historical abbreviated/missing revisions require a caller-supplied full commit
and an explicit acknowledgement that the revision is not historically verified.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "pplx": ("perplexity-ai/pplx-decider-v1.1-27b", "14c9d90137962d300f3de95633f7b6535e80c5ce"),
    "jina": ("jinaai/jina-reranker-v3.5", None),
    "bge": ("BAAI/bge-reranker-v2-m3", None),
    "qwen06": ("Qwen/Qwen3-Reranker-0.6B", None),
    "qwen4": ("Qwen/Qwen3-Reranker-4B", None),
}
FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--output-dir", required=True, type=Path, help="destination outside this repository")
    parser.add_argument("--revision", help="full immutable commit hash; required for models without a recovered full pin")
    parser.add_argument("--accept-unverified-revision", action="store_true", help="acknowledge this is not proven to match the historical run")
    args = parser.parse_args()

    repo_id, historical_pin = MODELS[args.model]
    revision = args.revision or historical_pin
    if not revision or not FULL_COMMIT.fullmatch(revision):
        raise SystemExit("A full 40-64 character hexadecimal commit revision is required; abbreviated tags are rejected.")
    if historical_pin is None or revision != historical_pin:
        if not args.accept_unverified_revision:
            raise SystemExit("This revision is not a verified historical pin. Supply --accept-unverified-revision to proceed explicitly.")

    output = args.output_dir.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit("Model snapshots must be stored outside the public repository.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("Destination must be absent or empty; refusing to overwrite files.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Install the optional huggingface_hub package to use this download-only helper.")

    path = snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(output))
    print(f"MODEL_SNAPSHOT_DOWNLOADED repo={repo_id} revision={revision} destination={path}")
    if historical_pin is None or revision != historical_pin:
        print("REVISION_STATUS=caller-supplied and not authenticated against the historical run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
