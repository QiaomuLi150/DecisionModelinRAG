#!/usr/bin/env bash
set -euo pipefail

# Idempotent entry point for the frozen v2.1 replication.  Stage 0 is the
# only admissible stage until both exact BM25 Top-20 input contracts pass.
# The Stage-0 script never performs model calls and is safe to resume.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python3}" scripts/v2_1_replication_stage0.py
