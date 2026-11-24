#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
PYTHON="./Production_Project/MultiModalRAG/venv/bin/python"
cd "$ROOT"
"$PYTHON" -m py_compile results/non_jev/v2/reviewer_response_final_v1_0/07_REPRODUCIBLE_ANALYSIS/*.py
"$PYTHON" results/non_jev/v2/reviewer_response_final_v1_0/07_REPRODUCIBLE_ANALYSIS/run_review_analysis.py
echo "REVIEWER_RESPONSE_CPU_SMOKE_PASS"
