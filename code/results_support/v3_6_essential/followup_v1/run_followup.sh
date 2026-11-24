#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT/RAG_system/Decision_Model_in_RAG"
echo "Resume N1 A/B and N2 C from their checkpoint files; GPU-Drex is a separate partial branch."
echo "A: MultiModalRAG/venv/bin/python scripts/v3_6_followup_n1.py --session A --gpu 3 --resume"
echo "B: MultiModalRAG/venv/bin/python scripts/v3_6_followup_n1.py --session B --gpu 6 --resume"
echo "C: MultiModalRAG/venv/bin/python scripts/v3_6_followup_n2.py --gpu 5 --resume"
echo "After all intended sessions finish: python3 scripts/v3_6_followup_closure.py --package"
