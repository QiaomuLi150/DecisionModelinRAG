# DecisionRAG v3.6 Test and Integrity Report

## Focused checks

- Python compilation: **PASS** for `v3_6_serving.py`, `v3_6_quality_analysis.py`, `v3_6_finalize.py`, `v3_6_report.py`, and `v3_6_package.py`.
- Frozen Stage 0 input and L=8192 identity receipts: **PASS**.
- Stage 1 quality join: **PASS**, 31,832 per-query rows and zero missing joins.
- Raw trace accounting: **PASS**, three serving namespaces reconciled to 12,960 derived raw records; every preserved raw file has a SHA-256 and record count.
- Raw-trace manifest recheck: **PASS**; all seven packaged raw-trace hashes and counts match the source files.
- Export hygiene: **PASS**; no `<redacted>, `<redacted>, `<redacted>, user-name, host-name, `nvidia-smi`, or GPU UUID matches in the staging directory.
- Archive exclusion: **PASS**; the ZIP contains no raw JSONL traces, checkpoints, passage text, process lists or device receipts.
- GPU release check: **PASS** for the two GPUs used by this task; GPU3 and GPU6 were at residual allocator memory only with zero utilization after completion. Other GPUs were not altered.

## Full regression

Command:

```text
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. MultiModalRAG/venv/bin/python -m pytest -q --disable-warnings
```

Result: **302 collected, 302 passed, 0 failed**.

The suite was CPU/offline-only. It made no model-serving requests, Jev calls, paid API calls, or protected-holdout accesses.

## Scientific gate

The v3.6 result remains **PARTIAL**. Stage 0 and Stage 1 pass. Two complete five-policy T1 serving sessions pass their internal accounting, but the third namespace contains only Jina policies; T2, solo interference, determinism rechecks and optional Stage 4 remain unverified. No three-replicate production-serving or cost claim is made.
