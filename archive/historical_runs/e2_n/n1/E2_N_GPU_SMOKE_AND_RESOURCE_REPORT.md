# E2-N N1 GPU Smoke and Resource Report

Status: `PASS_WITH_LIMITATIONS`

This report is generated only from completed N1 traces; it performs no model calls.

## Model gates

| Model | Rows | Output-complete | Physical-forward evidence | Input hashes | Gate |
|---|---:|---:|---:|---:|---|
| bge | 10 | 10 | 10 | 10 | PASS |
| clef | 20 | 20 | 20 | 20 | PASS |
| laya | 10 | 10 | 10 | 10 | PASS |
| qwen | 10 | 10 | 0 | 10 | PASS_WITH_TELEMETRY_LIMITATION |

## Resource and integrity notes

- Clef ran on GPU0; BGE and Laya ran on idle fallback GPU6. GPUs1/2/4 and other active workloads were not interrupted.
- No checkpoint downloads were authorized or performed. All local checkpoints were loaded from pinned snapshots.
- Qwen used the existing local service on port 8002. Its response alignment and raw-response hashes passed, but the service did not expose physical-forward telemetry; the N2/N3 Qwen branch is therefore fail-closed unless a durable service-level telemetry source is supplied.
- Laya emitted a checkpoint temperature warning. The warning is retained as a calibration limitation; no score ordering change was inferred.
- Jev calls: `0`; paid API calls: `0`; model training: `0`; production-service restarts: `0`.

## Artifact provenance

The machine-readable manifest contains SHA-256 hashes for both raw traces and summaries.
