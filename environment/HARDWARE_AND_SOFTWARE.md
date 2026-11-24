# Hardware and software provenance

The paper's principal serving measurements were recorded on NVIDIA A100 80 GB hardware using serial requests, with same-session Jina anchors in the timing protocols. This package does not claim that every historical run used isolated hardware: original contended CQADupStack Pass-B traces are retained as diagnostics, and the clean paired sessions are the paper's latency source.

| Component | Recorded information included here | Limitation |
|---|---|---|
| GPU class | NVIDIA A100 80 GB | Exact device UUIDs and process lists are intentionally removed. |
| Driver/CUDA | Not consistently available in release-safe run manifests | Not stated as a verified single version. |
| PyTorch/Transformers/vLLM | Run-specific metadata is uneven; the CPU verification package does not depend on them | Do not infer a version where it is absent. |
| Analysis runtime | Python 3.12.3; Tier 0/1 verifier uses only the standard library | This is the packaging/verification runtime, not a claim that every experiment used it. |
| Request policy | Per frozen experiment protocol; batch size 1 for clean serial timing; complete latency includes failures according to each protocol | See sanitized protocol copies and result metadata. |

No GPU was used for this package build or verification.
