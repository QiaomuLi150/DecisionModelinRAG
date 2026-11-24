# v3.6 GPU Runtime Repair Record

Status: PASS for the corrected execution context.

The initial Stage 1 attempt was launched from a device-isolated Python
context. `nvidia-smi` could enumerate the host GPUs, but PyTorch reported zero
CUDA devices and each request failed before a model forward with
`MODEL_LOAD_ERROR: No CUDA GPUs are available`. Those raw traces are preserved
under the original `stage1_quality/raw/` namespace and are not treated as
scientific model failures.

A non-destructive CUDA smoke test in the GPU-capable execution context reported
one visible NVIDIA A100 device and completed a CUDA matrix multiplication. A
one-query Jina smoke run then completed all six new Stage 1 policy records
successfully on GPU slot 4.

The corrected full sweep writes only to the append-only
`stage1_quality_v2/raw/` namespace. It uses the existing pinned checkpoints,
frozen inputs and frozen protocol; no historical result was overwritten.

This is an execution-environment deviation under the v3.6 conservative
decision rules. The original sandbox failure is retained as a setup failure,
and the corrected GPU-capable run is the scientific Stage 1 attempt.
