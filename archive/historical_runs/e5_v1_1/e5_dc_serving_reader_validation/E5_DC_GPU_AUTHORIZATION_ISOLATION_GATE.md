# E5-D/E5-C GPU Authorization and Isolation Gate

Pre-inference contract: GPU5 index **5**, UUID `<REDACTED_GPU_UUID>`, NVIDIA A100-SXM4-80GB. It is the only device authorized for matched E5-D timing in this run. GPU6 was observed occupied at preflight and was not substituted.

The final D gate is determined by the immediately pre-load and post-teardown snapshots in `E5_D_*_SUMMARY.json`. Existing unrelated processes must not be terminated. Reader calls use the existing local endpoint and its physical GPU attribution is explicitly `NOT_EXPOSED`.
