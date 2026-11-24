# E2-N Qwen Telemetry Remediation

The resident Qwen3-VL-Reranker-2B service was not restarted or modified. Its prior service trace remains non-promoted because the service boundary did not expose physical-forward telemetry.

A separate official local checkpoint branch was executed with direct model instrumentation:

- native smoke: 10/10, physical forwards 10/10;
- CAL: 30/30, physical forwards 30/30;
- historical: 4,267/4,267, physical forwards 4,267/4,267;
- actual tensor-input and raw-score hashes are recorded per row;
- gold labels were not included in model-visible inputs;
- Jev calls and paid calls: 0.

The direct branch is a separate scoring contract and must not be silently substituted for the resident service in historical reports. Future service runs must hash raw response bytes before normalization.
