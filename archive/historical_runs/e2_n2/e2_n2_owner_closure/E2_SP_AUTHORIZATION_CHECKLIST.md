# E2-S/P Authorization Checklist

All items below require explicit owner approval before execution:

- GPU IDs and exclusive execution window, with a contention check.
- Exact model/checkpoint roster and local-service ownership.
- Network/download permission and confirmation that no production service may be restarted or modified.
- Frozen cohort, candidate-slate hashes, arrival rates, concurrency, warm-up, repetition count, and p95 SLO.
- E2-P reader checkpoint, prompt, decoding, evidence presentation, and independently verified reference policy.
- Cache-disabled application policy and raw request/response hashing before normalization.
- Cost-rate assumptions and reporting boundary; no dollar claim without approved rates.
- No fresh holdout, Jev, paid API, training, or protocol tuning.
- Stop-gate owner and artifact-preservation procedure.
