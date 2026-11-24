# E5-D/E5-C Deviations and Stop Rules

Status: `FROZEN_BEFORE_NEW_INFERENCE`.

- Representative sampling used seed **731**, uniform source-group sampling over **200** groups, then one question per selected group. The draw occurred before E5-A success labels were read.
- Diagnostic enrichment is the complete historical E5-A discordance set: 31 PPLX-only and 24 Jina-only cases. It is never pooled with representative population estimates.
- E5-D requires GPU5 UUID `<REDACTED_GPU_UUID>` to be observable and compute-idle immediately before model loading. If not, serving is `E5_D_ISOLATION_BLOCKED`; no other GPU is silently substituted.
- Timed selector requests have concurrency 1, no retry, 30-second threshold, and retain every assigned result.
- E5-C uses the fixed Qwen reader contract from E4-1. T1 is a deterministic reference-agreement proxy; T2 and T3 are not promoted unless independently available.
- Any checkpoint, prompt, candidate, policy, holdout, Jev, paid-API or post-outcome protocol change stops the affected phase.
