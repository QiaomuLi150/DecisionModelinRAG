# E6-2 R0 Native Top-5 Contract Review

All eight proposed models have a machine-checkable Top-5 adapter record in E6-1/E6-1B, with the following scientific qualification:

- PPLX, Liquid, StartLux and Amos expose categorical/system-choice readouts. Sorting candidate-aligned probabilities is a **derived ranking policy**; it is not joint-set optimization or evidence-complementarity proof.
- Jina exposes native listwise reranking; Qwen3-4B, BGE-v2-m3 and Qwen3-0.6B expose within-view pairwise/cross-encoder scores. Their scores are comparable only within each submitted candidate view.
- All successful records require exactly five unique canonical IDs from the submitted candidate view, deterministic ties and explicit invalid-output status.
- PPLX, Liquid, Jina, StartLux and Amos showed 0/3 set-level reversal matches in the small E6-1/E6-1B diagnostic. Qwen3-4B, BGE and Qwen3-0.6B showed 3/3. These are diagnostics, not population estimates.
- Amos retains six K20 strict 16,384-token failures. Its physical forward count remains `NOT_MEASURED`; no estimate is substituted.

Admission is therefore conditional on the frozen failure policy and authentic future candidate slates. It does not imply quality equivalence, superiority, calibrated probabilities, or production readiness.
