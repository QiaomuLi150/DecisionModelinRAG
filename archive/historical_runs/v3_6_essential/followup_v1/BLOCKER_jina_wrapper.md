# Jina wrapper verification blocker

Status: `BLOCKED_UNVERIFIED`

The frozen follow-up protocol requires the corrected Jina wrapper to reproduce the
historical `jina_single_8192` selected lists at least 99% exactly on the matched
valid requests before any new Jina quality or timing result can be treated as a
verified replacement measurement.

## Matched comparison

- Historical trace: `results/v3_6_essential/stage1_quality/raw/jina_single_8192__8192_reuse.jsonl`
- Corrected rerun: `results/v3_6_essential/followup_v1/stage1_jina/raw/jina_rerun_combined.jsonl`
- Canonical-ID intersection: 1,384
- Matched valid requests after excluding the 32 historical failures: 1,352
- Exact selected-list agreement: 1,126 / 1,352 = 83.284%
- Selected-set agreement: 1,276 / 1,352 = 94.379%
- Corrected rerun builder-infeasible rows for this policy: 4; no native `None` return was observed in the pilot or full rerun
- Historical explicit wrapper errors: 32 `TypeError: cannot unpack non-iterable NoneType object` records, all in the robotics domain

The validity gate on budgetable inputs is satisfied for the corrected rerun, but
the selected-list agreement gate is not. The corrected run therefore remains a
diagnostic artifact and must not be substituted for the frozen historical Jina
quality result or used to claim a verified Jina timing/quality correction.

## Interpretation

The observed disagreements are consistent with score and near-tie differences
between the historical and corrected executions, but the existing evidence does
not identify whether the cause is model revision/runtime behavior, input
normalization, score parsing, tie handling, or the old wrapper. The historical
32 TypeError rows contain no native model-return payload, so they cannot prove
that the model returned `None`; the wrapper/normalization path is the conservative
fault boundary.

Both original and corrected raw traces are preserved and hashed in the follow-up
provenance artifacts. No rerun was performed to seek a favorable result. Any
future Jina comparison must first freeze a separately approved compatibility
protocol and resolve this blocker.
