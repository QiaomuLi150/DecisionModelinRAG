# E2-M1 Structural and Mechanism Analysis

**Stage status:** S0 smoke and S1 CAL status are `PASS`. Full exploratory traces are analyzed only when both branches have authentic `ok` outputs.

## Mechanism interpretation

- M1-A pointwise binary and M1-B native joint binary use the same Clef checkpoint, binary `noul` objective, canonical document views and candidate IDs.
- Joint scoring is a single native record with 20 candidate-specific binary fields. It is not five quartet calls and is not a calibrated independent-probability claim.
- M1-C policies reuse the same score vector; no rescoring occurs between global and quartet selection.
- Historical Clef-G5 remains a separate categorical reference and is never merged into the global score vector.

## Context feasibility

The runner records actual encoded token counts per query. K=20 is considered feasible only for rows that successfully encode and execute without silent truncation. A failure is `JOINT_K20_INFEASIBLE` or an explicit model/runtime error; it is not replaced by reduced-K data.

## Ordering and policy diagnostics

- Completed comparable pointwise/joint query rows: `4267`.
- Queries whose full candidate ordering changed between pointwise and joint scores: `4267`.
- Valid five-group/four-candidate group contracts checked: `4267`.
- Selection cardinality/group-contract violations: `0`.

## Quality interpretation

The policy CSV and paired statistics report Track A and Track B separately, with unconditional eligible and retrieval-conditional denominators. The paired intervals use 5,000 query-level bootstrap resamples with seed 731 and are exploratory follow-up evidence, not untouched confirmation. Tiny subgroup claims are not promoted to significance claims.

## Limitations

Any missing/error rows, context overflows, unmeasured physical GPU active seconds, or unavailable native score semantics remain explicit. The proposed 1-second SLO is not enforced here. No reader generation or new retrieval was run.
