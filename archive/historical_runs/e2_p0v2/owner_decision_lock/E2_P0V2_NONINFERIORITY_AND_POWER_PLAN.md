# E2-P0v2 Noninferiority and Power Plan

## Proposal

Propose `epsilon = 0.05`, an absolute five-percentage-point bound on complete annotated support retention. This is not approved or frozen.

Operationally, the proposal says that a replacement decision policy may lose at most five complete-support successes per 100 comparable queries, in exchange for a separately measured operational benefit. It may be too permissive for safety-critical evidence selection and too strict for low-cost exploratory routing; the owner must justify it for the target use case. Sensitivity should include smaller margins such as 0.03 after owner review, but those values are not selected here.

## Primary test

For each paired query, define `D_q = CSSR_decision(q) - CSSR_specialist(q)` on the same frozen candidate slate and eligible population. Establish noninferiority only if the lower 95% paired confidence bound is greater than `-epsilon`. A nonsignificant superiority test is not noninferiority.

Use query-level paired bootstrap with source-document cluster resampling where applicable. Report Track A and Track B separately; an equal-dataset macro is secondary and must not replace dataset-level estimates. Holm adjustment applies only after the owner defines the primary family and alpha.

## Power result at this stage

`NOT_ESTIMATED_FOR_CONFIRMATORY_H1`: the proposed comparator Qwen3-Reranker-4B has no verified local checkpoint, adapter, or paired historical outcomes in this workspace. Existing M0 Clef/Laya structural oracles are not a valid variance estimate for the proposed Clef/Qwen3-Reranker-4B H1 contrast. No fabricated power number is reported.

Required inputs for the calculation:

- approved epsilon and alpha;
- target power;
- exact fresh holdout `n` and dataset allocation;
- planning estimate of paired discordance/effect variance from an approved CAL-only pilot;
- primary-family size and multiplicity rule.

After those inputs are approved, calculate power/precision before TEST using only CAL or historical data for planning. Do not use TEST outcomes for configuration selection.
