# E6-3 R1 Reader Analysis Protocol

R1 freezes reader outputs only. It does not compute answer-reference agreement, citation validity, grounding, abstention quality, factual correctness, or model comparisons.

The primary R2 population is the 200-question representative cohort. The PPLX-versus-Jina R2 contrast will be paired by sample ID using the frozen output-status denominator. The disagreement (up to 60) and 64 no-positive populations remain diagnostic and are never added to the representative denominator.

R2 will use the authentic GaRAGe `answer_generate` reference field only after R1 output freeze, with answer-reference agreement explicitly labeled a proxy. Citation marker validity, selected-passage attribution, output failures, context failures and abstention will be reported separately. No token-overlap score will be called factual correctness. Proposed uncertainty is 5,000 paired query bootstrap resamples, seed 731, two-sided percentile intervals; component sensitivity is separately qualified because GaRAGe source independence is not certified.

No R2 scoring, reader adjudication, LLM judging, reserved access, Jev call, paid API call, selector rerun or new candidate pool is permitted in R1.
