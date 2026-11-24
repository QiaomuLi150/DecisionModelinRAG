# E2-P0v2 Hypotheses and Primary Contrasts

## H1 — replacement capability

A predefined native decision-model policy is noninferior to a predefined specialist when the paired lower confidence bound for `decision_quality - specialist_quality` exceeds `-epsilon`. A nonsignificant superiority test is not noninferiority. Recommendation for owner selection: evaluate Clef-Flash native optimized against one specialist (BGE-v2-m3 or Qwen3-VL-Reranker-2B) only after native contracts and the holdout are approved.

## H2a — candidate-context effect

Within one checkpoint, objective, document strings, and final Top-5 rule, compare independent document relevance with joint candidate-visible relevance. The treatment is additional candidate context; it is not described as a pure architecture effect.

## H2b — selection-policy effect

Use one frozen, globally comparable score vector for all 20 candidates. Apply unconstrained Top-5 and one-winner-per-authenticated-quartet Top-5. The historical categorical Clef/Laya G5 policies are separate baselines. Quartet-normalized probabilities may never be concatenated.

## H2c — conditional evidence gain

Exploratory only: estimate incremental gain conditioned on already selected evidence, without gold labels or true utility in the prompt. Count all iterative passes and do not call the estimate true marginal utility.

## H3/H4

H3 prespecifies interactions with candidate count, length, gold support cardinality, redundancy, order, overflow and truncation. H4 evaluates complete selection quality against p50/p95, sustainable QPS under a numeric SLO, GPU-seconds, memory and supported cost units.

## Primary family recommendation (not frozen)

Recommend one primary H1 pair after adapter review: Clef-Flash native complete Top-5 versus a single approved specialist, with Track A/B-style metrics reported separately and equal-dataset macro only as a secondary summary. The owner must choose the specialist, epsilon, holdout and correction family before TEST.
