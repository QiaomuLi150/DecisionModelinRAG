# E5-A Protocol and Pre-Outcome Freeze

Status: `FROZEN_BEFORE_EXPLORATORY_OUTCOMES`.

Primary contrast: PPLX Decider minus Jina v3.5, identical frozen natural Top-20 candidate IDs and canonical text, exactly five unique outputs.

Primary endpoint: set-aware unconditional CSSR-U; a query succeeds when any prespecified valid mapped annotated support set of cardinality 1–5 is wholly contained in the selected Top-5. Alternative sets are not unioned. Natural retrieval misses are unconditional failures. The original first-stage Top-5 is a separate retrieval-only control.

Primary estimator: equal-source-group mean of paired query-level differences over groups containing eligible questions. Uncertainty: 5,000 paired source-group bootstrap replicates, seed 731, two-sided percentile 95% interval. Query-weighted and valid-output sensitivities are secondary.

Native contracts, prompts, text caps, output parsing, tie breaking, retry policy, GPU identities, artifact paths and model revisions are frozen in `E5_A_IMMUTABLE_EXECUTION_MANIFEST.json`.

No E5-F payload, reserved outcome, reader answer, Jev, paid API, new retrieval or label-dependent routing is permitted.
