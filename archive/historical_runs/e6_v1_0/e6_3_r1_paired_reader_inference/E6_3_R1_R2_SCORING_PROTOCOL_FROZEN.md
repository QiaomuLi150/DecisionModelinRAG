# E6-3 R1 → R2 Scoring Protocol Freeze

Status: FROZEN_FOR_SEPARATE_R2_AUTHORIZATION

Primary R2 endpoint: answer-reference agreement against the authentic GaRAGe `answer_generate` field, reported only as a reference-agreement proxy with single-reference and answer-type limitations. No factual-correctness claim is permitted.

Secondary R2 endpoints: structural answer parse validity, citation-marker validity, citation attribution to selected passages, cited-passage relevance using the already frozen R5 annotations, explicit abstention, answer rate, context failures and reader output failures. These endpoints must not be conflated.

Primary population: 200 representative questions. Primary contrast: PPLX reader outcome minus Jina reader outcome, paired by sample ID. Diagnostic populations: disagreement cohort and no-positive cohort, analyzed separately. Uncertainty: 5,000 paired query bootstrap replicates, seed 731, two-sided percentile 95% interval, plus explicitly qualified exact-content component sensitivity if supported.

R2 must not access the RESERVED partition, change prompts, change the reader, rerun selectors, or treat reference agreement as independently verified factual correctness.
