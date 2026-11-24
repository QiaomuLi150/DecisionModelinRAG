# E6-3A Reader Reference-Metric Audit

The frozen 1,565 reader rows were recomputed offline. The stored normalized token-F1 and exact-match values were reproduced with `0` mismatches. No reader request was made.

The primary 200-question representative contrast remains PPLX minus Jina: PPLX mean `0.116365493883`, Jina mean `0.119772228125`, paired mean `-0.003406734242`, n=`200`. This is lexical reference overlap against the single stored `answer_generate` field, not factual correctness.

The frozen score rows expose one reference field only; multiple acceptable references and a formal answer-type taxonomy are not available in the scored artifact. `answer_validate` is retained as a source validation flag and was not promoted to an answer-type label. Five rows from one no-positive question have no nonempty reference and remain outside the metric denominator. Exact match is uniformly descriptive and does not establish correctness.

The low overlap is therefore compatible with long-form/short-answer mismatch, single-reference limitations, abstentions, and lexical variation. The audit does not convert mismatch into factual error and does not alter the primary endpoint. Citation-marker validity and annotated relevance remain structural/annotation diagnostics; claim-level factual correctness is `NOT_VERIFIED`.
