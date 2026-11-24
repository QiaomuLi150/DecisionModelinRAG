# E4-2 Facet Validity and Native Contract

Facet generation: deterministic question-only algorithm, frozen before inference. Gold/reference/support fields are excluded from facet and selector payloads.

Liquid: official `system_one` with native noul yes probabilities. Pointwise uses one candidate plus all frozen facets per call; joint uses all twenty candidates and candidate-facet questions in one shared state. Same checkpoint/revision and score orientation are used.

Jina: official local `rerank(query, documents)` interface; facet-aware branch runs one native query per frozen facet over the common Top-20 text views.

Raw magnitudes are not cross-model calibrated. The common optimizer uses within-facet rank normalization, identical weights/penalty/tie rules, and exact five-ID output. Any invalid facet, overflow, parse failure or model failure is explicit; no fallback policy is substituted.
