# E6-2 R0 Candidate-Order Robustness Protocol

For every admitted model and future frozen candidate slate, preserve canonical retrieval order as primary. On a prespecified diagnostic subset, evaluate reversed order and at least three deterministic random permutations generated from seeds 731, 732 and 733 using query ID plus candidate IDs. Do not select the subset or permutation after outcomes.

Record exact set match, Top-5 Jaccard, changed selected-ID count, quality change, output failures and context-overflow changes. Keep this robustness analysis separate from the canonical primary endpoint. Order sensitivity does not by itself invalidate a model for canonical ordered inputs, but it limits order-invariant claims.
