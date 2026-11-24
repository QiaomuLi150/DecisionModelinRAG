# R3 ordered completion

Gate: `R3_COMPLETE`

All 1,384 BRIGHT questions were evaluated under reversed order and five seeded permutations, preserving candidate identity and the original frozen slate hash. Each order had 1,381 valid and 3 failed rows; 8,304 rows were sealed before offline label joins.

The original-order all-attempt nDCG@5 was 0.210772. Ordered all-attempt nDCG@5 values were 0.202500 (reversed), 0.203606 (seed1), 0.209310 (seed2), 0.206848 (seed3), 0.202771 (seed4), and 0.202338 (seed5). The set changed for 92.847%–94.436% of queries. Reversed order produced paired query delta -0.008272 and source-group delta -0.009389, with domain-bootstrap 95% CI [-0.016259, -0.002473] and sign-flip diagnostic p=0.02832. These are order-sensitivity diagnostics, not independent model comparisons.
