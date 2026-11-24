# v3.6 Stage-3 frontier statistics

The frontier joins Stage-1 BRIGHT quality records (1,384 queries) to T2 C120 timing records (120 IDs x 3 attempts). The quality CI is a 40,000-draw fixed-domain bootstrap (seed 731); timing p95 ratios use 10,000 canonical-ID cluster resamples (seed 731). R, Q and S are separate release checks: validity >=99%, quality lower CI > -0.02 nDCG, and p95 ratio upper CI <=2.0. No point is treated as a production guarantee.
