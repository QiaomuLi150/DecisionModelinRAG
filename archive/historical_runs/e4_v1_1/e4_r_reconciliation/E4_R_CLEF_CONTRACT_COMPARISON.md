# E4-R Clef Contract Comparison

| Field | E2-M1 | E4-3 | Comparability |
|---|---|---|---|
| checkpoint hash/revision | Cloudflare/clef-flash@8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 | Cloudflare/clef-flash@8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 | MATCH |
| adapter/objective | pointwise INDEPENDENT_RELEVANCE:v2; native JOINT_BINARY_RELEVANCE:v2 | same named contracts; E4 K10 prefix and K20 full view | SAME_MODEL_CONTRACT_DIFFERENT_VIEWS |
| prompt/candidate serialization | E2-M1 make_views; pointwise batch 20; joint one native 20-field record | E3-D candidate-view hashes; deterministic K10 first-ten/K20 full twenty | HETEROGENEOUS_CANDIDATE_VIEW_LINEAGE |
| slate origin | Track A 3767 / Track B 500 frozen slates | E3-D MultiHop exploratory slate cohort, 300 representative + 80 diagnostic | DIFFERENT_POPULATIONS |
| support identity/feasibility | Track-specific qrels; conditional n=1490/n=330 | gold_support_ids_for_analysis_only; K10 feasible 206/300 and 47/80; K20 300/300 and 80/80 | DIFFERENT_DENOMINATORS |
| Top-5 selection | global score sort plus separate quartet quota policies | stable score sort over K-specific candidate view | DIFFERENT_SELECTION_PROTOCOLS |
| bootstrap | query-level seed 731, 5000 reps | query-level seed 731, 5000 reps; source sensitivity separately | SAME_PRIMARY_RESAMPLING_DIFFERENT_COHORT |
| exact overlap | 4267 | 380 | NOT_IDENTIFIABLE; matched_n=0 |

The common checkpoint and named objectives do not make the studies independently paired. Exact query/candidate/view overlap is required for a cross-study mechanism effect; that overlap is absent.
