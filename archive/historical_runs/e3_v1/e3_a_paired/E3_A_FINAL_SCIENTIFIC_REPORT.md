# DecisionRAG E3-A Final Scientific Report

Status: `E3_A_PASS_WITH_LIMITATIONS`

E3-A independently reconstructs the Jina/PPLX complete-support four-cell outcomes from the frozen E2-N2 selections. Track A has 3,767 questions (HotpotQA 800, MuSiQue 2,167, 2Wiki 800); Track B has 500 enterprise questions, with 443 identity-eligible and 330 retrieval-conditional under the accepted qst_0413 ambiguity treatment. Exact counts are in `E3_A_FOUR_CELL_COUNTS.csv` and paired CIs in `E3_A_PAIRED_EVIDENCE_STATS.csv` (B=5000, seed=731).

The fixed-reader branch is limited to the existing 300-query/3,300-output cohort. Reference-linked answer EM/F1 are deterministic normalized answer-string proxies for public multi-hop references, not independently adjudicated factual correctness. Enterprise reference correctness is unavailable. The R2 864-row artifact is retained in a separate no-reference local-judge table and is not promoted to gold.

The escalation curves are clairvoyant offline upper bounds. They do not establish a deployable routing policy or a cascade p95. No new inference, GPU activity, reader generation, model download, fresh holdout access, Jev call, or paid API request occurred.
