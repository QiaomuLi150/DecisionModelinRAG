# E3-F0 Confirmatory Protocol Options — DRAFT, NOT APPROVED

No option is selected or frozen by F0. All options require a genuinely new eligible population and owner/reviewer approval before any labels or outcomes are opened.

## Option A — Independent multi-hop evidence-selection confirmation

**Population:** newly acquired source-document-group-disjoint multi-hop questions with complete, prevalidated support annotations. Freeze one Top-20 candidate pool and Top-5 rule before model outcomes. **Roster:** Jina and PPLX as the primary pair; retrieval-only is a descriptive baseline.

**Primary estimand:** paired unconditional difference `CSSR_U(PPLX) - CSSR_U(Jina)` over all eligible questions, including model/context failures as failures. A reviewer may choose superiority or non-inferiority; a provisional illustrative non-inferiority margin is −5 percentage points, not approved. Conditional CSSR-C, Top-20 feasibility and source-cluster sensitivity are secondary.

**Statistics:** paired binary discordance analysis with exact or mid-p McNemar interval/test as appropriate, plus source-document-cluster bootstrap or robust cluster inference. One primary endpoint; multiplicity-controlled secondary endpoints. No 900/250 historical denominator substitution.

**Planning:** under baseline 46%, 20% discordance and one-sided superiority effects, the deterministic sensitivity file gives minimum n for +2/+5/+10 pp at 80%/90% targets as `{"2": {"0.8": 4100, "0.9": null}, "5": {"0.8": 800, "0.9": 1000}, "10": {"0.8": 300, "0.9": 300}}`. These are not claims about the future population; 2 pp may be infeasible within the simulated 5,000-question ceiling under the stated assumptions.

**Feasibility:** `NOT_FEASIBLE_NOW` until a new source release, complete support annotations and prospective reservation timestamp exist. Incomplete annotations require changing the endpoint before data collection.

## Option B — Authentic enterprise transfer / interaction confirmation

**Population:** new organizations, filings or reporting periods with independent source groups and verified document-level gold. Synthetic EnterpriseRAG-Bench and exposed FinanceBench are excluded from confirmation.

**Primary estimand:** within-enterprise paired PPLX-minus-Jina difference in complete evidence retention, or a predeclared model×domain interaction if the scientific question is transfer rather than replacement. A provisional +5 pp superiority target is illustrative only. Report organization-clustered uncertainty and per-domain results; do not pool incompatible page/passage metrics.

**Statistics and missingness:** intention-to-evaluate; overflow, unsupported, and failed native calls remain unconditional failures. Cluster bootstrap at organization/report level; one primary interaction/endpoint with controlled secondary analyses.

**Feasibility:** `DATA_CONSTRAINED`; requires rights, independent annotation, source-group isolation and adequate organization count. The current local corpus cannot satisfy this.

## Option C — Reference-supported downstream utility confirmation

**Population:** new independently referenced questions with frozen retrieved evidence and a fixed reader. Every answer is scored against prevalidated references, and ambiguous financial/numerical cases receive independent expert adjudication before unblinding model results.

**Primary estimand:** paired verified factual-correctness difference PPLX-minus-Jina on the complete eligible population. Normalized EM/F1 and numeric equivalence are secondary diagnostics, never substitutes for verified correctness.

**Statistics:** paired binary/graded analysis with exact paired intervals, source/issuer clustering, and blinded adjudication of disagreements. The disagreement subset may be a descriptive stratum but cannot become the population estimand. Missingness and abstention rules are frozen in advance.

**Feasibility:** `NOT_FEASIBLE_NOW`; no independently verified factual labels exist in E3-D-R, and local-judge agreement cannot supply them.

## Shared prerequisites and disqualification

All options require model revisions, prompts, text views, truncation, Top-20 construction, score direction, tie handling, cache policy, failure accounting, seed, source rights, and analysis code to be frozen before outcomes. A model cannot be silently omitted on context overflow. No option supports a causal architecture claim unless the architecture and all other conditions are controlled. A 5-pp margin, 1-second or 3-second p95 SLO, and any dollar cost are proposed planning choices, not current product facts.
