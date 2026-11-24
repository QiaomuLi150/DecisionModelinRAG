# DecisionRAG v2.1 — E2-P0v2 Final Owner Decision Package

**Package status:** `E2_P0V2_PARTIAL_BLOCKED`

**Scientific freeze marker:** `P0v2_NOT_FROZEN`

**M1 readiness:** `M1_PARTIAL_BLOCKED`

**Confirmatory N readiness:** `N_BLOCKED`

**Execution date:** 2026-10-12T21:21:40.393797+00:00

## Executive decision summary

This package resolves the remaining P0v2 questions into explicit owner-facing recommendations. Recommendations are not approvals. The current task authorizes offline documentation, metadata review, and CPU validation only.

| Decision | Recommended default | Status |
|---|---|---|
| M1 primary mechanism contrast | Clef-Flash pointwise binary relevance vs Clef-Flash native joint binary relevance, same checkpoint and candidate slate | `PROPOSED`; native joint adapter still requires verification |
| Selection-policy ablation | One globally comparable score vector: unconstrained Top-5 vs one-winner-per-authenticated-quartet Top-5 | `PROPOSED_FOR_M1`; not a historical quartet-probability merge |
| Conditional evidence gain | Exploratory secondary only, after native interface and budget validation | `PROPOSED`; not executable |
| Confirmatory H1 comparator | Qwen3-Reranker-4B, subject to exact checkpoint/adapter verification | `PROPOSED`; currently blocked on evidence |
| Noninferiority margin | `epsilon = 0.05` absolute complete-support probability | `PROPOSED`; owner approval required |
| Multiplicity | Holm across the owner-defined primary family; paired query bootstrap CIs | `PROPOSED`; owner approval required |
| Fresh holdout | Prefer local 2Wiki test release as a candidate only; it is not yet independently verified or sealed | `PROPOSED_CANDIDATE`; blocked |
| Serving target | Initial candidate: 1-second p95 Top-5 selection SLO | `PROPOSED`; no load test or approval |
| Core roster | Preserve the full declared roster; execute M1 first with already evidenced Clef-Flash | `PARTIAL_EVIDENCE`; external checks remain |
| GPU window | One isolated A100 80GB reservation for future M1 | `BLOCKED`; no reservation made |

## Preserved evidence

The E2-M0 findings remain unchanged: Track A conditional `N=1,490`, quota oracle `1,196`, Clef-G5 observed `689`; Track B conditional `N=330`, quota oracle `318`, Clef-G5 observed `206`. Historical quartet-normalized probabilities are not globally comparable and are not reused as a global score vector.

The existing P0v2 package contains 14 artifacts and its recorded output hashes still match. No historical result, dataset, gold record, source code, or model configuration was rewritten.

## Interpretation

M1 is an exploratory/follow-up mechanism investigation on previously inspected frozen populations. It may proceed independently of the new confirmatory holdout and noninferiority margin, but only after its native adapter, score semantics, comparison population, prompt/view hashes, and GPU execution window are separately approved.

N is a confirmatory replacement investigation. It remains blocked until a fresh sealed holdout, primary contrast, epsilon, power, multiplicity, checkpoint revisions, and production-serving criteria are all explicitly approved.

## Owner action required

The owner must approve or reject each proposed value in the checklist. Until then, no proposal is frozen and no model request is authorized.
