# E4-1 Final Scientific Report

Status: `E4_1_OWNER_REVIEW_READY` for exploratory intervention reporting, subject to the explicit proxy/neutrality limitations below.

Reader accounting: 1507 raw study rows (1493 accepted), 1397 physical reader calls (60 CAL, 1337 study), 156 exact historical A/B reuses (156 study), and 14 retained ARM_INFEASIBLE attempt rows. The interrupted pre-CAL attempt is preserved and excluded from accepted analysis.
Representative qids: 200; diagnostic-enriched qids: 50. These populations are analyzed separately.

Representative arm summaries: A_JINA_TOP5: n=200, match=98, unresolved=0, complete_support=91; B_PPLX_TOP5: n=200, match=101, unresolved=0, complete_support=90; C_ANNOTATED_ORACLE: n=200, match=103, unresolved=0, complete_support=200; D_SUPPORT_REMOVAL: n=198, match=81, unresolved=0, complete_support=0; D_CONTROL_NEUTRAL_SWAP: n=198, match=99, unresolved=0, complete_support=198; E_NO_EVIDENCE: n=200, match=21, unresolved=0, complete_support=0
Diagnostic-enriched arm summaries: A_JINA_TOP5: n=50, match=29, unresolved=0, complete_support=25; B_PPLX_TOP5: n=50, match=30, unresolved=0, complete_support=25; C_ANNOTATED_ORACLE: n=49, match=28, unresolved=0, complete_support=49; D_SUPPORT_REMOVAL: n=49, match=21, unresolved=0, complete_support=0; D_CONTROL_NEUTRAL_SWAP: n=49, match=28, unresolved=0, complete_support=49; E_NO_EVIDENCE: n=50, match=9, unresolved=0, complete_support=0

Representative primary B−A proxy delta: 0.015000 with CI [0.000000, 0.085714], n=200.
Representative primary D−D-control proxy delta: -0.090909 with CI [-0.310345, -0.071429], n=198.
Placebo-eligible paired denominators: representative D/DC=198; diagnostic-enriched D/DC=49; C oracle feasible rows are 249/250 overall. The seven infeasible study arm combinations are retained as missingness and excluded pairwise, not silently imputed.

Primary outcomes are deterministic reference agreement proxies. `UNRESOLVED` and missing/failed arms remain explicit. No answer is described as independently factually verified, and no local model adjudication was promoted to human truth.

The placebo arm is based on real frozen passages and a deterministic lexical screen. The screen cannot independently establish that a passage is non-contradictory; D-versus-placebo is therefore a qualified context-perturbation result, not a definitive necessity claim.

No citation metric is reported because the frozen reader prompt did not request candidate-ID citations. Grounding is `UNVERIFIED` except for structural support-completeness metadata.

Confirmation remains `CONFIRMATORY_DATA_NOT_READY`; E4-2, E4-3 and E4-5 were not run.

## Paired statistics

See `E4_1_PAIRED_INTERVENTION_STATS.csv`; all intervals use 5,000 source-group bootstrap resamples, seed 731, with the dominant-group limitation disclosed in E4-0.
