# E4-1 Execution Events

## Interrupted pre-CAL attempt

The first E4-1 executor revision selected ten CAL qids in memory but did not materialize their arm rows. Its CAL queue was therefore empty and the study executor began submitting requests. The run was interrupted before any study output was accepted. The raw output file at that point contained only exact historical A/B reuse rows and explicit infeasible-arm records; no new study response was promoted into the accepted results.

The implementation was corrected to materialize and hash all CAL arm rows, to write the CAL trace boundary before study submission, and to fail closed on any CAL request error. The interrupted attempt is preserved as an engineering limitation; its in-flight request count is not represented as completed scientific inference, and no corresponding answer is used in analysis.

No historical E2/E3 artifact was modified. Jev calls, paid calls and reserved-holdout access remained zero.
