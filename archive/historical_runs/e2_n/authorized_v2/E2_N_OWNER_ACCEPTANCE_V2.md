# DecisionRAG E2-N Owner Acceptance (v2)

Date: 2026-10-13

## Decision

Owner approval is recorded for the completed, authorized E2-N non-Jev work.

**Status: `OWNER_ACCEPTED_PASS_WITH_LIMITATIONS`**

N0 through N4 are closed for the authorized exploratory scope. The final status remains `PASS_WITH_LIMITATIONS`, as documented in the execution manifest and owner-review gate.

## Accepted evidence

- N3 physical evaluation completed for the authorized model roster, with the documented exception for Intern-Decision-4B.
- Qwen direct-checkpoint telemetry remediation completed without restarting or modifying the resident Qwen service.
- N4 offline statistical analysis and selection artifacts completed.
- The final offline regression suite passed: **65/65 tests**.
- Artifact hash verification passed for the N4 provenance manifest, quality summary, paired statistics, and Top-5 selection output.
- No fresh confirmatory holdout was accessed.

## Preserved limitation

Intern-Decision-4B remains blocked for the 500-row enterprise subset because the frozen native 8192-token contract rejects those inputs with context overflow. This is reported as native context infeasibility; no truncation, protocol change, model substitution, or fabricated result was applied.

The resulting CSSR values are complete-support selection proxies, not independently verified reader factual-accuracy measurements. Qwen direct-checkpoint telemetry and resident-service telemetry remain separate evidence streams. GPU dollar cost was not measured, so no monetary claim is made.

## Scope and exclusions

This acceptance does **not** authorize:

- Jev or any paid API request;
- fresh-holdout access;
- additional inference, training, or protocol changes;
- production-service restart or modification.

Jev remains separately gated and must remain at zero calls until a distinct owner authorization is issued after the applicable non-Jev work is reviewed.

## Primary records

- [Authorized E2-N report](E2_N_AUTHORIZED_EXPANSION_REPORT.md)
- [Execution manifest](E2_N_EXECUTION_MANIFEST_V2.json)
- [Owner-review gate](E2_N_OWNER_REVIEW_GATE_V2.md)
- [Qwen telemetry remediation](E2_N_QWEN_TELEMETRY_REMEDIATION.md)
- [N4 provenance manifest](../n4_authorized_v2/E2_N4_PROVENANCE_MANIFEST.json)

All preceding M0, P0v2, M1, Track A/B/C, HERB, and earlier E2-N artifacts remain preserved.
