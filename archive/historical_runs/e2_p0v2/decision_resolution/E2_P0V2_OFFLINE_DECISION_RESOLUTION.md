# E2-P0v2 Offline Decision-Resolution Report

**Owner action recorded:** `ACCEPT P0v2 FOR OFFLINE DECISION-RESOLUTION ONLY`

**Current verdict:** `E2_P0V2_PARTIAL_BLOCKED`

**Freeze marker:** `P0v2_NOT_FROZEN`

**Execution window:** CPU-only static and metadata audit, 2026-10-12T21:14:20.015121+00:00 through 2026-10-12T21:14:20.998714+00:00 UTC.

## Scope acceptance

The owner acceptance authorizes this offline resolution phase only. It does **not** authorize checkpoint loading, model inference, GPU serving, network/API requests, Qdrant access, retrieval/index construction, training, M1/N/S/P, Jev, paid inference, or opening a proposed fresh holdout gold sidecar. The original P0v2 artifacts remain preserved; this report is an append-only derived record.

## What was resolved

- The P0v2 planning package is accepted as the working specification for offline resolution.
- The explicit score contracts, candidate-view rules, historical quartet non-pooling rule, H2a/H2b/H2c separation, and telemetry requirements are retained for planning.
- Existing P0v2 artifact presence and recorded output hashes were rechecked.
- Existing ID-only metadata was audited for overlap among the local CAL/prior-test manifest, debug fixtures, and the historical `NEW_TEST_IDS` manifest. The audit did not inspect answers or fresh holdout gold.
- Local release files were inventoried by path, size, and SHA-256 only. Availability does not establish untouched-holdout independence.

## Artifact and execution checks

| Check | Result |
|---|---|
| Required P0v2 artifacts | `14 expected; PASS` |
| Existing P0v2 recorded hashes | `PASS` |
| M0 no-call/preservation gate | `PASS` |
| ID-only overlap audit | `PASS` |
| Proposed fresh holdout gold opened | `NO (PASS)` |
| Model/GPU/service/network calls | `0 (PASS)` |
| Jev/paid API calls | `0 (PASS)` |

## ID-only overlap audit

The source-record key is derived from an explicit `source_record_id`, or from the query ID after its dataset/debug namespace. This is an identity audit only; it is not a label audit.

| Dataset | CAL/prior-test | Debug | Historical new-test manifest | CAL∩debug | CAL∩new-test | Debug∩new-test | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| hotpot | 250 | 10 | 800 | 0 | 0 | 0 | PASS |
| musique | 250 | 10 | 2167 | 0 | 0 | 0 | PASS |
| wiki2 | 250 | 10 | 800 | 0 | 0 | 0 | PASS |

`NEW_TEST_IDS.jsonl` is a historical artifact and is not promoted to the new P0v2 confirmatory holdout. Any nonzero overlap would block a future immutable lock rather than trigger outcome-based resampling. Exact machine-readable counts are in `E2_P0V2_OFFLINE_DECISION_LOCK.json`.

## Local source inventory

The local MuSiQue, HotpotQA, 2Wiki, and EnterpriseRAG-Bench files were recorded by path/bytes/hash only. The 2Wiki `test.json` file was deliberately not parsed because it may contain answer/gold fields that must remain closed until a holdout is formally approved. No local file is therefore accepted as the fresh confirmatory holdout by this phase.

## Decision register after offline resolution

| Decision | Status | Consequence |
|---|---|---|
| owner_scope_acceptance | `ACCEPTED_OFFLINE_ONLY` | Owner accepted P0v2 for offline decision-resolution only; no execution or freeze authorization. |
| scoring_and_view_contract | `RESOLVED_FOR_PLANNING` | Use the existing explicit score-contract and candidate-view rules; native feasibility remains model-specific and unverified until authorized audit. |
| primary_H1_pair | `OWNER_VALUE_REQUIRED` | The existing Clef-Flash-versus-one-specialist recommendation remains a recommendation, not a frozen contrast. |
| H2b_policy_contrast | `RESOLVED_FOR_PLANNING` | If approved, use one globally comparable score vector with global Top-5 versus fixed authenticated-quota Top-5; no historical quartet probabilities are pooled. |
| fresh_confirmatory_holdout | `BLOCKED` | No locally verified, untouched release/revision/label lock is available. Local files are inventoried metadata-only; 2Wiki test gold was not opened. |
| noninferiority_epsilon | `OWNER_VALUE_REQUIRED` | No numeric epsilon was supplied; no margin is selected or frozen. |
| power_and_sample_size | `BLOCKED_ON_EPSILON_AND_HOLDOUT` | Cannot freeze power/sample size without approved epsilon, alpha, target power, discordance assumptions and holdout identity/n. |
| multiplicity | `OWNER_VALUE_REQUIRED` | Holm remains a recommendation only; primary family and alpha are not frozen. |
| numeric_p95_SLO_and_load | `OWNER_VALUE_REQUIRED` | No numeric serving SLO or load regime is frozen. The prior 2-second proposal remains non-binding. |
| core_model_roster | `PARTIAL_EVIDENCE` | Only four locally evidenced model revisions remain eligible for future audit planning; all additional roster entries require external verification. |
| GPU_execution_window | `BLOCKED` | This user authorization covers offline work only; no GPU window is opened. |
| Jev | `NOT_AUTHORIZED` | Jev remains last and untouched; zero calls. |

## Remaining freeze blockers

1. Approve and lock an exact untouched confirmatory holdout identity, release/revision, sample size, document/source independence, label-access protocol, and hashes.
2. Select the primary H1 pair or explicitly approve a CAL-only selection algorithm.
3. Approve an absolute noninferiority margin, alpha/power/sample-size plan, and multiplicity family.
4. Approve the core checkpoint roster/revisions and complete external capability/license/hash verification for any models beyond the four locally evidenced entries.
5. Approve a numeric p95 serving SLO, load regime, and an isolated execution window before any GPU/inference phase.

Until all mandatory items are resolved and independently validated, the required state remains **`P0v2_NOT_FROZEN`**. No scientific result, holdout, checkpoint roster, SLO, or paid phase has been frozen by this report.

## Stop gate

This phase stops here. No M1/N/S/P, model request, GPU workload, Qdrant/index operation, service restart, dataset expansion, Jev request, or paid API request was made.
