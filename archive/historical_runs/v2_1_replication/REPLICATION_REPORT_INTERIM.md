# DecisionRAG v2.1 controlled replication report

## Overall status

`BLOCKED`: both benchmark branches stopped before model scoring under A3-R2. This is a protocol-input blocker, not a model result.

## Stage 0 findings

- EnterpriseRAG-Bench: 500 questions; 443 `APPLICABLE` identity-resolved quality rows; 57 non-quality rows (30 empty gold, 26 over-five gold, 1 accepted qst_0413 ambiguity exception). A prior 500-row hybrid RRF Top-20 trace exists, but the exact BM25-only Top-20 artifact required by this protocol does not.
- MuSiQue: 2,167 selected records from the 2,417-row local dev release; support cardinalities were audited offline. A prior 2,167-row hybrid RRF Top-20 trace exists, but the exact BM25-only Top-20 artifact and preserved BM25 ordering required by this protocol do not.
- Label-free length scan: `COMPLETE_DIAGNOSTIC_ONLY` for the non-admissible MuSiQue reference trace; it is not a replication result.
- Protocol was written and hashed before any model scoring call; protocol SHA-256 is `b9c417a5932474941d0e1cba42687e8cae578ec2b1e8b49bb9393425b07813ab`.

## Stages 1–5

No selector inference, order robustness, serving timing, FS@5 statistics, or export package was run. There are zero model calls, zero GPU workloads, zero Jev calls and zero paid API calls.

## Conclusions

Transferred findings: none from this replication, because no admissible model outcomes exist. Divergent findings: none measured. Unverified: all six replication questions in the specification.

## Required unblock

Supply the exact hashable BM25-only Top-20 candidate slates and exact candidate-text manifests for both datasets, or approve a versioned protocol amendment that explicitly permits the existing hybrid RRF slates. The historical slates must remain preserved.
