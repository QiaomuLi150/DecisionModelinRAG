# E4-R Provenance Graph

This graph is generated offline from byte hashes; source artifacts remain read-only.

## Nodes

- Governing documents/code: 5 files.
- Historical E2/E3/E4 artifacts: 549 files.
- Manifest/gate checks: 32 files; hash mismatches: 0.

## Edges

- E2-M1 raw Clef scores and selected IDs -> E2-M1 case metrics -> E2-M1 paired statistics.
- E3-D metadata/candidates/views/traces -> E4-3 frozen cohort/raw traces -> E4-3 scorecard.
- E4-2 manifest and artifacts -> historical context only; no E4-R inference.
- E2/E3/E4 hashes and manifests -> E4-R independent audit outputs.

## Integrity caveats

- E2-M1 config status `RUNNING` conflicts with its execution manifest `E2_M1_PASS`; this is retained as a historical status inconsistency.
- E4-3 K20 specialist rows are exact historical reuse and do not represent new E4-R/E4-3 computation.
- E4-5 remains `CONFIRMATORY_DATA_NOT_READY`; no reserved outcomes were opened.
