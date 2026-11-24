# DecisionRAG E6-0 Owner Decision

Decision status: `E6_0_ACCEPTED_SUCCESSFUL_READINESS_AUDIT`

## Owner decisions

1. E6-0 is accepted as a successful CPU/offline readiness audit.
2. A graded-support/grounding endpoint amendment is **conditionally approved**.
3. A separate bounded E6-1 CAL adapter pilot is **authorized**.
4. The full eight-model E6 benchmark is **not authorized**.

## E6-1 authorization boundary

The E6-1 pilot is limited to CAL/native-adapter validation and contract checks. It must:

- use only the frozen CAL population;
- validate native input/output contracts, candidate alignment, Top-5 cardinality, score direction, tie handling, hashes and failure accounting;
- use the conditionally approved graded-support/grounding endpoint only where its dataset-specific contract is satisfied;
- preserve all E6-0 and E2-E5 artifacts;
- record physical inference, model revision and resource provenance if inference is later run;
- stop before any full eight-model exploratory benchmark.

No outcome-based model selection, endpoint revision, full benchmark expansion, confirmation evaluation, Jev call, paid API call or sealed-holdout access is authorized by this decision.

The E6-0 gate report remains unchanged and continues to document the original amendment requirement. This file records the subsequent owner decision rather than rewriting frozen audit evidence.
