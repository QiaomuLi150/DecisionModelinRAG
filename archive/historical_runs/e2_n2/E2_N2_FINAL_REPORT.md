# DecisionRAG E2-N2 Final Report

Status: `E2_N2_PASS_WITH_LIMITATIONS`

## Execution mode

E2-N2 was completed using authenticated E2-N physical smoke/CAL/full traces already present from E2-N, followed by a fresh offline N4 evaluation in the E2-N2 namespace. This avoided duplicate inference while preserving genuine model outputs. Current packaging used no GPU allocation.

Policy views completed: **13**; normalized trace rows: **55,471**; explicit blocked/error rows: **500**; offline quality rows: **84**.

## Model-family status

- Decision models: Nimble, Intern-Decision, and PPLX traces are present; Intern-Decision remains context-blocked on 500 enterprise rows.
- Specialist rerankers: Qwen 0.6B/4B, Jina v3.5, GTE-ModernBERT, and MiniLM traces are present.
- Existing controls: BGE, Qwen3-VL-Reranker-2B, Clef, and Laya are preserved through authenticated traces; resident Qwen-service telemetry is not promoted over direct-checkpoint telemetry.

## Scientific limitations

- Track A and Track B remain separate and exploratory.
- CSSR is a complete-support selection proxy, not independently verified reader factual accuracy.
- Cross-model comparisons with unequal context feasibility must report full-workload and feasible-subset denominators separately.
- GPU dollar cost and maximum production QPS were not measured.

Jev calls: `0`; paid API calls: `0`; fresh holdout access: `false`; training: `false`; retrieval/indexing: `false`.
