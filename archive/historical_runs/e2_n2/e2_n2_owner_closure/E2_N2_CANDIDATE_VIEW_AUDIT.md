# E2-N2-R Candidate-View and Score-Joining Audit

Status: `PASS_WITH_LIMITATIONS`

All checks below are offline checks over frozen normalized rows. No candidate view was regenerated with a model tokenizer and no inference was performed.

## Alignment and evidence scorecard

| Policy | Rows | OK | Canonical candidate alignment | Question hashes | 20-view hashes | Input hashes | Response hashes | Finite 20 scores | Forward evidence | Contract |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| retrieval5 | 4267 | 4267 | 4267 | 0 | 0 | 0 | 0 | 0 | 0 | frozen retrieval control; no model inference |
| bge_pointwise | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | independent pointwise relevance; semantic text view |
| laya_pointwise | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | independent pointwise relevance; adapter batches candidate payloads |
| clef_pointwise | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | pointwise binary true probability |
| clef_joint | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | native joint binary true probability |
| gte | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | pointwise text relevance |
| minilm | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | pointwise text relevance |
| jina35 | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | native listwise/chunked relevance contract |
| nimble | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | native slate categorical 20-choice contract |
| intern_decision | 4267 | 3767 | 4267 | 0 | 3767 | 3767 | 3767 | 3767 | 3767 | native joint binary contract; enterprise context blocked at >8192 |
| pplx_decider | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | native slate categorical 20-choice contract |
| qwen06 | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | pointwise text relevance |
| qwen4 | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | official relevance-token contract |
| qwen_direct | 4267 | 4267 | 4267 | 0 | 4267 | 4267 | 4267 | 4267 | 4267 | direct checkpoint independent relevance contract |

Candidate/score audit errors: `0`.

## Relevance direction, sorting, and joining

- Every successful score row was checked against the exact 20 canonical candidate IDs and order from the frozen Track A/B slate.
- Scores are interpreted in the recorded descending-relevance direction. Top-5 is recomputed from score descending, original candidate position, then candidate ID.
- The frozen selection JSONL was compared against the independently recomputed Top-5 IDs for every policy/query key, including explicit empty selections for Intern-Decision context-blocked rows.
- No duplicate policy/query score rows were accepted.
- `gold_in_inference_input` is false for all normalized rows. Gold labels are joined only in this offline evaluator.
- Canonical query identity is verified by frozen query IDs and the exact frozen slate joins. The normalized model rows do not carry a separate `question_sha256` field: question-hash coverage is therefore `0/54,971` and is disclosed as a telemetry gap, while `actual_model_input_sha256` is present for all successful rows.

## Model-specific view contracts

BGE and the text specialists use pointwise/text relevance contracts; Jina uses its native listwise/chunked contract; Nimble and PPLX use native 20-choice slate contracts; Clef has separately recorded pointwise and native-joint binary contracts; Qwen4 uses its official relevance-token contract; Intern-Decision uses its native joint contract and explicitly rejects the 500 enterprise views above its frozen context limit.
These are a common candidate slate but not identical tokenization, native prompt format, or context-budget experiments. Cross-model conclusions must retain that limitation.

## Candidate-view serialization limitation

The normalized rows preserve per-candidate `candidate_view_hashes` and the complete model-input hash, but do not embed the raw serialized candidate-view text. Therefore this audit proves identity/order and hash presence, not byte-level regeneration of every view. A future serving contract must retain raw request bytes or a separately verifiable serialization manifest.
