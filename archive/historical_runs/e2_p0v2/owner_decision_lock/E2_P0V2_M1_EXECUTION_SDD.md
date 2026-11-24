# E2-P0v2 M1 Execution SDD

**Status:** implementation-ready proposal; `M1_PARTIAL_BLOCKED`; no execution authorized.

## 1. Scientific purpose and boundaries

M1 is an exploratory mechanism investigation on previously inspected frozen Track A and Track B evidence. It is not a confirmatory replacement claim and does not consume or relabel a fresh holdout. Track A and Track B remain separate populations.

The historical structural findings are preserved: Track A conditional N=1,490, quota oracle=1,196, Clef-G5 observed=689; Track B conditional N=330, quota oracle=318, Clef-G5 observed=206. Historical quartet-normalized probabilities are not a global score vector.

## 2. Frozen inputs for a future authorized run

| Input | Contract |
|---|---|
| Questions | Canonical nonempty question strings; reject blank/whitespace-only input |
| Candidates | Existing frozen Top-20 IDs and original order; no candidate filtering or replacement |
| Track A | 3,767 previously inspected prospective questions; report eligible and retrieval-conditional denominators separately |
| Track B | 443 identity-resolved previously inspected enterprise questions; retain all 500 in manifest, exclude unresolved identity cases only from identity-dependent metrics |
| Conditional populations | Track A 1,490 and Track B 330 when all annotated gold identities are in Top-20 |
| Candidate view | Same semantic question/document text for compared adapters; historical pointwise view is q=64/passage=128 tokens, historical groupwise passage=64, but joint native capacity must be checked before execution |
| Seeds | Preserve frozen study seed 20261011; bootstrap seed 731 and 5,000 replicates unless a versioned M1 plan changes them before execution |
| Gold access | Gold is evaluation-only and never placed in prompts; no fresh holdout is used in M1 |

## 3. M1-A — Pointwise binary relevance

### Treatment

Run the same verified Clef-Flash checkpoint and binary relevance objective independently on each query-document pair. The output is one score/logit per canonical candidate ID. The model must not see sibling candidate documents.

### Input schema

```json
{"query_id":"...","question":"canonical nonempty text","candidate_id":"...","document_text":"frozen view","view_hash":"...","score_contract":"INDEPENDENT_RELEVANCE:v1"}
```

### Output schema

```json
{"query_id":"...","candidate_id":"...","score":0.0,"score_direction":"higher_is_more_relevant","model_id":"Cloudflare/clef-flash","revision":"...","actual_input_sha256":"...","raw_response_sha256":"...","status":"ok"}
```

The exact native readout, score denominator, tokenizer limit, and physical forward semantics must be verified before execution. If any is unknown, record `UNVERIFIED` and stop rather than approximate.

## 4. M1-B — Native joint binary relevance

### Treatment

Run the same Clef-Flash checkpoint and objective with the candidate documents exposed in one documented native joint binary request. Candidate-specific outputs must align to the canonical candidate IDs. The intentional difference from M1-A is shared candidate context; this is not called a pure architecture ablation.

### Input schema

```json
{"query_id":"...","question":"canonical nonempty text","candidate_ids":["c1","...","c20"],"documents":[{"candidate_id":"c1","text":"frozen view"}],"candidate_order":"frozen","view_hash":"...","score_contract":"JOINT_BINARY_RELEVANCE:v1"}
```

### Required gate

Do not assume that all 20 candidates fit Clef's runtime context. Compute tokenizer length and native context budget before inference. If the exact joint interface cannot be verified, M1-B is `BLOCKED`; do not construct a pseudo-joint prompt or reinterpret categorical quartet probabilities as joint binary relevance.

## 5. M1-C — Fixed-score selection-policy ablation

Use one authenticated, globally comparable score vector over the same 20 candidates, preferably a verified M1-B output or another explicitly approved global score contract. Apply:

1. unconstrained descending-score Top-5;
2. one-winner-per-authenticated-quartet Top-5 with deterministic fill/tie rules.

The scores, candidate IDs, partitions, question strings and view hashes are identical between policies. Report complete support retention, retrieval-conditional CSSR, unconditional support retention, per-document metrics where valid, candidate-order sensitivity, overflow and physical work. Historical Clef-G5 remains a separate categorical reference and is not recomputed or merged.

## 6. M1-D — Historical categorical reference

Use the saved Clef-G5 group records exactly as historical reference. Preserve quartet-specific denominators and authenticated partitions. Do not rerun, alter, globally sort or concatenate quartet-normalized probabilities. Clearly label this reference as a different score contract from M1-A/B/C.

## 7. M1-E — Conditional evidence gain

Exploratory secondary only. After a selected set S is recorded, request a model-native candidate-usefulness readout for the next candidate or next batch. Do not place gold labels, answers, qrels or support IDs in prompts. Do not call the output true marginal utility. Count every iterative pass, candidate list, token view, queue wait and physical forward. Abort if the native interface or budget cannot be validated.

## 8. Analysis

- Pair at the query level; preserve dataset boundaries.
- Report Track A and Track B separately; equal-dataset macro is secondary.
- CSSR uses the complete annotated support contract and explicit eligible/conditional denominators.
- Use paired bootstrap CIs with source-document clustering where appropriate.
- Treat M1-C as a selection-policy contrast, not model superiority.
- Report missing, overflow, invalid, and unsupported outputs explicitly; never impute.
- Candidate-order sensitivity uses deterministic shuffles with identity-preserving alignment.

## 9. Required telemetry

Capture canonical question hash, frozen candidate IDs/order, semantic view hash, actual model-visible input hash, raw response bytes/hash before normalization, model revision/config hash, score contract, tokenizer lengths, truncation/overflow, queue wait, model compute, serialization, physical forward/pass count if instrumentable, GPU UUID/memory/utilization, cache key/provenance, retries and errors.

Selection latency is complete Top-5 selection work, including all 20 scores and repeated/native passes. Do not substitute sums of stage percentiles for measured end-to-end latency.

## 10. Integrity tests and failure handling

Before execution validate: nonempty question, 20 unique in-slate candidates, exact image/text/document mapping, score cardinality and alignment, no hidden filtering, deterministic ties, stable order, context budget, no gold fields in prompt, prompt/config hashes, and authorized checkpoint revision.

Any missing score, duplicate/misaligned ID, unknown score denominator, unsupported native interface, overflow without prespecified handling, unexplained cache reuse, absent raw hash, or unauthorized GPU produces an explicit invalid/blocked record. Do not silently retry into a different policy or substitute a different model.

## 11. Calibration and execution cap proposal

Before the full exploratory run, use at most 100 existing CAL questions × 20 candidates per adapter for interface/schema and context-budget validation. This is a proposed cap, not a runtime authorization. Full M1 execution requires a new owner approval after the smoke report.

## 12. Stop conditions

Stop if Clef joint binary semantics cannot be verified, if the native context cannot fit the declared candidate view, if a checkpoint/revision is unavailable, if GPU isolation is not demonstrated, if physical work cannot be measured as required, or if any candidate identity cannot be aligned. Never convert a failed native check into an unsupported approximation.
