# DecisionRAG v1.5 GPU3 Continuation — Final Report

Gate: `V1_5_OWNER_REVIEW_READY_PARTIAL_BOUNDED_C3`

## Stage status

- C1 two-stage specialist: COMPLETE; 21 frozen shards × 1,384 = 29,064 new label-blind selection records.
- C2 latency uncertainty: COMPLETE from existing historical traces; no new model calls.
- C3 concurrency: `PASS_WITH_D_BLOCKED`; D_separate remains blocked because no second isolated A100 was available.
- C4 Winnow rerun: `COMPLETE`.

## C1 primary contrasts

| Contrast | n | delta | 95% CI | Holm p |
|---|---:|---:|---|---:|
| pplx_two_stage - jina_two_stage_full (original) | 1384 | 0.02485927951929999 | [0.010896294457117539, 0.04010445980261434] | 0.0146484375 |
| pplx_two_stage - jina_two_stage_full (mean six shuffled) | 1384 | 0.019521635338407645 | [0.0047152765812198995, 0.035860258890624615] | 0.0703125 |
| pplx_two_stage - jina_two_stage_pplxtext (original) | 1384 | 0.033217875265909864 | [0.018442187681208623, 0.04909507258648124] | 0.00390625 |
| jina_two_stage_full - jina_full (original) | 1384 | 0.0018695744133712694 | [-0.0020171386584172516, 0.005467839564477255] | 0.3583984375 |

C1 batch traces are valid for ranking quality but are not isolated-request latency measurements. R1 retiming is the dedicated-GPU timing source.

## Scientific boundaries

- All native outputs remain label-blind until separate offline joins.
- Failures remain in operational denominators and are not silently retried.
- C3 uses the exact fixed R1 routing request repeated with stable IDs because no 50-variant frozen pool artifact exists; this is disclosed rather than treated as a rich prompt-diversity test.
- No sustainable production QPS, monetary GPU cost, factual correctness, or replacement superiority is inferred from this continuation alone.
- No Jev or paid API calls occurred; historical artifacts were not modified.
