# E2-P0v2 Model Verification Gaps

## Locally evidenced models

| Model | Evidence status | Known limitation before M1/N |
|---|---|---|
| `Cloudflare/clef-flash` revision `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | Historical local pointwise/groupwise traces | Native joint binary schema, context capacity, score denominator, and physical forward semantics require adapter audit |
| `convaiinnovations/laya` revision `7b928d828b7b0e022f929d9bd2e44165aa270148` | Historical local pointwise/groupwise traces | Short-context and categorical group semantics remain limitations; no global score pooling |
| `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Historical pointwise trace | Local license/checkpoint verification and new physical telemetry remain outstanding |
| `Qwen/Qwen3-VL-Reranker-2B` revision `4bd860ac4f15ad1897a214615cccc700f8f71818` | Historical pointwise/service trace | Exact official token/readout semantics and physical telemetry remain outstanding |

## Declared roster retained

Decision models: Laya, Clef-Flash, Bespoke Nimble-v2, Intern-Decision-4B, Perplexity Decider 27B.

Specialists: BGE-v2-m3, Qwen3-VL-Reranker-2B, Qwen3-Reranker-0.6B, Qwen3-Reranker-4B, Jina-v3.5, GTE-ModernBERT, MiniLM-L6-v2.

Optional: Clef 27B, CLM 8B, Qwen3-Reranker-8B. Jev remains a separate paid reference and last phase.

For every unverified model, checkpoint/revision, license, official scoring interface, native context budget, batching semantics, output alignment, and 80GB feasibility remain outstanding. No difficult checkpoint is silently removed; it is marked `BLOCKED_LOCAL_EVIDENCE` until verified.
