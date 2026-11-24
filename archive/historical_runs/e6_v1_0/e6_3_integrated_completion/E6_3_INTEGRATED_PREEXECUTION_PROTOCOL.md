# DecisionRAG E6-3 Integrated Pre-execution Protocol

Status: FROZEN_BEFORE_REFERENCE_JOIN_AND_SERVING
Protocol seed: 731

## Reader scoring

The frozen R1 parsed outputs are scored without new model calls. The primary reader endpoint is normalized token F1 against the authorized GaRAGe `answer_generate` field, with normalized exact agreement reported secondarily. Normalization is Unicode NFKC, casefolding, removal of `[cite_N]` markers, punctuation-to-space conversion, and whitespace collapse. These are reference-overlap proxies, not factual correctness. The primary contrast is paired representative-cohort PPLX minus Jina; 5,000 query bootstrap resamples, seed 731, percentile two-sided 95% intervals. Diagnostic disagreement and no-positive cohorts remain separate. Missing or invalid references are excluded from that metric denominator and counted explicitly.

Citation validity is structural only. Annotated citation relevance uses the previously frozen `evidence_relevant` projection; `evidence_correct` is separately tabulated, including UNKNOWN and RELATED-INFORMATION. `CLAIM_LEVEL_FACTUAL_SUPPORT_NOT_VERIFIED` is mandatory. No LLM judge or human outcome is created.

Only the 313 IDs in the R1 cohort manifest may be joined to answer/reference/grounding data. Reserved rows are not parsed.

## Serving

Scenarios are fixed before selector inference: 100 SciFact queries from the 277 exploratory population, and 100 GaRAGe queries from the 1,397 exploratory population, selected by SHA256(seed|query_id), seed 731, independent of labels and latency. Each of eight pinned models receives the same two workload samples and two timed repetitions per query: 3,200 timed attempts plus 16 one-request-per-model/workload warmups. SciFact uses frozen natural Top-20 slates; GaRAGe uses frozen label-blind K15 inputs.

The request boundary is input-view construction, native serialization/tokenization, host-device transfer, native inference, required forwards, score/readout processing, stable Top-5 selection, and output packaging. CUDA is synchronized before and after each timed request. Load and warmup are separate. Failures remain in the denominator; failed latency is not imputed. No cached prediction reuse is permitted. Model order is deterministic counterbalanced: a SHA256-derived order for SciFact and its reverse for GaRAGe. Models run sequentially on one validated same-device GPU when possible.

Measured endpoints: attempts, valid selections, failures, p50/p90/p95/p99 for successful requests, full-attempt failure rate, peak allocated/reserved memory, native API calls, observed forwards (or NOT_MEASURED), input tokens, 1s/3s p95 SLO compliance. Sustainable QPS is NOT_MEASURED unless a separately controlled offered-rate test is executed; inverse latency is not QPS.

## Claim hierarchy

Quality advantage requires the paired effect analysis; operational feasibility requires measured workload-specific latency/failure/memory; replacement justification requires both and is qualitative because no prospective noninferiority margin was frozen. SciFact and GaRAGe quality are not pooled. E6-4 is not executed. Jev/paid calls remain zero.
