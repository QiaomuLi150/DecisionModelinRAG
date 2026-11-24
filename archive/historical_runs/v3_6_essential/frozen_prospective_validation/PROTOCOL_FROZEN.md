# DecisionRAG Frozen Prospective Validation Protocol

Protocol ID: `decisionrag-v3.6-frozen-prospective-validation-v1`
Status: FROZEN BEFORE MODEL INFERENCE
Freeze seed: `731`

## Objective

Evaluate the two BRIGHT-selected PPLX context settings on one fresh, text-only dataset whose candidate lists have not been used in the DecisionRAG experiments. The confirmation contrast is between `pplx_single_4096` and `pplx_single_6144`, with full-text Jina, Jina matched-text at 8192, and BM25 Top-5 as fixed comparators.

## Dataset admission and frozen population

The mandated dataset order is: CQADupStack, ArguAna, SciDocs, FiQA-2018, and Natural Questions (BEIR). ArguAna is the first locally complete and admissible dataset. The local BEIR ArguAna snapshot is used without modification:

- Corpus: 8,674 documents.
- Queries: 1,406.
- Qrels: 1,406 test-query entries.
- Dataset is not one of the excluded benchmarks: 2WikiMultiHopQA, BRIGHT, CRAG, EnterpriseRAG-Bench, FinanceBench, garage, HotpotQA, MultiHop-RAG, MuSiQue, QASPER, or SciFact.
- The complete 1,406-query population is frozen because it is below the 2,000 target and above the 1,000 minimum. No outcome-based sampling is performed.
- The dataset was selected before reading query or qrel content for evaluation decisions. All qrels remain absent from serving inputs and are joined only after raw selector outputs are frozen and hashed.

Dataset files and revisions are recorded in `DATASET_ADMISSION.json` and `DATASET_FILE_MANIFEST.json`.

## Retrieval contract

Use the same first-stage BM25 implementation and settings as the frozen BRIGHT retrieval implementation:

- Pyserini Lucene analyzer.
- Gensim `LuceneBM25Model` with `k1=0.9`, `b=0.4`.
- Global corpus search over the admitted ArguAna corpus.
- Candidate order is descending BM25 score with stable original-corpus order for exact ties.
- Freeze exactly 20 unique candidate IDs per query before any qrel join.
- No qrels, gold answers, or relevance fields are available to retrieval or selector processes.

The immutable candidate manifest is `ARGUANA_FROZEN_TOP20_SLATES.jsonl`; its hash is recorded in the final provenance manifest.

## Selector contracts

The pinned native revisions are:

- PPLX Decider: `perplexity-ai/pplx-decider-v1.1-27b`, revision `14c9d90137962d300f3de95633f7b6535e80c5ce`.
- Jina v3.5: `jinaai/jina-reranker-v3.5`, revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`.

Policies are fixed before inference:

1. `pplx_single_4096`: PPLX native choice scoring using the frozen exact R1 decrement-to-fit request builder at `L=4096`, safety margin 128 tokens.
2. `pplx_single_6144`: same at `L=6144`.
3. `jina_full_text`: Jina native scoring over the original Top-20 passage strings.
4. `jina_trunc_8192`: Jina native scoring over the frozen exact PPLX-builder strings at `L=8192`.
5. `bm25_top5`: first-stage BM25 order, taking the first five candidates; no model calls.

The PPLX input builder measures the exact serialized request with candidate IDs, template and separators, then decrements one shared per-candidate token cap until the serialized request is at most `L-128`. Builder failures are retained as failures for every policy using that builder; no retries or fill-in are allowed.

Every model policy must return exactly five unique canonical candidate IDs. Scores are joined to canonical IDs before ranking. Stable tie-breaking is original candidate order, then canonical ID. Failed requests receive quality score zero in all-request metrics.

## Execution and serving contract

1. Run a non-confirmatory dry run on 20 prespecified query IDs, excluded from the confirmation scorecard.
2. Run Pass 1 on all 1,406 queries, using a deterministic seeded random policy order independently per query.
3. Run Pass 2 as timing-only, reversing the Pass-1 policy order for each query.
4. Use serial requests within each worker, batch size 1, prefix caching off, the frozen serving stack and launch arguments. One worker is assigned per GPU shard. Workers use disjoint query shards.
5. Before model loading and after completion, record GPU device snapshots. Record a contention snapshot every 100 requests. If a GPU has less than approximately 56 GiB free for PPLX plus Jina, it may run only a single-model shard; the corresponding result is marked unpaired and is not treated as a matched resource comparison.
6. Each worker runs 60 full-text Jina anchor requests at the beginning and 60 at the end of the timed session. Anchor drift is reported; anchors are not part of the quality population.
7. Record preprocessing, native scoring, parsing, ID mapping, packaging, status, failure class, elapsed time, peak allocated memory, physical forward count, model revision, GPU assignment, request hash and raw trace hash. Do not record labels in serving traces.
8. No retries. A crashed shard may be resumed only for unfinished query IDs, with the restart and non-overlap recorded; completed records are immutable.

## Statistical contract

- Primary quality metric: known-positive nDCG@5, with failed or invalid selectors scored zero.
- Secondary: strict valid rate, Top-20 positive availability, judged-only nDCG@5, and BM25 Top-5 control.
- Report exact Clopper-Pearson lower bounds for strict validity.
- Bootstrap paired quality differences over query IDs, 40,000 draws, seed 731. The resampling unit is query ID; no natural topic clusters are available in the admitted dataset metadata.
- Apply Bonferroni correction across the two PPLX settings for the quality gate.
- `R` passes only when strict valid rate is at least 99% and its exact lower confidence bound supports the declared threshold.
- `Q` requires the lower 95% bound for PPLX minus BM25 to be positive, the lower bound for PPLX minus full-text Jina to be greater than -0.02, and the second margin bound to exceed 40% of (full-text Jina minus BM25), with the two PPLX settings corrected by Bonferroni.
- `S` passes when the upper paired 95% confidence bound for p95 latency ratio to same-query full-text Jina is at most 2; it fails when the lower bound is greater than 2; otherwise it is inconclusive.
- A PPLX setting is a supported decision only if R, Q and S all pass. All other outcomes are reported as fail, inconclusive or blocked; no favorable selection is permitted.

## Length and reporting strata

Before inference, compute PPLX-tokenizer lengths for the query and each original passage. Report all quality and latency results by passage-length tercile defined from the frozen full-candidate length distribution. Do not change tercile boundaries after outcomes are observed.

## Integrity restrictions

No paid APIs, Jev, new model families, training, reader inference, retrieval tuning, label-dependent candidate modification, or protected-holdout access. All new files are append-only below this directory. Historical artifacts are not overwritten.

