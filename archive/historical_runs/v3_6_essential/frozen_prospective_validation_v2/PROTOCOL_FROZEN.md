# DecisionRAG Frozen Prospective Validation v2

Protocol ID: `decisionrag-v3.6-frozen-prospective-validation-v2`
Status: FROZEN BEFORE CONFIRMATORY MODEL CALLS
Freeze seed: `731`

## 1. Fixed dataset and release

The sole dataset is the official BEIR CQADupStack `test` release, all twelve forums: `android`, `english`, `gaming`, `gis`, `mathematica`, `physics`, `programmers`, `stats`, `tex`, `unix`, `webmasters`, and `wordpress`. There is no fallback dataset and no post-outcome dataset search.

- Official release URL: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip`
- Downloaded archive SHA-256: `6072f7d345496387d24194c8af35c5fb6c2f0e5f9130c5b4a78b98bbfac88558`
- Official release table MD5: `4e41456d7df8ee7760a7f866133bda78`
- Release file date: 2021-03-09.
- Population: all 13,145 query records; IDs are domain-qualified as `forum:query_id`.
- Qrels are used only after raw selector traces are closed and hashed.

Per-forum corpus and qrels SHA-256 values are frozen here:

| Forum | Corpus SHA-256 | Qrels `test.tsv` SHA-256 |
|---|---|---|
| android | `aa1bad1bd193e374da8bf30299487d24059c7555a2d437f33e15b7e071cbae95` | `8965c2259e76144f857a36b197f2dedf073e838c5196e6720990ba10251e322b` |
| english | `1ba0eed2fc9c0f4afe93325bff8bddca52004884e2969c8a157604b6ecbfcfc8` | `d9377c26abd502f470a5ff331e36209662fc5d091fe824201e663620b5434b6a` |
| gaming | `48c63ce9f5bdff61344693de5b5b0e8657ecd08daa87d7cbb95ad77c074b8a78` | `a39736b2be1cf8922ed3a7c56017c02a5492131cc75156d77c6c3825124ba4dc` |
| gis | `18d89c9f82284310ec1438076c28c7411e098b55d23ecfa48915e74014d0e5f3` | `45a27e08f2371d72dcffff27a7d5d66e282fa2f022e6f46e2b7dc3f1dbf66b2e` |
| mathematica | `8d8824a0c7f8ee2aff8bc889d628e97d6d110b90253d6a5a23b8f70023cf9f63` | `328d123124f9cffde8508a39c05954ec8dce81fd1eae8658b729022c374cd342` |
| physics | `7c42540150ef9100e0b69d26d9908f8ee6d374020d3f12bbc4c7aa7d6988ab64` | `3550c476a608c75550dfdda2a9d9551416f514660e6268b8153131d80df393df` |
| programmers | `c652ce07f0040a7ee4edad8d63d13e93170665dd0f412ab65eb607c80e1a583f` | `ed69afa873c64d34298792d694c82c4293e4b7e3c6ea3edc7a73662d4b0cba76` |
| stats | `cb1e8ba2193dc71ce0efa878a068a7b7cae16bb00ad8e504c71282301beecb85` | `819b62de1d83923cf83def7f161726e7bff88f69c0840cb391b50c2f7f0f89e8` |
| tex | `4a233f5bfe30a9d8afbf6a1fd3c62292407c5eb6fb97943c78bc2bb1b6e9b5aa` | `ec6952e95a549782a8d3b304ef0effaf12f24c567a7cbc0ae9f6d6328a43ff04` |
| unix | `a52e7fb1f7370613eae79143b75ce0ea0a4be1bb2c908c6d9f83431fe2cf9a1a` | `a7bd24ce081b0bb892602e9b0067017dc66fd03e7593563ad095cb3cd05e92ed` |
| webmasters | `b9c00bb5ce58c648442e80f2d05d619b63567bf2cafede2979dba3282446abb5` | `6b56b9ff5ee60346934d12453dab985648c01ad2b2c67abec1d225e08ecc28de` |
| wordpress | `4820c9b476a9d48b9a915741941650203baf737a71a984ed591df3007d09ba5e` | `bfd319c103eac106e054399160f874466a018b54548b6a474645ed80a2281b78` |

The complete file manifest, query counts and qrel counts are in `DATASET_FILE_MANIFEST.json`.

## 2. Candidate generation

For each forum separately, generate a global-for-that-forum BM25 Top-20 list using the same implementation as the frozen BRIGHT retrieval:

- Pyserini Lucene analyzer.
- Gensim `LuceneBM25Model`, `k1=0.9`, `b=0.4`.
- Gensim 4.4.0, Transformers 5.16.1, Torch 2.13.0+cu130; the pyserini package is recorded by its installed distribution metadata.
- Stable descending score order, retaining original corpus order on exact ties.
- No qrels, labels, or relevance fields are loaded by the retrieval or serving process.
- Candidate IDs are serialized as `forum:document_id` and every frozen slate contains exactly 20 unique IDs.

The retrieval and preparation implementation is frozen at repository commit `246ff54e66b178eadd350351a5544879f5484ed2` and the candidate file hash is written to `val_candidates.sha256` before any confirmatory model call.

## 3. Frozen policies and model contracts

Exactly these policies are admitted:

1. `pplx_single_6144` — primary; native PPLX choice scoring with the frozen R1 equal-prefix builder, `b(q)=max(0,floor((L-h(q)-128)/20))`, followed by decrement until the exact serialized request is at most `L-128`.
2. `pplx_single_4096` — secondary; identical contract at `L=4096`.
3. `jina_full_text` — native Jina scoring on the original Top-20 text.
4. `jina_trunc_8192` — native Jina scoring on the exact PPLX-builder text at `L=8192`.
5. `bm25_top5` — first-stage BM25 prefix control; no model call.

Pinned revisions:

- PPLX Decider `perplexity-ai/pplx-decider-v1.1-27b`, `14c9d90137962d300f3de95633f7b6535e80c5ce`.
- Jina v3.5 `jinaai/jina-reranker-v3.5`, `e8a93f33f0b22108f8c2364f8484ce3422552fbc`.

Each selector must emit exactly five unique candidate IDs from the original Top-20. Any malformed, missing, duplicated, out-of-list or otherwise invalid output is a failed request scored zero. Ties are resolved by original candidate order and then canonical ID. No fallback, retry, fill-in or outcome-dependent prompt change is allowed. Serving launch arguments, library versions and code hashes are in `SERVING_ARGS.json`.

## 4. Cohorts and procedure

- Dry run: 20 IDs chosen from the sparse-qrel forums after sorting domain-qualified IDs; these IDs are outside the confirmatory scorecard and are engineering-only.
- Pass A quality/validity: all remaining frozen query IDs, sharded by SHA-256 of the domain-qualified ID across all usable GPUs. Busy GPUs may be shared for Pass A. There are no client timeouts that manufacture failures; genuine timeout/OOM/error records remain explicit, and only unfinished IDs may be resumed on a clean GPU.
- Latency cohort: the first 2,000 domain-qualified IDs after sorting by SHA-256(`ID + "731"`). The exact list is frozen in `LATENCY_COHORT_IDS.jsonl`.
- Pass B latency: two serial, batch-size-one passes on clean GPUs only. Each worker loads PPLX and Jina in one process, evaluates every admitted policy paired on the same GPU, disables prefix caching, uses deterministic seeded random policy order in pass 1 and the exact reverse order in pass 2. No PPLX-only/Jina-only latency shards are valid.
- Every Pass B worker executes 60 full-text Jina anchors at start and 60 at end. Other-process utilization is recorded every 100 requests. Sustained contention invalidates that worker block and requires a clean rerun.
- A worker crash is recorded; only unfinished query IDs may be resumed.

## 5. Statistical checks and verdicts

Primary unit is whole forum (12 forums). The primary pooled all-query mean uses 40,000 forum bootstrap draws, seed 731, with forum-level resampling. Query-level bootstrap stratified by forum is sensitivity only. Primary scoring is all-request nDCG@5, failures zero. Report the judged-positive Top-20 share and judged-only values separately.

- `R`: strict valid Top-5 rate at least 99%; report exact Clopper–Pearson interval.
- `Q`: lower 95% bound of nDCG@5 difference versus BM25 Top-5 greater than zero and versus full-text Jina greater than `-0.02`. The 40%-of-Jina-over-BM25 margin is sensitivity only.
- `S`: p95 latency ratio to same-query full-text Jina; upper 95% bound at most 2 passes, lower bound above 2 fails, otherwise inconclusive.
- Observed pass: observed R/Q/S pass.
- Supported pass: observed pass plus Bonferroni-2-adjusted criteria: exact Clopper–Pearson lower valid-rate bound at least 99%, Q lower bounds at two-sided 97.5%, and S upper ratio bound at 97.5% at most 2.
- Fail: a check is demonstrably not met, including validity upper bound below 99%, quality upper bound below threshold, or S lower ratio bound above 2.
- Inconclusive: none of the fail conditions and not a pass.

The headline category is for `pplx_single_6144`; the 4096 result is separate. No results are dropped, resampled, or re-chosen after model output.

## 6. Power and integrity notes

At approximately 99.2% reliability, on the order of 10,000 requests is needed for the exact lower validity bound to reach 99%; this motivates the full 13,145-query quality population and the fixed 2,000-query latency subset. Any implementation repair requires a versioned amendment and new code hash before confirmatory scoring. No paid APIs, Jev calls, new models, training, reader generation, protected holdouts, or manuscript work are permitted.

