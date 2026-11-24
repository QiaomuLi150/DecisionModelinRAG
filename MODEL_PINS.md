# Model checkpoint registry

No weights, tokenizers, caches, or model repositories are included. Revisions below are copied from the local run/manifests used for the release inventory; they are not inferred from current model-card defaults. Run-specific inputs and precision are recorded in `archive/historical_runs/` and the corresponding protocol exports.

| Paper model | Official repository | Recorded paper-row revision | Size from paper |
|---|---|---|---:|
| MiniLM-L6 | `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 0.02B |
| GTE-ModernBERT | `Alibaba-NLP/gte-reranker-modernbert-base` | `f7481e6055501a30fb19d090657df9ec1f79ab2c` | 0.15B |
| BGE-v2-m3 | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 0.57B |
| Qwen3-Reranker-0.6B | `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 0.6B |
| Jina v3.5 | `jinaai/jina-reranker-v3.5` | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | 0.6B |
| Mixedbread v2 | `mixedbread-ai/mxbai-rerank-large-v2` | `ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6` recorded in the historical model-lineage manifest | 1.5B |
| Qwen3-VL-Reranker-2B | `Qwen/Qwen3-VL-Reranker-2B` | `4bd860ac4f15ad1897a214615cccc700f8f71818` | 2.1B |
| Qwen3-Reranker-4B | `Qwen/Qwen3-Reranker-4B` | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | 4B |
| zerank-2 | `zeroentropy/zerank-2-reranker` | `5eae30d5ee3c6b2df2ef6d723bde45172d761c4c` | 4B |
| Laya | `convaiinnovations/laya` | `7b928d828b7b0e022f929d9bd2e44165aa270148` | 0.42B |
| Amos 1 Small | `mukaddamzaid/amos-1-small` | `09ba3e9d8f506cfcb96ea130845d5d76d9d92028` | 2B |
| Liquid d1-3B | `LiquidAI/d1-3B` | `051bcc464b01b9f92942b364d9586b0ef5912432` | 3.1B |
| Intern-Decision-4B | `internlm/Intern-Decision-4B` | `0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd` | 4.5B |
| StartLux-Decision-4B | `StartLuxLabs/StartLux-Decision` | `9c0a32e9809d6a3f612434dd0bce150295dcc274`; official model page needs owner confirmation | 4.7B |
| Drex-DLM | `nace-ai/drex-dlm` | `8f6f8244ce0b186b3caa4cdbaae1c9f3986bf63e` | 8.2B |
| Kev-9B | `jaredpalmer/kev-9b` | `db029f08b290afd9fee4aa4bbcd9ae48602d1eb0` | 9B |
| Bespoke Nimble-9B-v2 | `bespokelabs/Bespoke-Nimble-9B-v2` | `4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe` | 9B |
| Clef-Flash | `Cloudflare/clef-flash` | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | 9.4B |
| Winnow-12B | `EldanRing/Winnow-12B` | `0b5b7f98e718cf0faffd0acea3093b86ebf00b56` paper-row pin; separate later Q8 run `b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea` | 12B |
| PPLX Decider v1.1 | `perplexity-ai/pplx-decider-v1.1-27b` | `14c9d90137962d300f3de95633f7b6535e80c5ce` | 27B |
| Drex-v2 | `nace-ai/drex-v2.0` | `3e6d164cb5c8cf264eec167905cc2448d89ea97d` | 35B total, 3B active |

Official model pages were checked on Hugging Face for all listed IDs except StartLux, whose page could not be confirmed by the available lookup. The README uses `TODO-owner` for that link.

## Auxiliary checkpoints in historical code

These checkpoints support other archived pipeline work and are not part of the paper's 21 reranker-screen roster.

| Role | Repository | Revision | Notes |
|---|---|---|---|
| Qwen3-VL-Embedding-2B | `Qwen/Qwen3-VL-Embedding-2B` | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | Historical retrieval/embedding service; not a reranker-screen row |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Historical reader/adjudication utility; downstream reader evaluation is not part of the paper's main comparison |

Repository IDs and revisions can be fetched only after checking each model's terms and the relevant run protocol. This release does not provide a turnkey model downloader for unpinned or license-restricted checkpoints.
