# Data and model licensing decisions (reviewed 2026-11-24)

No model weights, tokenizer files, or benchmark text are included. Results and derived tables are not relicensed where upstream terms may apply. `CC BY 4.0` for derived data is a proposal only where permitted; owner/legal review is required before public release.

| Asset | Official source and observed license | Release decision |
|---|---|---|
| BRIGHT | Official repository `https://github.com/xlang-ai/BRIGHT` and dataset card `https://huggingface.co/datasets/xlangai/BRIGHT` identify the dataset; the official Hugging Face card declares CC-BY-4.0 (reviewed 2026-11-24). | No query/document text copied. IDs, hashes and derived scores only; attribution retained. |
| BEIR / CQADupStack | BEIR official dataset registry `https://github.com/beir-cellar/beir` explicitly says its preparation does not certify dataset rights; CQADupStack derives from Stack Exchange content, whose original content terms must be checked at `https://stackoverflow.com/legal/terms-of-service/public` and `https://creativecommons.org/licenses/by-sa/3.0/`. | Do not redistribute corpus, queries or qrels in this package. Publish IDs, hashes and derived scores only; downloader points to official BEIR source and requires user to accept upstream terms. |
| MuSiQue | Official repository `https://github.com/StonyBrookNLP/musique` states data is CC BY 4.0. | No text/qrels copied in this package; IDs, hashes and derived scores only. Attribution retained. |
| EnterpriseRAG-Bench | Official repository has an MIT `LICENSE` for repo materials at `https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/LICENSE`; the asset/license coverage issue is tracked in that repository and does not clearly grant the corpus/labels under the code license. | Treat benchmark files as not redistributable here; release IDs/hashes/derived scores only. Owner must obtain explicit dataset-asset clarification before broader redistribution. |
| PPLX Decider v1.1 27B | Model card/revision `https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/14c9d90137962d300f3de95633f7b6535e80c5ce` declares Apache-2.0. | Weights excluded. Pin and download instructions only. |
| Jina Reranker v3.5 | Model card/revision `https://huggingface.co/jinaai/jina-reranker-v3.5/tree/e8a93f3` declares CC BY-NC 4.0. | Weights excluded. Pin and download instructions only; non-commercial restriction must be respected. |
| BGE reranker v2-m3 | Official card `https://huggingface.co/BAAI/bge-reranker-v2-m3` declares Apache-2.0. | Weights excluded. |
| Qwen3 Reranker 0.6B / 4B | Official cards `https://huggingface.co/Qwen/Qwen3-Reranker-0.6B` and `https://huggingface.co/Qwen/Qwen3-Reranker-4B` declare Apache-2.0. | Weights excluded. |
| Remaining models in paper Table I | 21-model historical roster and exact pins are not available as one authenticated file in the local workspace. | Do not claim a complete license audit; do not redistribute weights. List as unresolved in `MISSING.md`. |

## Source review notes

The BEIR project warns that it only distributes datasets in a common format and does not claim that users have permission to use those data. The EnterpriseRAG-Bench repository's code license alone is not treated as proof that every benchmark asset is MIT-licensed. Unknown or ambiguous dataset rights default to non-redistribution. The public download script never downloads automatically during package verification.

Official sources: [BRIGHT repository](https://github.com/xlang-ai/BRIGHT), [BEIR dataset registry](https://github.com/beir-cellar/beir), [MuSiQue repository](https://github.com/StonyBrookNLP/musique), [EnterpriseRAG-Bench license](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/LICENSE), [PPLX Decider card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/14c9d90137962d300f3de95633f7b6535e80c5ce), [Jina v3.5 card](https://huggingface.co/jinaai/jina-reranker-v3.5/tree/e8a93f3), [BGE card](https://huggingface.co/BAAI/bge-reranker-v2-m3), and [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) / [4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B).
