# Can System-1-Style Decision Models Replace Dedicated Rerankers in RAG? A Systems Evaluation

## Paper

**Title:** Can System-1-Style Decision Models Replace Dedicated Rerankers in RAG? A Systems Evaluation<br>
**Authors:** Qiaomu Li and Ying Xie<br>
**arXiv:** [2610.34816](https://arxiv.org/abs/2610.34816) — the arXiv page title was verified to match.<br>
**Paper files:** [`paper/`](paper/), including [`paper/DecisionRAG_paper_v6_6.pdf`](paper/DecisionRAG_paper_v6_6.pdf) and the TeX source in [`paper/arxiv_source/`](paper/arxiv_source/).

```bibtex
@misc{li2026system1decisionrerankers,
  title={Can System-1-Style Decision Models Replace Dedicated Rerankers in RAG? A Systems Evaluation},
  author={Li, Qiaomu and Xie, Ying},
  year={2026},
  eprint={2610.34816},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.34816}
}
```

## Study summary and repository contents

This study evaluates whether a structured decision model can replace a specialist reranker at the fixed Top-20-to-Top-5 stage of text RAG. It combines a sparse historical screen with matched BRIGHT ranking, input-budget, candidate-order, two-stage, serving, transfer, and frozen CQADupStack evidence. Results vary by workload; failures, contended/superseded runs, and unresolved discrepancies remain visible. The historical screen is heterogeneous, not a controlled 21-by-11 matrix.

`results/01_model_screen/` contains Table I transcribed from the paper because the original per-run files were not found in the workspace. These transcribed values are not a substitute for the missing per-run records.

Included are experiment adapters and runners; request builders; serving, scoring, bootstrap, and table-generation code; sanitized score, selection, latency, failure, and protocol records; and locally available paper figures/tables. Benchmark text, qrels, model weights, tokenizers, and caches are excluded.

## Reproducibility status

This release supports **artifact-integrity checks and selected offline analyses**. It is **not a turnkey reproduction of the complete historical project**. Full native inference requires model weights, tokenizer assets, benchmark/corpus files, and machine-specific services that are not distributed. Some historical raw traces and result bundles are also unavailable or intentionally excluded; `MISSING.md` records the resulting claim-level limits. The included full historical test suite refers to those omitted stage artifacts and is not expected to pass from this release alone. Do not interpret the passing package checks as reproduction of every paper result.

On a clean checkout with Python 3.12, the self-contained checks are:

```sh
python3 verify/verify_hashes.py
python3 verify/run_integrity_tests.py
python3 verify/compile_scripts.py
python3 verify/audit_git_history.py
```

The broader historical suite can be inspected with `python3 -m pytest -q code/tests`; its missing-input failures are expected for this release and are not waived or converted into passes. No model inference is launched by the three self-contained checks above. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the audited scope and limitations and [`PUBLIC_REDACTION_REPORT.md`](PUBLIC_REDACTION_REPORT.md) for release sanitation and its exact-input caveat.

## Models

The 21 checkpoints below are those in the paper. Revisions are copied from available run/lineage manifests; see [MODEL_PINS.md](MODEL_PINS.md) for run-specific notes. `not recovered` means the exact revision for the paper row was not recorded in the available source manifests. Sizes follow the paper's Table I. Drex-v2 is a 35B MoE with 3B active parameters.

| Family | Model | Official page | Version / revision recorded | Parameters |
|---|---|---|---|---:|
| Specialist | MiniLM-L6 | [HF](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 0.02B |
| Specialist | GTE-ModernBERT | [HF](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | `f7481e6055501a30fb19d090657df9ec1f79ab2c` | 0.15B |
| Specialist | BGE-v2-m3 | [HF](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 0.57B |
| Specialist | Qwen3-Reranker-0.6B | [HF](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 0.6B |
| Specialist | Jina Reranker v3.5 | [HF](https://huggingface.co/jinaai/jina-reranker-v3.5) | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | 0.6B |
| Specialist | Mixedbread v2 | [HF](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) | `ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6` in model-lineage manifest | 1.5B |
| Specialist | Qwen3-VL-Reranker-2B | [HF](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | `4bd860ac4f15ad1897a214615cccc700f8f71818` | 2.1B |
| Specialist | Qwen3-Reranker-4B | [HF](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | 4B |
| Specialist | zerank-2 | [HF](https://huggingface.co/zeroentropy/zerank-2-reranker) | `5eae30d5ee3c6b2df2ef6d723bde45172d761c4c` | 4B |
| Decision | Laya | [HF](https://huggingface.co/convaiinnovations/laya) | `7b928d828b7b0e022f929d9bd2e44165aa270148` | 0.42B |
| Decision | Amos 1 Small | [HF](https://huggingface.co/mukaddamzaid/amos-1-small) | `09ba3e9d8f506cfcb96ea130845d5d76d9d92028` | 2B |
| Decision | Liquid d1-3B | [HF](https://huggingface.co/LiquidAI/d1-3B) | `051bcc464b01b9f92942b364d9586b0ef5912432` | 3.1B |
| Decision | Intern-Decision-4B | [HF](https://huggingface.co/internlm/Intern-Decision-4B) | `0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd` | 4.5B |
| Decision | StartLux-Decision-4B | `TODO-owner` (source repo ID: `StartLuxLabs/StartLux-Decision`; official page could not be confirmed) | `9c0a32e9809d6a3f612434dd0bce150295dcc274` | 4.7B |
| Decision | Drex-DLM | [HF](https://huggingface.co/nace-ai/drex-dlm) | `8f6f8244ce0b186b3caa4cdbaae1c9f3986bf63e` | 8.2B |
| Decision | Kev-9B | [HF](https://huggingface.co/jaredpalmer/kev-9b) | `db029f08b290afd9fee4aa4bbcd9ae48602d1eb0` | 9B |
| Decision | Bespoke Nimble-9B-v2 | [HF](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) | `4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe` | 9B |
| Decision | Clef-Flash | [HF](https://huggingface.co/Cloudflare/clef-flash) | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | 9.4B |
| Decision | Winnow-12B | [HF](https://huggingface.co/EldanRing/Winnow-12B) | `0b5b7f98e718cf0faffd0acea3093b86ebf00b56` (paper-row); later Q8 run has a separate revision | 12B |
| Decision | PPLX Decider v1.1 | [HF](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | `14c9d90137962d300f3de95633f7b6535e80c5ce` | 27B |
| Decision | Drex-v2 | [HF](https://huggingface.co/nace-ai/drex-v2.0) | `3e6d164cb5c8cf264eec167905cc2448d89ea97d` | 35B (3B active) |

Auxiliary reader and embedding checkpoints appear in historical code outside the 21-model reranker screen. They are listed separately in `MODEL_PINS.md`; no weights or tokenizers are included.

## Benchmark datasets

| Benchmark | Official page / paper | Split or frozen cohort used |
|---|---|---|
| BRIGHT | [repository](https://github.com/xlang-ai/BRIGHT) · [dataset card](https://huggingface.co/datasets/xlangai/BRIGHT) | 1,384 queries; frozen original BM25 Top-20 for controlled study |
| SciFact | [BEIR dataset card](https://huggingface.co/datasets/BeIR/scifact) | 277-query historical screen cohort |
| QASPER | [dataset card](https://huggingface.co/datasets/allenai/qasper) | 1,502-query frozen study cohort |
| HotpotQA | [official site](https://hotpotqa.github.io/) | 800-query historical screen cohort |
| MuSiQue | [repository](https://github.com/StonyBrookNLP/musique) | 2,167-query screen cohort; separate transfer cohort also has 2,167 frozen questions |
| 2WikiMultihopQA | [repository](https://github.com/Alab-NII/2wikimultihop) | 800-query historical screen cohort |
| MultiHop-RAG | [repository](https://github.com/yixuantt/MultiHop-RAG) | 900-query historical screen cohort |
| GaRAGe | [repository](https://github.com/amazon-science/GaRAGe) | 1,397-query fixed-pool screen cohort |
| EnterpriseRAG-Bench | [repository](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) · [dataset card](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) | 500 requests; 443 have resolvable labels |
| FinanceBench | [repository](https://github.com/patronus-ai/financebench) · [dataset card](https://huggingface.co/datasets/PatronusAI/financebench) | 109-query historical screen cohort |
| CRAG | [repository](https://github.com/facebookresearch/CRAG) | 300-query historical screen cohort |
| BEIR CQADupStack | [BEIR repository](https://github.com/beir-cellar/beir) | Test split, 12 forums; 13,145 frozen requests, 20 dry-run exclusions, 13,125 scored |

The historical screen used different candidate views and metrics across datasets; the 104 runs are not one shared-protocol grid. The exact cohort manifests and protocols are indexed under `protocols/` and `archive/`.

Models and datasets are not included in this repository. Obtain them from the official links above and follow their licenses.

## Repository map and recorded commands

| Path | Paper mapping |
|---|---|
| `code/scripts/`, `code/src/`, `code/tests/`, `code/configs/`, `code/results_support/`, `code/pyproject.toml` | Adapters/runners, candidate builders, truncation, serving, scoring, bootstrap, table/figure code, regression fixtures, recorded run wrappers, and project metadata |
| `results/01_model_screen/` | Table I transcribed from the paper; original per-run files were not found |
| `results/02_bright_default_and_budget/` | Tables III–IV and default/input-budget comparisons |
| `results/03_order_and_multistage/` | Table V candidate-order, multi-stage, and matched-text controls |
| `results/05_latency_and_memory/` | Tables VI–VII latency, reliability, memory, and replacement frontier |
| `results/06_transfer_enterprise_musique/` | Table VIII transfer evidence |
| `results/07_frozen_cqadupstack/` | Frozen BEIR CQADupStack validation and timing |
| `results/08_controlled_replication/` | Controlled replication evidence |
| `archive/historical_runs/` | Additional sanitized historical/diagnostic result records, including failures and superseded versions |
| `protocols/` | Frozen protocols, cohort files, hashes, and sanitized copies |
| `raw_traces/`, `data/` | Score/selection/latency projections; no benchmark text or qrels |
| `paper/` | Paper PDF, TeX source, figures, and tables |
| `environment/` | Recorded runtime/hardware notes |

The historical runners under `code/scripts/` are preserved for inspection, but many depend on local paths and inputs omitted from this release. They are not presented as executable end-to-end reproduction commands. The supported offline verification commands are listed above; [`scripts/make_paper_assets.py`](scripts/make_paper_assets.py) regenerates selected paper tables from packaged data, not the full inference study.

Run inventory and discrepancies are indexed in [RESULTS_INDEX.md](RESULTS_INDEX.md), [MISSING.md](MISSING.md), and [protocols/HASHES.md](protocols/HASHES.md). Size/data exclusions are in [RELEASE_ASSETS_TODO.md](RELEASE_ASSETS_TODO.md).

## Large files

No `data-v1` release assets are published. The 35 original source artifacts listed in [`RELEASE_ASSETS_TODO.md`](RELEASE_ASSETS_TODO.md) are not included in this repository; that inventory records each source path, byte size, SHA-256, and disposition. They include raw benchmark/model-input material or artifacts whose redistribution rights are unresolved. Under [`LICENSE-DATA.md`](LICENSE-DATA.md), these files are not cleared for redistribution, so no release download URLs are available.

The authors used LLM assistants to write analysis scripts, draft figures and tables, and edit the text. The authors reviewed and validated the resulting analyses, figures, tables, and text and take full responsibility for the paper's content and findings.
