# Paper-to-release crosswalk

Paper source used for this crosswalk: arXiv v1 PDF, <https://arxiv.org/pdf/2610.34816>. The local TeX/source bundle was not available. This is a mapping of extant evidence, not a claim that every paper table or figure can be regenerated.

| Paper item | Local source/evidence | Regeneration | Status / limitation |
|---|---|---|---|
| Table I: sparse model screen | `results/01_model_screen/e2_n_limited_screen/` | `verify/verify_paper_numbers.py` | Blocked for paper table: E2-N is a different 12-policy study; the 21-model/11-benchmark/104-run source matrix is missing. |
| Table II: benchmark counts and run totals | `verify/expected_numbers.json`; `protocols/cqadupstack_download_manifest.json`; `results/06_transfer_enterprise_musique/` | `verify/verify_paper_numbers.py` | Partial; only selected populations have source manifests. |
| Table III: default BRIGHT policies | `data/scores/bright_metrics.jsonl.gz`; `results/02_bright_default_and_budget/bright_summary.csv` | `scripts/make_paper_assets.py` → `paper/regenerated/bright_summary.csv`, `bright_e9_model_summary.csv` | Partial; coverage limited to released model policies. |
| Table IV: 8,192-token Jina truncation | `data/scores/bright_perquery.csv`; `results/02_bright_default_and_budget/bright_summary.csv` | `scripts/make_paper_assets.py` | Published 0.1765 differs from the current exported no-fallback mean 0.1738; the older reused row is not independently reconstructed here. |
| Table V: candidate order and two-stage controls | `results/03_order_and_multistage/`; `archive/jina_diagnostic/` | `scripts/make_paper_assets.py` | Both Jina two-stage values 0.1790 and 0.1855 remain separate; their cause is unresolved. |
| Table VI: input budgets and selection adjustment | `data/scores/bright_perquery.csv`; `results/05_latency_and_memory/frontier_points.csv` | `scripts/make_paper_assets.py`; `verify/verify_paper_numbers.py` | Budget summaries regenerate; max-t/Bonferroni/exact validity intervals remain `NOT_VERIFIABLE` because draws/source outputs are absent. |
| Table VII: replacement frontier | `results/05_latency_and_memory/`; `results/07_frozen_cqadupstack/` | `scripts/make_paper_assets.py` → frontier data CSV | Source data are partial; paper plot formatting and some exact timing/model-memory source populations are missing. |
| Table VIII: Enterprise/MuSiQue transfer | `data/scores/transfer_perquery.csv`; `results/06_transfer_enterprise_musique/` | `scripts/make_paper_assets.py` → `transfer_summary.csv` | 2,667 requests: 500 Enterprise (443 labeled), 2,167 MuSiQue. 333 MuSiQue slates were never frozen. |
| Figure 1: evaluation pipeline | No source generator/artwork located | None | Not regenerated. |
| Figure 2: failure decomposition | Partial diagnostic records only | None | Not regenerated; complete plotting inputs are absent. |
| Figure 3: quality/latency frontier | `results/05_latency_and_memory/frontier_points.csv` | `scripts/make_paper_assets.py` → `frontier_points.csv` | Data are generated; plot pixels are not, because the source plot code/settings are absent. |

Run `python verify/verify_paper_numbers.py` after generating the outputs. Its `FAIL` and `NOT_VERIFIABLE` records must be read as evidence gaps or mismatches, not resolved by editing values.
