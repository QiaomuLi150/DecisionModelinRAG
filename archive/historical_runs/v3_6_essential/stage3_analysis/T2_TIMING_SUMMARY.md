# v3.6 T2 timing summary

T2 completed three frozen C120 sessions (A/B/C), each with three attempts per ID, on GPUs 3/6/3. B6.2 completed a separate GPU-isolated solo C120 session on GPU3 with `jina_full`, `jina_single_8192`, and `pplx_budget`. All serving traces remained label-blind; failure elapsed time is retained and the all-request analysis adds only the measured label-free packaging fallback cost to failed rows. See `export/t2_ratios.csv`.
