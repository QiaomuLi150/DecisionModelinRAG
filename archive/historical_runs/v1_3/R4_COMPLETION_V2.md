# R4 completion

Gate: `R4_COMPLETE_WITH_VERSIONED_READER_REPAIRS`

The corrected versioned R4 run produced 2,400/2,400 successful reader outputs: four frozen selection policies × 300 CRAG questions × 512/1024-token windows. The Qwen3-VL-8B checkpoint revision is `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Raw output SHA-256 is `0a707cafb47cd1e416691d35c8dd0e5fc57eec9bf052eb9089e0ab6f94891a99`.

Offline scoring was corrected to use exact tokenizer-token answer offsets; the original failed attempts and the prior word-offset score are preserved. The corrected score files are `r4_crag_reader_window_v2/export/crag_reader_window_v3.csv` and `r4_crag_reader_window_v2/export/crag_reader_window_summary_v3.json`. The metrics are deterministic EM/F1/reference proxies, not independently verified factual correctness.

The post-run GPU snapshot showed an unattributed `python3` process using approximately 14,098 MiB on GPU3. It was not terminated. This limits the claim of a fully exclusive post-run resource state, but does not change the sealed reader outputs.
