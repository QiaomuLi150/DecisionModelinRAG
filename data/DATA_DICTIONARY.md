# Released tabular-data dictionary

This file inventories every released CSV and JSONL/JSONL.GZ path, its row count, and the columns/keys present. It is generated from the released headers/records; it does not infer semantic meanings beyond conventional field suffixes.

All query/document text columns were omitted from public projections. `valid` is a strict valid Top-5 selection indicator; failed requests contribute zero to all-request quality metrics. IDs retain canonical values and order where supplied, except path-like/private-looking E2-N IDs replaced by deterministic SHA-256 pseudonyms (see SANITIZATION_LOG.md). Ranking score ties follow the recorded stable original ordering; where an adapter does not expose a tied score order, do not assume a cross-model tie rule.

## `MISSING_SOURCE_FILES.csv`

Rows: 0.

| Field | Unit/type |
|---|---|
| `source_relative_path` | identifier / count / categorical / source-defined |
| `reason` | identifier / count / categorical / source-defined |

## `SANITIZATION_LOG.csv`

Rows: 5.

| Field | Unit/type |
|---|---|
| `release_relative_path` | identifier / count / categorical / source-defined |
| `removed_field_names` | identifier / count / categorical / source-defined |

## `archive/cqadupstack_contended/contended_passb_perrequest.csv`

Rows: 16240.

| Field | Unit/type |
|---|---|
| `pass` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `forum` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `anchor_stage` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `latency_ms` | milliseconds |
| `peak_memory_allocated_mib` | MiB |
| `n_calls` | identifier / count / categorical / source-defined |
| `physical_forward_count` | identifier / count / categorical / source-defined |
| `policy_order_index` | identifier / count / categorical / source-defined |
| `input_sha256` | SHA-256 or digest text |
| `source_slate_sha256` | SHA-256 or digest text |
| `model_revision` | identifier / count / categorical / source-defined |

## `archive/jina_diagnostic/fresh_runs_perquery.csv`

Rows: 1384.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `run1_valid` | identifier / count / categorical / source-defined |
| `run2_valid` | identifier / count / categorical / source-defined |
| `list_equal` | identifier / count / categorical / source-defined |
| `set_equal` | identifier / count / categorical / source-defined |
| `ndcg5_run1` | unit interval unless field name says percent |
| `ndcg5_run2` | unit interval unless field name says percent |

## `archive/jina_diagnostic/frozen_vs_corrected_perquery.csv`

Rows: 1384.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `valid_frozen` | identifier / count / categorical / source-defined |
| `valid_corrected` | identifier / count / categorical / source-defined |
| `ndcg5_frozen` | unit interval unless field name says percent |
| `ndcg5_corrected` | unit interval unless field name says percent |
| `list_equal` | identifier / count / categorical / source-defined |
| `set_equal` | identifier / count / categorical / source-defined |
| `difference_cause` | identifier / count / categorical / source-defined |

## `archive/transfer_wrong_denominator/wrong_denominator_ratios.csv`

Rows: 8.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `basis` | identifier / count / categorical / source-defined |
| `n_requests` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate_strict` | unit interval unless field name says percent |
| `p50_ms` | milliseconds |
| `p95_ms` | milliseconds |
| `anchor_n` | identifier / count / categorical / source-defined |
| `anchor_p95_ms` | milliseconds |
| `p95_ratio` | unitless ratio |
| `ratio_lo95` | unitless ratio |
| `ratio_hi95` | unitless ratio |
| `prob_ratio_le_2` | unitless ratio |
| `alloc_p95_mib` | MiB |
| `R_verdict` | identifier / count / categorical / source-defined |
| `S_verdict` | identifier / count / categorical / source-defined |

## `data/SOURCE_CROSSWALK.csv`

Rows: 73.

| Field | Unit/type |
|---|---|
| `source_relative_path` | identifier / count / categorical / source-defined |
| `release_relative_path` | identifier / count / categorical / source-defined |
| `source_sha256` | SHA-256 or digest text |
| `release_sha256` | SHA-256 or digest text |
| `fields_removed_or_transformation` | identifier / count / categorical / source-defined |

## `data/candidates/bright_top20_ids.jsonl.gz`

Rows: 1384.

| Field | Unit/type |
|---|---|
| `candidate_ids` | identifier / count / categorical / source-defined |
| `dataset` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `input_sha256` | SHA-256 or digest text |
| `query_id` | identifier / count / categorical / source-defined |
| `query_sha256` | SHA-256 or digest text |
| `slate_sha256` | SHA-256 or digest text |

## `data/candidates/cqadupstack_top20_ids.jsonl.gz`

Rows: 13145.

| Field | Unit/type |
|---|---|
| `candidate_ids` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `dataset` | identifier / count / categorical / source-defined |
| `forum` | identifier / count / categorical / source-defined |
| `input_sha256` | SHA-256 or digest text |
| `query_id` | identifier / count / categorical / source-defined |
| `query_sha256` | SHA-256 or digest text |
| `slate_sha256` | SHA-256 or digest text |

## `data/latency/bright_full_population_perrequest.csv`

Rows: 55360.

| Field | Unit/type |
|---|---|
| `session` | identifier / count / categorical / source-defined |
| `gpu_slot` | identifier / count / categorical / source-defined |
| `cohort` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `attempt` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `latency_ms` | milliseconds |
| `latency_with_fallback_ms` | milliseconds |
| `alloc_peak_mib` | MiB |

## `data/latency/bright_serving_perrequest.csv`

Rows: 12960.

| Field | Unit/type |
|---|---|
| `session` | identifier / count / categorical / source-defined |
| `gpu_slot` | identifier / count / categorical / source-defined |
| `cohort` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `attempt` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `latency_ms` | milliseconds |
| `alloc_peak_mib` | MiB |

## `data/latency/cqadupstack_clean_perrequest.csv`

Rows: 16240.

| Field | Unit/type |
|---|---|
| `pass` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `forum` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `anchor_stage` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `latency_ms` | milliseconds |
| `peak_memory_allocated_mib` | MiB |
| `n_calls` | identifier / count / categorical / source-defined |
| `physical_forward_count` | identifier / count / categorical / source-defined |
| `policy_order_index` | identifier / count / categorical / source-defined |
| `input_sha256` | SHA-256 or digest text |
| `source_slate_sha256` | SHA-256 or digest text |
| `model_revision` | identifier / count / categorical / source-defined |

## `data/scores/bright_metrics.jsonl.gz`

Rows: 6920.

| Field | Unit/type |
|---|---|
| `domain` | identifier / count / categorical / source-defined |
| `duration_ms` | unitless ratio |
| `gold_count` | identifier / count / categorical / source-defined |
| `gold_in_selected_top5` | identifier / count / categorical / source-defined |
| `gold_in_top20` | identifier / count / categorical / source-defined |
| `model` | identifier / count / categorical / source-defined |
| `operational_ndcg5` | unitless ratio |
| `peak_memory_allocated_mib` | MiB |
| `query_id` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `valid_ndcg5` | unit interval unless field name says percent |

## `data/scores/bright_perquery.csv`

Rows: 31832.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `model` | identifier / count / categorical / source-defined |
| `mode` | identifier / count / categorical / source-defined |
| `budget_L` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `ndcg5` | unit interval unless field name says percent |
| `ndcg5_bm25_fallback` | unit interval unless field name says percent |
| `n_calls` | identifier / count / categorical / source-defined |

## `data/scores/cqadupstack_perquery.csv`

Rows: 65625.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `forum` | identifier / count / categorical / source-defined |
| `length_tercile` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `gold_count` | identifier / count / categorical / source-defined |
| `top20_has_gold` | identifier / count / categorical / source-defined |
| `ndcg5` | unit interval unless field name says percent |
| `conditional_ndcg5` | unit interval unless field name says percent |
| `selected_ids_sha256` | SHA-256 or digest text |
| `failure_class` | identifier / count / categorical / source-defined |

## `data/scores/e2_n_top5_ids.jsonl.gz`

Rows: 59738.

| Field | Unit/type |
|---|---|
| `candidate_ids` | identifier / count / categorical / source-defined |
| `cohort` | identifier / count / categorical / source-defined |
| `dataset` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `query_id` | identifier / count / categorical / source-defined |
| `score_source` | identifier / count / categorical / source-defined |
| `selected_ids` | identifier / count / categorical / source-defined |

## `data/scores/transfer_perquery.csv`

Rows: 42672.

| Field | Unit/type |
|---|---|
| `dataset` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `L` | identifier / count / categorical / source-defined |
| `model` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `builder_failure` | identifier / count / categorical / source-defined |
| `model_output_failure` | identifier / count / categorical / source-defined |
| `labeled` | identifier / count / categorical / source-defined |
| `gold_count` | identifier / count / categorical / source-defined |
| `reachable` | identifier / count / categorical / source-defined |
| `ndcg5` | unit interval unless field name says percent |
| `fs5` | unit interval unless field name says percent |
| `serialized_tokens` | token count |
| `input_sha256` | SHA-256 or digest text |
| `control_source` | identifier / count / categorical / source-defined |

## `paper/local_assets/model_inventory.csv`

Rows: 7.

| Field | Unit/type |
|---|---|
| `model_id` | identifier / count / categorical / source-defined |
| `deployment` | identifier / count / categorical / source-defined |
| `revision` | identifier / count / categorical / source-defined |
| `input_contract` | identifier / count / categorical / source-defined |
| `calls_per_query` | identifier / count / categorical / source-defined |
| `gpu_topology` | identifier / count / categorical / source-defined |

## `paper/local_assets/policy_quality.csv`

Rows: 36.

| Field | Unit/type |
|---|---|
| `policy_id` | identifier / count / categorical / source-defined |
| `dataset` | identifier / count / categorical / source-defined |
| `n_test` | identifier / count / categorical / source-defined |
| `gold_present_n` | identifier / count / categorical / source-defined |
| `cssr_denominator` | identifier / count / categorical / source-defined |
| `cssr_numerator` | identifier / count / categorical / source-defined |
| `cssr_gold_present` | identifier / count / categorical / source-defined |
| `cssr_unconditional_numerator` | identifier / count / categorical / source-defined |
| `cssr_unconditional` | identifier / count / categorical / source-defined |
| `support_recall` | identifier / count / categorical / source-defined |
| `answer_em` | identifier / count / categorical / source-defined |
| `answer_f1` | identifier / count / categorical / source-defined |
| `reader_input_tokens_mean` | token count |
| `truncated_support_n` | identifier / count / categorical / source-defined |
| `truncated_support_rate` | unit interval unless field name says percent |

## `paper/local_assets/serving_cost.csv`

Rows: 12.

| Field | Unit/type |
|---|---|
| `policy_id` | identifier / count / categorical / source-defined |
| `n_requests` | identifier / count / categorical / source-defined |
| `request_p50_ms` | milliseconds |
| `request_p95_ms` | milliseconds |
| `selector_wall_seconds_per_1k_proxy` | identifier / count / categorical / source-defined |
| `gpu_seconds_per_1k` | identifier / count / categorical / source-defined |
| `gpu_peak_vram_gib_max` | GiB |
| `reader_input_tokens_mean` | token count |
| `paid_usd_per_1k` | identifier / count / categorical / source-defined |
| `local_compute_usd_per_1k` | identifier / count / categorical / source-defined |
| `scenario_total_usd_per_1k` | identifier / count / categorical / source-defined |
| `cost_status` | identifier / count / categorical / source-defined |

## `paper/regenerated/bright_e9_model_summary.csv`

Rows: 5.

| Field | Unit/type |
|---|---|
| `model` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `mean_operational_ndcg5` | unitless ratio |
| `source` | identifier / count / categorical / source-defined |

## `paper/regenerated/bright_summary.csv`

Rows: 23.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `budget_L` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate` | unit interval unless field name says percent |
| `mean_ndcg5_all` | unit interval unless field name says percent |
| `mean_ndcg5_valid` | unit interval unless field name says percent |
| `source` | identifier / count / categorical / source-defined |

## `paper/regenerated/cqadupstack_forum_contrasts.csv`

Rows: 4.

| Field | Unit/type |
|---|---|
| `contrast` | identifier / count / categorical / source-defined |
| `left_policy` | identifier / count / categorical / source-defined |
| `right_policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `delta_all_query_weighted` | identifier / count / categorical / source-defined |
| `forum_bootstrap_mean` | identifier / count / categorical / source-defined |
| `forum_bootstrap_lo95` | identifier / count / categorical / source-defined |
| `forum_bootstrap_hi95` | identifier / count / categorical / source-defined |
| `bootstrap_draws` | identifier / count / categorical / source-defined |
| `bootstrap_seed` | identifier / count / categorical / source-defined |
| `bootstrap_unit` | identifier / count / categorical / source-defined |

## `paper/regenerated/cqadupstack_summary.csv`

Rows: 5.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate` | unit interval unless field name says percent |
| `mean_ndcg5_all` | unit interval unless field name says percent |
| `source` | identifier / count / categorical / source-defined |

## `paper/regenerated/frontier_points.csv`

Rows: 19.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `model` | identifier / count / categorical / source-defined |
| `mode` | identifier / count / categorical / source-defined |
| `budget_L` | identifier / count / categorical / source-defined |
| `quality_valid_pct` | percent |
| `mean_ndcg5_all` | unit interval unless field name says percent |
| `delta_vs_jina_full` | identifier / count / categorical / source-defined |
| `delta_lo95` | identifier / count / categorical / source-defined |
| `delta_hi95` | identifier / count / categorical / source-defined |
| `p95_ms_pooled` | identifier / count / categorical / source-defined |
| `p95_ratio` | unitless ratio |
| `ratio_lo95` | unitless ratio |
| `ratio_hi95` | unitless ratio |
| `R` | identifier / count / categorical / source-defined |
| `Q` | identifier / count / categorical / source-defined |
| `S` | identifier / count / categorical / source-defined |
| `pareto_nondominated` | identifier / count / categorical / source-defined |
| `p_nondominated` | identifier / count / categorical / source-defined |
| `timing_population` | identifier / count / categorical / source-defined |

## `paper/regenerated/order_multistage_results.csv`

Rows: 38752.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `metric` | identifier / count / categorical / source-defined |
| `n_calls` | identifier / count / categorical / source-defined |
| `elapsed_ms` | milliseconds |
| `batch_elapsed_ms` | milliseconds |
| `batch_size` | identifier / count / categorical / source-defined |
| `timing_valid` | identifier / count / categorical / source-defined |

## `paper/regenerated/transfer_paired_contrasts.csv`

Rows: 8.

| Field | Unit/type |
|---|---|
| `dataset` | identifier / count / categorical / source-defined |
| `L` | identifier / count / categorical / source-defined |
| `contrast` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `delta` | identifier / count / categorical / source-defined |
| `lo95` | identifier / count / categorical / source-defined |
| `hi95` | identifier / count / categorical / source-defined |
| `resampling_unit` | identifier / count / categorical / source-defined |
| `draws` | identifier / count / categorical / source-defined |
| `seed` | identifier / count / categorical / source-defined |

## `paper/regenerated/transfer_summary.csv`

Rows: 32.

| Field | Unit/type |
|---|---|
| `dataset` | identifier / count / categorical / source-defined |
| `L` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `labeled_n` | identifier / count / categorical / source-defined |
| `mean_ndcg5` | unit interval unless field name says percent |
| `mean_fs5` | unit interval unless field name says percent |
| `source` | identifier / count / categorical / source-defined |

## `protocols/cqadupstack_latency_cohort_ids.jsonl`

Rows: 2000.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `rank` | identifier / count / categorical / source-defined |

## `raw_traces/TRACE_MANIFEST.csv`

Rows: 29.

| Field | Unit/type |
|---|---|
| `source_relative_path` | identifier / count / categorical / source-defined |
| `source_sha256` | SHA-256 or digest text |
| `source_record_count` | identifier / count / categorical / source-defined |
| `release_relative_path` | identifier / count / categorical / source-defined |
| `release_sha256` | SHA-256 or digest text |
| `release_record_count` | identifier / count / categorical / source-defined |
| `transformation` | identifier / count / categorical / source-defined |

## `raw_traces/bright_native_scores.jsonl.gz`

Rows: 5536.

| Field | Unit/type |
|---|---|
| `candidate_ids` | identifier / count / categorical / source-defined |
| `candidate_ids_sha256` | SHA-256 or digest text |
| `domain` | identifier / count / categorical / source-defined |
| `duration_ms` | unitless ratio |
| `input_manifest_sha256` | SHA-256 or digest text |
| `input_tokens` | token count |
| `model` | identifier / count / categorical / source-defined |
| `native_input_sha256` | SHA-256 or digest text |
| `peak_memory_allocated_mib` | MiB |
| `physical_forward_count` | identifier / count / categorical / source-defined |
| `query_id` | identifier / count / categorical / source-defined |
| `readout` | identifier / count / categorical / source-defined |
| `scores` | identifier / count / categorical / source-defined |
| `selected_ids` | identifier / count / categorical / source-defined |
| `slate_sha256` | SHA-256 or digest text |
| `status` | identifier / count / categorical / source-defined |

## `raw_traces/cqadupstack_native_scores.jsonl.gz`

Rows: 54852.

| Field | Unit/type |
|---|---|
| `candidate_ids` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `dataset` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `forum` | identifier / count / categorical / source-defined |
| `input_sha256` | SHA-256 or digest text |
| `latency_ms` | milliseconds |
| `length_tercile` | identifier / count / categorical / source-defined |
| `model_revision` | identifier / count / categorical / source-defined |
| `n_calls` | identifier / count / categorical / source-defined |
| `peak_memory_allocated_mib` | MiB |
| `physical_forward_count` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `query_id` | identifier / count / categorical / source-defined |
| `scores` | identifier / count / categorical / source-defined |
| `selected_ids` | identifier / count / categorical / source-defined |
| `source_slate_sha256` | SHA-256 or digest text |
| `status` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |

## `raw_traces/e2_n_native_scores.jsonl.gz`

Rows: 55471.

| Field | Unit/type |
|---|---|
| `actual_model_input_sha256` | SHA-256 or digest text |
| `attention_mode` | identifier / count / categorical / source-defined |
| `cache_provenance` | identifier / count / categorical / source-defined |
| `cache_reuse` | identifier / count / categorical / source-defined |
| `candidate_ids` | identifier / count / categorical / source-defined |
| `cohort` | identifier / count / categorical / source-defined |
| `dataset` | identifier / count / categorical / source-defined |
| `duration_ms` | unitless ratio |
| `input_forward_batches` | identifier / count / categorical / source-defined |
| `input_tokens` | token count |
| `model_id` | identifier / count / categorical / source-defined |
| `physical_forward_calls` | identifier / count / categorical / source-defined |
| `query_id` | identifier / count / categorical / source-defined |
| `raw_response_sha256` | SHA-256 or digest text |
| `real_inference` | identifier / count / categorical / source-defined |
| `request_hash` | SHA-256 or digest text |
| `revision` | identifier / count / categorical / source-defined |
| `score_contract` | identifier / count / categorical / source-defined |
| `score_direction` | identifier / count / categorical / source-defined |
| `scores` | identifier / count / categorical / source-defined |
| `seed` | identifier / count / categorical / source-defined |
| `stage` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `temperature` | identifier / count / categorical / source-defined |

## `results/01_model_screen/e2_n_limited_screen/model_dataset_summary.csv`

Rows: 84.

| Field | Unit/type |
|---|---|
| `cohort` | identifier / count / categorical / source-defined |
| `dataset` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `n_total` | identifier / count / categorical / source-defined |
| `n_eligible` | GiB |
| `n_conditional` | identifier / count / categorical / source-defined |
| `n_any_gold` | identifier / count / categorical / source-defined |
| `success_unconditional` | identifier / count / categorical / source-defined |
| `success_conditional` | identifier / count / categorical / source-defined |
| `cssr_unconditional` | identifier / count / categorical / source-defined |
| `cssr_conditional` | identifier / count / categorical / source-defined |
| `any_gold_coverage` | identifier / count / categorical / source-defined |
| `interpretation` | identifier / count / categorical / source-defined |

## `results/01_model_screen/e2_n_limited_screen/paired_statistics.csv`

Rows: 104.

| Field | Unit/type |
|---|---|
| `cohort` | identifier / count / categorical / source-defined |
| `population` | identifier / count / categorical / source-defined |
| `comparison` | identifier / count / categorical / source-defined |
| `left_policy` | identifier / count / categorical / source-defined |
| `right_policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `difference` | identifier / count / categorical / source-defined |
| `ci95_low` | identifier / count / categorical / source-defined |
| `ci95_high` | identifier / count / categorical / source-defined |
| `paired_wins` | identifier / count / categorical / source-defined |
| `paired_losses` | identifier / count / categorical / source-defined |
| `paired_ties` | identifier / count / categorical / source-defined |
| `bootstrap_reps` | identifier / count / categorical / source-defined |
| `bootstrap_seed` | identifier / count / categorical / source-defined |
| `interpretation` | identifier / count / categorical / source-defined |

## `results/02_bright_default_and_budget/bright_contrasts.csv`

Rows: 7.

| Field | Unit/type |
|---|---|
| `contrast` | identifier / count / categorical / source-defined |
| `family` | identifier / count / categorical / source-defined |
| `delta_query_weighted` | identifier / count / categorical / source-defined |
| `lo95` | identifier / count / categorical / source-defined |
| `hi95` | identifier / count / categorical / source-defined |
| `lo90` | identifier / count / categorical / source-defined |
| `hi90` | identifier / count / categorical / source-defined |
| `p_signflip` | identifier / count / categorical / source-defined |
| `p_holm` | identifier / count / categorical / source-defined |
| `domains_positive` | identifier / count / categorical / source-defined |

## `results/02_bright_default_and_budget/bright_determinism.csv`

Rows: 1.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `n_rechecked` | identifier / count / categorical / source-defined |
| `n_mismatch` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |

## `results/02_bright_default_and_budget/bright_summary.csv`

Rows: 23.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `model` | identifier / count / categorical / source-defined |
| `mode` | identifier / count / categorical / source-defined |
| `budget_L` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_pct` | percent |
| `mean_ndcg5_all` | unit interval unless field name says percent |
| `mean_ndcg5_valid` | unit interval unless field name says percent |
| `mean_ndcg5_bm25_fallback` | unit interval unless field name says percent |

## `results/02_bright_default_and_budget/e9_bright_comparison_table.csv`

Rows: 9.

| Field | Unit/type |
|---|---|
| `comparison` | identifier / count / categorical / source-defined |
| `population` | identifier / count / categorical / source-defined |
| `model_a` | identifier / count / categorical / source-defined |
| `model_b` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `domain_count` | identifier / count / categorical / source-defined |
| `a_macro_ndcg5` | unit interval unless field name says percent |
| `b_macro_ndcg5` | unit interval unless field name says percent |
| `delta_macro_a_minus_b` | identifier / count / categorical / source-defined |
| `ci95_domain_resample` | identifier / count / categorical / source-defined |
| `ci95_fixed_domain_query_resample` | identifier / count / categorical / source-defined |
| `ci95_hierarchical_domain_query_resample` | identifier / count / categorical / source-defined |
| `delta_query_weighted` | identifier / count / categorical / source-defined |
| `ci95_query_resample` | identifier / count / categorical / source-defined |
| `paired_wins_a` | identifier / count / categorical / source-defined |
| `paired_losses_a` | identifier / count / categorical / source-defined |
| `paired_ties` | identifier / count / categorical / source-defined |
| `selection_note` | identifier / count / categorical / source-defined |

## `results/03_order_and_multistage/order_perquery.csv`

Rows: 29064.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `selected_ids_hash` | SHA-256 or digest text |

## `results/03_order_and_multistage/order_summary.csv`

Rows: 3.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `mean_over_orders` | identifier / count / categorical / source-defined |
| `sd_over_orders` | identifier / count / categorical / source-defined |
| `min` | identifier / count / categorical / source-defined |
| `max` | identifier / count / categorical / source-defined |
| `top5_set_change_rate_vs_original` | unit interval unless field name says percent |

## `results/03_order_and_multistage/two_stage_jina_matched_text_domain_summary.csv`

Rows: 168.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `mean_ndcg5` | unit interval unless field name says percent |

## `results/03_order_and_multistage/two_stage_jina_matched_text_perquery.csv`

Rows: 19376.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `metric` | identifier / count / categorical / source-defined |
| `survivor_set_matches_pplx` | identifier / count / categorical / source-defined |

## `results/03_order_and_multistage/two_stage_order_stability.csv`

Rows: 1384.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `valid_order_count` | identifier / count / categorical / source-defined |
| `unique_selected_sets` | identifier / count / categorical / source-defined |
| `mean_ndcg5` | unit interval unless field name says percent |
| `std_ndcg5` | unit interval unless field name says percent |

## `results/03_order_and_multistage/two_stage_ordered_perquery.csv`

Rows: 8304.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `ndcg5` | unit interval unless field name says percent |
| `original_ndcg5` | unit interval unless field name says percent |
| `delta_vs_original` | identifier / count / categorical / source-defined |
| `selected_ids` | identifier / count / categorical / source-defined |
| `original_selected_ids` | identifier / count / categorical / source-defined |
| `selection_set_changed` | identifier / count / categorical / source-defined |
| `selection_order_changed` | identifier / count / categorical / source-defined |
| `n_calls` | identifier / count / categorical / source-defined |
| `total_ms` | milliseconds |

## `results/03_order_and_multistage/two_stage_specialist_contrasts.csv`

Rows: 4.

| Field | Unit/type |
|---|---|
| `contrast` | identifier / count / categorical / source-defined |
| `left` | identifier / count / categorical / source-defined |
| `right` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `query_weighted_effect` | identifier / count / categorical / source-defined |
| `domain_weighted_effect` | identifier / count / categorical / source-defined |
| `ci95_low` | identifier / count / categorical / source-defined |
| `ci95_high` | identifier / count / categorical / source-defined |
| `p_two_sided_sign_flip` | identifier / count / categorical / source-defined |
| `bootstrap_seed` | identifier / count / categorical / source-defined |
| `bootstrap_draws` | identifier / count / categorical / source-defined |
| `domain_n` | identifier / count / categorical / source-defined |
| `holm_adjusted_p_two_sided` | identifier / count / categorical / source-defined |

## `results/03_order_and_multistage/two_stage_specialist_perquery.csv`

Rows: 38752.

| Field | Unit/type |
|---|---|
| `canonical_id` | identifier / count / categorical / source-defined |
| `domain` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `order_id` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `metric` | identifier / count / categorical / source-defined |
| `n_calls` | identifier / count / categorical / source-defined |
| `elapsed_ms` | milliseconds |
| `batch_elapsed_ms` | milliseconds |
| `batch_size` | identifier / count / categorical / source-defined |
| `timing_valid` | identifier / count / categorical / source-defined |

## `results/05_latency_and_memory/bright_full_population_pooled_ratios.csv`

Rows: 60.

| Field | Unit/type |
|---|---|
| `session` | identifier / count / categorical / source-defined |
| `cohort` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `basis` | identifier / count / categorical / source-defined |
| `p50_ms` | milliseconds |
| `p95_ms` | milliseconds |
| `p95_ratio_point` | unitless ratio |
| `bootstrap_lo95` | identifier / count / categorical / source-defined |
| `bootstrap_hi95` | identifier / count / categorical / source-defined |
| `prob_ratio_le_2` | unitless ratio |
| `alloc_p95_mib` | MiB |
| `verdict` | identifier / count / categorical / source-defined |

## `results/05_latency_and_memory/bright_serving_ratios.csv`

Rows: 24.

| Field | Unit/type |
|---|---|
| `session` | identifier / count / categorical / source-defined |
| `cohort` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `basis` | identifier / count / categorical / source-defined |
| `p50_ms` | milliseconds |
| `p95_ms` | milliseconds |
| `p95_ratio_point` | unitless ratio |
| `bootstrap_lo95` | identifier / count / categorical / source-defined |
| `bootstrap_hi95` | identifier / count / categorical / source-defined |
| `prob_ratio_le_2` | unitless ratio |
| `alloc_p95_mib` | MiB |
| `verdict` | identifier / count / categorical / source-defined |

## `results/05_latency_and_memory/bright_serving_ratios_pooled.csv`

Rows: 10.

| Field | Unit/type |
|---|---|
| `session` | identifier / count / categorical / source-defined |
| `cohort` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `basis` | identifier / count / categorical / source-defined |
| `p50_ms` | milliseconds |
| `p95_ms` | milliseconds |
| `p95_ratio_point` | unitless ratio |
| `bootstrap_lo95` | identifier / count / categorical / source-defined |
| `bootstrap_hi95` | identifier / count / categorical / source-defined |
| `prob_ratio_le_2` | unitless ratio |
| `alloc_p95_mib` | MiB |
| `verdict` | identifier / count / categorical / source-defined |

## `results/05_latency_and_memory/frontier_points.csv`

Rows: 19.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `model` | identifier / count / categorical / source-defined |
| `mode` | identifier / count / categorical / source-defined |
| `budget_L` | identifier / count / categorical / source-defined |
| `quality_valid_pct` | percent |
| `mean_ndcg5_all` | unit interval unless field name says percent |
| `delta_vs_jina_full` | identifier / count / categorical / source-defined |
| `delta_lo95` | identifier / count / categorical / source-defined |
| `delta_hi95` | identifier / count / categorical / source-defined |
| `p95_ms_pooled` | identifier / count / categorical / source-defined |
| `p95_ratio` | unitless ratio |
| `ratio_lo95` | unitless ratio |
| `ratio_hi95` | unitless ratio |
| `R` | identifier / count / categorical / source-defined |
| `Q` | identifier / count / categorical / source-defined |
| `S` | identifier / count / categorical / source-defined |
| `pareto_nondominated` | identifier / count / categorical / source-defined |
| `p_nondominated` | identifier / count / categorical / source-defined |
| `timing_population` | identifier / count / categorical / source-defined |

## `results/05_latency_and_memory/release_check_frontier.csv`

Rows: 19.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `model` | identifier / count / categorical / source-defined |
| `mode` | identifier / count / categorical / source-defined |
| `budget_L` | identifier / count / categorical / source-defined |
| `quality_valid_pct` | percent |
| `mean_ndcg5_all` | unit interval unless field name says percent |
| `delta_vs_jina_full` | identifier / count / categorical / source-defined |
| `delta_lo95` | identifier / count / categorical / source-defined |
| `delta_hi95` | identifier / count / categorical / source-defined |
| `p95_ms_pooled` | identifier / count / categorical / source-defined |
| `p95_ratio` | unitless ratio |
| `ratio_lo95` | unitless ratio |
| `ratio_hi95` | unitless ratio |
| `R` | identifier / count / categorical / source-defined |
| `Q` | identifier / count / categorical / source-defined |
| `S` | identifier / count / categorical / source-defined |
| `pareto_nondominated` | identifier / count / categorical / source-defined |
| `p_nondominated` | identifier / count / categorical / source-defined |
| `all_pass` | identifier / count / categorical / source-defined |

## `results/05_latency_and_memory/two_stage_perrequest.csv`

Rows: 9360.

| Field | Unit/type |
|---|---|
| `session` | identifier / count / categorical / source-defined |
| `phase` | identifier / count / categorical / source-defined |
| `planned_cohort` | identifier / count / categorical / source-defined |
| `observed_records_per_policy` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `canonical_id` | identifier / count / categorical / source-defined |
| `attempt` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `valid` | identifier / count / categorical / source-defined |
| `error_class` | identifier / count / categorical / source-defined |
| `latency_ms` | milliseconds |
| `latency_with_fallback_ms` | milliseconds |
| `alloc_peak_mib` | MiB |
| `n_calls` | identifier / count / categorical / source-defined |
| `model_revision` | identifier / count / categorical / source-defined |

## `results/05_latency_and_memory/two_stage_ratios.csv`

Rows: 52.

| Field | Unit/type |
|---|---|
| `session` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `basis` | identifier / count / categorical / source-defined |
| `n_requests` | identifier / count / categorical / source-defined |
| `n_ids` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `p50_ms` | milliseconds |
| `p95_ms` | milliseconds |
| `p95_ratio` | unitless ratio |
| `ratio_lo95` | unitless ratio |
| `ratio_hi95` | unitless ratio |
| `prob_ratio_le_2` | unitless ratio |
| `alloc_p95_mib` | MiB |
| `failure_count` | identifier / count / categorical / source-defined |
| `error_classes` | identifier / count / categorical / source-defined |
| `verdict` | identifier / count / categorical / source-defined |
| `fallback_added_on_failures` | identifier / count / categorical / source-defined |

## `results/06_transfer_enterprise_musique/transfer_contrasts.csv`

Rows: 8.

| Field | Unit/type |
|---|---|
| `dataset` | identifier / count / categorical / source-defined |
| `L` | identifier / count / categorical / source-defined |
| `contrast` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `delta` | identifier / count / categorical / source-defined |
| `lo95` | identifier / count / categorical / source-defined |
| `hi95` | identifier / count / categorical / source-defined |
| `resampling_unit` | identifier / count / categorical / source-defined |
| `draws` | identifier / count / categorical / source-defined |
| `seed` | identifier / count / categorical / source-defined |

## `results/06_transfer_enterprise_musique/transfer_summary.csv`

Rows: 32.

| Field | Unit/type |
|---|---|
| `dataset` | identifier / count / categorical / source-defined |
| `L` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `gold_labeled_n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate` | unit interval unless field name says percent |
| `builder_failures` | identifier / count / categorical / source-defined |
| `model_output_failures` | identifier / count / categorical / source-defined |
| `ndcg5_all_labeled` | unit interval unless field name says percent |
| `fs5_all_labeled` | unit interval unless field name says percent |
| `mean_serialized_tokens` | token count |
| `max_serialized_tokens` | token count |
| `serialized_token_population` | token count |
| `resampling_unit` | identifier / count / categorical / source-defined |

## `results/07_frozen_cqadupstack/by_forum.csv`

Rows: 60.

| Field | Unit/type |
|---|---|
| `forum` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate` | unit interval unless field name says percent |
| `mean_ndcg5` | unit interval unless field name says percent |
| `top20_gold_share` | unit interval unless field name says percent |

## `results/07_frozen_cqadupstack/by_length_tercile.csv`

Rows: 15.

| Field | Unit/type |
|---|---|
| `length_tercile` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate` | unit interval unless field name says percent |
| `mean_ndcg5` | unit interval unless field name says percent |

## `results/07_frozen_cqadupstack/clean_latency_ratios.csv`

Rows: 9.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `basis` | identifier / count / categorical / source-defined |
| `n_query_ids` | identifier / count / categorical / source-defined |
| `p50_ms` | milliseconds |
| `p95_ms` | milliseconds |
| `jina_p50_ms` | milliseconds |
| `jina_p95_ms` | milliseconds |
| `p95_ratio` | unitless ratio |
| `ratio_lo95` | unitless ratio |
| `ratio_hi95` | unitless ratio |
| `prob_ratio_le_2` | unitless ratio |
| `bootstrap_draws` | identifier / count / categorical / source-defined |
| `bootstrap_seed` | identifier / count / categorical / source-defined |
| `bootstrap_unit` | identifier / count / categorical / source-defined |
| `latency_basis` | identifier / count / categorical / source-defined |

## `results/07_frozen_cqadupstack/clean_latency_summary.csv`

Rows: 8.

| Field | Unit/type |
|---|---|
| `pass` | identifier / count / categorical / source-defined |
| `policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate` | unit interval unless field name says percent |
| `failed_n` | identifier / count / categorical / source-defined |
| `error_classes` | identifier / count / categorical / source-defined |
| `p50_ms` | milliseconds |
| `p95_ms` | milliseconds |
| `p99_ms` | milliseconds |
| `mean_ms` | milliseconds |
| `alloc_p95_mib` | MiB |
| `physical_forward_total` | identifier / count / categorical / source-defined |

## `results/07_frozen_cqadupstack/cqadupstack_contrasts.csv`

Rows: 4.

| Field | Unit/type |
|---|---|
| `contrast` | identifier / count / categorical / source-defined |
| `left_policy` | identifier / count / categorical / source-defined |
| `right_policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `delta_all_query_weighted` | identifier / count / categorical / source-defined |
| `forum_bootstrap_mean` | identifier / count / categorical / source-defined |
| `forum_bootstrap_lo95` | identifier / count / categorical / source-defined |
| `forum_bootstrap_hi95` | identifier / count / categorical / source-defined |
| `bootstrap_draws` | identifier / count / categorical / source-defined |
| `bootstrap_seed` | identifier / count / categorical / source-defined |
| `bootstrap_unit` | identifier / count / categorical / source-defined |

## `results/07_frozen_cqadupstack/cqadupstack_summary.csv`

Rows: 5.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `n` | identifier / count / categorical / source-defined |
| `valid_n` | identifier / count / categorical / source-defined |
| `valid_rate` | unit interval unless field name says percent |
| `valid_cp_lo95` | identifier / count / categorical / source-defined |
| `valid_cp_hi95` | identifier / count / categorical / source-defined |
| `mean_ndcg5_all` | unit interval unless field name says percent |
| `mean_ndcg5_valid` | unit interval unless field name says percent |
| `forum_bootstrap_mean` | identifier / count / categorical / source-defined |
| `forum_bootstrap_lo95` | identifier / count / categorical / source-defined |
| `forum_bootstrap_hi95` | identifier / count / categorical / source-defined |
| `top20_gold_share` | unit interval unless field name says percent |
| `failures` | identifier / count / categorical / source-defined |

## `results/07_frozen_cqadupstack/judged_share.csv`

Rows: 5.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `all_n` | identifier / count / categorical / source-defined |
| `top20_judged_n` | identifier / count / categorical / source-defined |
| `top20_judged_share` | unit interval unless field name says percent |
| `all_mean_ndcg5` | unit interval unless field name says percent |
| `judged_only_mean_ndcg5` | unit interval unless field name says percent |

## `results/07_frozen_cqadupstack/p95_ratio_ci_97_5.csv`

Rows: 2.

| Field | Unit/type |
|---|---|
| `policy` | identifier / count / categorical / source-defined |
| `basis` | identifier / count / categorical / source-defined |
| `n_query_ids` | identifier / count / categorical / source-defined |
| `p95_ratio` | unitless ratio |
| `ratio_lo97_5` | unitless ratio |
| `ratio_hi97_5` | unitless ratio |
| `bootstrap_draws` | identifier / count / categorical / source-defined |
| `bootstrap_seed` | identifier / count / categorical / source-defined |
| `bootstrap_unit` | identifier / count / categorical / source-defined |
| `interval` | identifier / count / categorical / source-defined |
| `latency_basis` | identifier / count / categorical / source-defined |

## `results/08_controlled_replication/e2_e9_evidence_ledger.csv`

Rows: 46.

| Field | Unit/type |
|---|---|
| `phase` | identifier / count / categorical / source-defined |
| `claim_id` | identifier / count / categorical / source-defined |
| `claim` | identifier / count / categorical / source-defined |
| `population` | identifier / count / categorical / source-defined |
| `metric` | identifier / count / categorical / source-defined |
| `result` | identifier / count / categorical / source-defined |
| `ci` | identifier / count / categorical / source-defined |
| `status` | identifier / count / categorical / source-defined |
| `source_artifact` | identifier / count / categorical / source-defined |
| `limitation` | identifier / count / categorical / source-defined |
| `allowed_wording` | identifier / count / categorical / source-defined |
