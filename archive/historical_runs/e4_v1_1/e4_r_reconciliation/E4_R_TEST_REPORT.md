# E4-R Test Report

## Focused CPU tests
Return code: **0**
```text
...s                                                                     [100%]


```

## Full repository regression
Return code: **0**
```text
........................................................................ [ 50%]
................................s......................................  [100%]
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/test_e2_m0_frozen_mapping.py::FrozenMappingAuditTests::test_all_core_gate_statuses_pass
PASSED tests/test_e2_m0_frozen_mapping.py::FrozenMappingAuditTests::test_full_query_rows_and_unique_namespaces
PASSED tests/test_e2_m0_frozen_mapping.py::FrozenMappingAuditTests::test_output_manifest_and_owner_gate_are_machine_readable
PASSED tests/test_e2_m0_frozen_mapping.py::FrozenMappingAuditTests::test_qst_0413_retained_but_not_eligible
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_01_gold_collision_same_group_is_quota_loss
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_02_gold_in_different_groups_can_be_retained
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_03_absent_gold_is_retrieval_miss
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_04_five_gold_one_per_group_is_feasible
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_05_more_than_budget_is_excluded
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_06_empty_gold_is_explicitly_excluded
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_07_ambiguous_gold_is_explicitly_excluded
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_08_path_qualified_ids_do_not_collapse
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_09_duplicate_slate_and_selected_ids_block
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_10_out_of_slate_selected_id_blocks
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_11_duplicate_group_membership_blocks
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_12_missing_group_record_cannot_be_inferred
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_13_same_score_top5_has_stable_tie_break
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_14_namespaced_query_and_document_ids_remain_distinct
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_15_aggregate_output_has_separate_denominators
PASSED tests/test_e2_m0_structural_oracle.py::StructuralOraclePureTests::test_16_no_gpu_or_network_imports_in_audit_script
PASSED tests/test_e2_n_n0_contracts.py::test_registry_is_cpu_only_and_versioned
PASSED tests/test_e2_n_n0_contracts.py::test_slate_rejects_empty_questions_and_duplicate_ids
PASSED tests/test_e2_n_n0_contracts.py::test_slate_requires_exactly_twenty_unique_nonempty_ids
PASSED tests/test_e2_n_n0_contracts.py::test_scores_must_align_exactly_and_be_finite
PASSED tests/test_e2_n_n0_contracts.py::test_score_direction_and_stable_top5
PASSED tests/test_e2_n_n0_contracts.py::test_lower_is_more_relevant_is_explicit
PASSED tests/test_e2_n_n0_contracts.py::test_trace_template_requires_physical_fields_without_fabricating_them
PASSED tests/test_e2_sp_r2_offline.py::test_r2_manifest_and_zero_paid_calls
PASSED tests/test_e2_sp_r2_offline.py::test_isolated_r3_has_five_models_and_800_successes
PASSED tests/test_e2_sp_r2_offline.py::test_local_adjudication_is_blinded_and_complete
PASSED tests/test_e2_sp_r2_offline.py::test_cal_configuration_is_locked_and_all_models_passed
PASSED tests/test_e2_sp_r2_offline.py::test_historical_shared_artifacts_still_exist
PASSED tests/test_e2_sp_r_offline.py::test_r0_frozen_population_and_selection_reader_counts
PASSED tests/test_e2_sp_r_offline.py::test_r4_package_is_deterministic_complete_and_unadjudicated
PASSED tests/test_e2_sp_r_offline.py::test_r3_shared_execution_remains_non_isolated
PASSED tests/test_e3_a_offline.py::test_frozen_population_and_unique_ids
PASSED tests/test_e3_a_offline.py::test_qst_0413_is_retained_but_not_identity_eligible
PASSED tests/test_e3_a_offline.py::test_top5_containment_and_four_cell_partition
PASSED tests/test_e3_a_offline.py::test_historical_anchor_counts_are_reproduced
PASSED tests/test_e3_a_offline.py::test_reader_cohort_and_nested_judge_are_separate
PASSED tests/test_e3_a_offline.py::test_answer_proxy_and_oracle_do_not_claim_factual_gold
PASSED tests/test_e3_a_offline.py::test_governance_remains_pending_without_metadata
PASSED tests/test_e3_a_offline.py::test_output_gate_and_zero_paid_calls
PASSED tests/test_e3_b_offline.py::test_e3_b_owner_gate_and_provenance
PASSED tests/test_e3_b_offline.py::test_e3_b_sampling_and_panel_counts
PASSED tests/test_e3_b_offline.py::test_e3_b_native_detail_alignment_and_counts
PASSED tests/test_e3_b_offline.py::test_e3_b_offline_result_cardinalities
PASSED tests/test_e3_b_offline.py::test_e3_b_setr_label_and_cal_freeze
PASSED tests/test_e3_b_offline.py::test_e3_b_artifact_hashes
PASSED tests/test_e3_c_offline.py::test_e3_c_owner_gate_and_paid_holdout_provenance
PASSED tests/test_e3_c_offline.py::test_e3_c_native_final_cardinalities_and_forward_telemetry
PASSED tests/test_e3_c_offline.py::test_e3_c_repeated_native_attempts_are_score_stable
PASSED tests/test_e3_c_offline.py::test_e3_c_selected_top5_and_quality_denominators
PASSED tests/test_e3_c_offline.py::test_e3_c_paired_statistics_are_separated_and_pinned
PASSED tests/test_e3_c_offline.py::test_e3_c_timing_is_measured_on_isolated_gpu0
PASSED tests/test_e3_c_offline.py::test_e3_c_artifact_hashes
PASSED tests/test_e3_d_offline.py::test_governance_partition_is_reserved_disjoint
PASSED tests/test_e3_d_offline.py::test_frozen_candidate_contract_is_identical_top20
PASSED tests/test_e3_d_offline.py::test_selector_rows_have_aligned_scores_or_explicit_failure
PASSED tests/test_e3_d_offline.py::test_exploratory_allowlist_has_no_reserved_assignment
PASSED tests/test_e3_d_offline.py::test_reader_records_are_gold_free_and_hashable
PASSED tests/test_e3_d_offline.py::test_required_e3d_artifacts_exist
PASSED tests/test_e3_d_r_offline.py::test_r0_exact_reader_and_selector_cardinality
PASSED tests/test_e3_d_r_offline.py::test_finance_legacy_reproduction_scorecard
PASSED tests/test_e3_d_r_offline.py::test_multihop_matched_denominator_and_four_cells
PASSED tests/test_e3_d_r_offline.py::test_holdout_metadata_only_and_no_reserved_use
PASSED tests/test_e3_d_r_offline.py::test_r_provenance_and_no_paid_work
PASSED tests/test_e3_d_r_offline.py::test_normalizer_fixture_artifact_passes
PASSED tests/test_e3_f0_offline.py::test_f0_protocol_is_offline_and_protected_paths_denied
PASSED tests/test_e3_f0_offline.py::test_f0_denominators_and_nonpooling_dictionary
PASSED tests/test_e3_f0_offline.py::test_f0_evidence_tables_keep_cssr_u_c_separate
PASSED tests/test_e3_f0_offline.py::test_f0_finance_incomplete_qrels_guard_and_proxy_status
PASSED tests/test_e3_f0_offline.py::test_f0_external_denominators_and_selection_counts
PASSED tests/test_e3_f0_offline.py::test_f0_holdout_timestamp_ambiguity_is_not_promoted
PASSED tests/test_e3_f0_offline.py::test_f0_power_has_all_effects_targets_and_edge_policy
PASSED tests/test_e3_f0_offline.py::test_f0_options_are_proposals_not_approval
PASSED tests/test_e3_f0_offline.py::test_f0_source_hash_index_and_new_artifact_scope
PASSED tests/test_e3_f0_offline.py::test_f0_reviewer_packet_is_critical_and_has_questions
PASSED tests/test_e3_f0_offline.py::test_f0_no_verified_accuracy_or_fabricated_reviewers
PASSED tests/test_e4_0_e4_1.py::test_e4_foundation_gate_and_cohort_freeze
PASSED tests/test_e4_0_e4_1.py::test_e4_six_arms_and_unique_ids
PASSED tests/test_e4_0_e4_1.py::test_e4_no_gold_fields_in_reader_inputs
PASSED tests/test_e4_0_e4_1.py::test_e4_ab_archived_top5_alignment
PASSED tests/test_e4_0_e4_1.py::test_e4_support_removal_and_placebo_change_only_intended_context
PASSED tests/test_e4_0_e4_1.py::test_e4_confirmation_and_future_phases_locked
PASSED tests/test_e4_0_e4_1.py::test_e4_native_call_and_reuse_accounting
PASSED tests/test_e4_0_e4_1.py::test_e4_paired_statistics_are_frozen_and_separate
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_frozen_cohorts_and_question_only_facets
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_native_rows_have_complete_policy_alignment
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_common_candidate_views_and_facets_are_identical_across_policies
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_liquid_mechanism_uses_one_revision_and_explicit_work
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_quality_artifact_has_typed_records_and_separate_cohorts
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_work_reports_phase_and_shared_gpu_limitation
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_protocol_prohibits_holdout_paid_and_future_phases
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_provenance_manifest_input_hashes_exist
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_cal_smoke_and_failure_events_are_explicit
PASSED tests/test_e4_2_facet_pilot.py::test_e4_2_owner_gate_and_no_automatic_expansion
PASSED tests/test_e4_3_operating_regimes.py::test_frozen_cohort_is_unique_and_nonoverlapping_with_e4_2
PASSED tests/test_e4_3_operating_regimes.py::test_k10_is_exact_k20_prefix_and_top5_helper_is_stable
PASSED tests/test_e4_3_operating_regimes.py::test_native_protocol_has_no_label_inputs_and_has_all_six_policies
PASSED tests/test_e4_3_operating_regimes.py::test_bootstrap_is_deterministic_and_handles_empty
PASSED tests/test_e4_r_reconciliation.py::test_e4_r_scripts_are_cpu_offline_only
PASSED tests/test_e4_r_reconciliation.py::test_e4_cohort_and_raw_population_integrity
PASSED tests/test_e4_r_reconciliation.py::test_recheck_outputs_have_explicit_denominators
PASSED tests/test_m1_data.py::test_m1_counts_and_mapping_integrity
PASSED tests/test_m1_data.py::test_m1_split_and_document_isolation
PASSED tests/test_m1_data.py::test_m1_corpus_contains_no_gold_or_answer_fields
PASSED tests/test_m1_data.py::test_m1_gold_audit_is_stratified_20_per_dataset
PASSED tests/test_m2_retrieval.py::test_m2_frozen_slates_are_complete_and_unique
PASSED tests/test_m2_retrieval.py::test_m2_candidates_resolve_to_frozen_corpus
PASSED tests/test_m2_retrieval.py::test_m2_manifest_integrity
PASSED tests/test_m3_selection.py::test_m3_all_models_and_policies_are_real_and_complete
PASSED tests/test_m3_selection.py::test_m3_real_selector_traces_and_corrected_groupwise_limits
PASSED tests/test_m3_selection.py::test_m3_shared_view_hashes_match_across_models
PASSED tests/test_m4_reader.py::test_m4_reader_outputs_and_cache_contract
PASSED tests/test_m4_reader.py::test_m4_metrics_and_bootstrap_contract
PASSED tests/test_m5_serving.py::test_m5_all_policies_share_frozen_uncached_cohort
PASSED tests/test_m5_serving.py::test_m5_stage_boundaries_and_summary_integrity
PASSED tests/test_m5_serving.py::test_m5_required_analysis_artifacts_exist
PASSED tests/test_paid_gate.py::test_default_gate_denies_without_network
PASSED tests/test_paid_gate.py::test_key_does_not_override_incomplete_phase
PASSED tests/test_paid_gate.py::test_smoke_approval_cannot_authorize_bulk
PASSED tests/test_paid_gate.py::test_example_approval_is_not_real
PASSED tests/test_paid_gate.py::test_gate_budget_is_enforced_without_transport
PASSED tests/test_quality_audit.py::test_bge_model_input_uses_native_tokenizer_while_views_remain_common
PASSED tests/test_quality_audit.py::test_original_non_jev_artifacts_are_hash_manifested
PASSED tests/test_quality_audit.py::test_quality_audit_stop_gate_and_interleaved_traces
PASSED tests/test_quality_audit.py::test_qwen_corrected_contract_and_audit_artifacts
PASSED tests/test_selection_policy.py::test_title_normalization_and_generic_rejection
PASSED tests/test_selection_policy.py::test_quartets_are_deterministic_and_cover_twenty
PASSED tests/test_selection_policy.py::test_rescue_one_swap_is_gold_blind_and_tie_deterministic
PASSED tests/test_selection_policy.py::test_rescue_has_no_connected_outsider
PASSED tests/test_track_b_index.py::test_track_b_bm25_recovers_ids_from_contentless_fts
PASSED tests/test_track_c_r2.py::test_r2_manifest_has_exact_nonempty_canonical_questions
PASSED tests/test_track_c_r2.py::test_r2_schedule_attaches_the_right_question_hash_to_all_tasks
PASSED tests/test_track_c_r2.py::test_missing_question_fails_closed
PASSED tests/test_track_c_r2.py::test_whitespace_question_fails_closed
PASSED tests/test_track_c_r2.py::test_wrong_query_mapping_fails_closed
PASSED tests/test_track_c_r2.py::test_wrong_dataset_association_fails_closed
PASSED tests/test_track_c_r2.py::test_question_hash_mismatch_fails_closed
PASSED tests/test_track_c_r2.py::test_repair_resolution_is_deterministic_and_gold_free
PASSED tests/test_track_c_r2.py::test_repair_output_contract_and_schedule_are_unchanged
SKIPPED [1] tests/test_e4_r_reconciliation.py:62: owner manifest is refreshed after the audit test phase


```

Tests were run with CUDA hidden and Hugging Face offline flags. A nonzero result is preserved as an environment or regression blocker; no model inference is triggered by E4-R itself.
