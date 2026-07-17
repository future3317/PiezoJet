# PiezoJet experiment registry

This is the human-readable index of every top-level cohort under `outputs/`. The machine-readable registry contains subrun markers, seeds, passes, artifact pointers, and interpretation boundaries; the JSONL index inventories every persisted file.

Negative, failed, interrupted, partial, running, and historical runs are intentionally retained. A directory's existence never implies a valid performance result.

Registered top-level cohorts: **137**.

| Cohort | Family | Execution | Result | Convention | Paper use | Runs (complete/partial) |
|---|---|---|---|---|---|---:|
| `audit` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 0 (0/0) |
| `baselines` | support_or_pretraining | completed_or_partial | support_artifact | historical_pre_v7_or_run_local | appendix_provenance_if_used | 1 (0/1) |
| `bec_transpose_cartesian_train149_pretrain_v1` | structural_pretraining | completed | support_artifact | v7_bec_transpose | appendix_provenance | 0 (0/0) |
| `bec_transpose_observable_v1` | historical_observable_lift | completed_or_retained_partial | negative_or_nonidentifiable | run_local_observable_lift | historical_appendix_only | 9 (8/1) |
| `capacity_decomposition_v1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 8 (8/0) |
| `cartesian_capacity_32` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `cartesian_encoder_smoke_pretrain` | support_or_pretraining | completed_or_partial | support_artifact | historical_pre_v7_or_run_local | appendix_provenance_if_used | 1 (1/0) |
| `cartesian_encoder_smoke_pretrain_v2` | support_or_pretraining | completed_or_partial | support_artifact | historical_pre_v7_or_run_local | appendix_provenance_if_used | 1 (1/0) |
| `cartesian_formal` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `cartesian_formal_long_noes` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (0/1) |
| `cartesian_formal_long_noes_nw0` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `checkpoint_smoke` | engineering_validation | completed_or_retained_partial | smoke_or_resource_diagnostic | historical_pre_v7_or_run_local | project_ledger_only | 1 (0/0) |
| `checkpoint_smoke2` | engineering_validation | completed_or_retained_partial | smoke_or_resource_diagnostic | historical_pre_v7_or_run_local | project_ledger_only | 1 (1/0) |
| `collective_polar_infer_smoke` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (0/1) |
| `collective_polar_smoke` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `collective_polar_smoke_3epoch` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `data_recovery_2026-07-16` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 0 (0/0) |
| `dfpt_128_bond_loss_selected_seed123` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_loss_selected_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_loss_selected_seed7` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_v2_loss_seed123` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_v2_loss_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_v2_loss_seed7` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_v3_epsr_seed123` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_v3_epsr_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_bond_v3_epsr_seed7` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_converged_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_curriculum_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_curriculum_smoke` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_energy_oracle_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_energy_strain_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_frozen_factors_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_response_active_bond_matched_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_response_active_shared_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_response_active_star_frozen_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_response_active_star_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_128_star_loss_selected_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `dfpt_convention_audit_v1` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 0 (0/0) |
| `dfpt_convention_audit_v2` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 0 (0/0) |
| `dfpt_energy_refactor_smoke` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `dfpt_pilot_128_audit` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `dfpt_pilot_128_smoke` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `dfpt_pilot_audit` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `dfpt_pilot_smoke` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `dfpt_pilot_smoke_v2` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `dfpt_recovery_spotcheck_v1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `dfpt_response_active_smoke` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `diagnostic_smoke_20260712` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `direct_u_multistream_smoke_v1` | direct_u_implementation_smoke | completed | negative_one_pass_diagnostic | v7_bec_transpose_regularized_direct_u | appendix_implementation_check_only | 1 (1/0) |
| `electromechanical_jet_fold_adjudication` | electrostatic_generator_formula_disjoint_adjudication | partial_with_completed_a1_negative_and_a0_user_interruption | development_only_negative_and_incomplete_diagnostics | v10_full_public_electrostatic_a0_a3 | appendix_post_freeze_development_diagnostic | 9 (5/4) |
| `electronic_generator_adjudication_v1` | electronic_generator_model_class_adjudication | completed_controls_and_same_id_capacity_with_retained_failures | current_head_negative_global_irrep_first_order_jet_and_literal_positive_same_id_capacity_redundant_probe_mixed_negative_control | v10_global_l3_independent_u_electronic_irreps_autodiff_delta_p | appendix_post_freeze_diagnostic | 19 (16/3) |
| `exposure_matched_direct_u_v2_conditioning` | registered_direct_u_exposure_replay | completed | registered_result_available | v7_bec_transpose_regularized_direct_u | appendix_registered_result | 25 (24/0) |
| `factor_protected_norm_match_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 5 (3/0) |
| `factor_protected_projection_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 5 (3/0) |
| `feedback4_execution_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 0 (0/0) |
| `feedback5_execution_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 5 (3/0) |
| `full_corpus_multitask_detached_lift_v1` | historical_detached_macro_lift | failed_superseded | negative_nonidentifiable_parameterization | bec_transpose_with_removed_detached_lift | historical_appendix_only | 2 (1/1) |
| `full_corpus_multitask_detached_lift_v2` | historical_detached_macro_lift | completed | negative_nonidentifiable_parameterization | bec_transpose_with_removed_detached_lift | historical_appendix_only | 7 (7/0) |
| `global_l3_first_order_training_smoke_v1` | global_l3_implementation_smoke | completed_or_retained_partial | implementation_diagnostic | v10_global_l3_independent_u_first_order | appendix_implementation_history | 1 (1/0) |
| `global_l3_first_order_training_smoke_v2_teacher_factor` | global_l3_implementation_smoke | completed_or_retained_partial | implementation_diagnostic | v10_global_l3_independent_u_first_order | appendix_implementation_history | 1 (1/0) |
| `global_l3_isolated_u_train1603_val10_v2` | global_l3_validation_forensic | completed | negative_or_partial_seed42_validation_diagnostic | v10_global_l3_independent_u_first_order | appendix_post_freeze_diagnostic | 1 (1/0) |
| `global_l3_isolated_u_training_smoke_v1` | global_l3_implementation_smoke | completed_or_retained_partial | implementation_diagnostic | v10_global_l3_independent_u_first_order | appendix_implementation_history | 1 (1/0) |
| `global_l3_joint_optimizer_adjudication_v1` | global_l3_joint_gradient_adjudication | completed_seed42 | positive_validation_only_mechanism_diagnostic | v10_global_l3_independent_u_first_order | appendix_post_freeze_diagnostic | 6 (6/0) |
| `global_l3_matched_direct_validation_v1` | global_l3_matched_direct_validation_control | completed | matched_direct_outperforms_isolated_macro_tower | v10_complete_shell_cartesian_direct | appendix_post_freeze_validation_control | 3 (3/0) |
| `global_l3_no_redundant_sum_multiseed_v1` | global_l3_validation_replication | completed | positive_physical_validation_negative_total_comparison | v10_global_l3_independent_u_first_order | appendix_post_freeze_validation_diagnostic | 2 (2/0) |
| `global_l3_train1603_val10_v1` | global_l3_validation_forensic | completed | negative_or_partial_seed42_validation_diagnostic | v10_global_l3_independent_u_first_order | appendix_post_freeze_diagnostic | 1 (1/0) |
| `gmtnet_outcar_total_consistency_v1` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 1 (1/0) |
| `hessian_bond_laplacian_oracle_v1` | offline_hessian_model_class_oracle | completed | model_class_diagnostic | v7_bec_transpose_train149 | appendix_model_class_diagnostic | 0 (0/0) |
| `inference_cache_first` | engineering_validation | completed_or_retained_partial | smoke_or_resource_diagnostic | historical_pre_v7_or_run_local | project_ledger_only | 1 (0/1) |
| `inference_cache_second` | engineering_validation | completed_or_retained_partial | smoke_or_resource_diagnostic | historical_pre_v7_or_run_local | project_ledger_only | 1 (0/1) |
| `inference_optimization` | engineering_validation | completed_or_retained_partial | smoke_or_resource_diagnostic | historical_pre_v7_or_run_local | project_ledger_only | 0 (0/0) |
| `inference_smoke` | engineering_validation | completed_or_retained_partial | smoke_or_resource_diagnostic | historical_pre_v7_or_run_local | project_ledger_only | 1 (0/1) |
| `information_gain_cohort_v1` | jarvis_data_expansion | completed | retrieval_or_completion_audit | historical_v4_v6_strict_gates | appendix_data_history | 0 (0/0) |
| `information_gain_cohort_v2_test_crystal_coverage` | jarvis_data_expansion | completed | retrieval_or_completion_audit | historical_v4_v6_strict_gates | appendix_data_history | 0 (0/0) |
| `information_gain_retrieval_v1` | jarvis_data_expansion | completed | retrieval_or_completion_audit | historical_v4_v6_strict_gates | appendix_data_history | 0 (0/0) |
| `information_gain_retrieval_v2_test_crystal_coverage` | jarvis_data_expansion | completed | retrieval_or_completion_audit | historical_v4_v6_strict_gates | appendix_data_history | 0 (0/0) |
| `jarvis_dfpt_expansion_v1` | jarvis_data_expansion | completed | retrieval_or_completion_audit | historical_v4_v6_strict_gates | appendix_data_history | 2 (0/2) |
| `jarvis_dft3d_official_audit` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `lambda_diagnostics_smoke` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (0/0) |
| `lambda_diagnostics_v1` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (0/0) |
| `lambda_diagnostics_v2` | historical_dfpt_diagnostic | completed_or_retained_partial | development_diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (0/0) |
| `m2_1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `m3` | historical_milestone | interrupted | no_generalization_result | historical_pre_v7_or_run_local | historical_appendix_only | 2 (0/1) |
| `m4` | engineering_validation | completed_or_retained_partial | smoke_or_resource_diagnostic | historical_pre_v7_or_run_local | project_ledger_only | 0 (0/0) |
| `m5` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 2 (2/0) |
| `mode_aware_smoke_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 2 (2/0) |
| `mode_response_smoke` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `mode_response_smoke_final` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `observable_lift_geometry_v1` | historical_observable_lift | completed_or_retained_partial | negative_or_nonidentifiable | run_local_observable_lift | historical_appendix_only | 1 (1/0) |
| `observable_subspace_v1` | historical_observable_lift | completed_or_retained_partial | negative_or_nonidentifiable | run_local_observable_lift | historical_appendix_only | 2 (2/0) |
| `observable_subspace_v2` | historical_observable_lift | completed_or_retained_partial | negative_or_nonidentifiable | run_local_observable_lift | historical_appendix_only | 2 (2/0) |
| `ood` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 0 (0/0) |
| `operator_learning_capacity_smoke_v1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `operator_learning_capacity_v1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 3 (2/1) |
| `operator_learning_capacity_v2` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 7 (7/0) |
| `operator_learning_capacity_v4_independent_lambda_spectral_floor` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 0 (0/0) |
| `operator_learning_capacity_v5_independent_lambda_spectral_floor` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 0 (0/0) |
| `operator_learning_capacity_v6_independent_lambda_spectral_floor` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 7 (7/0) |
| `optimization_ablation_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 13 (12/0) |
| `optimization_ablation_v1_smoke` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 5 (4/0) |
| `optimization_smoke` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `p0_tmp_index` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 0 (0/0) |
| `piezojet_amplitude_factorized` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (0/1) |
| `piezojet_amplitude_factorized_gpu` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `piezojet_amplitude_factorized_smoke` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `piezojet_balanced_formula` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `piezojet_balanced_formula_smoke` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `piezojet_tensorial_response_operator_gpu` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (0/1) |
| `polar_fluctuation_smoke` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `pretrain` | support_or_pretraining | completed_or_partial | support_artifact | historical_pre_v7_or_run_local | appendix_provenance_if_used | 1 (1/0) |
| `pretrain_cartesian` | support_or_pretraining | completed_or_partial | support_artifact | historical_pre_v7_or_run_local | appendix_provenance_if_used | 1 (1/0) |
| `pymatgen_parser_crosscheck_v1` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 1 (1/0) |
| `pymatgen_phonopy_parser_crosscheck_v2` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 1 (1/0) |
| `response_audit` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 1 (0/1) |
| `response_operator_action_capacity_v1` | response_operator_action_capacity_probe | completed | noninductive_capacity_diagnostic_available | v7_bec_transpose_true_factor_operator_action | appendix_implementation_diagnostic_only | 4 (3/0) |
| `response_operator_action_cpu_smoke_v1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `schema4_provenance_rebuild_v1` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 0 (0/0) |
| `smoke_tensorial_operator_gpu` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `stratified_subset_resampling_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 6 (5/0) |
| `strict_completion_train108_protocol_b_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 5 (3/0) |
| `strict_completion_train97_protocol_b_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 5 (3/0) |
| `strict_completion_v4_factorcurriculum_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `strict_completion_v4_seed42` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `strict_completion_v4_seed43` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `strict_completion_v4_seed44` | historical_dfpt_model_development | completed | development_diagnostic | historical_pre_v7 | historical_appendix_only | 1 (1/0) |
| `strict_learning_curve_v1` | historical_registered_factor_forensics | completed | mixed_or_negative_forensic | historical_factorized_pre_direct_u | historical_appendix_only | 23 (22/0) |
| `strict_v8_provenance_completion_v1` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 0 (0/0) |
| `symfc_force_constant_audit_v1` | data_convention_and_provenance_audit | completed | support_audit | source_audited_bec_transpose | appendix_provenance | 1 (1/0) |
| `teacher_forced_zero_basin_cpu_smoke_v1` | identifiability_memorization_probe | partial | failed_or_incomplete_noninductive_diagnostic | v7_bec_transpose_teacher_forced_direct_u | appendix_implementation_diagnostic_only | 1 (1/0) |
| `teacher_forced_zero_basin_cpu_smoke_v2` | identifiability_memorization_probe | partial | failed_or_incomplete_noninductive_diagnostic | v7_bec_transpose_teacher_forced_direct_u | appendix_implementation_diagnostic_only | 1 (1/0) |
| `teacher_forced_zero_basin_cpu_smoke_v3` | identifiability_memorization_probe | completed | cpu_implementation_smoke | v7_bec_transpose_teacher_forced_direct_u | appendix_implementation_diagnostic_only | 2 (2/0) |
| `teacher_forced_zero_basin_v1` | identifiability_memorization_probe | completed | noninductive_capacity_diagnostic_available | v7_bec_transpose_teacher_forced_direct_u | appendix_implementation_diagnostic_only | 3 (3/0) |
| `u_capacity_adjudication_smoke_v1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 1 (1/0) |
| `u_capacity_adjudication_v1` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 5 (5/0) |
| `u_capacity_adjudication_v2_balanced_objective` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 5 (5/0) |
| `u_capacity_adjudication_v3_long_fit` | early_development | completed_or_retained_partial | diagnostic | historical_pre_v7_or_run_local | historical_appendix_only | 2 (2/0) |
| `u_capacity_adjudication_v4_global_l3` | global_l3_same_id_capacity_gate | completed | positive_noninductive_capacity_diagnostic | v10_global_l3_independent_u | appendix_capacity_diagnostic_only | 3 (3/0) |

## Non-comparable families

Historical pre-v7, observable-lift/pInv, protocol A--G, sketch/mode-aware, and early-development cohorts remain evidence about prior hypotheses only. They must not be pooled with the v7 BEC-transpose regularized direct-U replay.

## Current registered replay

`exposure_matched_direct_u_v2_conditioning` is the registered 1/5/10/20-pass by seed 42/7/1729 grid. While it is running, every completed point remains an intermediate diagnostic. The final summary must include all planned points and may not select a pass from test data.

## File-level preservation

`outputs/EXPERIMENT_ARTIFACT_INDEX.jsonl` records every artifact path, byte size, and modification time. JSON, CSV, Markdown, YAML, and text records up to 20 MiB also receive SHA-256 hashes. Large checkpoints are retained and indexed by path/size/time without being duplicated.
