module main

import experiment
import math

// ============================================================================
// SRM Tests
// ============================================================================

fn test__srm_test_no_srm() {
	// Perfect 50/50 split → no SRM
	result := experiment.srm_test(1000, 1000, 0.5, 0.05)
	assert result.srm_detected == false
	assert result.chi2_statistic < 1e-9
	assert result.p_value > 0.5
}

fn test__srm_test_detects_mismatch() {
	// Extreme imbalance: 900 vs 100 with 50% expected → strong SRM
	result := experiment.srm_test(900, 100, 0.5, 0.05)
	assert result.srm_detected == true
	assert result.chi2_statistic > 10.0
	assert result.p_value < 0.001
}

fn test__srm_test_unequal_expected_split() {
	// Expected 90/10 split, observed exactly that → no SRM
	result := experiment.srm_test(900, 100, 0.1, 0.05)
	assert result.srm_detected == false
	assert result.chi2_statistic < 1e-9
}

// ============================================================================
// Simpson's Check Tests
// ============================================================================

fn test__simpsons_check_no_reversal() {
	// Overall positive, all subgroups positive → no reversal
	result := experiment.simpsons_check(2.0, [1.0, 3.0, 2.5])
	assert result.reversal_found == false
}

fn test__simpsons_check_reversal_detected() {
	// Overall positive, one subgroup negative → reversal
	result := experiment.simpsons_check(1.0, [2.0, -0.5, 1.5])
	assert result.reversal_found == true
}

fn test__simpsons_check_negative_overall_reversal() {
	// Overall negative, one subgroup positive → reversal
	result := experiment.simpsons_check(-1.0, [-2.0, 0.5, -1.5])
	assert result.reversal_found == true
}

// ============================================================================
// HTE Subgroup Tests
// ============================================================================

fn test__hte_subgroup_two_groups() {
	// Subgroup 0: no effect (ctrl mean = trt mean = 10)
	// Subgroup 1: +5 effect
	mut y         := []f64{}
	mut treatment := []int{}
	mut subgroup  := []int{}
	for i in 0..10 {
		y         << 10.0 + f64(i % 3) * 0.1
		treatment << i % 2
		subgroup  << 0
	}
	for i in 0..10 {
		base := if i % 2 == 0 { 10.0 } else { 15.0 }
		y         << base + f64(i % 3) * 0.1
		treatment << i % 2
		subgroup  << 1
	}
	result := experiment.hte_subgroup(y, treatment, subgroup, ['sg0', 'sg1'])
	assert result.subgroup_labels.len == 2
	assert result.subgroup_effects.len == 2
	assert result.subgroup_p_values.len == 2
	assert result.subgroup_ns[0] == 10
	assert result.subgroup_ns[1] == 10
	// Subgroup 1 should show a larger effect than subgroup 0
	assert math.abs(result.subgroup_effects[1]) > math.abs(result.subgroup_effects[0])
}
