module main

import experiment
import math

fn test__novelty_check_declining_effect() {
	// Clearly declining effect — novelty pattern
	effects := [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
	result := experiment.novelty_primacy_check(effects)

	assert result.novelty_suspected == true
	assert result.primacy_suspected == false
	assert result.slope < 0
	assert result.slope_significant == true
	assert result.early_effect > result.late_effect
	assert result.trend < 0
	assert result.sufficient_periods == true
}

fn test__novelty_check_growing_effect() {
	// Clearly growing effect — primacy pattern
	effects := [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
	result := experiment.novelty_primacy_check(effects)

	assert result.primacy_suspected == true
	assert result.novelty_suspected == false
	assert result.slope > 0
	assert result.trend > 0
	assert result.late_effect > result.early_effect
}

fn test__novelty_check_stable_effect() {
	// Flat effect — no novelty or primacy
	effects := [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
	result := experiment.novelty_primacy_check(effects)

	assert result.novelty_suspected == false
	assert result.primacy_suspected == false
	assert math.abs(result.slope) < 1e-10
	assert math.abs(result.trend) < 1e-10
}

fn test__novelty_check_insufficient_periods() {
	// Only 2 periods — below min_periods_for_slope default of 3
	effects := [0.5, 0.1]
	result := experiment.novelty_primacy_check(effects)

	assert result.sufficient_periods == false
	assert result.slope_significant == false
	assert result.slope_p_value == 1.0
	assert result.n_periods == 2
}

fn test__novelty_check_early_late_split() {
	// 5 periods: early = [:2] = [0.1, 0.2], late = [2:] = [0.3, 0.4, 0.5]
	effects := [0.1, 0.2, 0.3, 0.4, 0.5]
	result := experiment.novelty_primacy_check(effects)

	assert math.abs(result.early_effect - 0.15) < 1e-10
	assert math.abs(result.late_effect - 0.4) < 1e-10
	assert result.n_periods == 5
}
