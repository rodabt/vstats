module main

import experiment
import math

fn test__sample_size_proportions_known_value() {
	// baseline=5%, mde=+1pp, alpha=0.05, power=0.80
	// Expected n ≈ 8000 per group (classic result for this setting)
	result := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.80)

	assert result.method == 'proportions'
	assert math.abs(f64(result.n_per_group) - 8100.0) < 500.0
	assert result.total_n == result.n_per_group * 2
	assert result.alpha == 0.05
	assert result.power == 0.80
	assert result.mde == 0.01
	assert result.baseline == 0.05
	assert result.baseline_std == 0.0
	assert result.effect_size > 0.0
}

fn test__sample_size_proportions_larger_effect() {
	// baseline=10%, mde=+5pp — bigger effect => fewer needed
	small := experiment.sample_size_proportions(0.10, 0.05, 0.05, 0.80)
	large := experiment.sample_size_proportions(0.10, 0.01, 0.05, 0.80)

	assert small.n_per_group < large.n_per_group
}

fn test__sample_size_proportions_higher_power() {
	// Higher power => larger sample
	low_power := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.80)
	high_power := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.90)

	assert high_power.n_per_group > low_power.n_per_group
}

fn test__sample_size_means_known_value() {
	// baseline_mean=100, std=20, mde=5, alpha=0.05, power=0.80
	// Cohen's d = 5/20 = 0.25, n = 2*((1.96+0.842)*20/5)^2 ≈ 252
	result := experiment.sample_size_means(100.0, 20.0, 5.0, 0.05, 0.80)

	assert result.method == 'means'
	assert math.abs(f64(result.n_per_group) - 252.0) < 15.0
	assert result.total_n == result.n_per_group * 2
	assert result.baseline == 100.0
	assert result.baseline_std == 20.0
	assert math.abs(result.effect_size - 0.25) < 0.01
	assert result.mde == 5.0
}

fn test__sample_size_means_higher_std() {
	// More variance => larger sample needed
	low_var := experiment.sample_size_means(100.0, 10.0, 5.0, 0.05, 0.80)
	high_var := experiment.sample_size_means(100.0, 30.0, 5.0, 0.05, 0.80)

	assert high_var.n_per_group > low_var.n_per_group
}

fn test__icc_identical_groups() {
	// Within each group all values are identical — maximum between-group variance
	// Groups: [1,1,1], [2,2,2], [3,3,3] — ICC should be high (near 1)
	g := [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
	val := experiment.icc(g)
	assert val > 0.8
}

fn test__icc_moderate() {
	// Tight within-group spread, meaningful between-group differences → 0 < ICC < 1
	g := [[10.0, 10.1, 9.9], [20.0, 20.1, 19.9], [30.0, 30.1, 29.9]]
	val := experiment.icc(g)
	assert val > 0.5 && val < 1.0
}

fn test__icc_zero_for_equal_means() {
	// If all groups have the same mean, SS_between = 0 → ICC ≤ 0 (clamped to 0)
	g := [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]]
	val := experiment.icc(g)
	assert val <= 0.0
}

fn test__design_effect_no_clustering() {
	// ICC = 0 → DEFF = 1 regardless of cluster size
	deff := experiment.design_effect(10.0, 0.0)
	assert deff == 1.0
}

fn test__design_effect_known_value() {
	// m_bar=10, ICC=0.05 → DEFF = 1 + 9*0.05 = 1.45
	deff := experiment.design_effect(10.0, 0.05)
	assert math.abs(deff - 1.45) < 1e-9
}
