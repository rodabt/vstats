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
