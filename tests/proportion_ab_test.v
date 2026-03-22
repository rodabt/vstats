module main

import experiment
import math

fn test__proportion_test_significant() {
	// 30/300 (10%) vs 48/300 (16%) — clear lift
	result := experiment.proportion_test(30, 300, 48, 300)

	assert result.rate_a == 0.10
	assert result.rate_b == 0.16
	assert math.abs(result.diff - 0.06) < 0.001
	assert result.z_statistic > 1.96
	assert result.p_value < 0.05
	assert result.significant == true
	assert result.ci_lower > 0.0   // CI excludes zero
	assert result.pooled_se > 0.0
	assert result.relative_lift > 0.0
}

fn test__proportion_test_not_significant() {
	// 50/500 vs 50/500 — identical rates
	result := experiment.proportion_test(50, 500, 50, 500)

	assert math.abs(result.diff) < 0.001
	assert math.abs(result.z_statistic) < 0.001
	assert result.p_value > 0.5
	assert result.significant == false
	assert result.ci_lower < 0.0
	assert result.ci_upper > 0.0
}

fn test__proportion_test_ci_width_scales_with_alpha() {
	// Tighter alpha => wider CI
	narrow := experiment.proportion_test(30, 300, 48, 300, experiment.ProportionTestConfig{ alpha: 0.10 })
	wide := experiment.proportion_test(30, 300, 48, 300, experiment.ProportionTestConfig{ alpha: 0.01 })

	narrow_width := narrow.ci_upper - narrow.ci_lower
	wide_width := wide.ci_upper - wide.ci_lower
	assert wide_width > narrow_width
}

fn test__proportion_test_relative_lift() {
	// rate_a=0.10, rate_b=0.20 => relative lift = (0.20-0.10)/0.10 = 1.0 (100%)
	result := experiment.proportion_test(100, 1000, 200, 1000)

	assert math.abs(result.relative_lift - 1.0) < 0.01
}

fn test__proportion_test_echoes_n() {
	result := experiment.proportion_test(30, 300, 48, 500)

	assert result.n_a == 300
	assert result.n_b == 500
}
