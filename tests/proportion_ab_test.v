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
	// Lower alpha => wider CI (more confidence requires wider interval)
	high_alpha := experiment.proportion_test(30, 300, 48, 300, experiment.ProportionTestConfig{ alpha: 0.10 })
	low_alpha  := experiment.proportion_test(30, 300, 48, 300, experiment.ProportionTestConfig{ alpha: 0.01 })

	high_alpha_width := high_alpha.ci_upper - high_alpha.ci_lower
	low_alpha_width  := low_alpha.ci_upper - low_alpha.ci_lower
	assert low_alpha_width > high_alpha_width
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

fn test__proportion_test_zero_baseline_lift() {
	// rate_a = 0 => relative_lift should be 0.0 (not infinity)
	result := experiment.proportion_test(0, 500, 10, 500)

	assert result.rate_a == 0.0
	assert result.relative_lift == 0.0
	assert result.diff > 0.0
}

fn test__proportion_test_one_sided_greater_significant() {
	// 50/1000 (5%) vs 80/1000 (8%) — clear positive lift
	result := experiment.proportion_test(50, 1000, 80, 1000,
		experiment.ProportionTestConfig{ alternative: .greater })

	assert result.p_value < 0.05
	assert result.significant == true
}

fn test__proportion_test_one_sided_less_not_significant_when_positive_lift() {
	// Same data — .less should NOT be significant
	result := experiment.proportion_test(50, 1000, 80, 1000,
		experiment.ProportionTestConfig{ alternative: .less })

	assert result.p_value > 0.95
	assert result.significant == false
}

fn test__proportion_test_one_sided_sum_to_one() {
	// p_greater + p_less = 1.0
	r_greater := experiment.proportion_test(50, 1000, 80, 1000,
		experiment.ProportionTestConfig{ alternative: .greater })
	r_less    := experiment.proportion_test(50, 1000, 80, 1000,
		experiment.ProportionTestConfig{ alternative: .less })

	assert math.abs(r_greater.p_value + r_less.p_value - 1.0) < 1e-10
}

fn test__proportion_test_one_sided_half_two_sided() {
	// p_greater ≈ p_two_sided / 2 when rate_b > rate_a
	r_two     := experiment.proportion_test(50, 1000, 80, 1000)
	r_greater := experiment.proportion_test(50, 1000, 80, 1000,
		experiment.ProportionTestConfig{ alternative: .greater })

	assert math.abs(r_greater.p_value - r_two.p_value / 2.0) < 1e-10
}

fn test__proportion_test_ci_always_two_sided() {
	// CIs must be identical regardless of alternative
	r_two     := experiment.proportion_test(50, 1000, 80, 1000)
	r_greater := experiment.proportion_test(50, 1000, 80, 1000,
		experiment.ProportionTestConfig{ alternative: .greater })
	r_less    := experiment.proportion_test(50, 1000, 80, 1000,
		experiment.ProportionTestConfig{ alternative: .less })

	assert math.abs(r_two.ci_lower - r_greater.ci_lower) < 1e-10
	assert math.abs(r_two.ci_upper - r_greater.ci_upper) < 1e-10
	assert math.abs(r_two.ci_lower - r_less.ci_lower) < 1e-10
	assert math.abs(r_two.ci_upper - r_less.ci_upper) < 1e-10
}
