import vstats.stats
import math
import rand

fn test__delta_method_ratio_detects_effect() {
	// control: revenue/sessions = 10/2 = 5.0 per user (mean)
	// treatment: revenue/sessions = 14/2 = 7.0 per user (mean)
	// true effect = 2.0
	a_ctrl := [10.0, 12.0, 8.0, 10.0]
	b_ctrl := [2.0, 2.0, 2.0, 2.0]
	a_trt  := [14.0, 16.0, 12.0, 14.0]
	b_trt  := [2.0, 2.0, 2.0, 2.0]
	mut a := []f64{}
	mut b := []f64{}
	a << a_ctrl
	a << a_trt
	b << b_ctrl
	b << b_trt
	treatment := [0, 0, 0, 0, 1, 1, 1, 1]
	result := stats.delta_method_ratio(a, b, treatment, stats.DeltaMethodConfig{})
	assert math.abs(result.ratio_ctrl - 5.0) < 1e-6
	assert math.abs(result.ratio_trt - 7.0) < 1e-6
	assert math.abs(result.effect - 2.0) < 1e-6
	assert result.p_value < 0.05
	assert result.ci_lower > 0.0   // CI entirely positive
	assert result.ci_upper > result.effect
}

fn test__delta_method_ratio_no_effect() {
	// Same ratio in both groups — effect should be 0, p large
	a := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	b := [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
	treatment := [0, 0, 0, 1, 1, 1]
	result := stats.delta_method_ratio(a, b, treatment, stats.DeltaMethodConfig{})
	assert math.abs(result.effect) < 1e-9
	assert result.p_value > 0.05
}

fn test__bootstrap_test_detects_large_effect() {
	rand.seed([u32(42), u32(0)])
	ctrl := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	trt  := [3.0, 3.1, 2.9, 3.0, 3.05, 2.95, 3.02, 2.98, 3.01, 2.99]
	result := stats.bootstrap_test(ctrl, trt, 2000)
	assert result.p_value < 0.05
	assert math.abs(result.observed_diff - 2.0) < 0.01
}

fn test__bootstrap_test_no_effect() {
	rand.seed([u32(99), u32(0)])
	// Identical arrays — observed_diff = 0, no permutation can be more extreme → p = 0 or near 0
	// Actually all permuted diffs will be >= 0 (= observed), so p ≈ 1.0
	ctrl := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	trt  := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	result := stats.bootstrap_test(ctrl, trt, 2000)
	assert result.p_value > 0.5
	assert math.abs(result.observed_diff) < 1e-9
}

fn test__bootstrap_test_ci_contains_truth() {
	rand.seed([u32(7), u32(0)])
	ctrl := [0.0, 0.1, -0.1, 0.0, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01]
	trt  := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	result := stats.bootstrap_test(ctrl, trt, 2000)
	// True effect ≈ 1.0; CI should contain it
	assert result.ci_lower < 1.0
	assert result.ci_upper > 1.0
}
