import hypothesis
import math

fn test__t_test_one_sample() {
	// Test with sample from N(5, 1)
	x := [4.9, 5.1, 5.0, 4.8, 5.2]
	t_stat, p_val := hypothesis.t_test_one_sample(x, 5.0, hypothesis.TestParams{})
	
	// t-statistic should be small (close to 0) since sample mean ~ 5
	assert math.abs(t_stat) < 0.5, "t-stat should be small for sample from H0 distribution"
	// p-value should be high
	assert p_val > 0.05, "p-value should be high for sample consistent with H0"
}

fn test__t_test_one_sample_reject() {
	// Test with sample clearly different from mu
	x := [10.0, 10.5, 9.8, 10.2, 10.1]
	t_stat, p_val := hypothesis.t_test_one_sample(x, 5.0, hypothesis.TestParams{})
	
	// t-statistic should be large
	assert math.abs(t_stat) > 5.0, "t-stat should be large for sample far from H0"
}

fn test__t_test_two_sample() {
	// Two samples with similar means
	x := [1.0, 1.1, 0.9, 1.0, 1.1]
	y := [1.0, 0.95, 1.05, 1.0, 1.0]
	t_stat, p_val := hypothesis.t_test_two_sample(x, y, hypothesis.TestParams{})
	
	// t-statistic should be small
	assert math.abs(t_stat) < 1.0, "t-stat should be small for similar samples"
}

fn test__t_test_two_sample_reject() {
	// Two clearly different samples
	x := [1.0, 1.0, 1.0, 1.0, 1.0]
	y := [5.0, 5.0, 5.0, 5.0, 5.0]
	t_stat, p_val := hypothesis.t_test_two_sample(x, y, hypothesis.TestParams{})
	
	// t-statistic should be large
	assert math.abs(t_stat) > 5.0, "t-stat should be large for different samples"
}

fn test__chi_squared_test() {
	// Fair die: observed = [10, 10, 10, 10, 10, 10], expected uniform
	observed := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	expected := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	chi2, p_val := hypothesis.chi_squared_test(observed, expected)
	
	// Chi-squared should be 0 (perfect fit)
	assert chi2 == 0.0, "chi-squared should be 0 for perfect fit"
}

fn test__chi_squared_test_deviate() {
	// Biased die: observed deviates from expected
	observed := [5.0, 15.0, 10.0, 10.0, 10.0, 10.0]
	expected := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	chi2, p_val := hypothesis.chi_squared_test(observed, expected)
	
	// Chi-squared should be > 0
	assert chi2 > 0.0, "chi-squared should be positive for deviate distribution"
}

fn test__correlation_test() {
	// Perfect positive correlation
	x := [1.0, 2.0, 3.0, 4.0, 5.0]
	y := [2.0, 4.0, 6.0, 8.0, 10.0]
	r, p_val := hypothesis.correlation_test(x, y, hypothesis.TestParams{})
	
	// r should be close to 1 (perfect correlation)
	assert r > 0.99, "correlation should be ~1 for perfect positive correlation"
}

fn test__correlation_test_no_correlation() {
	// No correlation
	x := [1.0, 2.0, 3.0, 4.0, 5.0]
	y := [2.0, 1.0, 4.0, 3.0, 5.0]
	r, p_val := hypothesis.correlation_test(x, y, hypothesis.TestParams{})
	
	// r should exist (between -1 and 1)
	assert r >= -1.0 && r <= 1.0, "correlation should be valid"
}

fn test__wilcoxon_signed_rank_test() {
	// Paired samples with difference ~ 0
	x := [1.0, 2.0, 3.0, 4.0]
	y := [1.1, 1.9, 3.1, 3.9]
	w_plus, p_val := hypothesis.wilcoxon_signed_rank_test(x, y)
	
	// w_plus should exist
	assert w_plus >= 0.0
}

fn test__wilcoxon_signed_rank_test_identical() {
	// Identical samples
	x := [1.0, 2.0, 3.0, 4.0]
	y := [1.0, 2.0, 3.0, 4.0]
	w_plus, p_val := hypothesis.wilcoxon_signed_rank_test(x, y)
	
	// All differences are 0, should return (0, 1)
	assert w_plus == 0.0
	assert p_val == 1.0
}

fn test__mann_whitney_u_test() {
	// Two samples
	x := [1.0, 2.0, 3.0]
	y := [4.0, 5.0, 6.0]
	u_stat, p_val := hypothesis.mann_whitney_u_test(x, y)
	
	// u_stat should be valid
	assert u_stat >= 0.0
	assert p_val >= 0.0 && p_val <= 1.0
}

fn test__shapiro_wilk_test() {
	// Approximately normal sample
	x := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	w_stat, p_val := hypothesis.shapiro_wilk_test(x)
	
	// w_stat should be in [0, 1]
	assert w_stat >= 0.0 && w_stat <= 1.0
	assert p_val >= 0.0 && p_val <= 1.0
}
