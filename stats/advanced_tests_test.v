module stats

import math

fn test_anova_one_way() {
	// Create three groups with different means
	group1 := [1.0, 2.0, 3.0]
	group2 := [4.0, 5.0, 6.0]
	group3 := [7.0, 8.0, 9.0]
	
	f_stat, p_val := anova_one_way([group1, group2, group3])
	
	// F-statistic should be positive
	assert f_stat > 0, "F-statistic should be positive"
	// P-value should be between 0 and 1
	assert p_val >= 0 && p_val <= 1, "p-value should be between 0 and 1"
	// With clearly different groups, p-value should be small
	assert p_val < 0.1, "p-value should indicate significant difference"
}

fn test_confidence_interval_mean() {
	data := [1.0, 2.0, 3.0, 4.0, 5.0]
	
	// 95% confidence
	lower, upper := confidence_interval_mean(data, 0.95)
	
	// Mean is 3.0, should be within interval
	mean_val := mean(data)
	assert lower <= mean_val && mean_val <= upper, "mean should be within confidence interval"
	assert lower < upper, "lower bound should be less than upper bound"
}

fn test_cohens_d() {
	group1 := [1.0, 2.0, 3.0, 4.0, 5.0]
	group2 := [6.0, 7.0, 8.0, 9.0, 10.0]
	
	d := cohens_d(group1, group2)
	
	// Cohen's d = (mean1 - mean2) / pooled_std, so it's negative when group2 > group1
	assert d < 0, "Cohen's d should be negative when second group has higher mean"
	// For these groups, should be large effect (magnitude > 1.2)
	assert math.abs(d) > 1.0, "large effect size"
}

fn test_cramers_v() {
	// Simple 2x2 contingency table
	contingency := [
		[50, 10],
		[20, 20]
	]
	
	v := cramers_v(contingency)
	
	// Should be between 0 and 1
	assert v >= 0 && v <= 1, "Cramér's V should be between 0 and 1"
	assert v > 0, "non-zero association"
}

fn test_skewness() {
	// Symmetric distribution
	symmetric := [1.0, 2.0, 3.0, 4.0, 5.0]
	skew_sym := skewness(symmetric)
	
	// Should be close to 0
	assert math.abs(skew_sym) < 0.5, "symmetric distribution should have near-zero skewness"
	
	// Right-skewed distribution
	right_skewed := [1.0, 1.0, 1.0, 2.0, 5.0]
	skew_right := skewness(right_skewed)
	
	// Should be positive
	assert skew_right > 0, "right-skewed distribution should have positive skewness"
}

fn test_kurtosis() {
	// Normal-like distribution
	normal_like := [1.0, 2.0, 3.0, 4.0, 5.0]
	kurt := kurtosis(normal_like)
	
	// Kurtosis value (excess kurtosis ~ 0 for normal)
	assert !math.is_nan(kurt), "kurtosis should be a valid number"
	
	// Distribution with heavy tails
	heavy_tails := [1.0, 1.0, 3.0, 5.0, 5.0]
	kurt_heavy := kurtosis(heavy_tails)
	
	// Should be a valid number
	assert !math.is_nan(kurt_heavy), "kurtosis with heavy tails should be valid"
}
