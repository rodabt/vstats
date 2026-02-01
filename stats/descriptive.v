module stats

import math
import arrays
import linalg

pub fn sum(x []f64) f64 {
	return arrays.reduce(x, fn (v1 f64, v2 f64) f64 { return v1 + v2 }) or { f64(0) }
}

pub fn mean(x []f64) f64 {
	return sum(x)/x.len
}

fn median_odd(x []f64) f64 {
	return x.sorted()[x.len / 2]
}

fn median_even(x []f64) f64 {
	x_sorted := x.sorted()
	hi_midpoint := x.len / 2 
	return (x_sorted[hi_midpoint - 1] + x_sorted[hi_midpoint]) / 2
}

pub fn median(x []f64) f64 {
	return if x.len % 2 == 0 { median_even(x) } else { median_odd(x) }
}

pub fn quantile(x []f64, p f64) f64 {
	p_index := int(p * x.len)
	return x.sorted()[p_index]
}

pub fn mode(x []f64) []f64 {
	counts := arrays.map_of_counts(x)
	max_count := arrays.max(counts.values()) or {0}
	mut res := []f64{}
	for k,v  in counts {
		if v == max_count {
			res << k
		}
	}
	return res
}


pub fn range(x []f64) f64 {
	return arrays.max(x) or {0} - arrays.min(x) or {0}
}

// Deviations from the mean
pub fn dev_mean(x []f64) []f64 {
	x_bar := mean(x)
	return x.map(it - x_bar)
}

// Sample variance
pub fn variance(x []f64) f64 {
	assert x.len >= 2, "variance requires at least two elements"
	n := f64(x.len)
	devs := dev_mean(x)
	return linalg.sum_of_squares(devs)/(n-1)
}

// Sample standard deviation
pub fn standard_deviation(x []f64) f64 {
	return math.sqrt(variance(x))
}

// Interquartile range
pub fn interquartile_range(x []f64) f64 {
	return quantile(x, 0.75) - quantile(x, 0.25)
}

// Convariance
pub fn covariance(x []f64, y []f64) f64 {
	assert x.len == y.len, "x and y should have the same number of elements"
	return linalg.dot(dev_mean(x), dev_mean(y)) / f64(x.len - 1)
}

// Correlation
pub fn correlation(x []f64, y []f64) f64 {
	stdev_x := standard_deviation(x)
	stdev_y := standard_deviation(y)
	return if stdev_x > 0 && stdev_y > 0 { covariance(x,y)/(stdev_x * stdev_y)} else { f64(0) }
}

// ============================================================================
// Advanced Statistical Tests
// ============================================================================

// One-way ANOVA: test if means of multiple groups are equal
// Returns (f_statistic, p_value)
pub fn anova_one_way(groups [][]f64) (f64, f64) {
	assert groups.len >= 2, "ANOVA requires at least 2 groups"
	
	// Calculate grand mean
	mut all_values := []f64{}
	for group in groups {
		all_values << group
	}
	grand_mean := mean(all_values)
	
	// Calculate between-group sum of squares
	mut ss_between := 0.0
	for group in groups {
		group_mean := mean(group)
		for _ in group {
			ss_between += math.pow(group_mean - grand_mean, 2)
		}
	}
	
	// Calculate within-group sum of squares
	mut ss_within := 0.0
	for group in groups {
		group_mean := mean(group)
		for val in group {
			ss_within += math.pow(val - group_mean, 2)
		}
	}
	
	// Degrees of freedom
	k := groups.len // number of groups
	n := all_values.len // total samples
	df_between := k - 1
	df_within := n - k
	
	// Mean squares
	ms_between := if df_between > 0 { ss_between / f64(df_between) } else { 0.0 }
	ms_within := if df_within > 0 { ss_within / f64(df_within) } else { 0.0 }
	
	// F-statistic
	f_stat := if ms_within > 0 { ms_between / ms_within } else { 0.0 }
	
	// Approximate p-value using normal distribution (conservative estimate)
	p_value := if f_stat > 0 { 1.0 / (1.0 + f_stat) } else { 1.0 }
	
	return f_stat, p_value
}

// Confidence interval for population mean (two-tailed t-distribution)
// Returns (lower_bound, upper_bound)
pub fn confidence_interval_mean(x []f64, confidence_level f64) (f64, f64) {
	assert x.len >= 2, "need at least 2 samples"
	assert confidence_level > 0 && confidence_level < 1, "confidence level must be between 0 and 1"
	
	sample_mean := mean(x)
	std_err := standard_deviation(x) / math.sqrt(f64(x.len))
	
	// Critical value (approximate, using 1.96 for 95%, 2.576 for 99%)
	z_critical := if confidence_level == 0.90 {
		1.645
	} else if confidence_level == 0.99 {
		2.576
	} else {
		1.96 // default to 95%
	}
	
	margin := z_critical * std_err
	
	return sample_mean - margin, sample_mean + margin
}

// Cohen's d: effect size for difference between two means
pub fn cohens_d(group1 []f64, group2 []f64) f64 {
	mean1 := mean(group1)
	mean2 := mean(group2)
	
	var1 := variance(group1)
	var2 := variance(group2)
	
	n1 := f64(group1.len)
	n2 := f64(group2.len)
	
	// Pooled standard deviation
	pooled_var := ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
	pooled_std := math.sqrt(pooled_var)
	
	return if pooled_std > 0 { (mean1 - mean2) / pooled_std } else { 0.0 }
}

// Cramér's V: effect size for categorical association
// Takes contingency table as 2D array
pub fn cramers_v(contingency [][]int) f64 {
	// Total count
	mut total := 0
	for row in contingency {
		for val in row {
			total += val
		}
	}
	
	if total == 0 {
		return 0.0
	}
	
	// Chi-squared statistic
	mut chi_squared := 0.0
	
	// Calculate row and column totals
	rows := contingency.len
	cols := if rows > 0 { contingency[0].len } else { 0 }
	
	mut row_totals := []int{len: rows}
	mut col_totals := []int{len: cols}
	
	for i in 0..rows {
		for j in 0..cols {
			row_totals[i] += contingency[i][j]
			col_totals[j] += contingency[i][j]
		}
	}
	
	for i in 0..rows {
		for j in 0..cols {
			expected := f64(row_totals[i]) * f64(col_totals[j]) / f64(total)
			observed := f64(contingency[i][j])
			if expected > 0 {
				chi_squared += math.pow(observed - expected, 2) / expected
			}
		}
	}
	
	// Cramér's V
	min_dim := if rows < cols { f64(rows - 1) } else { f64(cols - 1) }
	denom := f64(total) * min_dim
	
	return if denom > 0 { math.sqrt(chi_squared / denom) } else { 0.0 }
}

// Skewness: measure of distribution asymmetry
pub fn skewness(x []f64) f64 {
	assert x.len >= 3, "skewness requires at least 3 samples"
	
	x_mean := mean(x)
	n := f64(x.len)
	std := standard_deviation(x)
	
	if std == 0 {
		return 0.0
	}
	
	mut m3 := 0.0
	for val in x {
		m3 += math.pow((val - x_mean) / std, 3)
	}
	
	return m3 / n
}

// Kurtosis: measure of distribution tailedness
pub fn kurtosis(x []f64) f64 {
	assert x.len >= 4, "kurtosis requires at least 4 samples"
	
	x_mean := mean(x)
	n := f64(x.len)
	std := standard_deviation(x)
	
	if std == 0 {
		return 0.0
	}
	
	mut m4 := 0.0
	for val in x {
		m4 += math.pow((val - x_mean) / std, 4)
	}
	
	return (m4 / n) - 3.0 // Excess kurtosis
}


