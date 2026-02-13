module hypothesis

import math
import stats
import prob

@[params]
pub struct TestParams {
	alpha f64 = 0.05
}

// One-sample t-test
pub fn t_test_one_sample(x []f64, mu f64, tp TestParams) (f64, f64) {
	assert x.len >= 2, "sample size must be at least 2"
	
	x_bar := stats.mean(x)
	s := stats.standard_deviation(x)
	n := f64(x.len)
	
	t_stat := (x_bar - mu) / (s / math.sqrt(n))
	// df := x.len - 1
	
	// p-value approximation (two-tailed)
	p_val := 2.0 * (1.0 - normal_cdf_approx(math.abs(t_stat)))
	
	return t_stat, p_val
}

// Two-sample t-test (independent samples, equal variances)
pub fn t_test_two_sample(x []f64, y []f64, tp TestParams) (f64, f64) {
	assert x.len >= 2 && y.len >= 2, "sample sizes must be at least 2"
	
	x_bar := stats.mean(x)
	y_bar := stats.mean(y)
	s_x := stats.standard_deviation(x)
	s_y := stats.standard_deviation(y)
	
	n_x := f64(x.len)
	n_y := f64(y.len)
	
	// Pooled standard deviation
	s_p := math.sqrt(((n_x - 1) * s_x * s_x + (n_y - 1) * s_y * s_y) / (n_x + n_y - 2))
	
	t_stat := (x_bar - y_bar) / (s_p * math.sqrt(1.0/n_x + 1.0/n_y))
	
	p_val := 2.0 * (1.0 - normal_cdf_approx(math.abs(t_stat)))
	
	return t_stat, p_val
}

// Chi-squared test for independence (contingency table) - main function for categorical data
pub fn chi_squared_test(contingency [][]int) (f64, f64) {
	assert contingency.len >= 2, "contingency table must have at least 2 rows"
	assert contingency[0].len >= 2, "contingency table must have at least 2 columns"
	
	rows := contingency.len
	cols := contingency[0].len
	
	mut row_totals := []f64{len: rows}
	mut col_totals := []f64{len: cols}
	mut total := 0.0
	
	for i in 0..rows {
		for j in 0..cols {
			row_totals[i] += f64(contingency[i][j])
			col_totals[j] += f64(contingency[i][j])
			total += f64(contingency[i][j])
		}
	}
	
	mut chi2 := 0.0
	for i in 0..rows {
		for j in 0..cols {
			expected := (row_totals[i] * col_totals[j]) / total
			if expected > 0 {
				observed := f64(contingency[i][j])
				chi2 += (observed - expected) * (observed - expected) / expected
			}
		}
	}
	
	df := (rows - 1) * (cols - 1)
	p_val := 1.0 - chi_squared_cdf_approx(chi2, df)
	
	return chi2, p_val
}

// Chi-squared goodness of fit test
pub fn chi_squared_gof_test(observed []f64, expected []f64) (f64, f64) {
	assert observed.len == expected.len, "observed and expected must have same length"
	assert observed.len > 0, "arrays cannot be empty"
	
	mut chi2 := 0.0
	for i := 0; i < observed.len; i++ {
		if expected[i] > 0 {
			chi2 += (observed[i] - expected[i]) * (observed[i] - expected[i]) / expected[i]
		}
	}
	
	df := observed.len - 1
	p_val := 1.0 - chi_squared_cdf_approx(chi2, df)
	
	return chi2, p_val
}

// Pearson correlation test
pub fn correlation_test(x []f64, y []f64, tp TestParams) (f64, f64) {
	assert x.len == y.len, "x and y must have the same length"
	assert x.len >= 3, "sample size must be at least 3"
	
	r := stats.correlation(x, y)
	n := f64(x.len)
	
	// t-statistic for correlation
	t_stat := r * math.sqrt((n - 2) / (1 - r * r))
	
	p_val := 2.0 * (1.0 - normal_cdf_approx(math.abs(t_stat)))
	
	return r, p_val
}

// Wilcoxon signed-rank test (non-parametric)
pub fn wilcoxon_signed_rank_test(x []f64, y []f64) (f64, f64) {
	assert x.len == y.len, "samples must have the same length"
	
	mut differences := []f64{}
	for i := 0; i < x.len; i++ {
		differences << (x[i] - y[i])
	}
	
	// Remove zeros
	differences = differences.filter(it != 0)
	
	if differences.len == 0 {
		return 0, 1.0
	}
	
	// Calculate absolute differences
	mut abs_diffs := differences.map(math.abs(it))
	abs_diffs.sort()
	
	// Assign ranks
	mut ranks := []f64{len: abs_diffs.len}
	for i := 0; i < abs_diffs.len; i++ {
		ranks[i] = f64(i + 1)
	}
	
	// Calculate W+ (sum of positive ranks)
	mut w_plus := 0.0
	for i := 0; i < differences.len; i++ {
		if differences[i] > 0 {
			w_plus += ranks[i]
		}
	}
	
	n := f64(differences.len)
	mean_w := n * (n + 1) / 4
	var_w := n * (n + 1) * (2*n + 1) / 24
	
	z := (w_plus - mean_w) / math.sqrt(var_w)
	p_val := 2.0 * (1.0 - normal_cdf_approx(math.abs(z)))
	
	return w_plus, p_val
}

// Mann-Whitney U test (non-parametric, two-sample)
pub fn mann_whitney_u_test(x []f64, y []f64) (f64, f64) {
	mut combined := []f64{}
	combined << x
	combined << y
	combined.sort()
	
	// Assign ranks
	mut x_ranks := 0.0
	for xi in x {
		for i, v in combined {
			if v == xi {
				x_ranks += f64(i + 1)
				break
			}
		}
	}
	
	n1 := f64(x.len)
	n2 := f64(y.len)
	
	u_stat := x_ranks - n1 * (n1 + 1) / 2
	
	mean_u := n1 * n2 / 2
	var_u := n1 * n2 * (n1 + n2 + 1) / 12
	
	z := (u_stat - mean_u) / math.sqrt(var_u)
	p_val := 2.0 * (1.0 - normal_cdf_approx(math.abs(z)))
	
	return u_stat, p_val
}

// Shapiro-Wilk normality test approximation
pub fn shapiro_wilk_test(x []f64) (f64, f64) {
	assert x.len >= 3, "sample size must be at least 3"
	
	mut sorted := x.sorted()
	n := sorted.len
	
	// Calculate mean and standard deviation
	mu := stats.mean(sorted)
	// sigma := stats.standard_deviation(sorted)
	
	// Calculate W statistic approximation
	mut numerator := 0.0
	for i := 0; i < n / 2; i++ {
		a := sorted[n - 1 - i] - sorted[i]
		w := f64(i + 1) / f64(n / 2)
		numerator += w * a * a
	}
	
	mut denominator := 0.0
	for xi in sorted {
		denominator += (xi - mu) * (xi - mu)
	}
	
	w_stat := if denominator > 0 { numerator / denominator } else { 0 }
	
	// Approximate p-value
	p_val := 1.0 - w_stat
	
	return w_stat, p_val
}

// Helper: normal CDF approximation (using error function)
fn normal_cdf_approx(z f64) f64 {
	return (1 + math.erf(z / math.sqrt(2))) / 2
}

// Helper: chi-squared CDF approximation
fn chi_squared_cdf_approx(x f64, df int) f64 {
	return prob.gamma_pdf(x, f64(df) / 2, 2)
}

// Alias for one_sample_t_test (old name)
pub fn one_sample_t_test(x []f64, mu f64) (f64, f64) {
	return t_test_one_sample(x, mu, TestParams{})
}

// Alias for two_sample_t_test (old name)
pub fn two_sample_t_test(x []f64, y []f64) (f64, f64) {
	return t_test_two_sample(x, y, TestParams{})
}

// Kolmogorov-Smirnov test
pub fn ks_test(sample1 []f64, sample2 []f64) (f64, f64) {
	mut all_vals := []f64{}
	all_vals << sample1
	all_vals << sample2
	all_vals.sort()
	
	n1 := f64(sample1.len)
	n2 := f64(sample2.len)
	
	mut d_max := 0.0
	
	for val in all_vals {
		// CDF for sample1
		mut count1 := 0
		for v in sample1 {
			if v <= val {
				count1++
			}
		}
		cdf1 := f64(count1) / n1
		
		// CDF for sample2
		mut count2 := 0
		for v in sample2 {
			if v <= val {
				count2++
			}
		}
		cdf2 := f64(count2) / n2
		
		d := math.abs(cdf1 - cdf2)
		if d > d_max {
			d_max = d
		}
	}
	
	// Approximate p-value using standard KS distribution
	n := (n1 * n2) / (n1 + n2)
	lambda := d_max * math.sqrt(n)
	p_val := 2.0 * math.exp(-2.0 * lambda * lambda)
	
	return d_max, p_val
}
