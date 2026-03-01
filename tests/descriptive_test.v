import stats
import math

fn test__mean() {
	assert stats.mean([f64(1), 2 , 3]) == f64(2.0)
}

fn test__median() {
	assert stats.median([f64(1), 10, 2, 9, 5]) == f64(5)
	assert stats.median([f64(1), 9, 2, 10]) == f64(5.5)
}

fn test__mode() {
	assert stats.mode([f64(1),2,2,2,3,3,3,5,6,7,8,8,9]) == [f64(2), 3]
}

fn test__range() {
	assert stats.range([f64(1),2,3,4,5,7]) == f64(6.0)
}

fn test__sum() {
	assert stats.sum([f64(1), 2, 3, 4, 5]) == f64(15)
	assert stats.sum([f64(0)]) == f64(0)
}

fn test__variance() {
	// [2, 4, 6]: mean=4, deviations=[-2,0,2], sum_sq=8, sample var = 8/2 = 4.0
	data := [f64(2), 4, 6]
	assert math.abs(stats.variance(data) - 4.0) < 1e-9
}

fn test__standard_deviation() {
	// Same dataset [2,4,6]: std = sqrt(4) = 2.0
	data := [f64(2), 4, 6]
	assert math.abs(stats.standard_deviation(data) - 2.0) < 1e-9
}

fn test__covariance() {
	// Perfectly correlated: x = y → cov = var(x)
	x := [f64(1), 2, 3, 4, 5]
	cov_xx := stats.covariance(x, x)
	var_x := stats.variance(x)
	assert math.abs(cov_xx - var_x) < 1e-9
	// Negatively correlated: cov(x, -x) = -var(x)
	neg_x := x.map(-it)
	cov_neg := stats.covariance(x, neg_x)
	assert math.abs(cov_neg + var_x) < 1e-9
}

fn test__correlation() {
	// Perfectly correlated → r = 1.0
	x := [f64(1), 2, 3, 4, 5]
	assert math.abs(stats.correlation(x, x) - 1.0) < 1e-9
	// Perfectly anti-correlated → r = -1.0
	neg_x := x.map(-it)
	assert math.abs(stats.correlation(x, neg_x) + 1.0) < 1e-9
}

fn test__quantile() {
	data := [f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10]
	// p=0.25 → index 2 (0-based) → value 3
	q25 := stats.quantile(data, 0.25)
	assert math.abs(q25 - 3.0) < 1e-9
	// p=0.75 → index 7 → value 8
	q75 := stats.quantile(data, 0.75)
	assert math.abs(q75 - 8.0) < 1e-9
}

fn test__interquartile_range() {
	data := [f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10]
	iqr := stats.interquartile_range(data)
	// Q3 - Q1 = 8 - 3 = 5
	assert math.abs(iqr - 5.0) < 1e-9
}

fn test__skewness() {
	// Perfectly symmetric data centred on 0 → skewness = 0.0
	sym := [f64(-2), -1, 0, 1, 2]
	assert math.abs(stats.skewness(sym)) < 1e-9
}

fn test__kurtosis() {
	// Large uniform-ish dataset → excess kurtosis near -1.2 (platykurtic), just verify it's finite
	data := [f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10]
	k := stats.kurtosis(data)
	assert k < 0.0 // uniform-like data has negative excess kurtosis
}