module timeseries

import math
import stats
import linalg

// diff applies order-d differencing to series x.
// Returns a slice of length x.len - d.
pub fn diff(x []f64, d int) []f64 {
	if d == 0 {
		return x.clone()
	}
	mut result := []f64{len: x.len - 1}
	for i in 1 .. x.len {
		result[i - 1] = x[i] - x[i - 1]
	}
	return diff(result, d - 1)
}

// seasonal_diff applies seasonal differencing with the given period.
// Returns a slice of length x.len - period.
pub fn seasonal_diff(x []f64, period int) []f64 {
	mut result := []f64{len: x.len - period}
	for i in period .. x.len {
		result[i - period] = x[i] - x[i - period]
	}
	return result
}

// undiff inverts d-order differencing on a forecast array.
// original is the full pre-forecast series (needed for seed values).
// Returns forecast values in the original (undifferenced) scale.
pub fn undiff(forecast_diff []f64, original []f64, d int) []f64 {
	if d == 0 {
		return forecast_diff.clone()
	}
	// Seed is last value of the (d-1)-times differenced original
	orig_d1 := diff(original, d - 1)
	seed := orig_d1[orig_d1.len - 1]
	mut integrated := []f64{len: forecast_diff.len}
	integrated[0] = seed + forecast_diff[0]
	for i in 1 .. forecast_diff.len {
		integrated[i] = integrated[i - 1] + forecast_diff[i]
	}
	return undiff(integrated, original, d - 1)
}

// acf computes the sample autocorrelation at lags 1..nlags.
pub fn acf(x []f64, nlags int) []f64 {
	n := x.len
	xbar := stats.mean(x)
	mut denom := 0.0
	for v in x {
		denom += (v - xbar) * (v - xbar)
	}
	mut result := []f64{len: nlags}
	for k in 1 .. nlags + 1 {
		mut num := 0.0
		for t in k .. n {
			num += (x[t] - xbar) * (x[t - k] - xbar)
		}
		result[k - 1] = if denom < 1e-14 { 0.0 } else { num / denom }
	}
	return result
}

// acf_confidence_bound returns Bartlett's ±1.96/√n significance threshold.
pub fn acf_confidence_bound(n int) f64 {
	return 1.96 / math.sqrt(f64(n))
}

// pacf computes partial autocorrelation at lags 1..nlags via Levinson-Durbin.
pub fn pacf(x []f64, nlags int) []f64 {
	r := acf(x, nlags)
	// phi[k][j] = AR coefficient j in the AR(k) model; PACF[k] = phi[k][k]
	mut phi := [][]f64{len: nlags + 1, init: []f64{len: nlags + 1}}
	mut result := []f64{len: nlags}
	if nlags == 0 {
		return result
	}
	phi[1][1] = r[0]
	result[0] = r[0]
	for k in 2 .. nlags + 1 {
		mut num := r[k - 1]
		mut den := 1.0
		for j in 1 .. k {
			num -= phi[k - 1][j] * r[k - 1 - j]
			den -= phi[k - 1][j] * r[j - 1]
		}
		phi[k][k] = if math.abs(den) < 1e-12 { 0.0 } else { num / den }
		for j in 1 .. k {
			phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
		}
		result[k - 1] = phi[k][k]
	}
	return result
}
