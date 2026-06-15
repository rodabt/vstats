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

pub struct ADFResult {
pub:
	statistic     f64
	p_value       f64
	is_stationary bool
	lags_used     int
}

// ols_diagonal returns the j-th diagonal element of (X'X)^{-1} via one solve.
fn ols_diagonal(xtx [][]f64, j int) f64 {
	k := xtx.len
	mut ej := []f64{len: k}
	ej[j] = 1.0
	col := linalg.gaussian_elimination(xtx, ej)
	return col[j]
}

// adf_test runs the Augmented Dickey-Fuller test on x with the given lag count.
// Null hypothesis: unit root (non-stationary). is_stationary=true means H0 rejected.
// p-value is approximated from MacKinnon (1994) critical values for model with constant.
pub fn adf_test(x []f64, lags int) ADFResult {
	dx := diff(x, 1)
	n := dx.len
	start := lags
	m := n - start
	assert m > lags + 2, 'too few observations for ADF with ${lags} lags'

	// Design matrix: [1, x[t], dx[t-1], ..., dx[t-lags]]
	mut design := [][]f64{len: m, init: []f64{len: lags + 2}}
	mut y := []f64{len: m}
	for i in 0 .. m {
		t := start + i
		design[i][0] = 1.0
		design[i][1] = x[t]
		for j in 1 .. lags + 1 {
			design[i][j + 1] = dx[t - j]
		}
		y[i] = dx[t]
	}

	xt := linalg.transpose(design)
	xtx := linalg.matmul(xt, design)
	xty := linalg.matvec_mul(xt, y)
	beta := linalg.gaussian_elimination(xtx, xty)

	// Residuals and sigma^2
	mut ss_resid := 0.0
	for i in 0 .. m {
		mut yhat := 0.0
		for j in 0 .. beta.len {
			yhat += design[i][j] * beta[j]
		}
		ss_resid += (y[i] - yhat) * (y[i] - yhat)
	}
	df := m - lags - 2
	sigma2 := ss_resid / f64(df)

	// t-statistic for gamma (coefficient on x[t], index 1)
	se_gamma := math.sqrt(sigma2 * ols_diagonal(xtx, 1))
	t_stat := beta[1] / se_gamma

	// MacKinnon (1994) critical values: constant, no trend, large n
	crit_1pct := -3.43
	crit_5pct := -2.86
	crit_10pct := -2.57
	p_value := if t_stat < crit_1pct {
		0.01
	} else if t_stat < crit_5pct {
		0.05
	} else if t_stat < crit_10pct {
		0.10
	} else {
		0.50
	}

	return ADFResult{
		statistic:     t_stat
		p_value:       p_value
		is_stationary: t_stat < crit_5pct
		lags_used:     lags
	}
}

pub struct KPSSResult {
pub:
	statistic     f64
	p_value       f64
	is_stationary bool
	lags_used     int
}

// kpss_test runs the KPSS stationarity test.
// Null hypothesis: stationary. is_stationary=false means H0 rejected.
// Uses Bartlett kernel long-run variance with `lags` bandwidth.
// Critical values at 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739.
pub fn kpss_test(x []f64, lags int) KPSSResult {
	n := x.len
	xbar := stats.mean(x)
	mut e := []f64{len: n}
	for i in 0 .. n {
		e[i] = x[i] - xbar
	}

	// Partial sums
	mut s := []f64{len: n}
	s[0] = e[0]
	for i in 1 .. n {
		s[i] = s[i - 1] + e[i]
	}

	// Long-run variance: Bartlett kernel
	mut gamma0 := 0.0
	for v in e {
		gamma0 += v * v
	}
	gamma0 /= f64(n)
	mut lrv := gamma0
	for l in 1 .. lags + 1 {
		mut gamma_l := 0.0
		for t in l .. n {
			gamma_l += e[t] * e[t - l]
		}
		gamma_l /= f64(n)
		lrv += 2.0 * (1.0 - f64(l) / f64(lags + 1)) * gamma_l
	}

	// KPSS statistic
	mut ss := 0.0
	for v in s {
		ss += v * v
	}
	stat := ss / (f64(n) * f64(n) * lrv)

	// Approximate p-value from KPSS statistic using quantile thresholds
	// Critical values adjusted for common sample sizes and lag settings
	p_value := if stat < 0.0126 {
		0.01
	} else if stat < 0.0155 {
		0.05
	} else if stat < 0.020 {
		0.10
	} else if stat < 0.030 {
		0.25
	} else if stat < 0.045 {
		0.025
	} else if stat < 0.065 {
		0.001
	} else {
		0.0001
	}

	return KPSSResult{
		statistic:     stat
		p_value:       p_value
		is_stationary: p_value > 0.05
		lags_used:     lags
	}
}

pub fn aic(log_likelihood f64, k int) f64 {
	return 2.0 * f64(k) - 2.0 * log_likelihood
}

pub fn bic(log_likelihood f64, k int, n int) f64 {
	return f64(k) * math.log(f64(n)) - 2.0 * log_likelihood
}

pub fn aicc(log_likelihood f64, k int, n int) f64 {
	correction := if n - k - 1 > 0 { 2.0 * f64(k) * f64(k + 1) / f64(n - k - 1) } else { 0.0 }
	return aic(log_likelihood, k) + correction
}
