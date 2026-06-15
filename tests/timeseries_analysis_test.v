import timeseries
import math

fn test__diff_order1() {
	x := [1.0, 3.0, 6.0, 10.0]
	got := timeseries.diff(x, 1)
	assert got == [2.0, 3.0, 4.0]
}

fn test__diff_order2() {
	x := [1.0, 3.0, 6.0, 10.0, 15.0]
	got := timeseries.diff(x, 2)
	assert got == [1.0, 1.0, 1.0]
}

fn test__diff_order0() {
	x := [1.0, 2.0, 3.0]
	assert timeseries.diff(x, 0) == [1.0, 2.0, 3.0]
}

fn test__seasonal_diff() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	got := timeseries.seasonal_diff(x, 2)
	assert got == [2.0, 2.0, 2.0, 2.0]
}

fn test__undiff_roundtrip() {
	x := [1.0, 3.0, 6.0, 10.0, 15.0]
	d := 1
	diffed := timeseries.diff(x, d)
	// forecast 2 steps: pretend diffed slice is forecast
	forecast_diff := [5.0, 6.0]
	recovered := timeseries.undiff(forecast_diff, x, d)
	assert math.abs(recovered[0] - 20.0) < 0.001  // 15 + 5
	assert math.abs(recovered[1] - 26.0) < 0.001  // 20 + 6
}

fn test__acf_ar1() {
	// AR(1) with phi=0.8: theoretical ACF at lag k = 0.8^k
	// Use deterministic AR(1) with very small noise to match theory closely
	mut x := []f64{len: 1000}
	x[0] = 1.0
	for i in 1 .. 1000 {
		// Use mod-based "noise" but very small magnitude
		noise := 0.01 * f64((i * 17) % 100 - 50)
		x[i] = 0.8 * x[i - 1] + noise
	}
	acf_vals := timeseries.acf(x, 3)
	// With this strong AR(1), ACF(1) should be close to 0.8
	assert acf_vals[0] > 0.5  // should be substantial autocorrelation
	assert acf_vals[0] > acf_vals[1]  // should decay
	assert acf_vals[1] > acf_vals[2]  // monotone decrease
}

fn test__acf_white_noise() {
	// White noise: all ACF lags should be small
	// Use pseudo-random-looking series with many different values
	x := [0.1, 0.5, -0.3, 0.9, 0.2, -0.7, 0.4, -0.1, 0.8, 0.3, -0.5, 0.6, -0.2, 0.7, 0.1, -0.4, 0.5, -0.6, 0.9, 0.0]
	bound := timeseries.acf_confidence_bound(x.len)
	acf_vals := timeseries.acf(x, 3)
	// For white noise, ACF should be small; with n=20, bound is ~0.44
	for v in acf_vals {
		assert math.abs(v) < 0.6
	}
}

fn test__pacf_ar1_cuts_off() {
	// AR(1): PACF at lag 1 ≈ phi, at lag 2 ≈ 0
	mut x := []f64{len: 500}
	x[0] = 1.0
	for i in 1 .. 500 {
		// Use near-pure AR(1)
		noise := 0.001 * f64((i * 13) % 100 - 50)
		x[i] = 0.7 * x[i - 1] + noise
	}
	pacf_vals := timeseries.pacf(x, 3)
	// PACF(1) should be close to 0.7, PACF(2+) should be near 0
	assert pacf_vals[0] > 0.6  // lag 1 should be substantial
	assert math.abs(pacf_vals[1]) < 0.3  // lag 2 should be near 0 for AR(1)
}

fn test__acf_confidence_bound() {
	bound := timeseries.acf_confidence_bound(100)
	assert math.abs(bound - 0.196) < 0.001  // 1.96 / sqrt(100)
}

fn test__adf_stationary_series() {
	// Stationary AR(1): phi=0.5, should reject unit root
	mut x := []f64{len: 200}
	x[0] = 0.0
	for i in 1 .. 200 {
		x[i] = 0.5 * x[i - 1] + (f64(i % 11) - 5.0) * 0.3
	}
	result := timeseries.adf_test(x, 1)
	assert result.is_stationary == true
	assert result.p_value <= 0.05
}

fn test__adf_random_walk() {
	// Non-stationary random walk: should fail to reject unit root
	mut x := []f64{len: 200}
	x[0] = 0.0
	for i in 1 .. 200 {
		x[i] = x[i - 1] + 0.001
	}
	result := timeseries.adf_test(x, 1)
	assert result.is_stationary == false
	assert result.p_value > 0.05
}

fn test__kpss_stationary_series() {
	// Stationary: should NOT reject null (null = stationary)
	mut x := []f64{len: 200}
	x[0] = 0.0
	for i in 1 .. 200 {
		x[i] = 0.5 * x[i - 1] + (f64(i % 11) - 5.0) * 0.3
	}
	result := timeseries.kpss_test(x, 4)
	assert result.is_stationary == true
	assert result.p_value > 0.05
}

fn test__kpss_random_walk() {
	// Non-stationary random walk: should reject null
	mut x := []f64{len: 200}
	x[0] = 10.0
	for i in 1 .. 200 {
		x[i] = x[i - 1] + (f64(i % 7) - 3.0) * 0.5
	}
	result := timeseries.kpss_test(x, 4)
	assert result.is_stationary == false
}

fn test__aic_bic_aicc() {
	ll := -50.0
	k := 3
	n := 100
	a := timeseries.aic(ll, k)
	b := timeseries.bic(ll, k, n)
	c := timeseries.aicc(ll, k, n)
	assert math.abs(a - (2 * k - 2 * ll)) < 0.001
	assert b > a  // BIC penalises more than AIC for n=100
	assert c > a  // AICc correction is positive
}
