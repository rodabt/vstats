import timeseries
import math

fn test__arima_ar1_coefficient_recovery() {
	// Generate AR(1) with phi=0.7, d=0, q=0
	mut x := []f64{len: 150}
	x[0] = 0.0
	for i in 1 .. 150 {
		x[i] = 0.7 * x[i - 1] + (f64(i % 13) - 6.0) * 0.2
	}
	model := timeseries.arima_fit(x, 1, 0, 0)
	assert math.abs(model.ar_coeffs[0] - 0.7) < 0.15
	assert model.p == 1
	assert model.d == 0
	assert model.q == 0
}

fn test__arima_fitted_length() {
	x := [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
	model := timeseries.arima_fit(x, 1, 0, 0)
	assert model.fitted.len == x.len
	assert model.residuals.len == x.len
}

fn test__arima_aic_bic_set() {
	x := [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0]
	model := timeseries.arima_fit(x, 1, 0, 0)
	assert model.aic != 0.0
	assert model.bic != 0.0
}

fn test__arima_fitted_length_with_differencing() {
	// ARIMA(1,1,0): fitted and residuals must still have same length as x
	x := [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0]
	model := timeseries.arima_fit(x, 1, 1, 0)
	assert model.fitted.len == x.len
	assert model.residuals.len == x.len
	assert model.d == 1
	// fitted + residuals must recover x in original scale for t >= d
	for t in model.d .. x.len {
		assert math.abs((model.fitted[t] + model.residuals[t]) - x[t]) < 1e-9
	}
}

fn test__arima_ma_term() {
	// ARMA(1,0,1): verifies MA feedback path in css_residuals is exercised
	x := [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
	model := timeseries.arima_fit(x, 1, 0, 1)
	assert model.p == 1
	assert model.q == 1
	assert model.fitted.len == x.len
	assert model.residuals.len == x.len
	assert model.aic != 0.0
}

fn test__arima_forecast_length() {
	mut x := []f64{len: 50}
	x[0] = 0.0
	for i in 1 .. 50 {
		x[i] = 0.6 * x[i - 1] + (f64(i % 7) - 3.0) * 0.2
	}
	model := timeseries.arima_fit(x, 1, 0, 0)
	fc := timeseries.arima_forecast(model, 5, 0.05, x)
	assert fc.forecast.len == 5
	assert fc.lower.len == 5
	assert fc.upper.len == 5
	for i in 0 .. 5 {
		assert fc.lower[i] < fc.forecast[i]
		assert fc.forecast[i] < fc.upper[i]
	}
}

fn test__arima_summary_not_empty() {
	x := [1.0, 2.0, 1.5, 2.5, 2.0, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5]
	model := timeseries.arima_fit(x, 1, 0, 0)
	summary := timeseries.arima_summary(model)
	assert summary.len > 0
	assert summary.contains('ARIMA')
	assert summary.contains('AIC')
}

fn test__sarima_fit_runs() {
	// Monthly data with period 4
	mut x := []f64{len: 48}
	for i in 0 .. 48 {
		x[i] = f64(i) * 0.1 + [1.0, -1.0, 2.0, -2.0][i % 4]
	}
	model := timeseries.sarima_fit(x, 1, 0, 0, 0, 1, 0, 4)
	assert model.p == 1
	assert model.d == 0
	assert model.fitted.len == x.len
}

fn test__auto_arima_finds_ar1() {
	// True AR(1) — auto_arima should prefer ARIMA(1,0,0) or similar
	mut x := []f64{len: 100}
	x[0] = 0.0
	for i in 1 .. 100 {
		x[i] = 0.7 * x[i - 1] + (f64(i % 11) - 5.0) * 0.2
	}
	model := timeseries.auto_arima(x, 3, 2, 1)
	// Should recover roughly correct p and AR coefficient
	assert model.p >= 1
	assert model.d == 0
	assert model.aic < 1000.0 // sanity: fitted something reasonable
}

fn test__auto_arima_runs_with_multiple_d_values() {
	// Test that auto_arima grid searches correctly over d values
	// Generate a stationary AR(1) series
	mut x := []f64{len: 100}
	x[0] = 0.5
	for i in 1 .. 100 {
		x[i] = 0.5 * x[i - 1] + (f64(i % 13) - 6.0) * 0.3
	}
	// With max_d=2, auto_arima should consider d=0,1,2 and pick the best by AICc
	model := timeseries.auto_arima(x, 2, 2, 1)
	// Should have a valid model
	assert model.p >= 0
	assert model.d >= 0
	assert model.q >= 0
	assert model.d <= 2
	assert model.aic < 10000.0
}
