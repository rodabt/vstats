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
}
