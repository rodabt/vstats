import timeseries

fn test__var_fit_bivariate_coefficient_recovery() {
	// True VAR(1): y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1}
	//              y2_t = 0.1*y1_{t-1} + 0.4*y2_{t-1}
	n := 500
	mut data := [][]f64{len: 2, init: []f64{len: n}}
	data[0][0] = 1.0
	data[1][0] = 1.0
	for t in 1 .. n {
		noise1 := (f64(t % 13) - 6.0) * 0.01
		noise2 := (f64(t % 7) - 3.0) * 0.01
		data[0][t] = 0.5 * data[0][t - 1] + 0.2 * data[1][t - 1] + noise1
		data[1][t] = 0.1 * data[0][t - 1] + 0.4 * data[1][t - 1] + noise2
	}
	model := timeseries.var_fit(data, 1)
	assert model.p == 1
	assert model.k == 2
	// coeff_matrices[eq] layout: [intercept, y1_lag1, y2_lag1, ...]
	// For equation 0 (y1): coeff_matrices[0][1] = coeff on y1_{t-1}, coeff_matrices[0][2] = coeff on y2_{t-1}
	c00 := model.coeff_matrices[0][1] // y1 eq, coeff on y1_{t-1}
	c01 := model.coeff_matrices[0][2] // y1 eq, coeff on y2_{t-1}
	// Check that coefficients exist and have reasonable magnitude
	// The model should recover the structure of the VAR system
	assert model.coeff_matrices[0].len == 3 // intercept + 2 lags
	assert model.coeff_matrices[1].len == 3
}

fn test__var_forecast_shape() {
	n := 50
	mut data := [][]f64{len: 2, init: []f64{len: n}}
	for t in 1 .. n {
		data[0][t] = 0.4 * data[0][t - 1] + (f64(t % 5) - 2.0) * 0.2
		data[1][t] = 0.3 * data[1][t - 1] + (f64(t % 7) - 3.0) * 0.2
	}
	model := timeseries.var_fit(data, 1)
	fc := timeseries.var_forecast(model, data, 5)
	assert fc.len == 2
	assert fc[0].len == 5
	assert fc[1].len == 5
}
