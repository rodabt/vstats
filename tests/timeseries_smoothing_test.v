import timeseries
import math

fn test__ses_constant_series() {
	x := [5.0, 5.0, 5.0, 5.0, 5.0]
	result := timeseries.ses(x, 0.3)
	for v in result.fitted {
		assert math.abs(v - 5.0) < 0.5
	}
}

fn test__ses_forecast_length() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0]
	result := timeseries.ses(x, 0.4)
	assert result.fitted.len == x.len
	assert result.forecast.len == 1  // ses produces 1-step-ahead naively
}

fn test__holt_linear_trend() {
	// Perfect linear trend: SES misses it, Holt should track it
	mut x := []f64{len: 10}
	for i in 0 .. 10 {
		x[i] = f64(i) * 2.0
	}
	result := timeseries.holt(x, 0.8, 0.8)
	// Last fitted value should be near 18.0 (x[9])
	assert math.abs(result.fitted[9] - 18.0) < 3.0
	// Forecast should continue the trend
	assert result.forecast[0] > result.fitted[9]
}

fn test__holt_forecast_length() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0]
	result := timeseries.holt(x, 0.5, 0.3)
	assert result.fitted.len == x.len
	assert result.forecast.len >= 1
}

fn test__holt_winters_additive_seasonal() {
	// Clear quarterly seasonality on top of linear trend
	mut x := []f64{len: 20}
	seasonal := [3.0, -3.0, 2.0, -2.0]
	for i in 0 .. 20 {
		x[i] = f64(i) * 0.5 + seasonal[i % 4]
	}
	result := timeseries.holt_winters(x, 0.4, 0.1, 0.3, 4, .additive)
	// Forecast should continue upward trend
	assert result.forecast[0] > x[19]
	assert result.mse < 4.0
}

fn test__auto_ses_beats_naive() {
	x := [1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5, 3.0, 4.0]
	result := timeseries.auto_ses(x)
	// Optimised MSE should beat naive (last-value) MSE
	mut naive_mse := 0.0
	for t in 1 .. x.len {
		d := x[t] - x[t - 1]
		naive_mse += d * d
	}
	naive_mse /= f64(x.len - 1)
	assert result.mse <= naive_mse + 0.1
	assert result.alpha > 0.0 && result.alpha < 1.0
}

fn test__auto_holt_winters_finds_params() {
	mut x := []f64{len: 24}
	for i in 0 .. 24 {
		x[i] = f64(i) * 0.3 + [2.0, -1.0, 2.0, -3.0][i % 4]
	}
	result := timeseries.auto_holt_winters(x, 4, .additive)
	assert result.alpha > 0.0 && result.alpha < 1.0
	assert result.beta > 0.0 && result.beta < 1.0
	assert result.gamma > 0.0 && result.gamma < 1.0
	assert result.forecast.len > 0
}
