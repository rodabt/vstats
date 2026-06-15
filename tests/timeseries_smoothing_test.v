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
