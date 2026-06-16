// Scenario: ARIMA Time Series Forecasting
// Demonstrates: vstats.timeseries — ARIMA fitting, auto_arima, forecasting, decomposition
// Python equivalent: statsmodels.tsa.arima.model.ARIMA + seasonal_decompose
module main

import vstats.timeseries
import math

fn main() {
	println('=== ARIMA Time Series Forecasting ===\n')

	// --- Synthetic series: AR(1) with trend and quarterly seasonality ---
	// x[t] = 0.6*x[t-1] + trend + seasonal + noise
	mut x := []f64{len: 60}
	seasonal := [3.0, -2.0, 4.0, -5.0]
	for t in 0 .. 60 {
		noise := (f64(t % 11) - 5.0) * 0.3
		prev := if t == 0 { 0.0 } else { x[t - 1] }
		x[t] = 0.6 * prev + f64(t) * 0.2 + seasonal[t % 4] + noise
	}

	println('Series: 60 observations, AR(1) + linear trend + quarterly seasonality')
	println('First 8 values: ${x[0..8].map(fn (v f64) string { return '${v:.2f}' }).join(', ')}\n')

	// --- Unit root tests ---
	println('--- Unit Root Tests ---')
	adf := timeseries.adf_test(x, 1)
	kpss := timeseries.kpss_test(x, 4)
	println('ADF  statistic=${adf.statistic:.3f}  p=${adf.p_value:.2f}  stationary=${adf.is_stationary}')
	println('KPSS statistic=${kpss.statistic:.3f}  p=${kpss.p_value:.2f}  stationary=${kpss.is_stationary}')

	// First-difference the series and re-test
	dx := timeseries.diff(x, 1)
	adf_d1 := timeseries.adf_test(dx, 1)
	println('ADF on Δx: statistic=${adf_d1.statistic:.3f}  stationary=${adf_d1.is_stationary}')

	// --- ACF and PACF ---
	println('\n--- ACF / PACF (first 5 lags) ---')
	acf_vals := timeseries.acf(x, 5)
	pacf_vals := timeseries.pacf(x, 5)
	bound := timeseries.acf_confidence_bound(x.len)
	println('95% confidence bound: ±${bound:.3f}')
	for k in 0 .. 5 {
		sig_acf := if math.abs(acf_vals[k]) > bound { '*' } else { ' ' }
		sig_pacf := if math.abs(pacf_vals[k]) > bound { '*' } else { ' ' }
		println('  lag ${k + 1}: ACF=${acf_vals[k]:6.3f}${sig_acf}  PACF=${pacf_vals[k]:6.3f}${sig_pacf}')
	}

	// --- Classical decomposition ---
	println('\n--- Classical Decomposition (additive, period=4) ---')
	dec := timeseries.decompose(x, 4, .additive)
	println('Trend  (t=10..14): ${dec.trend[10..15].map(fn (v f64) string { return '${v:.2f}' }).join(', ')}')
	println('Seasonal pattern:  ${dec.seasonal[0..4].map(fn (v f64) string { return '${v:.2f}' }).join(', ')}')

	// --- Manual ARIMA fit ---
	println('\n--- ARIMA(1,1,0) Fit ---')
	model := timeseries.arima_fit(x, 1, 1, 0)
	println('AR coefficient: ${model.ar_coeffs[0]:.4f}')
	println('Intercept:      ${model.intercept:.4f}')
	println('Sigma²:         ${model.sigma2:.4f}')
	println('AIC=${model.aic:.2f}  BIC=${model.bic:.2f}  AICc=${model.aicc:.2f}')

	// --- Forecast 8 steps ahead ---
	println('\n--- 8-Step Forecast (95% CI) ---')
	fc := timeseries.arima_forecast(model, 8, 0.05, x)
	for i in 0 .. 8 {
		println('  t+${i + 1}: ${fc.forecast[i]:.2f}  [${fc.lower[i]:.2f}, ${fc.upper[i]:.2f}]')
	}

	// --- auto_arima ---
	println('\n--- auto_arima (max_p=3, max_q=2, max_d=2) ---')
	best := timeseries.auto_arima(x, 3, 2, 2)
	println('Best order: ARIMA(${best.p}, ${best.d}, ${best.q})')
	println('AICc: ${best.aicc:.2f}')
	println(timeseries.arima_summary(best))
}
