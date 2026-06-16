// Scenario: Multivariate VAR Analysis and Granger Causality
// Demonstrates: vstats.timeseries — VAR fitting, lag selection, Granger causality, IRF
// Python equivalent: statsmodels.tsa.vector_ar.var_model.VAR + grangercausalitytests
module main

import vstats.timeseries
import math

fn main() {
	println('=== VAR Model: Granger Causality and Impulse Response ===\n')

	// --- Synthetic bivariate VAR(1) ---
	// True model:
	//   y1[t] = 0.5*y1[t-1] + 0.4*y2[t-1] + noise   ← y2 causes y1
	//   y2[t] = 0.0*y1[t-1] + 0.6*y2[t-1] + noise   ← y1 does NOT cause y2
	n := 200
	mut data := [][]f64{len: 2, init: []f64{len: n}}
	data[0][0] = 1.0
	data[1][0] = 1.0
	for t in 1 .. n {
		e1 := (f64(t % 13) - 6.0) * 0.15
		e2 := (f64(t % 7) - 3.0) * 0.15
		data[0][t] = 0.5 * data[0][t - 1] + 0.4 * data[1][t - 1] + e1
		data[1][t] = 0.0 * data[0][t - 1] + 0.6 * data[1][t - 1] + e2
	}

	println('Series: 200 observations, 2 variables')
	println('True model: y1[t] = 0.5·y1[t-1] + 0.4·y2[t-1]  (y2 → y1)')
	println('            y2[t] = 0.6·y2[t-1]                  (y1 ↛ y2)\n')

	// --- Lag selection ---
	println('--- Lag Selection (max_p=4) ---')
	sel := timeseries.var_select_lag(data, 4)
	println('AIC suggests p=${sel.aic_order}')
	println('BIC suggests p=${sel.bic_order}')
	println('HQC suggests p=${sel.hqc_order}')
	println('\nFull criteria table (AIC | BIC | HQC):')
	for i, row in sel.criteria {
		println('  p=${i + 1}:  ${row[0]:8.2f}  ${row[1]:8.2f}  ${row[2]:8.2f}')
	}

	// --- Fit VAR(1) ---
	println('\n--- VAR(1) Fit ---')
	model := timeseries.var_fit(data, 1)
	println('k=${model.k}  p=${model.p}  AIC=${model.aic:.2f}  BIC=${model.bic:.2f}')
	println('\nEquation y1:  intercept=${model.coeff_matrices[0][0]:.4f}  φ(y1)=${model.coeff_matrices[0][1]:.4f}  φ(y2)=${model.coeff_matrices[0][2]:.4f}')
	println('Equation y2:  intercept=${model.coeff_matrices[1][0]:.4f}  φ(y1)=${model.coeff_matrices[1][1]:.4f}  φ(y2)=${model.coeff_matrices[1][2]:.4f}')
	println('(True:        y1: 0.0, 0.5, 0.4   y2: 0.0, 0.0, 0.6)')

	// --- Granger causality ---
	println('\n--- Granger Causality Tests (p=1) ---')
	g_y2_to_y1 := timeseries.granger_causality(data, 1, 1, 0)
	g_y1_to_y2 := timeseries.granger_causality(data, 1, 0, 1)
	println('y2 → y1:  F=${g_y2_to_y1.f_statistic:.3f}  p=${g_y2_to_y1.p_value:.4f}  significant=${g_y2_to_y1.significant}  ← expected: true')
	println('y1 → y2:  F=${g_y1_to_y2.f_statistic:.3f}  p=${g_y1_to_y2.p_value:.4f}  significant=${g_y1_to_y2.significant}  ← expected: false')

	// --- Forecast ---
	println('\n--- 5-Step Ahead Forecast ---')
	fc := timeseries.var_forecast(model, data, 5)
	println('Step  y1        y2')
	for i in 0 .. 5 {
		println('  +${i + 1}    ${fc[0][i]:7.3f}   ${fc[1][i]:7.3f}')
	}

	// --- Impulse Response Functions ---
	println('\n--- Orthogonalized IRF (10 periods) ---')
	irf_result := timeseries.irf(model, 10)
	println('Shock to y2 → response of y1 (should start positive and decay):')
	resp := irf_result.responses[1][0]
	for t in 0 .. 10 {
		bar_len := int(math.abs(resp[t]) * 20)
		bar := '█'.repeat(bar_len)
		sign := if resp[t] >= 0 { '+' } else { '-' }
		println('  t=${t:2d}  ${resp[t]:7.4f}  ${sign}${bar}')
	}

	println('\nShock to y1 → response of y2 (should be near zero):')
	resp2 := irf_result.responses[0][1]
	for t in 0 .. 5 {
		println('  t=${t:2d}  ${resp2[t]:7.4f}')
	}
}
