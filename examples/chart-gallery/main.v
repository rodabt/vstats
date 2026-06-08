// Scenario: Chart Gallery — Regression Diagnostics
// Demonstrates: vstats.chart — scatter, line, bar, histogram, legend, guide line, theming
// Python equivalent: matplotlib regression diagnostics (scatter+fit, residuals, hist, coef bar)
module main

import os
import math
import vstats.utils
import vstats.ml
import vstats.chart

struct XY {
	cx f64
	cy f64
}

fn main() {
	println('=== Chart Gallery: Regression Diagnostics ===\n')

	// --- Setup: one dataset, two regressions ---
	dataset := utils.load_boston_housing()!
	x := dataset.features // [][]f64, 3 features
	y := dataset.target   // []f64, median house price
	names := dataset.feature_names
	println('Dataset: ${dataset.name} (${y.len} samples, ${names.len} features)')

	out_dir := os.dir(@FILE) // directory of this source file

	// Single-feature regression: price ~ Crime Rate (feature index 0)
	crime := x.map(it[0])
	x1 := crime.map([it]) // [][]f64 single-feature design matrix
	model := ml.linear_regression(x1, y)
	pred := ml.linear_predict(model, x1)
	rmse := ml.rmse(y, pred)
	println('Simple model (price ~ Crime Rate): intercept=${model.intercept:.2f}, slope=${model.coefficients[0]:.3f}, RMSE=${rmse:.2f}')

	// residuals
	mut resid := []f64{len: y.len}
	for i in 0 .. y.len {
		resid[i] = y[i] - pred[i]
	}

	// --- Chart 1: scatter (observed) + line (fit) ---
	// sort points by crime so the fitted line draws cleanly left-to-right
	mut pairs := []XY{}
	for i in 0 .. crime.len {
		pairs << XY{crime[i], pred[i]}
	}
	pairs.sort(a.cx < b.cx)
	xs := pairs.map(it.cx)
	ys := pairs.map(it.cy)
	chart.new(title: 'Price vs Crime Rate', width: 640, height: 420)
		.scatter(crime, y, label: 'observed')
		.line(xs, ys, label: 'fit')
		.xlabel('Crime Rate')
		.ylabel('Median House Price')
		.save(os.join_path(out_dir, 'regression_fit.svg'))!

	// --- Chart 2: residuals vs fitted, with a zero reference line ---
	chart.new(title: 'Residuals vs Fitted', width: 640, height: 420)
		.scatter(pred, resid, label: 'residual')
		.axhline(0.0)
		.xlabel('Fitted value')
		.ylabel('Residual')
		.save(os.join_path(out_dir, 'residuals_vs_fitted.svg'))!

	// --- Chart 3: histogram of residuals (auto bins) ---
	chart.new(title: 'Residual Distribution', width: 640, height: 420)
		.histogram(resid)
		.xlabel('Residual')
		.ylabel('Count')
		.save(os.join_path(out_dir, 'residuals_hist.svg'))!

	// --- Chart 4: coefficient bar from the full multivariate regression ---
	full := ml.linear_regression(x, y)
	coefs := full.coefficients
	mut top := 0
	for j in 1 .. coefs.len {
		if math.abs(coefs[j]) > math.abs(coefs[top]) {
			top = j
		}
	}
	println('Full model strongest predictor: ${names[top]} (coef=${coefs[top]:.3f})')

	custom := chart.Theme{
		background: '#f7f7f7'
		palette:    ['#756bb1']
	}
	chart.new(title: 'Regression Coefficients', width: 640, height: 420, theme: custom)
		.bar(coefs, label: 'coefficient')
		.xlabel('Feature index (0=Crime, 1=ResLand, 2=Distance)')
		.ylabel('Coefficient')
		.save(os.join_path(out_dir, 'coefficients.svg'))!

	println('\nwrote 4 charts to ${out_dir}')
}
