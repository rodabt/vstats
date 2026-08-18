// Scenario: Diagnostic Plots Gallery — ROC, Q-Q, ECDF, Pair Plot
// Demonstrates: vstats.chart (Grid, pair_plot), vstats.stats (ecdf), vstats.prob (qq_points),
//               vstats.utils (roc_curve), vstats.ml, vstats.linalg — composed multi-panel diagnostics
// Python equivalent: sklearn.metrics.RocCurveDisplay, scipy.stats.probplot,
//                     statsmodels.distributions.ECDF, seaborn.pairplot
module main

import os
import vstats.utils
import vstats.ml
import vstats.stats
import vstats.prob
import vstats.linalg
import vstats.chart

fn main() {
	println('=== Diagnostic Plots Gallery ===\n')
	out_dir := os.dir(@FILE)

	// --- ROC curve: Random Forest on Breast Cancer ---
	cancer := utils.load_breast_cancer()!
	train, test := cancer.train_test_split(0.2)
	x_train, y_train := train.xy()
	x_test, y_test := test.xy()
	rf := ml.random_forest_classifier(x_train, y_train, 20, 5)
	rf_proba := ml.random_forest_classifier_predict_proba(rf, x_test)
	roc := utils.roc_curve(y_test, rf_proba)
	println('Random Forest AUC: ${roc.auc:.4f}')

	chart.new(title: 'ROC Curve (AUC=${roc.auc:.3f})', width: 420, height: 360)
		.line(roc.fpr, roc.tpr, label: 'ROC')
		.line([0.0, 1.0], [0.0, 1.0], label: 'chance')
		.xlabel('False Positive Rate')
		.ylabel('True Positive Rate')
		.save(os.join_path(out_dir, 'roc_curve.svg'))!

	// --- ECDF + Q-Q of regression residuals: Boston Housing ---
	housing := utils.load_boston_housing()!
	model := ml.linear_regression(housing.features, housing.target)
	pred := ml.linear_predict(model, housing.features)
	mut resid := []f64{len: housing.target.len}
	for i in 0 .. housing.target.len {
		resid[i] = housing.target[i] - pred[i]
	}

	ecdf_x, ecdf_p := stats.ecdf(resid)
	chart.new(title: 'ECDF of Residuals', width: 420, height: 360)
		.step(ecdf_x, ecdf_p, label: 'ECDF')
		.xlabel('Residual')
		.ylabel('Cumulative probability')
		.save(os.join_path(out_dir, 'ecdf.svg'))!

	qq_theoretical, qq_sample := prob.qq_points(resid)
	chart.new(title: 'Normal Q-Q Plot of Residuals', width: 420, height: 360)
		.scatter(qq_theoretical, qq_sample, label: 'residuals')
		.xlabel('Theoretical quantile')
		.ylabel('Sample quantile')
		.save(os.join_path(out_dir, 'qq_plot.svg'))!

	// --- Pair plot: Iris features ---
	iris := utils.load_iris()!
	columns := linalg.transpose(iris.features) // one []f64 per feature
	grid := chart.pair_plot(columns, iris.feature_names, title: 'Iris Feature Pair Plot')
	grid.save(os.join_path(out_dir, 'pair_plot.svg'))!

	println('\nWrote roc_curve.svg, ecdf.svg, qq_plot.svg, pair_plot.svg to ${out_dir}')
}
