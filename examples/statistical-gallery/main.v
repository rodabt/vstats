module main

import os
import math
import vstats.chart

fn main() {
	out := os.dir(@FILE)

	// ── step: empirical CDF ──────────────────────────────────────────────
	raw := [22.0, 25, 27, 28, 30, 31, 31, 33, 35, 36, 38, 39, 40, 42, 45,
		48, 50, 53, 57, 65]
	n := f64(raw.len)
	mut xs_cdf := raw.clone()
	xs_cdf.sort()
	mut ys_cdf := []f64{len: raw.len}
	for i in 0 .. raw.len {
		ys_cdf[i] = f64(i + 1) / n
	}
	chart.new(title: 'Empirical CDF', width: 640, height: 420,
		theme: chart.Theme{ grid: true })
		.step(xs_cdf, ys_cdf, label: 'CDF')
		.xlabel('Age')
		.ylabel('Cumulative probability')
		.save(os.join_path(out, 'step_cdf.svg'))!

	// ── step: survival curves ────────────────────────────────────────────
	ts := [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	control := [1.0, 0.95, 0.88, 0.79, 0.71, 0.63, 0.55, 0.48, 0.41, 0.35, 0.30]
	treatment := [1.0, 0.98, 0.94, 0.89, 0.83, 0.77, 0.71, 0.65, 0.59, 0.53, 0.47]
	chart.new(title: 'Survival Curves', subtitle: 'Kaplan-Meier estimate', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.step(ts, control, label: 'Control')
		.step(ts, treatment, label: 'Treatment')
		.axhline(0.5)
		.xlabel('Time (months)')
		.ylabel('Survival probability')
		.save(os.join_path(out, 'step_survival.svg'))!

	// ── box: distribution comparison ────────────────────────────────────
	a_data := [45.0, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60]
	b_data := [20.0, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80]
	c_data := [30.0, 32, 33, 34, 35, 35, 36, 37, 38, 42, 55, 68, 70]
	d_data := [40.0, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
	chart.new(title: 'Distribution Comparison', subtitle: 'Box plots — Q1/median/Q3, 1.5×IQR whiskers',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.box(a_data, label: 'A')
		.box(b_data, label: 'B')
		.box(c_data, label: 'C')
		.box(d_data, label: 'D')
		.ylabel('Value')
		.save(os.join_path(out, 'box_comparison.svg'))!

	// ── dot: Cleveland dot plot ──────────────────────────────────────────
	feat_labels := ['Age', 'Income', 'Education', 'Distance', 'Tenure', 'Score', 'Visits',
		'Days']
	feat_vals := [0.82, 0.74, 0.68, 0.61, 0.55, 0.49, 0.37, 0.22]
	chart.new(title: 'Feature Importance', subtitle: 'Cleveland dot plot', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.dot(feat_vals, labels: feat_labels)
		.xlabel('Importance score')
		.save(os.join_path(out, 'dot_ranking.svg'))!

	_ = math.pi // suppress unused import until later tasks use it
	println('done — wrote SVGs to ${out}')
}
