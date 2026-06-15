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

	_ = math.pi // suppress unused import until later tasks use it
	println('done — wrote SVGs to ${out}')
}
