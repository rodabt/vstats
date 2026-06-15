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
		.dot(feat_vals, labels: feat_labels, show_values: true)
		.xlabel('Importance score')
		.save(os.join_path(out, 'dot_ranking.svg'))!

	// ── violin: distribution shape ───────────────────────────────────────
	chart.new(title: 'Distribution Shape', subtitle: 'Violin plots — same groups as box',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.violin(a_data, label: 'A')
		.violin(b_data, label: 'B')
		.violin(c_data, label: 'C')
		.violin(d_data, label: 'D')
		.ylabel('Value')
		.save(os.join_path(out, 'violin_distribution.svg'))!

	// ── hbar: horizontal bar ─────────────────────────────────────────────
	hbar_labels := ['Control', 'Variant A', 'Variant B', 'Variant C', 'Variant D', 'Variant E']
	hbar_vals := [12.4, 14.1, 13.8, 16.2, 11.9, 15.5]
	chart.new(title: 'Conversion Rate by Variant', subtitle: 'Horizontal bar chart', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.hbar(hbar_vals, labels: hbar_labels, show_values: true)
		.xlabel('Conversion rate (%)')
		.save(os.join_path(out, 'hbar_comparison.svg'))!

	// ── heatmap: correlation matrix ──────────────────────────────────────
	heat_labels := ['A', 'B', 'C', 'D', 'E']
	corr := [
		[1.0, 0.8, 0.3, -0.2, 0.1],
		[0.8, 1.0, 0.4, -0.1, 0.2],
		[0.3, 0.4, 1.0, 0.5, -0.3],
		[-0.2, -0.1, 0.5, 1.0, 0.6],
		[0.1, 0.2, -0.3, 0.6, 1.0],
	]
	chart.new(title: 'Correlation Matrix', width: 500, height: 500)
		.heatmap(corr, row_labels: heat_labels, col_labels: heat_labels,
		color_lo: '#d73027', color_hi: '#4575b4')
		.save(os.join_path(out, 'heatmap_correlation.svg'))!

	// ── stacked_bar: revenue by segment ──────────────────────────────────
	quarters := ['Q1', 'Q2', 'Q3', 'Q4']
	stack_groups := [
		[120.0, 85.0, 60.0],
		[145.0, 90.0, 75.0],
		[110.0, 95.0, 80.0],
		[160.0, 100.0, 90.0],
	]
	chart.new(title: 'Revenue by Segment', subtitle: 'Stacked bar — Products A, B, C',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.stacked_bar(stack_groups, labels: quarters, colors: ['#e41a1c', '#377eb8', '#4daf4a'])
		.ylabel('Revenue (\$k)')
		.save(os.join_path(out, 'stacked_bar.svg'))!

	// ── line + band: CI ribbon ───────────────────────────────────────────
	xs_ci := [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	ys_ci := [2.1, 2.8, 3.9, 4.5, 5.2, 5.8, 6.1, 6.8, 7.2, 7.9, 8.3, 8.9]
	lo_ci := [1.6, 2.1, 3.0, 3.5, 4.0, 4.6, 4.8, 5.4, 5.7, 6.3, 6.6, 7.1]
	hi_ci := [2.6, 3.5, 4.8, 5.5, 6.4, 7.0, 7.4, 8.2, 8.7, 9.5, 10.0, 10.7]
	chart.new(title: 'Monthly Trend with 95% CI', width: 640, height: 420,
		theme: chart.Theme{ grid: true })
		.band(xs_ci, lo_ci, hi_ci, label: '95% CI')
		.line(xs_ci, ys_ci, label: 'mean')
		.xlabel('Month')
		.ylabel('Value')
		.save(os.join_path(out, 'line_ci.svg'))!

	// ── forest plot (composed: scatter + err + vline) ────────────────────
	study_y := [5.0, 4, 3, 2, 1, 0]
	effects := [0.32, 0.18, 0.45, -0.08, 0.28, 0.27]
	errs := [0.12, 0.18, 0.22, 0.15, 0.10, 0.06]
	chart.new(title: 'Forest Plot', subtitle: 'Effect sizes with 95% CI', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.scatter(effects, study_y, err: errs)
		.axvline(0.0)
		.xlabel('Effect size')
		.save(os.join_path(out, 'forest_plot.svg'))!

	// ── Q-Q plot (composed: scatter + reference line) ────────────────────
	sample_sorted := [-2.1, -1.7, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.1,
		0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.6, 1.9, 2.3]
	theory := [-1.87, -1.40, -1.13, -0.93, -0.76, -0.60, -0.45, -0.32, -0.18, -0.06,
		0.06, 0.18, 0.32, 0.45, 0.60, 0.76, 0.93, 1.13, 1.40, 1.87]
	qq_ref_x := [-2.0, 2.0]
	qq_ref_y := [-2.0, 2.0]
	chart.new(title: 'Normal Q-Q Plot', width: 480, height: 480,
		theme: chart.Theme{ grid: true })
		.line(qq_ref_x, qq_ref_y, color: '#cccccc', label: 'reference')
		.scatter(theory, sample_sorted, label: 'sample')
		.xlabel('Theoretical quantiles')
		.ylabel('Sample quantiles')
		.save(os.join_path(out, 'qq_normal.svg'))!

	// ── KDE density overlay on histogram (composed) ──────────────────────
	kde_data := [1.2, 1.5, 1.8, 2.0, 2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
		3.1, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.0, 4.1, 4.3, 4.5, 4.8, 5.0,
		5.3, 5.8]
	kde_n := f64(kde_data.len)
	mut kde_sum := 0.0
	for v in kde_data {
		kde_sum += v
	}
	kde_mean := kde_sum / kde_n
	mut kde_sq := 0.0
	for v in kde_data {
		kde_sq += (v - kde_mean) * (v - kde_mean)
	}
	kde_sigma := math.sqrt(kde_sq / (kde_n - 1.0))
	kde_h := 1.06 * kde_sigma * math.pow(kde_n, -0.2)
	kde_lo := kde_data[0] - 2.0 * kde_h
	kde_hi := kde_data[kde_data.len - 1] + 2.0 * kde_h
	n_kde := 60
	mut kde_xs := []f64{len: n_kde}
	mut kde_ys := []f64{len: n_kde}
	for i in 0 .. n_kde {
		gv := kde_lo + f64(i) * (kde_hi - kde_lo) / f64(n_kde - 1)
		kde_xs[i] = gv
		mut d := 0.0
		for x in kde_data {
			u := (gv - x) / kde_h
			d += math.exp(-0.5 * u * u)
		}
		kde_ys[i] = d / (kde_n * kde_h * math.sqrt(2.0 * math.pi))
	}
	bins := chart.histogram_bins(kde_data, 0)
	bin_w := bins.edges[1] - bins.edges[0]
	for i in 0 .. n_kde {
		kde_ys[i] *= kde_n * bin_w
	}
	chart.new(title: 'KDE Density Overlay', subtitle: 'Histogram with Gaussian kernel density',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.histogram(kde_data, label: 'data')
		.line(kde_xs, kde_ys, label: 'KDE')
		.xlabel('Value')
		.ylabel('Count')
		.save(os.join_path(out, 'kde_density.svg'))!

	println('done — wrote 12 SVGs to ${out}')
}
