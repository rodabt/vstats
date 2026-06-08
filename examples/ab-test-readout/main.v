// Scenario: A/B Test Readout
// Demonstrates: vstats.chart + vstats.experiment — bar chart with error bars, value labels, theming
// Python equivalent: statsmodels proportions z-test + matplotlib bar with yerr
module main

import os
import math
import vstats.experiment
import vstats.chart

fn main() {
	println('=== A/B Test Readout ===\n')

	// 0/1 conversion outcomes: control 24/200 = 12%, treatment 36/200 = 18%
	mut control := []f64{len: 200, init: 0.0}
	for i in 0 .. 24 {
		control[i] = 1.0
	}
	mut treatment := []f64{len: 200, init: 0.0}
	for i in 0 .. 36 {
		treatment[i] = 1.0
	}

	res := experiment.abtest(control, treatment)
	ci_c := 1.96 * res.control_std / math.sqrt(f64(res.n_control))
	ci_t := 1.96 * res.treatment_std / math.sqrt(f64(res.n_treatment))

	println('Control:   ${res.control_mean * 100:.1f}%  (n=${res.n_control})')
	println('Treatment: ${res.treatment_mean * 100:.1f}%  (n=${res.n_treatment})')
	println('Lift: ${res.relative_lift * 100:.1f}%   p=${res.p_value:.4f}   significant=${res.significant}')

	verdict := if res.significant {
		'Treatment lifts conversion ${res.relative_lift * 100:.1f}% (p=${res.p_value:.3f}, significant)'
	} else {
		'No significant difference (p=${res.p_value:.3f})'
	}

	out_dir := os.dir(@FILE)
	rates := [res.control_mean, res.treatment_mean]
	labels := ['${res.control_mean * 100:.1f}%', '${res.treatment_mean * 100:.1f}%']

	chart.new(title: 'A/B Test: Conversion Rate', subtitle: verdict, width: 560, height: 380,
		theme: chart.Theme{ grid: true })
		.bar(rates, color: '#2ca02c', show_values: true, labels: labels, err: [ci_c, ci_t])
		.xlabel('Arm (0 = control, 1 = treatment)')
		.ylabel('Conversion rate')
		.save(os.join_path(out_dir, 'ab_test_readout.svg'))!

	println('\nwrote 1 chart to ${out_dir}')
}
