module main

import os
import vstats.chart

fn main() {
	println('=== Chart Types Expansion Demo ===\n')
	out_dir := os.dir(@FILE)

	// combo: bar (primary axis, revenue) + line (secondary axis, conversion rate)
	combo := chart.new(title: 'Revenue vs Conversion Rate', width: 500, height: 320)
		.xcategories(['Jan', 'Feb', 'Mar', 'Apr'])
		.bar([120.0, 150.0, 90.0, 200.0], label: 'Revenue')
		.line([0.0, 1.0, 2.0, 3.0], [0.12, 0.18, 0.09, 0.22], label: 'Conv. Rate', secondary_axis: true)
	combo.save(os.join_path(out_dir, 'combo_demo.svg'))!
	println('Wrote combo_demo.svg (bar + line, dual y-axis)')

	// waterfall: revenue bridge
	wf := chart.new(title: 'Revenue Bridge', width: 500, height: 320)
		.waterfall([1000.0, 250.0, -80.0, 150.0, -40.0, 1280.0], [
			'Start',
			'New Sales',
			'Churn',
			'Upsell',
			'Refunds',
			'End',
		], show_values: true)
	wf.save(os.join_path(out_dir, 'waterfall_demo.svg'))!
	println('Wrote waterfall_demo.svg (cumulative build-up)')

	// grouped bar: quarterly comparison across regions
	gb := chart.new(title: 'Sales by Region', width: 500, height: 320)
		.xcategories(['Q1', 'Q2', 'Q3', 'Q4'])
		.grouped_bar([[30.0, 45.0], [40.0, 38.0], [35.0, 50.0], [50.0, 42.0]], labels: [
			'Q1',
			'Q2',
			'Q3',
			'Q4',
		], colors: ['#1f77b4', '#ff7f0e'])
	gb.save(os.join_path(out_dir, 'grouped_bar_demo.svg'))!
	println('Wrote grouped_bar_demo.svg (side-by-side clusters)')

	// dumbbell: before/after intervention
	db := chart.new(title: 'Latency Before/After Optimization (ms)', width: 500, height: 320)
		.dumbbell([320.0, 410.0, 275.0, 500.0], [180.0, 260.0, 190.0, 300.0], [
			'API A',
			'API B',
			'API C',
			'API D',
		], show_values: true)
	db.save(os.join_path(out_dir, 'dumbbell_demo.svg'))!
	println('Wrote dumbbell_demo.svg (paired before/after)')

	println('\nAll 4 demo charts written to ${out_dir}')
}
