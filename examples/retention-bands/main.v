// Scenario: Cohort Retention with Uncertainty
// Demonstrates: vstats.chart + vstats.growth — line + CI band, area fill, per-series color
// Python equivalent: matplotlib fill_between for retention bands + per-cohort lines
module main

import os
import vstats.growth
import vstats.chart

fn main() {
	println('=== Cohort Retention with Uncertainty ===\n')

	names := ['Jan', 'Feb', 'Mar', 'Apr']
	sizes := [1000, 1200, 900, 1100]
	// retained counts per period (period 0 = signup month)
	retention_data := [
		[1000, 720, 560, 470, 410, 380],
		[1200, 900, 740, 620, 560, 510],
		[900, 590, 430, 350, 300, 270],
		[1100, 800, 650, 560, 500, 460],
	]
	ca := growth.create_cohort_analysis(names, sizes, retention_data)

	periods := ca.avg_retention.len
	mut xs := []f64{len: periods}
	for j in 0 .. periods {
		xs[j] = f64(j)
	}

	// cross-cohort spread band (min..max retention at each period)
	mut lo := []f64{len: periods}
	mut hi := []f64{len: periods}
	for j in 0 .. periods {
		mut mn := ca.retention_matrix[0][j]
		mut mx := ca.retention_matrix[0][j]
		for i in 0 .. ca.retention_matrix.len {
			v := ca.retention_matrix[i][j]
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		lo[j] = mn
		hi[j] = mx
	}

	out_dir := os.dir(@FILE)

	// Chart 1: average retention line + cross-cohort band + faint per-cohort lines
	mut c := chart.new(title: 'Cohort Retention', subtitle: 'Average with cross-cohort min/max band',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
	c = c.band(xs, lo, hi, label: 'min/max')
	for i in 0 .. ca.retention_matrix.len {
		c = c.line(xs, ca.retention_matrix[i], color: '#cccccc')
	}
	c = c.line(xs, ca.avg_retention, label: 'average', color: '#1f77b4')
	c.xlabel('Months since signup')
		.ylabel('Retention')
		.save(os.join_path(out_dir, 'retention_bands.svg'))!

	// Chart 2: average retention as an area fill
	chart.new(title: 'Average Retention', subtitle: 'Area under the average retention curve',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.area(xs, ca.avg_retention, label: 'avg')
		.line(xs, ca.avg_retention, color: '#1f77b4')
		.xlabel('Months since signup')
		.ylabel('Retention')
		.save(os.join_path(out_dir, 'retention_area.svg'))!

	println('Final average retention: ${ca.avg_retention[periods - 1] * 100:.1f}%')
	println('wrote 2 charts to ${out_dir}')
}
