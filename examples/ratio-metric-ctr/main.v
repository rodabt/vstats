// Scenario: CTR A/B Test with Delta Method
// Demonstrates: ratio_metric_test — correct variance estimation for click-through rates
// Python equivalent: statsmodels MeanDiff via delta method / linearization
module main

import vstats.experiment

fn main() {
	println('=== CTR A/B Test (Delta Method) ===\n')

	// 30 users per arm. Each user: (clicks, impressions).
	// Control: alternating 1 and 2 clicks per 10 impressions → CTR ≈ 15%
	// Treatment: alternating 3 and 4 clicks per 10 impressions → CTR ≈ 35%
	mut clicks_ctrl := []f64{}
	mut imps_ctrl   := []f64{}
	mut clicks_trt  := []f64{}
	mut imps_trt    := []f64{}

	for i in 0 .. 30 {
		clicks_ctrl << if i % 2 == 0 { 1.0 } else { 2.0 }
		imps_ctrl   << 10.0
		clicks_trt  << if i % 2 == 0 { 3.0 } else { 4.0 }
		imps_trt    << 10.0
	}

	result := experiment.ratio_metric_test(clicks_ctrl, imps_ctrl, clicks_trt, imps_trt)

	println('Control CTR:    ${result.ratio_ctrl * 100:.2f}%  (n=${result.n_ctrl})')
	println('Treatment CTR:  ${result.ratio_trt * 100:.2f}%  (n=${result.n_trt})')
	println('Absolute diff:  ${result.diff * 100:.2f}pp')
	println('Relative lift:  ${result.relative_lift * 100:.1f}%')
	println('SE (delta):     ${result.se:.5f}')
	println('Z-statistic:    ${result.z_statistic:.3f}')
	println('P-value:        ${result.p_value:.4f}')
	println('Significant:    ${result.significant}')
	println('95% CI:         [${result.ci_lower * 100:.2f}pp, ${result.ci_upper * 100:.2f}pp]')
}
