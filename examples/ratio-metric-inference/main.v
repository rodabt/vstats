// Scenario: Ratio Metric Inference (Revenue per Session)
// Demonstrates: vstats.stats — delta method + permutation bootstrap
// Python equivalent: manual linearization + scipy.stats.ttest_ind + scipy.stats.bootstrap
module main

import vstats.experiment
import vstats.stats
import rand

fn main() {
	println('=== Ratio Metric Inference: Revenue per Session ===\n')
	println('Problem: a naive t-test on revenue/session is biased because the')
	println('numerator (revenue) and denominator (sessions) are correlated per user.')
	println('The delta method linearizes the ratio before testing.\n')

	rand.seed([u32(42), u32(0)])

	// --- Setup: 10 users per arm ---
	// Control: ~$5/session. Treatment: ~$7/session (+$2 lift).
	revenue_ctrl  := [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 7.0, 13.0, 10.0, 14.0]
	sessions_ctrl := [2.0,  2.0,  2.0,  3.0,  2.0,  2.0,  1.0,  3.0,  2.0,  2.0]
	revenue_trt   := [14.0, 16.0, 12.0, 21.0, 13.0, 15.0, 7.0, 19.0, 14.0, 18.0]
	sessions_trt  := [2.0,  2.0,  2.0,  3.0,  2.0,  2.0,  1.0,  3.0,  2.0,  2.0]

	mut revenue   := []f64{}
	mut sessions  := []f64{}
	mut treatment := []int{}

	for v in revenue_ctrl  { revenue << v }
	for v in revenue_trt   { revenue << v }
	for v in sessions_ctrl { sessions << v }
	for v in sessions_trt  { sessions << v }
	for _ in revenue_ctrl  { treatment << 0 }
	for _ in revenue_trt   { treatment << 1 }

	// --- Core analysis ---

	// 1. Naive t-test on raw revenue (biased for ratio metrics)
	naive := experiment.abtest(revenue_ctrl, revenue_trt)
	println('1. Naive t-test on raw revenue (ignores session denominator)')
	println('   ctrl_mean=${naive.control_mean:.2f}  trt_mean=${naive.treatment_mean:.2f}  p=${naive.p_value:.4f}')

	// 2. Delta method (correct approach for ratio metrics)
	dm := stats.delta_method_ratio(revenue, sessions, treatment)
	println('\n2. Delta method (correct for revenue/session)')
	println('   ctrl_ratio=${dm.ratio_ctrl:.4f}  trt_ratio=${dm.ratio_trt:.4f}')
	println('   effect=${dm.effect:.4f}  SE=${dm.se:.4f}')
	println('   t=${dm.t_statistic:.4f}  p=${dm.p_value:.4f}')
	println('   95% CI: [${dm.ci_lower:.4f}, ${dm.ci_upper:.4f}]')

	// 3. Permutation bootstrap on linearized residuals — non-parametric check
	r_global := stats.mean(revenue) / stats.mean(sessions)
	mut z_ctrl := []f64{}
	mut z_trt  := []f64{}
	for i in 0 .. revenue_ctrl.len {
		z_ctrl << revenue_ctrl[i] - r_global * sessions_ctrl[i]
		z_trt  << revenue_trt[i]  - r_global * sessions_trt[i]
	}
	boot := stats.bootstrap_test(z_ctrl, z_trt, 2000)
	println('\n3. Permutation bootstrap (non-parametric robustness check)')
	println('   observed_diff=${boot.observed_diff:.4f}  p=${boot.p_value:.4f}')
	println('   95% CI: [${boot.ci_lower:.4f}, ${boot.ci_upper:.4f}]')

	// --- Interpret output ---
	println('\n4. Conclusion')
	if dm.p_value < 0.05 {
		println('   Delta method detects significant lift: +${dm.effect:.2f} revenue/session.')
	} else {
		println('   No significant lift in revenue/session.')
	}
}
