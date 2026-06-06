// Scenario: Causal Difference-in-Differences
// Demonstrates: vstats.experiment — DiD regression + parallel trends test
// Python equivalent: statsmodels OLS with interaction term + manual trend test
module main

import vstats.experiment

fn main() {
	println('=== Causal DiD: Policy Impact Analysis ===\n')
	println('Setting: two regions, 10 units each. Treatment region receives new')
	println('pricing in period 2. True DiD effect: +3.0 units.\n')

	// --- Setup ---
	// Pre-period data for parallel trends test
	y_treat_pre := [10.0, 9.9, 10.2, 10.1, 9.8, 10.3, 9.7, 10.2, 10.0, 9.9]
	y_ctrl_pre  := [9.8,  10.1, 10.3, 9.9, 10.2, 10.0, 9.7, 10.4, 10.1, 9.9]
	time_pre    := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	// Full panel: pre + post for both groups
	mut y     := []f64{}
	mut group := []int{}  // 0=control, 1=treated
	mut time  := []int{}  // 0=pre, 1=post

	for v in y_ctrl_pre  { y << v; group << 0; time << 0 }
	// Control post: common trend +2
	for v in [11.9, 12.1, 12.2, 11.8, 12.3, 12.0, 11.7, 12.4, 12.1, 11.9] {
		y << v; group << 0; time << 1
	}
	for v in y_treat_pre { y << v; group << 1; time << 0 }
	// Treated post: common trend +2 plus treatment effect +3 = +5 total
	for v in [14.8, 15.1, 15.3, 14.9, 15.2, 15.0, 14.7, 15.4, 15.1, 14.9] {
		y << v; group << 1; time << 1
	}

	// --- Core analysis ---
	cfg := experiment.DiDConfig{}

	// 1. Parallel trends test (pre-period only — assumption check)
	trends := experiment.test_parallel_trends(y_treat_pre, y_ctrl_pre, time_pre, cfg)
	println('1. Parallel trends test (pre-period assumption check)')
	println('   slope_treated=${trends.slope_treated:.4f}  slope_control=${trends.slope_control:.4f}')
	println('   difference p=${trends.p_value:.4f}  holds=${trends.parallel_trends_hold}')

	// 2. DiD regression (OLS with treatment × post interaction)
	did := experiment.did_regression(y, [][]f64{}, group, time, cfg)
	println('\n2. DiD regression')
	println('   effect=${did.did_coefficient:.4f}  (true: 3.0)')
	println('   SE=${did.did_se:.4f}  t=${did.did_t_stat:.4f}  p=${did.did_p_value:.4f}')
	println('   95% CI: [${did.did_ci_lower:.4f}, ${did.did_ci_upper:.4f}]')
	println('   R²=${did.r_squared:.4f}')

	// --- Interpret output ---
	println('\n3. Verdict')
	if did.did_p_value < 0.05 {
		println('   Significant causal effect: DiD = ${did.did_coefficient:.3f}')
	} else {
		println('   No significant causal effect detected.')
	}
}
