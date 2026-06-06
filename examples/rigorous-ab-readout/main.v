// Scenario: Rigorous A/B Test Readout
// Demonstrates: vstats.experiment + vstats.stats
// Python equivalent: scipy.stats + statsmodels + custom SRM + multipletests = 5 imports, ~80 lines
module main

import vstats.experiment
import vstats.stats

fn main() {
	println('=== Rigorous A/B Test Readout ===\n')

	// --- Setup: checkout-flow experiment, 20 users per arm ---
	// Revenue per user (0 = non-purchaser). Pre-period used for CUPED.
	pre_ctrl := [40.0, 0, 0, 50.0, 0, 45.0, 0, 0, 0, 48.0,
	             42.0, 0, 0, 0, 49.0, 0, 44.0, 0, 0, 47.0]
	pre_trt  := [43.0, 0, 53.0, 0, 48.0, 0, 0, 45.0, 0, 51.0,
	             44.0, 0, 0, 54.0, 0, 49.0, 0, 0, 0, 50.0]
	ctrl     := [42.0, 0, 0, 55.0, 0, 48.0, 0, 0, 0, 51.0,
	             44.0, 0, 0, 0, 52.0, 0, 47.0, 0, 0, 49.0]
	trt      := [45.0, 0, 56.0, 0, 50.0, 0, 0, 48.0, 0, 53.0,
	             46.0, 0, 0, 57.0, 0, 51.0, 0, 0, 0, 52.0]

	// --- Core analysis ---

	// 1. SRM: verify assignment matches expected 50/50 split
	srm := experiment.srm_test(ctrl.len, trt.len, 0.5, 0.05)
	println('1. SRM check')
	println('   chi2=${srm.chi2_statistic:.4f}  p=${srm.p_value:.4f}  detected=${srm.srm_detected}')

	// 2. Winsorize to cap heavy tails before analysis
	ctrl_w := stats.winsorize(ctrl, 0.05, 0.95)
	trt_w  := stats.winsorize(trt, 0.05, 0.95)

	// 3. CUPED: use pre-period data to reduce variance
	cuped := experiment.cuped_test(ctrl_w, trt_w, pre_ctrl, pre_trt)
	println('\n2. CUPED')
	println('   theta=${cuped.theta:.4f}  variance_reduction=${cuped.variance_reduction * 100:.1f}%')
	result := cuped.adjusted_result

	// 4. BH correction across three metrics tested simultaneously
	p_revenue  := result.p_value
	p_sessions := 0.031   // second metric tested in same experiment
	p_bounce   := 0.210   // third metric
	bh := stats.bh_correction([p_revenue, p_sessions, p_bounce], 0.05)
	println('\n3. Multiple testing (BH, 3 metrics)')
	println('   revenue  adj_p=${bh.adjusted[0]:.4f}  reject=${bh.reject[0]}')
	println('   sessions adj_p=${bh.adjusted[1]:.4f}  reject=${bh.reject[1]}')
	println('   bounce   adj_p=${bh.adjusted[2]:.4f}  reject=${bh.reject[2]}')
	println('   total rejected: ${bh.n_rejected}/3')

	// --- Interpret output ---
	println('\n4. Verdict')
	println('   ' + experiment.null_verdict(result, 0.05))
}
