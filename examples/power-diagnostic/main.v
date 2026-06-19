// Scenario: Post-hoc Power Analysis on a Null Result
// Demonstrates: power_diagnostic — was the experiment underpowered?
// Python equivalent: statsmodels TTestPower / pingouin compute_effsize
module main

import vstats.experiment

fn main() {
	println('=== Post-hoc Power Diagnostic ===\n')

	// Small groups: typical of a rushed experiment
	ctrl := [10.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3, 10.0]
	trt  := [10.4, 10.9, 10.2, 10.6, 10.5, 10.3, 10.7, 10.4]

	result := experiment.abtest(ctrl, trt)
	println('A/B Test result:')
	println('  Control mean:    ${result.control_mean:.3f}')
	println('  Treatment mean:  ${result.treatment_mean:.3f}')
	println('  Relative lift:   ${result.relative_lift * 100:.1f}%')
	println('  P-value:         ${result.p_value:.4f}')
	println('  Significant:     ${result.significant}')
	println('  Effect size (d): ${result.effect_size:.3f}')

	diag := experiment.power_diagnostic(result, 0.05, 0.80)
	println('\nPower diagnostic (target power = 80%):')
	println('  N control:          ${diag.n_ctrl}')
	println('  N treatment:        ${diag.n_trt}')
	println('  Observed power:     ${diag.observed_power * 100:.1f}%')
	println('  MDE (Cohen\'s d):    ${diag.mde_effect_size:.3f}')
	println('  Underpowered:       ${diag.underpowered}')

	if diag.underpowered {
		println('\nConclusion: null result is likely due to insufficient sample size.')
		println('The experiment could only reliably detect effects >= d=${diag.mde_effect_size:.2f}.')
		println('To detect d=${result.effect_size:.2f} with 80% power, collect more data.')
	} else {
		println('\nConclusion: experiment was adequately powered. Null result is credible.')
	}
}
