// Scenario: Bayesian Revenue A/B Test with ROPE
// Demonstrates: bayesian_continuous_ab_test — P(trt > ctrl), expected loss, ROPE
// Python equivalent: PyMC Normal model with posterior sampling
module main

import vstats.experiment

fn main() {
	println('=== Bayesian Revenue A/B Test ===\n')

	// Revenue per user (USD). Control ≈ $10, Treatment ≈ $12.
	ctrl_rev := [9.5, 10.2, 8.8, 11.0, 9.7, 10.5, 9.1, 10.8, 9.4, 10.1,
	             10.3, 9.6, 11.2, 8.9, 10.7, 9.3, 10.4, 9.8, 10.6, 9.9]
	trt_rev  := [11.5, 12.2, 10.8, 13.0, 11.7, 12.5, 11.1, 12.8, 11.4, 12.1,
	             12.3, 11.6, 13.2, 10.9, 12.7, 11.3, 12.4, 11.8, 12.6, 11.9]

	result := experiment.bayesian_continuous_ab_test(ctrl_rev, trt_rev)

	println('Posterior estimates:')
	println('  Control:   mean = \$${result.posterior_mean_ctrl:.2f}  std = \$${result.posterior_std_ctrl:.3f}')
	println('  Treatment: mean = \$${result.posterior_mean_trt:.2f}  std = \$${result.posterior_std_trt:.3f}')

	println('\nDecision metrics:')
	println('  P(treatment > control):       ${result.prob_trt_beats_ctrl:.3f}')
	println('  Expected loss (deploy ctrl):  \$${result.expected_loss_ctrl:.3f}')
	println('  Expected loss (deploy trt):   \$${result.expected_loss_trt:.3f}')

	println('\n95% Credible intervals:')
	println('  Control:   [\$${result.ci_lower_ctrl:.2f}, \$${result.ci_upper_ctrl:.2f}]')
	println('  Treatment: [\$${result.ci_lower_trt:.2f}, \$${result.ci_upper_trt:.2f}]')

	// ROPE: effects smaller than $0.50 are commercially negligible
	result_rope := experiment.bayesian_continuous_ab_test(ctrl_rev, trt_rev,
		experiment.BayesianContinuousConfig{ rope_lower: -0.50, rope_upper: 0.50 })
	println('\nWith ROPE [-\$0.50, +\$0.50] (practically negligible threshold):')
	println('  P(effect negligible): ${result_rope.prob_rope:.3f}')
	println('  P(effect meaningful): ${1.0 - result_rope.prob_rope:.3f}')

	recommendation := if result.prob_trt_beats_ctrl > 0.95 {
		'Strong evidence for treatment. Ship it.'
	} else if result.prob_trt_beats_ctrl > 0.80 {
		'Moderate evidence for treatment. Consider collecting more data.'
	} else {
		'Insufficient evidence. Do not ship.'
	}
	println('\nRecommendation: ${recommendation}')
}
