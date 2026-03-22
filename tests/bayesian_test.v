module main

import experiment
import math

fn test__bayesian_b_clearly_better() {
	// A: 10/100 (10%), B: 40/100 (40%) — B is much better
	result := experiment.bayesian_ab_test(10, 100, 40, 100)

	assert result.prob_b_beats_a > 0.99
	assert result.expected_loss_b < result.expected_loss_a
	assert result.posterior_mean_a < result.posterior_mean_b
	assert result.successes_a == 10
	assert result.successes_b == 40
	assert result.n_a == 100
	assert result.n_b == 100
}

fn test__bayesian_groups_equal() {
	// A: 50/100 (50%), B: 50/100 (50%) — roughly equal
	result := experiment.bayesian_ab_test(50, 100, 50, 100)

	// With equal data, prob_b_beats_a should be close to 0.5
	assert result.prob_b_beats_a > 0.35
	assert result.prob_b_beats_a < 0.65
	// Expected losses should be symmetric and small
	assert math.abs(result.expected_loss_a - result.expected_loss_b) < 0.05
}

fn test__bayesian_credible_intervals_ordered() {
	result := experiment.bayesian_ab_test(10, 100, 40, 100)

	assert result.ci_lower_a < result.posterior_mean_a
	assert result.ci_upper_a > result.posterior_mean_a
	assert result.ci_lower_b < result.posterior_mean_b
	assert result.ci_upper_b > result.posterior_mean_b
}

fn test__bayesian_posterior_means_correct() {
	// Posterior mean for Beta(alpha+s, beta+f) = (alpha+s) / (alpha+s + beta+f)
	// With uniform prior (alpha=1, beta=1):
	// A: 10/100 => posterior mean = (1+10)/(1+10+1+90) = 11/102 ≈ 0.1078
	// B: 40/100 => posterior mean = (1+40)/(1+40+1+60) = 41/102 ≈ 0.4020
	result := experiment.bayesian_ab_test(10, 100, 40, 100)

	assert math.abs(result.posterior_mean_a - 11.0 / 102.0) < 0.001
	assert math.abs(result.posterior_mean_b - 41.0 / 102.0) < 0.001
}

fn test__bayesian_informative_prior() {
	// With informative prior, posterior should be pulled toward prior
	cfg := experiment.BayesianConfig{ alpha_prior: 10.0, beta_prior: 90.0 }
	result := experiment.bayesian_ab_test(10, 100, 40, 100, cfg)

	// Prior says rate ≈ 10%, strong pull on A, less on B
	// A posterior should be pulled toward 10% (prior mean)
	// B posterior should still be above A
	assert result.posterior_mean_b > result.posterior_mean_a
}
