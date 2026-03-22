module main

import experiment
import math

fn test__sprt_reject_null() {
	// Control: 50/1000 (rate=0.05). Treatment: 80/1000 (rate=0.08).
	// mde=0.02 means alternative is p0+0.02=0.07. Since 0.08 > 0.07 with enough data, should reject.
	// LLR = 80*log(0.07/0.05) + 920*log(0.93/0.95) ≈ 7.3 > upper_boundary ≈ 2.77
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 80, 1000, cfg)

	assert result.decision == .reject_null
	assert result.log_likelihood_ratio > result.upper_boundary
	assert math.abs(result.upper_boundary - math.log(0.8 / 0.05)) < 0.01
	assert math.abs(result.lower_boundary - math.log(0.2 / 0.95)) < 0.01
	assert result.rate_a == 0.05
	assert result.n_a == 1000
	assert result.n_b == 1000
}

fn test__sprt_accept_null() {
	// Control: 50/1000 (rate=0.05). Treatment: 40/1000 (rate=0.04).
	// Effect is in the wrong direction — LLR should fall below lower boundary.
	// LLR = 40*log(0.07/0.05) + 960*log(0.93/0.95) ≈ -7.0 < lower_boundary ≈ -1.56
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 40, 1000, cfg)

	assert result.decision == .accept_null
	assert result.log_likelihood_ratio < result.lower_boundary
}

fn test__sprt_continue_testing() {
	// Control: 50/1000 (rate=0.05). Treatment: 60/1000 (rate=0.06).
	// Small positive effect — not enough evidence yet.
	// LLR ≈ 0.19, between boundaries (-1.56, 2.77)
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 60, 1000, cfg)

	assert result.decision == .continue_testing
	assert result.log_likelihood_ratio > result.lower_boundary
	assert result.log_likelihood_ratio < result.upper_boundary
}

fn test__sprt_boundaries_correct() {
	// Upper = log((1-beta)/alpha) = log(0.8/0.05) = log(16) ≈ 2.773
	// Lower = log(beta/(1-alpha)) = log(0.2/0.95) ≈ -1.558
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 60, 1000, cfg)

	assert math.abs(result.upper_boundary - math.log(16.0)) < 0.01
	assert math.abs(result.lower_boundary - math.log(0.2 / 0.95)) < 0.01
}
