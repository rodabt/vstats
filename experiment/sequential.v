module experiment

import math

pub enum SPRTDecision {
	continue_testing
	reject_null
	accept_null
}

// SPRTConfig configures the Sequential Probability Ratio Test.
// Note: NOT @[params] because mde has no sensible default — callers must
// set it explicitly: SPRTConfig{ mde: 0.02 }
pub struct SPRTConfig {
pub:
	alpha f64 = 0.05
	beta  f64 = 0.20
	mde   f64
}

pub struct SPRTResult {
pub:
	log_likelihood_ratio f64
	decision             SPRTDecision
	upper_boundary       f64
	lower_boundary       f64
	rate_a               f64
	rate_b               f64
	n_a                  int
	n_b                  int
}

// sprt_test applies the Sequential Probability Ratio Test (SPRT) to accumulated
// experiment data. Call this function repeatedly as data accumulates — it is
// stateless and works on cumulative totals.
//
// The SPRT solves the "peeking problem": it allows you to check results at any
// time without inflating the false positive rate, by using Wald's likelihood
// ratio boundaries.
//
// successes_a/n_a: cumulative events and observations for control
// successes_b/n_b: cumulative events and observations for treatment
// cfg.mde:         minimum detectable effect (absolute rate difference, must be > 0)
pub fn sprt_test(successes_a int, n_a int, successes_b int, n_b int, cfg SPRTConfig) SPRTResult {
	assert cfg.mde > 0, 'mde must be positive'
	assert n_a > 0 && n_b > 0, 'group sizes must be positive'
	assert successes_a >= 0 && successes_a <= n_a, 'successes_a out of range'
	assert successes_b >= 0 && successes_b <= n_b, 'successes_b out of range'

	// Wald boundaries
	upper := math.log((1.0 - cfg.beta) / cfg.alpha)
	lower := math.log(cfg.beta / (1.0 - cfg.alpha))

	// H0: p_b = p0 (observed control rate)
	// H1: p_b = p1 = p0 + mde
	p0 := f64(successes_a) / f64(n_a)
	p1 := p0 + cfg.mde

	// Clamp to avoid log(0) — rates must be in (0, 1)
	p0c := math.min(math.max(p0, 1e-10), 1.0 - 1e-10)
	p1c := math.min(math.max(p1, 1e-10), 1.0 - 1e-10)

	sb := f64(successes_b)
	fb := f64(n_b - successes_b)
	llr := sb * math.log(p1c / p0c) + fb * math.log((1.0 - p1c) / (1.0 - p0c))

	decision := if llr >= upper {
		SPRTDecision.reject_null
	} else if llr <= lower {
		SPRTDecision.accept_null
	} else {
		SPRTDecision.continue_testing
	}

	return SPRTResult{
		log_likelihood_ratio: llr
		decision:             decision
		upper_boundary:       upper
		lower_boundary:       lower
		rate_a:               p0
		rate_b:               f64(successes_b) / f64(n_b)
		n_a:                  n_a
		n_b:                  n_b
	}
}
