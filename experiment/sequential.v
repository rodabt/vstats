module experiment

import math
import stats

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

@[params]
pub struct MSPRTConfig {
pub:
	alpha           f64 = 0.05
	beta            f64 = 0.20
	tau_sigma_ratio f64 = 1.0
	sigma           f64 = 0.0
}

pub struct MSPRTResult {
pub:
	log_mixture_ratio f64
	decision          SPRTDecision
	upper_boundary    f64
	lower_boundary    f64
	control_mean      f64
	treatment_mean    f64
	effect            f64
	sigma_hat         f64
	n_control         int
	n_treatment       int
}

// msprt_test applies the mixture Sequential Probability Ratio Test (mSPRT) to
// continuous outcome data. Like sprt_test, it is stateless — call it repeatedly
// as cumulative data accumulates; it never inflates the false positive rate.
//
// The mixture likelihood ratio (MLR) is the Bayes factor between a normal-mixture
// H₁ (effect ~ N(0, τ²)) and the point null H₀: no effect. τ = tau_sigma_ratio × σ̂.
//
// If cfg.sigma > 0, it is used as σ directly. Otherwise σ is estimated from the
// pooled within-group sample variance of ctrl and trt.
pub fn msprt_test(ctrl []f64, trt []f64, cfg MSPRTConfig) MSPRTResult {
	assert ctrl.len >= 2 && trt.len >= 2, 'each group needs at least 2 observations'

	n_c := ctrl.len
	n_t := trt.len
	m_c := stats.mean(ctrl)
	m_t := stats.mean(trt)

	sigma_hat := if cfg.sigma > 0.0 {
		cfg.sigma
	} else {
		mut ss_c := 0.0
		mut ss_t := 0.0
		for x in ctrl {
			ss_c += (x - m_c) * (x - m_c)
		}
		for y in trt {
			ss_t += (y - m_t) * (y - m_t)
		}
		pooled_var := (ss_c + ss_t) / f64(n_c + n_t - 2)
		math.sqrt(pooled_var)
	}

	sigma2 := sigma_hat * sigma_hat
	tau2 := cfg.tau_sigma_ratio * sigma_hat * cfg.tau_sigma_ratio * sigma_hat
	d := m_t - m_c
	s2 := sigma2 * (1.0 / f64(n_c) + 1.0 / f64(n_t))

	upper := math.log((1.0 - cfg.beta) / cfg.alpha)
	lower := math.log(cfg.beta / (1.0 - cfg.alpha))

	log_mlr := if s2 == 0.0 {
		0.0
	} else {
		0.5 * math.log(s2 / (s2 + tau2)) + d * d * tau2 / (2.0 * s2 * (s2 + tau2))
	}

	decision := if log_mlr >= upper {
		SPRTDecision.reject_null
	} else if log_mlr <= lower {
		SPRTDecision.accept_null
	} else {
		SPRTDecision.continue_testing
	}

	return MSPRTResult{
		log_mixture_ratio: log_mlr
		decision:          decision
		upper_boundary:    upper
		lower_boundary:    lower
		control_mean:      m_c
		treatment_mean:    m_t
		effect:            d
		sigma_hat:         sigma_hat
		n_control:         n_c
		n_treatment:       n_t
	}
}
