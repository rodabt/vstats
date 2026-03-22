module experiment

import math
import rand
import stats

@[params]
pub struct BayesianConfig {
pub:
	alpha_prior f64 = 1.0
	beta_prior  f64 = 1.0
	n_samples   int = 10000
}

pub struct BayesianResult {
pub:
	posterior_mean_a f64
	posterior_mean_b f64
	prob_b_beats_a   f64
	expected_loss_a  f64
	expected_loss_b  f64
	ci_lower_a       f64
	ci_upper_a       f64
	ci_lower_b       f64
	ci_upper_b       f64
	successes_a      int
	successes_b      int
	n_a              int
	n_b              int
}

// box_muller returns a standard normal sample using the Box-Muller transform.
fn box_muller() f64 {
	mut u1 := rand.f64()
	if u1 <= 0.0 {
		u1 = 1e-15
	}
	u2 := rand.f64()
	return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
}

// gamma_sample returns a sample from Gamma(shape, 1) using Marsaglia-Tsang (2000).
// Works for shape > 0.
fn gamma_sample(shape f64) f64 {
	if shape < 1.0 {
		u := rand.f64()
		if u <= 0.0 {
			return 0.0
		}
		return gamma_sample(1.0 + shape) * math.pow(u, 1.0 / shape)
	}
	d := shape - 1.0 / 3.0
	c := 1.0 / math.sqrt(9.0 * d)
	for {
		mut x := f64(0)
		mut v := f64(0)
		for {
			x = box_muller()
			v = 1.0 + c * x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := rand.f64()
		x2 := x * x
		if u < 1.0 - 0.0331 * x2 * x2 {
			return d * v
		}
		if math.log(u) < 0.5 * x2 + d * (1.0 - v + math.log(v)) {
			return d * v
		}
	}
	return 0.0 // unreachable
}

// beta_sample returns a sample from Beta(a, b) using the Gamma ratio method.
fn beta_sample(a f64, b f64) f64 {
	x := gamma_sample(a)
	y := gamma_sample(b)
	total := x + y
	if total <= 0.0 {
		return 0.5
	}
	return x / total
}

// bayesian_ab_test runs a Bayesian A/B test using the Beta-Binomial conjugate model.
// Returns the probability that B beats A, expected losses, and 95% credible intervals.
//
// Uses a uniform (non-informative) prior by default: Beta(1, 1).
// For informative priors, set alpha_prior and beta_prior in BayesianConfig.
//
// The result does NOT include p-values. Instead interpret:
//   prob_b_beats_a: probability B's true rate exceeds A's
//   expected_loss_b: expected regret if you deploy B (and A was actually better)
pub fn bayesian_ab_test(successes_a int, n_a int, successes_b int, n_b int, cfg BayesianConfig) BayesianResult {
	assert n_a > 0 && n_b > 0, 'group sizes must be positive'
	assert successes_a >= 0 && successes_a <= n_a, 'successes_a out of range'
	assert successes_b >= 0 && successes_b <= n_b, 'successes_b out of range'
	assert cfg.alpha_prior > 0 && cfg.beta_prior > 0, 'priors must be positive'
	assert cfg.n_samples > 0, 'n_samples must be positive'

	// Posterior parameters: Beta(alpha_prior + successes, beta_prior + failures)
	alpha_a := cfg.alpha_prior + f64(successes_a)
	beta_a := cfg.beta_prior + f64(n_a - successes_a)
	alpha_b := cfg.alpha_prior + f64(successes_b)
	beta_b := cfg.beta_prior + f64(n_b - successes_b)

	// Analytic posterior means
	mean_a := alpha_a / (alpha_a + beta_a)
	mean_b := alpha_b / (alpha_b + beta_b)

	// Monte Carlo
	n := cfg.n_samples
	mut samples_a := []f64{len: n}
	mut samples_b := []f64{len: n}
	mut b_beats := 0
	mut loss_a := 0.0
	mut loss_b := 0.0

	for i in 0 .. n {
		sa := beta_sample(alpha_a, beta_a)
		sb := beta_sample(alpha_b, beta_b)
		samples_a[i] = sa
		samples_b[i] = sb
		if sb > sa {
			b_beats++
		}
		diff := sb - sa
		if diff > 0 {
			loss_a += diff
		} else {
			loss_b += -diff
		}
	}

	prob_b := f64(b_beats) / f64(n)
	exp_loss_a := loss_a / f64(n)
	exp_loss_b := loss_b / f64(n)

	// Credible intervals via percentiles of sorted samples
	ci_lo_a := stats.quantile(samples_a, 0.025)
	ci_hi_a := stats.quantile(samples_a, 0.975)
	ci_lo_b := stats.quantile(samples_b, 0.025)
	ci_hi_b := stats.quantile(samples_b, 0.975)

	return BayesianResult{
		posterior_mean_a: mean_a
		posterior_mean_b: mean_b
		prob_b_beats_a:   prob_b
		expected_loss_a:  exp_loss_a
		expected_loss_b:  exp_loss_b
		ci_lower_a:       ci_lo_a
		ci_upper_a:       ci_hi_a
		ci_lower_b:       ci_lo_b
		ci_upper_b:       ci_hi_b
		successes_a:      successes_a
		successes_b:      successes_b
		n_a:              n_a
		n_b:              n_b
	}
}
