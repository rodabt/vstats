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

@[params]
pub struct BayesianContinuousConfig {
pub:
	prior_mean f64 = 0.0
	prior_std  f64 = 1000.0 // diffuse non-informative default
	n_samples  int = 10000
	rope_lower f64 = 0.0   // region of practical equivalence lower bound
	rope_upper f64 = 0.0   // upper bound; both 0.0 = no ROPE
}

pub struct BayesianContinuousResult {
pub:
	posterior_mean_ctrl f64
	posterior_mean_trt  f64
	posterior_std_ctrl  f64
	posterior_std_trt   f64
	prob_trt_beats_ctrl f64
	expected_loss_ctrl  f64
	expected_loss_trt   f64
	ci_lower_ctrl       f64
	ci_upper_ctrl       f64
	ci_lower_trt        f64
	ci_upper_trt        f64
	prob_rope           f64 // P(effect in ROPE); 0.0 when ROPE not set
}

// bayesian_continuous_ab_test runs a Bayesian A/B test for continuous outcomes
// (e.g. revenue, session duration, score) using a Normal-Normal conjugate model.
// Returns P(treatment > control), expected losses, and 95% credible intervals.
//
// Use BayesianContinuousConfig.rope_lower/rope_upper to define a region of practical
// equivalence — effects smaller than this are considered negligible even if real.
pub fn bayesian_continuous_ab_test(ctrl []f64, trt []f64, cfg BayesianContinuousConfig) BayesianContinuousResult {
	assert ctrl.len >= 2 && trt.len >= 2, 'each group needs at least 2 observations'
	assert cfg.prior_std > 0, 'prior_std must be positive'
	assert cfg.n_samples > 0, 'n_samples must be positive'

	n_c := ctrl.len
	n_t := trt.len

	sample_mean_c := stats.mean(ctrl)
	sample_mean_t := stats.mean(trt)

	// Use per-arm sample variance as the known likelihood variance (σ²)
	sigma2_c := if stats.variance(ctrl) > 0 { stats.variance(ctrl) } else { 1.0 }
	sigma2_t := if stats.variance(trt) > 0 { stats.variance(trt) } else { 1.0 }

	// Normal-Normal conjugate update:
	// Prior: μ ~ N(prior_mean, prior_std²)
	// Posterior: μ | data ~ N(post_mean, post_var)
	// post_var  = 1 / (1/prior_var + n/sigma²)
	// post_mean = post_var * (prior_mean/prior_var + n*sample_mean/sigma²)
	prior_var := cfg.prior_std * cfg.prior_std

	post_var_c := 1.0 / (1.0 / prior_var + f64(n_c) / sigma2_c)
	post_mean_c := post_var_c * (cfg.prior_mean / prior_var + f64(n_c) * sample_mean_c / sigma2_c)
	post_std_c := math.sqrt(post_var_c)

	post_var_t := 1.0 / (1.0 / prior_var + f64(n_t) / sigma2_t)
	post_mean_t := post_var_t * (cfg.prior_mean / prior_var + f64(n_t) * sample_mean_t / sigma2_t)
	post_std_t := math.sqrt(post_var_t)

	n := cfg.n_samples
	mut samples_c := []f64{len: n}
	mut samples_t := []f64{len: n}
	mut trt_beats := 0
	mut loss_c := 0.0
	mut loss_t := 0.0
	mut in_rope := 0
	has_rope := cfg.rope_lower != 0.0 || cfg.rope_upper != 0.0

	for i in 0 .. n {
		sc := post_mean_c + post_std_c * box_muller()
		st := post_mean_t + post_std_t * box_muller()
		samples_c[i] = sc
		samples_t[i] = st
		if st > sc {
			trt_beats++
		}
		diff := st - sc
		if diff > 0 {
			loss_c += diff
		} else {
			loss_t += -diff
		}
		if has_rope && diff >= cfg.rope_lower && diff <= cfg.rope_upper {
			in_rope++
		}
	}

	prob_beats := f64(trt_beats) / f64(n)
	exp_loss_c := loss_c / f64(n)
	exp_loss_t := loss_t / f64(n)
	p_rope := if has_rope { f64(in_rope) / f64(n) } else { 0.0 }

	return BayesianContinuousResult{
		posterior_mean_ctrl: post_mean_c
		posterior_mean_trt:  post_mean_t
		posterior_std_ctrl:  post_std_c
		posterior_std_trt:   post_std_t
		prob_trt_beats_ctrl: prob_beats
		expected_loss_ctrl:  exp_loss_c
		expected_loss_trt:   exp_loss_t
		ci_lower_ctrl:       stats.quantile(samples_c, 0.025)
		ci_upper_ctrl:       stats.quantile(samples_c, 0.975)
		ci_lower_trt:        stats.quantile(samples_t, 0.025)
		ci_upper_trt:        stats.quantile(samples_t, 0.975)
		prob_rope:           p_rope
	}
}
