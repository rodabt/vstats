module experiment

import vstats.prob
import math
import rand

pub struct EffectDist {
pub:
	mean f64
	std  f64
}

pub struct OptimizerConfig {
pub:
	baseline      f64
	daily_traffic int
	post_test_n   int
	effect_dist   EffectDist
	n_sims        int
	seed          u32
}

pub struct DesignPoint {
pub:
	alpha        f64
	beta         f64
	mde          f64
	n_per_arm    int
	runtime_days int
}

pub struct DesignScore {
pub:
	design         DesignPoint
	expected_value f64
	correct_rate   f64
}

pub struct OptimizationResult {
pub:
	best DesignScore
	grid []DesignScore
}

// compute_design_point derives sample size and runtime from (alpha, beta, mde) via
// the one-sided two-proportion z-test power formula. alpha is the one-sided type-I
// error rate: z_alpha = Φ⁻¹(1 − alpha), not Φ⁻¹(1 − alpha/2).
pub fn compute_design_point(alpha f64, beta f64, mde f64, baseline f64, daily_traffic int) DesignPoint {
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0, 1)'
	assert beta > 0.0 && beta < 1.0, 'beta must be in (0, 1)'
	assert mde > 0.0, 'mde must be positive'
	assert baseline > 0.0 && baseline < 1.0, 'baseline must be in (0, 1)'
	assert daily_traffic > 0, 'daily_traffic must be positive'
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(1.0 - beta, 0.0, 1.0)
	p := baseline
	n_raw := (z_alpha + z_beta) * (z_alpha + z_beta) * 2.0 * p * (1.0 - p) / (mde * mde)
	n := int(math.ceil(n_raw))
	runtime := int(math.ceil(f64(n) / f64(daily_traffic)))
	return DesignPoint{
		alpha:        alpha
		beta:         beta
		mde:          mde
		n_per_arm:    n
		runtime_days: runtime
	}
}

fn clamp_f64(x f64, lo f64, hi f64) f64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

// simulate_one runs one simulated experiment and returns (marginal_value, decision_correct).
// z_alpha is the pre-computed one-sided critical value for dp.alpha.
fn simulate_one(config OptimizerConfig, dp DesignPoint, z_alpha f64) (f64, bool) {
	p := config.baseline
	n := f64(dp.n_per_arm)

	// Draw true effect from N(mean, std^2); may be negative (harmful treatment)
	true_effect := config.effect_dist.mean + config.effect_dist.std * box_muller()
	p_treat := clamp_f64(p + true_effect, 0.0, 1.0)

	// Control arm: N(n*p, n*p*(1-p))
	ctrl_std := math.sqrt(n * p * (1.0 - p))
	ctrl_conv := clamp_f64(n * p + ctrl_std * box_muller(), 0.0, n)

	// Treatment arm: N(n*p_treat, n*p_treat*(1-p_treat))
	trt_std := math.sqrt(n * p_treat * (1.0 - p_treat))
	trt_conv := clamp_f64(n * p_treat + trt_std * box_muller(), 0.0, n)

	// One-sided z-statistic using pooled SE
	p_ctrl_obs := ctrl_conv / n
	p_trt_obs := trt_conv / n
	pooled_p := (ctrl_conv + trt_conv) / (2.0 * n)
	se := math.sqrt(2.0 * pooled_p * (1.0 - pooled_p) / n)
	z := if se > 0 { (p_trt_obs - p_ctrl_obs) / se } else { 0.0 }

	shipped := z > z_alpha

	// Marginal value over never-testing baseline
	test_excess := trt_conv - n * p
	post_test_value := if shipped { true_effect * f64(config.post_test_n) } else { 0.0 }
	marginal := test_excess + post_test_value

	correct := (true_effect > 0 && shipped) || (true_effect <= 0 && !shipped)
	return marginal, correct
}

// score_design_point evaluates a DesignPoint across n_sims simulations and returns
// average marginal conversions and correct decision rate.
pub fn score_design_point(config OptimizerConfig, dp DesignPoint) DesignScore {
	assert config.n_sims > 0, 'n_sims must be positive'
	rand.seed([config.seed, u32(0)])
	z_alpha := prob.inverse_normal_cdf(1.0 - dp.alpha, 0.0, 1.0)

	mut total_value := 0.0
	mut correct_count := 0
	for _ in 0 .. config.n_sims {
		v, correct := simulate_one(config, dp, z_alpha)
		total_value += v
		if correct {
			correct_count++
		}
	}
	return DesignScore{
		design:         dp
		expected_value: total_value / f64(config.n_sims)
		correct_rate:   f64(correct_count) / f64(config.n_sims)
	}
}

// optimize sweeps all (alpha, beta, mde) combinations and returns the DesignScore
// with the highest expected_value alongside the full scored grid.
pub fn optimize(config OptimizerConfig, alpha_values []f64, beta_values []f64, mde_values []f64) OptimizationResult {
	assert alpha_values.len > 0 && beta_values.len > 0 && mde_values.len > 0, 'alpha_values, beta_values, and mde_values must be non-empty'
	mut grid := []DesignScore{}
	for alpha in alpha_values {
		for beta in beta_values {
			for mde in mde_values {
				dp := compute_design_point(alpha, beta, mde, config.baseline, config.daily_traffic)
				score := score_design_point(config, dp)
				grid << score
			}
		}
	}
	mut best_idx := 0
	for i, s in grid {
		if s.expected_value > grid[best_idx].expected_value {
			best_idx = i
		}
	}
	return OptimizationResult{
		best: grid[best_idx]
		grid: grid
	}
}
