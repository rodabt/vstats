module experiment

import vstats.prob

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
// the one-sided two-proportion z-test power formula.
pub fn compute_design_point(alpha f64, beta f64, mde f64, baseline f64, daily_traffic int) DesignPoint {
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(1.0 - beta, 0.0, 1.0)
	p := baseline
	n_raw := (z_alpha + z_beta) * (z_alpha + z_beta) * 2.0 * p * (1.0 - p) / (mde * mde)
	n := if n_raw > f64(int(n_raw)) { int(n_raw) + 1 } else { int(n_raw) }
	rt_raw := f64(n) / f64(daily_traffic)
	runtime := if rt_raw > f64(int(rt_raw)) { int(rt_raw) + 1 } else { int(rt_raw) }
	return DesignPoint{
		alpha:        alpha
		beta:         beta
		mde:          mde
		n_per_arm:    n
		runtime_days: runtime
	}
}
