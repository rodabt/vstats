module experiment

import vstats.prob
import math
import rand

pub struct MixturePrior {
pub:
	null_frac f64 = 0.40
	neg_frac  f64 = 0.30
	neg_mean  f64 = -0.02
	neg_std   f64 = 0.01
	pos_mean  f64 = 0.02
	pos_std   f64 = 0.01
	n_samples int = 100_000
}

pub struct OptimizerConfig {
pub:
	baseline                  f64
	daily_traffic_per_variant int
	mde_tolerance             f64
	alpha                     f64 = 0.05
	prior                     MixturePrior
	seasonality_min_days      int = 14
	min_power                 f64 = 0.80
	max_days                  int = 90
	seed                      u32
}

pub struct RuntimeResult {
pub:
	runtime_days           int
	monthly_detection_rate f64
}

pub struct OptimizationResult {
pub:
	optimal_days           int
	monthly_detection_rate f64
	all_results            []RuntimeResult
	worth_running          bool
	power_min_days         int
	effective_min_days     int
	power_at_optimal       f64
	no_go_reason           string
}

fn sample_effects(prior MixturePrior) []f64 {
	n := prior.n_samples
	pos_threshold := prior.null_frac + prior.neg_frac
	mut theta := []f64{len: n}
	for i in 0 .. n {
		u := rand.f64()
		if u >= pos_threshold {
			theta[i] = prior.pos_mean + prior.pos_std * box_muller()
		} else if u >= prior.null_frac {
			theta[i] = prior.neg_mean + prior.neg_std * box_muller()
		}
	}
	return theta
}

fn power_for_effect(effect f64, baseline f64, sample_per_variant int, alpha f64) f64 {
	p1 := baseline
	p2_raw := baseline + effect
	p2 := if p2_raw < 0.0001 { 0.0001 } else if p2_raw > 0.9999 { 0.9999 } else { p2_raw }
	denom := f64(sample_per_variant)
	se := math.sqrt(p1 * (1.0 - p1) / denom + p2 * (1.0 - p2) / denom)
	if se <= 0.0 {
		return 0.0
	}
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	noncentrality := math.abs(effect) / se
	return 1.0 - prob.normal_cdf(z_alpha - noncentrality, 0.0, 1.0)
}

fn detection_rate(runtime_days int, config OptimizerConfig, effects []f64) f64 {
	n := runtime_days * config.daily_traffic_per_variant
	mut total := 0.0
	for effect in effects {
		if effect >= config.mde_tolerance {
			total += power_for_effect(effect, config.baseline, n, config.alpha)
		}
	}
	return (total / f64(effects.len)) * (30.0 / f64(runtime_days))
}

fn power_floor(config OptimizerConfig) int {
	effect := config.mde_tolerance
	if effect <= 0.0 {
		return config.max_days
	}
	p1 := config.baseline
	p2_raw := config.baseline + effect
	p2 := if p2_raw < 0.0001 { 0.0001 } else if p2_raw > 0.9999 { 0.9999 } else { p2_raw }
	z_alpha := prob.inverse_normal_cdf(1.0 - config.alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(config.min_power, 0.0, 1.0)
	variance := p1 * (1.0 - p1) + p2 * (1.0 - p2)
	n_raw := (z_alpha + z_beta) * (z_alpha + z_beta) * variance / (effect * effect)
	n := int(math.ceil(n_raw))
	days_raw := f64(n) / f64(config.daily_traffic_per_variant)
	return int(math.ceil(days_raw))
}

pub fn find_optimal_runtime(config OptimizerConfig) OptimizationResult {
	assert config.baseline > 0.0 && config.baseline < 1.0, 'baseline must be in (0, 1)'
	assert config.daily_traffic_per_variant > 0, 'daily_traffic_per_variant must be positive'
	assert config.mde_tolerance > 0.0, 'mde_tolerance must be positive'
	assert config.seasonality_min_days > 0, 'seasonality_min_days must be positive'
	assert config.min_power > 0.0 && config.min_power < 1.0, 'min_power must be in (0, 1)'
	assert config.max_days > 0, 'max_days must be positive'

	pmdays := power_floor(config)
	eff_min := if pmdays > config.seasonality_min_days { pmdays } else { config.seasonality_min_days }

	if pmdays > config.max_days {
		return OptimizationResult{
			worth_running:      false
			power_min_days:     pmdays
			effective_min_days: eff_min
			no_go_reason:       'need ${pmdays} days for ${config.min_power * 100:.0f}% sensitivity to ${config.mde_tolerance * 100:.1f}pp MDE but max_days is ${config.max_days} — increase traffic or max_days'
		}
	}

	rand.seed([config.seed, u32(0)])
	effects := sample_effects(config.prior)

	mut all_results := []RuntimeResult{}
	for days in eff_min .. config.max_days + 1 {
		dr := detection_rate(days, config, effects)
		all_results << RuntimeResult{ runtime_days: days, monthly_detection_rate: dr }
	}

	mut best_idx := 0
	for i, r in all_results {
		if r.monthly_detection_rate > all_results[best_idx].monthly_detection_rate {
			best_idx = i
		}
	}

	opt_days := all_results[best_idx].runtime_days
	opt_dr := all_results[best_idx].monthly_detection_rate
	opt_power := power_for_effect(config.mde_tolerance, config.baseline,
		opt_days * config.daily_traffic_per_variant, config.alpha)

	worth := opt_dr > 0.0
	reason := if !worth {
		'no effects >= ${config.mde_tolerance * 100:.1f}pp in prior, or power too low — check prior.pos_mean vs mde_tolerance'
	} else {
		''
	}

	return OptimizationResult{
		optimal_days:           opt_days
		monthly_detection_rate: opt_dr
		all_results:            all_results
		worth_running:          worth
		power_min_days:         pmdays
		effective_min_days:     eff_min
		power_at_optimal:       opt_power
		no_go_reason:           reason
	}
}
