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
	// Minimum acceptable expected monthly detection rate. If the prior-weighted
	// probability of finding a true positive in any given month is below this value,
	// the experiment is not worth the opportunity cost of running. 0.05 means you
	// expect to detect something real roughly once every 20 months.
	min_monthly_detection_rate f64 = 0.05
	// Standard deviation of the metric per observation. Leave at 0 (default) for
	// proportion metrics (conversion rate, click-through rate), where variance is
	// derived from the baseline. Set to a positive value for continuous metrics
	// (revenue per user, session duration, order value) measured from historical data.
	metric_std_dev f64 = 0.0
	seed           u32
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

// DesignParams is the simplified entry point for experiment design.
// It replaces the need to choose an MDE in absolute pp or construct a MixturePrior
// manually. Use optimizer_config() to convert it to an OptimizerConfig.
pub struct DesignParams {
pub:
	baseline                  f64
	daily_traffic_per_variant int
	// Smallest lift worth detecting, expressed as a fraction of the baseline.
	// 0.05 = "I want to detect improvements of at least 5% above the current rate."
	// Halving this requires roughly 4× more traffic or runtime.
	min_relative_lift    f64 = 0.05
	// Prior belief that this experiment will produce a meaningful positive effect.
	// 0.0 = very skeptical: most experiments have no effect or harm the metric.
	// 1.0 = very confident: most experiments move the metric positively.
	// Typical values: 0.2 for speculative ideas, 0.5 for user-researched features,
	// 0.8 for incremental changes with strong historical signal.
	prior_conviction     f64 = 0.50
	max_days             int = 90
	// Standard deviation of the metric per observation. Required for continuous
	// metrics (revenue, session duration, order value). Leave at 0 for proportion
	// metrics (conversion rate, CTR) — variance is derived from the baseline.
	// Obtain from historical data: std_dev of the per-user metric value.
	metric_std_dev       f64 = 0.0
	// Advanced overrides — defaults are sensible for most experiments.
	alpha                f64 = 0.05
	min_power            f64 = 0.80
	seasonality_min_days int  = 14
	seed                 u32
}

// conviction_to_prior maps a conviction score in [0, 1] to a MixturePrior and
// a monthly-detection-rate threshold. The prior is anchored to mde so that the
// positive and negative effect distributions scale with the sensitivity target.
// Returns (prior, min_monthly_detection_rate).
pub fn conviction_to_prior(conviction f64, mde f64) (MixturePrior, f64) {
	c := math.min(math.max(conviction, 0.0), 1.0)
	// At low conviction, most experiments are null or harmful; few have a positive effect.
	// At high conviction, the distribution shifts toward positive effects.
	prior := MixturePrior{
		null_frac: 0.65 - 0.55 * c  // 0.65 (skeptical) → 0.10 (confident)
		neg_frac:  0.30 - 0.20 * c  // 0.30 (skeptical) → 0.10 (confident)
		neg_mean:  -(mde * 3.0)
		neg_std:   mde * 1.5
		pos_mean:  mde * 2.0
		pos_std:   mde * 0.5
		n_samples: 45_000
	}
	// At low conviction, require a high monthly detection rate before running —
	// skeptical experiments should only proceed if even the pessimistic prior still
	// yields a meaningful signal. At high conviction, a low threshold suffices because
	// the prior already expects frequent positive outcomes.
	// Calibrated so that conviction ≈ 0.20 is the crossover: below it the experiment
	// is unlikely to be worth the opportunity cost.
	min_dr := 0.50 - 0.48 * c  // 0.50 (very skeptical) → 0.02 (very confident)
	return prior, min_dr
}

// optimizer_config converts a DesignParams into a fully-specified OptimizerConfig,
// deriving the MDE, prior, and detection-rate threshold from the simplified inputs.
pub fn optimizer_config(params DesignParams) OptimizerConfig {
	mde := params.baseline * params.min_relative_lift
	prior, min_dr := conviction_to_prior(params.prior_conviction, mde)
	return OptimizerConfig{
		baseline:                   params.baseline
		daily_traffic_per_variant:  params.daily_traffic_per_variant
		mde_tolerance:              mde
		prior:                      prior
		min_monthly_detection_rate: min_dr
		metric_std_dev:             params.metric_std_dev
		max_days:                   params.max_days
		alpha:                      params.alpha
		min_power:                  params.min_power
		seasonality_min_days:       params.seasonality_min_days
		seed:                       params.seed
	}
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

fn power_for_effect(effect f64, baseline f64, sample_per_variant int, z_alpha f64, std_dev f64) f64 {
	denom := f64(sample_per_variant)
	se := if std_dev > 0.0 {
		// Two-sample t-test for continuous metrics: variance is σ²/n per arm.
		math.sqrt(2.0 * std_dev * std_dev / denom)
	} else {
		// Two-proportion z-test: variance is p(1-p)/n per arm.
		p1 := baseline
		p2_raw := baseline + effect
		p2 := if p2_raw < 0.0001 { 0.0001 } else if p2_raw > 0.9999 { 0.9999 } else { p2_raw }
		math.sqrt(p1 * (1.0 - p1) / denom + p2 * (1.0 - p2) / denom)
	}
	if se <= 0.0 {
		return 0.0
	}
	noncentrality := math.abs(effect) / se
	return 1.0 - prob.normal_cdf(z_alpha - noncentrality, 0.0, 1.0)
}

fn detection_rate(runtime_days int, config OptimizerConfig, effects []f64, z_alpha f64) f64 {
	n := runtime_days * config.daily_traffic_per_variant
	mut total := 0.0
	for effect in effects {
		if effect >= config.mde_tolerance {
			total += power_for_effect(effect, config.baseline, n, z_alpha, config.metric_std_dev)
		}
	}
	// p_exp is the prior-weighted probability that a single experiment of this
	// runtime detects a true positive effect. Over a 30-day window you can run
	// 30/runtime_days back-to-back experiments; the probability of detecting at
	// least one true positive compounds as 1 - (1 - p_exp)^(30/runtime_days).
	// This stays bounded in [0, 1] (the old linear scaling could exceed 1) and
	// reduces to p_exp * 30/runtime_days for small p_exp.
	p_exp := total / f64(effects.len)
	return 1.0 - math.pow(1.0 - p_exp, 30.0 / f64(runtime_days))
}

fn power_floor(config OptimizerConfig) int {
	effect := config.mde_tolerance
	if effect <= 0.0 {
		return config.max_days
	}
	n := int(math.ceil(sample_size_for(effect, config.baseline, config.metric_std_dev,
		config.alpha, config.min_power)))
	days_raw := f64(n) / f64(config.daily_traffic_per_variant)
	return int(math.ceil(days_raw))
}

pub fn find_optimal_runtime(config OptimizerConfig) OptimizationResult {
	assert config.baseline > 0.0, 'baseline must be positive'
	if config.metric_std_dev == 0.0 {
		assert config.baseline < 1.0, 'proportion baseline must be in (0, 1) — set metric_std_dev for continuous metrics'
	}
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
			no_go_reason:       'need ${pmdays} days for ${config.min_power * 100:.0f}% power at MDE ${config.mde_tolerance:.4f} but max_days is ${config.max_days} — increase traffic or max_days'
		}
	}

	rand.seed([config.seed, u32(0)])
	effects := sample_effects(config.prior)

	// z_alpha depends only on alpha, so compute it once here rather than on every
	// power_for_effect call inside the day × sample loops below.
	z_alpha := prob.inverse_normal_cdf(1.0 - config.alpha / 2.0, 0.0, 1.0)

	mut all_results := []RuntimeResult{}
	for days in eff_min .. config.max_days + 1 {
		dr := detection_rate(days, config, effects, z_alpha)
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
		opt_days * config.daily_traffic_per_variant, z_alpha, config.metric_std_dev)

	worth := opt_dr > 0.0 && opt_dr >= config.min_monthly_detection_rate
	reason := if !worth {
		if opt_dr <= 0.0 {
			'no effects >= ${config.mde_tolerance * 100:.1f}pp in prior — check prior.pos_mean vs mde_tolerance'
		} else {
			'monthly detection rate ${opt_dr:.3f} is below threshold ${config.min_monthly_detection_rate:.3f} — prior assigns too little probability to positive effects above MDE'
		}
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
