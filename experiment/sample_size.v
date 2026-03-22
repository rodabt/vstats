module experiment

import math
import prob

pub struct SampleSizeResult {
pub:
	n_per_group  int
	total_n      int
	alpha        f64
	power        f64
	mde          f64
	baseline     f64
	effect_size  f64
	baseline_std f64
	method       string
}

// sample_size_proportions computes the required sample size per group to detect
// a minimum detectable effect (mde) in a proportion metric.
//
// baseline_rate: current conversion rate (e.g. 0.05 for 5%)
// mde:           absolute change to detect (e.g. 0.01 to detect 5% -> 6%)
// alpha:         significance level (e.g. 0.05)
// power:         desired power (e.g. 0.80)
pub fn sample_size_proportions(baseline_rate f64, mde f64, alpha f64, power f64) SampleSizeResult {
	assert baseline_rate > 0 && baseline_rate < 1, 'baseline_rate must be in (0, 1)'
	assert mde > 0, 'mde must be positive'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	assert power > 0 && power < 1, 'power must be in (0, 1)'

	p1 := baseline_rate
	p2 := baseline_rate + mde
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)

	numerator := math.pow(z_alpha + z_beta, 2) * (p1 * (1.0 - p1) + p2 * (1.0 - p2))
	denominator := math.pow(p2 - p1, 2)
	n_raw := numerator / denominator
	n := int(math.ceil(n_raw))

	pooled_var := (p1 * (1.0 - p1) + p2 * (1.0 - p2)) / 2.0
	effect := if pooled_var > 0 { math.abs(mde) / math.sqrt(pooled_var) } else { 0.0 }

	return SampleSizeResult{
		n_per_group:  n
		total_n:      n * 2
		alpha:        alpha
		power:        power
		mde:          mde
		baseline:     baseline_rate
		effect_size:  effect
		baseline_std: 0.0
		method:       'proportions'
	}
}

// sample_size_means computes the required sample size per group to detect
// a minimum detectable effect in a continuous metric.
//
// baseline_mean: current mean value
// baseline_std:  current standard deviation (must be estimated from historical data)
// mde_absolute:  absolute change in mean to detect
// alpha:         significance level (e.g. 0.05)
// power:         desired power (e.g. 0.80)
pub fn sample_size_means(baseline_mean f64, baseline_std f64, mde_absolute f64, alpha f64, power f64) SampleSizeResult {
	assert baseline_std > 0, 'baseline_std must be positive'
	assert mde_absolute > 0, 'mde_absolute must be positive'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	assert power > 0 && power < 1, 'power must be in (0, 1)'

	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)

	n_raw := 2.0 * math.pow((z_alpha + z_beta) * baseline_std / mde_absolute, 2)
	n := int(math.ceil(n_raw))
	cohens_d := mde_absolute / baseline_std

	return SampleSizeResult{
		n_per_group:  n
		total_n:      n * 2
		alpha:        alpha
		power:        power
		mde:          mde_absolute
		baseline:     baseline_mean
		effect_size:  cohens_d
		baseline_std: baseline_std
		method:       'means'
	}
}
