module experiment

import math
import prob

pub struct PowerDiagnosticResult {
pub:
	n_ctrl               int
	n_trt                int
	observed_effect_size f64
	observed_power       f64
	mde_effect_size      f64  // in Cohen's d units (absolute / baseline_std)
	alpha                f64
	target_power         f64
	underpowered         bool
}

// observed_power returns the probability of detecting an effect of size `effect_size`
// (Cohen's d) given the sample sizes and significance level.
pub fn observed_power(n_ctrl int, n_trt int, effect_size f64, alpha f64) f64 {
	assert n_ctrl >= 2 && n_trt >= 2, 'each group needs at least 2 observations'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	if effect_size == 0.0 {
		return alpha // trivial: only false positives
	}
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	// Noncentrality parameter for two-sample test
	ncp := math.abs(effect_size) * math.sqrt(f64(n_ctrl * n_trt) / f64(n_ctrl + n_trt))
	// Two-tailed power (second tail negligible for ncp > 0)
	power := prob.normal_cdf(ncp - z_alpha, 0.0, 1.0) + prob.normal_cdf(-ncp - z_alpha, 0.0, 1.0)
	return math.min(power, 1.0)
}

// mde_from_n_means returns the minimum detectable absolute effect (in the same units as
// baseline_std) for a continuous metric, given sample sizes and a target power level.
// Dividing the result by baseline_std gives the MDE in Cohen's d units.
pub fn mde_from_n_means(n_ctrl int, n_trt int, baseline_std f64, alpha f64, power f64) f64 {
	assert n_ctrl >= 2 && n_trt >= 2, 'each group needs at least 2 observations'
	assert baseline_std > 0, 'baseline_std must be positive'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	assert power > 0 && power < 1, 'power must be in (0, 1)'
	// Bisect for Cohen's d such that observed_power == power
	mut lo := 0.0
	mut hi := 20.0
	for _ in 0 .. 60 {
		mid := (lo + hi) / 2.0
		if observed_power(n_ctrl, n_trt, mid, alpha) < power {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2.0 * baseline_std
}

// mde_from_n_proportions returns the minimum detectable absolute rate difference
// for a proportion metric, given sample sizes and a target power level.
pub fn mde_from_n_proportions(n_ctrl int, n_trt int, baseline_rate f64, alpha f64, power f64) f64 {
	assert n_ctrl >= 2 && n_trt >= 2, 'each group needs at least 2 observations'
	assert baseline_rate > 0 && baseline_rate < 1, 'baseline_rate must be in (0, 1)'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	assert power > 0 && power < 1, 'power must be in (0, 1)'
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	max_delta := if baseline_rate <= 0.5 {
		1.0 - baseline_rate - 1e-9
	} else {
		baseline_rate - 1e-9
	}
	// Bisect for delta such that the two-proportion test achieves target power.
	// Power derived from: n = (z_α*se_null + z_β*se_alt)² / delta²
	// → z_β = (delta*√n_h - z_α*se_null) / se_alt → power = Φ(z_β)
	// where n_h = sqrt(harmonic_mean(n_ctrl, n_trt)) = sqrt(2*n1*n2/(n1+n2)).
	// Harmonic mean acts as effective per-group n; equal groups → n_h = sqrt(n).
	n_h := math.sqrt(2.0 * f64(n_ctrl) * f64(n_trt) / f64(n_ctrl + n_trt))
	mut lo := 0.0
	mut hi := max_delta
	for _ in 0 .. 60 {
		delta := (lo + hi) / 2.0
		p2 := baseline_rate + delta
		if p2 <= 0.0 || p2 >= 1.0 {
			lo = (lo + hi) / 2.0
			continue
		}
		p_bar := (baseline_rate + p2) / 2.0
		se_null := math.sqrt(2.0 * p_bar * (1.0 - p_bar))
		se_alt := math.sqrt(baseline_rate * (1.0 - baseline_rate) + p2 * (1.0 - p2))
		z_b := if se_alt > 0 { (delta * n_h - z_alpha * se_null) / se_alt } else { 0.0 }
		pwr := prob.normal_cdf(z_b, 0.0, 1.0)
		if pwr < power {
			lo = delta
		} else {
			hi = delta
		}
	}
	return (lo + hi) / 2.0
}

// power_diagnostic computes post-hoc power and MDE from an ABTestResult.
// Call this when an experiment shows a null result to determine if it was underpowered.
// alpha: significance level used in the original test.
// power: target power level (e.g. 0.80).
pub fn power_diagnostic(result ABTestResult, alpha f64, power f64) PowerDiagnosticResult {
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	assert power > 0 && power < 1, 'power must be in (0, 1)'
	d := result.effect_size // Cohen's d
	obs_pwr := observed_power(result.n_control, result.n_treatment, d, alpha)
	// mde_from_n_means with baseline_std=1.0 returns Cohen's d directly
	mde_d := mde_from_n_means(result.n_control, result.n_treatment, 1.0, alpha, power)
	return PowerDiagnosticResult{
		n_ctrl:               result.n_control
		n_trt:                result.n_treatment
		observed_effect_size: d
		observed_power:       obs_pwr
		mde_effect_size:      mde_d
		alpha:                alpha
		target_power:         power
		underpowered:         obs_pwr < power
	}
}
