module experiment

import vstats.prob
import math

// sample_size_for returns the raw (un-rounded) samples-per-arm needed to detect
// `effect` (absolute) at the given alpha and power. Proportions use a pooled-variance
// null and unpooled alternative; continuous metrics use the 2σ² two-sample form.
// This is the single source of truth shared by power_floor and the timeline solver.
fn sample_size_for(effect f64, baseline f64, std_dev f64, alpha f64, power f64) f64 {
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)
	if std_dev > 0.0 {
		// Two-sample t-test: n = 2σ²(z_α + z_β)² / δ²
		return 2.0 * std_dev * std_dev * math.pow(z_alpha + z_beta, 2.0) / (effect * effect)
	}
	// Two-proportion z-test (pooled-variance null, unpooled alternative).
	p1 := baseline
	p2_raw := baseline + effect
	p2 := if p2_raw < 0.0001 { 0.0001 } else if p2_raw > 0.9999 { 0.9999 } else { p2_raw }
	p_bar := (p1 + p2) / 2.0
	var_null := 2.0 * p_bar * (1.0 - p_bar)
	var_alt := p1 * (1.0 - p1) + p2 * (1.0 - p2)
	return math.pow(z_alpha * math.sqrt(var_null) + z_beta * math.sqrt(var_alt), 2.0) / (effect * effect)
}

// power_min_days is the pure power-based runtime (in days) to detect `effect_abs`,
// before any seasonality floor is applied.
fn power_min_days(effect_abs f64, baseline f64, std_dev f64, daily_traffic int, alpha f64, power f64) int {
	n := int(math.ceil(sample_size_for(effect_abs, baseline, std_dev, alpha, power)))
	return int(math.ceil(f64(n) / f64(daily_traffic)))
}

// required_runtime returns the runtime in days needed to detect `target_lift`
// (relative to baseline) at the given alpha and power, raised to the seasonality floor.
pub fn required_runtime(target_lift f64, baseline f64, std_dev f64, daily_traffic int, alpha f64, power f64, seasonality_min_days int) int {
	pd := power_min_days(target_lift * baseline, baseline, std_dev, daily_traffic, alpha, power)
	return if pd > seasonality_min_days { pd } else { seasonality_min_days }
}

pub struct DetectableLift {
pub:
	relative f64 // lift as a fraction of baseline (the MDE the PM reads)
	absolute f64 // lift in metric units (pp for proportions, $/units for continuous)
}

// detectable_lift inverts the power identity: given a runtime, what lift can we
// detect at `power`? Continuous metrics are closed-form; proportions are solved by
// bisection against sample_size_for (which is monotonically decreasing in the effect).
pub fn detectable_lift(runtime_days int, baseline f64, std_dev f64, daily_traffic int, alpha f64, power f64) DetectableLift {
	n_per_arm := f64(runtime_days) * f64(daily_traffic)
	if std_dev > 0.0 {
		z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
		z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)
		abs := (z_alpha + z_beta) * math.sqrt(2.0 * std_dev * std_dev / n_per_arm)
		return DetectableLift{
			relative: abs / baseline
			absolute: abs
		}
	}
	// Proportion: find the absolute effect whose required sample size equals n_per_arm.
	mut lo := 1e-9
	mut hi := 1.0 - baseline
	for _ in 0 .. 60 {
		mid := 0.5 * (lo + hi)
		need := sample_size_for(mid, baseline, std_dev, alpha, power)
		if need > n_per_arm {
			// Not enough samples for this effect → need a LARGER effect to detect.
			lo = mid
		} else {
			hi = mid
		}
	}
	abs := 0.5 * (lo + hi)
	return DetectableLift{
		relative: abs / baseline
		absolute: abs
	}
}
