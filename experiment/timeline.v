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
