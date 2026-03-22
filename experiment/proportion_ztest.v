module experiment

import math
import prob

@[params]
pub struct ProportionTestConfig {
pub:
	alpha f64 = 0.05
}

pub struct ProportionTestResult {
pub:
	rate_a        f64
	rate_b        f64
	diff          f64
	relative_lift f64
	z_statistic   f64
	p_value       f64
	significant   bool
	ci_lower      f64
	ci_upper      f64
	pooled_se     f64
	n_a           int
	n_b           int
}

// proportion_test runs a two-proportion z-test comparing the conversion rates
// of two groups. Use this for binary outcomes: click, signup, purchase, etc.
//
// successes_a/n_a: events and total observations for group A (control)
// successes_b/n_b: events and total observations for group B (treatment)
pub fn proportion_test(successes_a int, n_a int, successes_b int, n_b int, cfg ProportionTestConfig) ProportionTestResult {
	assert n_a > 0 && n_b > 0, 'group sizes must be positive'
	assert successes_a >= 0 && successes_a <= n_a, 'successes_a out of range'
	assert successes_b >= 0 && successes_b <= n_b, 'successes_b out of range'

	rate_a := f64(successes_a) / f64(n_a)
	rate_b := f64(successes_b) / f64(n_b)
	diff := rate_b - rate_a
	lift := if rate_a > 0 { diff / rate_a } else { 0.0 }

	// Pooled SE under H0 (used for z-statistic)
	p_pool := f64(successes_a + successes_b) / f64(n_a + n_b)
	pse := math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / f64(n_a) + 1.0 / f64(n_b)))
	z := if pse > 0 { diff / pse } else { 0.0 }
	p_val := 2.0 * prob.normal_cdf(-math.abs(z), 0.0, 1.0)

	// Unpooled SE for CI (uses alpha from config, never hard-coded)
	se_diff := math.sqrt(rate_a * (1.0 - rate_a) / f64(n_a) + rate_b * (1.0 - rate_b) / f64(n_b))
	z_ci := prob.inverse_normal_cdf(1.0 - cfg.alpha / 2.0, 0.0, 1.0)
	ci_lo := diff - z_ci * se_diff
	ci_hi := diff + z_ci * se_diff

	return ProportionTestResult{
		rate_a:        rate_a
		rate_b:        rate_b
		diff:          diff
		relative_lift: lift
		z_statistic:   z
		p_value:       p_val
		significant:   p_val < cfg.alpha
		ci_lower:      ci_lo
		ci_upper:      ci_hi
		pooled_se:     pse
		n_a:           n_a
		n_b:           n_b
	}
}
