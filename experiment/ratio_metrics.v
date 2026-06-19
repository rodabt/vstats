module experiment

import math
import stats
import prob

pub struct RatioMetricResult {
pub:
	ratio_ctrl    f64
	ratio_trt     f64
	diff          f64
	relative_lift f64
	se            f64
	z_statistic   f64
	p_value       f64
	significant   bool
	ci_lower      f64
	ci_upper      f64
	n_ctrl        int
	n_trt         int
}

// ratio_metric_test runs a delta-method z-test for ratio metrics (e.g. CTR, ARPU).
// num_ctrl/den_ctrl: per-unit numerator and denominator for control.
// num_trt/den_trt:   per-unit numerator and denominator for treatment.
// Example: for CTR, num = clicks per user, den = impressions per user.
// For simple rates where each unit has one event, pass den = []f64{len: n, init: 1.0}.
pub fn ratio_metric_test(num_ctrl []f64, den_ctrl []f64, num_trt []f64, den_trt []f64, cfg ABTestConfig) RatioMetricResult {
	assert num_ctrl.len == den_ctrl.len, 'num_ctrl and den_ctrl must have same length'
	assert num_trt.len == den_trt.len, 'num_trt and den_trt must have same length'
	assert num_ctrl.len >= 2 && num_trt.len >= 2, 'each group needs at least 2 observations'

	n_c := num_ctrl.len
	n_t := num_trt.len

	// Per-arm ratio: R = mean(num) / mean(den)
	mu_x_c := stats.mean(num_ctrl)
	mu_y_c := stats.mean(den_ctrl)
	assert mu_y_c != 0.0, 'control denominator mean must be nonzero'
	ratio_c := mu_x_c / mu_y_c

	mu_x_t := stats.mean(num_trt)
	mu_y_t := stats.mean(den_trt)
	assert mu_y_t != 0.0, 'treatment denominator mean must be nonzero'
	ratio_t := mu_x_t / mu_y_t

	// Delta-method variance:
	// Var(R) ≈ (1/n) * [Var(X)/μ_Y² - 2·μ_X·Cov(X,Y)/μ_Y³ + μ_X²·Var(Y)/μ_Y⁴]
	var_x_c := stats.variance(num_ctrl)
	var_y_c := stats.variance(den_ctrl)
	cov_c   := stats.covariance(num_ctrl, den_ctrl)
	my2_c   := mu_y_c * mu_y_c
	my3_c   := mu_y_c * my2_c
	my4_c   := my2_c * my2_c
	term1_c := var_x_c / my2_c
	term2_c := 2.0 * mu_x_c * cov_c / my3_c
	term3_c := mu_x_c * mu_x_c * var_y_c / my4_c
	var_rc  := (term1_c - term2_c + term3_c) / f64(n_c)

	var_x_t := stats.variance(num_trt)
	var_y_t := stats.variance(den_trt)
	cov_t   := stats.covariance(num_trt, den_trt)
	my2_t   := mu_y_t * mu_y_t
	my3_t   := mu_y_t * my2_t
	my4_t   := my2_t * my2_t
	term1_t := var_x_t / my2_t
	term2_t := 2.0 * mu_x_t * cov_t / my3_t
	term3_t := mu_x_t * mu_x_t * var_y_t / my4_t
	var_rt  := (term1_t - term2_t + term3_t) / f64(n_t)

	se := math.sqrt(math.max(var_rc + var_rt, 0.0))
	diff := ratio_t - ratio_c
	z := if se > 0 { diff / se } else { 0.0 }
	p_val := 2.0 * prob.normal_cdf(-math.abs(z), 0.0, 1.0)
	z_ci := prob.inverse_normal_cdf(1.0 - cfg.alpha / 2.0, 0.0, 1.0)
	lift := if ratio_c != 0 { diff / math.abs(ratio_c) } else { 0.0 }

	return RatioMetricResult{
		ratio_ctrl:    ratio_c
		ratio_trt:     ratio_t
		diff:          diff
		relative_lift: lift
		se:            se
		z_statistic:   z
		p_value:       p_val
		significant:   p_val < cfg.alpha
		ci_lower:      diff - z_ci * se
		ci_upper:      diff + z_ci * se
		n_ctrl:        n_c
		n_trt:         n_t
	}
}
