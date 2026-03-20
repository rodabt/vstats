module experiment

import math
import stats
import prob

@[params]
pub struct ABTestConfig {
pub:
	alpha          f64 = 0.05
	equal_variance bool
}

pub struct ABTestResult {
pub:
	control_mean    f64
	treatment_mean  f64
	control_std     f64
	treatment_std   f64
	relative_lift   f64
	effect_size     f64
	t_statistic     f64
	degrees_freedom f64
	p_value         f64
	significant     bool
	ci_lower        f64
	ci_upper        f64
	n_control       int
	n_treatment     int
}

pub struct PowerAnalysisResult {
pub:
	n_per_group int
	power       f64
	alpha       f64
	effect_size f64
}

pub struct CUPEDResult {
pub:
	theta              f64
	variance_reduction f64
	adjusted_result    ABTestResult
}

// welch_t computes (t_statistic, degrees_of_freedom) via Welch's formula
fn welch_t(m1 f64, m2 f64, v1 f64, v2 f64, n1 int, n2 int) (f64, f64) {
	a := v1 / f64(n1)
	b := v2 / f64(n2)
	se := math.sqrt(a + b)
	t := if se > 0 { (m1 - m2) / se } else { 0.0 }
	// Welch-Satterthwaite degrees of freedom
	df_num := (a + b) * (a + b)
	df_den := a * a / f64(n1 - 1) + b * b / f64(n2 - 1)
	df := if df_den > 0 { df_num / df_den } else { f64(n1 + n2 - 2) }
	return t, df
}

// t_pvalue_approx computes two-tailed p-value via normal CDF approximation
fn t_pvalue_approx(t f64, df f64) f64 {
	_ = df
	return 2.0 * prob.normal_cdf(-math.abs(t), 0.0, 1.0)
}

// welch_ci computes the confidence interval for the difference in means
fn welch_ci(m1 f64, m2 f64, v1 f64, v2 f64, n1 int, n2 int, alpha f64) (f64, f64) {
	a := v1 / f64(n1)
	b := v2 / f64(n2)
	se := math.sqrt(a + b)
	z := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	diff := m1 - m2
	return diff - z * se, diff + z * se
}

// abtest runs a two-sample A/B test (Welch's t-test by default)
pub fn abtest(control []f64, treatment []f64, cfg ABTestConfig) ABTestResult {
	assert control.len >= 2 && treatment.len >= 2, 'each group needs at least 2 observations'

	m_c := stats.mean(control)
	m_t := stats.mean(treatment)
	v_c := stats.variance(control)
	v_t := stats.variance(treatment)
	s_c := math.sqrt(v_c)
	s_t := math.sqrt(v_t)
	n_c := control.len
	n_t := treatment.len

	d := stats.cohens_d(treatment, control)
	t, df := welch_t(m_t, m_c, v_t, v_c, n_t, n_c)
	p := t_pvalue_approx(t, df)
	ci_lo, ci_hi := welch_ci(m_t, m_c, v_t, v_c, n_t, n_c, cfg.alpha)
	lift := if m_c != 0 { (m_t - m_c) / math.abs(m_c) } else { 0.0 }

	return ABTestResult{
		control_mean:    m_c
		treatment_mean:  m_t
		control_std:     s_c
		treatment_std:   s_t
		relative_lift:   lift
		effect_size:     d
		t_statistic:     t
		degrees_freedom: df
		p_value:         p
		significant:     p < cfg.alpha
		ci_lower:        ci_lo
		ci_upper:        ci_hi
		n_control:       n_c
		n_treatment:     n_t
	}
}

// power_analysis computes the required sample size per group
pub fn power_analysis(effect_size f64, alpha f64, power f64) PowerAnalysisResult {
	assert effect_size > 0, 'effect_size must be positive'
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)
	n_raw := 2.0 * math.pow((z_alpha + z_beta) / effect_size, 2)
	n := if n_raw > f64(int(n_raw)) { int(n_raw) + 1 } else { int(n_raw) }

	return PowerAnalysisResult{
		n_per_group: n
		power:       power
		alpha:       alpha
		effect_size: effect_size
	}
}

// cuped_test runs CUPED-adjusted A/B test using pre-experiment covariates
pub fn cuped_test(y_ctrl []f64, y_treat []f64, pre_ctrl []f64, pre_treat []f64, cfg ABTestConfig) CUPEDResult {
	assert y_ctrl.len == pre_ctrl.len, 'control pre/post lengths must match'
	assert y_treat.len == pre_treat.len, 'treatment pre/post lengths must match'

	// Pool pre and post data to estimate theta
	mut y_all := []f64{}
	mut pre_all := []f64{}
	y_all << y_ctrl
	y_all << y_treat
	pre_all << pre_ctrl
	pre_all << pre_treat

	pre_mean := stats.mean(pre_all)
	v_pre := stats.variance(pre_all)
	cov_yp := stats.covariance(y_all, pre_all)
	theta := if v_pre > 0 { cov_yp / v_pre } else { 0.0 }

	// Variance reduction fraction = rho^2 = theta^2 * Var(pre) / Var(y)
	v_y := stats.variance(y_all)
	variance_reduction := if v_y > 0 { theta * theta * v_pre / v_y } else { 0.0 }

	// Adjust outcomes: y_adj = y - theta * (y_pre - mean(y_pre))
	mut y_ctrl_adj := []f64{len: y_ctrl.len}
	mut y_treat_adj := []f64{len: y_treat.len}
	for i in 0 .. y_ctrl.len {
		y_ctrl_adj[i] = y_ctrl[i] - theta * (pre_ctrl[i] - pre_mean)
	}
	for i in 0 .. y_treat.len {
		y_treat_adj[i] = y_treat[i] - theta * (pre_treat[i] - pre_mean)
	}

	adj_result := abtest(y_ctrl_adj, y_treat_adj, cfg)

	return CUPEDResult{
		theta:              theta
		variance_reduction: variance_reduction
		adjusted_result:    adj_result
	}
}
