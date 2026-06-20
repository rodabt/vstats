module experiment

import math
import stats
import prob
import ml

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

pub struct ProportionPowerResult {
pub:
	n_per_group int
	power       f64
	alpha       f64
	p_baseline  f64
	p_treatment f64
	mde         f64
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

// t_pvalue returns the two-tailed p-value for a t-statistic with df degrees of freedom.
fn t_pvalue(t f64, df int) f64 {
	return 2.0 * prob.students_t_cdf(-math.abs(t), df)
}

// t_critical returns the t critical value for a two-tailed CI at the given alpha level.
fn t_critical(alpha f64, df int) f64 {
	return prob.inverse_students_t_cdf(1.0 - alpha / 2.0, df)
}

// t_pvalue_approx computes two-tailed p-value via normal CDF approximation
fn t_pvalue_approx(t f64, df f64) f64 {
	return t_pvalue(t, int(df))
}

// welch_ci computes the confidence interval for the difference in means
fn welch_ci(m1 f64, m2 f64, v1 f64, v2 f64, n1 int, n2 int, alpha f64, df f64) (f64, f64) {
	a := v1 / f64(n1)
	b := v2 / f64(n2)
	se := math.sqrt(a + b)
	crit := t_critical(alpha, int(df))
	diff := m1 - m2
	return diff - crit * se, diff + crit * se
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
	ci_lo, ci_hi := welch_ci(m_t, m_c, v_t, v_c, n_t, n_c, cfg.alpha, df)
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

// proportion_power_analysis computes the required sample size per group for binary outcomes
// (conversion rates, click-through rates) using the two-proportion z-test formula.
// Use this instead of power_analysis when your metric is a 0/1 proportion.
pub fn proportion_power_analysis(p_baseline f64, p_treatment f64, alpha f64, power f64) ProportionPowerResult {
	assert p_baseline > 0 && p_baseline < 1, 'p_baseline must be in (0, 1)'
	assert p_treatment > 0 && p_treatment < 1, 'p_treatment must be in (0, 1)'
	assert p_baseline != p_treatment, 'p_baseline and p_treatment must differ'
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)
	p_bar := (p_baseline + p_treatment) / 2.0
	se_null := math.sqrt(2.0 * p_bar * (1.0 - p_bar))
	se_alt := math.sqrt(p_baseline * (1.0 - p_baseline) + p_treatment * (1.0 - p_treatment))
	diff := math.abs(p_treatment - p_baseline)
	n_raw := math.pow(z_alpha * se_null + z_beta * se_alt, 2) / (diff * diff)
	n := if n_raw > f64(int(n_raw)) { int(n_raw) + 1 } else { int(n_raw) }
	return ProportionPowerResult{
		n_per_group: n
		power:       power
		alpha:       alpha
		p_baseline:  p_baseline
		p_treatment: p_treatment
		mde:         diff
	}
}

pub struct ANCOVAResult {
pub:
	adjusted_effect f64
	se              f64
	t_statistic     f64
	p_value         f64
	ci_lower        f64
	ci_upper        f64
	significant     bool
	n_control       int
	n_treatment     int
}

// ancova runs an ANCOVA-adjusted A/B test: fits y ~ treatment + covariates via OLS
// and returns the treatment coefficient with its standard error and p-value.
// x_ctrl and x_trt are covariate matrices (one row per unit, one column per covariate).
// Pass empty slices ([][]f64{}) when there are no covariates.
pub fn ancova(ctrl []f64, trt []f64, x_ctrl [][]f64, x_trt [][]f64, cfg ABTestConfig) ANCOVAResult {
	assert ctrl.len >= 2 && trt.len >= 2, 'each group needs at least 2 observations'
	n_c := ctrl.len
	n_t := trt.len
	n := n_c + n_t

	n_cov := if x_ctrl.len > 0 { x_ctrl[0].len } else { 0 }
	n_params := 1 + n_cov // treatment indicator + covariates

	// Build design matrix (no intercept column — ols_se adds it)
	mut x_design := [][]f64{len: n}
	for i in 0 .. n_c {
		x_design[i] = []f64{len: n_params}
		x_design[i][0] = 0.0
		for j in 0 .. n_cov {
			x_design[i][1 + j] = x_ctrl[i][j]
		}
	}
	for i in 0 .. n_t {
		x_design[n_c + i] = []f64{len: n_params}
		x_design[n_c + i][0] = 1.0
		for j in 0 .. n_cov {
			x_design[n_c + i][1 + j] = x_trt[i][j]
		}
	}

	mut y := []f64{len: n}
	for i in 0 .. n_c {
		y[i] = ctrl[i]
	}
	for i in 0 .. n_t {
		y[n_c + i] = trt[i]
	}

	model := ml.linear_regression(x_design, y)

	// treatment coefficient is model.coefficients[0]
	adj_effect := model.coefficients[0]

	mut full_coefs := [model.intercept]
	for c in model.coefficients {
		full_coefs << c
	}
	se_vec := ols_se(x_design, y, full_coefs)
	// se_vec[0]=intercept, se_vec[1]=treatment
	se_trt := if se_vec.len > 1 { se_vec[1] } else { 0.0 }

	t_stat := if se_trt > 0 { adj_effect / se_trt } else { 0.0 }
	ancova_df := n - (n_cov + 2)
	p_val := t_pvalue(t_stat, ancova_df)
	z_crit := t_critical(cfg.alpha, ancova_df)

	return ANCOVAResult{
		adjusted_effect: adj_effect
		se:              se_trt
		t_statistic:     t_stat
		p_value:         p_val
		ci_lower:        adj_effect - z_crit * se_trt
		ci_upper:        adj_effect + z_crit * se_trt
		significant:     p_val < cfg.alpha
		n_control:       n_c
		n_treatment:     n_t
	}
}

// null_verdict returns a plain-English readout verdict for an A/B test result.
// sig_digits controls decimal places in the formatted numbers (default 3).
pub fn null_verdict(result ABTestResult, alpha f64) string {
	dir := if result.treatment_mean > result.control_mean { 'higher' } else { 'lower' }
	lift_pct := result.relative_lift * 100.0
	lift_str := if lift_pct >= 0 { '+${lift_pct:.1f}%' } else { '${lift_pct:.1f}%' }
	if result.p_value < alpha {
		return 'Significant: treatment is ${dir} than control (lift ${lift_str}, ' +
			'p=${result.p_value:.4f}, 95% CI [${result.ci_lower:.4f}, ${result.ci_upper:.4f}]).'
	} else {
		return 'Not significant: insufficient evidence that treatment differs from control ' +
			'(lift ${lift_str}, p=${result.p_value:.4f}, 95% CI [${result.ci_lower:.4f}, ${result.ci_upper:.4f}]).'
	}
}

pub struct ITTPPResult {
pub:
	itt ABTestResult
	pp  ABTestResult
}

// itt_and_pp runs both Intent-to-Treat and Per-Protocol A/B tests.
// assigned: 0=control, 1=treatment (by randomization).
// complied: true if the unit actually received their assigned treatment.
// y: outcome for each unit.
// ITT uses all units as assigned. PP excludes non-compliers.
pub fn itt_and_pp(y []f64, assigned []int, complied []bool, cfg ABTestConfig) ITTPPResult {
	assert y.len == assigned.len && y.len == complied.len, 'y, assigned, complied must have same length'

	// ITT: split by assignment, regardless of compliance
	mut y_ctrl_itt := []f64{}
	mut y_trt_itt  := []f64{}
	for i in 0 .. y.len {
		if assigned[i] == 0 {
			y_ctrl_itt << y[i]
		} else {
			y_trt_itt << y[i]
		}
	}

	// PP: keep only compliers in each arm
	mut y_ctrl_pp := []f64{}
	mut y_trt_pp  := []f64{}
	for i in 0 .. y.len {
		if !complied[i] { continue }
		if assigned[i] == 0 {
			y_ctrl_pp << y[i]
		} else {
			y_trt_pp << y[i]
		}
	}

	assert y_ctrl_itt.len >= 2 && y_trt_itt.len >= 2, 'ITT: each arm needs at least 2 units'
	assert y_ctrl_pp.len >= 2 && y_trt_pp.len >= 2, 'PP: each compliant arm needs at least 2 units'

	return ITTPPResult{
		itt: abtest(y_ctrl_itt, y_trt_itt, cfg)
		pp:  abtest(y_ctrl_pp,  y_trt_pp,  cfg)
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
