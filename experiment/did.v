module experiment

import math
import stats
import prob
import ml
import linalg

@[params]
pub struct DiDConfig {
pub:
	alpha f64 = 0.05
}

pub struct DiDResult {
pub:
	did_effect     f64
	se             f64
	t_statistic    f64
	p_value        f64
	ci_lower       f64
	ci_upper       f64
	treated_change f64
	control_change f64
	n_treated_pre  int
	n_treated_post int
	n_control_pre  int
	n_control_post int
}

pub struct DiDRegressionResult {
pub:
	did_coefficient f64
	did_se          f64
	did_t_stat      f64
	did_p_value     f64
	did_ci_lower    f64
	did_ci_upper    f64
	r_squared       f64
	n               int
}

pub struct ParallelTrendsResult {
pub:
	slope_treated        f64
	slope_control        f64
	slope_difference     f64
	t_statistic          f64
	p_value              f64
	parallel_trends_hold bool
}

pub struct EventStudyResult {
pub:
	relative_times []int
	effects        []f64
	std_errors     []f64
	t_statistics   []f64
	p_values       []f64
	ci_lowers      []f64
	ci_uppers      []f64
}

// matrix_inverse computes the inverse of a square matrix via Gaussian elimination
fn matrix_inverse(m [][]f64) [][]f64 {
	n_dim := m.len
	mut result := [][]f64{len: n_dim}
	for i in 0 .. n_dim {
		result[i] = []f64{len: n_dim}
	}
	for j in 0 .. n_dim {
		mut e := []f64{len: n_dim, init: 0.0}
		e[j] = 1.0
		col := linalg.gaussian_elimination(m, e)
		for i in 0 .. n_dim {
			result[i][j] = col[i]
		}
	}
	return result
}

// ols_se computes OLS standard errors from residuals
// x_mat is the design matrix WITHOUT intercept; coefficients is [intercept, coef_1, ..., coef_p]
fn ols_se(x_mat [][]f64, y []f64, coefficients []f64) []f64 {
	n := x_mat.len
	n_params := coefficients.len

	if n <= n_params {
		return []f64{len: n_params, init: 0.0}
	}

	// Build augmented X with intercept column prepended
	mut x_aug := [][]f64{len: n}
	for i in 0 .. n {
		x_aug[i] = []f64{len: n_params}
		x_aug[i][0] = 1.0
		for j in 1 .. n_params {
			if j - 1 < x_mat[i].len {
				x_aug[i][j] = x_mat[i][j - 1]
			}
		}
	}

	// Compute sum of squared residuals
	mut ssr := 0.0
	for i in 0 .. n {
		mut y_pred := 0.0
		for j in 0 .. n_params {
			y_pred += coefficients[j] * x_aug[i][j]
		}
		resid := y[i] - y_pred
		ssr += resid * resid
	}
	s2 := ssr / f64(n - n_params)

	// Compute (X'X)^{-1}
	xt := linalg.transpose(x_aug)
	xtx := linalg.matmul(xt, x_aug)
	xtx_inv := matrix_inverse(xtx)

	// SE_j = sqrt(s^2 * [(X'X)^{-1}]_{jj})
	mut se := []f64{len: n_params}
	for j in 0 .. n_params {
		v_j := s2 * xtx_inv[j][j]
		se[j] = if v_j > 0 { math.sqrt(v_j) } else { 0.0 }
	}
	return se
}

// did_ci computes a confidence interval via normal approximation
fn did_ci(effect f64, se f64, alpha f64) (f64, f64) {
	z := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	return effect - z * se, effect + z * se
}

// did_2x2 computes the simple 2x2 Difference-in-Differences estimator
pub fn did_2x2(y_treat_pre []f64, y_treat_post []f64, y_ctrl_pre []f64, y_ctrl_post []f64, cfg DiDConfig) DiDResult {
	assert y_treat_pre.len >= 2 && y_treat_post.len >= 2, 'treated groups need at least 2 observations'
	assert y_ctrl_pre.len >= 2 && y_ctrl_post.len >= 2, 'control groups need at least 2 observations'

	m_tp := stats.mean(y_treat_post)
	m_t0 := stats.mean(y_treat_pre)
	m_cp := stats.mean(y_ctrl_post)
	m_c0 := stats.mean(y_ctrl_pre)

	treated_change := m_tp - m_t0
	control_change := m_cp - m_c0
	did := treated_change - control_change

	// SE via delta method: Var(DiD) = sum of Var(mean) for each cell
	v_tp := stats.variance(y_treat_post) / f64(y_treat_post.len)
	v_t0 := stats.variance(y_treat_pre) / f64(y_treat_pre.len)
	v_cp := stats.variance(y_ctrl_post) / f64(y_ctrl_post.len)
	v_c0 := stats.variance(y_ctrl_pre) / f64(y_ctrl_pre.len)
	se := math.sqrt(v_tp + v_t0 + v_cp + v_c0)

	t_stat := if se > 0 { did / se } else { 0.0 }
	p_val := 2.0 * prob.normal_cdf(-math.abs(t_stat), 0.0, 1.0)
	ci_lo, ci_hi := did_ci(did, se, cfg.alpha)

	return DiDResult{
		did_effect:     did
		se:             se
		t_statistic:    t_stat
		p_value:        p_val
		ci_lower:       ci_lo
		ci_upper:       ci_hi
		treated_change: treated_change
		control_change: control_change
		n_treated_pre:  y_treat_pre.len
		n_treated_post: y_treat_post.len
		n_control_pre:  y_ctrl_pre.len
		n_control_post: y_ctrl_post.len
	}
}

// did_regression estimates DiD via OLS with interaction term
pub fn did_regression(y []f64, x [][]f64, group []int, time []int, cfg DiDConfig) DiDRegressionResult {
	assert y.len == group.len && y.len == time.len, 'y, group, and time must have same length'

	n := y.len
	n_extra := if x.len > 0 { x[0].len } else { 0 }
	n_features := 3 + n_extra

	// Build design matrix: [treat, post, treat*post, x_covariates...]
	mut x_design := [][]f64{len: n}
	for i in 0 .. n {
		x_design[i] = []f64{len: n_features}
		x_design[i][0] = f64(group[i])
		x_design[i][1] = f64(time[i])
		x_design[i][2] = f64(group[i]) * f64(time[i])
		for j in 0 .. n_extra {
			x_design[i][3 + j] = x[i][j]
		}
	}

	model := ml.linear_regression(x_design, y)

	// DiD coefficient is on the interaction term (index 2 in model.coefficients)
	did_coef := model.coefficients[2]

	// Compute OLS standard errors
	mut full_coefs := []f64{}
	full_coefs << model.intercept
	for c in model.coefficients {
		full_coefs << c
	}
	se := ols_se(x_design, y, full_coefs)
	did_se_val := if se.len > 3 { se[3] } else { 0.0 }

	t_stat := if did_se_val > 0 { did_coef / did_se_val } else { 0.0 }
	p_val := 2.0 * prob.normal_cdf(-math.abs(t_stat), 0.0, 1.0)
	ci_lo, ci_hi := did_ci(did_coef, did_se_val, cfg.alpha)

	// R-squared
	preds := ml.linear_predict(model, x_design)
	r2 := ml.r_squared(y, preds)

	return DiDRegressionResult{
		did_coefficient: did_coef
		did_se:          did_se_val
		did_t_stat:      t_stat
		did_p_value:     p_val
		did_ci_lower:    ci_lo
		did_ci_upper:    ci_hi
		r_squared:       r2
		n:               n
	}
}

// test_parallel_trends tests whether pre-period trends are parallel between groups
pub fn test_parallel_trends(y_treated_pre []f64, y_control_pre []f64, time_pre []int, cfg DiDConfig) ParallelTrendsResult {
	assert y_treated_pre.len == time_pre.len, 'treated pre and time_pre must have same length'
	assert y_control_pre.len == time_pre.len, 'control pre and time_pre must have same length'

	n_t := y_treated_pre.len
	n_c := y_control_pre.len
	n := n_t + n_c

	// Pool data
	mut y_all := []f64{len: n}
	for i in 0 .. n_t {
		y_all[i] = y_treated_pre[i]
	}
	for i in 0 .. n_c {
		y_all[n_t + i] = y_control_pre[i]
	}

	// Design matrix: [group, time, group*time]
	mut x_design := [][]f64{len: n}
	for i in 0 .. n_t {
		t := f64(time_pre[i])
		x_design[i] = [1.0, t, t]
	}
	for i in 0 .. n_c {
		t := f64(time_pre[i])
		x_design[n_t + i] = [0.0, t, 0.0]
	}

	model := ml.linear_regression(x_design, y_all)

	// model.coefficients: [beta_group, beta_time, beta_interaction]
	slope_c := model.coefficients[1]
	slope_t := model.coefficients[1] + model.coefficients[2]
	slope_diff := model.coefficients[2]

	// SE of interaction coefficient
	mut full_coefs := []f64{}
	full_coefs << model.intercept
	for c in model.coefficients {
		full_coefs << c
	}
	se := ols_se(x_design, y_all, full_coefs)
	interaction_se := if se.len > 3 { se[3] } else { 0.0 }

	t_stat := if interaction_se > 0 { slope_diff / interaction_se } else { 0.0 }
	p_val := 2.0 * prob.normal_cdf(-math.abs(t_stat), 0.0, 1.0)

	return ParallelTrendsResult{
		slope_treated:        slope_t
		slope_control:        slope_c
		slope_difference:     slope_diff
		t_statistic:          t_stat
		p_value:              p_val
		parallel_trends_hold: p_val > cfg.alpha
	}
}

// event_study computes period-by-period DiD estimates relative to period -1
pub fn event_study(y []f64, group []int, relative_time []int, cfg DiDConfig) EventStudyResult {
	n := y.len

	// Find unique periods excluding reference period (-1)
	mut seen := map[int]bool{}
	mut periods := []int{}
	for t in relative_time {
		if t != -1 && !seen[t] {
			seen[t] = true
			periods << t
		}
	}
	periods.sort()

	// Extract reference period data
	mut y_treat_ref := []f64{}
	mut y_ctrl_ref := []f64{}
	for i in 0 .. n {
		if relative_time[i] == -1 {
			if group[i] == 1 {
				y_treat_ref << y[i]
			} else {
				y_ctrl_ref << y[i]
			}
		}
	}

	k := periods.len
	mut effects := []f64{len: k}
	mut std_errors := []f64{len: k}
	mut t_statistics := []f64{len: k}
	mut p_values := []f64{len: k}
	mut ci_lowers := []f64{len: k}
	mut ci_uppers := []f64{len: k}

	for idx, period in periods {
		mut y_treat_t := []f64{}
		mut y_ctrl_t := []f64{}
		for i in 0 .. n {
			if relative_time[i] == period {
				if group[i] == 1 {
					y_treat_t << y[i]
				} else {
					y_ctrl_t << y[i]
				}
			}
		}

		if y_treat_t.len >= 2 && y_ctrl_t.len >= 2 && y_treat_ref.len >= 2
			&& y_ctrl_ref.len >= 2 {
			did_result := did_2x2(y_treat_ref, y_treat_t, y_ctrl_ref, y_ctrl_t, cfg)
			effects[idx] = did_result.did_effect
			std_errors[idx] = did_result.se
			t_statistics[idx] = did_result.t_statistic
			p_values[idx] = did_result.p_value
			ci_lowers[idx] = did_result.ci_lower
			ci_uppers[idx] = did_result.ci_upper
		}
	}

	return EventStudyResult{
		relative_times: periods
		effects:        effects
		std_errors:     std_errors
		t_statistics:   t_statistics
		p_values:       p_values
		ci_lowers:      ci_lowers
		ci_uppers:      ci_uppers
	}
}
