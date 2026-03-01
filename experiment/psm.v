module experiment

import math
import stats
import ml
import hypothesis

pub struct PropensityModel {
pub:
	logistic_model ml.LogisticModel[f64]
	scores         []f64
	treatment      []f64
}

@[params]
pub struct PropensityConfig {
pub:
	iterations    int = 1000
	learning_rate f64 = 0.1
	trim          f64 = 0.0
}

pub struct MatchedPair {
pub:
	treated_idx int
	control_idx int
	ps_distance f64
}

@[params]
pub struct MatchingConfig {
pub:
	caliper     f64  = -1.0
	replacement bool = true
}

pub struct MatchingResult {
pub:
	pairs               []MatchedPair
	n_matched_treated   int
	n_unmatched_treated int
	avg_distance        f64
}

pub struct BalanceResult {
pub:
	smd_before          []f64
	smd_after           []f64
	mean_abs_smd_before f64
	mean_abs_smd_after  f64
	balanced            bool
}

pub struct ATEResult {
pub:
	ate         f64
	se          f64
	ci_lower    f64
	ci_upper    f64
	t_statistic f64
	p_value     f64
	n_treated   int
	n_control   int
}

// smd computes standardised mean difference for a single covariate
fn smd_calc(x_treated []f64, x_control []f64) f64 {
	if x_treated.len < 2 || x_control.len < 2 {
		return 0.0
	}
	mu_t := stats.mean(x_treated)
	mu_c := stats.mean(x_control)
	v_t := stats.variance(x_treated)
	v_c := stats.variance(x_control)
	denom := math.sqrt((v_t + v_c) / 2.0)
	return if denom > 0 { (mu_t - mu_c) / denom } else { 0.0 }
}

// estimate_propensity_scores fits a logistic regression model and returns propensity scores
pub fn estimate_propensity_scores(x [][]f64, treatment []f64, cfg PropensityConfig) PropensityModel {
	assert x.len == treatment.len, 'x and treatment must have same length'

	config := ml.OptimLogisticConfig{
		iterations:    cfg.iterations
		batch_size:    x.len
		learning_rate: cfg.learning_rate
		momentum:      0.9
		lambda:        0.01
		shuffle:       false
	}
	model := ml.logistic_regression_fast(x, treatment, config)
	mut scores := ml.logistic_predict_proba(model, x)

	// Common-support trimming
	if cfg.trim > 0 {
		lo := cfg.trim
		hi := 1.0 - cfg.trim
		for i in 0 .. scores.len {
			if scores[i] < lo {
				scores[i] = lo
			}
			if scores[i] > hi {
				scores[i] = hi
			}
		}
	}

	return PropensityModel{
		logistic_model: model
		scores:         scores
		treatment:      treatment
	}
}

// match_nearest_neighbor performs greedy nearest-neighbor matching on propensity scores
pub fn match_nearest_neighbor(model PropensityModel, cfg MatchingConfig) MatchingResult {
	scores := model.scores
	treatment := model.treatment
	n := scores.len

	mut treated_idx := []int{}
	mut control_idx := []int{}
	for i in 0 .. n {
		if treatment[i] >= 0.5 {
			treated_idx << i
		} else {
			control_idx << i
		}
	}

	mut used_control := []bool{len: n, init: false}
	mut pairs := []MatchedPair{}

	for ti in treated_idx {
		mut best_dist := 1e18
		mut best_ci := -1
		for ci in control_idx {
			if !cfg.replacement && used_control[ci] {
				continue
			}
			dist := math.abs(scores[ti] - scores[ci])
			if dist < best_dist {
				best_dist = dist
				best_ci = ci
			}
		}
		if best_ci >= 0 && (cfg.caliper < 0 || best_dist <= cfg.caliper) {
			pairs << MatchedPair{
				treated_idx: ti
				control_idx: best_ci
				ps_distance: best_dist
			}
			if !cfg.replacement {
				used_control[best_ci] = true
			}
		}
	}

	mut total_dist := 0.0
	for p in pairs {
		total_dist += p.ps_distance
	}
	avg_dist := if pairs.len > 0 { total_dist / f64(pairs.len) } else { 0.0 }

	return MatchingResult{
		pairs:               pairs
		n_matched_treated:   pairs.len
		n_unmatched_treated: treated_idx.len - pairs.len
		avg_distance:        avg_dist
	}
}

// check_balance computes standardised mean differences before and after matching
pub fn check_balance(x [][]f64, treatment []f64, result MatchingResult) BalanceResult {
	if x.len == 0 {
		return BalanceResult{}
	}
	n_cols := x[0].len
	n := x.len

	// Before matching: split all observations by treatment
	mut x_t_cols := [][]f64{}
	mut x_c_cols := [][]f64{}
	for _ in 0 .. n_cols {
		x_t_cols << []f64{}
		x_c_cols << []f64{}
	}
	for i in 0 .. n {
		for j in 0 .. n_cols {
			if treatment[i] >= 0.5 {
				x_t_cols[j] << x[i][j]
			} else {
				x_c_cols[j] << x[i][j]
			}
		}
	}

	mut smd_before := []f64{len: n_cols}
	for j in 0 .. n_cols {
		smd_before[j] = smd_calc(x_t_cols[j], x_c_cols[j])
	}

	// After matching: use matched pairs
	mut x_t_matched := [][]f64{}
	mut x_c_matched := [][]f64{}
	for _ in 0 .. n_cols {
		x_t_matched << []f64{}
		x_c_matched << []f64{}
	}
	for pair in result.pairs {
		for j in 0 .. n_cols {
			x_t_matched[j] << x[pair.treated_idx][j]
			x_c_matched[j] << x[pair.control_idx][j]
		}
	}

	mut smd_after := []f64{len: n_cols}
	for j in 0 .. n_cols {
		smd_after[j] = smd_calc(x_t_matched[j], x_c_matched[j])
	}

	mut sum_before := 0.0
	mut sum_after := 0.0
	for j in 0 .. n_cols {
		sum_before += math.abs(smd_before[j])
		sum_after += math.abs(smd_after[j])
	}
	mean_before := if n_cols > 0 { sum_before / f64(n_cols) } else { 0.0 }
	mean_after := if n_cols > 0 { sum_after / f64(n_cols) } else { 0.0 }

	mut all_balanced := true
	for j in 0 .. n_cols {
		if math.abs(smd_after[j]) >= 0.1 {
			all_balanced = false
			break
		}
	}

	return BalanceResult{
		smd_before:          smd_before
		smd_after:           smd_after
		mean_abs_smd_before: mean_before
		mean_abs_smd_after:  mean_after
		balanced:            all_balanced
	}
}

// ate_matched computes the average treatment effect from matched pairs
pub fn ate_matched(y []f64, treatment []f64, result MatchingResult) ATEResult {
	_ = treatment

	mut y_treated := []f64{}
	mut y_control := []f64{}
	for pair in result.pairs {
		y_treated << y[pair.treated_idx]
		y_control << y[pair.control_idx]
	}

	if y_treated.len < 2 || y_control.len < 2 {
		return ATEResult{}
	}

	ate := stats.mean(y_treated) - stats.mean(y_control)
	v_t := stats.variance(y_treated)
	v_c := stats.variance(y_control)
	n_t := f64(y_treated.len)
	n_c := f64(y_control.len)
	se := math.sqrt(v_t / n_t + v_c / n_c)

	t_stat, p_val := hypothesis.t_test_two_sample(y_treated, y_control)

	z := 1.96
	ci_lo := ate - z * se
	ci_hi := ate + z * se

	return ATEResult{
		ate:         ate
		se:          se
		ci_lower:    ci_lo
		ci_upper:    ci_hi
		t_statistic: t_stat
		p_value:     p_val
		n_treated:   y_treated.len
		n_control:   y_control.len
	}
}
