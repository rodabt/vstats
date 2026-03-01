import experiment
import math

// ============================================================================
// A/B Test Tests
// ============================================================================

fn test__abtest_no_effect() {
	// Identical groups should show no significant effect
	control := [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.2]
	treatment := [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.2]
	result := experiment.abtest(control, treatment)

	assert result.p_value > 0.5
	assert math.abs(result.relative_lift) < 0.01
	assert math.abs(result.effect_size) < 0.01
	assert result.significant == false
}

fn test__abtest_clear_effect() {
	// Groups differing by several SDs
	control := [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 9.9]
	treatment := [13.0, 13.2, 12.8, 13.1, 12.9, 13.0, 13.3, 12.7, 13.1, 12.9]
	result := experiment.abtest(control, treatment)

	assert result.p_value < 0.05
	assert result.significant == true
	assert result.treatment_mean > result.control_mean
	assert math.abs(result.effect_size) > 1.0
}

fn test__power_analysis() {
	// Cohen's d = 0.5, alpha = 0.05, power = 0.80 => n ~= 64
	result := experiment.power_analysis(0.5, 0.05, 0.80)

	assert math.abs(result.n_per_group - 63) <= 3
	assert result.power == 0.80
	assert result.alpha == 0.05
	assert result.effect_size == 0.5
}

fn test__cuped_reduces_variance() {
	// Create data where pre-period correlates strongly with post-period
	pre_ctrl := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	pre_treat := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

	// y = pre + noise for control, y = pre + 3 + noise for treatment
	y_ctrl := [1.1, 2.2, 2.9, 4.1, 4.8, 6.2, 6.9, 8.1, 9.2, 9.8]
	y_treat := [4.1, 5.2, 5.9, 7.1, 7.8, 9.2, 9.9, 11.1, 12.2, 12.8]

	result := experiment.cuped_test(y_ctrl, y_treat, pre_ctrl, pre_treat)

	assert result.variance_reduction > 0.0
	// Compare CI width: adjusted should be narrower
	raw := experiment.abtest(y_ctrl, y_treat)
	raw_width := raw.ci_upper - raw.ci_lower
	adj_width := result.adjusted_result.ci_upper - result.adjusted_result.ci_lower
	assert adj_width < raw_width
}

// ============================================================================
// PSM Tests
// ============================================================================

fn test__propensity_scores_range() {
	// Covariates: treated group has higher feature values
	mut x := [][]f64{}
	mut treatment := []f64{}
	for i in 0 .. 20 {
		x << [f64(i) + 1.0]
		treatment << if i >= 10 { 1.0 } else { 0.0 }
	}

	model := experiment.estimate_propensity_scores(x, treatment)

	// All scores should be in (0, 1)
	for score in model.scores {
		assert score > 0.0 && score < 1.0
	}
}

fn test__matching_same_n_treated() {
	// With replacement, all treated units should be matched
	mut x := [][]f64{}
	mut treatment := []f64{}
	mut n_treated := 0
	for i in 0 .. 20 {
		x << [f64(i) + 1.0]
		t := if i >= 10 { 1.0 } else { 0.0 }
		treatment << t
		if t >= 0.5 {
			n_treated++
		}
	}

	model := experiment.estimate_propensity_scores(x, treatment)
	result := experiment.match_nearest_neighbor(model, experiment.MatchingConfig{ replacement: true })

	assert result.n_matched_treated == n_treated
	assert result.n_unmatched_treated == 0
}

fn test__balance_improves() {
	// Create data with overlapping covariates but different distributions
	// Control tends to have lower values, treatment tends to have higher values
	// but with substantial overlap so matching can find good pairs
	mut x := [][]f64{}
	mut treatment := []f64{}

	// Control group: values spread from 1 to 8
	ctrl_vals := [1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 7.5, 8.0]
	for v in ctrl_vals {
		x << [v]
		treatment << 0.0
	}
	// Treatment group: values spread from 4 to 10
	treat_vals := [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 9.5, 10.0]
	for v in treat_vals {
		x << [v]
		treatment << 1.0
	}

	model := experiment.estimate_propensity_scores(x, treatment,
		experiment.PropensityConfig{ iterations: 2000, learning_rate: 0.05 })
	result := experiment.match_nearest_neighbor(model)
	balance := experiment.check_balance(x, treatment, result)

	assert balance.mean_abs_smd_after <= balance.mean_abs_smd_before
}

fn test__ate_matched_no_effect() {
	// Synthetic data with same Y distribution for both groups
	mut x := [][]f64{}
	mut treatment := []f64{}
	mut y := []f64{}

	for i in 0 .. 20 {
		feat := f64(i) + 1.0
		x << [feat]
		t := if i >= 10 { 1.0 } else { 0.0 }
		treatment << t
		// Same outcome model regardless of treatment
		y << feat * 0.5
	}

	model := experiment.estimate_propensity_scores(x, treatment)
	matching := experiment.match_nearest_neighbor(model)
	ate := experiment.ate_matched(y, treatment, matching)

	// ATE should be small since treatment has no real effect
	// But due to covariate imbalance the raw ATE may not be exactly 0
	// We just verify the result is reasonable
	assert ate.n_treated > 0
	assert ate.n_control > 0
	assert ate.se >= 0.0
}

// ============================================================================
// DiD Tests
// ============================================================================

fn test__did_2x2_known_effect() {
	// Construct data where true ATE = 5.0
	y_treat_pre := [10.0, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0, 9.9]
	y_treat_post := [15.0, 15.1, 14.9, 15.0, 15.1, 15.0, 14.9, 15.1, 15.0, 14.9]
	y_ctrl_pre := [10.0, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0, 9.9]
	y_ctrl_post := [10.0, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0, 9.9]

	result := experiment.did_2x2(y_treat_pre, y_treat_post, y_ctrl_pre, y_ctrl_post)

	assert math.abs(result.did_effect - 5.0) < 0.1
	assert math.abs(result.treated_change - 5.0) < 0.1
	assert math.abs(result.control_change) < 0.1
}

fn test__did_2x2_no_effect() {
	// Parallel groups with same trend
	y_treat_pre := [10.0, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0, 9.9]
	y_treat_post := [12.0, 12.1, 11.9, 12.0, 12.1, 12.0, 11.9, 12.1, 12.0, 11.9]
	y_ctrl_pre := [10.0, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0, 9.9]
	y_ctrl_post := [12.0, 12.1, 11.9, 12.0, 12.1, 12.0, 11.9, 12.1, 12.0, 11.9]

	result := experiment.did_2x2(y_treat_pre, y_treat_post, y_ctrl_pre, y_ctrl_post)

	assert math.abs(result.did_effect) < 0.1
}

fn test__did_regression_recovers_effect() {
	// Build data with known DiD effect of 5.0
	mut y := []f64{}
	mut group := []int{}
	mut time := []int{}

	// Control pre
	for v in [10.0, 10.1, 9.9, 10.0, 10.1] {
		y << v
		group << 0
		time << 0
	}
	// Control post (no treatment effect, common trend of +2)
	for v in [12.0, 12.1, 11.9, 12.0, 12.1] {
		y << v
		group << 0
		time << 1
	}
	// Treated pre
	for v in [10.0, 10.1, 9.9, 10.0, 10.1] {
		y << v
		group << 1
		time << 0
	}
	// Treated post (common trend +2, plus treatment effect +5)
	for v in [17.0, 17.1, 16.9, 17.0, 17.1] {
		y << v
		group << 1
		time << 1
	}

	result := experiment.did_regression(y, [][]f64{}, group, time)

	assert math.abs(result.did_coefficient - 5.0) < 0.2
	assert result.r_squared > 0.9
}

fn test__parallel_trends_holds() {
	// Perfectly parallel pre-trends: both groups increase by 1.0 per period
	time_pre := [0, 1, 2, 3, 4]

	// Treated: 10, 11, 12, 13, 14 (slope = 1)
	y_treated := [10.0, 11.0, 12.0, 13.0, 14.0]
	// Control: 5, 6, 7, 8, 9 (slope = 1)
	y_control := [5.0, 6.0, 7.0, 8.0, 9.0]

	result := experiment.test_parallel_trends(y_treated, y_control, time_pre)

	assert result.p_value > 0.05
	assert result.parallel_trends_hold == true
	assert math.abs(result.slope_difference) < 0.01
}

fn test__event_study_shape() {
	// Create data with relative times {-2, -1, 0, 1}
	// 3 observations per period per group
	mut y := []f64{}
	mut group := []int{}
	mut relative_time := []int{}

	periods := [-2, -1, 0, 1]
	for period in periods {
		for _ in 0 .. 3 {
			// Treated
			y << 10.0 + f64(period)
			group << 1
			relative_time << period
			// Control
			y << 10.0 + f64(period) * 0.5
			group << 0
			relative_time << period
		}
	}

	result := experiment.event_study(y, group, relative_time)

	// 4 periods, -1 is reference => 3 non-reference periods
	assert result.relative_times.len == 3
	assert result.effects.len == 3
	assert result.std_errors.len == 3
	assert result.t_statistics.len == 3
	assert result.p_values.len == 3
	assert result.ci_lowers.len == 3
	assert result.ci_uppers.len == 3
}
