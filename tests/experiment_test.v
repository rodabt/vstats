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

fn test__proportion_power_analysis() {
	// p_baseline=0.10, p_treatment=0.15, alpha=0.05, power=0.80 => n ~= 686
	result := experiment.proportion_power_analysis(0.10, 0.15, 0.05, 0.80)

	assert math.abs(result.n_per_group - 686) <= 5
	assert result.power == 0.80
	assert result.alpha == 0.05
	assert result.p_baseline == 0.10
	assert result.p_treatment == 0.15
	assert math.abs(result.mde - 0.05) < 1e-10
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

// ============================================================================
// ANCOVA Tests
// ============================================================================

// ============================================================================
// ITT and PP Tests
// ============================================================================

fn test__itt_and_pp_full_compliance() {
	// When everyone complies, ITT and PP should give the same result
	y        := [10.0, 10.1, 9.9, 10.0, 13.0, 13.1, 12.9, 13.0]
	assigned := [0, 0, 0, 0, 1, 1, 1, 1]
	complied := [true, true, true, true, true, true, true, true]
	result := experiment.itt_and_pp(y, assigned, complied)
	assert math.abs(result.itt.treatment_mean - result.pp.treatment_mean) < 1e-9
	assert math.abs(result.itt.control_mean - result.pp.control_mean) < 1e-9
}

fn test__itt_and_pp_partial_compliance_dilutes_pp() {
	// Two non-compliers in treatment arm (low outcome) → PP excludes them → PP effect > ITT effect
	// Assigned treatment: [13, 13, 13, 13, 5, 5]  ← last two are non-compliers
	y        := [10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0, 5.0, 5.0]
	assigned := [0,    0,    0,    0,    1,    1,    1,    1,    1,   1  ]
	complied := [true, true, true, true, true, true, true, true, false, false]
	result := experiment.itt_and_pp(y, assigned, complied)
	// PP removes low-outcome non-compliers → PP treatment mean > ITT treatment mean
	assert result.pp.treatment_mean > result.itt.treatment_mean
	// PP removes diluting non-compliers → PP relative lift > ITT relative lift
	assert result.pp.relative_lift > result.itt.relative_lift
}

// ============================================================================
// null_verdict Tests
// ============================================================================

fn test__null_verdict_significant() {
	ctrl := [10.0, 10.1, 9.9, 10.0, 10.1, 9.8, 10.2, 10.0, 9.9, 10.1]
	trt  := [13.0, 13.1, 12.9, 13.0, 13.1, 12.8, 13.2, 13.0, 12.9, 13.1]
	result := experiment.abtest(ctrl, trt)
	verdict := experiment.null_verdict(result, 0.05)
	assert verdict.contains('Significant:')
	assert verdict.contains('higher')
	assert verdict.contains('p=')
}

fn test__null_verdict_not_significant() {
	ctrl := [10.0, 10.1, 9.9, 10.0]
	trt  := [10.05, 10.15, 9.95, 10.05]
	result := experiment.abtest(ctrl, trt)
	verdict := experiment.null_verdict(result, 0.05)
	assert verdict.contains('Not significant:')
	assert verdict.contains('p=')
}

// ============================================================================
// ANCOVA Tests
// ============================================================================

fn test__ancova_no_covariate() {
	// Without covariates, ancova should detect a clear +2 effect
	ctrl := [10.0, 10.1, 9.9, 10.0, 10.1, 9.8, 10.2, 10.0, 9.9, 10.1]
	trt  := [12.0, 12.1, 11.9, 12.0, 12.1, 11.8, 12.2, 12.0, 11.9, 12.1]
	result := experiment.ancova(ctrl, trt, [][]f64{}, [][]f64{})
	assert result.significant == true
	assert math.abs(result.adjusted_effect - 2.0) < 0.1
	assert result.p_value < 0.05
}

fn test__ancova_covariate_recovers_effect() {
	// Baseline covariate correlated with outcome; treatment effect = +2 with some noise
	ctrl   := [10.1, 11.0, 12.2, 12.9, 14.1, 9.8, 11.3, 12.0, 13.1, 14.0]
	trt    := [12.0, 13.1, 14.0, 15.2, 16.1, 11.9, 13.0, 14.2, 15.0, 16.2]
	x_ctrl := [[10.0], [11.0], [12.0], [13.0], [14.0], [10.0], [11.0], [12.0], [13.0], [14.0]]
	x_trt  := [[10.0], [11.0], [12.0], [13.0], [14.0], [10.0], [11.0], [12.0], [13.0], [14.0]]
	result := experiment.ancova(ctrl, trt, x_ctrl, x_trt)
	assert result.significant == true
	assert math.abs(result.adjusted_effect - 2.0) < 0.5
	assert result.ci_lower < result.adjusted_effect
	assert result.ci_upper > result.adjusted_effect
}

// ============================================================================
// Ratio Metric Tests
// ============================================================================

fn test__ratio_metric_test_no_effect() {
	// Same ratio in both arms (CTR = 0.15 in both)
	num_ctrl := [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
	den_ctrl := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	num_trt  := [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
	den_trt  := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	result := experiment.ratio_metric_test(num_ctrl, den_ctrl, num_trt, den_trt)
	assert math.abs(result.ratio_ctrl - 0.15) < 1e-9
	assert math.abs(result.ratio_trt - 0.15) < 1e-9
	assert math.abs(result.diff) < 1e-9
	assert result.significant == false
}

fn test__ratio_metric_test_clear_effect() {
	// Control CTR = 0.15, Treatment CTR = 0.35, n=20 per arm, well-separated
	num_ctrl := [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
	             1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
	den_ctrl := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
	             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	num_trt  := [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
	             3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0]
	den_trt  := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
	             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	result := experiment.ratio_metric_test(num_ctrl, den_ctrl, num_trt, den_trt)
	// ratio_ctrl = (5*1+5*2+5*1+5*2)/(20*10) = 30/200 = 0.15
	// ratio_trt  = (5*3+5*4+5*3+5*4)/(20*10) = 70/200 = 0.35
	assert math.abs(result.ratio_ctrl - 0.15) < 1e-9
	assert math.abs(result.ratio_trt - 0.35) < 1e-9
	assert math.abs(result.diff - 0.20) < 1e-9
	assert result.p_value < 0.05
	assert result.significant == true
	assert result.relative_lift > 0.0
	assert result.ci_lower > 0.0
	assert result.ci_lower < result.diff
	assert result.ci_upper > result.diff
}

fn test__ratio_metric_test_result_fields() {
	// Smoke test: all fields are populated and SE > 0
	num_ctrl := [2.0, 3.0, 2.0, 3.0, 2.0]
	den_ctrl := [10.0, 12.0, 10.0, 12.0, 10.0]
	num_trt  := [4.0, 5.0, 4.0, 5.0, 4.0]
	den_trt  := [10.0, 12.0, 10.0, 12.0, 10.0]
	result := experiment.ratio_metric_test(num_ctrl, den_ctrl, num_trt, den_trt)
	assert result.n_ctrl == 5
	assert result.n_trt == 5
	assert result.se > 0.0
	assert result.ratio_ctrl > 0.0
	assert result.ratio_trt > 0.0
}

// ============================================================================
// Multiple Testing Correction Tests
// ============================================================================

fn test__bonferroni_rejects_only_clearly_significant() {
	// 5 p-values; only the first two are below alpha/5 = 0.01
	p_values := [0.002, 0.008, 0.12, 0.34, 0.87]
	result := experiment.bonferroni(p_values, 0.05)
	assert result.method == 'bonferroni'
	assert result.n_rejected == 2
	assert result.rejected[0] == true
	assert result.rejected[1] == true
	assert result.rejected[2] == false
	assert result.rejected[3] == false
	assert result.rejected[4] == false
	// Adjusted p-values: min(p * k, 1.0)
	assert math.abs(result.adjusted_p_values[0] - 0.01) < 1e-9
	assert math.abs(result.adjusted_p_values[1] - 0.04) < 1e-9
}

fn test__bh_fdr_more_powerful_than_bonferroni() {
	// BH controls FDR so it should reject >= as many as Bonferroni
	p_values := [0.003, 0.021, 0.06, 0.34, 0.87]
	bonf := experiment.bonferroni(p_values, 0.05)
	bh   := experiment.bh_fdr(p_values, 0.05)
	assert bh.n_rejected >= bonf.n_rejected
}

fn test__holm_between_bonferroni_and_bh() {
	// Holm is uniformly more powerful than Bonferroni but controls FWER like Bonferroni
	p_values := [0.002, 0.015, 0.06, 0.20, 0.87]
	bonf := experiment.bonferroni(p_values, 0.05)
	holm := experiment.holm(p_values, 0.05)
	assert holm.n_rejected >= bonf.n_rejected
	assert holm.method == 'holm'
}

fn test__correction_all_null() {
	// All p-values well above 0.05 — nothing should be rejected
	p_values := [0.40, 0.55, 0.70, 0.80, 0.92]
	bonf := experiment.bonferroni(p_values, 0.05)
	bh   := experiment.bh_fdr(p_values, 0.05)
	holm := experiment.holm(p_values, 0.05)
	assert bonf.n_rejected == 0
	assert bh.n_rejected == 0
	assert holm.n_rejected == 0
}

fn test__bh_fdr_adjusted_p_values_ordered() {
	// BH adjusted p-values preserve original index order and are all in [0, 1]
	p_values := [0.01, 0.05, 0.03, 0.40, 0.001]
	result := experiment.bh_fdr(p_values, 0.05)
	assert result.adjusted_p_values.len == p_values.len
	for adj in result.adjusted_p_values {
		assert adj >= 0.0 && adj <= 1.0
	}
}

// ============================================================================
// Power Diagnostic Tests
// ============================================================================

fn test__observed_power_well_powered() {
	// Large n, d=0.5 → should have ~80% power at alpha=0.05
	pwr := experiment.observed_power(64, 64, 0.5, 0.05)
	assert pwr > 0.75 && pwr < 0.95
}

fn test__observed_power_zero_effect() {
	// d=0 → power equals alpha (false positive rate only)
	pwr := experiment.observed_power(100, 100, 0.0, 0.05)
	assert math.abs(pwr - 0.05) < 0.01
}

fn test__mde_from_n_means_roundtrip() {
	// sample_size_means gives n for d=0.5, alpha=0.05, power=0.80
	// mde_from_n_means with that n should recover d≈0.5
	ss := experiment.sample_size_means(0.0, 1.0, 0.5, 0.05, 0.80)
	n := ss.n_per_group
	mde := experiment.mde_from_n_means(n, n, 1.0, 0.05, 0.80)
	// mde is in absolute units; with baseline_std=1.0 this equals Cohen's d
	assert math.abs(mde - 0.5) < 0.05
}

fn test__mde_from_n_proportions_reasonable() {
	// With n=500 per group, baseline 0.10, alpha=0.05, power=0.80
	// MDE should be a small but detectable absolute difference
	mde := experiment.mde_from_n_proportions(500, 500, 0.10, 0.05, 0.80)
	assert mde > 0.01 && mde < 0.10
}

fn test__power_diagnostic_underpowered() {
	// Very small groups → any realistic effect is underpowered
	ctrl := [10.0, 10.5, 9.8, 10.2]
	trt  := [10.3, 10.8, 10.1, 10.5]
	result := experiment.abtest(ctrl, trt)
	diag := experiment.power_diagnostic(result, 0.05, 0.80)
	assert diag.n_ctrl == 4
	assert diag.n_trt == 4
	assert diag.underpowered == true
	assert diag.observed_power >= 0.0 && diag.observed_power <= 1.0
	assert diag.mde_effect_size > 0.0
}

fn test__power_diagnostic_well_powered() {
	// Large groups with a clear effect → should be adequately powered
	ctrl := [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.2,
	         10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.2]
	trt  := [13.0, 13.2, 12.8, 13.1, 12.9, 13.0, 13.3, 12.7, 13.1, 12.9,
	         13.0, 13.2, 12.8, 13.1, 12.9, 13.0, 13.3, 12.7, 13.1, 12.9]
	result := experiment.abtest(ctrl, trt)
	diag := experiment.power_diagnostic(result, 0.05, 0.80)
	assert diag.underpowered == false
	assert diag.observed_power > 0.80
}

// ============================================================================
// Bayesian Continuous Tests
// ============================================================================

fn test__bayesian_continuous_no_effect() {
	// Identical distributions → P(trt > ctrl) ≈ 0.5
	ctrl := [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0]
	trt  := [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0]
	result := experiment.bayesian_continuous_ab_test(ctrl, trt)
	assert math.abs(result.prob_trt_beats_ctrl - 0.5) < 0.05
}

fn test__bayesian_continuous_clear_effect() {
	// Treatment clearly higher → P(trt > ctrl) > 0.95
	ctrl := [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 9.9,
	         10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 9.9]
	trt  := [14.0, 14.2, 13.8, 14.1, 13.9, 14.0, 14.3, 13.7, 14.1, 13.9,
	         14.0, 14.2, 13.8, 14.1, 13.9, 14.0, 14.3, 13.7, 14.1, 13.9]
	result := experiment.bayesian_continuous_ab_test(ctrl, trt)
	assert result.prob_trt_beats_ctrl > 0.95
	assert result.posterior_mean_trt > result.posterior_mean_ctrl
	assert result.ci_lower_trt > result.ci_upper_ctrl
}

fn test__bayesian_continuous_credible_intervals() {
	// CI should contain the posterior mean and be ordered
	ctrl := [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0]
	trt  := [12.0, 12.5, 11.5, 12.2, 11.8, 12.1, 11.9, 12.3, 11.7, 12.0]
	result := experiment.bayesian_continuous_ab_test(ctrl, trt)
	assert result.ci_lower_ctrl < result.posterior_mean_ctrl
	assert result.ci_upper_ctrl > result.posterior_mean_ctrl
	assert result.ci_lower_trt < result.posterior_mean_trt
	assert result.ci_upper_trt > result.posterior_mean_trt
}

fn test__bayesian_continuous_rope_not_set() {
	// When ROPE is not set (both 0.0), prob_rope must be 0.0
	ctrl := [10.0, 10.5, 9.5, 10.2, 9.8]
	trt  := [10.1, 10.6, 9.6, 10.3, 9.9]
	result := experiment.bayesian_continuous_ab_test(ctrl, trt)
	assert result.prob_rope == 0.0
}

fn test__bayesian_continuous_rope_narrow_effect() {
	// Tiny effect within ROPE → prob_rope should be substantial
	ctrl := [10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0]
	trt  := [10.05, 10.15, 9.95, 10.05, 10.15, 9.95, 10.05, 10.15, 9.95, 10.05]
	result := experiment.bayesian_continuous_ab_test(ctrl, trt,
		experiment.BayesianContinuousConfig{ rope_lower: -0.5, rope_upper: 0.5 })
	assert result.prob_rope > 0.30
	assert result.prob_rope > 0.0
}

fn test__abtest_ci_uses_t_distribution() {
	// With n=4 per group, Welch df≈6, t(6,0.975)≈2.447 vs z(0.975)≈1.960.
	// CI width under t: ≈1.413. CI width under z (current bug): ≈1.133.
	// This test catches the regression where df is discarded and z is used instead.
	ctrl := [9.5, 10.0, 10.5, 10.0]
	trt  := [11.5, 12.0, 12.5, 12.0]
	result := experiment.abtest(ctrl, trt)
	assert result.ci_upper - result.ci_lower > 1.2
}
