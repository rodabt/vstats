# experiment

`import vstats.experiment`

A/B testing, variance reduction (CUPED), causal inference (DiD, PSM), and rigorous
readout checks (SRM, ITT/PP, subgroup analysis).

> **vs Python:** replaces `scipy.stats.ttest_ind` + `statsmodels` OLS + custom SRM logic
> + `multipletests` — four libraries in one import.

## A/B Testing

```v
abtest(control []f64, treatment []f64, cfg ABTestConfig) ABTestResult
// ABTestConfig{ alpha f64 = 0.05, equal_variance bool }
// ABTestResult: control_mean, treatment_mean, control_std, treatment_std,
//               relative_lift, effect_size, t_statistic, degrees_freedom,
//               p_value, significant bool, ci_lower, ci_upper f64, n_control, n_treatment int

null_verdict(result ABTestResult, alpha f64) string
// Plain-English: "Significant: ..." or "Not significant: ..."

cuped_test(y_ctrl []f64, y_treat []f64, pre_ctrl []f64, pre_treat []f64, cfg ABTestConfig) CUPEDResult
// CUPEDResult{ theta f64, variance_reduction f64, adjusted_result ABTestResult }

ancova(ctrl []f64, trt []f64, x_ctrl [][]f64, x_trt [][]f64, cfg ABTestConfig) ANCOVAResult
// Covariate-adjusted A/B test via OLS. Pass [][]f64{} for no covariates.
// ANCOVAResult{ adjusted_effect, se, t_statistic, p_value, ci_lower, ci_upper f64;
//               significant bool; n_control, n_treatment int }

itt_and_pp(y []f64, assigned []int, complied []bool, cfg ABTestConfig) ITTPPResult
// Intent-to-Treat and Per-Protocol. assigned: 0=ctrl, 1=trt.
// ITTPPResult{ itt ABTestResult, pp ABTestResult }
```

## Sample Size

> **vs Python:** replaces `statsmodels.stats.power.tt_ind_solve_power`.

```v
power_analysis(effect_size f64, alpha f64, power f64) PowerAnalysisResult
// PowerAnalysisResult{ n_per_group int }

proportion_power_analysis(p_baseline f64, p_treatment f64, alpha f64, power f64) ProportionPowerResult

sample_size_means(baseline_mean f64, baseline_std f64, mde_absolute f64, alpha f64, power f64) SampleSizeResult
sample_size_proportions(baseline_rate f64, mde f64, alpha f64, power f64) SampleSizeResult

icc(groups [][]f64) f64                   // Intraclass Correlation Coefficient
design_effect(m_bar f64, icc_val f64) f64 // DEFF = 1 + (m̄-1)·ICC for cluster-randomized trials
```

## Readout Checks

```v
srm_test(n_ctrl int, n_trt int, expected_ratio f64, alpha f64) SRMResult
// SRMResult{ expected_ctrl, expected_trt f64; observed_ctrl, observed_trt int;
//            chi2_statistic, p_value f64; srm_detected bool }

simpsons_check(overall_effect f64, subgroup_effects []f64) SimpsonsCheckResult
// reversal_found=true if any subgroup has opposite sign to overall

hte_subgroup(y []f64, treatment []int, subgroup []int, labels []string, cfg ABTestConfig) HTEResult
// HTEResult{ subgroup_labels []string; subgroup_effects, subgroup_p_values []f64; subgroup_ns []int }
```

## Difference-in-Differences

> **vs Python:** replaces `statsmodels.formula.api.ols` with interaction term.

```v
did_2x2(y_treat_pre []f64, y_treat_post []f64, y_ctrl_pre []f64, y_ctrl_post []f64, cfg DiDConfig) DiDResult
did_regression(y []f64, x [][]f64, group []int, time []int, cfg DiDConfig) DiDRegressionResult
// DiDRegressionResult{ did_coefficient, did_se, did_t_stat, did_p_value, did_ci_lower, did_ci_upper, r_squared f64 }

test_parallel_trends(y_treated_pre []f64, y_control_pre []f64, time_pre []int, cfg DiDConfig) ParallelTrendsResult
event_study(y []f64, group []int, relative_time []int, cfg DiDConfig) EventStudyResult
```

## Propensity Score Matching

> **vs Python:** replaces `sklearn.linear_model.LogisticRegression` + manual matching loop.

```v
estimate_propensity_scores(x [][]f64, treatment []int) []f64
match_units(propensity_scores []f64, treatment []int, caliper f64) []MatchedPair
estimate_att(outcomes []f64, treatment []int, matched_pairs []MatchedPair) f64
```

## See Also

- [rigorous-ab-readout example](../examples.html#rigorous-ab-readout)
- [causal-did example](../examples.html#causal-did)
