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

## Design Optimizer

> Answers: "Should I run this experiment, and for how long?" without requiring you
> to choose alpha, power, or an MDE manually.

### Simplified entry point

```v
// DesignParams — human-friendly inputs; pass to optimizer_config() to get OptimizerConfig.
DesignParams{
    baseline                  f64        // observed metric value (rate or mean)
    daily_traffic_per_variant int        // eligible users per variant per day
    min_relative_lift         f64 = 0.05 // smallest lift worth detecting, as fraction of baseline
    prior_conviction          f64 = 0.50 // belief that the experiment will find a positive effect
                                         // 0.0 = very skeptical · 1.0 = very confident
    metric_std_dev            f64 = 0.0  // historical σ per user; leave 0 for proportion metrics
    max_days                  int = 90
    // advanced overrides (sensible defaults):
    alpha                f64 = 0.05
    min_power            f64 = 0.80
    seasonality_min_days int = 14
    seed                 u32
}

optimizer_config(params DesignParams) OptimizerConfig
// Derives mde_tolerance, MixturePrior, and min_monthly_detection_rate from params.

conviction_to_prior(conviction f64, mde f64) (MixturePrior, f64)
// Returns (prior, min_monthly_detection_rate) for a given conviction score.
// Exposed for inspection; called internally by optimizer_config().

find_optimal_runtime(config OptimizerConfig) OptimizationResult
// OptimizationResult{ optimal_days int; monthly_detection_rate f64;
//                     all_results []RuntimeResult; worth_running bool;
//                     power_min_days, effective_min_days int;
//                     power_at_optimal f64; no_go_reason string }
```

### Metric types

| `metric_std_dev` | Metric type | Power formula |
|---|---|---|
| `0` (default) | Proportion (conversion rate, CTR) | Two-proportion z-test |
| `> 0` | Continuous (revenue, session time, AOV) | Two-sample t-test (`n = 2σ²(z_α+z_β)²/δ²`) |

### `prior_conviction` mapping

| conviction | null % | neg % | pos % | detection threshold |
|---|---|---|---|---|
| 0.0 (very skeptical) | 65% | 30% | 5% | 0.50 |
| 0.2 (crossover) | 54% | 26% | 20% | 0.40 |
| 0.5 (neutral) | 38% | 20% | 42% | 0.26 |
| 1.0 (very confident) | 10% | 10% | 80% | 0.02 |

The `monthly_detection_rate` is the prior-weighted probability of finding a true positive in any given month. The experiment is not worth running (`worth_running = false`) when this falls below the conviction-derived threshold.

### Advanced: `OptimizerConfig` directly

```v
OptimizerConfig{
    baseline                  f64
    daily_traffic_per_variant int
    mde_tolerance             f64        // absolute MDE (in metric units)
    alpha                     f64 = 0.05
    prior                     MixturePrior
    seasonality_min_days      int = 14
    min_power                 f64 = 0.80
    max_days                  int = 90
    min_monthly_detection_rate f64 = 0.05
    metric_std_dev            f64 = 0.0
    seed                      u32
}

MixturePrior{
    null_frac f64 = 0.40   // fraction with no effect
    neg_frac  f64 = 0.30   // fraction with harmful effect
    neg_mean  f64 = -0.02  // centre of negative-effect distribution
    neg_std   f64 = 0.01
    pos_mean  f64 = 0.02   // centre of positive-effect distribution
    pos_std   f64 = 0.01
    n_samples int = 100_000
}
```

### Example — proportion metric

```v
import vstats.experiment

params := experiment.DesignParams{
    baseline:                  0.41   // 41% conversion rate
    daily_traffic_per_variant: 1370
    min_relative_lift:         0.05   // detect ≥5% relative improvement
    prior_conviction:          0.30   // moderately skeptical
    max_days:                  30
}
config := experiment.optimizer_config(params)
result := experiment.find_optimal_runtime(config)
println(result.worth_running)           // true
println(result.optimal_days)            // 14
println(result.monthly_detection_rate)  // 0.584
```

### Example — continuous metric (revenue per user)

```v
params := experiment.DesignParams{
    baseline:                  47.50   // $47.50 mean order value
    metric_std_dev:            82.00   // $82 historical σ — required for continuous
    daily_traffic_per_variant: 620
    min_relative_lift:         0.05    // detect ≥$2.38 lift
    prior_conviction:          0.40
    max_days:                  60
}
config := experiment.optimizer_config(params)
result := experiment.find_optimal_runtime(config)
```

## See Also

- [rigorous-ab-readout example](../examples.html#rigorous-ab-readout)
- [causal-did example](../examples.html#causal-did)
- [ab-design-optimizer example](../examples.html#ab-design-optimizer)
- [revenue-per-user-optimizer example](../examples.html#revenue-per-user-optimizer)
