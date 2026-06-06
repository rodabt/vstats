# stats

`import vstats.stats`

Descriptive statistics, outlier handling, multiple testing corrections, and
ratio-metric inference.

> **vs Python:** `stats.mean`, `stats.variance`, `stats.correlation` replace
> `numpy.mean`, `numpy.var`, `scipy.stats.pearsonr`. Functions are generic —
> they accept `[]int` or `[]f64` and return `f64`.

## Descriptive

```v
sum[T](x []T) T
mean[T](x []T) f64
median(x []f64) f64                        // requires []f64
quantile(x []f64, p f64) f64               // uses int(p*n) truncation, not rounding
mode(x []f64) []f64
range[T](x []T) T
variance[T](x []T) f64                     // sample variance (÷n-1)
standard_deviation[T](x []T) f64
interquartile_range(x []f64) f64
covariance[T](x []T, y []T) f64
correlation[T](x []T, y []T) f64
dev_mean[T](x []T) []f64
```

## Effect Sizes & Tests

```v
cohens_d[T](group1 []T, group2 []T) f64
cramers_v(contingency [][]int) f64
skewness[T](x []T) f64
kurtosis[T](x []T) f64
anova_one_way[T](groups [][]T) (f64, f64)               // (F-stat, p-value)
confidence_interval_mean[T](x []T, level f64) (f64, f64)
```

## Outlier Handling

> **vs Python:** `stats.winsorize` replaces `scipy.stats.mstats.winsorize`.
> `stats.rtm_correction` has no direct scipy equivalent.

```v
winsorize(x []f64, q_low f64, q_high f64) []f64
rtm_correction(baseline []f64, followup []f64, selection_threshold f64) f64
```

## Multiple Testing

> **vs Python:** replaces `statsmodels.stats.multitest.multipletests`.

```v
bh_correction(p_values []f64, alpha f64) BHResult
// BHResult{ adjusted []f64, reject []bool, n_rejected int }

bonferroni_correction(p_values []f64, alpha f64) BonferroniResult
```

## Ratio Metrics & Bootstrap

> **vs Python:** `delta_method_ratio` replaces manual linearization + `scipy.stats.ttest_ind`.
> `bootstrap_test` replaces `scipy.stats.bootstrap`.

```v
delta_method_ratio(a []f64, b []f64, treatment []int, cfg DeltaMethodConfig) DeltaMethodResult
// DeltaMethodConfig{ alpha f64 = 0.05 }
// DeltaMethodResult{ ratio_ctrl, ratio_trt, effect, se, t_statistic, p_value, ci_lower, ci_upper f64 }

bootstrap_test(ctrl []f64, trt []f64, n_resamples int) BootstrapResult
// BootstrapResult{ p_value, observed_diff, ci_lower, ci_upper f64; n_resamples int }
```

## See Also

- [ratio-metric-inference example](../examples.html#ratio-metric-inference)
- [rigorous-ab-readout example](../examples.html#rigorous-ab-readout)
