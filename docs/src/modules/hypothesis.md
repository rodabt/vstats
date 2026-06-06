# hypothesis

`import vstats.hypothesis`

Statistical hypothesis tests. All tests return `(statistic f64, p_value f64)`.

> **vs Python:** mirrors `scipy.stats` test functions. The return convention
> `(statistic, p_value)` is the same as scipy.

## Usage Pattern

```v
tp := hypothesis.TestParams{ alpha: 0.05 }

t_stat, p := hypothesis.t_test_two_sample(group_a, group_b, tp)
```

## Parametric Tests

```v
t_test_one_sample(x []f64, mu f64, tp TestParams) (f64, f64)
t_test_two_sample(x []f64, y []f64, tp TestParams) (f64, f64)   // Welch t-test
one_sample_t_test(x []f64, mu f64) (f64, f64)                   // alpha=0.05 shorthand
two_sample_t_test(x []f64, y []f64) (f64, f64)                  // alpha=0.05 shorthand

anova_one_way[T](groups [][]T) (f64, f64)                       // in stats module
correlation_test(x []f64, y []f64, tp TestParams) (f64, f64)
```

## Non-Parametric Tests

> **vs Python:** `mann_whitney_u_test` replaces `scipy.stats.mannwhitneyu`.
> `wilcoxon_signed_rank_test` replaces `scipy.stats.wilcoxon`.

```v
mann_whitney_u_test(x []f64, y []f64) (f64, f64)        // unpaired, robust to outliers
wilcoxon_signed_rank_test(x []f64, y []f64) (f64, f64)  // paired non-parametric
ks_test(sample1 []f64, sample2 []f64) (f64, f64)        // Kolmogorov-Smirnov
```

## Normality & Goodness of Fit

> **vs Python:** `shapiro_wilk_test` replaces `scipy.stats.shapiro`.
> `chi_squared_gof_test` replaces `scipy.stats.chisquare`.

```v
shapiro_wilk_test(x []f64) (f64, f64)
chi_squared_test(contingency [][]int) (f64, f64)                // independence
chi_squared_gof_test(observed []f64, expected []f64) (f64, f64) // goodness-of-fit
```

## Spearman & Runs

```v
// Available in hypothesis module
// spearman_correlation_test, wald_wolfowitz_runs_test
// See hypothesis/tests.v for signatures
```

## Decision Pattern

```v
_, p_sw := hypothesis.shapiro_wilk_test(x)
if p_sw > 0.05 {
    // Normal — use parametric
    _, p := hypothesis.t_test_two_sample(x, y, tp)
} else {
    // Non-normal — use non-parametric
    _, p := hypothesis.mann_whitney_u_test(x, y)
}
```

## See Also

- [hypothesis-battery example](../examples.html#hypothesis-battery)
