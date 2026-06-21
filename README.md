# VStats 0.3.0

A dependency-free statistics, linear algebra, and machine learning library for the [V programming language](https://vlang.io), with a focus on product analytics and experimentation. Includes A/B testing, funnel analysis, cohort retention, causal inference, time series analysis, and growth metrics alongside classical stats and ML — all built from scratch with no external dependencies.

## Installation

```bash
v install https://github.com/rodabt/vstats
```

## Quick Start

```v
import vstats.stats
import vstats.linalg
import vstats.experiment
import vstats.growth

// Descriptive statistics — works with int or f64
mean_val := stats.mean([1, 2, 3, 4, 5])
std_dev := stats.standard_deviation([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

// Linear algebra
dot := linalg.dot([1, 2, 3], [4, 5, 6])
result := linalg.matmul(matrix_a, matrix_b)

// A/B testing
ab := experiment.abtest([10.1, 9.8, 10.2], [13.1, 12.8, 13.2])
println('p-value: ${ab.p_value:.4f}, lift: ${ab.relative_lift:.2f}')

// Funnel analysis
funnel := growth.create_funnel(
    ['Visit', 'Signup', 'Purchase'],
    [1000, 350, 120],
)
println('Overall conversion: ${funnel.conversion_rate:.2f}')
```

## Modules

| Module         | Purpose                              | Status     |
| -------------- | ------------------------------------ | ---------- |
| **linalg**     | Vector & matrix operations           | Complete   |
| **stats**      | Descriptive & inferential statistics | Complete   |
| **prob**       | Probability distributions (PDF/CDF)  | Complete   |
| **optim**      | Numerical optimization               | Complete   |
| **utils**      | Metrics, datasets, feature tools     | Complete   |
| **ml**         | Regression, classification, clustering | Complete |
| **nn**         | Neural network layers & training     | Complete   |
| **hypothesis** | Statistical hypothesis tests         | Complete   |
| **experiment** | A/B testing, PSM, DiD, CUPED        | Complete   |
| **growth**     | Funnels, cohorts, attribution        | Complete   |
| **timeseries** | ARIMA, VAR, decomposition, smoothing | Complete   |
| **symbol**     | Symbolic computation                 | WIP        |

## Generic Type Support

Most functions accept generic numeric types (`int` or `f64`). The convention is:

- **Same-type output** (linalg): `linalg.add[T](v []T, w []T) []T`
- **f64 output** (stats, ml): `stats.mean[T](x []T) f64` — ensures precision
- **f64-only** where required: `median`, `quantile`, `mode` (need sorting/hashing)

## Documentation

- **[docs/index.html](docs/index.html)** — full API reference, concepts, examples
- **[docs/getting-started.html](docs/getting-started.html)** — quick-start for Python/R users
- **[examples/](examples/)** — runnable scenario files covering ML, experiment analysis, and time series

To regenerate the HTML docs after editing `docs/src/`:
```bash
make docs
```

## Build & Test

```bash
make test              # run all tests
make fulltest          # run with verbose stats
make docs              # regenerate HTML docs from docs/src/
v test tests/          # same as make test
v test tests/stats_test.v   # single test file
```

## Changelog

### v0.3.3

**experiment module improvements**
- **Non-parametric tests** (`nonparametric.v`): `mann_whitney_ab_test` (two-sample) and `wilcoxon_paired_test` (paired) return rank-biserial correlation effect size and percentile bootstrap CI; both support `alternative: .greater / .less`
- **Winsorized/trimmed means**: `utils.winsorize` and `utils.trim` clip/remove outliers by percentile; `experiment.winsorized_abtest` and `experiment.trimmed_abtest` apply them before Welch's t-test for outlier-robust comparisons
- **Bootstrap CI** (`stats/inference.v`): `stats.bootstrap_ci(x, stat_fn, n_boot, alpha)` returns a percentile CI for any scalar statistic — mean, median, 90th percentile, or custom lambda; `BootstrapCIResult` carries the point estimate alongside bounds

### v0.3.2

**experiment module improvements**
- **One-sided hypothesis tests**: `abtest`, `ancova`, and `proportion_test` now accept `alternative: .greater` or `alternative: .less` (via `TestAlternative` enum) for directional p-values; CIs remain two-sided; default `.two_sided` is fully backward-compatible
- **mSPRT for continuous metrics** (`sequential.v`): `msprt_test` applies the mixture Sequential Probability Ratio Test to continuous outcome data — stateless, always-valid inference, normal-mixture prior with tunable `tau_sigma_ratio`; σ estimated from data or provided by caller
- **Novelty/primacy detection** (`readout.v`): `novelty_primacy_check` takes a per-period effect time series and returns early/late split means, OLS slope with t-test significance, and `novelty_suspected` / `primacy_suspected` flags
- **CUPED for proportions**: documented in `cuped_test` that binary 0/1 outcomes work directly — pass conversion events as 0.0/1.0 `[]f64` with any continuous pre-covariate

### v0.3.1

**experiment module improvements**
- **A/B design optimizer** (`design_optimizer.v`): `find_optimal_runtime` replaces the EU-based objective with a detection-rate objective — `E[I(effect ≥ mde_tolerance) × power(effect, T)] × 30/T` — eliminating the need for unknown `annual_revenue` / `day_cost`. Given a `MixturePrior` over possible true effects, the optimizer finds the runtime that maximises expected meaningful detections per month. Answers: "can we catch a ≥ N pp effect within a month, given our traffic?"
- **t-distribution corrections**: all p-values and CIs in `abtest.v` (Welch's t-test, ANCOVA) and `did.v` (DiD regression) now use the exact t-distribution instead of the normal approximation

### v0.3.0

**New module: timeseries**
- **analysis**: `diff`, `seasonal_diff`, `undiff` (differencing and inversion); `acf`, `pacf`, `acf_confidence_bound` (Levinson-Durbin); `adf_test` (MacKinnon critical values), `kpss_test` (Bartlett kernel); `aic`, `bic`, `aicc`
- **decomposition**: Classical additive/multiplicative decomposition (`decompose`); STL decomposition with iterative LOESS and robustness weights (`stl`)
- **smoothing**: Simple Exponential Smoothing (`ses`), Holt's double smoothing (`holt`), Holt-Winters triple smoothing (`holt_winters`); auto-optimization via Nelder-Mead (`auto_ses`, `auto_holt`, `auto_holt_winters`)
- **arima**: CSS fitting (`arima_fit`), forecasting with CIs (`arima_forecast`), model summary (`arima_summary`), SARIMA (`sarima_fit`), grid-search order selection (`auto_arima`)
- **var**: VAR(p) OLS fitting (`var_fit`), recursive forecasting (`var_forecast`), AIC/BIC/HQC lag selection (`var_select_lag`), Granger causality F-test (`granger_causality`), Cholesky-orthogonalized impulse response functions (`irf`)

**New examples**
- `examples/timeseries-arima-forecast/` — unit root tests, ACF/PACF, decomposition, ARIMA fit and 8-step forecast, auto_arima
- `examples/timeseries-var-granger/` — VAR lag selection, coefficient recovery, Granger causality, IRF visualization

### v0.2.0

**New modules**
- **experiment**: A/B testing (Welch's t-test, Bayesian Beta-Binomial), CUPED variance reduction, power analysis, sample size calculators, SPRT sequential testing, proportion tests
- **experiment**: Propensity Score Matching with balance checks and ATE estimation
- **experiment**: Difference-in-Differences (2×2, regression, parallel trends, event study)
- **growth**: Funnel analysis (stage conversion/drop-off), cohort retention tables, marketing attribution (first-touch, last-touch, linear, time-decay, position-based), growth metrics (DAU/MAU ratios, retention rates)

**New features**
- **hypothesis**: Wilcoxon signed-rank test, Mann-Whitney U test
- **stats**: ANOVA, confidence intervals, Cohen's d, Cramér's V, skewness, kurtosis
- **utils**: ROC/AUC curves, confusion matrix, feature normalization, early stopping, LR decay, grid search, built-in datasets (Iris, Wine, Breast Cancer, Boston Housing, Titanic)
- **nn**: Full set of loss functions (MSE, MAE, Huber, hinge, cross-entropy, KL divergence, triplet, contrastive)
- **ml**: SVM, Naive Bayes, Random Forest, DBSCAN, hierarchical clustering

**Improvements**
- Generics migration across linalg and ml (replaced `_f64` suffix functions)
- Bug fixes in matrix reshape, inverse normal CDF, chi-squared approximation, silhouette coefficient
- HTML documentation site with API reference, concepts, and examples

### v0.1.0

- Initial release with linalg, stats, prob, optim, utils, ml, nn modules

## Disclaimer

- Written as an exercise to bring V closer to data analytics and ML workflows
- Inspired by Joel Grus's *Data Science from Scratch*
- Focus is on correctness and API design, not raw performance
- Contributions welcome!

## References

- [V Language](https://vlang.io)
- [Data Science from Scratch](https://github.com/joelgrus/data-science-from-scratch) by Joel Grus
