# VStats 0.2.1

A dependency-free statistics, linear algebra, and machine learning library for the [V programming language](https://vlang.io), with a focus on product analytics and experimentation. Includes A/B testing, funnel analysis, cohort retention, causal inference, and growth metrics alongside classical stats and ML — all built from scratch with no external dependencies.

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
| **symbol**     | Symbolic computation                 | WIP        |

## Generic Type Support

Most functions accept generic numeric types (`int` or `f64`). The convention is:

- **Same-type output** (linalg): `linalg.add[T](v []T, w []T) []T`
- **f64 output** (stats, ml): `stats.mean[T](x []T) f64` — ensures precision
- **f64-only** where required: `median`, `quantile`, `mode` (need sorting/hashing)

## Documentation

Full API reference, conceptual guides, worked examples, and module docs are available in the [`docs/`](docs/) directory. Open `docs/index.html` in your browser to get started.

## Build & Test

```bash
make test              # run all tests
make fulltest          # run with verbose stats
v test tests/          # same as make test
v test tests/stats_test.v   # single test file
```

## Changelog

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
