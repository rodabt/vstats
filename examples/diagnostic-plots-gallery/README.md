# Diagnostic Plots Gallery — ROC, Q-Q, ECDF, Pair Plot

Four common ML/stats diagnostic plots vstats couldn't produce before:

- **ROC curve** — Random Forest classifier on the Breast Cancer dataset, using
  `utils.roc_curve()` (already existed) fed straight into `chart.line()`.
- **ECDF** — empirical CDF of Boston Housing regression residuals, via the new
  `stats.ecdf()` plotted with `chart.step()`.
- **Q-Q plot** — normality check of the same residuals, via the new
  `prob.qq_points()` plotted with `chart.scatter()`.
- **Pair plot** — scatter-matrix of the four Iris features via the new
  `chart.pair_plot()`, which composes a `chart.Grid` with pairwise scatter panels
  and histogram diagonals.

Running the example regenerates the four `.svg` files in this directory.

**Run:** `v run examples/diagnostic-plots-gallery/main.v`

**Modules used:** `vstats.chart`, `vstats.stats`, `vstats.prob`, `vstats.ml`,
`vstats.utils`, `vstats.linalg`

**Python equivalent:** `sklearn.metrics.RocCurveDisplay`, `scipy.stats.probplot`,
`statsmodels.distributions.ECDF`, `seaborn.pairplot`.
