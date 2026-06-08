# Chart Gallery — Regression Diagnostics

Fits a regression on the Boston Housing dataset and renders four SVG charts that
together cover every `chart` type: a scatter of observed prices vs crime rate with
the fitted **line** overlaid, a **scatter** of residuals vs fitted values with a
zero **guide line**, a **histogram** of residuals, and a **bar** chart of the
multivariate regression coefficients with a custom theme.

Running the example regenerates the four `.svg` files in this directory.

**Run:** `v run examples/chart-gallery/main.v`

**Modules used:** `vstats.chart`, `vstats.ml`, `vstats.utils`

**Python equivalent:** matplotlib regression-diagnostics plots — `scatter` + fitted
line, residuals-vs-fitted, `hist` of residuals, and a coefficient `bar` chart.
