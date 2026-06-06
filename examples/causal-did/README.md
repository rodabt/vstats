# Causal Difference-in-Differences

Estimates a causal treatment effect from observational panel data using DiD
regression. Includes the parallel trends assumption check as a pre-analysis step.

**Run:** `v run examples/causal-did/main.v`

**Modules used:** `vstats.experiment`

**Python equivalent:** `statsmodels.formula.api.ols('y ~ group * time', data).fit()`
plus manual parallel trends slope comparison.
