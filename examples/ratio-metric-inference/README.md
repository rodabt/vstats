# Ratio Metric Inference

Demonstrates why a naive t-test on ratio metrics (revenue/session) is biased,
and how the delta method correctly linearizes the ratio before testing.
Includes a permutation bootstrap as a non-parametric robustness check.

**Run:** `v run examples/ratio-metric-inference/main.v`

**Modules used:** `vstats.stats`, `vstats.experiment`

**Python equivalent:** manual linearization + `scipy.stats.ttest_ind` +
`scipy.stats.bootstrap` — three separate steps with no built-in ratio support.
