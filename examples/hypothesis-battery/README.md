# Hypothesis Testing Battery

Demonstrates the normality-first decision pattern: run Shapiro-Wilk, then choose
Welch t-test (normal) or Mann-Whitney U (non-normal). Applied to two scenarios —
near-normal data and bimodal data — to show when the choice matters.

**Run:** `v run examples/hypothesis-battery/main.v`

**Modules used:** `vstats.hypothesis`, `vstats.stats`

**Python equivalent:** `scipy.stats.shapiro` + `scipy.stats.ttest_ind` +
`scipy.stats.mannwhitneyu` — same pattern, same three functions.
