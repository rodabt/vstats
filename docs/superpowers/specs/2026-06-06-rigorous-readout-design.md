# Rigorous Readout Functions — Design Spec

**Date:** 2026-06-06
**Scope:** Standalone functions for a complete experiment readout workflow. No orchestrator.
**Approach:** Option B — all companion-doc TODOs relevant to readout + `bootstrap_test` beyond the roadmap.

---

## Motivation

The vstats `experiment/` module covers the core A/B test and causal methods but is missing the surrounding functions needed for a production-grade readout: metric pre-processing (winsorize, ratio correction), data quality checks (SRM, Simpson's), effect interpretation (HTE, ITT/PP, null verdict), and multiple testing correction. This spec fills those gaps as 12 standalone functions.

---

## File Organization

| File | Status | New additions |
|------|--------|--------------|
| `stats/descriptive.v` | existing | `winsorize`, `rtm_correction` |
| `stats/multiple_testing.v` | **new** | `bh_correction`, `bonferroni_correction` |
| `stats/inference.v` | **new** | `delta_method_ratio`, `bootstrap_test` |
| `experiment/abtest.v` | existing | `null_verdict`, `ancova`, `itt_and_pp` |
| `experiment/sample_size.v` | existing | `icc`, `design_effect` |
| `experiment/readout.v` | **new** | `srm_test`, `simpsons_check`, `hte_subgroup` |

---

## `stats/descriptive.v` — 2 new functions

### `stats.winsorize`

```v
pub fn winsorize(x []f64, q_low f64, q_high f64) []f64
```

Clips values in `x` at its own `q_low` and `q_high` quantile thresholds. Returns a new array; input is unchanged. For experiment use, always pass the pooled (treatment + control) array — caller's responsibility to pool before calling.

- Uses existing `stats.quantile` for thresholds.
- `q_low = 0.0, q_high = 0.99` is the standard configuration.
- No effect on binary (0/1) arrays — values are already in [0, 1].

### `stats.rtm_correction`

```v
pub fn rtm_correction(baseline []f64, followup []f64, selection_threshold f64) f64
```

Estimates the regression-to-the-mean baseline shift for units selected because their baseline exceeded `selection_threshold`. Uses the OLS relationship between baseline and followup to predict the expected followup value of the selected cohort under the null. Returns the expected RTM shift (subtract from an observed pre/post gain before concluding there is a real effect).

- Fits a linear model `followup ~ baseline` on all units.
- Predicts the expected followup for units with `baseline > selection_threshold`.
- Returns `predicted_followup_mean - overall_followup_mean`.

---

## `stats/multiple_testing.v` — 2 new functions (new file)

Both functions operate on a slice of p-values and return adjusted p-values plus rejection flags. Same result shape for easy interchangeability.

### `stats.bh_correction`

```v
pub struct BHResult {
pub:
    adjusted  []f64
    reject    []bool
    n_rejected int
}

pub fn bh_correction(p_values []f64, alpha f64) BHResult
```

Benjamini-Hochberg FDR correction. Algorithm:
1. Sort p-values ascending, track original indices.
2. `p_adj[i] = p[i] * n / rank[i]` (rank is 1-based).
3. Enforce monotonicity from right: `p_adj[i] = min(p_adj[i], p_adj[i+1])`.
4. Reject if `p_adj[i] <= alpha`.
5. Return results in original input order.

### `stats.bonferroni_correction`

```v
pub struct BonferroniResult {
pub:
    adjusted  []f64
    reject    []bool
    n_rejected int
}

pub fn bonferroni_correction(p_values []f64, alpha f64) BonferroniResult
```

`p_adj[i] = min(p[i] * n, 1.0)`. Reject if `p_adj[i] <= alpha`. Conservative; use BH for exploratory HTE, Bonferroni for pre-registered primary metrics.

---

## `stats/inference.v` — 2 new functions (new file)

### `stats.delta_method_ratio`

```v
pub struct DeltaMethodResult {
pub:
    ratio_ctrl   f64
    ratio_trt    f64
    effect       f64
    se           f64
    t_statistic  f64
    p_value      f64
    ci_lower     f64
    ci_upper     f64
}

pub fn delta_method_ratio(a []f64, b []f64, treatment []int) DeltaMethodResult
```

Correct SE for a ratio metric (ARPU, CTR, sessions/user) where the naive `SE(a)/mean(b)` is wrong because it ignores the covariance between numerator and denominator.

Algorithm:
1. Compute pooled ratio `R = mean(a) / mean(b)`.
2. Linearize: `z_i = a_i - R * b_i`.
3. Run a two-sample t-test on `z[treatment==1]` vs `z[treatment==0]`.
4. SE on ratio scale: `SE(z) / mean(b)`.
5. Effect: `ratio_trt - ratio_ctrl`.

`a` and `b` are user-level numerator and denominator (e.g., revenue and sessions). `treatment` is 0/1.

### `stats.bootstrap_test`

```v
pub struct BootstrapResult {
pub:
    p_value       f64
    observed_diff f64
    ci_lower      f64
    ci_upper      f64
    n_resamples   int
}

pub fn bootstrap_test(ctrl []f64, trt []f64, n_resamples int) BootstrapResult
```

Permutation test for difference in means. Non-parametric — no normality assumption. Use as a robustness check alongside the parametric `abtest()`, especially for heavy-tailed metrics.

Algorithm:
1. Compute observed difference `d_obs = mean(trt) - mean(ctrl)`.
2. Pool all values. Repeat `n_resamples` times: shuffle, split at `ctrl.len`, compute mean difference.
3. `p_value = proportion of permuted diffs >= |d_obs|` (two-tailed).
4. CI from the percentile bootstrap on the observed difference (resample with replacement from each group separately).

Uses `rand` (already a dependency in `experiment/bayesian.v`). Default `n_resamples = 10_000` is sufficient; 1_000 is acceptable for interactive use.

---

## `experiment/sample_size.v` — 2 new functions

### `experiment.icc`

```v
pub fn icc(y []f64, cluster_ids []int) f64
```

Intraclass Correlation Coefficient via one-way ANOVA decomposition:

```
ICC = (MS_between - MS_within) / (MS_between + (m̄ - 1) * MS_within)
```

where `m̄` is the average cluster size. Uses `stats.anova_one_way` for the F-statistic and derives MS_between and MS_within from it. Returns 0.0 if `MS_between <= MS_within` (no clustering signal). Clamps result to [0, 1].

### `experiment.design_effect`

```v
pub fn design_effect(icc f64, avg_cluster_size f64) f64
```

`DEFF = 1.0 + (avg_cluster_size - 1.0) * icc`. Multiply the required sample size from `sample_size_means` or `sample_size_proportions` by DEFF to get the cluster-adjusted requirement.

---

## `experiment/abtest.v` — 3 new functions

### `experiment.ancova`

```v
@[params]
pub struct ANCOVAConfig {
pub:
    alpha f64 = 0.05
}

pub struct ANCOVAResult {
pub:
    effect      f64
    se          f64
    t_statistic f64
    p_value     f64
    ci_lower    f64
    ci_upper    f64
    r_squared   f64
    n           int
}

pub fn ancova(y []f64, treatment []int, covariates [][]f64, cfg ANCOVAConfig) ANCOVAResult
```

OLS-based ANCOVA: `Y = α + β·treatment + Σγ_k·cov_k + ε`. The `β` coefficient is the covariate-adjusted treatment effect. More flexible than CUPED — handles categorical strata and multiple continuous covariates simultaneously.

- Builds design matrix `[treatment, cov_0, ..., cov_k]` and calls `ml.linear_regression`.
- Extracts SE for the treatment coefficient using `ols_se`, which is currently a private function in `did.v`. It must be extracted into a private helper in a new `experiment/ols_helpers.v` file (or made `pub fn`) so both `did.v` and the new `abtest.v` additions can use it without duplication.
- Returns `effect = β`, `se`, t-stat, p-value, CI.

### `experiment.null_verdict`

```v
pub enum NullVerdictKind {
    significant
    true_null
    inconclusive
    below_mde
}

pub struct NullVerdictResult {
pub:
    verdict   NullVerdictKind
    label     string
    ci_lower  f64
    ci_upper  f64
    mde       f64
    p_value   f64
}

pub fn null_verdict(effect f64, se f64, mde f64, alpha f64) NullVerdictResult
```

Classifies the result of a completed test:

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| `significant` | p < alpha | Ship decision warranted |
| `true_null` | CI upper < MDE | Effect is real and small — safely below business threshold |
| `inconclusive` | CI spans MDE | Can't distinguish "no effect" from "effect below MDE" — need more data |
| `below_mde` | point estimate < MDE/2 and CI entirely below MDE | Effect exists but is commercially irrelevant |

CI is `effect ± z_{alpha/2} * se`. p-value uses normal approximation (consistent with `abtest`).

### `experiment.itt_and_pp`

```v
pub struct ITTPPResult {
pub:
    itt             ABTestResult
    pp              ABTestResult
    compliance_rate f64
    late            f64
    late_se         f64
}

pub fn itt_and_pp(y []f64, treatment []int, activated []int, cfg ABTestConfig) ITTPPResult
```

- **ITT** (intent-to-treat): runs `abtest` on all assigned users, including non-activators. This is the primary estimand.
- **Per-protocol**: filters to `activated[i] == 1` in treatment and all control users. Upward-biased but useful as a sanity check.
- **LATE** (IV/Wald): `late = itt.effect / compliance_rate`, where `compliance_rate = mean(activated[treatment==1])`. SE via delta method: `late_se = itt_se / compliance_rate`.

`activated` is 0/1: 1 means the user actually engaged with the treatment (triggered the feature).

---

## `experiment/readout.v` — 3 new functions (new file)

### `experiment.srm_test`

```v
pub struct SRMResult {
pub:
    chi2         f64
    p_value      f64
    srm_detected bool
    observed     []int
    expected     []f64
}

pub fn srm_test(observed_counts []int, expected_proportions []f64, alpha f64) SRMResult
```

Chi-squared goodness-of-fit test on assignment counts. Wraps `hypothesis.chi_squared_gof_test`. `expected_proportions` should sum to 1.0 (e.g., `[0.5, 0.5]` for a 50/50 split). `srm_detected = p_value < alpha`. Run this before looking at any metric results.

### `experiment.simpsons_check`

```v
pub struct SimpsonsResult {
pub:
    aggregate_ate     f64
    segment_ates      []f64
    segment_ids       []int
    reversal_detected bool
}

pub fn simpsons_check(y []f64, treatment []int, segment []int) SimpsonsResult
```

Computes aggregate ATE (`mean(y[treatment==1]) - mean(y[treatment==0])`) and per-segment ATEs. Sets `reversal_detected = true` when any segment ATE has the opposite sign from the aggregate ATE. Does not perform significance testing — use `hte_subgroup` for that. This is purely a direction-check.

### `experiment.hte_subgroup`

```v
pub struct SubgroupResult {
pub:
    segment_id int
    n_ctrl     int
    n_trt      int
    ate        f64
    se         f64
    p_value    f64
    ci_lower   f64
    ci_upper   f64
}

pub struct HTEResult {
pub:
    subgroups []SubgroupResult
    overall   ABTestResult
}

pub fn hte_subgroup(y []f64, treatment []int, segment []int, cfg ABTestConfig) HTEResult
```

Runs `abtest` within each unique segment value. Also runs `abtest` on the full sample for the `overall` field. Does not apply multiple testing correction — caller should pass the resulting `p_values` to `stats.bh_correction` or `bonferroni_correction`. Pre-register segments before unblinding.

---

## Skill Updates (`/home/rabt/.claude/skills/vstats/`)

### `references/modules.md` — corrections to existing entries

The following entries in the current `modules.md` are stale and need fixing:

| Location | Current (wrong) | Correct |
|----------|----------------|---------|
| `experiment` — ABTestConfig | `two_tailed: true` | `equal_variance: false` |
| `experiment` — SPRT functions | `sprt_update`, `sprt_boundaries` | `sprt_test(successes_a, n_a, successes_b, n_b, cfg SPRTConfig)` |
| `experiment` — DiD functions | `did_estimate(...)` | `did_2x2(...)`, `did_regression(...)` |
| `experiment` — PSM functions | `estimate_propensity_scores(x, treatment []int)` | `estimate_propensity_scores(x, treatment []f64, cfg PropensityConfig) PropensityModel` |
| `experiment` — PSM functions | `match_units(...)` | `match_nearest_neighbor(model, cfg MatchingConfig) MatchingResult` |
| `experiment` — PSM functions | `estimate_att(...)` | `ate_matched(y, treatment, result MatchingResult) ATEResult` |

### `references/modules.md` — new entries to add

All 12 new functions with signatures, in the appropriate module sections.

### `SKILL.md` — updates

- Add a **Rigorous Readout Workflow** section showing the canonical call sequence: `srm_test` → `winsorize` → `delta_method_ratio` or `abtest` → `bh_correction` → `hte_subgroup` → `null_verdict`.
- Update the **Hypothesis Testing Quick Reference** to mention `bootstrap_test` as the non-parametric alternative.
- Fix the experiment workflow code snippet: `ABTestConfig` field `two_tailed` → `equal_variance`.

---

## Tests

One test file per new source file, plus additions to the existing `tests/stats_test.v` and `tests/experiment_test.v`:

| Test file | What it covers |
|-----------|----------------|
| `tests/stats_test.v` (additions) | `winsorize`, `rtm_correction` |
| `tests/multiple_testing_test.v` (new) | `bh_correction`, `bonferroni_correction` |
| `tests/inference_test.v` (new) | `delta_method_ratio`, `bootstrap_test` |
| `tests/experiment_test.v` (additions) | `null_verdict`, `ancova`, `itt_and_pp`, `icc`, `design_effect` |
| `tests/readout_test.v` (new) | `srm_test`, `simpsons_check`, `hte_subgroup` |

All tests use known synthetic data with analytically checkable answers.

---

## Out of Scope

- `experiment.mediation` (Baron-Kenny) — useful but not core readout
- `experiment.mde_to_business` — design-time tool, not readout
- `experiment.exposure_funnel` / `spillover_test` — pre-readout data quality, separate concern
- Any orchestrator or `ReadoutResult` struct — explicitly deferred (Option A chosen)
- Cross-validation — ML concern, not experiment readout
- Survival analysis — separate future module
