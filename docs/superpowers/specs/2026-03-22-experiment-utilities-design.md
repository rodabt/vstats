# Design: Experiment Module Utilities & Documentation

**Date:** 2026-03-22
**Status:** Approved

## Summary

Extend the `experiment/` module with four new focused files covering sample size calculation, proportion testing, sequential testing, and Bayesian A/B testing. Improve documentation with full API reference, conceptual explanations, and end-to-end worked examples.

---

## New Code

### File Layout

All new files live inside `experiment/` following the existing per-concern pattern (`abtest.v`, `did.v`, `psm.v`).

```
experiment/
  abtest.v          (existing)
  did.v             (existing)
  psm.v             (existing)
  sample_size.v     (NEW)
  proportion_test.v (NEW)
  sequential.v      (NEW)
  bayesian.v        (NEW)
```

Tests go in `tests/` as per project convention.

---

### `experiment/sample_size.v`

**Purpose:** Compute required sample size before running an experiment.

**Public API:**

```v
pub struct SampleSizeResult {
pub:
    n_per_group  int
    total_n      int
    alpha        f64
    power        f64
    mde          f64
    baseline     f64
    effect_size  f64   // Cohen's d for means; |p2 - p1| / pooled_se for proportions
    baseline_std f64   // echoed back for means; 0.0 for proportions
    method       string  // "proportions" or "means"
}

// For conversion/proportion metrics (click-through, signup, purchase rates).
// baseline_rate: current observed rate (e.g. 0.05 for 5%)
// mde: minimum detectable effect as absolute change in rate (e.g. 0.01 for +1pp)
//      i.e. p1 = baseline_rate, p2 = baseline_rate + mde
// alpha: significance level (e.g. 0.05)
// power: desired statistical power (e.g. 0.80)
pub fn sample_size_proportions(baseline_rate f64, mde f64, alpha f64, power f64) SampleSizeResult

// For continuous metrics (revenue, session length, latency).
// baseline_mean: current observed mean
// baseline_std: current observed standard deviation
// mde_absolute: minimum detectable effect as absolute change in mean
// alpha: significance level
// power: desired statistical power
pub fn sample_size_means(baseline_mean f64, baseline_std f64, mde_absolute f64, alpha f64, power f64) SampleSizeResult
```

**Math:**
- Proportions: `p1 = baseline_rate`, `p2 = baseline_rate + mde`. Formula: `n = (z_alpha/2 + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p2 - p1)^2`. Effect size field = `|mde| / sqrt((p1*(1-p1) + p2*(1-p2)) / 2)`.
- Means: `n = 2 * ((z_alpha/2 + z_beta) * baseline_std / mde_absolute)^2`. Effect size field = `mde_absolute / baseline_std` (Cohen's d).
- Both use `prob.inverse_normal_cdf` for z critical values.

---

### `experiment/proportion_test.v`

**Purpose:** Two-proportion z-test for comparing conversion rates directly.

**Public API:**

```v
@[params]
pub struct ProportionTestConfig {
pub:
    alpha f64 = 0.05
}

pub struct ProportionTestResult {
pub:
    rate_a        f64
    rate_b        f64
    diff          f64    // rate_b - rate_a
    relative_lift f64
    z_statistic   f64
    p_value       f64
    significant   bool
    ci_lower      f64    // CI on (rate_b - rate_a), uses alpha from config (not hard-coded 1.96)
    ci_upper      f64
    pooled_se     f64    // SE under H0 (pooled), used for z-statistic
    n_a           int
    n_b           int
}

pub fn proportion_test(successes_a int, n_a int, successes_b int, n_b int, cfg ProportionTestConfig) ProportionTestResult
```

**Math:** Pooled z-test. Under H₀: `p_pool = (successes_a + successes_b) / (n_a + n_b)`. `pooled_se = sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))`. `z = (rate_b - rate_a) / pooled_se`. CI on the difference uses the unpooled SE: `se_diff = sqrt(rate_a*(1-rate_a)/n_a + rate_b*(1-rate_b)/n_b)`, and `z_ci = inverse_normal_cdf(1 - alpha/2)` (from config, not hard-coded).

---

### `experiment/sequential.v`

**Purpose:** Sequential Probability Ratio Test (SPRT) — frequentist approach to safe interim analysis. Solves the "peeking problem": allows stopping early for significance or futility without inflating the false positive rate.

**Public API:**

```v
pub enum SPRTDecision {
    continue_testing
    reject_null      // significant — stop, B differs from A by at least MDE
    accept_null      // futile — stop, effect is below MDE
}

// Note: SPRTConfig is NOT @[params] because mde has no sensible default.
// Callers must construct it explicitly: SPRTConfig{ mde: 0.02 }
pub struct SPRTConfig {
pub:
    alpha f64 = 0.05   // desired false positive rate
    beta  f64 = 0.20   // desired false negative rate (1 - power)
    mde   f64          // minimum detectable effect (absolute rate difference); must be > 0
}

pub struct SPRTResult {
pub:
    log_likelihood_ratio f64
    decision             SPRTDecision
    upper_boundary       f64   // log((1 - beta) / alpha)  — reject null if LLR exceeds this
    lower_boundary       f64   // log(beta / (1 - alpha))  — accept null if LLR falls below this
    rate_a               f64
    rate_b               f64
    n_a                  int
    n_b                  int
}

pub fn sprt_test(successes_a int, n_a int, successes_b int, n_b int, cfg SPRTConfig) SPRTResult
```

**Note:** `sprt_test` is a stateless, one-shot function over cumulative totals. Callers are responsible for accumulating `successes` and `n` across observations and calling this function repeatedly at each interim check.

**Math:** Bernoulli SPRT. Null hypothesis: `p_b = p_a`. Alternative: `p_b = p_a + mde`. Let `p0 = successes_a / n_a` (observed control rate), `p1 = p0 + mde` (alternative rate). LLR: `successes_b * log(p1/p0) + (n_b - successes_b) * log((1-p1)/(1-p0))`. Upper boundary: `A = log((1 - beta) / alpha)`. Lower boundary: `B = log(beta / (1 - alpha))`. Decision: LLR > A → `reject_null`; LLR < B → `accept_null`; else → `continue_testing`. Assert `mde > 0` inside function.

---

### `experiment/bayesian.v`

**Purpose:** Bayesian A/B test for proportion data using the Beta-Binomial conjugate model. Returns probability that variant B beats A, expected loss, and credible intervals — no p-values.

**Public API:**

```v
@[params]
pub struct BayesianConfig {
pub:
    alpha_prior f64 = 1.0    // Beta prior shape parameter (default: uniform / non-informative)
    beta_prior  f64 = 1.0    // Beta prior shape parameter (default: uniform / non-informative)
    n_samples   int = 10000  // Monte Carlo samples for probability and loss estimation
}

pub struct BayesianResult {
pub:
    posterior_mean_a   f64
    posterior_mean_b   f64
    prob_b_beats_a     f64   // P(theta_B > theta_A) via Monte Carlo
    expected_loss_a    f64   // E[max(theta_B - theta_A, 0)] — cost of choosing A when B is better
    expected_loss_b    f64   // E[max(theta_A - theta_B, 0)] — cost of choosing B when A is better
    ci_lower_a         f64   // 95% credible interval for A's true rate
    ci_upper_a         f64
    ci_lower_b         f64
    ci_upper_b         f64
    successes_a        int
    successes_b        int
    n_a                int
    n_b                int
}

pub fn bayesian_ab_test(successes_a int, n_a int, successes_b int, n_b int, cfg BayesianConfig) BayesianResult
```

**Math:** Posterior for A: `Beta(alpha_prior + successes_a, beta_prior + n_a - successes_a)`. Same for B. Monte Carlo using Cheng's beta acceptance-rejection sampler (implemented as a private `beta_sample` helper — no external deps, uses `rand.f64()` from V stdlib):

```
// Cheng's BB algorithm for Beta(a, b) sampling when a > 1 and b > 1
// Falls back to ratio-of-uniforms for shape params <= 1
```

From N samples of each posterior:
- `prob_b_beats_a = count(sample_b > sample_a) / N`
- `expected_loss_a = mean(max(sample_b - sample_a, 0))`
- `expected_loss_b = mean(max(sample_a - sample_b, 0))`
- Credible intervals: 2.5th and 97.5th percentiles of sorted samples (use `stats.quantile`)

Posterior means: `alpha_post / (alpha_post + beta_post)` (analytic, not sampled).

---

## Tests

New test files in `tests/` using shorter names consistent with existing conventions:
- `tests/sample_size_test.v`
- `tests/proportion_test_test.v` — or `tests/prop_test_test.v` to avoid the double "test"
- `tests/sequential_test.v`
- `tests/bayesian_test.v`

Each covers: happy path with known-good values, edge cases (zero successes, equal groups, very small/large N), and assertions on `significant`/`decision` fields.

---

## Documentation

### `docs/modules/experiment.html` — Full Rewrite

Replaces the current 3-function stub. Structure:
1. **Overview & Method Selection Guide** — prose decision flowchart: use `proportion_test` for conversion rates, `abtest` for continuous metrics, `sequential` / `sprt_test` for early stopping, `bayesian_ab_test` when you want probability statements
2. **Full function reference** — every public function with parameter docs, return field descriptions, short code snippet
3. **Structs reference** — all config and result structs with field-level explanations

### `docs/concepts.html` — New "Experimentation" Section

Added after "Machine Learning Essentials":
- **Statistical power and sample size** — alpha, beta, MDE in plain English; why you compute N first
- **Frequentist vs. Bayesian** — intuition and when each is appropriate
- **The peeking problem** — why mid-experiment checks inflate false positives; how SPRT resolves this
- **Effect sizes** — Cohen's d for means; absolute vs. relative lift for proportions

### `docs/examples.html` — New "Experimentation Workflows" Section

Four end-to-end worked examples, each with business context + full code + annotated expected output:

1. **Conversion Rate Test** — `sample_size_proportions` → run experiment → `proportion_test` → interpret
2. **Continuous Metric Test** — `sample_size_means` → run experiment → `abtest` → interpret CI
3. **Sequential Test (SPRT)** — when to use early stopping, `sprt_test` called iteratively → interpret `decision`
4. **Bayesian Test** — when to choose Bayesian, `bayesian_ab_test` → interpret `prob_b_beats_a` and `expected_loss`

CUPED (`cuped_test`) is also annotated with a brief "when to use" note in the method selection guide.

---

## Constraints & Conventions

- No external dependencies — all math uses `prob`/`stats` modules and V stdlib `rand`
- Beta sampling uses Cheng's BB acceptance-rejection algorithm (private helper), which is exact and works for all shape parameter values
- Follow naming conventions: `snake_case` functions, `PascalCase` structs
- `@[params]` only on structs where ALL fields have sensible defaults; `SPRTConfig` is NOT `@[params]` because `mde` has no default
- CIs always use `alpha` from config, never hard-coded critical values
- All tests in `tests/`, not alongside source
- No markdown docs files (per CLAUDE.md)
