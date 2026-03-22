# Experiment Utilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sample size calculators, proportion z-test, SPRT sequential testing, and Bayesian A/B testing to the `experiment/` module, plus comprehensive documentation rewrites.

**Architecture:** Four new focused files in `experiment/` (one concern each), four test files in `tests/`, and HTML doc updates. The Bayesian sampler uses a private Gamma/Beta rejection sampler (Marsaglia-Tsang algorithm) so no external dependencies are introduced.

**Tech Stack:** V language, vstats internal modules (`prob`, `stats`, `math`, `rand`), HTML/CSS for docs.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `experiment/sample_size.v` | `sample_size_proportions`, `sample_size_means` |
| Create | `experiment/proportion_test.v` | `proportion_test` (two-proportion z-test) |
| Create | `experiment/sequential.v` | `sprt_test` (SPRT sequential testing) |
| Create | `experiment/bayesian.v` | `bayesian_ab_test` + private Beta sampler |
| Create | `tests/sample_size_test.v` | Tests for sample_size.v |
| Create | `tests/proportion_ab_test.v` | Tests for proportion_test.v |
| Create | `tests/sequential_test.v` | Tests for sequential.v |
| Create | `tests/bayesian_test.v` | Tests for bayesian.v |
| Modify | `docs/modules/experiment.html` | Full API reference rewrite |
| Modify | `docs/concepts.html` | Add Experimentation Concepts section |
| Modify | `docs/examples.html` | Add Experimentation Workflows section |

---

## Task 1: Sample Size Calculator

**Files:**
- Create: `experiment/sample_size.v`
- Create: `tests/sample_size_test.v`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/sample_size_test.v`:

```v
import experiment
import math

fn test__sample_size_proportions_known_value() {
	// baseline=5%, mde=+1pp, alpha=0.05, power=0.80
	// Expected n ≈ 8000 per group (classic result for this setting)
	result := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.80)

	assert result.method == 'proportions'
	assert math.abs(f64(result.n_per_group) - 8100.0) < 500.0
	assert result.total_n == result.n_per_group * 2
	assert result.alpha == 0.05
	assert result.power == 0.80
	assert result.mde == 0.01
	assert result.baseline == 0.05
	assert result.baseline_std == 0.0
	assert result.effect_size > 0.0
}

fn test__sample_size_proportions_larger_effect() {
	// baseline=10%, mde=+5pp — bigger effect => fewer needed
	small := experiment.sample_size_proportions(0.10, 0.05, 0.05, 0.80)
	large := experiment.sample_size_proportions(0.10, 0.01, 0.05, 0.80)

	assert small.n_per_group < large.n_per_group
}

fn test__sample_size_proportions_higher_power() {
	// Higher power => larger sample
	low_power := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.80)
	high_power := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.90)

	assert high_power.n_per_group > low_power.n_per_group
}

fn test__sample_size_means_known_value() {
	// baseline_mean=100, std=20, mde=5, alpha=0.05, power=0.80
	// Cohen's d = 5/20 = 0.25, n = 2*((1.96+0.842)*20/5)^2 ≈ 252
	result := experiment.sample_size_means(100.0, 20.0, 5.0, 0.05, 0.80)

	assert result.method == 'means'
	assert math.abs(f64(result.n_per_group) - 252.0) < 15.0
	assert result.total_n == result.n_per_group * 2
	assert result.baseline == 100.0
	assert result.baseline_std == 20.0
	assert math.abs(result.effect_size - 0.25) < 0.01
	assert result.mde == 5.0
}

fn test__sample_size_means_higher_std() {
	// More variance => larger sample needed
	low_var := experiment.sample_size_means(100.0, 10.0, 5.0, 0.05, 0.80)
	high_var := experiment.sample_size_means(100.0, 30.0, 5.0, 0.05, 0.80)

	assert high_var.n_per_group > low_var.n_per_group
}
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
v test tests/sample_size_test.v
```
Expected: compilation error — `experiment.sample_size_proportions` not found.

- [ ] **Step 1.3: Implement `experiment/sample_size.v`**

```v
module experiment

import math
import prob

pub struct SampleSizeResult {
pub:
	n_per_group  int
	total_n      int
	alpha        f64
	power        f64
	mde          f64
	baseline     f64
	effect_size  f64
	baseline_std f64
	method       string
}

// sample_size_proportions computes the required sample size per group to detect
// a minimum detectable effect (mde) in a proportion metric.
//
// baseline_rate: current conversion rate (e.g. 0.05 for 5%)
// mde:           absolute change to detect (e.g. 0.01 to detect 5% -> 6%)
// alpha:         significance level (e.g. 0.05)
// power:         desired power (e.g. 0.80)
pub fn sample_size_proportions(baseline_rate f64, mde f64, alpha f64, power f64) SampleSizeResult {
	assert baseline_rate > 0 && baseline_rate < 1, 'baseline_rate must be in (0, 1)'
	assert mde > 0, 'mde must be positive'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	assert power > 0 && power < 1, 'power must be in (0, 1)'

	p1 := baseline_rate
	p2 := baseline_rate + mde
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)

	numerator := math.pow(z_alpha + z_beta, 2) * (p1 * (1.0 - p1) + p2 * (1.0 - p2))
	denominator := math.pow(p2 - p1, 2)
	n_raw := numerator / denominator
	n := int(math.ceil(n_raw))

	pooled_var := (p1 * (1.0 - p1) + p2 * (1.0 - p2)) / 2.0
	effect := if pooled_var > 0 { math.abs(mde) / math.sqrt(pooled_var) } else { 0.0 }

	return SampleSizeResult{
		n_per_group:  n
		total_n:      n * 2
		alpha:        alpha
		power:        power
		mde:          mde
		baseline:     baseline_rate
		effect_size:  effect
		baseline_std: 0.0
		method:       'proportions'
	}
}

// sample_size_means computes the required sample size per group to detect
// a minimum detectable effect in a continuous metric.
//
// baseline_mean: current mean value
// baseline_std:  current standard deviation (must be estimated from historical data)
// mde_absolute:  absolute change in mean to detect
// alpha:         significance level (e.g. 0.05)
// power:         desired power (e.g. 0.80)
pub fn sample_size_means(baseline_mean f64, baseline_std f64, mde_absolute f64, alpha f64, power f64) SampleSizeResult {
	assert baseline_std > 0, 'baseline_std must be positive'
	assert mde_absolute > 0, 'mde_absolute must be positive'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	assert power > 0 && power < 1, 'power must be in (0, 1)'

	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)

	n_raw := 2.0 * math.pow((z_alpha + z_beta) * baseline_std / mde_absolute, 2)
	n := int(math.ceil(n_raw))
	cohens_d := mde_absolute / baseline_std

	return SampleSizeResult{
		n_per_group:  n
		total_n:      n * 2
		alpha:        alpha
		power:        power
		mde:          mde_absolute
		baseline:     baseline_mean
		effect_size:  cohens_d
		baseline_std: baseline_std
		method:       'means'
	}
}
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
v test tests/sample_size_test.v
```
Expected: all 5 tests PASS.

- [ ] **Step 1.5: Run full test suite to check nothing broke**

```bash
make test
```
Expected: all tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add experiment/sample_size.v tests/sample_size_test.v
git commit -m "feat(experiment): add sample_size_proportions and sample_size_means"
```

---

## Task 2: Two-Proportion Z-Test

**Files:**
- Create: `experiment/proportion_test.v`
- Create: `tests/proportion_ab_test.v`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/proportion_ab_test.v`:

```v
import experiment
import math

fn test__proportion_test_significant() {
	// 30/300 (10%) vs 48/300 (16%) — clear lift
	result := experiment.proportion_test(30, 300, 48, 300)

	assert result.rate_a == 0.10
	assert result.rate_b == 0.16
	assert math.abs(result.diff - 0.06) < 0.001
	assert result.z_statistic > 1.96
	assert result.p_value < 0.05
	assert result.significant == true
	assert result.ci_lower > 0.0   // CI excludes zero
	assert result.pooled_se > 0.0
	assert result.relative_lift > 0.0
}

fn test__proportion_test_not_significant() {
	// 50/500 vs 50/500 — identical rates
	result := experiment.proportion_test(50, 500, 50, 500)

	assert math.abs(result.diff) < 0.001
	assert math.abs(result.z_statistic) < 0.001
	assert result.p_value > 0.5
	assert result.significant == false
	assert result.ci_lower < 0.0
	assert result.ci_upper > 0.0
}

fn test__proportion_test_ci_width_scales_with_alpha() {
	// Tighter alpha => wider CI
	narrow := experiment.proportion_test(30, 300, 48, 300, experiment.ProportionTestConfig{ alpha: 0.10 })
	wide := experiment.proportion_test(30, 300, 48, 300, experiment.ProportionTestConfig{ alpha: 0.01 })

	narrow_width := narrow.ci_upper - narrow.ci_lower
	wide_width := wide.ci_upper - wide.ci_lower
	assert wide_width > narrow_width
}

fn test__proportion_test_relative_lift() {
	// rate_a=0.10, rate_b=0.20 => relative lift = (0.20-0.10)/0.10 = 1.0 (100%)
	result := experiment.proportion_test(100, 1000, 200, 1000)

	assert math.abs(result.relative_lift - 1.0) < 0.01
}

fn test__proportion_test_echoes_n() {
	result := experiment.proportion_test(30, 300, 48, 500)

	assert result.n_a == 300
	assert result.n_b == 500
}
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
v test tests/proportion_ab_test.v
```
Expected: compilation error — `experiment.proportion_test` not found.

- [ ] **Step 2.3: Implement `experiment/proportion_test.v`**

```v
module experiment

import math
import prob

@[params]
pub struct ProportionTestConfig {
pub:
	alpha f64 = 0.05
}

pub struct ProportionTestResult {
pub:
	rate_a        f64
	rate_b        f64
	diff          f64
	relative_lift f64
	z_statistic   f64
	p_value       f64
	significant   bool
	ci_lower      f64
	ci_upper      f64
	pooled_se     f64
	n_a           int
	n_b           int
}

// proportion_test runs a two-proportion z-test comparing the conversion rates
// of two groups. Use this for binary outcomes: click, signup, purchase, etc.
//
// successes_a/n_a: events and total observations for group A (control)
// successes_b/n_b: events and total observations for group B (treatment)
pub fn proportion_test(successes_a int, n_a int, successes_b int, n_b int, cfg ProportionTestConfig) ProportionTestResult {
	assert n_a > 0 && n_b > 0, 'group sizes must be positive'
	assert successes_a >= 0 && successes_a <= n_a, 'successes_a out of range'
	assert successes_b >= 0 && successes_b <= n_b, 'successes_b out of range'

	rate_a := f64(successes_a) / f64(n_a)
	rate_b := f64(successes_b) / f64(n_b)
	diff := rate_b - rate_a
	lift := if rate_a > 0 { diff / rate_a } else { 0.0 }

	// Pooled SE under H0 (used for z-statistic)
	p_pool := f64(successes_a + successes_b) / f64(n_a + n_b)
	pse := math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / f64(n_a) + 1.0 / f64(n_b)))
	z := if pse > 0 { diff / pse } else { 0.0 }
	p_val := 2.0 * prob.normal_cdf(-math.abs(z), 0.0, 1.0)

	// Unpooled SE for CI (uses alpha from config, never hard-coded)
	se_diff := math.sqrt(rate_a * (1.0 - rate_a) / f64(n_a) + rate_b * (1.0 - rate_b) / f64(n_b))
	z_ci := prob.inverse_normal_cdf(1.0 - cfg.alpha / 2.0, 0.0, 1.0)
	ci_lo := diff - z_ci * se_diff
	ci_hi := diff + z_ci * se_diff

	return ProportionTestResult{
		rate_a:        rate_a
		rate_b:        rate_b
		diff:          diff
		relative_lift: lift
		z_statistic:   z
		p_value:       p_val
		significant:   p_val < cfg.alpha
		ci_lower:      ci_lo
		ci_upper:      ci_hi
		pooled_se:     pse
		n_a:           n_a
		n_b:           n_b
	}
}
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
v test tests/proportion_ab_test.v
```
Expected: all 5 tests PASS.

- [ ] **Step 2.5: Run full test suite**

```bash
make test
```
Expected: all tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add experiment/proportion_test.v tests/proportion_ab_test.v
git commit -m "feat(experiment): add proportion_test (two-proportion z-test)"
```

---

## Task 3: Sequential Testing (SPRT)

**Files:**
- Create: `experiment/sequential.v`
- Create: `tests/sequential_test.v`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/sequential_test.v`:

```v
import experiment
import math

fn test__sprt_reject_null() {
	// Control: 50/1000 (rate=0.05). Treatment: 80/1000 (rate=0.08).
	// mde=0.02 means alternative is p0+0.02=0.07. Since 0.08 > 0.07 with enough data, should reject.
	// LLR = 80*log(0.07/0.05) + 920*log(0.93/0.95) ≈ 7.3 > upper_boundary ≈ 2.77
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 80, 1000, cfg)

	assert result.decision == .reject_null
	assert result.log_likelihood_ratio > result.upper_boundary
	assert math.abs(result.upper_boundary - math.log(0.8 / 0.05)) < 0.01
	assert math.abs(result.lower_boundary - math.log(0.2 / 0.95)) < 0.01
	assert result.rate_a == 0.05
	assert result.n_a == 1000
	assert result.n_b == 1000
}

fn test__sprt_accept_null() {
	// Control: 50/1000 (rate=0.05). Treatment: 40/1000 (rate=0.04).
	// Effect is in the wrong direction — LLR should fall below lower boundary.
	// LLR = 40*log(0.07/0.05) + 960*log(0.93/0.95) ≈ -7.0 < lower_boundary ≈ -1.56
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 40, 1000, cfg)

	assert result.decision == .accept_null
	assert result.log_likelihood_ratio < result.lower_boundary
}

fn test__sprt_continue_testing() {
	// Control: 50/1000 (rate=0.05). Treatment: 60/1000 (rate=0.06).
	// Small positive effect — not enough evidence yet.
	// LLR ≈ 0.19, between boundaries (-1.56, 2.77)
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 60, 1000, cfg)

	assert result.decision == .continue_testing
	assert result.log_likelihood_ratio > result.lower_boundary
	assert result.log_likelihood_ratio < result.upper_boundary
}

fn test__sprt_boundaries_correct() {
	// Upper = log((1-beta)/alpha) = log(0.8/0.05) = log(16) ≈ 2.773
	// Lower = log(beta/(1-alpha)) = log(0.2/0.95) ≈ -1.558
	cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }
	result := experiment.sprt_test(50, 1000, 60, 1000, cfg)

	assert math.abs(result.upper_boundary - math.log(16.0)) < 0.01
	assert math.abs(result.lower_boundary - math.log(0.2 / 0.95)) < 0.01
}
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
v test tests/sequential_test.v
```
Expected: compilation error — `experiment.SPRTConfig` not found.

- [ ] **Step 3.3: Implement `experiment/sequential.v`**

```v
module experiment

import math

pub enum SPRTDecision {
	continue_testing
	reject_null
	accept_null
}

// SPRTConfig configures the Sequential Probability Ratio Test.
// Note: NOT @[params] because mde has no sensible default — callers must
// set it explicitly: SPRTConfig{ mde: 0.02 }
pub struct SPRTConfig {
pub:
	alpha f64 = 0.05
	beta  f64 = 0.20
	mde   f64
}

pub struct SPRTResult {
pub:
	log_likelihood_ratio f64
	decision             SPRTDecision
	upper_boundary       f64
	lower_boundary       f64
	rate_a               f64
	rate_b               f64
	n_a                  int
	n_b                  int
}

// sprt_test applies the Sequential Probability Ratio Test (SPRT) to accumulated
// experiment data. Call this function repeatedly as data accumulates — it is
// stateless and works on cumulative totals.
//
// The SPRT solves the "peeking problem": it allows you to check results at any
// time without inflating the false positive rate, by using Wald's likelihood
// ratio boundaries.
//
// successes_a/n_a: cumulative events and observations for control
// successes_b/n_b: cumulative events and observations for treatment
// cfg.mde:         minimum detectable effect (absolute rate difference, must be > 0)
pub fn sprt_test(successes_a int, n_a int, successes_b int, n_b int, cfg SPRTConfig) SPRTResult {
	assert cfg.mde > 0, 'mde must be positive'
	assert n_a > 0 && n_b > 0, 'group sizes must be positive'
	assert successes_a >= 0 && successes_a <= n_a, 'successes_a out of range'
	assert successes_b >= 0 && successes_b <= n_b, 'successes_b out of range'

	// Wald boundaries
	upper := math.log((1.0 - cfg.beta) / cfg.alpha)
	lower := math.log(cfg.beta / (1.0 - cfg.alpha))

	// H0: p_b = p0 (observed control rate)
	// H1: p_b = p1 = p0 + mde
	p0 := f64(successes_a) / f64(n_a)
	p1 := p0 + cfg.mde

	// Clamp to avoid log(0) — rates must be in (0, 1)
	p0c := math.min(math.max(p0, 1e-10), 1.0 - 1e-10)
	p1c := math.min(math.max(p1, 1e-10), 1.0 - 1e-10)

	sb := f64(successes_b)
	fb := f64(n_b - successes_b)
	llr := sb * math.log(p1c / p0c) + fb * math.log((1.0 - p1c) / (1.0 - p0c))

	decision := if llr >= upper {
		SPRTDecision.reject_null
	} else if llr <= lower {
		SPRTDecision.accept_null
	} else {
		SPRTDecision.continue_testing
	}

	return SPRTResult{
		log_likelihood_ratio: llr
		decision:             decision
		upper_boundary:       upper
		lower_boundary:       lower
		rate_a:               p0
		rate_b:               f64(successes_b) / f64(n_b)
		n_a:                  n_a
		n_b:                  n_b
	}
}
```

Note: `sequential.v` only uses `math` — `prob` is intentionally omitted.

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
v test tests/sequential_test.v
```
Expected: all 4 tests PASS.

- [ ] **Step 3.5: Run full test suite**

```bash
make test
```
Expected: all tests pass.

- [ ] **Step 3.6: Commit**

```bash
git add experiment/sequential.v tests/sequential_test.v
git commit -m "feat(experiment): add sprt_test (sequential probability ratio test)"
```

---

## Task 4: Bayesian A/B Test

**Files:**
- Create: `experiment/bayesian.v`
- Create: `tests/bayesian_test.v`

The Bayesian implementation uses a private Beta sampler based on Gamma random variates (Marsaglia-Tsang algorithm). This is a standard rejection sampler — no external dependencies needed.

- [ ] **Step 4.1: Write the failing tests**

Create `tests/bayesian_test.v`:

```v
import experiment
import math

fn test__bayesian_b_clearly_better() {
	// A: 10/100 (10%), B: 40/100 (40%) — B is much better
	result := experiment.bayesian_ab_test(10, 100, 40, 100)

	assert result.prob_b_beats_a > 0.99
	assert result.expected_loss_b < result.expected_loss_a
	assert result.posterior_mean_a < result.posterior_mean_b
	assert result.successes_a == 10
	assert result.successes_b == 40
	assert result.n_a == 100
	assert result.n_b == 100
}

fn test__bayesian_groups_equal() {
	// A: 50/100 (50%), B: 50/100 (50%) — roughly equal
	result := experiment.bayesian_ab_test(50, 100, 50, 100)

	// With equal data, prob_b_beats_a should be close to 0.5
	assert result.prob_b_beats_a > 0.35
	assert result.prob_b_beats_a < 0.65
	// Expected losses should be symmetric and small
	assert math.abs(result.expected_loss_a - result.expected_loss_b) < 0.05
}

fn test__bayesian_credible_intervals_ordered() {
	result := experiment.bayesian_ab_test(10, 100, 40, 100)

	assert result.ci_lower_a < result.posterior_mean_a
	assert result.ci_upper_a > result.posterior_mean_a
	assert result.ci_lower_b < result.posterior_mean_b
	assert result.ci_upper_b > result.posterior_mean_b
}

fn test__bayesian_posterior_means_correct() {
	// Posterior mean for Beta(alpha+s, beta+f) = (alpha+s) / (alpha+s + beta+f)
	// With uniform prior (alpha=1, beta=1):
	// A: 10/100 => posterior mean = (1+10)/(1+10+1+90) = 11/102 ≈ 0.1078
	// B: 40/100 => posterior mean = (1+40)/(1+40+1+60) = 41/102 ≈ 0.4020
	result := experiment.bayesian_ab_test(10, 100, 40, 100)

	assert math.abs(result.posterior_mean_a - 11.0 / 102.0) < 0.001
	assert math.abs(result.posterior_mean_b - 41.0 / 102.0) < 0.001
}

fn test__bayesian_informative_prior() {
	// With informative prior, posterior should be pulled toward prior
	cfg := experiment.BayesianConfig{ alpha_prior: 10.0, beta_prior: 90.0 }
	result := experiment.bayesian_ab_test(10, 100, 40, 100, cfg)

	// Prior says rate ≈ 10%, strong pull on A, less on B
	// A posterior should be pulled toward 10% (prior mean)
	// B posterior should still be above A
	assert result.posterior_mean_b > result.posterior_mean_a
}
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
v test tests/bayesian_test.v
```
Expected: compilation error — `experiment.bayesian_ab_test` not found.

- [ ] **Step 4.3: Implement `experiment/bayesian.v`**

```v
module experiment

import math
import rand
import stats

@[params]
pub struct BayesianConfig {
pub:
	alpha_prior f64 = 1.0
	beta_prior  f64 = 1.0
	n_samples   int = 10000
}

pub struct BayesianResult {
pub:
	posterior_mean_a f64
	posterior_mean_b f64
	prob_b_beats_a   f64
	expected_loss_a  f64
	expected_loss_b  f64
	ci_lower_a       f64
	ci_upper_a       f64
	ci_lower_b       f64
	ci_upper_b       f64
	successes_a      int
	successes_b      int
	n_a              int
	n_b              int
}

// box_muller returns a standard normal sample using the Box-Muller transform.
fn box_muller() f64 {
	mut u1 := rand.f64()
	if u1 <= 0.0 {
		u1 = 1e-15
	}
	u2 := rand.f64()
	return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
}

// gamma_sample returns a sample from Gamma(shape, 1) using Marsaglia-Tsang (2000).
// Works for shape > 0.
fn gamma_sample(shape f64) f64 {
	if shape < 1.0 {
		u := rand.f64()
		if u <= 0.0 {
			return 0.0
		}
		return gamma_sample(1.0 + shape) * math.pow(u, 1.0 / shape)
	}
	d := shape - 1.0 / 3.0
	c := 1.0 / math.sqrt(9.0 * d)
	for {
		mut x := f64(0)
		mut v := f64(0)
		for {
			x = box_muller()
			v = 1.0 + c * x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := rand.f64()
		x2 := x * x
		if u < 1.0 - 0.0331 * x2 * x2 {
			return d * v
		}
		if math.log(u) < 0.5 * x2 + d * (1.0 - v + math.log(v)) {
			return d * v
		}
	}
	return 0.0 // unreachable
}

// beta_sample returns a sample from Beta(a, b) using the Gamma ratio method.
fn beta_sample(a f64, b f64) f64 {
	x := gamma_sample(a)
	y := gamma_sample(b)
	total := x + y
	if total <= 0.0 {
		return 0.5
	}
	return x / total
}

// bayesian_ab_test runs a Bayesian A/B test using the Beta-Binomial conjugate model.
// Returns the probability that B beats A, expected losses, and 95% credible intervals.
//
// Uses a uniform (non-informative) prior by default: Beta(1, 1).
// For informative priors, set alpha_prior and beta_prior in BayesianConfig.
//
// The result does NOT include p-values. Instead interpret:
//   prob_b_beats_a: probability B's true rate exceeds A's
//   expected_loss_b: expected regret if you deploy B (and A was actually better)
pub fn bayesian_ab_test(successes_a int, n_a int, successes_b int, n_b int, cfg BayesianConfig) BayesianResult {
	assert n_a > 0 && n_b > 0, 'group sizes must be positive'
	assert successes_a >= 0 && successes_a <= n_a, 'successes_a out of range'
	assert successes_b >= 0 && successes_b <= n_b, 'successes_b out of range'
	assert cfg.alpha_prior > 0 && cfg.beta_prior > 0, 'priors must be positive'
	assert cfg.n_samples > 0, 'n_samples must be positive'

	// Posterior parameters: Beta(alpha_prior + successes, beta_prior + failures)
	alpha_a := cfg.alpha_prior + f64(successes_a)
	beta_a := cfg.beta_prior + f64(n_a - successes_a)
	alpha_b := cfg.alpha_prior + f64(successes_b)
	beta_b := cfg.beta_prior + f64(n_b - successes_b)

	// Analytic posterior means
	mean_a := alpha_a / (alpha_a + beta_a)
	mean_b := alpha_b / (alpha_b + beta_b)

	// Monte Carlo
	n := cfg.n_samples
	mut samples_a := []f64{len: n}
	mut samples_b := []f64{len: n}
	mut b_beats := 0
	mut loss_a := 0.0
	mut loss_b := 0.0

	for i in 0 .. n {
		sa := beta_sample(alpha_a, beta_a)
		sb := beta_sample(alpha_b, beta_b)
		samples_a[i] = sa
		samples_b[i] = sb
		if sb > sa {
			b_beats++
		}
		diff := sb - sa
		if diff > 0 {
			loss_a += diff
		} else {
			loss_b += -diff
		}
	}

	prob_b := f64(b_beats) / f64(n)
	exp_loss_a := loss_a / f64(n)
	exp_loss_b := loss_b / f64(n)

	// Credible intervals via percentiles of sorted samples
	ci_lo_a := stats.quantile(samples_a, 0.025)
	ci_hi_a := stats.quantile(samples_a, 0.975)
	ci_lo_b := stats.quantile(samples_b, 0.025)
	ci_hi_b := stats.quantile(samples_b, 0.975)

	return BayesianResult{
		posterior_mean_a: mean_a
		posterior_mean_b: mean_b
		prob_b_beats_a:   prob_b
		expected_loss_a:  exp_loss_a
		expected_loss_b:  exp_loss_b
		ci_lower_a:       ci_lo_a
		ci_upper_a:       ci_hi_a
		ci_lower_b:       ci_lo_b
		ci_upper_b:       ci_hi_b
		successes_a:      successes_a
		successes_b:      successes_b
		n_a:              n_a
		n_b:              n_b
	}
}
```

- [ ] **Step 4.4: Run tests to confirm they pass**

```bash
v test tests/bayesian_test.v
```
Expected: all 5 tests PASS. Note: the Monte Carlo tests (`prob_b_beats_a`) use wide bounds (> 0.99, 0.35–0.65) to tolerate natural sampling variation.

- [ ] **Step 4.5: Run full test suite**

```bash
make test
```
Expected: all tests pass.

- [ ] **Step 4.6: Commit**

```bash
git add experiment/bayesian.v tests/bayesian_test.v
git commit -m "feat(experiment): add bayesian_ab_test with Beta-Binomial conjugate model"
```

---

## Task 5: Experiment Module API Reference

**Files:**
- Modify: `docs/modules/experiment.html`

Replace the thin 3-function stub with a complete API reference covering all 10+ public functions.

- [ ] **Step 5.1: Rewrite `docs/modules/experiment.html`**

Replace the entire file content with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>experiment - VStats</title>
    <link rel="stylesheet" href="../css/style.css">
</head>
<body>
    <div class="layout">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="../index.html" class="logo">VStats</a>
            </div>
            <nav class="sidebar-nav">
                <ul class="nav-list">
                    <li><a href="../index.html">Home</a></li>
                    <li><a href="../concepts.html">Concepts</a></li>
                    <li><a href="../examples.html">Examples</a></li>
                    <li><a href="../api.html">API Reference</a></li>
                </ul>
            </nav>
            <div class="toc">
                <h3>On This Page</h3>
                <ul class="toc-list">
                    <li><a href="#overview">Overview</a></li>
                    <li><a href="#method-guide">Which Method?</a></li>
                    <li><a href="#sample-size">Sample Size</a></li>
                    <li><a href="#proportion-test">Proportion Test</a></li>
                    <li><a href="#abtest">A/B Test (Continuous)</a></li>
                    <li><a href="#sequential">Sequential Testing</a></li>
                    <li><a href="#bayesian">Bayesian Test</a></li>
                    <li><a href="#cuped">CUPED</a></li>
                    <li><a href="#causal">Causal Inference</a></li>
                </ul>
            </div>
        </aside>
        <main class="content">
            <header class="module-header">
                <h1>experiment</h1>
                <p class="module-description">A/B Testing, Causal Inference &amp; Experimentation</p>
            </header>

            <section id="overview">
                <h2>Overview</h2>
                <p>The <code>experiment</code> module provides tools for designing and analyzing controlled experiments. It covers the full lifecycle: calculating required sample size before launch, running statistical tests after the experiment, and applying variance reduction or causal inference methods.</p>
                <p>Import with: <code>import vstats.experiment</code></p>
            </section>

            <section id="method-guide">
                <h2>Which Method Should I Use?</h2>
                <table>
                    <tr>
                        <th>Situation</th>
                        <th>Function</th>
                        <th>Why</th>
                    </tr>
                    <tr>
                        <td>Planning: how many users do I need?</td>
                        <td><a href="#sample-size"><code>sample_size_proportions</code> / <code>sample_size_means</code></a></td>
                        <td>Always compute sample size <em>before</em> running your experiment</td>
                    </tr>
                    <tr>
                        <td>Comparing click-through, signup, or purchase rates</td>
                        <td><a href="#proportion-test"><code>proportion_test</code></a></td>
                        <td>Binary outcomes need a z-test, not a t-test</td>
                    </tr>
                    <tr>
                        <td>Comparing revenue, session length, or other continuous metrics</td>
                        <td><a href="#abtest"><code>abtest</code></a></td>
                        <td>Welch's t-test handles unequal variance between groups</td>
                    </tr>
                    <tr>
                        <td>Need to check results early without inflating false positives</td>
                        <td><a href="#sequential"><code>sprt_test</code></a></td>
                        <td>SPRT gives statistically valid early-stopping decisions</td>
                    </tr>
                    <tr>
                        <td>Want probability statements, not p-values</td>
                        <td><a href="#bayesian"><code>bayesian_ab_test</code></a></td>
                        <td>Bayesian inference gives "P(B beats A)" directly</td>
                    </tr>
                    <tr>
                        <td>High-variance metric; want to reduce required sample size</td>
                        <td><a href="#cuped"><code>cuped_test</code></a></td>
                        <td>CUPED uses pre-experiment data to absorb variance</td>
                    </tr>
                    <tr>
                        <td>Treatment was not randomly assigned; observational data</td>
                        <td><a href="#causal">PSM / DiD</a></td>
                        <td>Propensity matching and difference-in-differences adjust for confounders</td>
                    </tr>
                </table>
            </section>

            <section id="sample-size">
                <h2>Sample Size Calculators</h2>
                <p>Always calculate your required sample size <strong>before</strong> running an experiment. Starting with too few observations leads to underpowered tests that miss real effects.</p>

                <div class="function-item">
                    <h3 id="sample_size_proportions">
                        <code>fn sample_size_proportions(baseline_rate f64, mde f64, alpha f64, power f64) SampleSizeResult</code>
                    </h3>
                    <p>Computes required observations per group for a binary metric (clicks, signups, purchases).</p>
                    <table>
                        <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
                        <tr><td><code>baseline_rate</code></td><td>f64</td><td>Your current conversion rate, e.g. <code>0.05</code> for 5%</td></tr>
                        <tr><td><code>mde</code></td><td>f64</td><td>Minimum detectable effect as an <em>absolute</em> change, e.g. <code>0.01</code> to detect a 5% → 6% lift</td></tr>
                        <tr><td><code>alpha</code></td><td>f64</td><td>Significance level. Use <code>0.05</code> for 95% confidence</td></tr>
                        <tr><td><code>power</code></td><td>f64</td><td>Desired statistical power. Use <code>0.80</code> (80% chance of detecting a real effect)</td></tr>
                    </table>
                    <pre><code>result := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.80)
println(result.n_per_group)  // ~8100 — needed per group
println(result.total_n)      // ~16200 — total experiment size</code></pre>
                </div>

                <div class="function-item">
                    <h3 id="sample_size_means">
                        <code>fn sample_size_means(baseline_mean f64, baseline_std f64, mde_absolute f64, alpha f64, power f64) SampleSizeResult</code>
                    </h3>
                    <p>Computes required observations per group for a continuous metric (revenue, latency, engagement).</p>
                    <table>
                        <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
                        <tr><td><code>baseline_mean</code></td><td>f64</td><td>Current average value</td></tr>
                        <tr><td><code>baseline_std</code></td><td>f64</td><td>Current standard deviation (estimate from historical data)</td></tr>
                        <tr><td><code>mde_absolute</code></td><td>f64</td><td>Absolute change in mean to detect, e.g. <code>5.0</code> to detect a $5 revenue lift</td></tr>
                        <tr><td><code>alpha</code></td><td>f64</td><td>Significance level</td></tr>
                        <tr><td><code>power</code></td><td>f64</td><td>Desired power</td></tr>
                    </table>
                    <pre><code>result := experiment.sample_size_means(45.0, 20.0, 3.0, 0.05, 0.80)
println(result.n_per_group)  // observations per group
println(result.effect_size)  // Cohen's d = mde / std</code></pre>
                </div>

                <div class="struct-item">
                    <h3><code>struct SampleSizeResult</code></h3>
                    <table>
                        <tr><th>Field</th><th>Type</th><th>Description</th></tr>
                        <tr><td><code>n_per_group</code></td><td>int</td><td>Required observations per group (ceiling-rounded)</td></tr>
                        <tr><td><code>total_n</code></td><td>int</td><td><code>n_per_group * 2</code></td></tr>
                        <tr><td><code>alpha</code></td><td>f64</td><td>Significance level used</td></tr>
                        <tr><td><code>power</code></td><td>f64</td><td>Power used</td></tr>
                        <tr><td><code>mde</code></td><td>f64</td><td>MDE used</td></tr>
                        <tr><td><code>baseline</code></td><td>f64</td><td>Baseline rate or mean used</td></tr>
                        <tr><td><code>effect_size</code></td><td>f64</td><td>Implied effect size (Cohen's d for means; h-style for proportions)</td></tr>
                        <tr><td><code>baseline_std</code></td><td>f64</td><td>Std dev used (means only; 0 for proportions)</td></tr>
                        <tr><td><code>method</code></td><td>string</td><td><code>"proportions"</code> or <code>"means"</code></td></tr>
                    </table>
                </div>
            </section>

            <section id="proportion-test">
                <h2>Proportion Test</h2>
                <p>Use this when your outcome is binary: each user either converted or didn't. This is the correct test for click-through rates, signup rates, purchase rates, etc.</p>

                <div class="function-item">
                    <h3 id="proportion_test">
                        <code>fn proportion_test(successes_a int, n_a int, successes_b int, n_b int, cfg ProportionTestConfig) ProportionTestResult</code>
                    </h3>
                    <p>Two-proportion z-test using a pooled standard error under H₀. The confidence interval uses the unpooled standard error.</p>
                    <pre><code>// 30 conversions / 300 users vs 48 conversions / 300 users
result := experiment.proportion_test(30, 300, 48, 300)
println(result.rate_a)        // 0.10 (10%)
println(result.rate_b)        // 0.16 (16%)
println(result.relative_lift) // 0.60 (60% relative lift)
println(result.p_value)       // e.g. 0.012
println(result.significant)   // true
println('${(result.diff * 100):.1f}% ± ${((result.ci_upper - result.ci_lower) / 2 * 100):.1f}pp')</code></pre>
                </div>

                <div class="struct-item">
                    <h3><code>struct ProportionTestResult</code></h3>
                    <table>
                        <tr><th>Field</th><th>Description</th></tr>
                        <tr><td><code>rate_a / rate_b</code></td><td>Observed conversion rates</td></tr>
                        <tr><td><code>diff</code></td><td><code>rate_b - rate_a</code> (absolute difference)</td></tr>
                        <tr><td><code>relative_lift</code></td><td><code>diff / rate_a</code></td></tr>
                        <tr><td><code>z_statistic</code></td><td>Test statistic (pooled SE)</td></tr>
                        <tr><td><code>p_value</code></td><td>Two-tailed p-value</td></tr>
                        <tr><td><code>significant</code></td><td>Whether <code>p_value &lt; alpha</code></td></tr>
                        <tr><td><code>ci_lower / ci_upper</code></td><td>Confidence interval on the difference (unpooled SE)</td></tr>
                        <tr><td><code>pooled_se</code></td><td>Standard error under H₀ (used for z-statistic)</td></tr>
                        <tr><td><code>n_a / n_b</code></td><td>Group sizes</td></tr>
                    </table>
                </div>
            </section>

            <section id="abtest">
                <h2>A/B Test — Continuous Metrics</h2>
                <div class="function-item">
                    <h3 id="abtest_fn">
                        <code>fn abtest(control []f64, treatment []f64, cfg ABTestConfig) ABTestResult</code>
                    </h3>
                    <p>Welch's t-test for continuous metrics. Handles unequal variances and unequal group sizes. Returns Cohen's d effect size, confidence interval on the mean difference, and relative lift.</p>
                    <pre><code>control   := [45.2, 43.1, 47.8, 44.3, 46.0]  // revenue per user
treatment := [48.1, 50.3, 47.9, 51.2, 49.5]
result := experiment.abtest(control, treatment)
println(result.p_value)      // e.g. 0.031
println(result.relative_lift) // e.g. 0.092 (9.2% lift)
println(result.ci_lower)     // lower bound of CI on difference</code></pre>
                    <p><strong>ABTestConfig fields:</strong> <code>alpha f64 = 0.05</code></p>
                </div>
            </section>

            <section id="sequential">
                <h2>Sequential Testing (SPRT)</h2>
                <p>The SPRT lets you check experiment results at any point without inflating the false positive rate. Call it repeatedly with cumulative totals as data comes in.</p>

                <div class="function-item">
                    <h3 id="sprt_test">
                        <code>fn sprt_test(successes_a int, n_a int, successes_b int, n_b int, cfg SPRTConfig) SPRTResult</code>
                    </h3>
                    <p>Stateless function — pass cumulative totals each time. Stop the experiment when <code>decision</code> is not <code>.continue_testing</code>.</p>
                    <pre><code>cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 }

// Check on day 3 (cumulative totals so far)
r := experiment.sprt_test(120, 2400, 145, 2400, cfg)
if r.decision == .reject_null {
    println('Significant! Stop the experiment.')
} else if r.decision == .accept_null {
    println('Futile. The effect is smaller than your MDE.')
} else {
    println('Keep collecting data.')
}</code></pre>
                </div>

                <div class="struct-item">
                    <h3><code>struct SPRTConfig</code></h3>
                    <p>Not a <code>@[params]</code> struct — you must construct it explicitly because <code>mde</code> has no sensible default.</p>
                    <table>
                        <tr><th>Field</th><th>Default</th><th>Description</th></tr>
                        <tr><td><code>alpha</code></td><td>0.05</td><td>Max acceptable false positive rate</td></tr>
                        <tr><td><code>beta</code></td><td>0.20</td><td>Max acceptable false negative rate (1 - power)</td></tr>
                        <tr><td><code>mde</code></td><td><em>required</em></td><td>Minimum detectable effect (absolute rate difference)</td></tr>
                    </table>
                </div>

                <div class="struct-item">
                    <h3><code>enum SPRTDecision</code></h3>
                    <table>
                        <tr><th>Value</th><th>Meaning</th></tr>
                        <tr><td><code>.continue_testing</code></td><td>Not enough evidence yet — keep running</td></tr>
                        <tr><td><code>.reject_null</code></td><td>Significant effect detected — safe to stop</td></tr>
                        <tr><td><code>.accept_null</code></td><td>Effect is below MDE — futile to continue</td></tr>
                    </table>
                </div>
            </section>

            <section id="bayesian">
                <h2>Bayesian A/B Test</h2>
                <p>Instead of a p-value, Bayesian testing gives you direct probability statements: "there is a 94% chance that B is better than A." Uses the Beta-Binomial conjugate model with configurable priors.</p>

                <div class="function-item">
                    <h3 id="bayesian_ab_test">
                        <code>fn bayesian_ab_test(successes_a int, n_a int, successes_b int, n_b int, cfg BayesianConfig) BayesianResult</code>
                    </h3>
                    <pre><code>result := experiment.bayesian_ab_test(30, 300, 48, 300)
println('P(B beats A): ${(result.prob_b_beats_a * 100):.1f}%')
println('Expected loss if we ship B: ${result.expected_loss_b:.4f}')
println('A credible interval: [${result.ci_lower_a:.3f}, ${result.ci_upper_a:.3f}]')
println('B credible interval: [${result.ci_lower_b:.3f}, ${result.ci_upper_b:.3f}]')</code></pre>
                    <p><strong>Interpreting expected loss:</strong> <code>expected_loss_b</code> is the average rate you "give up" by shipping B if A was actually better. A decision rule: ship B if <code>expected_loss_b &lt; threshold</code> (e.g. 0.001).</p>
                </div>

                <div class="struct-item">
                    <h3><code>struct BayesianConfig</code> (optional, <code>@[params]</code>)</h3>
                    <table>
                        <tr><th>Field</th><th>Default</th><th>Description</th></tr>
                        <tr><td><code>alpha_prior</code></td><td>1.0</td><td>Beta prior shape — set higher to encode prior knowledge of the success rate</td></tr>
                        <tr><td><code>beta_prior</code></td><td>1.0</td><td>Beta prior shape — Beta(1,1) is the uniform (non-informative) prior</td></tr>
                        <tr><td><code>n_samples</code></td><td>10000</td><td>Monte Carlo samples — increase for more precise estimates</td></tr>
                    </table>
                </div>
            </section>

            <section id="cuped">
                <h2>CUPED — Variance Reduction</h2>
                <div class="function-item">
                    <h3 id="cuped_test">
                        <code>fn cuped_test(y_ctrl []f64, y_treat []f64, pre_ctrl []f64, pre_treat []f64, cfg ABTestConfig) CUPEDResult</code>
                    </h3>
                    <p>CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance by adjusting outcomes using a correlated pre-experiment covariate. A strong correlation (e.g. last week's revenue predicts this week's revenue) can cut required sample size by 50% or more.</p>
                    <p>Pass the same pre-experiment metric as <code>pre_ctrl</code> / <code>pre_treat</code> (e.g. revenue in the week before the experiment). The function estimates the adjustment coefficient <code>theta</code> and runs the A/B test on adjusted values.</p>
                    <pre><code>result := experiment.cuped_test(y_control, y_treatment, pre_control, pre_treatment)
println('Variance reduced by: ${(result.variance_reduction * 100):.1f}%')
println('Adjusted p-value: ${result.adjusted_result.p_value:.4f}')</code></pre>
                </div>
            </section>

            <section id="causal">
                <h2>Causal Inference</h2>
                <p>Use these when treatment was <em>not</em> randomly assigned (observational data).</p>

                <div class="function-item">
                    <h3>Propensity Score Matching (PSM)</h3>
                    <p>Matches treated and control units on the probability of receiving treatment, controlling for selection bias.</p>
                    <pre><code>ps_model := experiment.estimate_propensity_scores(x_covariates, treatment)
matching := experiment.match_nearest_neighbor(ps_model)
balance  := experiment.check_balance(x_covariates, treatment, matching)
ate      := experiment.ate_matched(y_outcomes, treatment, matching)</code></pre>
                </div>

                <div class="function-item">
                    <h3>Difference-in-Differences (DiD)</h3>
                    <p>Estimates causal effects using pre/post data across treated and control groups, assuming parallel pre-period trends.</p>
                    <pre><code>// Simple 2x2 DiD
result := experiment.did_2x2(y_treat_pre, y_treat_post, y_ctrl_pre, y_ctrl_post)
println('DiD effect: ${result.did_effect:.4f}')

// Test whether parallel trends assumption holds
pt := experiment.test_parallel_trends(y_treated_pre, y_control_pre, time_pre)
println('Parallel trends hold: ${pt.parallel_trends_hold}')</code></pre>
                </div>
            </section>
        </main>
    </div>
    <script src="../js/script.js"></script>
</body>
</html>
```

- [ ] **Step 5.2: Verify the file renders correctly by checking for broken HTML**

Open `docs/modules/experiment.html` in a browser or run a basic syntax check.

- [ ] **Step 5.3: Commit**

```bash
git add docs/modules/experiment.html
git commit -m "docs: rewrite experiment module API reference"
```

---

## Task 6: Concepts Page — Experimentation Section

**Files:**
- Modify: `docs/concepts.html`

Add a new "Experimentation" section after the "Machine Learning Essentials" section (before the closing `</main>`).

- [ ] **Step 6.1: Insert the Experimentation section into `docs/concepts.html`**

Find the closing `</main>` tag and insert before it:

```html
            <section class="concept" id="experimentation">
                <h2>Experimentation</h2>

                <h3>Statistical Power and Sample Size</h3>
                <p>Before running an experiment, you must decide how many observations you need. Four numbers determine this:</p>
                <ul>
                    <li><strong>Alpha (α):</strong> The false positive rate — how often you'll declare a winner when there's no real effect. Typically 0.05 (5%).</li>
                    <li><strong>Power (1 - β):</strong> The probability of detecting a real effect when one exists. Typically 0.80 (80%). Setting power too low means your experiment may end with no result even when the treatment works.</li>
                    <li><strong>Minimum Detectable Effect (MDE):</strong> The smallest improvement worth detecting. A tighter MDE (detecting smaller changes) requires more data. Be honest about what effect size would change a business decision — don't chase tiny effects that aren't actionable.</li>
                    <li><strong>Baseline variance:</strong> Higher variance in your metric means more noise, requiring more data to see signal. Use historical data to estimate this.</li>
                </ul>
                <p>The formula for continuous metrics: <code>n = 2 × ((z_α/2 + z_β) × σ / Δ)²</code> per group, where σ is the standard deviation and Δ is the MDE.</p>

                <h3>Frequentist vs. Bayesian</h3>
                <p>Both are valid frameworks with different outputs:</p>
                <ul>
                    <li><strong>Frequentist (p-values):</strong> Answers "if there were no effect, how unlikely is this data?" You reject the null hypothesis when p &lt; α. The result is binary: significant or not. Does not give the probability that B is better.</li>
                    <li><strong>Bayesian (posteriors):</strong> Answers "given this data, what is the probability that B beats A?" More intuitive for business decisions. Requires specifying a prior — use Beta(1,1) (uniform) when you have no prior knowledge.</li>
                </ul>
                <p>Bayesian is preferable when you need to communicate results to non-statisticians ("94% chance B is better"), when you want to incorporate prior knowledge, or when you need to make a decision before collecting enough data for a frequentist test.</p>

                <h3>The Peeking Problem</h3>
                <p>A common mistake: checking an experiment's p-value every day and stopping when it first crosses 0.05. This <em>inflates the false positive rate dramatically</em> — you may declare significance by chance, especially early in an experiment.</p>
                <p>Two solutions:</p>
                <ul>
                    <li><strong>Sequential testing (SPRT):</strong> Uses Wald's likelihood ratio test with boundaries that account for repeated looks. You can check at any time; the false positive rate stays at α. Use <code>sprt_test</code>.</li>
                    <li><strong>Pre-commit to a fixed end date:</strong> Calculate your sample size, run the experiment until you have it, then do one final test. Never look at results before the end.</li>
                </ul>

                <h3>Effect Sizes</h3>
                <p>A p-value tells you whether an effect exists; effect size tells you how big it is. Always report both.</p>
                <ul>
                    <li><strong>Cohen's d:</strong> For continuous metrics. d = (mean_B - mean_A) / pooled_std. Small: 0.2, Medium: 0.5, Large: 0.8.</li>
                    <li><strong>Absolute lift:</strong> For proportions. "The signup rate increased by 1.2 percentage points (from 5.0% to 6.2%)." Always prefer absolute over relative for communication.</li>
                    <li><strong>Relative lift:</strong> Percentage change relative to control. "The signup rate increased 24% relative to control." Useful for comparing across metrics with different baselines, but can be misleading for small baselines.</li>
                </ul>

                <h3>Variance Reduction with CUPED</h3>
                <p>CUPED (Controlled-experiment Using Pre-Experiment Data) exploits the correlation between a user's pre-experiment behavior and their in-experiment behavior. If last week's revenue predicts this week's revenue, you can "remove" that predictable component from both groups, reducing noise without introducing bias. A correlation of ρ = 0.7 reduces required sample size by approximately 1 - ρ² = 51%.</p>
            </section>
```

- [ ] **Step 6.2: Commit**

```bash
git add docs/concepts.html
git commit -m "docs: add Experimentation concepts section"
```

---

## Task 7: Examples Page — Experimentation Workflows

**Files:**
- Modify: `docs/examples.html`

Add a new "Experimentation Workflows" section to `docs/examples.html`. Find the `<section class="example-category" id="experimentation">` section (it already exists) and replace its content with the full workflows below.

- [ ] **Step 7.1: Locate the experimentation section in examples.html**

Read `docs/examples.html` and find where the `id="experimentation"` section begins. Replace the section's inner content (or insert after the existing content) with the four workflows below. Wrap in `<section class="example-category" id="experimentation">`.

**Workflow 1: Conversion Rate Test**

```html
<h2 id="experimentation">2. Experimentation Workflows</h2>
<p class="lead">End-to-end examples for designing, running, and interpreting A/B experiments.</p>

<h3>Workflow 1: Conversion Rate Test</h3>
<div class="business-context">
<strong>Business Context:</strong> Your landing page has a 5% signup rate. You want to test a redesigned hero section and need to detect at least a 1 percentage-point improvement (5% → 6%). You have 80% power budget and use a 5% significance level.
</div>

<h4>Step 1: Calculate sample size before starting</h4>
<pre><code>import vstats.experiment

fn main() {
    // How many users do we need per group?
    // baseline_rate=0.05, mde=+0.01 (5%→6%), alpha=0.05, power=0.80
    plan := experiment.sample_size_proportions(0.05, 0.01, 0.05, 0.80)
    println('Need ${plan.n_per_group} users per group (${plan.total_n} total)')
    println('Implied effect size (h): ${plan.effect_size:.3f}')
    // Output: Need ~8100 users per group (16200 total)
}</code></pre>

<h4>Step 2: Run the experiment, then analyze</h4>
<pre><code>    // After collecting data:
    // Control (original): 405 signups from 8100 visitors
    // Treatment (new design): 520 signups from 8100 visitors
    result := experiment.proportion_test(405, 8100, 520, 8100)

    println('Control rate:   ${(result.rate_a * 100):.2f}%')   // 5.00%
    println('Treatment rate: ${(result.rate_b * 100):.2f}%')   // 6.42%
    println('Absolute lift:  +${(result.diff * 100):.2f}pp')   // +1.42pp
    println('Relative lift:  +${(result.relative_lift * 100):.1f}%') // +28.4%
    println('p-value:        ${result.p_value:.4f}')
    println('95% CI on diff: [${(result.ci_lower*100):.2f}pp, ${(result.ci_upper*100):.2f}pp]')
    println('Significant:    ${result.significant}')
    // Interpretation: The new design increased signups by 1.42pp (28% relative lift).
    // This exceeds our MDE of 1pp and is statistically significant. Ship it.</code></pre>
```

**Workflow 2: Continuous Metric Test**

```html
<h3>Workflow 2: Continuous Metric Test (Revenue)</h3>
<div class="business-context">
<strong>Business Context:</strong> Average revenue per user is $45 with a standard deviation of $20. You want to detect a $3 increase (6.7% lift). Calculate sample size, run the experiment, then analyze with a t-test.
</div>

<pre><code>import vstats.experiment

fn main() {
    // Step 1: Sample size
    // baseline_mean=45.0, std=20.0, mde=$3, alpha=0.05, power=0.80
    plan := experiment.sample_size_means(45.0, 20.0, 3.0, 0.05, 0.80)
    println('Need ${plan.n_per_group} users per group')
    println("Cohen's d: ${plan.effect_size:.3f}")  // 0.15 (small effect)

    // Step 2: Analyze after collecting data
    // (In practice, load your data from a file or database)
    control   := [43.2, 45.1, 46.8, 44.3, 45.7, 43.9, 47.2, 44.8, 45.5, 46.1]
    treatment := [47.8, 49.2, 46.5, 50.1, 48.3, 47.9, 49.5, 48.7, 50.2, 47.4]

    result := experiment.abtest(control, treatment)
    println('Control mean:   \$${result.control_mean:.2f}')
    println('Treatment mean: \$${result.treatment_mean:.2f}')
    println('Lift:           \$${result.treatment_mean - result.control_mean:.2f} (${(result.relative_lift * 100):.1f}%)')
    println("Cohen's d:      ${result.effect_size:.3f}")
    println('p-value:        ${result.p_value:.4f}')
    println('95% CI: [\$${result.ci_lower:.2f}, \$${result.ci_upper:.2f}]')
    println('Significant:    ${result.significant}')
}</code></pre>
```

**Workflow 3: Sequential Test (Early Stopping)**

```html
<h3>Workflow 3: Sequential Testing (Safe Early Stopping)</h3>
<div class="business-context">
<strong>Business Context:</strong> You're running a high-traffic experiment where results come in fast. You want to check daily and stop early if a clear winner emerges — without inflating your false positive rate.
</div>

<pre><code>import vstats.experiment

fn main() {
    cfg := experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.01 }

    // Simulate daily checks with cumulative data
    // Day 1: 500 users per group
    r1 := experiment.sprt_test(25, 500, 28, 500, cfg)
    println('Day 1: LLR=${r1.log_likelihood_ratio:.3f}, decision=${r1.decision}')

    // Day 5: 2500 users per group
    r5 := experiment.sprt_test(125, 2500, 160, 2500, cfg)
    println('Day 5: LLR=${r5.log_likelihood_ratio:.3f}, decision=${r5.decision}')

    // Day 10: 5000 users per group — effect is clear
    r10 := experiment.sprt_test(250, 5000, 340, 5000, cfg)
    println('Day 10: LLR=${r10.log_likelihood_ratio:.3f}, decision=${r10.decision}')

    match r10.decision {
        .reject_null      { println('Significant — ship the treatment!') }
        .accept_null      { println('Futile — effect is smaller than your MDE.') }
        .continue_testing { println('Keep running.') }
    }
    // Note: boundaries are log((1-β)/α) and log(β/(1-α))
    println('Upper boundary: ${r10.upper_boundary:.3f}')  // ~2.77
    println('Lower boundary: ${r10.lower_boundary:.3f}')  // ~-1.56
}</code></pre>
```

**Workflow 4: Bayesian Test**

```html
<h3>Workflow 4: Bayesian A/B Test</h3>
<div class="business-context">
<strong>Business Context:</strong> You have a smaller experiment (300 users per group) and want to make a decision even before reaching classical significance. The Bayesian approach gives you a direct probability that variant B is better.
</div>

<pre><code>import vstats.experiment

fn main() {
    // 30 conversions / 300 users vs 48 conversions / 300 users
    result := experiment.bayesian_ab_test(30, 300, 48, 300)

    println('A conversion rate (posterior mean): ${(result.posterior_mean_a * 100):.1f}%')
    println('B conversion rate (posterior mean): ${(result.posterior_mean_b * 100):.1f}%')
    println('')
    println('P(B beats A):   ${(result.prob_b_beats_a * 100):.1f}%')
    println('Expected loss (ship B): ${result.expected_loss_b:.5f}')
    println('Expected loss (ship A): ${result.expected_loss_a:.5f}')
    println('')
    println('95% Credible Interval for A: [${result.ci_lower_a:.3f}, ${result.ci_upper_a:.3f}]')
    println('95% Credible Interval for B: [${result.ci_lower_b:.3f}, ${result.ci_upper_b:.3f}]')

    // Decision rule: ship B if expected_loss_b < threshold
    threshold := 0.001
    if result.expected_loss_b < threshold {
        println('\nDecision: Ship B (expected loss ${result.expected_loss_b:.5f} < ${threshold})')
    } else if result.expected_loss_a < threshold {
        println('\nDecision: Keep A (expected loss of shipping B is too high)')
    } else {
        println('\nDecision: Collect more data')
    }

    // With an informative prior (you know the baseline is around 10%):
    informed_cfg := experiment.BayesianConfig{ alpha_prior: 10.0, beta_prior: 90.0 }
    informed := experiment.bayesian_ab_test(30, 300, 48, 300, informed_cfg)
    println('\nWith informative prior: P(B beats A) = ${(informed.prob_b_beats_a * 100):.1f}%')
}</code></pre>
```

- [ ] **Step 7.2: Read the current examples.html and insert the workflows into the experimentation section**

First `Read docs/examples.html` to locate the `id="experimentation"` section, then use `Edit` to replace its content with the four workflows above.

- [ ] **Step 7.3: Commit**

```bash
git add docs/examples.html
git commit -m "docs: add four experimentation workflow examples"
```

---

## Final Verification

- [ ] **Update MODULES.md experiment section**

Open `MODULES.md` and update the `experiment/` module entry to list all 7 files and their new functions (`sample_size_proportions`, `sample_size_means`, `proportion_test`, `sprt_test`, `bayesian_ab_test`, plus existing functions).

```bash
git add MODULES.md
git commit -m "docs: update MODULES.md with new experiment functions"
```

- [ ] **Run the full test suite one last time**

```bash
make test
```
Expected: all tests pass, including the 4 new test files.

- [ ] **Verify test count increased**

```bash
v -stats test tests/ 2>&1 | grep -E "Tests|PASS|FAIL"
```

- [ ] **Final commit if any cleanup needed**

```bash
git add -u
git commit -m "chore: finalize experiment utilities implementation"
```
