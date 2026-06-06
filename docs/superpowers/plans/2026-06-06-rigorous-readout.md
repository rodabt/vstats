# Rigorous Readout Functions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 12 standalone functions covering the full experiment readout workflow — metric pre-processing, data quality checks, effect estimation, multiple testing correction, result interpretation, and a skill update.

**Architecture:** Functions are added as standalone utilities to existing and new files in `stats/` and `experiment/`. No orchestrator; each function is independently callable. New files (`stats/multiple_testing.v`, `stats/inference.v`, `experiment/ols_helpers.v`, `experiment/readout.v`) follow existing module conventions. All test files live in `tests/`.

**Tech Stack:** V language, vstats module. Tests use V's built-in test runner (`v test`). `import math`, `import rand` are standard library. `import vstats.stats`, `import vstats.prob`, `import vstats.ml`, `import vstats.experiment`, `import vstats.hypothesis` follow established patterns.

---

## File Map

| File | Status | What changes |
|------|--------|-------------|
| `stats/descriptive.v` | modify | add `winsorize`, `rtm_correction` |
| `stats/multiple_testing.v` | **create** | `BHResult`, `BonferroniResult`, `bh_correction`, `bonferroni_correction` |
| `stats/inference.v` | **create** | `DeltaMethodResult`, `DeltaMethodConfig`, `BootstrapResult`, `delta_method_ratio`, `bootstrap_test`, private `inv_normal_cdf` |
| `experiment/ols_helpers.v` | **create** | extract `ols_se` from `did.v` so both `did.v` and `abtest.v` can use it |
| `experiment/did.v` | modify | remove the now-duplicate `ols_se` definition |
| `experiment/sample_size.v` | modify | add `icc`, `design_effect` |
| `experiment/abtest.v` | modify | add `ANCOVAConfig`, `ANCOVAResult`, `NullVerdictKind`, `NullVerdictResult`, `ITTPPResult`, `ancova`, `null_verdict`, `itt_and_pp` |
| `experiment/readout.v` | **create** | `SRMResult`, `SimpsonsResult`, `SubgroupResult`, `HTEResult`, `srm_test`, `simpsons_check`, `hte_subgroup` |
| `tests/descriptive_test.v` | modify | add tests for `winsorize`, `rtm_correction` |
| `tests/multiple_testing_test.v` | **create** | tests for `bh_correction`, `bonferroni_correction` |
| `tests/inference_test.v` | **create** | tests for `delta_method_ratio`, `bootstrap_test` |
| `tests/experiment_test.v` | modify | add tests for `icc`, `design_effect`, `ancova`, `null_verdict`, `itt_and_pp` |
| `tests/readout_test.v` | **create** | tests for `srm_test`, `simpsons_check`, `hte_subgroup` |
| `/home/rabt/.claude/skills/vstats/references/modules.md` | modify | fix stale entries + add all new function signatures |
| `/home/rabt/.claude/skills/vstats/SKILL.md` | modify | fix `ABTestConfig` snippet, add readout workflow section |

---

## Task 1: `stats.winsorize` and `stats.rtm_correction`

**Files:**
- Modify: `stats/descriptive.v`
- Modify: `tests/descriptive_test.v`

Note: `quantile(x, p)` uses `p_index = int(p * x.len)` — integer truncation, not rounding. Account for this in test values.

- [ ] **Step 1: Write failing tests** — add to `tests/descriptive_test.v`:

```v
fn test__winsorize_clamps_extremes() {
	x := [1.0, 2.0, 3.0, 4.0, 100.0]
	// quantile uses int(p * len): q_low=0.2 → int(0.2*5)=1 → sorted[1]=2.0
	//                             q_high=0.6 → int(0.6*5)=3 → sorted[3]=4.0
	result := stats.winsorize(x, 0.2, 0.6)
	assert result[0] == 2.0   // 1.0 clamped up to lo=2.0
	assert result[1] == 2.0   // 2.0 == lo, unchanged
	assert result[2] == 3.0   // 3.0 in range, unchanged
	assert result[3] == 4.0   // 4.0 == hi, unchanged
	assert result[4] == 4.0   // 100.0 clamped down to hi=4.0
}

fn test__winsorize_no_clamp_when_full_range() {
	x := [1.0, 2.0, 3.0]
	result := stats.winsorize(x, 0.0, 1.0)
	// q_low=0.0 → int(0*3)=0 → sorted[0]=1.0
	// q_high=1.0 → int(1.0*3)=3 → out of bounds; clamp to len-1 → sorted[2]=3.0
	// Actually int(1.0*3)=3 which is x.len — need to handle this edge case
	// The implementation should clamp p_index to len-1
	assert result[0] == 1.0
	assert result[2] == 3.0
}

fn test__rtm_correction_perfect_correlation() {
	// followup = baseline (perfect correlation, slope=1, intercept=0)
	// selected units above threshold=7: baseline=[8,9,10], mean=9.0
	// overall followup mean = (1+2+...+10)/10 = 5.5
	// RTM shift = 9.0 - 5.5 = 3.5
	baseline := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	followup := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	result := stats.rtm_correction(baseline, followup, 7.0)
	assert math.abs(result - 3.5) < 1e-6
}

fn test__rtm_correction_no_selection() {
	baseline := [1.0, 2.0, 3.0]
	followup := [2.0, 4.0, 6.0]
	// threshold above all values → no selected units → returns 0.0
	result := stats.rtm_correction(baseline, followup, 100.0)
	assert result == 0.0
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/descriptive_test.v
```
Expected: compile error — `stats.winsorize` and `stats.rtm_correction` undefined.

- [ ] **Step 3: Implement** — add to end of `stats/descriptive.v`:

```v
pub fn winsorize(x []f64, q_low f64, q_high f64) []f64 {
	assert x.len > 0, 'x must not be empty'
	assert q_low >= 0.0 && q_low < q_high, 'q_low must be < q_high'
	assert q_high <= 1.0, 'q_high must be <= 1.0'
	lo := quantile(x, q_low)
	hi_idx := int(q_high * f64(x.len))
	mut x_sorted := x.clone()
	x_sorted.sort()
	hi := x_sorted[if hi_idx >= x_sorted.len { x_sorted.len - 1 } else { hi_idx }]
	mut result := []f64{len: x.len}
	for i, v in x {
		result[i] = if v < lo { lo } else if v > hi { hi } else { v }
	}
	return result
}

pub fn rtm_correction(baseline []f64, followup []f64, selection_threshold f64) f64 {
	assert baseline.len == followup.len, 'baseline and followup must have same length'
	assert baseline.len >= 2, 'need at least 2 observations'
	b_mean := mean(baseline)
	f_mean := mean(followup)
	cov_bf := covariance(baseline, followup)
	var_b := variance(baseline)
	slope := if var_b > 0 { cov_bf / var_b } else { 0.0 }
	intercept := f_mean - slope * b_mean
	mut sel_sum := 0.0
	mut sel_n := 0
	for v in baseline {
		if v > selection_threshold {
			sel_sum += intercept + slope * v
			sel_n++
		}
	}
	if sel_n == 0 {
		return 0.0
	}
	return sel_sum / f64(sel_n) - f_mean
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/descriptive_test.v
```
Expected: all tests PASS.

- [ ] **Step 5: Full test suite passes**

```bash
make test
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add stats/descriptive.v tests/descriptive_test.v
git commit -m "feat(stats): add winsorize and rtm_correction"
```

---

## Task 2: Multiple Testing Corrections

**Files:**
- Create: `stats/multiple_testing.v`
- Create: `tests/multiple_testing_test.v`

- [ ] **Step 1: Write failing tests** — create `tests/multiple_testing_test.v`:

```v
import vstats.stats
import math

fn test__bh_correction_rejects_three() {
	// BH with known result: p=[0.001,0.01,0.02,0.2,0.5], alpha=0.05
	// BH thresholds (rank*alpha/n): 0.01,0.02,0.03,0.04,0.05
	// p[0]=0.001<=0.01 ✓, p[1]=0.01<=0.02 ✓, p[2]=0.02<=0.03 ✓, rest: NO
	p_values := [0.001, 0.01, 0.02, 0.2, 0.5]
	result := stats.bh_correction(p_values, 0.05)
	assert result.n_rejected == 3
	assert result.reject[0] == true
	assert result.reject[1] == true
	assert result.reject[2] == true
	assert result.reject[3] == false
	assert result.reject[4] == false
	// Adjusted p-values are in [0,1]
	for adj in result.adjusted {
		assert adj >= 0.0 && adj <= 1.0
	}
}

fn test__bh_correction_rejects_none_when_all_large() {
	p_values := [0.3, 0.4, 0.5, 0.6, 0.7]
	result := stats.bh_correction(p_values, 0.05)
	assert result.n_rejected == 0
}

fn test__bonferroni_correction_rejects_one() {
	// With n=5, alpha=0.05, Bonferroni threshold = 0.01
	// p=0.005 < 0.01 → reject; p=0.02 → adj=0.1 > 0.05 → keep
	p_values := [0.005, 0.02, 0.04, 0.1, 0.5]
	result := stats.bonferroni_correction(p_values, 0.05)
	assert result.n_rejected == 1
	assert result.reject[0] == true
	assert result.reject[1] == false
	// adjusted[0] = 0.005 * 5 = 0.025
	assert math.abs(result.adjusted[0] - 0.025) < 1e-9
	// adjusted[2] = min(0.04*5, 1.0) = 0.2
	assert math.abs(result.adjusted[2] - 0.2) < 1e-9
}

fn test__bonferroni_adjusted_capped_at_one() {
	p_values := [0.5, 0.5, 0.5, 0.5, 0.5]
	result := stats.bonferroni_correction(p_values, 0.05)
	for adj in result.adjusted {
		assert adj == 1.0
	}
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/multiple_testing_test.v
```
Expected: compile error — module not found.

- [ ] **Step 3: Implement** — create `stats/multiple_testing.v`:

```v
module stats

pub struct BHResult {
pub:
	adjusted  []f64
	reject    []bool
	n_rejected int
}

pub struct BonferroniResult {
pub:
	adjusted  []f64
	reject    []bool
	n_rejected int
}

pub fn bh_correction(p_values []f64, alpha f64) BHResult {
	n := p_values.len
	assert n > 0, 'p_values must not be empty'
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0, 1)'

	// Sort indices by p-value (insertion sort — no external deps)
	mut order := []int{len: n, init: index}
	for i := 1; i < n; i++ {
		key := order[i]
		mut j := i - 1
		for j >= 0 && p_values[order[j]] > p_values[key] {
			order[j + 1] = order[j]
			j--
		}
		order[j + 1] = key
	}

	// Compute adjusted p-values: p_adj[rank] = p[rank] * n / (rank+1)
	mut adj := []f64{len: n}
	for rank, orig_idx in order {
		adj[orig_idx] = p_values[orig_idx] * f64(n) / f64(rank + 1)
	}

	// Enforce monotonicity (step-down): scan sorted order right to left
	mut min_so_far := 1.0
	for r := n - 1; r >= 0; r-- {
		orig_idx := order[r]
		if adj[orig_idx] < min_so_far {
			min_so_far = adj[orig_idx]
		} else {
			adj[orig_idx] = min_so_far
		}
	}

	// Clamp to [0, 1] and build rejection flags
	mut reject := []bool{len: n}
	mut n_rejected := 0
	for i in 0 .. n {
		if adj[i] > 1.0 {
			adj[i] = 1.0
		}
		reject[i] = adj[i] <= alpha
		if reject[i] {
			n_rejected++
		}
	}
	return BHResult{ adjusted: adj, reject: reject, n_rejected: n_rejected }
}

pub fn bonferroni_correction(p_values []f64, alpha f64) BonferroniResult {
	n := p_values.len
	assert n > 0, 'p_values must not be empty'
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0, 1)'
	mut adj := []f64{len: n}
	mut reject := []bool{len: n}
	mut n_rejected := 0
	for i, p in p_values {
		a := p * f64(n)
		adj[i] = if a > 1.0 { 1.0 } else { a }
		reject[i] = adj[i] <= alpha
		if reject[i] {
			n_rejected++
		}
	}
	return BonferroniResult{ adjusted: adj, reject: reject, n_rejected: n_rejected }
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/multiple_testing_test.v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Full suite**

```bash
make test
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add stats/multiple_testing.v tests/multiple_testing_test.v
git commit -m "feat(stats): add bh_correction and bonferroni_correction"
```

---

## Task 3: `stats.delta_method_ratio` and `stats.bootstrap_test`

**Files:**
- Create: `stats/inference.v`
- Create: `tests/inference_test.v`

Note: `stats/inference.v` is in the `stats` module. It may call `mean`, `variance`, `normal_cdf` (defined in `descriptive.v`) without any import — same module. Needs `import math` and `import rand`.

- [ ] **Step 1: Write failing tests** — create `tests/inference_test.v`:

```v
import vstats.stats
import math
import rand

fn test__delta_method_ratio_detects_effect() {
	// control: revenue/sessions = 10/2 = 5.0 per user (mean)
	// treatment: revenue/sessions = 14/2 = 7.0 per user (mean)
	// true effect = 2.0
	a_ctrl := [10.0, 12.0, 8.0, 10.0]
	b_ctrl := [2.0, 2.0, 2.0, 2.0]
	a_trt  := [14.0, 16.0, 12.0, 14.0]
	b_trt  := [2.0, 2.0, 2.0, 2.0]
	mut a := []f64{}
	mut b := []f64{}
	a << a_ctrl
	a << a_trt
	b << b_ctrl
	b << b_trt
	treatment := [0, 0, 0, 0, 1, 1, 1, 1]
	result := stats.delta_method_ratio(a, b, treatment, stats.DeltaMethodConfig{})
	assert math.abs(result.ratio_ctrl - 5.0) < 1e-6
	assert math.abs(result.ratio_trt - 7.0) < 1e-6
	assert math.abs(result.effect - 2.0) < 1e-6
	assert result.p_value < 0.05
	assert result.ci_lower > 0.0   // CI entirely positive
	assert result.ci_upper > result.effect
}

fn test__delta_method_ratio_no_effect() {
	// same ratio in both groups
	a := [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
	b := [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
	treatment := [0, 0, 0, 1, 1, 1]
	result := stats.delta_method_ratio(a, b, treatment, stats.DeltaMethodConfig{})
	assert math.abs(result.effect) < 1e-9
	assert result.p_value > 0.05
}

fn test__bootstrap_test_detects_large_effect() {
	rand.seed([u32(42), u32(0)])
	ctrl := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	trt  := [3.0, 3.1, 2.9, 3.0, 3.05, 2.95, 3.02, 2.98, 3.01, 2.99]
	result := stats.bootstrap_test(ctrl, trt, 2000)
	assert result.p_value < 0.05
	assert math.abs(result.observed_diff - 2.0) < 0.01
}

fn test__bootstrap_test_no_effect() {
	rand.seed([u32(99), u32(0)])
	// Same distribution — p-value should not be consistently small
	ctrl := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	trt  := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	result := stats.bootstrap_test(ctrl, trt, 2000)
	// observed_diff = 0 exactly → p_value should be 1.0 (no permutation is more extreme)
	assert result.p_value > 0.5
	assert math.abs(result.observed_diff) < 1e-9
}

fn test__bootstrap_test_ci_contains_truth() {
	rand.seed([u32(7), u32(0)])
	ctrl := [0.0, 0.1, -0.1, 0.0, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01]
	trt  := [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99]
	result := stats.bootstrap_test(ctrl, trt, 2000)
	// True effect ≈ 1.0; CI should contain it
	assert result.ci_lower < 1.0
	assert result.ci_upper > 1.0
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/inference_test.v
```
Expected: compile error.

- [ ] **Step 3: Implement** — create `stats/inference.v`:

```v
module stats

import math
import rand

@[params]
pub struct DeltaMethodConfig {
pub:
	alpha f64 = 0.05
}

pub struct DeltaMethodResult {
pub:
	ratio_ctrl  f64
	ratio_trt   f64
	effect      f64
	se          f64
	t_statistic f64
	p_value     f64
	ci_lower    f64
	ci_upper    f64
}

pub struct BootstrapResult {
pub:
	p_value       f64
	observed_diff f64
	ci_lower      f64
	ci_upper      f64
	n_resamples   int
}

// Rational approximation for inverse normal CDF (Abramowitz & Stegun 26.2.17)
// Accuracy ~0.0005. Used internally since stats cannot import prob.
fn inv_normal_cdf(p f64) f64 {
	if p <= 0.0 { return -8.0 }
	if p >= 1.0 { return 8.0 }
	mut q := p
	mut sgn := 1.0
	if p < 0.5 {
		q = 1.0 - p
		sgn = -1.0
	}
	t := math.sqrt(-2.0 * math.log(1.0 - q))
	x := t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
		(1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
	return sgn * x
}

pub fn delta_method_ratio(a []f64, b []f64, treatment []int, cfg DeltaMethodConfig) DeltaMethodResult {
	assert a.len == b.len && a.len == treatment.len, 'a, b, treatment must have same length'
	assert a.len >= 4, 'need at least 4 observations'

	n := a.len
	mut sum_a := 0.0
	mut sum_b := 0.0
	for i in 0 .. n {
		sum_a += a[i]
		sum_b += b[i]
	}
	r := if sum_b != 0.0 { sum_a / sum_b } else { 0.0 }

	// Linearize: z_i = a_i - R * b_i
	mut z_ctrl := []f64{}
	mut z_trt  := []f64{}
	mut a_ctrl := []f64{}
	mut a_trt  := []f64{}
	mut b_ctrl := []f64{}
	mut b_trt  := []f64{}
	for i in 0 .. n {
		z := a[i] - r * b[i]
		if treatment[i] == 0 {
			z_ctrl << z
			a_ctrl << a[i]
			b_ctrl << b[i]
		} else {
			z_trt << z
			a_trt << a[i]
			b_trt << b[i]
		}
	}
	assert z_ctrl.len >= 2 && z_trt.len >= 2, 'each group needs at least 2 observations'

	mean_b := sum_b / f64(n)
	v_c := variance(z_ctrl)
	v_t := variance(z_trt)
	n_c := f64(z_ctrl.len)
	n_t := f64(z_trt.len)
	se_z := math.sqrt(v_c / n_c + v_t / n_t)
	se_ratio := if mean_b != 0.0 { se_z / math.abs(mean_b) } else { 0.0 }

	ratio_ctrl := if mean(b_ctrl) != 0.0 { mean(a_ctrl) / mean(b_ctrl) } else { 0.0 }
	ratio_trt  := if mean(b_trt)  != 0.0 { mean(a_trt)  / mean(b_trt)  } else { 0.0 }
	effect := ratio_trt - ratio_ctrl
	t_stat := if se_ratio > 0.0 { effect / se_ratio } else { 0.0 }
	p_val := 2.0 * normal_cdf(-math.abs(t_stat), 0.0, 1.0)
	z_crit := inv_normal_cdf(1.0 - cfg.alpha / 2.0)

	return DeltaMethodResult{
		ratio_ctrl:  ratio_ctrl
		ratio_trt:   ratio_trt
		effect:      effect
		se:          se_ratio
		t_statistic: t_stat
		p_value:     p_val
		ci_lower:    effect - z_crit * se_ratio
		ci_upper:    effect + z_crit * se_ratio
	}
}

pub fn bootstrap_test(ctrl []f64, trt []f64, n_resamples int) BootstrapResult {
	assert ctrl.len >= 2 && trt.len >= 2, 'each group needs at least 2 observations'
	assert n_resamples > 0, 'n_resamples must be positive'

	n_c := ctrl.len
	n_t := trt.len
	observed_diff := mean(trt) - mean(ctrl)

	// Permutation test: pool, shuffle, split, count extreme diffs
	mut pooled := []f64{}
	pooled << ctrl
	pooled << trt
	n_pool := pooled.len

	mut count_extreme := 0
	for _ in 0 .. n_resamples {
		// Fisher-Yates shuffle
		mut perm := pooled.clone()
		for i := n_pool - 1; i > 0; i-- {
			j := rand.intn(i + 1) or { 0 }
			tmp := perm[i]
			perm[i] = perm[j]
			perm[j] = tmp
		}
		mut s_ctrl := 0.0
		mut s_trt  := 0.0
		for k in 0 .. n_c { s_ctrl += perm[k] }
		for k in n_c .. n_pool { s_trt += perm[k] }
		perm_diff := s_trt / f64(n_t) - s_ctrl / f64(n_c)
		if math.abs(perm_diff) >= math.abs(observed_diff) {
			count_extreme++
		}
	}
	p_val := f64(count_extreme) / f64(n_resamples)

	// Percentile bootstrap CI (resample with replacement)
	mut boot_diffs := []f64{len: n_resamples}
	for i in 0 .. n_resamples {
		mut bc_sum := 0.0
		mut bt_sum := 0.0
		for _ in 0 .. n_c {
			bc_sum += ctrl[rand.intn(n_c) or { 0 }]
		}
		for _ in 0 .. n_t {
			bt_sum += trt[rand.intn(n_t) or { 0 }]
		}
		boot_diffs[i] = bt_sum / f64(n_t) - bc_sum / f64(n_c)
	}
	boot_diffs.sort()
	ci_lo := quantile(boot_diffs, 0.025)
	ci_hi := quantile(boot_diffs, 0.975)

	return BootstrapResult{
		p_value:       p_val
		observed_diff: observed_diff
		ci_lower:      ci_lo
		ci_upper:      ci_hi
		n_resamples:   n_resamples
	}
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/inference_test.v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Full suite**

```bash
make test
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add stats/inference.v tests/inference_test.v
git commit -m "feat(stats): add delta_method_ratio and bootstrap_test"
```

---

## Task 4: Extract `ols_se` to shared helper

**Files:**
- Create: `experiment/ols_helpers.v`
- Modify: `experiment/did.v` (remove duplicate)

The `ols_se` function is currently private in `did.v`. Moving it to `ols_helpers.v` (same module) makes it available to `abtest.v` without any import.

- [ ] **Step 1: Run current tests as baseline**

```bash
make test
```
Expected: all PASS. Note this before refactoring.

- [ ] **Step 2: Create `experiment/ols_helpers.v`** with the function extracted verbatim from `did.v`:

```v
module experiment

import linalg
import math

// ols_se computes OLS standard errors for each coefficient.
// x_mat is the feature matrix WITHOUT an intercept column.
// coefficients must be [intercept, coef_0, coef_1, ...] (intercept first).
// Returns se[0]=SE(intercept), se[1]=SE(coef_0), etc.
fn ols_se(x_mat [][]f64, y []f64, coefficients []f64) []f64 {
	n := x_mat.len
	n_params := coefficients.len
	if n <= n_params {
		return []f64{len: n_params, init: 0.0}
	}
	mut x_aug := [][]f64{len: n}
	for i in 0 .. n {
		x_aug[i] = []f64{len: n_params}
		x_aug[i][0] = 1.0
		for j in 1 .. n_params {
			if j - 1 < x_mat[i].len {
				x_aug[i][j] = x_mat[i][j - 1]
			}
		}
	}
	mut ssr := 0.0
	for i in 0 .. n {
		mut y_pred := 0.0
		for j in 0 .. n_params {
			y_pred += coefficients[j] * x_aug[i][j]
		}
		resid := y[i] - y_pred
		ssr += resid * resid
	}
	s2 := ssr / f64(n - n_params)
	xt := linalg.transpose(x_aug)
	xtx := linalg.matmul(xt, x_aug)
	xtx_inv := matrix_inverse(xtx)
	mut se := []f64{len: n_params}
	for j in 0 .. n_params {
		v_j := s2 * xtx_inv[j][j]
		se[j] = if v_j > 0 { math.sqrt(v_j) } else { 0.0 }
	}
	return se
}
```

Note: `matrix_inverse` is also a private function in `did.v`. It must also be moved to `ols_helpers.v`, or kept in `did.v` with `ols_se` staying there and a wrapper exposed. Simpler: move **both** `matrix_inverse` and `ols_se` to `ols_helpers.v`, then remove them from `did.v`.

- [ ] **Step 3: Move `matrix_inverse` too** — `ols_helpers.v` should contain both:

Add `matrix_inverse` above `ols_se` in `ols_helpers.v`:

```v
fn matrix_inverse(m [][]f64) [][]f64 {
	n_dim := m.len
	mut result := [][]f64{len: n_dim}
	for i in 0 .. n_dim {
		result[i] = []f64{len: n_dim}
	}
	for j in 0 .. n_dim {
		mut e := []f64{len: n_dim, init: 0.0}
		e[j] = 1.0
		col := linalg.gaussian_elimination(m, e)
		for i in 0 .. n_dim {
			result[i][j] = col[i]
		}
	}
	return result
}
```

- [ ] **Step 4: Remove from `did.v`** — delete the `fn matrix_inverse` and `fn ols_se` definitions from `experiment/did.v`. Do not change any of the callers in `did.v` — they will now resolve from `ols_helpers.v` (same module).

- [ ] **Step 5: Verify**

```bash
make test
```
Expected: all tests still PASS. If there are compile errors, the function bodies were not moved correctly — compare `ols_helpers.v` against what was in `did.v`.

- [ ] **Step 6: Commit**

```bash
git add experiment/ols_helpers.v experiment/did.v
git commit -m "refactor(experiment): extract ols_se and matrix_inverse to ols_helpers.v"
```

---

## Task 5: `experiment.icc` and `experiment.design_effect`

**Files:**
- Modify: `experiment/sample_size.v`
- Modify: `tests/sample_size_test.v`

- [ ] **Step 1: Write failing tests** — add to `tests/sample_size_test.v`:

```v
fn test__icc_maximum_clustering() {
	// Clusters with identical within-cluster values → ICC = 1.0
	// cluster_ids: [0,0,1,1,2,2], y: [1,1,2,2,3,3]
	y           := [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
	cluster_ids := [0, 0, 1, 1, 2, 2]
	result := experiment.icc(y, cluster_ids)
	assert math.abs(result - 1.0) < 1e-6
}

fn test__icc_known_value() {
	// y=[1,2,3,4,5,6], cluster_ids=[0,0,0,1,1,1]
	// cluster 0: mean=2, cluster 1: mean=5, grand_mean=3.5
	// SS_between = 3*(2-3.5)^2 + 3*(5-3.5)^2 = 6.75+6.75 = 13.5
	// SS_within  = (1-2)^2+(2-2)^2+(3-2)^2+(4-5)^2+(5-5)^2+(6-5)^2 = 4.0
	// MS_between = 13.5/1 = 13.5, MS_within = 4.0/4 = 1.0, m_bar=3
	// ICC = (13.5-1.0)/(13.5+(3-1)*1.0) = 12.5/15.5 ≈ 0.8065
	y           := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	cluster_ids := [0, 0, 0, 1, 1, 1]
	result := experiment.icc(y, cluster_ids)
	assert math.abs(result - 12.5 / 15.5) < 1e-6
}

fn test__icc_no_clustering() {
	// All values identical within group means, zero between → ICC ≤ 0 → clamped to 0
	y           := [5.0, 5.0, 5.0, 5.0]
	cluster_ids := [0, 0, 1, 1]
	result := experiment.icc(y, cluster_ids)
	assert result == 0.0
}

fn test__design_effect_formula() {
	// DEFF = 1 + (m-1)*ICC
	deff := experiment.design_effect(0.1, 10.0)
	assert math.abs(deff - (1.0 + 9.0 * 0.1)) < 1e-9  // = 1.9
}

fn test__design_effect_no_clustering() {
	deff := experiment.design_effect(0.0, 20.0)
	assert math.abs(deff - 1.0) < 1e-9
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/sample_size_test.v
```
Expected: compile error — `experiment.icc` and `experiment.design_effect` undefined.

- [ ] **Step 3: Implement** — add to end of `experiment/sample_size.v`:

```v
pub fn icc(y []f64, cluster_ids []int) f64 {
	assert y.len == cluster_ids.len, 'y and cluster_ids must have same length'
	assert y.len >= 2, 'need at least 2 observations'

	n := y.len
	mut grand_sum := 0.0
	for v in y { grand_sum += v }
	grand_mean := grand_sum / f64(n)

	// Group values by cluster
	mut cluster_map := map[int][]f64{}
	for i in 0 .. n {
		cluster_map[cluster_ids[i]] << y[i]
	}
	k := cluster_map.len
	if k < 2 { return 0.0 }

	mut ss_between := 0.0
	mut ss_within := 0.0
	mut total_m := 0
	for _, vals in cluster_map {
		m := vals.len
		total_m += m
		mut cm_sum := 0.0
		for v in vals { cm_sum += v }
		cluster_mean := cm_sum / f64(m)
		ss_between += f64(m) * (cluster_mean - grand_mean) * (cluster_mean - grand_mean)
		for v in vals {
			ss_within += (v - cluster_mean) * (v - cluster_mean)
		}
	}

	df_between := f64(k - 1)
	df_within := f64(n - k)
	if df_within <= 0 { return 0.0 }

	ms_between := ss_between / df_between
	ms_within  := ss_within / df_within
	if ms_between <= ms_within { return 0.0 }

	m_bar := f64(total_m) / f64(k)
	icc_val := (ms_between - ms_within) / (ms_between + (m_bar - 1.0) * ms_within)
	return if icc_val > 1.0 { 1.0 } else { icc_val }
}

pub fn design_effect(icc_ f64, avg_cluster_size f64) f64 {
	return 1.0 + (avg_cluster_size - 1.0) * icc_
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/sample_size_test.v
```
Expected: all tests PASS.

- [ ] **Step 5: Full suite**

```bash
make test
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add experiment/sample_size.v tests/sample_size_test.v
git commit -m "feat(experiment): add icc and design_effect for clustered experiments"
```

---

## Task 6: `experiment.ancova`

**Files:**
- Modify: `experiment/abtest.v`
- Modify: `tests/experiment_test.v`

Note: `x_design` is passed to `ml.linear_regression` without an intercept column (that function adds it internally). `ols_se` is called with `[intercept] + coefficients` and returns `se[1]` for the first feature's SE. Treatment is always `x_design[i][0]`.

- [ ] **Step 1: Write failing test** — add to `tests/experiment_test.v`:

```v
fn test__ancova_reduces_noise() {
	// True treatment effect = 3.0. Covariate (baseline) explains variance.
	// Without ANCOVA the effect is still estimable; with ANCOVA SE is smaller.
	// Use 20 observations with a baseline covariate.
	ctrl_y  := [10.0, 12.0, 11.0, 9.0, 10.5, 11.5, 9.5, 10.0, 11.0, 10.0]
	trt_y   := [13.0, 15.0, 14.0, 12.0, 13.5, 14.5, 12.5, 13.0, 14.0, 13.0]
	// baseline covariate for each user
	ctrl_cov := [[8.0], [10.0], [9.0], [7.0], [8.5], [9.5], [7.5], [8.0], [9.0], [8.0]]
	trt_cov  := [[8.0], [10.0], [9.0], [7.0], [8.5], [9.5], [7.5], [8.0], [9.0], [8.0]]

	mut y := []f64{}
	y << ctrl_y
	y << trt_y
	mut treatment := []int{}
	for _ in 0 .. 10 { treatment << 0 }
	for _ in 0 .. 10 { treatment << 1 }
	mut covariates := [][]f64{}
	covariates << ctrl_cov
	covariates << trt_cov

	result := experiment.ancova(y, treatment, covariates, experiment.ANCOVAConfig{})
	// Effect should be close to 3.0
	assert math.abs(result.effect - 3.0) < 0.5
	assert result.p_value < 0.05
	assert result.ci_lower > 0.0
	assert result.r_squared > 0.0 && result.r_squared <= 1.0
	assert result.n == 20
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/experiment_test.v
```
Expected: compile error — `experiment.ancova` and `experiment.ANCOVAConfig` undefined.

- [ ] **Step 3: Implement** — add to `experiment/abtest.v` (after the existing `cuped_test` function):

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

pub fn ancova(y []f64, treatment []int, covariates [][]f64, cfg ANCOVAConfig) ANCOVAResult {
	assert y.len == treatment.len, 'y and treatment must have same length'
	assert covariates.len == y.len || covariates.len == 0, 'covariates must match y length or be empty'
	n := y.len
	assert n >= 4, 'need at least 4 observations'

	n_cov := if covariates.len > 0 { covariates[0].len } else { 0 }

	// Build design matrix: [treatment, cov_0, cov_1, ...]
	mut x_design := [][]f64{len: n}
	for i in 0 .. n {
		x_design[i] = []f64{len: 1 + n_cov}
		x_design[i][0] = f64(treatment[i])
		for j in 0 .. n_cov {
			x_design[i][1 + j] = covariates[i][j]
		}
	}

	model := ml.linear_regression(x_design, y)
	effect := model.coefficients[0]  // treatment coefficient

	mut full_coefs := []f64{}
	full_coefs << model.intercept
	for c in model.coefficients {
		full_coefs << c
	}
	se_arr := ols_se(x_design, y, full_coefs)
	// se_arr[0]=SE(intercept), se_arr[1]=SE(treatment), se_arr[2..]=SE(covariates)
	se := if se_arr.len > 1 { se_arr[1] } else { 0.0 }

	t_stat := if se > 0 { effect / se } else { 0.0 }
	p_val := 2.0 * prob.normal_cdf(-math.abs(t_stat), 0.0, 1.0)
	z := prob.inverse_normal_cdf(1.0 - cfg.alpha / 2.0, 0.0, 1.0)

	preds := ml.linear_predict(model, x_design)
	r2 := ml.r_squared(y, preds)

	return ANCOVAResult{
		effect:      effect
		se:          se
		t_statistic: t_stat
		p_value:     p_val
		ci_lower:    effect - z * se
		ci_upper:    effect + z * se
		r_squared:   r2
		n:           n
	}
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/experiment_test.v
```
Expected: all tests PASS.

- [ ] **Step 5: Full suite**

```bash
make test
```

- [ ] **Step 6: Commit**

```bash
git add experiment/abtest.v tests/experiment_test.v
git commit -m "feat(experiment): add ancova for covariate-adjusted treatment effect"
```

---

## Task 7: `experiment.null_verdict`

**Files:**
- Modify: `experiment/abtest.v`
- Modify: `tests/experiment_test.v`

- [ ] **Step 1: Write failing tests** — add to `tests/experiment_test.v`:

```v
fn test__null_verdict_significant() {
	// effect=0.5, se=0.1: t=5.0, p≈0 → significant
	result := experiment.null_verdict(0.5, 0.1, 0.2, 0.05)
	assert result.verdict == experiment.NullVerdictKind.significant
}

fn test__null_verdict_true_null() {
	// effect=0.0, se=0.01, mde=0.2, alpha=0.05
	// CI: [-0.0196, 0.0196] — entirely below MDE=0.2 → true_null
	result := experiment.null_verdict(0.0, 0.01, 0.2, 0.05)
	assert result.verdict == experiment.NullVerdictKind.true_null
	assert result.ci_upper < result.mde
}

fn test__null_verdict_inconclusive() {
	// effect=0.1, se=0.15, mde=0.2, alpha=0.05
	// CI: [0.1-1.96*0.15, 0.1+1.96*0.15] = [-0.194, 0.394]
	// CI spans MDE=0.2 → inconclusive
	result := experiment.null_verdict(0.1, 0.15, 0.2, 0.05)
	assert result.verdict == experiment.NullVerdictKind.inconclusive
	assert result.ci_lower < result.mde
	assert result.ci_upper > result.mde
}

fn test__null_verdict_below_mde() {
	// effect=0.05, se=0.02, mde=0.2
	// CI: [0.05-0.039, 0.05+0.039] = [0.011, 0.089] — CI upper < mde → true_null
	// (This maps to true_null not below_mde — ci_upper < mde)
	result := experiment.null_verdict(0.05, 0.02, 0.2, 0.05)
	assert result.verdict == experiment.NullVerdictKind.true_null
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/experiment_test.v
```
Expected: compile error.

- [ ] **Step 3: Implement** — add to `experiment/abtest.v`:

```v
pub enum NullVerdictKind {
	significant
	true_null
	inconclusive
	below_mde
}

pub struct NullVerdictResult {
pub:
	verdict  NullVerdictKind
	label    string
	ci_lower f64
	ci_upper f64
	mde      f64
	p_value  f64
}

pub fn null_verdict(effect f64, se f64, mde f64, alpha f64) NullVerdictResult {
	assert se >= 0.0, 'se must be non-negative'
	assert mde > 0.0, 'mde must be positive'
	z := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	ci_lower := effect - z * se
	ci_upper := effect + z * se
	p_val := if se > 0.0 {
		2.0 * prob.normal_cdf(-math.abs(effect / se), 0.0, 1.0)
	} else {
		if effect == 0.0 { 1.0 } else { 0.0 }
	}

	verdict := if p_val < alpha {
		NullVerdictKind.significant
	} else if ci_upper < mde {
		NullVerdictKind.true_null
	} else if ci_lower < mde && ci_upper >= mde {
		NullVerdictKind.inconclusive
	} else {
		NullVerdictKind.below_mde
	}

	label := match verdict {
		.significant  { 'Significant — effect detected above threshold' }
		.true_null    { 'True null — effect is real but below MDE' }
		.inconclusive { 'Inconclusive — underpowered, cannot distinguish null from below-MDE' }
		.below_mde    { 'Below MDE — effect exists but is commercially irrelevant' }
	}

	return NullVerdictResult{
		verdict:  verdict
		label:    label
		ci_lower: ci_lower
		ci_upper: ci_upper
		mde:      mde
		p_value:  p_val
	}
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/experiment_test.v
```
Expected: all tests PASS.

- [ ] **Step 5: Full suite**

```bash
make test
```

- [ ] **Step 6: Commit**

```bash
git add experiment/abtest.v tests/experiment_test.v
git commit -m "feat(experiment): add null_verdict for result classification"
```

---

## Task 8: `experiment.itt_and_pp`

**Files:**
- Modify: `experiment/abtest.v`
- Modify: `tests/experiment_test.v`

- [ ] **Step 1: Write failing tests** — add to `tests/experiment_test.v`:

```v
fn test__itt_and_pp_compliance_rate() {
	// 10 control, 10 treatment — 7 of treatment activated (compliance=0.7)
	y         := [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	treatment := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	              1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	activated := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	              1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  // 7 of 10 activated
	result := experiment.itt_and_pp(y, treatment, activated, experiment.ABTestConfig{})
	assert math.abs(result.compliance_rate - 0.7) < 1e-9
	// ITT effect = mean(trt) - mean(ctrl) = 1.0 - 0.0 = 1.0
	assert math.abs(result.itt.treatment_mean - 1.0) < 1e-9
	assert math.abs(result.itt.control_mean - 0.0) < 1e-9
	// LATE = ITT_effect / compliance = 1.0 / 0.7 ≈ 1.429
	assert math.abs(result.late - 1.0 / 0.7) < 1e-6
	assert result.late_se > 0.0
}

fn test__itt_and_pp_full_compliance_late_equals_pp() {
	// When compliance = 1.0, LATE should equal ITT effect
	y         := [10.0, 11.0, 9.0, 10.0, 13.0, 14.0, 12.0, 13.0]
	treatment := [0, 0, 0, 0, 1, 1, 1, 1]
	activated := [0, 0, 0, 0, 1, 1, 1, 1]  // all activated
	result := experiment.itt_and_pp(y, treatment, activated, experiment.ABTestConfig{})
	assert math.abs(result.compliance_rate - 1.0) < 1e-9
	assert math.abs(result.late - result.itt.relative_lift * result.itt.control_mean) < 0.5
	// ITT == PP when compliance is 1.0
	assert math.abs(result.itt.treatment_mean - result.pp.treatment_mean) < 1e-9
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/experiment_test.v
```
Expected: compile error.

- [ ] **Step 3: Implement** — add to `experiment/abtest.v`:

```v
pub struct ITTPPResult {
pub:
	itt             ABTestResult
	pp              ABTestResult
	compliance_rate f64
	late            f64
	late_se         f64
}

pub fn itt_and_pp(y []f64, treatment []int, activated []int, cfg ABTestConfig) ITTPPResult {
	assert y.len == treatment.len && y.len == activated.len, 'y, treatment, activated must have same length'

	// ITT: all assigned users
	mut y_ctrl := []f64{}
	mut y_trt  := []f64{}
	for i in 0 .. y.len {
		if treatment[i] == 0 { y_ctrl << y[i] } else { y_trt << y[i] }
	}
	itt := abtest(y_ctrl, y_trt, cfg)

	// Compliance rate: fraction of treatment-assigned who activated
	mut n_trt := 0
	mut n_activated := 0
	for i in 0 .. y.len {
		if treatment[i] == 1 {
			n_trt++
			if activated[i] == 1 { n_activated++ }
		}
	}
	compliance := if n_trt > 0 { f64(n_activated) / f64(n_trt) } else { 0.0 }

	// Per-protocol: activated treatment vs all control
	mut y_pp_ctrl := []f64{}
	mut y_pp_trt  := []f64{}
	for i in 0 .. y.len {
		if treatment[i] == 0 {
			y_pp_ctrl << y[i]
		} else if activated[i] == 1 {
			y_pp_trt << y[i]
		}
	}
	pp := if y_pp_ctrl.len >= 2 && y_pp_trt.len >= 2 {
		abtest(y_pp_ctrl, y_pp_trt, cfg)
	} else {
		ABTestResult{}
	}

	// IV/Wald LATE = ITT_effect / compliance
	itt_effect := itt.treatment_mean - itt.control_mean
	late := if compliance > 0.0 { itt_effect / compliance } else { 0.0 }
	itt_se := if itt.n_treatment > 0 && itt.n_control > 0 {
		math.sqrt(
			itt.treatment_std * itt.treatment_std / f64(itt.n_treatment) +
			itt.control_std * itt.control_std / f64(itt.n_control)
		)
	} else { 0.0 }
	late_se := if compliance > 0.0 { itt_se / compliance } else { 0.0 }

	return ITTPPResult{
		itt:             itt
		pp:              pp
		compliance_rate: compliance
		late:            late
		late_se:         late_se
	}
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/experiment_test.v
```
Expected: all tests PASS.

- [ ] **Step 5: Full suite**

```bash
make test
```

- [ ] **Step 6: Commit**

```bash
git add experiment/abtest.v tests/experiment_test.v
git commit -m "feat(experiment): add itt_and_pp for compliance-adjusted effect estimation"
```

---

## Task 9: `experiment/readout.v` — `srm_test`, `simpsons_check`, `hte_subgroup`

**Files:**
- Create: `experiment/readout.v`
- Create: `tests/readout_test.v`

`readout.v` imports `stats` and `prob` (both available to `experiment`). `hte_subgroup` calls `abtest` (same module, no import needed).

- [ ] **Step 1: Write failing tests** — create `tests/readout_test.v`:

```v
import vstats.experiment
import math

fn test__srm_test_no_srm_equal_split() {
	// 500 and 500 with expected 50/50 — no SRM
	result := experiment.srm_test([500, 500], [0.5, 0.5], 0.01)
	assert result.srm_detected == false
	assert result.chi2 == 0.0
	assert result.p_value == 1.0
}

fn test__srm_test_detects_imbalance() {
	// 800 vs 200 expected 50/50 → severe SRM
	result := experiment.srm_test([800, 200], [0.5, 0.5], 0.01)
	assert result.srm_detected == true
	assert result.chi2 > 100.0  // very large chi2
	assert result.p_value < 0.001
}

fn test__srm_test_three_variants() {
	// 330/330/340 vs expected 1/3 each — minor imbalance, should not trigger at strict alpha
	result := experiment.srm_test([330, 330, 340], [0.333, 0.333, 0.334], 0.001)
	assert result.srm_detected == false
}

fn test__simpsons_check_no_reversal() {
	// aggregate ATE positive, all segment ATEs positive → no reversal
	y         := [1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0]
	treatment := [0, 0, 1, 1, 0, 0, 1, 1]
	segment   := [0, 0, 0, 0, 1, 1, 1, 1]
	result := experiment.simpsons_check(y, treatment, segment)
	assert result.reversal_detected == false
	assert result.aggregate_ate > 0.0
}

fn test__simpsons_check_detects_reversal() {
	// Aggregate ATE is positive but segment 0 has negative ATE
	// segment 0: ctrl=[10,10], trt=[8,8] → ATE=-2 (negative)
	// segment 1: ctrl=[1,1], trt=[9,9] → ATE=+8 (positive)
	// aggregate: ctrl=[10,10,1,1]/4=5.5, trt=[8,8,9,9]/4=8.5 → ATE=+3 (positive)
	y         := [10.0, 10.0, 1.0, 1.0,   8.0, 8.0, 9.0, 9.0]
	treatment := [0, 0, 0, 0,              1, 1, 1, 1]
	segment   := [0, 0, 1, 1,              0, 0, 1, 1]
	result := experiment.simpsons_check(y, treatment, segment)
	assert result.reversal_detected == true
	assert result.aggregate_ate > 0.0
	// Segment 0: ctrl=[10,10], trt=[8,8] → ATE=-2
	seg0_idx := result.segment_ids.index(0)
	assert result.segment_ates[seg0_idx] < 0.0
}

fn test__hte_subgroup_returns_overall_and_segments() {
	// 2 segments, clear effect in both
	y         := [1.0, 1.0, 1.0, 5.0, 5.0, 5.0,   // segment 0
	              2.0, 2.0, 2.0, 6.0, 6.0, 6.0]    // segment 1
	treatment := [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
	segment   := [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
	result := experiment.hte_subgroup(y, treatment, segment, experiment.ABTestConfig{})
	assert result.subgroups.len == 2
	assert result.overall.significant == true
	// Both segments should show positive effect
	for sg in result.subgroups {
		assert sg.ate > 0.0
		assert sg.p_value < 0.05
	}
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
v test tests/readout_test.v
```
Expected: compile error — module file does not exist.

- [ ] **Step 3: Implement** — create `experiment/readout.v`:

```v
module experiment

import math
import prob
import stats

pub struct SRMResult {
pub:
	chi2         f64
	p_value      f64
	srm_detected bool
	observed     []int
	expected     []f64
}

pub struct SimpsonsResult {
pub:
	aggregate_ate     f64
	segment_ates      []f64
	segment_ids       []int
	reversal_detected bool
}

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

pub fn srm_test(observed_counts []int, expected_proportions []f64, alpha f64) SRMResult {
	n := observed_counts.len
	assert n == expected_proportions.len && n >= 2, 'observed and expected must have same length >= 2'
	mut total := 0
	for c in observed_counts { total += c }
	assert total > 0, 'total observed must be positive'

	mut expected := []f64{len: n}
	for i in 0 .. n {
		expected[i] = expected_proportions[i] * f64(total)
	}

	mut chi2 := 0.0
	for i in 0 .. n {
		if expected[i] > 0.0 {
			diff := f64(observed_counts[i]) - expected[i]
			chi2 += diff * diff / expected[i]
		}
	}

	df := f64(n - 1)
	p_val := 1.0 - prob.chi_squared_cdf(chi2, int(df))

	return SRMResult{
		chi2:         chi2
		p_value:      p_val
		srm_detected: p_val < alpha
		observed:     observed_counts
		expected:     expected
	}
}

pub fn simpsons_check(y []f64, treatment []int, segment []int) SimpsonsResult {
	assert y.len == treatment.len && y.len == segment.len, 'y, treatment, segment must have same length'

	mut y_ctrl := []f64{}
	mut y_trt  := []f64{}
	for i in 0 .. y.len {
		if treatment[i] == 0 { y_ctrl << y[i] } else { y_trt << y[i] }
	}
	agg_ate := stats.mean(y_trt) - stats.mean(y_ctrl)

	mut seg_map := map[int]bool{}
	for s in segment { seg_map[s] = true }

	mut seg_ids  := []int{}
	mut seg_ates := []f64{}
	for seg_id, _ in seg_map {
		mut sc := []f64{}
		mut st := []f64{}
		for i in 0 .. y.len {
			if segment[i] == seg_id {
				if treatment[i] == 0 { sc << y[i] } else { st << y[i] }
			}
		}
		if sc.len == 0 || st.len == 0 { continue }
		seg_ids  << seg_id
		seg_ates << stats.mean(st) - stats.mean(sc)
	}

	mut reversal := false
	for ate in seg_ates {
		if (ate > 0.0) != (agg_ate > 0.0) {
			reversal = true
			break
		}
	}

	return SimpsonsResult{
		aggregate_ate:     agg_ate
		segment_ates:      seg_ates
		segment_ids:       seg_ids
		reversal_detected: reversal
	}
}

pub fn hte_subgroup(y []f64, treatment []int, segment []int, cfg ABTestConfig) HTEResult {
	assert y.len == treatment.len && y.len == segment.len, 'y, treatment, segment must have same length'

	mut seg_map := map[int]bool{}
	for s in segment { seg_map[s] = true }

	mut subgroups := []SubgroupResult{}
	for seg_id, _ in seg_map {
		mut y_c := []f64{}
		mut y_t := []f64{}
		for i in 0 .. y.len {
			if segment[i] == seg_id {
				if treatment[i] == 0 { y_c << y[i] } else { y_t << y[i] }
			}
		}
		if y_c.len < 2 || y_t.len < 2 { continue }
		r := abtest(y_c, y_t, cfg)
		se := math.sqrt(
			r.treatment_std * r.treatment_std / f64(r.n_treatment) +
			r.control_std   * r.control_std   / f64(r.n_control)
		)
		subgroups << SubgroupResult{
			segment_id: seg_id
			n_ctrl:     r.n_control
			n_trt:      r.n_treatment
			ate:        r.treatment_mean - r.control_mean
			se:         se
			p_value:    r.p_value
			ci_lower:   r.ci_lower
			ci_upper:   r.ci_upper
		}
	}

	mut y_ctrl := []f64{}
	mut y_trt  := []f64{}
	for i in 0 .. y.len {
		if treatment[i] == 0 { y_ctrl << y[i] } else { y_trt << y[i] }
	}
	overall := abtest(y_ctrl, y_trt, cfg)

	return HTEResult{ subgroups: subgroups, overall: overall }
}
```

- [ ] **Step 4: Run tests**

```bash
v test tests/readout_test.v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Full suite**

```bash
make test
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add experiment/readout.v tests/readout_test.v
git commit -m "feat(experiment): add srm_test, simpsons_check, hte_subgroup to readout module"
```

---

## Task 10: Final full-suite verification

Before skill updates, confirm everything compiles and passes clean.

- [ ] **Step 1: Run full test suite**

```bash
make test
```
Expected: all tests PASS with no warnings.

- [ ] **If anything fails:** check that:
  - `experiment/ols_helpers.v` was created and `did.v` no longer defines `matrix_inverse` or `ols_se`
  - `stats/inference.v` has `module stats` at the top (not `module inference`)
  - `stats/multiple_testing.v` has `module stats` at the top
  - `experiment/readout.v` has `module experiment` at the top
  - All new files import `math` where `math.abs`, `math.sqrt` are used

---

## Task 11: Skill updates

**Files:**
- Modify: `/home/rabt/.claude/skills/vstats/references/modules.md`
- Modify: `/home/rabt/.claude/skills/vstats/SKILL.md`

### 11a — Fix stale entries in `modules.md`

- [ ] **Step 1: Fix `ABTestConfig`** — in the `experiment` section, find:
  ```
  // ABTestConfig{ alpha: 0.05, two_tailed: true }
  ```
  Replace with:
  ```
  // ABTestConfig{ alpha: 0.05, equal_variance: false }
  ```

- [ ] **Step 2: Fix SPRT functions** — find the SPRT description block and replace:
  ```
  Available in `abtest.v` — use for ongoing rollouts where you look at data continuously without inflating false-positive rate. Functions: `sprt_update`, `sprt_boundaries`.
  ```
  With:
  ```
  Available in `sequential.v`. Use for ongoing rollouts without inflating false-positive rate.
  ```
  Then add a code block:
  ```v
  result := experiment.sprt_test(successes_a, n_a, successes_b, n_b,
      experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.02 })
  // result.decision: .continue_testing | .reject_null | .accept_null
  ```

- [ ] **Step 3: Fix DiD functions** — find the DiD section and replace the stale signatures:
  ```
  did_estimate(treated_pre []f64, treated_post []f64, control_pre []f64, control_post []f64) DiDResult
  test_parallel_trends(treated_pre [][]f64, control_pre [][]f64) ParallelTrendsResult
  event_study(treated [][]f64, control [][]f64, treatment_period int) EventStudyResult
  ```
  With:
  ```v
  did_2x2(y_treat_pre []f64, y_treat_post []f64, y_ctrl_pre []f64, y_ctrl_post []f64, cfg DiDConfig) DiDResult
  did_regression(y []f64, x [][]f64, group []int, time []int, cfg DiDConfig) DiDRegressionResult
  test_parallel_trends(y_treated_pre []f64, y_control_pre []f64, time_pre []int, cfg DiDConfig) ParallelTrendsResult
  event_study(y []f64, group []int, relative_time []int, cfg DiDConfig) EventStudyResult
  ```

- [ ] **Step 4: Fix PSM functions** — find the PSM section and replace:
  ```
  estimate_propensity_scores(x [][]f64, treatment []int) []f64
  match_units(propensity_scores []f64, treatment []int, caliper f64) []MatchedPair
  estimate_att(outcomes []f64, treatment []int, matched_pairs []MatchedPair) f64
  ```
  With:
  ```v
  estimate_propensity_scores(x [][]f64, treatment []f64, cfg PropensityConfig) PropensityModel
  match_nearest_neighbor(model PropensityModel, cfg MatchingConfig) MatchingResult
  check_balance(x [][]f64, treatment []f64, result MatchingResult) BalanceResult
  ate_matched(y []f64, treatment []f64, result MatchingResult) ATEResult
  ```

- [ ] **Step 5: Add new `stats` functions** — in the `stats` section, append:

  ```v
  // Winsorization & RTM
  winsorize(x []f64, q_low f64, q_high f64) []f64
  rtm_correction(baseline []f64, followup []f64, selection_threshold f64) f64

  // Multiple testing corrections (stats/multiple_testing.v)
  bh_correction(p_values []f64, alpha f64) BHResult
  // BHResult{ adjusted []f64; reject []bool; n_rejected int }
  bonferroni_correction(p_values []f64, alpha f64) BonferroniResult
  // BonferroniResult{ adjusted []f64; reject []bool; n_rejected int }

  // Ratio metrics & bootstrap (stats/inference.v)
  delta_method_ratio(a []f64, b []f64, treatment []int, cfg DeltaMethodConfig) DeltaMethodResult
  // DeltaMethodConfig{ alpha f64 = 0.05 }
  // DeltaMethodResult{ ratio_ctrl f64; ratio_trt f64; effect f64; se f64; t_statistic f64; p_value f64; ci_lower f64; ci_upper f64 }
  bootstrap_test(ctrl []f64, trt []f64, n_resamples int) BootstrapResult
  // BootstrapResult{ p_value f64; observed_diff f64; ci_lower f64; ci_upper f64; n_resamples int }
  ```

- [ ] **Step 6: Add new `experiment` functions** — in the `experiment` section, append after the existing entries:

  ```v
  // Cluster design (experiment/sample_size.v)
  icc(y []f64, cluster_ids []int) f64
  design_effect(icc_ f64, avg_cluster_size f64) f64

  // Covariate adjustment (experiment/abtest.v)
  ancova(y []f64, treatment []int, covariates [][]f64, cfg ANCOVAConfig) ANCOVAResult
  // ANCOVAConfig{ alpha f64 = 0.05 }
  // ANCOVAResult{ effect f64; se f64; t_statistic f64; p_value f64; ci_lower f64; ci_upper f64; r_squared f64; n int }

  // Result interpretation (experiment/abtest.v)
  null_verdict(effect f64, se f64, mde f64, alpha f64) NullVerdictResult
  // NullVerdictResult{ verdict NullVerdictKind; label string; ci_lower f64; ci_upper f64; mde f64; p_value f64 }
  // NullVerdictKind: .significant | .true_null | .inconclusive | .below_mde

  // Compliance-adjusted effects (experiment/abtest.v)
  itt_and_pp(y []f64, treatment []int, activated []int, cfg ABTestConfig) ITTPPResult
  // ITTPPResult{ itt ABTestResult; pp ABTestResult; compliance_rate f64; late f64; late_se f64 }

  // Readout checks (experiment/readout.v)
  srm_test(observed_counts []int, expected_proportions []f64, alpha f64) SRMResult
  // SRMResult{ chi2 f64; p_value f64; srm_detected bool; observed []int; expected []f64 }
  simpsons_check(y []f64, treatment []int, segment []int) SimpsonsResult
  // SimpsonsResult{ aggregate_ate f64; segment_ates []f64; segment_ids []int; reversal_detected bool }
  hte_subgroup(y []f64, treatment []int, segment []int, cfg ABTestConfig) HTEResult
  // HTEResult{ subgroups []SubgroupResult; overall ABTestResult }
  // SubgroupResult{ segment_id int; n_ctrl int; n_trt int; ate f64; se f64; p_value f64; ci_lower f64; ci_upper f64 }
  ```

### 11b — Update `SKILL.md`

- [ ] **Step 7: Fix the experiment snippet** — in the `## Experiment / A/B Testing Workflow` section, find:
  ```v
  result := experiment.abtest(control, treatment, experiment.ABTestConfig{
      alpha: 0.05
      two_tailed: true
  })
  ```
  Replace with:
  ```v
  result := experiment.abtest(control, treatment, experiment.ABTestConfig{
      alpha: 0.05
  })
  ```

- [ ] **Step 8: Add readout workflow section** — add after the existing `## Experiment / A/B Testing Workflow` section:

  ````markdown
  ## Rigorous Readout Workflow

  Canonical order for a complete experiment readout:

  ```v
  import vstats.experiment
  import vstats.stats

  // 1. Check randomization was clean (run before looking at metrics)
  srm := experiment.srm_test([n_ctrl, n_trt], [0.5, 0.5], 0.01)
  if srm.srm_detected { panic('SRM detected — do not read out') }

  // 2. Pre-process metric (winsorize at p99 on pooled data)
  mut pooled := y_ctrl.clone(); pooled << y_trt
  y_ctrl_w := stats.winsorize(pooled, 0.0, 0.99)[..y_ctrl.len]
  y_trt_w  := stats.winsorize(pooled, 0.0, 0.99)[y_ctrl.len..]

  // 3a. For ratio metrics (ARPU, CTR): use delta method
  ratio_result := stats.delta_method_ratio(revenue, sessions, treatment, stats.DeltaMethodConfig{})

  // 3b. For continuous means: run t-test (optionally with CUPED or ANCOVA)
  result := experiment.abtest(y_ctrl_w, y_trt_w, experiment.ABTestConfig{})

  // 4. Multiple testing correction (if testing multiple metrics)
  bh := stats.bh_correction([result.p_value, secondary_p], 0.05)

  // 5. Interpret null result (if not significant)
  if !result.significant {
      verdict := experiment.null_verdict(result.treatment_mean - result.control_mean,
          result.control_std, mde, 0.05)
      println(verdict.label)
  }

  // 6. Subgroup analysis (pre-registered segments only)
  hte := experiment.hte_subgroup(y, treatment, segment, experiment.ABTestConfig{})
  simp := experiment.simpsons_check(y, treatment, segment)
  if simp.reversal_detected { println('Warning: Simpson reversal detected') }
  ```
  ````

- [ ] **Step 9: Commit skill updates**

  ```bash
  git -C /home/rabt/.claude add skills/vstats/references/modules.md skills/vstats/SKILL.md
  git -C /home/rabt/.claude commit -m "feat(skills/vstats): fix stale entries and add rigorous readout functions"
  ```

  Note: the skills directory is in `~/.claude`, a separate git repo. Use `-C /home/rabt/.claude` to target it.

---

## Self-Review Checklist

- **Spec coverage:**
  - `stats.winsorize` ✓ Task 1
  - `stats.rtm_correction` ✓ Task 1
  - `stats.bh_correction` + `bonferroni_correction` ✓ Task 2
  - `stats.delta_method_ratio` ✓ Task 3
  - `stats.bootstrap_test` ✓ Task 3
  - `experiment.icc` + `design_effect` ✓ Task 5
  - `experiment.ancova` ✓ Task 6
  - `experiment.null_verdict` ✓ Task 7
  - `experiment.itt_and_pp` ✓ Task 8
  - `experiment.srm_test` ✓ Task 9
  - `experiment.simpsons_check` ✓ Task 9
  - `experiment.hte_subgroup` ✓ Task 9
  - Skill stale-entry fixes ✓ Task 11
  - Skill new-function entries ✓ Task 11

- **Type consistency check:**
  - `ANCOVAConfig` defined in Task 6, used in Task 6 ✓
  - `NullVerdictKind` enum defined in Task 7, referenced in tests as `experiment.NullVerdictKind.significant` ✓
  - `ITTPPResult.itt` is `ABTestResult` — `ABTestResult` is already defined in `abtest.v` ✓
  - `HTEResult.overall` is `ABTestResult` ✓
  - `SubgroupResult.ate` computed as `r.treatment_mean - r.control_mean` — `ABTestResult` has those fields ✓
  - `ols_se` called in Task 6 (`ancova`) — exists after Task 4 extraction ✓

- **No placeholders:** all steps contain complete code or exact commands.
