# Sequential Testing with a 1% Holdout During Progressive Rollouts

When you ship a feature gradually — 1% → 5% → 20% → 50% → 100% — you want to
know as early as possible whether something is broken or working. This tutorial
shows how to do that correctly using vstats, without inflating your false
positive rate by peeking at data repeatedly.

## The Problem with Peeking

A standard t-test is designed to be evaluated **once**, after a fixed sample
size determined upfront. If you run it every day while accumulating data, the
probability of seeing a spurious "significant" result climbs fast:

| Times you check | Effective false positive rate (targeting 5%) |
|---|---|
| 1 | 5% |
| 5 | ~14% |
| 20 | ~25% |
| 100 | ~37% |

This is why you need a **sequential testing** strategy.

---

## Setup: The 1% Permanent Holdout

The holdout is a slice of users who **never** receive the feature — not even at
100% rollout. They are your long-term control group.

```
Day 1:  [1% holdout] [1% treatment]  [98% not yet exposed]
Day 3:  [1% holdout] [5% treatment]  [94% not yet exposed]
Day 7:  [1% holdout] [20% treatment] [79% not yet exposed]
Day 14: [1% holdout] [50% treatment] [49% not yet exposed]
Day 30: [1% holdout] [99% treatment]
```

Every analysis compares **holdout vs. current treatment cohort**. The holdout
bucket never changes; only the treatment side grows.

**Assignment rule:** Use a stable hash of user ID (e.g. `hash(user_id) % 100`).
Bucket 0 is always holdout. Buckets 1–N are treatment as rollout percentage N
grows. This ensures the same user stays in the same group across sessions.

---

## Step 0 — Pre-flight: How Long Will This Take?

Before launching, estimate how many holdout users you need before your test has
meaningful power. The holdout grows at `DAU * 0.01` users per day, so this
calculation tells you the minimum runtime.

### Binary metric (conversion rate, click-through)

```v
module main

import vstats.experiment

fn main() {
    // Scenario: 5% baseline conversion rate.
    // We want to detect a +0.5 percentage point change (absolute).
    // 100K DAU → holdout grows at 1000 users/day.
    ss := experiment.sample_size_proportions(0.05, 0.005, 0.05, 0.80)

    days := ss.n_per_group / 1000   // holdout is the bottleneck
    println('Need ${ss.n_per_group} holdout users → ~${days} days to reach 80% power')
    // Output: Need 14752 holdout users → ~15 days
}
```

`sample_size_proportions(baseline_rate, mde, alpha, power)` returns a
`SampleSizeResult`. The `n_per_group` field is the **minimum holdout size**.
Power is almost entirely determined by the smaller group — the 1% holdout —
regardless of how large the treatment group grows.

### Continuous metric (revenue, session duration)

```v
module main

import vstats.experiment

fn main() {
    // Scenario: mean session = 180s, std = 90s.
    // We want to detect a 5-second improvement.
    ss := experiment.sample_size_means(180.0, 90.0, 5.0, 0.05, 0.80)

    days := ss.n_per_group / 1000
    println('Need ${ss.n_per_group} holdout users → ~${days} days')
    println('Cohen\'s d: ${ss.effect_size:.3f}')
    // Output: Need 5070 holdout users → ~5 days
}
```

`sample_size_means(baseline_mean, baseline_std, mde_absolute, alpha, power)`.
Estimate `baseline_std` from at least 2 weeks of historical data. A noisy
metric (high std) requires more time, not more treatment users.

---

## Step 1 — Sample Ratio Mismatch Check

Before running any statistical test, verify that users are assigned to groups
in the right proportions. A mismatch means there is a logging bug, a bot filter
applied unevenly, or a hashing collision — not a real effect.

```v
module main

import math

fn check_srm(n_holdout int, n_treatment int, rollout_pct f64) bool {
    // At rollout_pct% treatment and 1% holdout, the expected holdout fraction
    // of the (holdout + treatment) population is:
    //   1 / (1 + rollout_pct)
    expected_frac := 1.0 / (1.0 + rollout_pct)
    observed_frac := f64(n_holdout) / f64(n_holdout + n_treatment)
    relative_deviation := math.abs(observed_frac - expected_frac) / expected_frac

    if relative_deviation > 0.05 {
        println('SRM DETECTED: expected holdout fraction ${expected_frac:.4f}, ' +
                'observed ${observed_frac:.4f} (${relative_deviation * 100:.1f}% deviation)')
        return false
    }
    println('SRM check passed (${relative_deviation * 100:.1f}% deviation)')
    return true
}

fn main() {
    // Day 7: 20% rollout. 100K DAU, so ~1000 holdout and ~20000 treatment.
    // Simulate a small imbalance — 950 holdout instead of 1000.
    ok := check_srm(950, 20000, 20.0)
    if !ok {
        // Stop. Do not interpret test results until the bug is fixed.
        return
    }
}
```

**Always run this check first, every time.** An SRM is a fundamental flaw in
the experiment — statistical results from an SRM experiment are uninterpretable.

---

## Step 2A — Sequential Testing for Binary Metrics (SPRT)

For binary outcomes (did the user convert? did they click?), use
`experiment.sprt_test`. This implements Wald's Sequential Probability Ratio
Test: you can call it at any point as data accumulates and it will tell you
whether to stop.

```v
module main

import vstats.experiment

fn main() {
    // You call this at each analysis checkpoint as cumulative totals grow.
    // control = holdout group, treatment = feature-receiving group.
    //
    // Day 7 snapshot:
    //   holdout:   1200 users, 60 conversions  → 5.0% rate
    //   treatment: 8500 users, 459 conversions → 5.4% rate

    cfg := experiment.SPRTConfig{
        alpha: 0.05   // false positive tolerance
        beta:  0.20   // false negative tolerance (1 - power)
        mde:   0.005  // minimum detectable effect: 0.5 percentage points
    }

    r := experiment.sprt_test(60, 1200, 459, 8500, cfg)

    println('Log-likelihood ratio: ${r.log_likelihood_ratio:.3f}')
    println('  Upper boundary (reject H0): ${r.upper_boundary:.3f}')
    println('  Lower boundary (accept H0): ${r.lower_boundary:.3f}')
    println('  Holdout rate:   ${r.rate_a * 100:.2f}%')
    println('  Treatment rate: ${r.rate_b * 100:.2f}%')

    match r.decision {
        .reject_null    { println('Decision: STOP — effect detected, safe to accelerate rollout') }
        .accept_null    { println('Decision: STOP — no effect, consider rollback') }
        .continue_testing { println('Decision: CONTINUE — not enough evidence yet') }
    }
}
```

**How SPRT works:** It maintains a log-likelihood ratio comparing two
hypotheses — H0 (no effect) vs H1 (effect of size `mde`). When the ratio
crosses the upper boundary `log((1-β)/α)`, reject H0. When it crosses the
lower boundary `log(β/(1-α))`, accept H0 (futility). In the middle: keep
accumulating data.

**Setting `mde`:** Use the smallest effect you would actually act on. Smaller
`mde` = more data needed = slower to decide. A common starting point is 10–20%
of your baseline rate.

**The SPRT is stateless** — it takes cumulative totals, not incremental data.
Store your running sums and call `sprt_test` with the updated totals each time.

---

## Step 2B — Sequential Testing for Continuous Metrics (Group Sequential)

For continuous outcomes (revenue per user, session duration, latency), use
`experiment.abtest` at pre-planned checkpoints combined with an
**O'Brien-Fleming spending function** to allocate your α budget.

The O'Brien-Fleming boundary at information fraction `t` (where `t = 1.0`
means all the data you planned to collect) is:

```
critical_z(t) = z_{α/2} / √t
```

This is deliberately conservative early (hard to stop at 1% of data) and
nearly standard at the end (close to `z_{1.96}` at `t = 1.0`).

```v
module main

import vstats.experiment
import vstats.prob
import math

// obf_boundary returns the critical |z| value at information fraction t.
// If |result.t_statistic| >= obf_boundary(t, alpha), the result is significant.
fn obf_boundary(t f64, alpha f64) f64 {
    z := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
    return z / math.sqrt(t)
}

fn main() {
    // Pre-flight: we planned for 5000 holdout users (from sample_size_means).
    target_holdout := 5000

    // ---- Day 7 look: 20% rollout ----
    // Holdout: 700 users measured. Information fraction = 700/5000 = 0.14.
    // Treatment: 14000 users measured (we only need holdout size for t).

    // Simulated session durations (seconds):
    holdout_day7 := [165.0, 190.0, 172.0, 185.0, 168.0, 195.0, 178.0,
                     160.0, 188.0, 174.0] // ... in practice: 700 values

    treatment_day7 := [170.0, 198.0, 183.0, 191.0, 172.0, 204.0, 186.0,
                       167.0, 196.0, 181.0] // ... in practice: 14000 values

    t := f64(holdout_day7.len) / f64(target_holdout)
    boundary := obf_boundary(t, 0.05)

    r := experiment.abtest(holdout_day7, treatment_day7)

    println('Information fraction: ${t:.3f}')
    println('O\'Brien-Fleming boundary: |z| >= ${boundary:.3f}')
    println('Observed |t-statistic|:    ${math.abs(r.t_statistic):.3f}')
    println('Treatment lift: ${r.relative_lift * 100:.1f}%')
    println('95% CI for difference: [${r.ci_lower:.1f}, ${r.ci_upper:.1f}]')

    if math.abs(r.t_statistic) >= boundary {
        println('Decision: STOP — crosses O\'Brien-Fleming boundary')
    } else {
        println('Decision: CONTINUE — below boundary, keep rolling out')
    }
}
```

**Why not just use the p-value?** The p-value from `abtest` is calibrated for
a single look. At early looks (small `t`), the O'Brien-Fleming boundary is much
stricter than p < 0.05 — you need overwhelming evidence to stop early. At the
final planned look, it's essentially equivalent to p < 0.05.

**Planned look schedule:** Set your checkpoints before launching. A natural
schedule follows the rollout milestones:

| Rollout % | Approx. information fraction | O'Brien-Fleming (z) boundary |
|---|---|---|
| 1% treatment | ~0.01 | ~9.8 |
| 5% treatment | ~0.05 | ~4.4 |
| 20% treatment | ~0.20 | ~2.2 |
| 50% treatment | ~0.50 | ~1.4 |
| 99% treatment | ~1.00 | ~1.0 |

*(Assuming α = 0.05, z_{α/2} = 1.96)*

Early stopping requires a very strong signal. This protects against false
positives caused by catching an unusual week or a short-term novelty spike.

---

## Step 3 — Interpreting Results and Making Decisions

### Binary metrics (SPRT)

```
sprt_test decision  │ What it means
────────────────────┼──────────────────────────────────────────────────────────
reject_null         │ Effect is real and ≥ MDE. Safe to accelerate or ship.
accept_null         │ No meaningful effect. Consider rollback or abandoning.
continue_testing    │ Insufficient evidence. Check again at the next milestone.
```

Never stop early at `continue_testing` just because the directional trend looks
good — that is exactly the peeking problem this framework prevents.

### Continuous metrics (group sequential)

```
|t_statistic| vs boundary  │ What it means
───────────────────────────┼──────────────────────────────────────────────────
>= boundary                │ Cross the boundary. Stop. Effect is significant.
< boundary                 │ Stay below boundary. Continue to next checkpoint.
```

Additionally, check the confidence interval from `result.ci_lower/ci_upper`.
Even if the test is significant, a CI that barely excludes zero (e.g.
`[0.01, 4.8]` seconds on a 180s baseline) may not be practically meaningful.
Use the CI to communicate the range of plausible effect sizes to stakeholders.

### Stopping for futility

If you reach 50% rollout and the effect is directionally zero with a tight CI
(e.g. `[-1.2, 1.4]` seconds), that is strong evidence of no effect — even
before the SPRT formally accepts H0. You can stop for futility using:

```v
module main

import vstats.experiment

fn futility_check(r experiment.ABTestResult, mde_absolute f64) bool {
    // Futility: CI is entirely inside [-mde, +mde], meaning even the upper
    // bound of plausible effects is too small to matter.
    return r.ci_upper < mde_absolute && r.ci_lower > -mde_absolute
}

fn main() {
    holdout   := [180.2, 179.8, 180.5, 181.0, 179.3]
    treatment := [180.4, 180.1, 179.9, 180.7, 180.2]

    r := experiment.abtest(holdout, treatment)
    if futility_check(r, 5.0) {
        println('Futility: effect is smaller than MDE. Safe to ship without concern.')
    }
}
```

---

## Step 4 — Post-launch Monitoring (Keeping the Holdout as a Canary)

After reaching 100% rollout, **keep the 1% holdout permanently**. Run a
standard `abtest` against it weekly. This catches:

- Metric degradation that emerged after novelty wore off
- Interaction effects with seasonal patterns or other launches
- Long-term engagement changes invisible in a short experiment

```v
module main

import vstats.experiment

fn weekly_canary_check(holdout []f64, treatment_sample []f64) {
    r := experiment.abtest(holdout, treatment_sample)

    println('=== Weekly canary check ===')
    println('  Holdout mean:   ${r.control_mean:.2f}')
    println('  Treatment mean: ${r.treatment_mean:.2f}')
    println('  Lift:           ${r.relative_lift * 100:.1f}%')
    println('  p-value:        ${r.p_value:.4f}')
    if r.significant {
        println('  ⚠ SIGNIFICANT CHANGE — investigate before next release')
    } else {
        println('  Healthy: no significant deviation from holdout')
    }
}

fn main() {
    // Pull weekly samples from your data warehouse.
    // These are random samples of ~500 users per group per week.
    holdout_week := [180.0, 178.5, 182.0, 179.0, 181.5]         // real: ~500 values
    treatment_week := [181.0, 179.5, 183.0, 180.0, 182.5]       // real: ~500 values

    weekly_canary_check(holdout_week, treatment_week)
}
```

This weekly check uses a standard (non-sequential) t-test, which is correct
here — you are making one decision per week, not peeking at the same
accumulating dataset.

---

## Full Example — Simulated 4-Week Rollout

The following is a complete, runnable program that simulates a progressive
rollout for a continuous metric (session duration). It includes all four steps:
pre-flight, SRM check, sequential testing at each milestone, and a final
canary check.

```v
module main

import vstats.experiment
import vstats.prob
import math
import rand

// obf_boundary returns the critical |z| value at information fraction t
// using the O'Brien-Fleming spending function.
fn obf_boundary(t f64, alpha f64) f64 {
    z := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
    return z / math.sqrt(t)
}

// check_srm returns false if the holdout fraction deviates more than 5%
// from the expected value given the rollout percentage.
fn check_srm(n_holdout int, n_treatment int, rollout_pct f64) bool {
    expected := 1.0 / (1.0 + rollout_pct)
    observed := f64(n_holdout) / f64(n_holdout + n_treatment)
    return math.abs(observed - expected) / expected <= 0.05
}

// normal_sample draws a single value from N(mean, std) using Box-Muller.
fn normal_sample(mean f64, std f64) f64 {
    mut u1 := rand.f64()
    if u1 <= 0.0 { u1 = 1e-15 }
    u2 := rand.f64()
    z := math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z
}

// generate_users returns n simulated session durations.
fn generate_users(n int, mean f64, std f64) []f64 {
    mut out := []f64{len: n}
    for i in 0 .. n {
        out[i] = normal_sample(mean, std)
    }
    return out
}

fn main() {
    rand.seed([u32(42), 0])

    // Experiment parameters
    baseline_mean := 180.0   // seconds
    baseline_std  := 60.0    // seconds
    true_effect   := 6.0     // treatment truly adds 6 seconds
    mde           := 5.0     // we care about effects >= 5s
    alpha         := 0.05
    dau           := 100_000

    // ── Step 0: Pre-flight ──────────────────────────────────────────────────
    ss := experiment.sample_size_means(baseline_mean, baseline_std, mde, alpha, 0.80)
    target_holdout := ss.n_per_group
    days_needed := target_holdout / (dau / 100)  // holdout grows at 1% of DAU/day
    println('=== Pre-flight ===')
    println('  Target holdout size: ${target_holdout} users')
    println('  Holdout grows at ${dau / 100} users/day → need ~${days_needed} days')
    println('')

    // Simulate cumulative data at each rollout milestone
    // Information fractions based on holdout growth (not treatment size)
    milestones := [
        // (day, rollout_pct, holdout_n, treatment_n)
        (7,  20.0, 700,   14000),
        (14, 50.0, 1400,  70000),
        (21, 99.0, ss.n_per_group, ss.n_per_group * 99),
    ]

    // Accumulate holdout and treatment data across days
    mut holdout_data   := []f64{}
    mut treatment_data := []f64{}
    mut stopped        := false

    for milestone in milestones {
        day, rollout_pct, holdout_total, _ := milestone

        // Generate new users since last look
        new_holdout   := holdout_total - holdout_data.len
        new_treatment := holdout_total * int(rollout_pct) - treatment_data.len

        for _ in 0 .. new_holdout {
            holdout_data << normal_sample(baseline_mean, baseline_std)
        }
        for _ in 0 .. new_treatment {
            treatment_data << normal_sample(baseline_mean + true_effect, baseline_std)
        }

        println('=== Day ${day} — ${int(rollout_pct)}% rollout ===')

        // ── Step 1: SRM check ──────────────────────────────────────────────
        if !check_srm(holdout_data.len, treatment_data.len, rollout_pct) {
            println('  ⚠ SRM detected! Stopping analysis — investigate assignment bug.')
            break
        }
        println('  SRM check: OK (holdout=${holdout_data.len}, treatment=${treatment_data.len})')

        // ── Step 2: Sequential test ────────────────────────────────────────
        t := f64(holdout_data.len) / f64(target_holdout)
        boundary := obf_boundary(t, alpha)
        r := experiment.abtest(holdout_data, treatment_data)

        println('  Information fraction: ${t:.3f}')
        println('  O\'BF boundary:        |z| >= ${boundary:.2f}')
        println('  |t-statistic|:         ${math.abs(r.t_statistic):.2f}')
        println('  Treatment lift:        +${r.relative_lift * 100:.1f}%')
        println('  95% CI:               [${r.ci_lower:.1f}s, ${r.ci_upper:.1f}s]')

        if math.abs(r.t_statistic) >= boundary {
            println('  Decision: ✓ STOP — crosses O\'BF boundary. Effect confirmed.')
            stopped = true
            break
        } else {
            println('  Decision: → CONTINUE')
        }
        println('')
    }

    if !stopped {
        println('Reached final look without crossing boundary.')
    }

    // ── Step 4: Post-launch canary (after reaching 99%) ──────────────────
    println('')
    println('=== Post-launch canary check (week 5) ===')
    canary_holdout   := generate_users(500, baseline_mean, baseline_std)
    canary_treatment := generate_users(500, baseline_mean + true_effect, baseline_std)
    cr := experiment.abtest(canary_holdout, canary_treatment)
    println('  p-value: ${cr.p_value:.4f}')
    println('  Lift:    ${cr.relative_lift * 100:.1f}%')
    if cr.significant {
        println('  ⚠ Significant change vs holdout — verify intentional')
    } else {
        println('  Healthy: metric stable vs holdout')
    }
}
```

### Running the example

```bash
v run examples/rollout_sequential.v
```

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| Running a standard t-test at every rollout stage | False positive rate 3–5× above target | Use SPRT (`sprt_test`) or O'Brien-Fleming (`abtest` + `obf_boundary`) |
| Dropping holdout at 100% | No canary for regressions | Keep 1% holdout permanently |
| Skipping SRM check | Assignment bugs look like real effects | Always run SRM before any statistical test |
| Setting `mde` too small in SPRT | Very slow decisions | Set MDE to the smallest effect you would actually act on |
| Analyzing users who haven't been exposed long enough | Novelty effect inflates early results | Apply a 7-day exposure filter: only include users with ≥7 days in their group |
| Looking at the test when `continue_testing` but the trend looks good | Defeats the purpose | Trust the framework. Only stop at boundary crossings. |

---

## Which Test to Use

| Metric type | Recommended approach | vstats function |
|---|---|---|
| Binary (click, convert, sign up) | SPRT — check any time | `experiment.sprt_test` |
| Continuous (revenue, duration, latency) | Group sequential with O'Brien-Fleming | `experiment.abtest` + `obf_boundary` |
| Binary, Bayesian decision-making | Beta-Binomial posterior | `experiment.bayesian_ab_test` |
| Pre-experiment covariate available | CUPED-adjusted sequential | `experiment.cuped_test` + `obf_boundary` |
