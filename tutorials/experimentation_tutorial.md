# Experimentation Tutorial: A/B Tests, Progressive Rollouts, Pre/Post Evaluations, and DiD

**Scenario**: DataBoard is a SaaS analytics platform. The product team wants to test
a redesigned onboarding wizard they believe will increase 14-day activation rate
(completing 3+ core actions) and first-month revenue. Four different data situations
arise depending on how and when the feature ships.

| Data situation | Right method |
|---|---|
| Randomized 50/50 split, simultaneous control | A/B test |
| Gradual rollout with a permanent holdout bucket | Progressive rollout + SPRT |
| Before/after data, all users got the change | Pre/post evaluation |
| Regional rollout — some markets treated early | Difference-in-Differences |

---

## Part 1 — Classic A/B Test

Random assignment: half of new users see the old onboarding (control), half see the
new one (treatment). All users are measured simultaneously.

### Step 0 — Plan your sample size before launching

Never start an experiment without knowing when to stop. An underpowered test wastes
time; an overpowered one wastes users.

```v
module main

import vstats.experiment

fn main() {
    // Binary metric: 14-day activation rate.
    // Baseline: 32%, minimum detectable effect: +6 percentage points.
    act := experiment.sample_size_proportions(0.32, 0.06, 0.05, 0.80)
    println('Activation: ${act.n_per_group} users/group (${act.total_n} total)')
    println('  Effect size: ${act.effect_size:.3f}')
    // Activation: 445 users/group (890 total)
    //   Effect size: 0.130

    // Continuous metric: first-month revenue.
    // Baseline: $48 mean, $14 std dev. Minimum detectable effect: +$6.
    rev := experiment.sample_size_means(48.0, 14.0, 6.0, 0.05, 0.80)
    println('Revenue:    ${rev.n_per_group} users/group')
    println("  Cohen's d: ${rev.effect_size:.3f}")
    // Revenue:    87 users/group
    //   Cohen's d: 0.429
}
```

**Rule of thumb**: plan for your hardest metric. With 445 users per group you easily
cover the revenue test (needs only 87). Collect exactly 445 per group, then stop.

---

### Step 1 — Test the binary metric (activation rate)

After 3 weeks you have 1,000 users per group. Use a two-proportion z-test.

```v
module main

import vstats.experiment

fn main() {
    //   Control  (1,000 users): 320 activated  → 32.0%
    //   Treatment (1,000 users): 382 activated → 38.2%
    r := experiment.proportion_test(320, 1000, 382, 1000)

    println('Control rate:   ${r.rate_a * 100:.1f}%')
    println('Treatment rate: ${r.rate_b * 100:.1f}%')
    println('Absolute lift:  +${r.diff * 100:.1f} pp')
    println('Relative lift:  +${r.relative_lift * 100:.1f}%')
    println('z-statistic:    ${r.z_statistic:.3f}')
    println('p-value:        ${r.p_value:.4f}')
    println('95%% CI:         [+${r.ci_lower * 100:.2f}pp, +${r.ci_upper * 100:.2f}pp]')
    println('Significant:    ${r.significant}')
    // Control rate:   32.0%
    // Treatment rate: 38.2%
    // Absolute lift:  +6.2 pp
    // Relative lift:  +19.4%
    // z-statistic:    2.905
    // p-value:        0.0037
    // 95% CI:         [+2.03pp, +10.37pp]
    // Significant:    true
}
```

**What the numbers tell you**: activation improved by 6.2pp (p = 0.004). The 95% CI
excludes zero, and the lower bound (+2.0pp) is still a meaningful gain for the
business. Ship it.

---

### Step 2 — Test the continuous metric (revenue)

```v
module main

import vstats.experiment

fn main() {
    // First-month revenue per user (USD). Showing 20 users per group;
    // in practice pass all observations from your full dataset.
    control := [36.0, 52.0, 44.0, 61.0, 31.0, 55.0, 38.0, 48.0, 57.0, 42.0,
                67.0, 33.0, 51.0, 44.0, 58.0, 39.0, 45.0, 63.0, 47.0, 53.0]
    treatment := [52.0, 68.0, 60.0, 77.0, 47.0, 71.0, 54.0, 64.0, 73.0, 58.0,
                  83.0, 49.0, 67.0, 60.0, 74.0, 55.0, 61.0, 79.0, 63.0, 69.0]

    r := experiment.abtest(control, treatment)

    println('Control:   \$${r.control_mean:.2f} ± \$${r.control_std:.2f}')
    println('Treatment: \$${r.treatment_mean:.2f} ± \$${r.treatment_std:.2f}')
    println('Lift:      +\$${r.treatment_mean - r.control_mean:.2f} (+${r.relative_lift * 100:.1f}%%)')
    println("Cohen's d: ${r.effect_size:.3f}")
    println('t:         ${r.t_statistic:.3f}')
    println('p-value:   ${r.p_value:.6f}')
    println('95%% CI:    [\$${r.ci_lower:.2f}, \$${r.ci_upper:.2f}]')
    // Control:   $48.20 ± $10.13
    // Treatment: $64.20 ± $10.13
    // Lift:      +$16.00 (+33.2%)
    // Cohen's d: 1.580
    // t:         4.993
    // p-value:   0.000001
    // 95% CI:    [$9.72, $22.28]
}
```

---

### Step 3 — Reduce noise with CUPED

If you collected pre-experiment revenue for each user (revenue from the month
*before* the test started), you can use CUPED to reduce variance. It removes
individual-level baseline differences, shrinking the confidence interval without
changing the expected effect estimate.

```v
module main

import vstats.experiment

fn main() {
    // Post-experiment outcomes (same users as Step 2)
    y_ctrl  := [36.0, 52.0, 44.0, 61.0, 31.0, 55.0, 38.0, 48.0, 57.0, 42.0,
                67.0, 33.0, 51.0, 44.0, 58.0, 39.0, 45.0, 63.0, 47.0, 53.0]
    y_treat := [52.0, 68.0, 60.0, 77.0, 47.0, 71.0, 54.0, 64.0, 73.0, 58.0,
                83.0, 49.0, 67.0, 60.0, 74.0, 55.0, 61.0, 79.0, 63.0, 69.0]

    // Pre-experiment revenue for the same users (month before test started).
    // Both groups have similar means — confirms randomization worked.
    pre_ctrl  := [30.0, 46.0, 38.0, 55.0, 25.0, 49.0, 32.0, 42.0, 51.0, 36.0,
                  61.0, 27.0, 45.0, 38.0, 52.0, 33.0, 39.0, 57.0, 41.0, 47.0]
    pre_treat := [31.0, 47.0, 40.0, 56.0, 27.0, 50.0, 34.0, 43.0, 52.0, 38.0,
                  62.0, 29.0, 46.0, 40.0, 53.0, 35.0, 41.0, 58.0, 43.0, 49.0]
    // pre_ctrl mean ≈ $42.2   pre_treat mean ≈ $43.7 (no group difference pre-test ✓)

    c := experiment.cuped_test(y_ctrl, y_treat, pre_ctrl, pre_treat)

    println('Theta (regression coeff): ${c.theta:.3f}')
    println('Variance reduction:       ${c.variance_reduction * 100:.1f}%%')
    println('Adjusted result:')
    println('  Treatment – Control: \$${c.adjusted_result.treatment_mean - c.adjusted_result.control_mean:.2f}')
    println('  p-value:             ${c.adjusted_result.p_value:.6f}')
    // Theta (regression coeff): 1.038
    // Variance reduction:       66.1%
    // Adjusted result:
    //   Treatment – Control: $16.00
    //   p-value:             0.000000
}
```

**What the numbers tell you**: CUPED reduced variance by 66%, shrinking the
confidence interval substantially. The effect estimate is unchanged ($16.00) because
CUPED removes noise but not signal. Use CUPED whenever you have pre-experiment data —
it is always at least as powerful as a standard t-test.

---

### Step 4 — Bayesian alternative

If you prefer to reason about probability rather than p-values, use the Bayesian test.
It works directly on conversion counts.

```v
module main

import vstats.experiment

fn main() {
    // Same counts as Step 1.
    r := experiment.bayesian_ab_test(320, 1000, 382, 1000)

    println('Posterior rate A: ${r.posterior_mean_a * 100:.1f}%  95%% CI [${r.ci_lower_a * 100:.1f}%%, ${r.ci_upper_a * 100:.1f}%%]')
    println('Posterior rate B: ${r.posterior_mean_b * 100:.1f}%  95%% CI [${r.ci_lower_b * 100:.1f}%%, ${r.ci_upper_b * 100:.1f}%%]')
    println('P(B beats A):     ${r.prob_b_beats_a * 100:.1f}%%')
    println('Expected loss if you deploy B:  ${r.expected_loss_b * 100:.3f} pp')
    println('Expected loss if you keep A:    ${r.expected_loss_a * 100:.3f} pp')
    // Posterior rate A: 32.0%  95% CI [29.1%, 34.9%]
    // Posterior rate B: 38.2%  95% CI [35.2%, 41.2%]
    // P(B beats A):     99.7%
    // Expected loss if you deploy B:  0.002 pp
    // Expected loss if you keep A:    6.187 pp
}
```

**What the numbers tell you**: there is a 99.7% probability that the new onboarding
is better. Deploying it risks losing a negligible 0.002pp on average; keeping the
old flow risks forgoing 6.2pp on average. Deploy B.

| Decision framework | Threshold | Conclusion |
|---|---|---|
| Frequentist | p < 0.05 | Ship (p = 0.004) |
| Bayesian | P(B > A) > 95% | Ship (99.7%) |
| Expected loss | loss_B < 0.5pp | Ship (0.002pp) |

---

## Part 2 — Progressive Rollout with Holdout Group

Instead of a simultaneous 50/50 split, the feature ships gradually:
1% → 5% → 20% → 100%. A permanent 1% holdout bucket never receives the feature.
This lets you detect problems early and avoid exposing all users to a bad change.

> **For the full theory** (O'Brien-Fleming boundaries, SRM checks, joint metric
> decisions), see `sequential_rollout.md`. This section focuses on the code.

**Bucket assignment** (stable across sessions):
```
hash(user_id) % 100:
  Bucket  0     → holdout  (1%, never gets the feature)
  Buckets 1–N   → treatment (grows as rollout_pct increases)
  Buckets N+1–99 → not yet exposed
```

### Sequential testing with SPRT

For binary metrics (activation rate), use `sprt_test`. You can call it at any
point with cumulative counts — it will tell you whether to stop.

```v
module main

import vstats.experiment

// Call at any rollout checkpoint with cumulative totals.
fn check(label string, s_hold int, n_hold int, s_treat int, n_treat int) {
    cfg := experiment.SPRTConfig{
        alpha: 0.05   // tolerate 5% false positive rate
        beta:  0.20   // tolerate 20% false negative rate (80% power)
        mde:   0.05   // detect a +5 percentage-point change
    }
    r := experiment.sprt_test(s_hold, n_hold, s_treat, n_treat, cfg)
    println('${label}')
    println('  Holdout: ${r.rate_a * 100:.1f}% (${n_hold} users)  Treatment: ${r.rate_b * 100:.1f}% (${n_treat} users)')
    println('  LLR: ${r.log_likelihood_ratio:.2f}  boundaries [${r.lower_boundary:.2f}, ${r.upper_boundary:.2f}]')
    println('  Decision: ${r.decision}')
}

fn main() {
    // Day 1  — early look. Treatment ramped to 1%; n=100 per group.
    check('Day 1',  32, 100,  38, 100)
    // Day 1
    //   Holdout: 32.0% (100 users)  Treatment: 38.0% (100 users)
    //   LLR: 0.78  boundaries [-1.56, 2.77]
    //   Decision: continue_testing

    // Day 3  — more data, still accumulating.
    check('Day 3',  96, 300, 114, 300)
    // Day 3
    //   Holdout: 32.0% (300 users)  Treatment: 38.0% (300 users)
    //   LLR: 2.35  boundaries [-1.56, 2.77]
    //   Decision: continue_testing

    // Day 5  — enough evidence.
    check('Day 5', 160, 500, 190, 500)
    // Day 5
    //   Holdout: 32.0% (500 users)  Treatment: 38.0% (500 users)
    //   LLR: 3.91  boundaries [-1.56, 2.77]
    //   Decision: reject_null → ramp up treatment
}
```

**Decision key**:

| Decision | Action |
|---|---|
| `continue_testing` | Keep accumulating data |
| `reject_null` | Effect confirmed — accelerate rollout |
| `accept_null` | No meaningful effect detected — ship or rollback (business decision) |

**When SPRT says `reject_null` on a positive metric**: ramp treatment to the next
stage (e.g., 1% → 20%). Do not disable the holdout — it becomes your long-term
canary after full launch.

---

## Part 3 — Pre/Post Evaluation

A re-engagement email campaign was sent to **all** active users at once — there is
no concurrent control group. You compare revenue in the 4 weeks before vs 4 weeks
after the campaign.

**Important limitation**: a before/after comparison cannot separate the campaign's
effect from anything else that changed at the same time (seasonality, product
changes, market conditions). Use this method when randomization was impossible,
and state its limits clearly.

### Step 1 — Naive before/after test

```v
module main

import vstats.experiment

fn main() {
    // Weekly revenue per user (40 users measured before and after).
    // Real data: export pre_period and post_period from your data warehouse.
    pre := [28.0, 45.0, 38.0, 52.0, 31.0, 47.0, 34.0, 41.0, 55.0, 36.0,
            62.0, 29.0, 44.0, 38.0, 51.0, 33.0, 40.0, 58.0, 43.0, 48.0,
            26.0, 44.0, 37.0, 51.0, 30.0, 46.0, 33.0, 40.0, 54.0, 35.0,
            61.0, 28.0, 43.0, 37.0, 50.0, 32.0, 39.0, 57.0, 42.0, 47.0]

    post := [40.0, 59.0, 52.0, 68.0, 44.0, 62.0, 48.0, 55.0, 70.0, 50.0,
             78.0, 43.0, 59.0, 52.0, 67.0, 46.0, 54.0, 74.0, 57.0, 63.0,
             38.0, 58.0, 51.0, 67.0, 43.0, 61.0, 47.0, 54.0, 69.0, 49.0,
             77.0, 42.0, 58.0, 51.0, 66.0, 45.0, 53.0, 73.0, 56.0, 62.0]

    // Treat "before" as control and "after" as treatment.
    r := experiment.abtest(pre, post)

    println('Before: \$${r.control_mean:.2f}/user  After: \$${r.treatment_mean:.2f}/user')
    println('Observed change: +\$${r.treatment_mean - r.control_mean:.2f} (+${r.relative_lift * 100:.1f}%%)')
    println('p-value: ${r.p_value:.5f}  Significant: ${r.significant}')
    println('95%% CI: [\$${r.ci_lower:.2f}, \$${r.ci_upper:.2f}]')
    // Before: $42.13/user  After: $56.53/user
    // Observed change: +$14.40 (+34.2%)
    // p-value: 0.00000  Significant: true
    // 95% CI: [$11.26, $17.54]
}
```

**What the numbers tell you**: users spent $14.40 more per week after the campaign
(p < 0.001). However, this includes any concurrent trend — holiday season, a
separate product launch, or natural growth. The number is an upper bound on the
campaign's true causal effect.

---

### Step 2 — Remove individual-level noise with CUPED

If you have data from an even-earlier period (weeks −8 to −4, before the campaign
was planned), you can use it as a CUPED covariate. This removes user-level
baseline differences without changing the effect estimate.

```v
module main

import vstats.experiment

fn main() {
    pre := [28.0, 45.0, 38.0, 52.0, 31.0, 47.0, 34.0, 41.0, 55.0, 36.0,
            62.0, 29.0, 44.0, 38.0, 51.0, 33.0, 40.0, 58.0, 43.0, 48.0,
            26.0, 44.0, 37.0, 51.0, 30.0, 46.0, 33.0, 40.0, 54.0, 35.0,
            61.0, 28.0, 43.0, 37.0, 50.0, 32.0, 39.0, 57.0, 42.0, 47.0]

    post := [40.0, 59.0, 52.0, 68.0, 44.0, 62.0, 48.0, 55.0, 70.0, 50.0,
             78.0, 43.0, 59.0, 52.0, 67.0, 46.0, 54.0, 74.0, 57.0, 63.0,
             38.0, 58.0, 51.0, 67.0, 43.0, 61.0, 47.0, 54.0, 69.0, 49.0,
             77.0, 42.0, 58.0, 51.0, 66.0, 45.0, 53.0, 73.0, 56.0, 62.0]

    // Revenue from two months before the campaign (same 40 users).
    // Use this as a covariate to control for user-level revenue differences.
    pre_pre := [24.0, 41.0, 34.0, 48.0, 27.0, 43.0, 30.0, 37.0, 51.0, 32.0,
                58.0, 25.0, 40.0, 34.0, 47.0, 29.0, 36.0, 54.0, 39.0, 44.0,
                22.0, 40.0, 33.0, 47.0, 26.0, 42.0, 29.0, 36.0, 50.0, 31.0,
                57.0, 24.0, 39.0, 33.0, 46.0, 28.0, 35.0, 53.0, 38.0, 43.0]

    // Since it's the same users in both periods, pass pre_pre as the covariate
    // for both the "control" (before) and "treatment" (after) groups.
    c := experiment.cuped_test(pre, post, pre_pre, pre_pre)

    println('Variance reduction: ${c.variance_reduction * 100:.1f}%%')
    println('Adjusted change:    +\$${c.adjusted_result.treatment_mean - c.adjusted_result.control_mean:.2f}')
    println('Adjusted p-value:   ${c.adjusted_result.p_value:.5f}')
    // Variance reduction: 94.3%
    // Adjusted change:    +$14.40
    // Adjusted p-value:   0.00000
}
```

**What CUPED does and doesn't do**: it removes user-level heterogeneity (big spenders
vs small spenders), making the test more sensitive. It does **not** remove
confounders — if your company also ran a discount promotion in the post period,
that contamination remains in the estimate.

| What CUPED removes | What CUPED cannot remove |
|---|---|
| High-vs-low spender variance | Concurrent product changes |
| Seasonal pattern at user level | Holiday / macro trends |
| Historical usage correlation | Any other event at the same time |

When possible, upgrade to Difference-in-Differences (Part 4) if you can find any
group that was not affected by the campaign.

---

## Part 4 — Difference-in-Differences

DataBoard launched the new onboarding in 6 cities (treated group) but not yet
in the other 6 cities (control group), due to a phased regional rollout.
You have weekly new-user revenue for all 12 cities across 5 time periods.

DiD estimates the causal effect by asking: *did the treated cities grow more than
the control cities grew over the same period?*

```
DiD = (Treated_post − Treated_pre) − (Control_post − Control_pre)
                          ↑                         ↑
                 "treatment effect            "what would have
                  + natural trend"             happened anyway"
```

### Step 1 — Check the parallel trends assumption

DiD is only valid if, before treatment, both groups trended the same way.
Test this with 3 pre-periods.

```v
module main

import vstats.experiment

fn main() {
    // Weekly new-user revenue by city, 3 pre-periods.
    // 18 observations per group (3 periods × 6 cities).

    // Treated cities — periods 1, 2, 3 (period 3 is the reference, just before launch)
    y_treat_pre := [
        8300.0, 8500.0, 8100.0, 8600.0, 8400.0, 8200.0,  // period 1
        8400.0, 8600.0, 8200.0, 8700.0, 8500.0, 8300.0,  // period 2
        8500.0, 8700.0, 8300.0, 8800.0, 8600.0, 8400.0,  // period 3
    ]

    // Control cities — same 3 periods
    y_ctrl_pre := [
        8000.0, 8300.0, 7900.0, 8500.0, 8100.0, 8200.0,  // period 1
        8100.0, 8400.0, 8000.0, 8600.0, 8200.0, 8300.0,  // period 2
        8200.0, 8500.0, 8100.0, 8700.0, 8300.0, 8400.0,  // period 3
    ]

    time_pre := [1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2,  3, 3, 3, 3, 3, 3]

    pt := experiment.test_parallel_trends(y_treat_pre, y_ctrl_pre, time_pre)

    println('Treated pre-slope:  +\$${pt.slope_treated:.0f}/period')
    println('Control pre-slope:  +\$${pt.slope_control:.0f}/period')
    println('Slope difference:   \$${pt.slope_difference:.1f}')
    println('p-value:            ${pt.p_value:.3f}')
    println('Parallel trends OK: ${pt.parallel_trends_hold}')
    // Treated pre-slope:  +$100/period
    // Control pre-slope:  +$100/period
    // Slope difference:   $0.0
    // p-value:            1.000
    // Parallel trends OK: true
}
```

**Interpretation**: both groups grew at the same rate before treatment (+$100/period).
The slope difference is $0 and p = 1.0, confirming the assumption holds. If this
test fails (p < 0.05), DiD is not valid — consider a different estimator or
look for confounders.

---

### Step 2 — Simple 2×2 DiD estimate

Use the period just before and just after launch.

```v
module main

import vstats.experiment

fn main() {
    // Period just before launch (period 3 from the parallel trends data above)
    y_treat_before := [8500.0, 8700.0, 8300.0, 8800.0, 8600.0, 8400.0]
    y_ctrl_before  := [8200.0, 8500.0, 8100.0, 8700.0, 8300.0, 8400.0]

    // Period just after launch (new onboarding is live in treated cities)
    y_treat_after := [10100.0, 10300.0, 9900.0, 10500.0, 10200.0, 10000.0]
    y_ctrl_after  := [8300.0,  8600.0,  8200.0,  8800.0,  8400.0,  8500.0]

    r := experiment.did_2x2(y_treat_before, y_treat_after, y_ctrl_before, y_ctrl_after)

    println('Treated cities:  +\$${r.treated_change:.0f}  (before → after)')
    println('Control cities:  +\$${r.control_change:.0f}  (before → after)')
    println('DiD effect:       \$${r.did_effect:.0f}')
    println('Std error:        \$${r.se:.0f}')
    println('t-statistic:     ${r.t_statistic:.2f}')
    println('p-value:         ${r.p_value:.4f}')
    println('95%% CI:          [\$${r.ci_lower:.0f}, \$${r.ci_upper:.0f}]')
    // Treated cities:  +$1617  (before → after)
    // Control cities:  +$100   (before → after)
    // DiD effect:       $1517
    // Std error:        $188
    // t-statistic:     8.07
    // p-value:         0.0000
    // 95% CI:          [$1148, $1885]
}
```

**What the numbers tell you**: the new onboarding caused an additional $1,517/week in
new-user revenue for treated cities, compared to what control cities gained naturally
in the same period (p < 0.001). The 95% CI [$1,148–$1,885] is tight and entirely
positive.

---

### Step 3 — DiD regression (preferred for panel data)

The 2×2 DiD uses only two periods. If you have a full panel of observations,
regression gives cleaner standard errors and lets you add covariates.

```v
module main

import vstats.experiment

fn main() {
    // Panel data: 24 observations (6 treated + 6 control cities × 2 periods).
    // y: outcome, group: 0=control 1=treated, time: 0=pre 1=post
    y := [
        // Treated, pre  (group=1, time=0)
        8500.0, 8700.0, 8300.0, 8800.0, 8600.0, 8400.0,
        // Treated, post (group=1, time=1)
        10100.0, 10300.0, 9900.0, 10500.0, 10200.0, 10000.0,
        // Control, pre  (group=0, time=0)
        8200.0, 8500.0, 8100.0, 8700.0, 8300.0, 8400.0,
        // Control, post (group=0, time=1)
        8300.0, 8600.0, 8200.0, 8800.0, 8400.0, 8500.0,
    ]

    group := [
        1, 1, 1, 1, 1, 1,   // treated pre
        1, 1, 1, 1, 1, 1,   // treated post
        0, 0, 0, 0, 0, 0,   // control pre
        0, 0, 0, 0, 0, 0,   // control post
    ]

    time := [
        0, 0, 0, 0, 0, 0,   // pre
        1, 1, 1, 1, 1, 1,   // post
        0, 0, 0, 0, 0, 0,   // pre
        1, 1, 1, 1, 1, 1,   // post
    ]

    // No extra covariates for this example.
    r := experiment.did_regression(y, [][]f64{}, group, time)

    println('DiD coefficient: \$${r.did_coefficient:.0f}/week')
    println('Std error:       \$${r.did_se:.0f}')
    println('t-statistic:    ${r.did_t_stat:.2f}')
    println('p-value:        ${r.did_p_value:.4f}')
    println('95%% CI:         [\$${r.did_ci_lower:.0f}, \$${r.did_ci_upper:.0f}]')
    println('R²:             ${r.r_squared:.3f}')
    // DiD coefficient: $1517/week
    // Std error:       $170
    // t-statistic:    8.92
    // p-value:        0.0000
    // 95% CI:         [$1183, $1850]
    // R²:             0.975
}
```

---

### Step 4 — Event study (did the effect emerge at the right time?)

An event study plots the DiD effect for each period relative to the launch. It
answers two questions:
1. Was there a pre-existing trend? (pre-treatment periods should be near zero)
2. Did the effect appear *when the feature launched* and not before? (post-treatment
   periods should show the effect)

```v
module main

import vstats.experiment

fn main() {
    // 60 observations: 12 cities × 5 periods.
    // Relative time: -3, -2, -1 (reference, just before launch), 0, 1.
    // Treatment group: cities 1–6, control group: cities 7–12.

    y := [
        // t=-3: treated (6 cities)
        8300.0, 8500.0, 8100.0, 8600.0, 8400.0, 8200.0,
        // t=-3: control (6 cities)
        8000.0, 8300.0, 7900.0, 8500.0, 8100.0, 8200.0,
        // t=-2: treated
        8400.0, 8600.0, 8200.0, 8700.0, 8500.0, 8300.0,
        // t=-2: control
        8100.0, 8400.0, 8000.0, 8600.0, 8200.0, 8300.0,
        // t=-1: treated (reference period)
        8500.0, 8700.0, 8300.0, 8800.0, 8600.0, 8400.0,
        // t=-1: control (reference period)
        8200.0, 8500.0, 8100.0, 8700.0, 8300.0, 8400.0,
        // t=0:  treated (feature launched)
        10100.0, 10300.0, 9900.0, 10500.0, 10200.0, 10000.0,
        // t=0:  control (no change)
        8300.0, 8600.0, 8200.0, 8800.0, 8400.0, 8500.0,
        // t=1:  treated (effect sustained)
        10300.0, 10500.0, 10100.0, 10700.0, 10400.0, 10200.0,
        // t=1:  control
        8400.0, 8700.0, 8300.0, 8900.0, 8500.0, 8600.0,
    ]

    group := [
        1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0,  // t=-3
        1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0,  // t=-2
        1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0,  // t=-1 (reference)
        1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0,  // t=0
        1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0,  // t=1
    ]

    relative_time := [
        -3, -3, -3, -3, -3, -3,  -3, -3, -3, -3, -3, -3,
        -2, -2, -2, -2, -2, -2,  -2, -2, -2, -2, -2, -2,
        -1, -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,   1,  1,  1,  1,  1,  1,
    ]

    es := experiment.event_study(y, group, relative_time)

    println('Period  Effect        95%% CI                  p-value')
    println('------  ------        -------                  -------')
    for i, t in es.relative_times {
        println('  t=${t:2d}   \$${es.effects[i]:6.0f}    [\$${es.ci_lowers[i]:6.0f}, \$${es.ci_uppers[i]:6.0f}]    ${es.p_values[i]:.4f}')
    }
    // Period  Effect        95% CI                  p-value
    // ------  ------        -------                  -------
    //   t=-3   $    0    [$   -369, $    369]    1.0000
    //   t=-2   $    0    [$   -369, $    369]    1.0000
    //   t= 0   $ 1517    [$ 1148,  $ 1885]       0.0000
    //   t= 1   $ 1617    [$ 1248,  $ 1985]       0.0000
}
```

**What the numbers tell you**: pre-treatment effects (t=−3 and t=−2) are zero —
the parallel trends assumption holds in the data. The treatment effect appears
precisely at t=0 (when the feature launched) and grows slightly at t=1 as users
complete onboarding. This "correct timing" pattern is strong evidence that the
effect is causal.

A well-executed event study looks like this:

```
Revenue effect
   $2000 |                              ●  ●
   $1500 |                         ●
   $1000 |
    $500 |
      $0 |  ●  ●   (reference)
   −$500 |
         +--+--+--------+--+--
          -3 -2  -1  |  0  1
                 launch
```

Flat pre-trend, sharp jump at launch, sustained post-treatment = strong evidence.

---

## Choosing the Right Method

| Question | Answer | Method |
|---|---|---|
| Can you randomly assign users? | Yes, simultaneously | **A/B test** |
| Must you ship gradually? | Yes, feature flag | **Progressive rollout + SPRT** |
| All users got the change at once? | No concurrent control | **Pre/Post** (with caveats) |
| Some units treated early, others later? | Panel data available | **DiD** |
| Have pre-experiment data per user? | Yes | Add **CUPED** to A/B or Pre/Post |
| Want probability statements? | Yes | **Bayesian A/B** |

**Common pitfalls**:

| Mistake | Consequence | Fix |
|---|---|---|
| Peeking at p-values daily in a fixed-horizon test | Up to 37% false positive rate | Use SPRT or set a single analysis date |
| Pre/post without checking for concurrent events | Overestimate effect | Check calendar; prefer DiD if any untreated group exists |
| DiD without parallel trends test | Invalid causal claim | Always run `test_parallel_trends` first |
| Sample size too small before launch | Underpowered, noisy results | Use `sample_size_proportions` / `sample_size_means` |
| CUPED with weak covariate (r < 0.3) | Near-zero variance reduction | Check correlation; pick a stronger covariate |

---

## Quick Reference

```v
import vstats.experiment

// Sample size planning
experiment.sample_size_proportions(baseline_rate, mde, alpha, power)
experiment.sample_size_means(baseline_mean, baseline_std, mde_absolute, alpha, power)

// A/B test
experiment.proportion_test(successes_a, n_a, successes_b, n_b)
experiment.abtest(control []f64, treatment []f64)
experiment.cuped_test(y_ctrl, y_treat, pre_ctrl, pre_treat)
experiment.bayesian_ab_test(successes_a, n_a, successes_b, n_b)

// Progressive rollout
experiment.sprt_test(successes_a, n_a, successes_b, n_b, SPRTConfig{ mde: 0.05 })

// DiD
experiment.test_parallel_trends(y_treated_pre, y_control_pre, time_pre)
experiment.did_2x2(y_treat_before, y_treat_after, y_ctrl_before, y_ctrl_after)
experiment.did_regression(y, x, group, time)
experiment.event_study(y, group, relative_time)
```
