# Chatbot Feature Rollout: Sequential Testing for Revenue Per Report and Activation Rate

You are rolling out a chatbot to your SaaS product. Two primary metrics will
determine whether the rollout continues, pauses, or rolls back:

- **Revenue Per Report (RPR)** — average revenue generated per completed report,
  measured over the first **7 days** after account creation. Continuous metric.
- **Activation Rate (AR)** — percentage of accounts that complete at least one
  report within **14 days** of account creation. Binary metric.

Rollout plan:
```
Permanent holdout: 1%
Treatment stages:  1% → 5% → 30% → 60% → 99%
```

---

## The Measurement Windows Problem

Both metrics have a **minimum observation window** before they can be measured:

| Metric             | Window  | Reason                                          |
| ------------------ | ------- | ----------------------------------------------- |
| Revenue Per Report | 7 days  | Users need time to discover and use the product |
| Activation Rate    | 14 days | Definition: completed report within 14 days     |

This means you cannot analyze a user the moment they join the experiment. A
user who created their account yesterday cannot be counted toward your 14-day
Activation Rate today. Only **matured users** — those who have been in their
group long enough — are valid observations.

```
Day 0:  User A enters holdout                    [not yet countable for either metric]
Day 7:  User A has 7 days of revenue data        [countable for RPR]
Day 14: User A's 14-day activation window closes [countable for AR]
```

Every analysis in this tutorial filters for matured users. Using un-matured
users is the most common mistake in rollout analysis — it underestimates both
metrics and inflates variance.

**Activation Rate is the binding constraint.** It requires twice the maturation
time of RPR, so the full experiment timeline is driven by AR, not RPR.

---

## Step 0 — Pre-flight: Sample Sizes and Timeline

Before launching, compute how many matured holdout users you need and how long
that will take. Assume **10,000 new accounts per day** (DAU) and a 1% holdout,
so the holdout grows at **100 matured users per day** (after the respective
window closes).

### Activation Rate (binary)

```v
module main

import vstats.experiment

fn main() {
    // Baseline: 20% of new accounts complete a report within 14 days.
    // MDE: detect a +3 percentage-point improvement (20% → 22%).
    ss_ar := experiment.sample_size_proportions(0.20, 0.01, 0.05, 0.80)

    // Holdout grows at 100 matured accounts/day (after the 14-day window).
    days_collecting := ss_ar.n_per_group / 100
    total_days := 14 + days_collecting   // 14-day maturation lag + collection
    println('AR target holdout: ${ss_ar.n_per_group} matured accounts')
    println('  → ~${days_collecting} days collecting + 14-day lag = ${total_days} days total')
    // AR target holdout: 4034 matured accounts
    // → ~40 days collecting + 14-day lag = 54 days total
}
```

### Revenue Per Report (continuous)

```v
module main

import vstats.experiment

fn main() {
    // Baseline: $47.00 per report, std $2.00 (estimated from historical data).
    // MDE: detect a $1.50 change per report.
    ss_rpr := experiment.sample_size_means(47.0, 5.0, 1.50, 0.05, 0.80)

    days_collecting := ss_rpr.n_per_group / 100
    total_days := 7 + days_collecting   // 7-day maturation lag + collection
    println('RPR target holdout: ${ss_rpr.n_per_group} matured accounts')
    println('  → ~${days_collecting} days collecting + 7-day lag = ${total_days} days total')
    // RPR target holdout: 2258 matured accounts
    // → ~23 days collecting + 7-day lag = 30 days total
}
```

### Derived timeline

AR needs 54 days; RPR needs 30 days. AR is the binding constraint.

Align your rollout milestones with when matured holdout counts reach meaningful
information fractions of the AR target (4034):

| Day | Action                        | Matured holdout (AR) | Info fraction t |
| --- | ----------------------------- | -------------------- | --------------- |
| 0   | Launch 1% treatment           | 0                    | —               |
| 14  | First analysis → ramp to 5%   | 1,400                | 0.35            |
| 21  | Second analysis → ramp to 30% | 2,100                | 0.52            |
| 35  | Third analysis → ramp to 60%  | 3,500                | 0.87            |
| 50  | Fourth analysis → ramp to 99% | 5,000                | 1.00            |
| 60  | Final analysis                | 6,000                | 1.00 (capped)   |

You have **4 pre-planned analysis points**. You will use an O'Brien-Fleming
boundary to allocate your α budget across these looks for RPR, and SPRT for AR.

---

## Step 1 — SRM Check (Run Before Every Analysis)

A sample ratio mismatch invalidates the experiment. Run this before touching
any test result.

```v
module main

import math

// srm_check verifies that the holdout fraction matches the expected ratio.
// rollout_pct: current treatment percentage (e.g. 5.0 for 5%)
// Returns false and prints a warning if the mismatch exceeds 5% relative.
fn srm_check(n_holdout int, n_treatment int, rollout_pct f64) bool {
    // Expected holdout fraction of the combined (holdout + treatment) pool:
    //   1 / (1 + rollout_pct)
    expected := 1.0 / (1.0 + rollout_pct)
    observed := f64(n_holdout) / f64(n_holdout + n_treatment)
    deviation := math.abs(observed - expected) / expected

    if deviation > 0.05 {
        println('!! SRM DETECTED')
        println('   Expected holdout fraction: ${expected:.4f}')
        println('   Observed holdout fraction: ${observed:.4f}')
        println('   Relative deviation:        ${deviation * 100:.1f}%')
        println('   DO NOT interpret test results. Fix the assignment bug first.')
        return false
    }
    println('SRM check passed (${deviation * 100:.2f}% deviation)')
    return true
}

fn main() {
    // Day 14, 5% treatment rollout.
    // Expected: 1400 holdout, 7000 treatment.
    // Observed: 1385 holdout, 7012 treatment (small rounding difference — OK).
    if !srm_check(1385, 7012, 5.0) {
        return
    }
}
```

Common causes of SRM in chatbot rollouts: the chatbot UI only loads on certain
browsers (treatment users with unsupported browsers silently fall through to
control), bot filtering applied differently across groups, or the chatbot
triggering an additional server-side session that inflates treatment event
counts.

---

## Step 2 — Activation Rate: Sequential Testing with SPRT

Activation Rate is binary (activated: yes or no), so use `sprt_test`. It is
stateless and always-valid — you pass cumulative matured totals and it tells
you whether to stop.

**Critical:** only pass users whose 14-day window has **closed**. A user who
signed up 10 days ago has not yet had the chance to activate — including them
would dilute the signal.

```v
module main

import vstats.experiment

fn check_activation_rate(
    holdout_matured    int   // accounts in holdout with account_age >= 14 days
    holdout_activated  int   // of those, how many completed a report
    treatment_matured  int   // same for treatment group
    treatment_activated int
    analysis_day       int
) {
    cfg := experiment.SPRTConfig{
        alpha: 0.05   // false positive tolerance
        beta:  0.20   // false negative tolerance (= 1 - 0.80 power)
        mde:   0.03   // detect +3pp or worse: 35% → 38% or 35% → 32%
    }

    r := experiment.sprt_test(
        holdout_activated, holdout_matured,
        treatment_activated, treatment_matured,
        cfg
    )

    println('=== Activation Rate — Day ${analysis_day} ===')
    println('  Holdout:   ${holdout_activated}/${holdout_matured} = ' +
            '${r.rate_a * 100:.1f}%')
    println('  Treatment: ${treatment_activated}/${treatment_matured} = ' +
            '${r.rate_b * 100:.1f}%')
    lift := (r.rate_b - r.rate_a) / r.rate_a * 100
    println('  Relative lift: ${lift:.1f}%')
    println('  Log-likelihood ratio: ${r.log_likelihood_ratio:.3f}')
    println('  Upper boundary (reject H0): +${r.upper_boundary:.3f}')
    println('  Lower boundary (accept H0):  ${r.lower_boundary:.3f}')

    match r.decision {
        .reject_null {
            if r.rate_b > r.rate_a {
                println('  Decision: STOP — positive effect confirmed. Safe to accelerate.')
            } else {
                println('  Decision: STOP — NEGATIVE effect detected. Pause rollout immediately.')
            }
        }
        .accept_null {
            println('  Decision: STOP (futility) — chatbot has no meaningful effect on activation.')
        }
        .continue_testing {
            println('  Decision: CONTINUE — insufficient evidence. Proceed to next rollout stage.')
        }
    }
}

fn main() {
    // Day 14: 1,400 matured holdout accounts, 7,000 matured treatment accounts.
    // Holdout activated: 490 (35.0%). Treatment activated: 2,520 (36.0%).
    check_activation_rate(1400, 490, 7000, 2520, 14)
    println('')
    // Day 21: 2,100 matured holdout, 21,000 matured treatment.
    // Holdout activated: 735 (35.0%). Treatment activated: 7,980 (38.0%).
    check_activation_rate(2100, 735, 21000, 7980, 21)
}
```

**How SPRT handles the imbalanced design:** The 1% holdout vs 5%+ treatment
creates a large size ratio. SPRT handles this correctly — the likelihood ratio
uses both group sizes in the calculation. Power is dominated by the smaller
group (holdout), so you are effectively constrained by the holdout size, not
the treatment side.

**What `mde` means for SPRT:** It defines both the "minimum effect worth
detecting" and where H1 is placed. A user completing exactly `mde` more
activations than baseline triggers the upper boundary. Set it to the smallest
change you would actually act on — here, 3 percentage points.

---

## Step 3 — Revenue Per Report: Group Sequential Testing

RPR is a continuous per-user metric. Compute it as:

```
rpr_i = total_revenue_i / reports_completed_i
```

for every user `i` in the matured holdout or treatment cohort who completed at
least one report (i.e. `reports_completed_i >= 1`). Users who completed zero
reports are excluded — they have no defined RPR.

> **Coupling note:** If AR increased, more users completed reports. Some of
> those newly-activated users may have different revenue profiles than users
> who would have activated without the chatbot. Interpret RPR alongside AR:
> an RPR increase on top of an AR increase is strong evidence of value.
> An RPR decrease alongside an AR increase may indicate the chatbot brings in
> lower-value users — worth investigating before shipping.

Use `abtest` at the four pre-planned checkpoints with an O'Brien-Fleming
boundary to avoid inflating the false positive rate.

```v
module main

import vstats.experiment
import vstats.prob
import math

// obf_boundary returns the critical |z| value at information fraction t.
// Use alpha = 0.05 for a two-sided test targeting 5% false positive rate.
// If |result.t_statistic| >= obf_boundary(t, alpha): stop, effect is significant.
fn obf_boundary(t f64, alpha f64) f64 {
    // O'Brien-Fleming spending: z_critical(t) = z_{alpha/2} / sqrt(t)
    // Conservative early (large boundary), nearly standard at t=1.
    z := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
    return z / math.sqrt(t)
}

fn check_revenue_per_report(
    holdout_rpr   []f64   // per-user RPR for matured holdout (len = matured holdout count)
    treatment_rpr []f64   // per-user RPR for matured treatment
    target_holdout int    // from pre-flight: sample_size_means result
    analysis_day   int
) {
    t := math.min(f64(holdout_rpr.len) / f64(target_holdout), 1.0)
    boundary := obf_boundary(t, 0.05)

    r := experiment.abtest(holdout_rpr, treatment_rpr)

    println('=== Revenue Per Report — Day ${analysis_day} ===')
    println('  Holdout:   n=${r.n_control}, mean=\$${r.control_mean:.2f}, ' +
            'std=\$${r.control_std:.2f}')
    println('  Treatment: n=${r.n_treatment}, mean=\$${r.treatment_mean:.2f}, ' +
            'std=\$${r.treatment_std:.2f}')
    println('  Lift: ${r.relative_lift * 100:.1f}%')
    println('  95% CI for difference: [\$${r.ci_lower:.2f}, \$${r.ci_upper:.2f}]')
    println('  Information fraction: ${t:.2f}')
    println('  O\'Brien-Fleming boundary: |z| >= ${boundary:.2f}')
    println('  Observed |t-statistic|:    ${math.abs(r.t_statistic):.2f}')

    if math.abs(r.t_statistic) >= boundary {
        if r.treatment_mean > r.control_mean {
            println('  Decision: STOP — positive RPR effect confirmed.')
        } else {
            println('  Decision: STOP — NEGATIVE RPR effect. Pause rollout immediately.')
        }
    } else if r.ci_upper < 0 {
        // CI entirely below zero: evidence of harm even without crossing boundary
        println('  Decision: PAUSE — CI is entirely negative. Likely harmful.')
    } else {
        println('  Decision: CONTINUE — below boundary. Proceed to next stage.')
    }
}

fn main() {
    // Day 14: 700 matured RPR holdout accounts (account_age >= 7 days, >= 1 report).
    // RPR values are per-user: total_revenue / reports_completed.
    // In practice these come from your data warehouse.
    // Simulated here as representative samples:
    holdout_d14 := [
        11.50, 13.20, 9.80, 14.70, 12.30, 10.90, 15.40, 11.10, 13.80, 12.60,
        10.20, 14.30, 11.90, 13.50, 12.10, 9.60,  15.80, 11.70, 12.90, 10.50,
    ] // ... in practice: ~700 values from your data warehouse

    treatment_d14 := [
        12.10, 14.50, 10.30, 15.80, 13.20, 11.60, 16.90, 12.40, 14.70, 13.40,
        10.80, 15.20, 12.70, 14.30, 13.10, 10.40, 17.20, 12.90, 13.80, 11.30,
    ] // ... in practice: ~3500 values

    check_revenue_per_report(holdout_d14, treatment_d14, 2258, 14)
}
```

**O'Brien-Fleming boundaries for this rollout's 4 looks** (α = 0.05, z_{α/2} = 1.96):

| Analysis day | Info fraction t | Critical | z |  |
| ------------ | --------------- | -------- |
| Day 14       | 0.35            | 3.31     |
| Day 21       | 0.52            | 2.72     |
| Day 35       | 0.87            | 2.10     |
| Day 50       | 1.00            | 1.96     |

Early looks are intentionally very strict (|z| ≥ 3.31 on day 14). You would
only stop early on overwhelming evidence — a modest positive trend does not
qualify.

---

## Step 4 — Joint Decision: When Metrics Disagree

Both metrics must be evaluated before each rollout decision. Use this matrix:

```
RPR decision       │ AR decision          │ Rollout decision
───────────────────┼──────────────────────┼──────────────────────────────────────
positive effect    │ positive effect      │ Accelerate: skip to next stage early
positive effect    │ continue_testing     │ Continue rollout as planned
continue_testing   │ positive effect      │ Continue rollout as planned
continue_testing   │ continue_testing     │ Continue rollout as planned
positive effect    │ futility             │ Investigate: chatbot helps revenue
                   │                      │ but doesn't activate more users — why?
negative effect    │ any                  │ PAUSE. Investigate before proceeding.
any                │ negative effect      │ PAUSE. Investigate before proceeding.
futility (both)    │ futility (both)      │ Ship without concern, or drop feature
```

The rule is asymmetric: **a positive result on one metric never overrides a
negative result on the other.** A chatbot that increases revenue but suppresses
activation is actively harming your funnel and must be investigated before
shipping.

```v
module main

import vstats.experiment
import math

pub enum RolloutDecision {
    accelerate
    continue_rollout
    pause_and_investigate
    ship_neutral
}

pub struct JointDecision {
pub:
    decision   RolloutDecision
    reason     string
}

fn joint_decision(ar experiment.SPRTResult, rpr_t_stat f64, rpr_boundary f64) JointDecision {
    ar_positive  := ar.decision == .reject_null && ar.rate_b > ar.rate_a
    ar_negative  := ar.decision == .reject_null && ar.rate_b < ar.rate_a
    ar_futile    := ar.decision == .accept_null
    ar_continue  := ar.decision == .continue_testing

    rpr_positive := math.abs(rpr_t_stat) >= rpr_boundary && rpr_t_stat > 0
    rpr_negative := math.abs(rpr_t_stat) >= rpr_boundary && rpr_t_stat < 0
    rpr_continue := math.abs(rpr_t_stat) < rpr_boundary

    if ar_negative || rpr_negative {
        return JointDecision{
            decision: .pause_and_investigate
            reason: if ar_negative {
                'AR is significantly negative — chatbot harms activation.'
            } else {
                'RPR is significantly negative — chatbot harms revenue per report.'
            }
        }
    }
    if ar_positive && rpr_positive {
        return JointDecision{
            decision: .accelerate
            reason: 'Both metrics show significant positive effect.'
        }
    }
    if ar_futile && rpr_continue || ar_continue && rpr_futile || ar_futile {
        return JointDecision{
            decision: .ship_neutral
            reason: 'No evidence of meaningful effect. Safe to ship or drop.'
        }
    }
    return JointDecision{
        decision: .continue_rollout
        reason: if ar_continue { 'AR needs more data.' } else { 'RPR needs more data.' }
    }
}

fn rpr_futile(r experiment.ABTestResult, mde f64) bool {
    return r.ci_upper < mde && r.ci_lower > -mde
}

fn main() {
    // Day 21 example: AR shows strong positive, RPR still accumulating.
    ar_result := experiment.sprt_test(
        735, 2100,   // holdout: 735 activated out of 2100 matured (35.0%)
        8190, 21000, // treatment: 8190 activated out of 21000 matured (39.0%)
        experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.03 }
    )

    // RPR: |t| = 1.8, boundary at t=0.52 is 2.72 → below boundary
    rpr_t_stat  := 1.8
    rpr_boundary := 2.72

    decision := joint_decision(ar_result, rpr_t_stat, rpr_boundary)

    println('Joint decision: ${decision.decision}')
    println('Reason:         ${decision.reason}')
    // Joint decision: continue_rollout
    // Reason:         RPR needs more data.
}
```

---

## Full Simulation — 60-Day Chatbot Rollout

The following program simulates the complete rollout from day 0 through day 60,
running both sequential tests at each pre-planned analysis point and printing
a rollout decision at each stage.

```v
module main

import vstats.experiment
import vstats.prob
import math
import rand

// ── Helpers ────────────────────────────────────────────────────────────────

fn obf_boundary(t f64, alpha f64) f64 {
    z := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
    return z / math.sqrt(t)
}

fn srm_ok(n_holdout int, n_treatment int, rollout_pct f64) bool {
    expected := 1.0 / (1.0 + rollout_pct)
    observed := f64(n_holdout) / f64(n_holdout + n_treatment)
    return math.abs(observed - expected) / expected <= 0.05
}

fn normal_sample(mean f64, std f64) f64 {
    mut u1 := rand.f64()
    if u1 <= 0.0 { u1 = 1e-15 }
    u2 := rand.f64()
    z := math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z
}

// ── Simulation state ───────────────────────────────────────────────────────

struct Account {
    group          string   // 'holdout' or 'treatment'
    created_day    int
    activated      bool     // completed a report within 14 days
    revenue_7d     f64      // total revenue in first 7 days
    reports_7d     int      // reports completed in first 7 days
}

fn simulate_account(group string, day int, true_ar f64, true_rpr f64) Account {
    activated := rand.f64() < true_ar
    reports := if rand.f64() < 0.60 { rand.intn(4) or { 1 } + 1 } else { 0 }
    revenue := if reports > 0 {
        math.max(0.0, normal_sample(true_rpr * f64(reports), 12.0))
    } else {
        0.0
    }
    return Account{
        group:       group
        created_day: day
        activated:   activated
        revenue_7d:  revenue
        reports_7d:  reports
    }
}

fn main() {
    rand.seed([u32(2026), u32(3)])

    // ── Experiment parameters ──────────────────────────────────────────────
    true_ar_holdout   := 0.35   // holdout activation rate
    true_ar_treatment := 0.39   // treatment: +4pp (above MDE of 3pp)
    true_rpr_holdout  := 12.0   // holdout RPR
    true_rpr_treatment := 13.80 // treatment: +$1.80 (above MDE of $1.50)

    new_accounts_per_day := 10_000
    target_ar  := 4034   // from sample_size_proportions(0.35, 0.03, 0.05, 0.80)
    target_rpr := 2258   // from sample_size_means(12.0, 18.0, 1.50, 0.05, 0.80)

    // Rollout schedule: (day, treatment_pct)
    rollout_schedule := [(0, 1.0), (14, 5.0), (21, 30.0), (35, 60.0), (50, 99.0)]

    // Analysis checkpoints: days on which we run both tests
    analysis_days := [14, 21, 35, 50]

    // ── Accumulate accounts day by day ────────────────────────────────────
    mut accounts := []Account{}
    mut current_rollout := 1.0

    for day in 0 .. 61 {
        // Update rollout percentage if today is a milestone
        for milestone in rollout_schedule {
            d, pct := milestone
            if d == day { current_rollout = pct }
        }

        // Add new accounts: 1% holdout, current_rollout% treatment
        holdout_n  := new_accounts_per_day / 100
        treatment_n := int(f64(new_accounts_per_day) * current_rollout / 100.0)

        for _ in 0 .. holdout_n {
            accounts << simulate_account('holdout', day, true_ar_holdout, true_rpr_holdout)
        }
        for _ in 0 .. treatment_n {
            accounts << simulate_account('treatment', day, true_ar_treatment, true_rpr_treatment)
        }

        if day !in analysis_days { continue }

        // ── SRM check ─────────────────────────────────────────────────────
        total_holdout   := accounts.filter(it.group == 'holdout').len
        total_treatment := accounts.filter(it.group == 'treatment').len
        println('\n══════════════════════════════════════════════')
        println('Day ${day} Analysis  (${int(current_rollout)}% rollout → ramping to next stage)')
        println('══════════════════════════════════════════════')

        if !srm_ok(total_holdout, total_treatment, current_rollout) {
            println('!! SRM detected — skipping statistical tests')
            continue
        }
        println('SRM: OK  (holdout=${total_holdout}, treatment=${total_treatment})')

        // ── Activation Rate (SPRT) ─────────────────────────────────────────
        // Only include accounts with account_age >= 14 days
        ar_holdout   := accounts.filter(it.group == 'holdout'   && day - it.created_day >= 14)
        ar_treatment := accounts.filter(it.group == 'treatment' && day - it.created_day >= 14)

        ar_h_matured := ar_holdout.len
        ar_t_matured := ar_treatment.len
        ar_h_active  := ar_holdout.filter(it.activated).len
        ar_t_active  := ar_treatment.filter(it.activated).len

        ar_result := experiment.sprt_test(
            ar_h_active, ar_h_matured,
            ar_t_active, ar_t_matured,
            experiment.SPRTConfig{ alpha: 0.05, beta: 0.20, mde: 0.03 }
        )

        println('\nActivation Rate (14-day window):')
        println('  Matured holdout:   ${ar_h_matured} accounts, ${ar_h_active} activated ' +
                '(${ar_result.rate_a * 100:.1f}%)')
        println('  Matured treatment: ${ar_t_matured} accounts, ${ar_t_active} activated ' +
                '(${ar_result.rate_b * 100:.1f}%)')
        println('  LLR: ${ar_result.log_likelihood_ratio:.2f}  ' +
                '[${ar_result.lower_boundary:.2f}, +${ar_result.upper_boundary:.2f}]')
        match ar_result.decision {
            .reject_null {
                dir := if ar_result.rate_b > ar_result.rate_a { 'POSITIVE ↑' } else { 'NEGATIVE ↓' }
                println('  AR decision: STOP — ${dir} effect detected')
            }
            .accept_null    { println('  AR decision: STOP (futility) — no meaningful effect') }
            .continue_testing { println('  AR decision: CONTINUE — accumulating evidence') }
        }

        // ── Revenue Per Report (group sequential) ─────────────────────────
        // Only include accounts with account_age >= 7 days AND >= 1 report
        rpr_holdout   := accounts.filter(it.group == 'holdout'   &&
                                         day - it.created_day >= 7 && it.reports_7d >= 1)
        rpr_treatment := accounts.filter(it.group == 'treatment' &&
                                         day - it.created_day >= 7 && it.reports_7d >= 1)

        rpr_h_vals := rpr_holdout.map(it.revenue_7d / f64(it.reports_7d))
        rpr_t_vals := rpr_treatment.map(it.revenue_7d / f64(it.reports_7d))

        t_rpr     := math.min(f64(rpr_h_vals.len) / f64(target_rpr), 1.0)
        t_ar_info := math.min(f64(ar_h_matured) / f64(target_ar), 1.0)
        boundary_rpr := obf_boundary(math.max(t_rpr, 0.01), 0.05)

        rpr_result := experiment.abtest(rpr_h_vals, rpr_t_vals)

        println('\nRevenue Per Report (7-day window, users with ≥1 report):')
        println('  Holdout:   n=${rpr_result.n_control}, mean=\$${rpr_result.control_mean:.2f}, ' +
                'std=\$${rpr_result.control_std:.2f}')
        println('  Treatment: n=${rpr_result.n_treatment}, mean=\$${rpr_result.treatment_mean:.2f}, ' +
                'std=\$${rpr_result.treatment_std:.2f}')
        println('  Lift: ${rpr_result.relative_lift * 100:.1f}%  ' +
                'CI: [\$${rpr_result.ci_lower:.2f}, \$${rpr_result.ci_upper:.2f}]')
        println('  Info fraction: t_AR=${t_ar_info:.2f}  t_RPR=${t_rpr:.2f}  ' +
                'OBF boundary: |z| >= ${boundary_rpr:.2f}  observed: ${math.abs(rpr_result.t_statistic):.2f}')

        if math.abs(rpr_result.t_statistic) >= boundary_rpr {
            dir := if rpr_result.treatment_mean > rpr_result.control_mean { 'POSITIVE ↑' } else { 'NEGATIVE ↓' }
            println('  RPR decision: STOP — ${dir} effect detected')
        } else {
            println('  RPR decision: CONTINUE — below O\'BF boundary')
        }

        // ── Rollout recommendation ─────────────────────────────────────────
        ar_harm  := ar_result.decision == .reject_null && ar_result.rate_b < ar_result.rate_a
        rpr_harm := math.abs(rpr_result.t_statistic) >= boundary_rpr &&
                    rpr_result.treatment_mean < rpr_result.control_mean

        println('\n→ Rollout recommendation: ' + if ar_harm || rpr_harm {
            'PAUSE — investigate regression before proceeding'
        } else {
            'PROCEED to next stage'
        })
    }

    // ── Post-launch canary (after 99% rollout, day 60) ────────────────────
    println('\n══════════════════════════════════════════════')
    println('Post-launch canary (week-over-week standard check)')
    println('══════════════════════════════════════════════')

    // Sample 500 recent accounts per group for the weekly canary check.
    // Use standard abtest here — one look per week, no alpha spending needed.
    recent_holdout := accounts.filter(it.group == 'holdout' && it.reports_7d >= 1)
    recent_treatment := accounts.filter(it.group == 'treatment' && it.reports_7d >= 1)

    sample_h := recent_holdout[recent_holdout.len - 500..].map(
        it.revenue_7d / f64(it.reports_7d))
    sample_t := recent_treatment[recent_treatment.len - 500..].map(
        it.revenue_7d / f64(it.reports_7d))

    canary := experiment.abtest(sample_h, sample_t)
    println('  p-value: ${canary.p_value:.4f}  lift: ${canary.relative_lift * 100:.1f}%')
    println(if canary.significant && canary.treatment_mean < canary.control_mean {
        '  !! RPR regression detected — open incident'
    } else {
        '  Healthy — no significant regression vs holdout'
    })
}
```

Run it:

```bash
v run examples/chatbot_rollout_simulation.v
```

---

## Checklist: Before Each Rollout Stage

```
□ Run SRM check — never skip this, even at routine checkpoints
□ Filter AR data: account_age >= 14 days only
□ Filter RPR data: account_age >= 7 days AND reports_completed >= 1
□ AR: call sprt_test with cumulative matured counts
□ RPR: call abtest, compare |t_statistic| to obf_boundary(t, 0.05)
□ Apply joint decision matrix — neither metric can override harm in the other
□ Document the information fractions and boundaries used (for audit trail)
□ If either metric says STOP with a negative effect, open an incident
   before touching the rollout percentage
```

---

## Common Mistakes in This Specific Setup

| Mistake                                                                | What goes wrong                                                                                                         |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Including un-matured accounts in AR                                    | Inflates the denominator, suppresses the activation rate — makes the chatbot look worse than it is                      |
| Including zero-report users in RPR                                     | Creates undefined values (`revenue / 0`); always filter `reports_completed >= 1`                                        |
| Interpreting RPR without AR context                                    | An RPR drop might just mean the chatbot activated lower-value accounts (actually a good thing if AR is up)              |
| Using a fixed 5% p-value threshold instead of O'Brien-Fleming          | Inflates false positive rate across 4 looks from 5% to ~18%                                                             |
| Pausing when SPRT says `continue_testing` because the trend looks good | Defeats the whole purpose — only act on `reject_null` or `accept_null` decisions                                        |
| Releasing the holdout at 99%                                           | Lose the canary. Keep 1% holdout permanently.                                                                           |
| Checking results on a Monday after a weekend spike                     | Session behavior on weekends differs from weekdays. Always include at least one full business week per analysis window. |
