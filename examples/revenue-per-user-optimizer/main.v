module main

import vstats.experiment

// ─────────────────────────────────────────────────────────────────────────────
// A/B test design optimizer — continuous metric (revenue per user)
// ─────────────────────────────────────────────────────────────────────────────
//
// NOTE: this demonstrates the optional prior-based runtime *recommender*
// (find_optimal_runtime). Reach for it when you have neither a runtime nor a
// sensitivity target in mind and want one suggested under a prior belief. To
// sequence experiments on a timeline — picking a runtime or sensitivity per
// experiment — see examples/experiment-timeline (the primary planning tool).
//
// Scenario
// --------
// An e-commerce team is testing a post-purchase upsell widget on the order
// confirmation page. The success metric is REVENUE PER USER over the 7 days after
// checkout — a continuous, right-skewed quantity, not a rate. Continuous metrics
// behave very differently from conversion rates in power analysis: the binomial
// p(1−p)/n variance no longer applies, so you MUST supply the historical standard
// deviation of the per-user value. High variance (σ ≫ mean, typical for revenue)
// is what makes these experiments expensive.
//
// The optimizer finds the runtime that maximizes the expected number of true
// positive effects detected per month, subject to a power floor and a seasonality
// floor, and tells you up front whether the design is viable at all within max_days.
//
// The headline lesson of this example is the SENSITIVITY of the answer to σ: the
// historical standard deviation is an estimate, and revenue σ is volatile. We sweep
// it at the end to show how much the recommendation moves when σ is off by ±35%.

fn main() {
	// ── Primary design ────────────────────────────────────────────────────────
	params := experiment.DesignParams{
		// Mean revenue per user over the last 90 days, in dollars. This is the metric
		// baseline; the MDE below is taken relative to it.
		baseline: 47.50

		// Historical standard deviation of revenue per user, in dollars. REQUIRED for
		// continuous metrics — it drives the two-sample power formula in place of the
		// binomial variance used for proportions. For right-skewed revenue, σ is often
		// 1.5–3× the mean. Source: SELECT STDDEV(revenue_7d) FROM orders WHERE …
		metric_std_dev: 82.00

		// Users reaching the confirmation page per variant per day (i.e. who actually
		// see the widget). Eligibility — not total site traffic — is what counts. This
		// is a busy checkout; thinner traffic would push the design toward no-go.
		daily_traffic_per_variant: 900

		// Smallest relative lift worth acting on, as a fraction of baseline.
		// 0.05 = detect a $2.38 increase on the $47.50 baseline. Revenue metrics
		// usually need a LARGER MDE than conversion rates because σ ≫ mean inflates
		// the sample size; chasing a 1–2% lift here is often simply unaffordable.
		min_relative_lift: 0.05

		// Confidence the widget will lift revenue. Past upsell tests were mixed, so
		// this is moderate skepticism — 0.35. Conviction sets both the prior over
		// effect sizes and the minimum monthly detection rate required to say "go".
		prior_conviction: 0.35

		// Span at least two full weekly cycles so paydays / weekend spend balance out.
		seasonality_min_days: 14

		// Revenue experiments often need more calendar time than conversion tests, so
		// allow a longer cap before declaring the design infeasible.
		max_days: 60
	}

	config := experiment.optimizer_config(params)
	result := experiment.find_optimal_runtime(config)

	print_header('REVENUE-PER-USER A/B DESIGN', config, params)
	print_readout(result, config)
	print_runtime_curve(result)

	// ── Sensitivity analysis ───────────────────────────────────────────────────
	// σ is an estimate from historical data and revenue variance drifts with the mix
	// of products and promotions. Re-run the design at σ = $60 / $82 / $110 to see
	// how the optimal runtime — and even the go/no-go verdict — shift. If the answer
	// is stable across this band, you can trust it; if it flips, tighten the σ
	// estimate before committing engineering time.
	println('')
	println('Sensitivity to the historical σ estimate (all else fixed):')
	for sigma in [60.0, 82.0, 110.0] {
		sweep_sigma(params, sigma)
	}
}

// design_with_sigma clones the base design but overrides metric_std_dev.
// DesignParams fields are read-only outside the module, so we rebuild rather than
// mutate a copy.
fn design_with_sigma(base experiment.DesignParams, sigma f64) experiment.DesignParams {
	return experiment.DesignParams{
		baseline:                  base.baseline
		metric_std_dev:            sigma
		daily_traffic_per_variant: base.daily_traffic_per_variant
		min_relative_lift:         base.min_relative_lift
		prior_conviction:          base.prior_conviction
		seasonality_min_days:      base.seasonality_min_days
		max_days:                  base.max_days
	}
}

fn sweep_sigma(base experiment.DesignParams, sigma f64) {
	cfg := experiment.optimizer_config(design_with_sigma(base, sigma))
	res := experiment.find_optimal_runtime(cfg)
	label := 'σ = \$${sigma:.0f}'
	if res.worth_running {
		println('  ${label:-12}  →  optimal ${res.optimal_days:2} d   power ${res.power_at_optimal * 100:5.1f}%   det-rate ${res.monthly_detection_rate:.3f}')
	} else {
		println('  ${label:-12}  →  NO-GO   ${res.no_go_reason}')
	}
}

// ── Shared printing helpers ────────────────────────────────────────────────────

fn print_header(title string, config experiment.OptimizerConfig, params experiment.DesignParams) {
	mde_abs := params.baseline * params.min_relative_lift
	pos_frac := 1.0 - config.prior.null_frac - config.prior.neg_frac
	println('━━━ ${title} ━━━')
	println('Baseline           : \$${config.baseline:.2f} revenue/user')
	println('Metric type        : continuous  (σ = \$${params.metric_std_dev:.2f}, i.e. ${params.metric_std_dev / params.baseline:.1f}× the mean)')
	println('Traffic            : ${config.daily_traffic_per_variant}/variant/day')
	println('Min lift           : ${params.min_relative_lift * 100:.0f}% relative  (\$${mde_abs:.2f} absolute MDE)')
	println('Prior              : conviction ${params.prior_conviction:.2f}  →  pos ${pos_frac * 100:.0f}%  null ${config.prior.null_frac * 100:.0f}%  neg ${config.prior.neg_frac * 100:.0f}%')
	println('Go threshold       : monthly detection rate ≥ ${config.min_monthly_detection_rate:.3f}')
	println('')
}

fn print_readout(result experiment.OptimizationResult, config experiment.OptimizerConfig) {
	if !result.worth_running {
		println('Verdict            : ✗ NOT worth running')
		println('Reason             : ${result.no_go_reason}')
		return
	}
	binding := if result.power_min_days >= result.effective_min_days { 'power' } else { 'seasonality' }
	println('Verdict            : ✓ worth running')
	println('Power floor        : ${result.power_min_days} days  (${config.min_power * 100:.0f}% power at the MDE)')
	println('Seasonality floor  : ${config.seasonality_min_days} days')
	println('Effective minimum  : ${result.effective_min_days} days  (binding constraint: ${binding})')
	println('Optimal runtime    : ${result.optimal_days} days')
	println('Power at optimal   : ${result.power_at_optimal * 100:.1f}%')
	println('Monthly detections : ${result.monthly_detection_rate:.3f}  (expected true positives caught per 30 days)')
}

// print_runtime_curve renders the detection-rate-vs-runtime curve as an inline
// ASCII bar chart so the interior optimum is visible at a glance — no SVG needed.
// It samples ~12 evenly spaced runtimes and always marks the earliest valid day,
// the optimum, and the longest considered.
fn print_runtime_curve(result experiment.OptimizationResult) {
	rows := result.all_results
	if rows.len == 0 {
		return
	}
	println('')
	println('Monthly detection rate by runtime (✓ = chosen optimum):')
	target := 12
	stride := if rows.len <= target { 1 } else { rows.len / target }
	for i, r in rows {
		is_opt := r.runtime_days == result.optimal_days
		is_min := r.runtime_days == result.effective_min_days
		if i % stride == 0 || is_opt || i == rows.len - 1 {
			bar := '█'.repeat(int(r.monthly_detection_rate * 40.0))
			marker := if is_opt {
				' ✓ optimum'
			} else if is_min {
				' ← earliest valid'
			} else {
				''
			}
			println('  ${r.runtime_days:3} d  ${r.monthly_detection_rate:.3f}  ${bar}${marker}')
		}
	}
}
