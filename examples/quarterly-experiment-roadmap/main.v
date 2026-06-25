module main

import vstats.experiment

// ─────────────────────────────────────────────────────────────────────────────
// Quarterly experiment roadmap — comprehensive timeline planner showcase
// ─────────────────────────────────────────────────────────────────────────────
//
// A growth team plans a full quarter of experiments across the funnel. The order
// is fixed by product dependencies (you can't test the activation tour before the
// new signup form ships, etc.), so the planner takes the chain as given and answers
// the project-management questions: when does each experiment run, what lift can it
// detect, and are any slots too short to be worth running?
//
// This example exercises every feature of plan_timeline:
//   • both input levers — some experiments fix a RUNTIME (a roadmap slot), others
//     fix a target SENSITIVITY and let the planner compute the runtime;
//   • proportion metrics (rates) AND a continuous metric (revenue per user);
//   • analysis/decision BUFFERS between dependent experiments;
//   • the seasonality floor bumping a too-short slot up to two weeks;
//   • a too-coarse flag when a slot cannot reach a useful sensitivity.
//
// For a minimal three-step introduction, see examples/experiment-timeline. The
// optional prior-based runtime *recommender* (find_optimal_runtime) lives in
// examples/ab-design-optimizer and examples/revenue-per-user-optimizer.

fn main() {
	plan := experiment.TimelinePlan{
		start_day:   1
		experiments: [
			// 1. ONBOARDING — fixed 2-week slot we've already blocked on the calendar.
			experiment.ExperimentSpec{
				name:                      'Onboarding · Simplified signup form'
				baseline:                  0.55 // signup-completion rate
				daily_traffic_per_variant: 1200
				runtime_days:              14
				buffer_days_after:         3 // analyze + decide before the tour test
			},
			// 2. ACTIVATION — depends on (1) shipping. We care about a 6% relative lift;
			//    let the planner size the runtime.
			experiment.ExperimentSpec{
				name:                      'Activation · Interactive product tour'
				baseline:                  0.34 // day-1 activation rate
				daily_traffic_per_variant: 1100
				target_lift:               0.06
				buffer_days_after:         3
			},
			// 3. PRICING — low base rate + small target lift ⇒ this is the long pole.
			experiment.ExperimentSpec{
				name:                      'Pricing · Annual-plan default toggle'
				baseline:                  0.08 // trial → paid conversion
				daily_traffic_per_variant: 700
				target_lift:               0.10
				buffer_days_after:         3
			},
			// 4. CHECKOUT — continuous metric (needs metric_std_dev). Fixed 3-week slot,
			//    with a 5% sensitivity bar. Lands right at the edge of "good enough".
			experiment.ExperimentSpec{
				name:                      'Checkout · One-click upsell widget'
				baseline:                  47.50 // revenue per user ($)
				metric_std_dev:            82.0
				daily_traffic_per_variant: 900
				runtime_days:              21
				max_acceptable_lift:       0.05
				buffer_days_after:         3
			},
			// 5. RETENTION — thin traffic + short slot. The seasonality floor bumps it to
			//    14 days, and even then it can't see an 8% lift ⇒ flagged too coarse.
			experiment.ExperimentSpec{
				name:                      'Retention · Win-back email cadence'
				baseline:                  0.12 // dormant-user reactivation rate
				daily_traffic_per_variant: 400
				runtime_days:              10
				max_acceptable_lift:       0.08
				buffer_days_after:         3
			},
			// 6. REFERRAL — sensitivity-driven; a big relative lift on a tiny base rate.
			experiment.ExperimentSpec{
				name:                      'Referral · Reward amount bump'
				baseline:                  0.03 // invite-send rate
				daily_traffic_per_variant: 1500
				target_lift:               0.15
			},
		]
	}

	result := experiment.plan_timeline(plan)

	println('Q3 experiment roadmap — ${result.slots.len} experiments in fixed dependency order')
	println('Start: day ${plan.start_day}\n')

	mut feasible_count := 0
	for i, s in result.slots {
		spec := plan.experiments[i]
		lever := if spec.runtime_days > 0 {
			'runtime ${s.runtime_days}d'
		} else {
			'target ${spec.target_lift * 100:.0f}%'
		}
		flag := if s.feasible {
			feasible_count++
			'✓'
		} else {
			'⚠'
		}
		println('  day ${s.start_day:3}–${s.end_day:3}  ${s.name:-38}  ${lever:-12}  detect ≥ ${s.detectable_lift * 100:4.1f}%  ${flag}')
		if s.warning != '' {
			println('             ↳ ${s.warning}')
		}
	}

	println('\n${feasible_count}/${result.slots.len} slots clear their sensitivity bar.')
	println('Roadmap spans ${result.total_days} days (start day ${plan.start_day} → end day ${result.end_day}).')
}
