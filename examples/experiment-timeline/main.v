module main

import vstats.experiment

// ─────────────────────────────────────────────────────────────────────────────
// Experiment timeline planner
// ─────────────────────────────────────────────────────────────────────────────
//
// A PM has an ordered roadmap of experiments — the order is fixed by dependencies,
// so the question is NOT "which to run first" but "when does each run, and what can
// it detect?" The planner takes the chain as given and, for each experiment, fills
// in whichever of {runtime, sensitivity} the PM did not specify, then lays them
// back-to-back on a day-offset timeline. Nothing is optimized — every number is a
// direct power-identity calculation.

fn main() {
	plan := experiment.TimelinePlan{
		start_day:   1
		experiments: [
			// Lever = RUNTIME. We have a fixed 2-week slot; report what it can detect.
			experiment.ExperimentSpec{
				name:                      'Onboarding CTA copy'
				baseline:                  0.22 // signup-completion rate
				daily_traffic_per_variant: 900
				runtime_days:              14
			},
			// Lever = SENSITIVITY. We need to catch a 10% relative lift; let the planner
			// compute the runtime that buys 80% power for it.
			experiment.ExperimentSpec{
				name:                      'Pricing page layout'
				baseline:                  0.08 // trial-start rate
				daily_traffic_per_variant: 1300
				target_lift:               0.10
			},
			// Lever = RUNTIME, continuous metric (revenue/user needs metric_std_dev),
			// with a sensitivity bar: flag the slot if 10 days can't see a ≤3% lift.
			experiment.ExperimentSpec{
				name:                      'Checkout upsell widget'
				baseline:                  47.50
				metric_std_dev:            82.0
				daily_traffic_per_variant: 900
				runtime_days:              10
				max_acceptable_lift:       0.03
			},
		]
	}

	result := experiment.plan_timeline(plan)

	println('Experiment roadmap (start day ${plan.start_day}):')
	for s in result.slots {
		flag := if s.feasible { '✓' } else { '⚠' }
		println('  day ${s.start_day:3}–${s.end_day:3}  ${s.name:-24}  ${s.runtime_days:2}d  detect ≥ ${s.detectable_lift * 100:4.1f}%  power ${s.power * 100:.0f}%  ${flag}')
		if s.warning != '' {
			println('             ↳ ${s.warning}')
		}
	}
	println('Total: ${result.total_days} days (ends day ${result.end_day})')
}
