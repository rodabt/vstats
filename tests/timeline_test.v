import vstats.experiment
import math

fn test__required_runtime_classical_proportion() {
	// baseline 0.10, 20% relative lift (0.02 absolute), alpha 0.05, power 0.80, 1000/day.
	// Classical n ≈ 3837/arm → ceil(3837/1000) = 4 days. Seasonality floor of 1 is non-binding.
	days := experiment.required_runtime(0.20, 0.10, 0.0, 1000, 0.05, 0.80, 1)
	assert days == 4
}

fn test__required_runtime_seasonality_floor_binds() {
	// Same design but 10_000/day → power needs 1 day; seasonality floor of 14 dominates.
	days := experiment.required_runtime(0.20, 0.10, 0.0, 10_000, 0.05, 0.80, 14)
	assert days == 14
}

fn test__required_runtime_continuous() {
	// baseline $47.50, σ=$82, 5% relative lift ($2.375), 900/day, alpha 0.05, power 0.80.
	// n/arm ≈ 18.7k → ~21 days. Allow a small tolerance around the classical value.
	days := experiment.required_runtime(0.05, 47.50, 82.0, 900, 0.05, 0.80, 1)
	assert days >= 19
	assert days <= 23
}

fn test__detectable_lift_roundtrip_proportion() {
	// required_runtime rounds days UP, so the detectable lift at that runtime should be
	// no worse than (i.e. ≤) the original target.
	baseline := 0.10
	traffic := 1000
	days := experiment.required_runtime(0.20, baseline, 0.0, traffic, 0.05, 0.80, 1) // 0.02 absolute target
	dl := experiment.detectable_lift(days, baseline, 0.0, traffic, 0.05, 0.80)
	assert dl.absolute <= 0.02 + 1e-6
	assert dl.absolute > 0.015 // ceiling of days shouldn't overshoot wildly
	assert math.abs(dl.relative - dl.absolute / baseline) < 1e-12
}

fn test__detectable_lift_roundtrip_continuous() {
	baseline := 47.50
	sigma := 82.0
	traffic := 900
	days := experiment.required_runtime(0.05, baseline, sigma, traffic, 0.05, 0.80, 1) // $2.375 target
	dl := experiment.detectable_lift(days, baseline, sigma, traffic, 0.05, 0.80)
	assert dl.absolute <= 2.375 + 1e-6
	assert dl.absolute > 2.0
	assert math.abs(dl.relative - dl.absolute / baseline) < 1e-12
}

fn test__detectable_lift_more_days_finer() {
	// More runtime ⇒ smaller detectable lift (strictly finer sensitivity).
	a := experiment.detectable_lift(10, 0.10, 0.0, 1000, 0.05, 0.80)
	b := experiment.detectable_lift(40, 0.10, 0.0, 1000, 0.05, 0.80)
	assert b.absolute < a.absolute
}

fn test__plan_timeline_back_to_back_dates() {
	// Two runtime-driven experiments, no buffer, start_day 1 → contiguous slots.
	plan := experiment.TimelinePlan{
		start_day:   1
		experiments: [
			experiment.ExperimentSpec{
				name:                      'A'
				baseline:                  0.20
				daily_traffic_per_variant: 900
				runtime_days:              14
			},
			experiment.ExperimentSpec{
				name:                      'B'
				baseline:                  0.10
				daily_traffic_per_variant: 1300
				runtime_days:              21
			},
		]
	}
	res := experiment.plan_timeline(plan)
	assert res.slots.len == 2
	assert res.slots[0].start_day == 1
	assert res.slots[0].end_day == 15 // 1 + 14
	assert res.slots[1].start_day == 15
	assert res.slots[1].end_day == 36 // 15 + 21
	assert res.end_day == 36
	assert res.total_days == 35 // 36 - 1
	assert res.slots[0].detectable_lift > 0.0
	assert res.slots[1].detectable_lift > 0.0
}

fn test__plan_timeline_sensitivity_driven_resolves_runtime() {
	// A sensitivity-driven spec: runtime is computed from the target lift.
	plan := experiment.TimelinePlan{
		experiments: [
			experiment.ExperimentSpec{
				name:                      'C'
				baseline:                  0.10
				daily_traffic_per_variant: 1000
				target_lift:               0.20
				seasonality_min_days:      1
			},
		]
	}
	res := experiment.plan_timeline(plan)
	assert res.slots[0].runtime_days == 4 // matches required_runtime classical anchor
	assert res.slots[0].detectable_lift <= 0.20 + 1e-6
}

fn test__plan_timeline_buffer_between() {
	plan := experiment.TimelinePlan{
		start_day:   0
		experiments: [
			experiment.ExperimentSpec{
				name:                      'A'
				baseline:                  0.20
				daily_traffic_per_variant: 900
				runtime_days:              14
				seasonality_min_days:      1
				buffer_days_after:         3
			},
			experiment.ExperimentSpec{
				name:                      'B'
				baseline:                  0.20
				daily_traffic_per_variant: 900
				runtime_days:              10
				seasonality_min_days:      1
			},
		]
	}
	res := experiment.plan_timeline(plan)
	assert res.slots[0].end_day == 14
	assert res.slots[1].start_day == 17 // 14 + 3 buffer
	assert res.slots[1].end_day == 27
}

fn test__plan_timeline_seasonality_bump_warns() {
	// runtime_days below the seasonality floor is bumped up and warned.
	plan := experiment.TimelinePlan{
		experiments: [
			experiment.ExperimentSpec{
				name:                      'A'
				baseline:                  0.20
				daily_traffic_per_variant: 900
				runtime_days:              5
				seasonality_min_days:      14
			},
		]
	}
	res := experiment.plan_timeline(plan)
	assert res.slots[0].runtime_days == 14
	assert res.slots[0].feasible == true
	assert res.slots[0].warning.contains('seasonality')
}

fn test__plan_timeline_too_coarse_flag() {
	// A short runtime whose detectable lift exceeds max_acceptable_lift → not feasible.
	plan := experiment.TimelinePlan{
		experiments: [
			experiment.ExperimentSpec{
				name:                      'A'
				baseline:                  0.10
				daily_traffic_per_variant: 50
				runtime_days:              14
				max_acceptable_lift:       0.05
			},
		]
	}
	res := experiment.plan_timeline(plan)
	assert res.slots[0].feasible == false
	assert res.slots[0].warning.contains('exceeds target')
}

fn test__plan_timeline_empty_plan() {
	res := experiment.plan_timeline(experiment.TimelinePlan{})
	assert res.slots.len == 0
	assert res.total_days == 0
	assert res.end_day == 0
}
