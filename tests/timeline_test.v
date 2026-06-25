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
