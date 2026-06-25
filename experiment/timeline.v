module experiment

import vstats.prob
import math

// sample_size_for returns the raw (un-rounded) samples-per-arm needed to detect
// `effect` (absolute) at the given alpha and power. Proportions use a pooled-variance
// null and unpooled alternative; continuous metrics use the 2σ² two-sample form.
// This is the single source of truth shared by power_floor and the timeline solver.
fn sample_size_for(effect f64, baseline f64, std_dev f64, alpha f64, power f64) f64 {
	z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
	z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)
	if std_dev > 0.0 {
		// Two-sample t-test: n = 2σ²(z_α + z_β)² / δ²
		return 2.0 * std_dev * std_dev * math.pow(z_alpha + z_beta, 2.0) / (effect * effect)
	}
	// Two-proportion z-test (pooled-variance null, unpooled alternative).
	p1 := baseline
	p2_raw := baseline + effect
	p2 := if p2_raw < 0.0001 { 0.0001 } else if p2_raw > 0.9999 { 0.9999 } else { p2_raw }
	p_bar := (p1 + p2) / 2.0
	var_null := 2.0 * p_bar * (1.0 - p_bar)
	var_alt := p1 * (1.0 - p1) + p2 * (1.0 - p2)
	return math.pow(z_alpha * math.sqrt(var_null) + z_beta * math.sqrt(var_alt), 2.0) / (effect * effect)
}

// power_min_days is the pure power-based runtime (in days) to detect `effect_abs`,
// before any seasonality floor is applied.
fn power_min_days(effect_abs f64, baseline f64, std_dev f64, daily_traffic int, alpha f64, power f64) int {
	n := int(math.ceil(sample_size_for(effect_abs, baseline, std_dev, alpha, power)))
	return int(math.ceil(f64(n) / f64(daily_traffic)))
}

// required_runtime returns the runtime in days needed to detect `target_lift`
// (relative to baseline) at the given alpha and power, raised to the seasonality floor.
pub fn required_runtime(target_lift f64, baseline f64, std_dev f64, daily_traffic int, alpha f64, power f64, seasonality_min_days int) int {
	pd := power_min_days(target_lift * baseline, baseline, std_dev, daily_traffic, alpha, power)
	return if pd > seasonality_min_days { pd } else { seasonality_min_days }
}

pub struct DetectableLift {
pub:
	relative f64 // lift as a fraction of baseline (the MDE the PM reads)
	absolute f64 // lift in metric units (pp for proportions, $/units for continuous)
}

// detectable_lift inverts the power identity: given a runtime, what lift can we
// detect at `power`? Continuous metrics are closed-form; proportions are solved by
// bisection against sample_size_for (which is monotonically decreasing in the effect).
pub fn detectable_lift(runtime_days int, baseline f64, std_dev f64, daily_traffic int, alpha f64, power f64) DetectableLift {
	n_per_arm := f64(runtime_days) * f64(daily_traffic)
	if std_dev > 0.0 {
		z_alpha := prob.inverse_normal_cdf(1.0 - alpha / 2.0, 0.0, 1.0)
		z_beta := prob.inverse_normal_cdf(power, 0.0, 1.0)
		abs := (z_alpha + z_beta) * math.sqrt(2.0 * std_dev * std_dev / n_per_arm)
		return DetectableLift{
			relative: abs / baseline
			absolute: abs
		}
	}
	// Proportion: find the absolute effect whose required sample size equals n_per_arm.
	mut lo := 1e-9
	mut hi := 1.0 - baseline
	for _ in 0 .. 60 {
		mid := 0.5 * (lo + hi)
		need := sample_size_for(mid, baseline, std_dev, alpha, power)
		if need > n_per_arm {
			// Not enough samples for this effect → need a LARGER effect to detect.
			lo = mid
		} else {
			hi = mid
		}
	}
	abs := 0.5 * (lo + hi)
	return DetectableLift{
		relative: abs / baseline
		absolute: abs
	}
}

pub struct ExperimentSpec {
pub:
	name                      string
	baseline                  f64
	metric_std_dev            f64 = 0.0 // >0 ⇒ continuous; 0 ⇒ proportion
	daily_traffic_per_variant int
	// Exactly ONE of the next two must be set:
	runtime_days int // PM fixes runtime    → tool reports lift
	target_lift  f64 // PM fixes sensitivity → tool reports runtime
	// Overrides:
	alpha                f64 = 0.05
	min_power            f64 = 0.80
	seasonality_min_days int = 14
	buffer_days_after    int // analysis/decision gap before the next starts
	max_acceptable_lift  f64 // optional "too coarse" bar; 0 = no flag
}

pub struct TimelinePlan {
pub:
	experiments []ExperimentSpec
	start_day   int // day-offset origin; default 0
}

pub struct ScheduledExperiment {
pub:
	name                string
	start_day           int
	end_day             int
	runtime_days        int
	detectable_lift     f64 // relative MDE
	detectable_lift_abs f64
	power               f64
	alpha               f64
	feasible            bool
	warning             string
}

pub struct ScheduleResult {
pub:
	slots      []ScheduledExperiment
	total_days int
	end_day    int
}

// plan_timeline lays an ordered backlog onto a day-offset timeline, computing each
// experiment's dates and the effect size it can detect. Order is taken as given —
// nothing is reordered or optimized.
pub fn plan_timeline(plan TimelinePlan) ScheduleResult {
	mut slots := []ScheduledExperiment{}
	mut cursor := plan.start_day
	for spec in plan.experiments {
		assert spec.baseline > 0.0, 'baseline must be positive'
		if spec.metric_std_dev == 0.0 {
			assert spec.baseline < 1.0, 'proportion baseline must be in (0, 1) — set metric_std_dev for continuous metrics'
		}
		assert spec.daily_traffic_per_variant > 0, 'daily_traffic_per_variant must be positive'
		assert spec.alpha > 0.0 && spec.alpha < 1.0, 'alpha must be in (0, 1)'
		assert spec.min_power > 0.0 && spec.min_power < 1.0, 'min_power must be in (0, 1)'
		has_runtime := spec.runtime_days > 0
		has_target := spec.target_lift > 0.0
		assert has_runtime != has_target, 'set exactly one of runtime_days or target_lift'

		pd := if has_runtime {
			spec.runtime_days
		} else {
			power_min_days(spec.target_lift * spec.baseline, spec.baseline, spec.metric_std_dev,
				spec.daily_traffic_per_variant, spec.alpha, spec.min_power)
		}
		floor := spec.seasonality_min_days
		runtime := if pd > floor { pd } else { floor }
		bumped := pd < floor

		dl := detectable_lift(runtime, spec.baseline, spec.metric_std_dev, spec.daily_traffic_per_variant,
			spec.alpha, spec.min_power)

		mut warning := ''
		mut feasible := true
		if bumped {
			warning = 'runtime raised to seasonality floor (${floor} days)'
		}
		if spec.max_acceptable_lift > 0.0 && dl.relative > spec.max_acceptable_lift {
			too := 'detectable lift ${dl.relative * 100:.1f}% exceeds target ${spec.max_acceptable_lift * 100:.1f}%'
			warning = if warning == '' { too } else { warning + '; ' + too }
			feasible = false
		}

		start := cursor
		end := start + runtime
		slots << ScheduledExperiment{
			name:                spec.name
			start_day:           start
			end_day:             end
			runtime_days:        runtime
			detectable_lift:     dl.relative
			detectable_lift_abs: dl.absolute
			power:               spec.min_power
			alpha:               spec.alpha
			feasible:            feasible
			warning:             warning
		}
		cursor = end + spec.buffer_days_after
	}
	return ScheduleResult{
		slots:      slots
		total_days: cursor - plan.start_day
		end_day:    cursor
	}
}
