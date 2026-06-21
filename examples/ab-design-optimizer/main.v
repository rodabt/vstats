module main

import os
import vstats.experiment
import vstats.chart

fn main() {
	// SaaS trial-to-paid conversion at 5 %. 500 users/day per variant.
	// Minimum meaningful lift: 1.5pp (mde_tolerance=0.015).
	// Monthly cadence: max_days=30. Seasonality floor: 14 days.
	// Power floor uses mde_tolerance directly (no revenue/cost needed).
	config := experiment.OptimizerConfig{
		baseline:                  0.05
		daily_traffic_per_variant: 500
		mde_tolerance:             0.015
		alpha:                     0.05
		seasonality_min_days:      14
		min_power:                 0.80
		max_days:                  30
		prior:                     experiment.MixturePrior{
			null_frac: 0.40
			neg_frac:  0.30
			neg_mean:  -0.01
			neg_std:   0.005
			pos_mean:  0.02
			pos_std:   0.01
			n_samples: 100_000
		}
		seed: 42
	}

	result := experiment.find_optimal_runtime(config)

	if result.worth_running {
		println('Worth running      : yes')
		println('Power floor        : ${result.power_min_days} days  (${config.min_power * 100:.0f}% power for ${config.mde_tolerance * 100:.1f}pp MDE)')
		println('Seasonality floor  : ${config.seasonality_min_days} days')
		println('Effective min      : ${result.effective_min_days} days  (binding: ${if result.power_min_days >= result.effective_min_days {
			'power'
		} else {
			'seasonality'
		}})')
		println('Optimal runtime    : ${result.optimal_days} days')
		println('Power at optimal   : ${result.power_at_optimal * 100:.1f}%')
		println('Monthly detections : ${result.monthly_detection_rate:.3f}')
	} else {
		println('Worth running      : NO')
		println('Reason             : ${result.no_go_reason}')
	}

	out_dir := os.dir(@FILE)

	if result.worth_running {
		days_x := result.all_results.map(f64(it.runtime_days))
		dr_y := result.all_results.map(it.monthly_detection_rate)

		chart.new(
			title:    'Optimal runtime: ${result.optimal_days} days  (power floor: ${result.power_min_days} days)'
			subtitle: 'Baseline ${config.baseline * 100:.0f}%  ·  ${config.daily_traffic_per_variant}/day  ·  MDE ≥ ${config.mde_tolerance * 100:.1f}pp  ·  max ${config.max_days} days'
			width:    700
			height:   420
		)
			.line(days_x, dr_y)
			.axvline(f64(result.effective_min_days))
			.xlabel('Runtime (days)')
			.ylabel('Monthly detection rate')
			.save(os.join_path(out_dir, 'detection_rate_vs_runtime.svg'))!
	} else {
		days_x := result.all_results.map(f64(it.runtime_days))
		dr_y := result.all_results.map(it.monthly_detection_rate)
		if days_x.len > 0 {
			chart.new(
				title:    'Not worth running: ${result.no_go_reason}'
				subtitle: 'Baseline ${config.baseline * 100:.0f}%  ·  ${config.daily_traffic_per_variant}/day  ·  MDE ≥ ${config.mde_tolerance * 100:.1f}pp  ·  max ${config.max_days} days'
				width:    700
				height:   420
			)
				.line(days_x, dr_y)
				.xlabel('Runtime (days)')
				.ylabel('Monthly detection rate')
				.save(os.join_path(out_dir, 'detection_rate_vs_runtime.svg'))!
		} else {
			println('(no chart — experiment cannot reach target power within max_days)')
		}
	}
}
