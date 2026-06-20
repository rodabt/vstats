module main

import os
import vstats.experiment
import vstats.chart

fn main() {
	// SaaS trial-to-paid conversion at 5 %. 500 users/day per variant.
	// $5 M ARR, trying to detect a 1 pp absolute lift (20 % relative).
	// day_cost=$500 ≈ 0.01 % of ARR — a standard calibration.
	// Classical formula requires ~30 days for 80 % power; seasonality floor is 14.
	// Power floor is the binding constraint → effective_min ≈ 30 days.
	config := experiment.OptimizerConfig{
		baseline:                  0.05
		daily_traffic_per_variant: 500
		annual_revenue:            5_000_000.0
		alpha:                     0.05
		day_cost:                  500.0
		seasonality_min_days:      14
		min_power:                 0.80
		prior:                     experiment.MixturePrior{
			null_frac: 0.40
			neg_frac:  0.30
			neg_mean:  -0.01
			neg_std:   0.005
			pos_mean:  0.01
			pos_std:   0.005
			n_samples: 100_000
		}
		max_days: 90
		seed:     42
	}

	result := experiment.find_optimal_runtime(config)

	if result.worth_running {
		println('Worth running   : yes')
		println('Power floor     : ${result.power_min_days} days  (${config.min_power * 100:.0f}% power for ${config.prior.pos_mean * 100:.1f}pp MDE)')
		println('Seasonality floor: ${config.seasonality_min_days} days')
		println('Effective min   : ${result.effective_min_days} days  (binding: ${if result.power_min_days >= result.effective_min_days { 'power' } else { 'seasonality' }})')
		println('Optimal runtime : ${result.optimal_days} days')
		println('Power at optimal: ${result.power_at_optimal * 100:.1f}%')
		println('Annual utility  : \$${result.annual_utility:.0f}')
	} else {
		println('Worth running   : NO')
		println('Reason          : ${result.no_go_reason}')
	}

	out_dir := os.dir(@FILE)

	if result.worth_running {
		days_x := result.all_results.map(f64(it.runtime_days))
		au_y := result.all_results.map(it.annual_utility)

		chart.new(
			title:    'Optimal runtime: ${result.optimal_days} days  (power floor: ${result.power_min_days} days)'
			subtitle: 'Baseline ${config.baseline * 100:.0f}%  ·  ${config.daily_traffic_per_variant}/day  ·  MDE ${config.prior.pos_mean * 100:.1f}pp  ·  \$${config.day_cost:.0f}/day  ·  \$${config.annual_revenue / 1_000_000:.0f}M ARR'
			width:    700
			height:   420
		)
			.line(days_x, au_y)
			.axvline(f64(result.effective_min_days))
			.xlabel('Runtime (days)')
			.ylabel('Annual utility (\$)')
			.save(os.join_path(out_dir, 'annual_utility_vs_runtime.svg'))!
	} else {
		// Show the (negative) curve so the user can see how far off the economics are
		days_x := result.all_results.map(f64(it.runtime_days))
		au_y := result.all_results.map(it.annual_utility)
		if days_x.len > 0 {
			chart.new(
				title:    'Not worth running: ${result.no_go_reason}'
				subtitle: 'Baseline ${config.baseline * 100:.0f}%  ·  ${config.daily_traffic_per_variant}/day  ·  MDE ${config.prior.pos_mean * 100:.1f}pp  ·  \$${config.day_cost:.0f}/day  ·  \$${config.annual_revenue / 1_000_000:.0f}M ARR'
				width:    700
				height:   420
			)
				.line(days_x, au_y)
				.xlabel('Runtime (days)')
				.ylabel('Annual utility (\$)')
				.save(os.join_path(out_dir, 'annual_utility_vs_runtime.svg'))!
		} else {
			println('(no chart — experiment cannot reach target power within max_days)')
		}
	}
}
