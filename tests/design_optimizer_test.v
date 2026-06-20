import vstats.experiment

// ── updated existing tests ────────────────────────────────────────────────────

fn test__find_optimal_runtime_result_in_range() {
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 5000
		seasonality_min_days:      3
		min_power:                 0.10
		max_days:                  30
		prior:                     experiment.MixturePrior{ n_samples: 5_000 }
		seed:                      1
	}
	result := experiment.find_optimal_runtime(config)
	assert result.worth_running == true
	assert result.optimal_days >= result.effective_min_days
	assert result.optimal_days <= config.max_days
	assert result.all_results.len == config.max_days - result.effective_min_days + 1
}

fn test__find_optimal_runtime_best_is_all_results_max() {
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 5000
		seasonality_min_days:      3
		min_power:                 0.10
		max_days:                  30
		prior:                     experiment.MixturePrior{ n_samples: 5_000 }
		seed:                      2
	}
	result := experiment.find_optimal_runtime(config)
	for r in result.all_results {
		assert result.annual_utility >= r.annual_utility
	}
}

fn test__find_optimal_runtime_all_results_cover_full_range() {
	// all_results must enumerate every day from effective_min_days to max_days
	config := experiment.OptimizerConfig{
		baseline:                  0.20
		daily_traffic_per_variant: 1000
		seasonality_min_days:      5
		min_power:                 0.10
		max_days:                  20
		prior:                     experiment.MixturePrior{ n_samples: 5_000 }
		seed:                      3
	}
	result := experiment.find_optimal_runtime(config)
	// min_power=0.10 → power_min_days=1; seasonality (5) is the binding floor
	assert result.effective_min_days == 5
	assert result.all_results[0].runtime_days == 5
	assert result.all_results[result.all_results.len - 1].runtime_days == config.max_days
	for i in 1 .. result.all_results.len {
		assert result.all_results[i].runtime_days == result.all_results[i - 1].runtime_days + 1
	}
}

fn test__find_optimal_runtime_high_traffic_shorter_optimal() {
	config_lo := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 200
		seasonality_min_days:      3
		min_power:                 0.10
		max_days:                  30
		prior:                     experiment.MixturePrior{ n_samples: 5_000 }
		seed:                      4
	}
	config_hi := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 10_000
		seasonality_min_days:      3
		min_power:                 0.10
		max_days:                  30
		prior:                     experiment.MixturePrior{ n_samples: 5_000 }
		seed:                      4
	}
	result_lo := experiment.find_optimal_runtime(config_lo)
	result_hi := experiment.find_optimal_runtime(config_hi)
	assert result_hi.optimal_days <= result_lo.optimal_days
}

fn test__find_optimal_runtime_positive_utility_for_strong_positive_prior() {
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 5000
		annual_revenue:            10_000_000.0
		day_cost:                  100.0
		seasonality_min_days:      3
		min_power:                 0.10
		prior:                     experiment.MixturePrior{
			null_frac: 0.10
			neg_frac:  0.10
			pos_mean:  0.05
			pos_std:   0.01
			n_samples: 5_000
		}
		max_days: 30
		seed:     5
	}
	result := experiment.find_optimal_runtime(config)
	assert result.annual_utility > 0.0
	assert result.worth_running == true
}

// ── new tests ─────────────────────────────────────────────────────────────────

fn test__power_floor_reflected_in_result() {
	// baseline=0.10, pos_mean=0.02, alpha=0.05, min_power=0.80, 1000/day
	// classical formula: n ≈ 3837 → power_min_days = ceil(3837/1000) = 4
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 1000
		alpha:                     0.05
		min_power:                 0.80
		seasonality_min_days:      3
		prior:                     experiment.MixturePrior{ pos_mean: 0.02, n_samples: 1000 }
		max_days:                  90
		seed:                      1
	}
	result := experiment.find_optimal_runtime(config)
	// allow ±2 days tolerance around classical formula value of 4
	assert result.power_min_days >= 2
	assert result.power_min_days <= 6
	// power floor > seasonality (3), so it is the binding constraint
	assert result.effective_min_days == result.power_min_days
}

fn test__effective_min_is_seasonality_when_binding() {
	// 10k/day + 2pp effect → power_min_days ≈ 1; seasonality=14 dominates
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 10_000
		alpha:                     0.05
		min_power:                 0.80
		seasonality_min_days:      14
		prior:                     experiment.MixturePrior{ pos_mean: 0.02, n_samples: 1000 }
		max_days:                  90
		seed:                      1
	}
	result := experiment.find_optimal_runtime(config)
	assert result.power_min_days < 14
	assert result.effective_min_days == 14
	assert result.all_results[0].runtime_days == 14
}

fn test__effective_min_is_power_when_binding() {
	// 100/day + 2pp effect → power_min_days ≈ 39; seasonality=7 is non-binding
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 100
		alpha:                     0.05
		min_power:                 0.80
		seasonality_min_days:      7
		prior:                     experiment.MixturePrior{ pos_mean: 0.02, n_samples: 1000 }
		max_days:                  90
		seed:                      1
	}
	result := experiment.find_optimal_runtime(config)
	assert result.power_min_days > 7
	assert result.effective_min_days == result.power_min_days
	assert result.all_results[0].runtime_days == result.effective_min_days
}

fn test__no_go_insufficient_traffic() {
	// 10/day trying to detect 0.5pp on 10% baseline at 80% power needs ~5775 days
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 10
		alpha:                     0.05
		min_power:                 0.80
		seasonality_min_days:      7
		prior:                     experiment.MixturePrior{ pos_mean: 0.005, n_samples: 1000 }
		max_days:                  14
		seed:                      1
	}
	result := experiment.find_optimal_runtime(config)
	assert result.worth_running == false
	assert result.all_results.len == 0
	assert result.optimal_days == 0
	assert result.no_go_reason.contains('max_days')
	assert result.power_min_days > config.max_days
}

fn test__no_go_negative_economics() {
	// revenue too small relative to day_cost → EU always negative
	config := experiment.OptimizerConfig{
		baseline:                  0.10
		daily_traffic_per_variant: 5000
		annual_revenue:            10_000.0
		day_cost:                  50_000.0
		alpha:                     0.05
		min_power:                 0.80
		seasonality_min_days:      3
		prior:                     experiment.MixturePrior{ pos_mean: 0.02, n_samples: 5_000 }
		max_days:                  30
		seed:                      1
	}
	result := experiment.find_optimal_runtime(config)
	assert result.worth_running == false
	assert result.annual_utility < 0.0
	assert result.no_go_reason.contains('day_cost')
}

fn test__worth_running_true_and_power_at_optimal_meets_floor() {
	// well-calibrated settings from spec example
	config := experiment.OptimizerConfig{
		baseline:                  0.05
		daily_traffic_per_variant: 500
		annual_revenue:            5_000_000.0
		day_cost:                  500.0
		alpha:                     0.05
		min_power:                 0.80
		seasonality_min_days:      14
		prior:                     experiment.MixturePrior{
			null_frac: 0.40
			neg_frac:  0.30
			neg_mean:  -0.01
			neg_std:   0.005
			pos_mean:  0.01
			pos_std:   0.005
			n_samples: 5_000
		}
		max_days: 90
		seed:     42
	}
	result := experiment.find_optimal_runtime(config)
	assert result.worth_running == true
	assert result.no_go_reason == ''
	assert result.power_at_optimal >= config.min_power
}
