import experiment
import math

fn test__compute_design_point_standard_case() {
	// alpha=0.05 one-sided, beta=0.20 (power=0.80), baseline=0.50, mde=0.05
	// z_alpha=1.6449, z_beta=0.8416
	// n = (1.6449+0.8416)^2 * 2 * 0.50 * 0.50 / 0.05^2 = 1236.54 -> 1237
	dp := experiment.compute_design_point(0.05, 0.20, 0.05, 0.50, 1000)
	assert math.abs(f64(dp.n_per_arm) - 1237) <= 5
	assert dp.runtime_days == 2 // ceil(1237/1000)
	assert dp.alpha == 0.05
	assert dp.beta == 0.20
	assert dp.mde == 0.05
}

fn test__compute_design_point_tighter_alpha() {
	// alpha=0.10 one-sided, beta=0.20, baseline=0.30, mde=0.03
	// z_alpha=1.2816, z_beta=0.8416
	// n = (2.1232)^2 * 2 * 0.30 * 0.70 / 0.03^2 = 2103.7 -> 2104
	dp := experiment.compute_design_point(0.10, 0.20, 0.03, 0.30, 500)
	assert math.abs(f64(dp.n_per_arm) - 2104) <= 10
	assert dp.runtime_days == 5 // ceil(2104/500)
}
