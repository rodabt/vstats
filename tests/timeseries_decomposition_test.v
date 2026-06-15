import timeseries
import math

fn test__classical_decompose_additive_recovers_trend() {
	// Linear trend + constant seasonal + zero noise
	mut x := []f64{len: 24}
	for i in 0 .. 24 {
		x[i] = f64(i) + [1.0, -1.0, 2.0, -2.0][i % 4]
	}
	result := timeseries.decompose(x, 4, .additive)
	// Trend should be approximately linear; check middle values
	for i in 4 .. 20 {
		if result.trend[i] != 0.0 {
			assert math.abs(result.trend[i] - f64(i)) < 1.5
		}
	}
	assert result.model == .additive
}

fn test__classical_decompose_multiplicative() {
	// Multiplicative: x[t] = trend * seasonal
	mut x := []f64{len: 24}
	for i in 0 .. 24 {
		x[i] = f64(i + 1) * [1.1, 0.9, 1.2, 0.8][i % 4]
	}
	result := timeseries.decompose(x, 4, .multiplicative)
	assert result.model == .multiplicative
	// Residuals should be close to 1.0 for multiplicative decomposition
	for i in 2 .. 22 {
		if result.residual[i] != 0.0 {
			assert math.abs(result.residual[i] - 1.0) < 0.5
		}
	}
}

fn test__classical_decompose_components_add_up() {
	mut x := []f64{len: 24}
	for i in 0 .. 24 {
		x[i] = f64(i) + [2.0, -1.0, 3.0, -4.0][i % 4]
	}
	result := timeseries.decompose(x, 4, .additive)
	for i in 0 .. 24 {
		if result.trend[i] != 0.0 {
			reconstructed := result.trend[i] + result.seasonal[i] + result.residual[i]
			assert math.abs(reconstructed - x[i]) < 0.001
		}
	}
}
