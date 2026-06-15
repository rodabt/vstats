module timeseries

import math

pub enum DecompositionModel {
	additive
	multiplicative
}

pub struct ClassicalDecomposition {
pub:
	trend    []f64
	seasonal []f64
	residual []f64
	model    DecompositionModel
}

// decompose performs classical time series decomposition into trend, seasonal, residual.
// Uses centered moving average for trend; period averages for seasonal component.
// Positions where trend cannot be computed (edges) are set to 0.0.
pub fn decompose(x []f64, period int, model DecompositionModel) ClassicalDecomposition {
	n := x.len
	// Centered moving average trend
	half := period / 2
	mut trend := []f64{len: n}
	for t in half .. n - half {
		mut s := 0.0
		if period % 2 == 0 {
			// Even period: average of half-period on each side with 0.5 weight at ends
			s += 0.5 * x[t - half]
			for k in t - half + 1 .. t + half {
				s += x[k]
			}
			s += 0.5 * x[t + half]
			trend[t] = s / f64(period)
		} else {
			for k in t - half .. t + half + 1 {
				s += x[k]
			}
			trend[t] = s / f64(period)
		}
	}

	// De-trend
	mut detrended := []f64{len: n}
	for t in 0 .. n {
		if trend[t] == 0.0 {
			continue
		}
		detrended[t] = if model == .additive { x[t] - trend[t] } else { x[t] / trend[t] }
	}

	// Average by seasonal position
	mut season_sums := []f64{len: period}
	mut season_counts := []int{len: period}
	for t in half .. n - half {
		season_sums[t % period] += detrended[t]
		season_counts[t % period]++
	}
	mut season_avgs := []f64{len: period}
	for s in 0 .. period {
		season_avgs[s] = if season_counts[s] > 0 {
			season_sums[s] / f64(season_counts[s])
		} else {
			0.0
		}
	}
	// Normalise additive seasonal so it sums to zero
	if model == .additive {
		mut s_mean := 0.0
		for v in season_avgs {
			s_mean += v
		}
		s_mean /= f64(period)
		for s in 0 .. period {
			season_avgs[s] -= s_mean
		}
	} else {
		// Multiplicative: normalise so seasonal factors average to 1
		mut s_mean := 0.0
		for v in season_avgs {
			s_mean += v
		}
		s_mean /= f64(period)
		for s in 0 .. period {
			season_avgs[s] /= s_mean
		}
	}

	mut seasonal := []f64{len: n}
	for t in 0 .. n {
		seasonal[t] = season_avgs[t % period]
	}

	// Residual
	mut residual := []f64{len: n}
	for t in 0 .. n {
		if trend[t] == 0.0 {
			continue
		}
		residual[t] = if model == .additive {
			x[t] - trend[t] - seasonal[t]
		} else {
			if trend[t] * seasonal[t] != 0.0 { x[t] / (trend[t] * seasonal[t]) } else { 0.0 }
		}
	}

	return ClassicalDecomposition{
		trend:    trend
		seasonal: seasonal
		residual: residual
		model:    model
	}
}

pub struct STLConfig {
pub:
	seasonal_window int // width of LOESS smoother for seasonal step (odd integer)
	trend_window    int // width of LOESS smoother for trend step (odd integer)
	n_iter          int = 2 // number of outer robustness iterations
}

pub struct STLResult {
pub:
	trend    []f64
	seasonal []f64
	residual []f64
}

// loess fits a local linear regression at each point using tricubic weights.
// window must be odd; points at the edges use the nearest available window.
fn loess(t []f64, y []f64, window int, weights []f64) []f64 {
	n := y.len
	half := window / 2
	mut fitted := []f64{len: n}
	for i in 0 .. n {
		lo := if i - half < 0 { 0 } else { i - half }
		hi := if i + half >= n { n - 1 } else { i + half }
		max_dist := f64(if i - lo > hi - i { i - lo } else { hi - i }) + 1.0
		mut w_sum := 0.0
		mut wt := 0.0
		mut wx := 0.0
		mut wxx := 0.0
		mut wy := 0.0
		mut wxy := 0.0
		for j in lo .. hi + 1 {
			u := math.abs(t[j] - t[i]) / max_dist
			tri := if u < 1.0 {
				rb := 1.0 - u * u * u
				rb * rb * rb
			} else {
				0.0
			}
			w := tri * weights[j]
			w_sum += w
			wt += w
			wx += w * t[j]
			wxx += w * t[j] * t[j]
			wy += w * y[j]
			wxy += w * t[j] * y[j]
		}
		if w_sum < 1e-12 {
			fitted[i] = y[i]
			continue
		}
		// Weighted linear regression: [beta0, beta1]
		denom := wt * wxx - wx * wx
		if math.abs(denom) < 1e-12 {
			fitted[i] = wy / w_sum
		} else {
			beta0 := (wxx * wy - wx * wxy) / denom
			beta1 := (wt * wxy - wx * wy) / denom
			fitted[i] = beta0 + beta1 * t[i]
		}
	}
	return fitted
}

// stl decomposes x into trend, seasonal, and residual using iterative LOESS.
pub fn stl(x []f64, period int, cfg STLConfig) STLResult {
	n := x.len
	mut t_index := []f64{len: n}
	for i in 0 .. n {
		t_index[i] = f64(i)
	}
	mut ones := []f64{len: n}
	for i in 0 .. n {
		ones[i] = 1.0
	}

	mut seasonal := []f64{len: n}
	mut trend := []f64{len: n}
	mut rob_weights := ones.clone()

	n_iter := if cfg.n_iter < 1 { 2 } else { cfg.n_iter }

	for _ in 0 .. n_iter {
		// Step 1: Detrend
		mut detrended := []f64{len: n}
		for i in 0 .. n {
			detrended[i] = x[i] - trend[i]
		}
		// Step 2: Smooth each subseries (one per seasonal position)
		for s in 0 .. period {
			mut sub_t := []f64{}
			mut sub_y := []f64{}
			mut sub_w := []f64{}
			for i in s .. n {
				if i % period == s {
					sub_t << f64(i)
					sub_y << detrended[i]
					sub_w << rob_weights[i]
				}
			}
			smoothed := loess(sub_t, sub_y, if cfg.seasonal_window < 3 { 7 } else { cfg.seasonal_window }, sub_w)
			mut idx := 0
			for i in s .. n {
				if i % period == s {
					seasonal[i] = smoothed[idx]
					idx++
				}
			}
		}
		// Step 3: Deseasonalise and smooth trend
		mut deseasonalised := []f64{len: n}
		for i in 0 .. n {
			deseasonalised[i] = x[i] - seasonal[i]
		}
		trend = loess(t_index, deseasonalised,
			if cfg.trend_window < 3 { 2 * period + 1 } else { cfg.trend_window }, rob_weights)
		// Step 4: Update robustness weights from residuals
		mut residual := []f64{len: n}
		for i in 0 .. n {
			residual[i] = x[i] - trend[i] - seasonal[i]
		}
		// h = 6 * median(|residual|)
		mut abs_resid := residual.map(math.abs(it))
		abs_resid.sort()
		median_r := abs_resid[n / 2]
		h := 6.0 * median_r + 1e-10
		for i in 0 .. n {
			u := math.abs(residual[i]) / h
			rob_weights[i] = if u < 1.0 {
				rb := 1.0 - u * u
				rb * rb
			} else {
				0.0
			}
		}
	}

	mut residual := []f64{len: n}
	for i in 0 .. n {
		residual[i] = x[i] - trend[i] - seasonal[i]
	}

	return STLResult{
		trend:    trend
		seasonal: seasonal
		residual: residual
	}
}
