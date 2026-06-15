module timeseries

import math

pub enum HWSeasonalType {
	additive
	multiplicative
}

pub struct SmoothingResult {
pub:
	fitted   []f64
	forecast []f64
	level    []f64
	trend    []f64  // empty for SES
	seasonal []f64  // empty for SES/Holt
	alpha    f64
	beta     f64
	gamma    f64
	mse      f64
}

// ses fits Simple Exponential Smoothing with smoothing parameter alpha in (0,1).
// Returns 1-step-ahead in-sample fitted values and a 1-step forecast.
pub fn ses(x []f64, alpha f64) SmoothingResult {
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0,1)'
	n := x.len
	mut level := []f64{len: n}
	mut fitted := []f64{len: n}
	level[0] = x[0]
	fitted[0] = x[0]
	for t in 1 .. n {
		fitted[t] = level[t - 1]
		level[t] = alpha * x[t] + (1.0 - alpha) * level[t - 1]
	}
	mut ss := 0.0
	for t in 1 .. n {
		d := x[t] - fitted[t]
		ss += d * d
	}
	return SmoothingResult{
		fitted:   fitted
		forecast: [level[n - 1]]
		level:    level
		trend:    []f64{}
		seasonal: []f64{}
		alpha:    alpha
		beta:     0.0
		gamma:    0.0
		mse:      ss / f64(n - 1)
	}
}

// holt fits Holt's double exponential smoothing (trend-corrected).
// alpha smooths the level; beta smooths the trend.
pub fn holt(x []f64, alpha f64, beta f64) SmoothingResult {
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0,1)'
	assert beta > 0.0 && beta < 1.0, 'beta must be in (0,1)'
	n := x.len
	mut level := []f64{len: n}
	mut trend := []f64{len: n}
	mut fitted := []f64{len: n}
	level[0] = x[0]
	trend[0] = if n > 1 { x[1] - x[0] } else { 0.0 }
	fitted[0] = x[0]
	for t in 1 .. n {
		fitted[t] = level[t - 1] + trend[t - 1]
		level[t] = alpha * x[t] + (1.0 - alpha) * (level[t - 1] + trend[t - 1])
		trend[t] = beta * (level[t] - level[t - 1]) + (1.0 - beta) * trend[t - 1]
	}
	mut ss := 0.0
	for t in 1 .. n {
		d := x[t] - fitted[t]
		ss += d * d
	}
	return SmoothingResult{
		fitted:   fitted
		forecast: [level[n - 1] + trend[n - 1]]
		level:    level
		trend:    trend
		seasonal: []f64{}
		alpha:    alpha
		beta:     beta
		gamma:    0.0
		mse:      ss / f64(n - 1)
	}
}

// holt_winters fits triple exponential smoothing (trend + seasonality).
pub fn holt_winters(x []f64, alpha f64, beta f64, gamma f64, period int, seasonal HWSeasonalType) SmoothingResult {
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0,1)'
	assert beta > 0.0 && beta < 1.0, 'beta must be in (0,1)'
	assert gamma > 0.0 && gamma < 1.0, 'gamma must be in (0,1)'
	n := x.len
	assert n >= 2 * period, 'need at least 2 full seasons'

	mut level := []f64{len: n}
	mut trend_arr := []f64{len: n}
	mut seas := []f64{len: n + period}
	mut fitted := []f64{len: n}

	mut s1 := 0.0
	mut s2 := 0.0
	for i in 0 .. period {
		s1 += x[i]
		s2 += x[i + period]
	}
	level[0] = s1 / f64(period)
	trend_arr[0] = (s2 - s1) / f64(period * period)

	mut season_mean := 0.0
	for i in 0 .. period {
		season_mean += x[i]
	}
	season_mean /= f64(period)
	for i in 0 .. period {
		seas[i] = if seasonal == .additive {
			x[i] - season_mean
		} else {
			if season_mean != 0.0 { x[i] / season_mean } else { 1.0 }
		}
	}
	fitted[0] = x[0]

	for t in 1 .. n {
		s_tm := seas[t]
		fitted[t] = if seasonal == .additive {
			level[t - 1] + trend_arr[t - 1] + s_tm
		} else {
			(level[t - 1] + trend_arr[t - 1]) * s_tm
		}
		if seasonal == .additive {
			level[t] = alpha * (x[t] - s_tm) + (1.0 - alpha) * (level[t - 1] + trend_arr[t - 1])
			seas[t + period] = gamma * (x[t] - level[t]) + (1.0 - gamma) * s_tm
		} else {
			level[t] = alpha * (if s_tm != 0.0 { x[t] / s_tm } else { x[t] }) +
				(1.0 - alpha) * (level[t - 1] + trend_arr[t - 1])
			seas[t + period] = gamma * (if level[t] != 0.0 { x[t] / level[t] } else { 1.0 }) +
				(1.0 - gamma) * s_tm
		}
		trend_arr[t] = beta * (level[t] - level[t - 1]) + (1.0 - beta) * trend_arr[t - 1]
	}

	s_next := seas[n]
	forecast_val := if seasonal == .additive {
		level[n - 1] + trend_arr[n - 1] + s_next
	} else {
		(level[n - 1] + trend_arr[n - 1]) * s_next
	}

	mut ss := 0.0
	for t in 1 .. n {
		d := x[t] - fitted[t]
		ss += d * d
	}

	return SmoothingResult{
		fitted:   fitted
		forecast: [forecast_val]
		level:    level
		trend:    trend_arr
		seasonal: seas[0..n].clone()
		alpha:    alpha
		beta:     beta
		gamma:    gamma
		mse:      ss / f64(n - 1)
	}
}

// nelder_mead minimises f over `dim` parameters bounded in (lo, hi) per dimension.
fn nelder_mead(f fn ([]f64) f64, dim int, lo f64, hi f64, max_iter int) []f64 {
	center := (lo + hi) / 2.0
	step := (hi - lo) * 0.05
	mut simplex := [][]f64{len: dim + 1, init: []f64{len: dim}}
	for i in 0 .. dim + 1 {
		for j in 0 .. dim {
			simplex[i][j] = center
		}
		if i > 0 {
			simplex[i][i - 1] = center + step
		}
	}

	alpha_nm := 1.0
	gamma_nm := 2.0
	rho_nm := 0.5
	sigma_nm := 0.5

	for _ in 0 .. max_iter {
		mut vals := simplex.map(f(it))
		for i in 0 .. dim + 1 {
			for j in i + 1 .. dim + 1 {
				if vals[j] < vals[i] {
					tmp := vals[i]
					vals[i] = vals[j]
					vals[j] = tmp
					tmp2 := simplex[i].clone()
					simplex[i] = simplex[j].clone()
					simplex[j] = tmp2
				}
			}
		}
		mut centroid := []f64{len: dim}
		for i in 0 .. dim {
			for j in 0 .. dim {
				centroid[j] += simplex[i][j]
			}
		}
		for j in 0 .. dim {
			centroid[j] /= f64(dim)
		}
		mut reflected := []f64{len: dim}
		for j in 0 .. dim {
			reflected[j] = centroid[j] + alpha_nm * (centroid[j] - simplex[dim][j])
			reflected[j] = math.min(hi - 0.001, math.max(lo + 0.001, reflected[j]))
		}
		f_reflected := f(reflected)
		if f_reflected < vals[0] {
			mut expanded := []f64{len: dim}
			for j in 0 .. dim {
				expanded[j] = centroid[j] + gamma_nm * (reflected[j] - centroid[j])
				expanded[j] = math.min(hi - 0.001, math.max(lo + 0.001, expanded[j]))
			}
			if f(expanded) < f_reflected {
				simplex[dim] = expanded
			} else {
				simplex[dim] = reflected
			}
		} else if f_reflected < vals[dim - 1] {
			simplex[dim] = reflected
		} else {
			mut contracted := []f64{len: dim}
			for j in 0 .. dim {
				contracted[j] = centroid[j] + rho_nm * (simplex[dim][j] - centroid[j])
				contracted[j] = math.min(hi - 0.001, math.max(lo + 0.001, contracted[j]))
			}
			if f(contracted) < vals[dim] {
				simplex[dim] = contracted
			} else {
				for i in 1 .. dim + 1 {
					for j in 0 .. dim {
						simplex[i][j] = simplex[0][j] + sigma_nm * (simplex[i][j] - simplex[0][j])
						simplex[i][j] = math.min(hi - 0.001, math.max(lo + 0.001, simplex[i][j]))
					}
				}
			}
		}
	}
	return simplex[0]
}

pub fn auto_ses(x []f64) SmoothingResult {
	best := nelder_mead(fn [x] (p []f64) f64 {
		return ses(x, p[0]).mse
	}, 1, 0.01, 0.99, 200)
	return ses(x, best[0])
}

pub fn auto_holt(x []f64) SmoothingResult {
	best := nelder_mead(fn [x] (p []f64) f64 {
		return holt(x, p[0], p[1]).mse
	}, 2, 0.01, 0.99, 300)
	return holt(x, best[0], best[1])
}

pub fn auto_holt_winters(x []f64, period int, seasonal HWSeasonalType) SmoothingResult {
	best := nelder_mead(fn [x, period, seasonal] (p []f64) f64 {
		return holt_winters(x, p[0], p[1], p[2], period, seasonal).mse
	}, 3, 0.01, 0.99, 400)
	return holt_winters(x, best[0], best[1], best[2], period, seasonal)
}
