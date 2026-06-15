module timeseries

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
