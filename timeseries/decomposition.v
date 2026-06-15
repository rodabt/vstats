module timeseries

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
