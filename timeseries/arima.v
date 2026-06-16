module timeseries

import math
import stats

pub struct ARIMAModel {
pub:
	p         int
	d         int
	q         int
	ar_coeffs []f64
	ma_coeffs []f64
	intercept f64
	sigma2    f64
	fitted    []f64
	residuals []f64
	aic       f64
	bic       f64
	aicc      f64
}

// ForecastResult holds point forecasts and prediction intervals.
// Populated by arima_forecast (Task 10).
struct ForecastResult {
	forecast []f64
	lower    []f64
	upper    []f64
	alpha    f64
}

// css_residuals computes ARMA residuals recursively given coefficients.
fn css_residuals(x []f64, ar []f64, ma []f64, intercept f64) []f64 {
	p := ar.len
	q := ma.len
	n := x.len
	mut resid := []f64{len: n}
	for t in 0 .. n {
		mut xhat := intercept
		for j in 0 .. p {
			if t - j - 1 >= 0 {
				xhat += ar[j] * x[t - j - 1]
			}
		}
		for j in 0 .. q {
			if t - j - 1 >= 0 {
				xhat -= ma[j] * resid[t - j - 1]
			}
		}
		resid[t] = x[t] - xhat
	}
	return resid
}

// css_loss returns the Conditional Sum of Squares for params on series x.
// params layout: [ar_0, ..., ar_{p-1}, ma_0, ..., ma_{q-1}, intercept]
fn css_loss(params []f64, x []f64, p int, q int) f64 {
	ar := params[0..p].clone()
	ma := params[p..p + q].clone()
	intercept := params[p + q]
	resid := css_residuals(x, ar, ma, intercept)
	mut ss := 0.0
	start := if p > q { p } else { q }
	for t in start .. resid.len {
		ss += resid[t] * resid[t]
	}
	return ss
}

// css_gradient computes numerical gradient of css_loss.
fn css_gradient(params []f64, x []f64, p int, q int, h f64) []f64 {
	f0 := css_loss(params, x, p, q)
	mut grad := []f64{len: params.len}
	for i in 0 .. params.len {
		mut p2 := params.clone()
		p2[i] += h
		grad[i] = (css_loss(p2, x, p, q) - f0) / h
	}
	return grad
}

// arima_fit fits ARIMA(p, d, q) to series x using CSS estimation.
pub fn arima_fit(x []f64, p int, d int, q int) ARIMAModel {
	x_diff := diff(x, d)
	n := x_diff.len
	n_params := p + q + 1

	mut params := []f64{len: n_params}
	params[n_params - 1] = stats.mean(x_diff)

	// Initialize AR/MA coefficients with ACF hints
	if p > 0 {
		acf_vals := acf(x_diff, p)
		for i in 0 .. p {
			params[i] = acf_vals[i] * 0.5
		}
	}

	mut lr := 0.001
	mut prev_loss := css_loss(params, x_diff, p, q)
	for iter in 0 .. 1000 {
		grad := css_gradient(params, x_diff, p, q, 1e-5)
		for i in 0 .. n_params {
			params[i] -= lr * grad[i]
		}
		for i in 0 .. p {
			if params[i] > 0.99 {
				params[i] = 0.99
			}
			if params[i] < -0.99 {
				params[i] = -0.99
			}
		}
		for i in p .. p + q {
			if params[i] > 0.99 {
				params[i] = 0.99
			}
			if params[i] < -0.99 {
				params[i] = -0.99
			}
		}
		if iter % 100 == 99 {
			lr *= 0.8
		}
		if iter % 50 == 49 {
			cur_loss := css_loss(params, x_diff, p, q)
			if math.abs(cur_loss - prev_loss) < 1e-8 {
				break
			}
			prev_loss = cur_loss
		}
	}

	ar_coeffs := params[0..p].clone()
	ma_coeffs := params[p..p + q].clone()
	intercept := params[n_params - 1]

	resid_diff := css_residuals(x_diff, ar_coeffs, ma_coeffs, intercept)
	mut fitted_diff := []f64{len: n}
	for t in 0 .. n {
		fitted_diff[t] = x_diff[t] - resid_diff[t]
	}

	// Build fitted and residuals in the ORIGINAL scale, both of length x.len.
	// The first d positions (consumed by differencing) have no prediction: use x[t] so residual = 0.
	// For d > 0, in-sample fitted values are recovered by integrating fitted_diff forward
	// from the last pre-differencing seed x[d-1] (i.e. cumsum, not undiff which seeds from end).
	x_len := x.len
	mut fitted := []f64{len: x_len}
	mut resid := []f64{len: x_len}
	for t in 0 .. d {
		fitted[t] = x[t]
		resid[t] = 0.0
	}
	if d == 0 {
		for t in 0 .. n {
			fitted[t] = fitted_diff[t]
			resid[t] = resid_diff[t]
		}
	} else {
		// Integrate fitted_diff back to original scale via d nested cumsums.
		// Each integration step k (from d down to 1) uses the last value of
		// diff(x, k-1) as its seed, which is diff(x, k-1)[0] = the first
		// element of the (k-1)-times differenced x.
		mut cur := fitted_diff.clone()
		for k in 1 .. d + 1 {
			seed_series := diff(x, d - k)
			seed := seed_series[0]
			mut next := []f64{len: cur.len}
			next[0] = seed + cur[0]
			for t in 1 .. cur.len {
				next[t] = next[t - 1] + cur[t]
			}
			cur = next.clone()
		}
		// cur is now in original scale, length n = x.len - d
		for t in 0 .. n {
			fitted[d + t] = cur[t]
			resid[d + t] = x[d + t] - cur[t]
		}
	}

	start := if p > q { p } else { q }
	mut ss := 0.0
	for t in start .. n {
		ss += resid_diff[t] * resid_diff[t]
	}
	effective_n := n - start
	sigma2 := ss / f64(effective_n)
	log_lik := -0.5 * f64(effective_n) * (1.0 + math.log(2.0 * math.pi * sigma2))

	k := n_params
	return ARIMAModel{
		p:         p
		d:         d
		q:         q
		ar_coeffs: ar_coeffs
		ma_coeffs: ma_coeffs
		intercept: intercept
		sigma2:    sigma2
		fitted:    fitted
		residuals: resid
		aic:       aic(log_lik, k)
		bic:       bic(log_lik, k, n)
		aicc:      aicc(log_lik, k, n)
	}
}
