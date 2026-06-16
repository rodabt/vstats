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
pub struct ForecastResult {
pub:
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

// arima_forecast produces h-step ahead forecasts with (1-alpha) confidence intervals.
// original_x is the undifferenced series (needed for undiff reconstruction).
pub fn arima_forecast(model ARIMAModel, h int, alpha f64, original_x []f64) ForecastResult {
	p := model.p
	q := model.q
	n := model.residuals.len
	x_diff := diff(original_x, model.d)
	mut hist_x := []f64{}
	start_x := if x_diff.len - p > 0 { x_diff.len - p } else { 0 }
	for i in start_x .. x_diff.len {
		hist_x << x_diff[i]
	}
	mut hist_e := []f64{}
	start_e := if n - q > 0 { n - q } else { 0 }
	for i in start_e .. n {
		hist_e << model.residuals[i]
	}

	mut fc_diff := []f64{len: h}
	for t in 0 .. h {
		mut xhat := model.intercept
		for j in 0 .. p {
			if t - j - 1 >= 0 {
				xhat += model.ar_coeffs[j] * fc_diff[t - j - 1]
			} else {
				idx := hist_x.len - (j - t + 1)
				if idx >= 0 {
					xhat += model.ar_coeffs[j] * hist_x[idx]
				}
			}
		}
		if t == 0 {
			for j in 0 .. q {
				idx := hist_e.len - j - 1
				if idx >= 0 {
					xhat -= model.ma_coeffs[j] * hist_e[idx]
				}
			}
		}
		fc_diff[t] = xhat
	}

	fc_orig := if model.d == 0 { fc_diff } else { undiff(fc_diff, original_x, model.d) }

	z := 1.96
	mut lower := []f64{len: h}
	mut upper := []f64{len: h}
	for t in 0 .. h {
		se := math.sqrt(model.sigma2 * f64(t + 1))
		lower[t] = fc_orig[t] - z * se
		upper[t] = fc_orig[t] + z * se
	}

	return ForecastResult{
		forecast: fc_orig
		lower:    lower
		upper:    upper
		alpha:    alpha
	}
}

// arima_summary returns a formatted model summary string.
pub fn arima_summary(model ARIMAModel) string {
	mut sb := '================================\n'
	sb += 'ARIMA(${model.p}, ${model.d}, ${model.q}) Results\n'
	sb += '================================\n'
	sb += 'Intercept: ${model.intercept:.4f}\n'
	if model.p > 0 {
		sb += 'AR Coefficients:\n'
		for i, c in model.ar_coeffs {
			sb += '  phi[${i + 1}] = ${c:.4f}\n'
		}
	}
	if model.q > 0 {
		sb += 'MA Coefficients:\n'
		for i, c in model.ma_coeffs {
			sb += '  theta[${i + 1}] = ${c:.4f}\n'
		}
	}
	sb += 'Sigma^2: ${model.sigma2:.4f}\n'
	sb += 'AIC:     ${model.aic:.4f}\n'
	sb += 'BIC:     ${model.bic:.4f}\n'
	sb += 'AICc:    ${model.aicc:.4f}\n'
	return sb
}

// sarima_fit fits SARIMA(p,d,q)(sp,sd,sq,m) by applying seasonal differencing first,
// then fitting with augmented AR/MA orders to cover seasonal lags.
pub fn sarima_fit(x []f64, p int, d int, q int, sp int, sd int, sq int, m int) ARIMAModel {
	mut xs := x.clone()
	sd_count := sd * m  // Total observations lost to seasonal differencing
	for _ in 0 .. sd {
		xs = seasonal_diff(xs, m)
	}
	effective_p := if sp > 0 { p + sp * m } else { p }
	effective_q := if sq > 0 { q + sq * m } else { q }
	model := arima_fit(xs, effective_p, d, effective_q)

	// Extend fitted and residuals back to original length
	// First sd*m positions use original series (no prediction error)
	if sd_count > 0 {
		mut extended_fitted := []f64{len: x.len}
		mut extended_residuals := []f64{len: x.len}
		for i in 0 .. sd_count {
			extended_fitted[i] = x[i]
			extended_residuals[i] = 0.0
		}
		for i in 0 .. model.fitted.len {
			extended_fitted[sd_count + i] = model.fitted[i]
			extended_residuals[sd_count + i] = model.residuals[i]
		}
		return ARIMAModel{
			p:         model.p
			d:         model.d
			q:         model.q
			ar_coeffs: model.ar_coeffs
			ma_coeffs: model.ma_coeffs
			intercept: model.intercept
			sigma2:    model.sigma2
			fitted:    extended_fitted
			residuals: extended_residuals
			aic:       model.aic
			bic:       model.bic
			aicc:      model.aicc
		}
	}

	return model
}

// auto_arima performs grid search over ARIMA(p,d,q) orders up to max_p, max_d, max_q.
// Uses ADF test to determine minimum d, capped at max_d.
// Selects the order with lowest AICc.
pub fn auto_arima(x []f64, max_p int, max_d int, max_q int) ARIMAModel {
	// Determine d via ADF
	mut best_d := 0
	mut xs := x.clone()
	for d_try in 0 .. max_d + 1 {
		adf := adf_test(xs, 1)
		if adf.is_stationary {
			best_d = d_try
			break
		}
		best_d = d_try + 1
		if d_try < max_d {
			xs = diff(xs, 1)
		}
	}
	if best_d > max_d {
		best_d = max_d
	}

	mut best_model := arima_fit(x, 0, best_d, 0)
	mut best_aicc := best_model.aicc

	for p in 0 .. max_p + 1 {
		for q in 0 .. max_q + 1 {
			if p == 0 && q == 0 {
				continue
			}
			candidate := arima_fit(x, p, best_d, q)
			if candidate.aicc < best_aicc {
				best_aicc = candidate.aicc
				best_model = candidate
			}
		}
	}
	return best_model
}
