module timeseries

import math
import linalg

pub struct VARModel {
pub:
	p              int
	k              int
	// coeff_matrices[eq] = [intercept, y1_lag1, y2_lag1, ..., y1_lag2, y2_lag2, ...]
	// Shape per equation: k*p + 1 coefficients
	coeff_matrices [][]f64
	sigma_u        [][]f64
	fitted         [][]f64   // [k][T-p]
	residuals      [][]f64   // [k][T-p]
	aic            f64
	bic            f64
	hqc            f64
}

pub struct VARLagSelection {
pub:
	aic_order int
	bic_order int
	hqc_order int
	criteria  [][]f64   // [max_p][3]: AIC, BIC, HQC per lag order
}

pub struct GrangerResult {
pub:
	cause_var   int
	effect_var  int
	f_statistic f64
	p_value     f64
	significant bool
}

// var_build_design builds the OLS design matrix for VAR(p).
// Returns X (T-p × k*p+1) and y_all (k × T-p).
// Design layout: [1, y1_{t-1}, y2_{t-1}, ..., yk_{t-1}, y1_{t-2}, ..., yk_{t-p}]
fn var_build_design(data [][]f64, p int) ([][]f64, [][]f64) {
	k := data.len
	t_total := data[0].len
	t_eff := t_total - p
	mut design := [][]f64{len: t_eff, init: []f64{len: k * p + 1}}
	mut y_all := [][]f64{len: k, init: []f64{len: t_eff}}
	for t in 0 .. t_eff {
		design[t][0] = 1.0
		for lag in 1 .. p + 1 {
			for j in 0 .. k {
				design[t][lag * k - k + j + 1] = data[j][t + p - lag]
			}
		}
		for j in 0 .. k {
			y_all[j][t] = data[j][t + p]
		}
	}
	return design, y_all
}

// var_fit estimates a VAR(p) model by OLS equation-by-equation.
pub fn var_fit(data [][]f64, p int) VARModel {
	k := data.len
	t_total := data[0].len
	t_eff := t_total - p
	design, y_all := var_build_design(data, p)
	xt := linalg.transpose(design)
	xtx := linalg.matmul(xt, design)

	mut coeff_matrices := [][]f64{len: k}
	mut fitted_all := [][]f64{len: k, init: []f64{len: t_eff}}
	mut resid_all := [][]f64{len: k, init: []f64{len: t_eff}}

	for eq in 0 .. k {
		xty := linalg.matvec_mul(xt, y_all[eq])
		beta := linalg.gaussian_elimination(xtx, xty)
		coeff_matrices[eq] = beta
		for t in 0 .. t_eff {
			mut yhat := 0.0
			for j in 0 .. beta.len {
				yhat += design[t][j] * beta[j]
			}
			fitted_all[eq][t] = yhat
			resid_all[eq][t] = y_all[eq][t] - yhat
		}
	}

	// Residual covariance matrix sigma_u
	mut sigma_u := [][]f64{len: k, init: []f64{len: k}}
	dof := t_eff - k * p - 1
	for i in 0 .. k {
		for j in 0 .. k {
			mut cov := 0.0
			for t in 0 .. t_eff {
				cov += resid_all[i][t] * resid_all[j][t]
			}
			sigma_u[i][j] = if dof > 0 { cov / f64(dof) } else { cov / f64(t_eff) }
		}
	}

	// Approximate log-likelihood
	n_params := k * (k * p + 1)
	mut log_det := 0.0
	for i in 0 .. k {
		if sigma_u[i][i] > 0.0 {
			log_det += math.log(sigma_u[i][i])
		}
	}
	log_lik := -0.5 * f64(t_eff) * (f64(k) * (1.0 + math.log(2.0 * math.pi)) + log_det)

	hqc_val := 2.0 * f64(n_params) * math.log(math.log(f64(t_eff))) - 2.0 * log_lik

	return VARModel{
		p:              p
		k:              k
		coeff_matrices: coeff_matrices
		sigma_u:        sigma_u
		fitted:         fitted_all
		residuals:      resid_all
		aic:            aic(log_lik, n_params)
		bic:            bic(log_lik, n_params, t_eff)
		hqc:            hqc_val
	}
}

// chi2_cdf_approx computes chi-squared CDF using Wilson-Hilferty approximation
fn chi2_cdf_approx(x f64, k int) f64 {
	if x <= 0.0 {
		return 0.0
	}
	kf := f64(k)
	z := (math.pow(x / kf, 1.0 / 3.0) - (1.0 - 2.0 / (9.0 * kf))) / math.sqrt(2.0 / (9.0 * kf))
	// Standard normal CDF via erf
	return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
}

// var_select_lag evaluates VAR(1)..VAR(max_p) and returns AIC/BIC/HQC-optimal orders.
pub fn var_select_lag(data [][]f64, max_p int) VARLagSelection {
	mut criteria := [][]f64{len: max_p, init: []f64{len: 3}}
	mut best_aic := math.inf(1)
	mut best_bic := math.inf(1)
	mut best_hqc := math.inf(1)
	mut aic_order := 1
	mut bic_order := 1
	mut hqc_order := 1
	for p in 1 .. max_p + 1 {
		m := var_fit(data, p)
		criteria[p - 1][0] = m.aic
		criteria[p - 1][1] = m.bic
		criteria[p - 1][2] = m.hqc
		if m.aic < best_aic {
			best_aic = m.aic
			aic_order = p
		}
		if m.bic < best_bic {
			best_bic = m.bic
			bic_order = p
		}
		if m.hqc < best_hqc {
			best_hqc = m.hqc
			hqc_order = p
		}
	}
	return VARLagSelection{
		aic_order: aic_order
		bic_order: bic_order
		hqc_order: hqc_order
		criteria:  criteria
	}
}

// granger_causality tests whether cause_idx Granger-causes effect_idx using VAR(p).
// Uses an F-test comparing unrestricted vs restricted (cause lags excluded) models.
pub fn granger_causality(data [][]f64, p int, cause_idx int, effect_idx int) GrangerResult {
	k := data.len
	t_total := data[0].len
	t_eff := t_total - p

	// Unrestricted: full VAR equation for effect_idx
	design_u, y_all := var_build_design(data, p)
	xt_u := linalg.transpose(design_u)
	xtx_u := linalg.matmul(xt_u, design_u)
	xty_u := linalg.matvec_mul(xt_u, y_all[effect_idx])
	beta_u := linalg.gaussian_elimination(xtx_u, xty_u)
	mut rss_u := 0.0
	for t in 0 .. t_eff {
		mut yhat := 0.0
		for j in 0 .. beta_u.len {
			yhat += design_u[t][j] * beta_u[j]
		}
		d := y_all[effect_idx][t] - yhat
		rss_u += d * d
	}

	// Restricted: remove p lags of cause_idx from design matrix
	mut keep_cols := [0]
	for lag in 1 .. p + 1 {
		for j in 0 .. k {
			col := lag * k - k + j + 1
			if j != cause_idx {
				keep_cols << col
			}
		}
	}
	n_cols_r := keep_cols.len
	mut design_r := [][]f64{len: t_eff, init: []f64{len: n_cols_r}}
	for t in 0 .. t_eff {
		for ci, c in keep_cols {
			design_r[t][ci] = design_u[t][c]
		}
	}
	xt_r := linalg.transpose(design_r)
	xtx_r := linalg.matmul(xt_r, design_r)
	xty_r := linalg.matvec_mul(xt_r, y_all[effect_idx])
	beta_r := linalg.gaussian_elimination(xtx_r, xty_r)
	mut rss_r := 0.0
	for t in 0 .. t_eff {
		mut yhat := 0.0
		for ci in 0 .. n_cols_r {
			yhat += design_r[t][ci] * beta_r[ci]
		}
		d := y_all[effect_idx][t] - yhat
		rss_r += d * d
	}

	// F statistic: ((RSS_R - RSS_U) / p) / (RSS_U / (T - k*p - 1))
	df1 := f64(p)
	df2 := f64(t_eff - k * p - 1)
	f_stat := if rss_u > 1e-14 { ((rss_r - rss_u) / df1) / (rss_u / df2) } else { 0.0 }

	// p-value approximation: large-sample F → chi²(df1) / df1
	chi2_val := f_stat * df1
	p_value := 1.0 - chi2_cdf_approx(chi2_val, int(df1))

	return GrangerResult{
		cause_var:   cause_idx
		effect_var:  effect_idx
		f_statistic: f_stat
		p_value:     p_value
		significant: p_value < 0.05
	}
}

// var_forecast produces h-step ahead recursive forecasts for all k variables.
// Returns [k][steps] forecast matrix.
pub fn var_forecast(model VARModel, data [][]f64, h int) [][]f64 {
	k := model.k
	p := model.p
	t_total := data[0].len
	// Seed with last p observations from data
	mut fc_hist := [][]f64{len: k, init: []f64{len: p}}
	for j in 0 .. k {
		for lag in 0 .. p {
			fc_hist[j][lag] = data[j][t_total - p + lag]
		}
	}
	mut result := [][]f64{len: k, init: []f64{len: h}}
	for step in 0 .. h {
		// Build regressor vector: [1, y_{t-1}[0], y_{t-1}[1], ..., y_{t-p}[k-1]]
		mut row := []f64{len: k * p + 1}
		row[0] = 1.0
		hist_len := fc_hist[0].len
		for lag in 1 .. p + 1 {
			t_idx := hist_len - lag
			for j in 0 .. k {
				row[lag * k - k + j + 1] = fc_hist[j][t_idx]
			}
		}
		mut new_vals := []f64{len: k}
		for eq in 0 .. k {
			mut yhat := 0.0
			for j in 0 .. row.len {
				yhat += model.coeff_matrices[eq][j] * row[j]
			}
			new_vals[eq] = yhat
			result[eq][step] = yhat
		}
		for j in 0 .. k {
			fc_hist[j] << new_vals[j]
		}
	}
	return result
}

pub struct IRFResult {
pub:
	responses [][][]f64  // [shock_var][response_var][period]
	periods   int
}

// cholesky_lower computes the lower Cholesky factor L such that A = L L'.
// A must be symmetric positive definite.
fn cholesky_lower(a [][]f64) [][]f64 {
	n := a.len
	mut l := [][]f64{len: n, init: []f64{len: n}}
	for i in 0 .. n {
		for j in 0 .. i + 1 {
			mut s := a[i][j]
			for kk in 0 .. j {
				s -= l[i][kk] * l[j][kk]
			}
			if i == j {
				l[i][j] = if s > 0.0 { math.sqrt(s) } else { math.sqrt(math.abs(s)) + 1e-10 }
			} else {
				l[i][j] = if l[j][j] > 1e-14 { s / l[j][j] } else { 0.0 }
			}
		}
	}
	return l
}

// irf computes orthogonalized Impulse Response Functions for `steps` periods.
// Uses Cholesky decomposition of sigma_u for orthogonalization.
pub fn irf(model VARModel, steps int) IRFResult {
	k := model.k
	p := model.p

	// Cholesky factor of sigma_u for orthogonalisation
	chol := cholesky_lower(model.sigma_u)

	// Compute MA(inf) coefficients Φ_s via recursion: Φ_0 = I, Φ_s = Σ_{j=1}^{p} A_j Φ_{s-j}
	mut phi := [][][]f64{}
	// Φ_0 = I
	mut phi0 := [][]f64{len: k, init: []f64{len: k}}
	for i in 0 .. k {
		phi0[i][i] = 1.0
	}
	phi << phi0

	for s in 1 .. steps {
		mut phi_s := [][]f64{len: k, init: []f64{len: k}}
		for j in 0 .. p {
			if s - j - 1 < 0 {
				continue
			}
			// A_j: k×k matrix of lag-j coefficients from coeff_matrices
			// coeff_matrices[eq] = [intercept, y1_lag1, y2_lag1, ..., y1_lagp, y2_lagp, ...]
			// Lag j (1-indexed) coefficients start at column j*k - k + 1
			mut a_j := [][]f64{len: k, init: []f64{len: k}}
			for eq in 0 .. k {
				for v in 0 .. k {
					col := j * k + v + 1  // +1 for intercept
					a_j[eq][v] = model.coeff_matrices[eq][col]
				}
			}
			prev := phi[s - j - 1]
			contrib := linalg.matmul(a_j, prev)
			for r in 0 .. k {
				for c in 0 .. k {
					phi_s[r][c] += contrib[r][c]
				}
			}
		}
		phi << phi_s
	}

	// Orthogonalized IRF: Ψ_s = Φ_s · chol
	// responses[shock][response][period]
	mut responses := [][][]f64{len: k, init: [][]f64{len: k, init: []f64{len: steps}}}
	for s in 0 .. steps {
		psi_s := linalg.matmul(phi[s], chol)
		for shock in 0 .. k {
			for resp in 0 .. k {
				responses[shock][resp][s] = psi_s[resp][shock]
			}
		}
	}

	return IRFResult{
		responses: responses
		periods:   steps
	}
}
