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
