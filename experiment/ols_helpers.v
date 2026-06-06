module experiment

import math
import linalg

// matrix_inverse computes the inverse of a square matrix via Gaussian elimination.
fn matrix_inverse(m [][]f64) [][]f64 {
	n_dim := m.len
	mut result := [][]f64{len: n_dim}
	for i in 0 .. n_dim {
		result[i] = []f64{len: n_dim}
	}
	for j in 0 .. n_dim {
		mut e := []f64{len: n_dim, init: 0.0}
		e[j] = 1.0
		col := linalg.gaussian_elimination(m, e)
		for i in 0 .. n_dim {
			result[i][j] = col[i]
		}
	}
	return result
}

// ols_se computes OLS standard errors from residuals.
// x_mat is the design matrix WITHOUT intercept; coefficients is [intercept, coef_1, ..., coef_p].
fn ols_se(x_mat [][]f64, y []f64, coefficients []f64) []f64 {
	n := x_mat.len
	n_params := coefficients.len

	if n <= n_params {
		return []f64{len: n_params, init: 0.0}
	}

	// Build augmented X with intercept column prepended
	mut x_aug := [][]f64{len: n}
	for i in 0 .. n {
		x_aug[i] = []f64{len: n_params}
		x_aug[i][0] = 1.0
		for j in 1 .. n_params {
			if j - 1 < x_mat[i].len {
				x_aug[i][j] = x_mat[i][j - 1]
			}
		}
	}

	// Compute sum of squared residuals
	mut ssr := 0.0
	for i in 0 .. n {
		mut y_pred := 0.0
		for j in 0 .. n_params {
			y_pred += coefficients[j] * x_aug[i][j]
		}
		resid := y[i] - y_pred
		ssr += resid * resid
	}
	s2 := ssr / f64(n - n_params)

	// Compute (X'X)^{-1}
	xt := linalg.transpose(x_aug)
	xtx := linalg.matmul(xt, x_aug)
	xtx_inv := matrix_inverse(xtx)

	// SE_j = sqrt(s^2 * [(X'X)^{-1}]_{jj})
	mut se := []f64{len: n_params}
	for j in 0 .. n_params {
		v_j := s2 * xtx_inv[j][j]
		se[j] = if v_j > 0 { math.sqrt(v_j) } else { 0.0 }
	}
	return se
}
