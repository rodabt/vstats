module ml

import math
import stats
import utils

pub struct LinearModel[T] {
	coefficients []T
	intercept    T
}

pub struct LogisticModel[T] {
	coefficients []T
	intercept    T
}

// Linear Regression - Ordinary Least Squares (converts to f64 internally for numerical stability)
pub fn linear_regression[T](x [][]T, y []T) LinearModel[T] {
	assert x.len == y.len, "number of samples must match"
	assert x.len > 0, "must have at least one sample"
	
	// Convert input to f64 for numerical computation
	mut x_f64 := [][]f64{len: x.len}
	mut y_f64 := []f64{len: y.len}
	
	for i in 0 .. x.len {
		x_f64[i] = []f64{len: x[i].len}
		for j in 0 .. x[i].len {
			x_f64[i][j] = f64(x[i][j])
		}
		y_f64[i] = f64(y[i])
	}
	
	p := x_f64[0].len
	
	// Add intercept term (column of 1s)
	mut x_mat := [][]f64{len: x_f64.len}
	for i in 0 .. x_f64.len {
		x_mat[i] = []f64{len: p + 1}
		x_mat[i][0] = 1.0
		for j in 0 .. p {
			x_mat[i][j + 1] = x_f64[i][j]
		}
	}
	
	// Normal equations: β = (X^T X)^(-1) X^T y
	xt := matrix_transpose_f64(x_mat)
	xtx := matrix_multiply_f64(xt, x_mat)
	xty := matrix_vector_multiply_f64(xt, y_f64)
	
	// Solve using Gaussian elimination
	beta := gaussian_elimination_f64(xtx, xty)
	
	// Convert results back to T
	mut coeffs := []T{len: beta.len - 1}
	for i in 0 .. coeffs.len {
		coeffs[i] = T(beta[i + 1])
	}
	
	return LinearModel[T]{
		coefficients: coeffs
		intercept: T(beta[0])
	}
}

// Predict using linear model
pub fn linear_predict[T](model LinearModel[T], x [][]T) []T {
	mut predictions := []T{len: x.len}
	for i in 0 .. x.len {
		mut pred := model.intercept
		for j in 0 .. model.coefficients.len {
			if j < x[i].len {
				pred += model.coefficients[j] * x[i][j]
			}
		}
		predictions[i] = pred
	}
	return predictions
}

// Logistic Regression (binary classification) with gradient descent and L2 regularization
// Features should be normalized for best results
pub fn logistic_regression[T](x [][]T, y []T, iterations int, learning_rate T) LogisticModel[T] {
	assert x.len == y.len, "number of samples must match"
	assert x.len > 0, "must have at least one sample"
	
	n := T(x.len)
	p := x[0].len
	
	// Initialize coefficients
	mut coefficients := []T{len: p}
	for j in 0 .. p {
		coefficients[j] = T(0)
	}
	mut intercept := T(0)
	
	// Regularization parameter (prevents overfitting by penalizing large coefficients)
	lambda := T(1) / n  // L2 regularization strength
	
	// Gradient descent
	for iter in 0 .. iterations {
		// Adaptive learning rate: decay schedule
		decay_factor := T(1) + T(iter) * T(0.0005)
		current_lr := learning_rate / decay_factor
		
		mut batch_intercept_grad := T(0)
		mut batch_coeff_grad := []T{len: p}
		for _ in 0 .. p {
			batch_coeff_grad << T(0)
		}
		
		// Forward pass and gradient accumulation
		for i in 0 .. x.len {
			// Compute logit z = intercept + sum(coeff[j] * x[i][j])
			mut z := intercept
			for j in 0 .. p {
				z += coefficients[j] * x[i][j]
			}
			
			// Clamp z to prevent numerical overflow in sigmoid
			if z > T(100) {
				z = T(100)
			} else if z < T(-100) {
				z = T(-100)
			}
			
			// Prediction via sigmoid
			pred := utils.sigmoid(z)
			
			// Gradient of loss: (pred - y[i])
			grad := pred - y[i]
			
			// Accumulate gradients
			batch_intercept_grad += grad
			for j in 0 .. p {
				batch_coeff_grad[j] += grad * x[i][j]
			}
		}
		
		// Update parameters with regularization
		// Intercept update (no regularization on intercept)
		avg_intercept_grad := batch_intercept_grad / n
		intercept -= current_lr * avg_intercept_grad
		
		// Coefficient updates with L2 regularization
		for j in 0 .. p {
			avg_coeff_grad := batch_coeff_grad[j] / n
			// Add L2 regularization term: lambda * coeff[j]
			reg_term := lambda * coefficients[j]
			coefficients[j] -= current_lr * (avg_coeff_grad + reg_term)
		}
	}
	
	return LogisticModel[T]{
		coefficients: coefficients
		intercept: intercept
	}
}

// Predict probabilities using logistic model
pub fn logistic_predict_proba[T](model LogisticModel[T], x [][]T) []T {
	mut predictions := []T{len: x.len}
	for i in 0 .. x.len {
		mut z := model.intercept
		for j in 0 .. model.coefficients.len {
			if j < x[i].len {
				z += model.coefficients[j] * x[i][j]
			}
		}
		predictions[i] = utils.sigmoid(z)
	}
	return predictions
}

// Predict class using logistic model
pub fn logistic_predict[T](model LogisticModel[T], x [][]T, threshold T) []T {
	proba := logistic_predict_proba(model, x)
	mut predictions := []T{len: proba.len}
	for i in 0 .. proba.len {
		predictions[i] = if proba[i] >= threshold { T(1) } else { T(0) }
	}
	return predictions
}

// Wrapper functions that delegate to utils
pub fn sigmoid[T](x T) T {
	return utils.sigmoid(x)
}

pub fn mse[T](y_true []T, y_pred []T) f64 {
	return utils.mse(y_true, y_pred)
}

pub fn rmse[T](y_true []T, y_pred []T) f64 {
	return utils.rmse(y_true, y_pred)
}

pub fn mae[T](y_true []T, y_pred []T) f64 {
	return utils.mae(y_true, y_pred)
}

// R-squared coefficient of determination - Generic input, f64 output
pub fn r_squared[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	y_mean := stats.mean(y_true)
	
	mut ss_res := 0.0
	mut ss_tot := 0.0
	
	for i in 0 .. y_true.len {
		y_t := f64(y_true[i])
		y_p := f64(y_pred[i])
		diff_res := y_t - y_p
		diff_tot := y_t - y_mean
		ss_res += diff_res * diff_res
		ss_tot += diff_tot * diff_tot
	}
	
	return if ss_tot > 0 { 1.0 - ss_res / ss_tot } else { 0.0 }
}

// Helper: Matrix transpose (f64 version for numerical stability)
fn matrix_transpose_f64(m [][]f64) [][]f64 {
	if m.len == 0 {
		return [][]f64{}
	}
	
	rows := m[0].len
	cols := m.len
	
	mut result := [][]f64{len: rows}
	for i in 0 .. rows {
		result[i] = []f64{len: cols}
		for j in 0 .. cols {
			result[i][j] = m[j][i]
		}
	}
	return result
}

// Helper: Matrix multiply (f64 version)
fn matrix_multiply_f64(a [][]f64, b [][]f64) [][]f64 {
	assert a[0].len == b.len, "incompatible dimensions"
	
	rows := a.len
	cols := b[0].len
	
	mut result := [][]f64{len: rows}
	for i in 0 .. rows {
		result[i] = []f64{len: cols}
		for j in 0 .. cols {
			mut sum := 0.0
			for k in 0 .. a[0].len {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

// Helper: Matrix-vector multiply (f64 version)
fn matrix_vector_multiply_f64(m [][]f64, v []f64) []f64 {
	mut result := []f64{len: m.len}
	for i in 0 .. m.len {
		mut sum := 0.0
		for j in 0 .. v.len {
			sum += m[i][j] * v[j]
		}
		result[i] = sum
	}
	return result
}

// Helper: Gaussian elimination for solving Ax = b (f64 version)
fn gaussian_elimination_f64(a [][]f64, b []f64) []f64 {
	n := a.len
	mut matrix := [][]f64{len: n}
	mut rhs := []f64{len: n}
	
	// Copy input
	for i in 0 .. n {
		matrix[i] = a[i].clone()
		rhs[i] = b[i]
	}
	
	// Forward elimination
	for i in 0 .. n {
		// Find pivot
		mut max_row := i
		for k in (i + 1) .. n {
			if math.abs(matrix[k][i]) > math.abs(matrix[max_row][i]) {
				max_row = k
			}
		}
		
		// Swap rows
		matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
		rhs[i], rhs[max_row] = rhs[max_row], rhs[i]
		
		// Make all rows below this one 0 in current column
		for k in (i + 1) .. n {
			if matrix[i][i] != 0 {
				factor := matrix[k][i] / matrix[i][i]
				for j in i .. n {
					matrix[k][j] -= factor * matrix[i][j]
				}
				rhs[k] -= factor * rhs[i]
			}
		}
	}
	
	// Back substitution
	mut solution := []f64{len: n}
	for i := n - 1; i >= 0; i-- {
		solution[i] = rhs[i]
		for j in (i + 1) .. n {
			solution[i] -= matrix[i][j] * solution[j]
		}
		if matrix[i][i] != 0 {
			solution[i] /= matrix[i][i]
		}
	}
	
	return solution
}


