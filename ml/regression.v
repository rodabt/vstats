module ml

import math
import stats

pub struct LinearModel[T] {
	coefficients []T
	intercept    T
}

pub struct LogisticModel[T] {
	coefficients []T
	intercept    T
}

// Linear Regression - Ordinary Least Squares
pub fn linear_regression[T](x [][]T, y []T) LinearModel[T] {
	assert x.len == y.len, "number of samples must match"
	assert x.len > 0, "must have at least one sample"
	
	// n := T(x.len)
	p := x[0].len
	
	// Add intercept term (column of 1s)
	mut x_mat := [][]T{len: x.len}
	for i in 0 .. x.len {
		x_mat[i] = []T{len: p + 1}
		x_mat[i][0] = 1.0
		for j in 0 .. p {
			x_mat[i][j + 1] = x[i][j]
		}
	}
	
	// Normal equations: β = (X^T X)^(-1) X^T y
	xt := matrix_transpose(x_mat)
	xtx := matrix_multiply(xt, x_mat)
	xty := matrix_vector_multiply(xt, y)
	
	// Solve using Gaussian elimination (simplified)
	beta := gaussian_elimination(xtx, xty)
	
	return LinearModel{
		coefficients: beta[1..]
		intercept: beta[0]
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
			pred := sigmoid(z)
			
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
		predictions[i] = sigmoid(z)
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

// Mean Squared Error
pub fn mse[T](y_true []T, y_pred []T) T {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		error := y_true[i] - y_pred[i]
		sum += error * error
	}
	return sum / T(y_true.len)
}

// Root Mean Squared Error
pub fn rmse[T](y_true []T, y_pred []T) T {
	return math.sqrt(mse(y_true, y_pred))
}

// Mean Absolute Error
pub fn mae[T](y_true []T, y_pred []T) T {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		sum += math.abs(y_true[i] - y_pred[i])
	}
	return sum / T(y_true.len)
}

// R-squared coefficient of determination
pub fn r_squared[T](y_true []T, y_pred []T) T {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	y_mean := stats.mean(y_true)
	
	mut ss_res := 0.0
	mut ss_tot := 0.0
	
	for i in 0 .. y_true.len {
		ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i])
		ss_tot += (y_true[i] - y_mean) * (y_true[i] - y_mean)
	}
	
	return if ss_tot > 0 { 1 - ss_res / ss_tot } else { 0 }
}

// Helper: Matrix transpose
fn matrix_transpose[T](m [][]T) [][]T {
	if m.len == 0 {
		return [][]T{}
	}
	
	rows := m[0].len
	cols := m.len
	
	mut result := [][]T{len: rows}
	for i in 0 .. rows {
		result[i] = []T{len: cols}
		for j in 0 .. cols {
			result[i][j] = m[j][i]
		}
	}
	return result
}

// Helper: Matrix multiply
fn matrix_multiply[T](a [][]T, b [][]T) [][]T {
	assert a[0].len == b.len, "incompatible dimensions"
	
	rows := a.len
	cols := b[0].len
	
	mut result := [][]T{len: rows}
	for i in 0 .. rows {
		result[i] = []T{len: cols}
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

// Helper: Matrix-vector multiply
fn matrix_vector_multiply[T](m [][]T, v []T) []T {
	mut result := []T{len: m.len}
	for i in 0 .. m.len {
		mut sum := 0.0
		for j in 0 .. v.len {
			sum += m[i][j] * v[j]
		}
		result[i] = sum
	}
	return result
}

// Helper: Gaussian elimination for solving Ax = b
fn gaussian_elimination[T](a [][]T, b []T) []T {
	n := a.len
	mut matrix := [][]T{len: n}
	mut rhs := []T{len: n}
	
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
	mut solution := []T{len: n}
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

// Helper: Sigmoid function
fn sigmoid[T](x T) T {
	return 1.0 / (1.0 + math.exp(-x))
}
