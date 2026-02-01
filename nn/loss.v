module nn

import math

// Mean Squared Error Loss
pub fn mse_loss[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		error := f64(y_true[i]) - f64(y_pred[i])
		sum += error * error
	}
	
	return sum / f64(y_true.len)
}

// MSE Loss Gradient
pub fn mse_loss_gradient[T](y_true []T, y_pred []T) []f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut grad := []f64{len: y_true.len}
	n := f64(y_true.len)
	
	for i in 0 .. y_true.len {
		grad[i] = -2.0 * (f64(y_true[i]) - f64(y_pred[i])) / n
	}
	
	return grad
}

// Binary Crossentropy Loss
pub fn binary_crossentropy_loss(y_true []f64, y_pred []f64) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		// Clip predictions to avoid log(0)
		y_clipped := math.max(1e-7, math.min(1 - 1e-7, y_pred[i]))
		sum += y_true[i] * math.log(y_clipped) + (1 - y_true[i]) * math.log(1 - y_clipped)
	}
	
	return -sum / f64(y_true.len)
}

// Binary Crossentropy Loss Gradient
pub fn binary_crossentropy_loss_gradient(y_true []f64, y_pred []f64) []f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut grad := []f64{len: y_true.len}
	n := f64(y_true.len)
	
	for i in 0 .. y_true.len {
		y_clipped := math.max(1e-7, math.min(1 - 1e-7, y_pred[i]))
		grad[i] = (y_clipped - y_true[i]) / (y_clipped * (1 - y_clipped) * n)
	}
	
	return grad
}

// Categorical Crossentropy Loss
pub fn categorical_crossentropy_loss(y_true [][]f64, y_pred [][]f64) f64 {
	assert y_true.len == y_pred.len, "batch sizes must match"
	
	mut sum := 0.0
	
	for i in 0 .. y_true.len {
		for j in 0 .. y_true[i].len {
			// Clip predictions to avoid log(0)
			y_clipped := math.max(1e-7, math.min(1 - 1e-7, y_pred[i][j]))
			if y_true[i][j] > 0 {
				sum += y_true[i][j] * math.log(y_clipped)
			}
		}
	}
	
	return -sum / f64(y_true.len)
}

// Sparse Categorical Crossentropy Loss
pub fn sparse_categorical_crossentropy_loss(y_true []int, y_pred [][]f64) f64 {
	assert y_true.len == y_pred.len, "batch sizes must match"
	
	mut sum := 0.0
	
	for i in 0 .. y_true.len {
		class_idx := y_true[i]
		y_clipped := math.max(1e-7, math.min(1 - 1e-7, y_pred[i][class_idx]))
		sum += math.log(y_clipped)
	}
	
	return -sum / f64(y_true.len)
}

// Huber Loss (robust to outliers)
pub fn huber_loss[T](y_true []T, y_pred []T, delta f64) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	
	for i in 0 .. y_true.len {
		error := math.abs(f64(y_true[i]) - f64(y_pred[i]))
		if error <= delta {
			sum += 0.5 * error * error
		} else {
			sum += delta * (error - 0.5 * delta)
		}
	}
	
	return sum / f64(y_true.len)
}

// Mean Absolute Error Loss
pub fn mae_loss[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		sum += math.abs(f64(y_true[i]) - f64(y_pred[i]))
	}
	
	return sum / f64(y_true.len)
}

// MAE Loss Gradient
pub fn mae_loss_gradient[T](y_true []T, y_pred []T) []f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut grad := []f64{len: y_true.len}
	n := f64(y_true.len)
	
	for i in 0 .. y_true.len {
		diff := f64(y_pred[i]) - f64(y_true[i])
		grad[i] = if diff > 0 { 1.0 / n } else { -1.0 / n }
	}
	
	return grad
}

// Hinge Loss (for SVM-like classifiers)
pub fn hinge_loss[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		// Assume y_true in {-1, 1}
		loss := math.max(0.0, 1.0 - f64(y_true[i]) * f64(y_pred[i]))
		sum += loss
	}
	
	return sum / f64(y_true.len)
}

// Squared Hinge Loss
pub fn squared_hinge_loss[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		loss := math.max(0.0, 1.0 - f64(y_true[i]) * f64(y_pred[i]))
		sum += loss * loss
	}
	
	return sum / f64(y_true.len)
}

// Kullback-Leibler Divergence Loss
pub fn kl_divergence_loss(y_true []f64, y_pred []f64) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	
	for i in 0 .. y_true.len {
		if y_true[i] > 0 {
			y_clipped := math.max(1e-7, math.min(1 - 1e-7, y_pred[i]))
			sum += y_true[i] * math.log(y_true[i] / y_clipped)
		}
	}
	
	return sum
}

// Cosine Similarity Loss
pub fn cosine_similarity_loss[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut dot_product := 0.0
	mut norm_true := 0.0
	mut norm_pred := 0.0
	
	for i in 0 .. y_true.len {
		dot_product += f64(y_true[i]) * f64(y_pred[i])
		norm_true += f64(y_true[i]) * f64(y_true[i])
		norm_pred += f64(y_pred[i]) * f64(y_pred[i])
	}
	
	denominator := math.sqrt(norm_true) * math.sqrt(norm_pred)
	
	if denominator == 0 {
		return 0
	}
	
	cosine_sim := dot_product / denominator
	return 1 - cosine_sim
}

// Contrastive Loss (for siamese networks)
pub fn contrastive_loss(y_true f64, distance f64, margin f64) f64 {
	if y_true > 0.5 {
		// Same class
		return distance * distance
	} else {
		// Different class
		return math.max(0.0, margin - distance) * math.max(0.0, margin - distance)
	}
}

// Triplet Loss (for metric learning)
pub fn triplet_loss[T](anchor []T, positive []T, negative []T, margin f64) f64 {
	// Compute euclidean distances
	mut dist_ap := 0.0
	mut dist_an := 0.0
	
	assert anchor.len == positive.len && anchor.len == negative.len, "vector sizes must match"
	
	for i in 0 .. anchor.len {
		diff_p := f64(anchor[i]) - f64(positive[i])
		diff_n := f64(anchor[i]) - f64(negative[i])
		dist_ap += diff_p * diff_p
		dist_an += diff_n * diff_n
	}
	
	dist_ap = math.sqrt(dist_ap)
	dist_an = math.sqrt(dist_an)
	
	return math.max(0.0, dist_ap - dist_an + margin)
}
