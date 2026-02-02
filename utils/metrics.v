module utils

import math
import arrays

// ============================================================================
// Core Metrics
// ============================================================================

// Mean Squared Error
pub fn mse[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		error := f64(y_true[i]) - f64(y_pred[i])
		sum += error * error
	}
	return sum / f64(y_true.len)
}

// Root Mean Squared Error
pub fn rmse[T](y_true []T, y_pred []T) f64 {
	return math.sqrt(mse(y_true, y_pred))
}

// Mean Absolute Error
pub fn mae[T](y_true []T, y_pred []T) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut sum := 0.0
	for i in 0 .. y_true.len {
		diff := f64(y_true[i]) - f64(y_pred[i])
		sum += math.abs(diff)
	}
	return sum / f64(y_true.len)
}

// ============================================================================
// Classification Metrics
// ============================================================================

// ConfusionMatrix: structure to hold confusion matrix components
pub struct ConfusionMatrix {
	pub mut:
	true_positives  int
	true_negatives  int
	false_positives int
	false_negatives int
}

// Build confusion matrix from predictions and ground truth
pub fn build_confusion_matrix(y_true []int, y_pred []int) ConfusionMatrix {
	assert y_true.len == y_pred.len, "y_true and y_pred must have same length"
	
	mut tp := 0
	mut tn := 0
	mut fp := 0
	mut fn_ := 0
	
	for i in 0..y_true.len {
		if y_true[i] == 1 {
			if y_pred[i] == 1 {
				tp++
			} else {
				fn_++
			}
		} else {
			if y_pred[i] == 1 {
				fp++
			} else {
				tn++
			}
		}
	}
	
	return ConfusionMatrix{
		true_positives: tp
		true_negatives: tn
		false_positives: fp
		false_negatives: fn_
	}
}

// Accuracy: (TP + TN) / (TP + TN + FP + FN)
pub fn (cm ConfusionMatrix) accuracy() f64 {
	total := cm.true_positives + cm.true_negatives + cm.false_positives + cm.false_negatives
	if total == 0 {
		return 0.0
	}
	return f64(cm.true_positives + cm.true_negatives) / f64(total)
}

// Precision: TP / (TP + FP)
pub fn (cm ConfusionMatrix) precision() f64 {
	denom := cm.true_positives + cm.false_positives
	if denom == 0 {
		return 0.0
	}
	return f64(cm.true_positives) / f64(denom)
}

// Recall (Sensitivity): TP / (TP + FN)
pub fn (cm ConfusionMatrix) recall() f64 {
	denom := cm.true_positives + cm.false_negatives
	if denom == 0 {
		return 0.0
	}
	return f64(cm.true_positives) / f64(denom)
}

// Specificity: TN / (TN + FP)
pub fn (cm ConfusionMatrix) specificity() f64 {
	denom := cm.true_negatives + cm.false_positives
	if denom == 0 {
		return 0.0
	}
	return f64(cm.true_negatives) / f64(denom)
}

// F1 Score: 2 * (precision * recall) / (precision + recall)
pub fn (cm ConfusionMatrix) f1_score() f64 {
	prec := cm.precision()
	rec := cm.recall()
	denom := prec + rec
	if denom == 0 {
		return 0.0
	}
	return 2.0 * (prec * rec) / denom
}

// FPR (False Positive Rate): FP / (FP + TN)
pub fn (cm ConfusionMatrix) false_positive_rate() f64 {
	denom := cm.false_positives + cm.true_negatives
	if denom == 0 {
		return 0.0
	}
	return f64(cm.false_positives) / f64(denom)
}

// Summary returns formatted string with all metrics
pub fn (cm ConfusionMatrix) summary() string {
	mut result := "Confusion Matrix Summary\n"
	result += "========================\n"
	result += "TP: ${cm.true_positives}, TN: ${cm.true_negatives}\n"
	result += "FP: ${cm.false_positives}, FN: ${cm.false_negatives}\n\n"
	result += "Accuracy:  ${cm.accuracy():.4f}\n"
	result += "Precision: ${cm.precision():.4f}\n"
	result += "Recall:    ${cm.recall():.4f}\n"
	result += "Specificity: ${cm.specificity():.4f}\n"
	result += "F1 Score:  ${cm.f1_score():.4f}\n"
	return result
}

// ============================================================================
// ROC-AUC Metrics
// ============================================================================

// ROC_Curve: structure for ROC curve data
pub struct ROC_Curve {
	pub mut:
	thresholds []f64
	tpr        []f64 // True Positive Rate
	fpr        []f64 // False Positive Rate
	auc        f64   // Area Under Curve
}

// Calculate ROC curve from probability predictions
pub fn roc_curve(y_true []int, y_proba []f64) ROC_Curve {
	assert y_true.len == y_proba.len, "y_true and y_proba must have same length"
	
	// Create sorted indices by probability (descending)
	mut indices := []int{len: y_proba.len}
	for i in 0..indices.len {
		indices[i] = i
	}
	
	// Manual sort by probability (descending)
	for i in 0..indices.len {
		for j in i+1..indices.len {
			if y_proba[indices[j]] > y_proba[indices[i]] {
				// Swap
				temp := indices[i]
				indices[i] = indices[j]
				indices[j] = temp
			}
		}
	}
	
	// Count positives and negatives
	mut n_pos := 0
	mut n_neg := 0
	for label in y_true {
		if label == 1 {
			n_pos++
		} else {
			n_neg++
		}
	}
	
	if n_pos == 0 || n_neg == 0 {
		return ROC_Curve{
			thresholds: [0.0]
			tpr: [0.0]
			fpr: [0.0]
			auc: 0.0
		}
	}
	
	// Calculate TPR and FPR at different thresholds
	mut thresholds := []f64{}
	mut tpr := []f64{}
	mut fpr := []f64{}
	
	mut tp := 0
	mut fp := 0
	
	thresholds << 2.0 // Start above max probability
	tpr << 0.0
	fpr << 0.0
	
	mut prev_prob := -1.0
	for idx in indices {
		curr_prob := y_proba[idx]
		
		// Add point when threshold changes
		if curr_prob != prev_prob && prev_prob >= 0 {
			thresholds << curr_prob
			tpr << f64(tp) / f64(n_pos)
			fpr << f64(fp) / f64(n_neg)
		}
		
		if y_true[idx] == 1 {
			tp++
		} else {
			fp++
		}
		prev_prob = curr_prob
	}
	
	// Add final point
	thresholds << -1.0
	tpr << f64(tp) / f64(n_pos)
	fpr << f64(fp) / f64(n_neg)
	
	// Calculate AUC using trapezoidal rule
	mut auc := 0.0
	for i in 1..fpr.len {
		auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
	}
	auc = math.abs(auc)
	
	return ROC_Curve{
		thresholds: thresholds
		tpr: tpr
		fpr: fpr
		auc: auc
	}
}

// Area Under Curve for ROC
pub fn (roc ROC_Curve) auc_value() f64 {
	return roc.auc
}

// ============================================================================
// Hyperparameter Utilities
// ============================================================================

// GridSearchResult: result from grid search
pub struct GridSearchResult {
	pub mut:
	best_params map[string]f64
	best_score  f64
	scores      []f64
	param_grids []map[string]f64
}

// Generate parameter grid combinations (for simple numeric parameters)
pub fn generate_param_grid(param_ranges map[string][]f64) []map[string]f64 {
	if param_ranges.len == 0 {
		return []map[string]f64{}
	}
	
	// Get parameter names and ranges
	mut param_names := []string{}
	mut ranges := [][]f64{}
	
	for name, values in param_ranges {
		param_names << name
		ranges << values
	}
	
	// Generate all combinations (brute force for small grids)
	mut combinations := []map[string]f64{}
	
	// Recursive helper would be better, but using nested loops for simplicity
	if param_names.len == 1 {
		for val in ranges[0] {
			mut combo := map[string]f64{}
			combo[param_names[0]] = val
			combinations << combo
		}
	} else if param_names.len == 2 {
		for val1 in ranges[0] {
			for val2 in ranges[1] {
				mut combo := map[string]f64{}
				combo[param_names[0]] = val1
				combo[param_names[1]] = val2
				combinations << combo
			}
		}
	} else if param_names.len == 3 {
		for val1 in ranges[0] {
			for val2 in ranges[1] {
				for val3 in ranges[2] {
					mut combo := map[string]f64{}
					combo[param_names[0]] = val1
					combo[param_names[1]] = val2
					combo[param_names[2]] = val3
					combinations << combo
				}
			}
		}
	}
	
	return combinations
}

// ============================================================================
// Prediction Analysis Utilities
// ============================================================================

// Prediction metrics: returns common evaluation metrics for regression/classification
pub struct PredictionMetrics {
	pub mut:
	metric_name string
	value       f64
}

// Calculate all binary classification metrics in one pass
pub fn binary_classification_metrics(y_true []int, y_pred []int) map[string]f64 {
	cm := build_confusion_matrix(y_true, y_pred)
	
	return {
		'accuracy': cm.accuracy()
		'precision': cm.precision()
		'recall': cm.recall()
		'specificity': cm.specificity()
		'f1_score': cm.f1_score()
		'fpr': cm.false_positive_rate()
	}
}

// Calculate all regression metrics in one pass
pub fn regression_metrics(y_true []f64, y_pred []f64) map[string]f64 {
	assert y_true.len == y_pred.len, "y_true and y_pred must have same length"
	
	mut mse_val := 0.0
	mut mae_val := 0.0
	mut ss_res := 0.0
	mut ss_tot := 0.0
	
	y_mean := arrays.sum(y_true) or { 0.0 } / f64(y_true.len)
	
	for i in 0..y_true.len {
		error := y_true[i] - y_pred[i]
		mse_val += error * error
		mae_val += math.abs(error)
		ss_res += error * error
		ss_tot += (y_true[i] - y_mean) * (y_true[i] - y_mean)
	}
	
	mse_val = mse_val / f64(y_true.len)
	mae_val = mae_val / f64(y_true.len)
	rmse_val := math.sqrt(mse_val)
	r2_val := if ss_tot > 0 { 1.0 - (ss_res / ss_tot) } else { 0.0 }
	
	return {
		'mse': mse_val
		'rmse': rmse_val
		'mae': mae_val
		'r2': r2_val
	}
}

// ============================================================================
// Training Progress Utilities
// ============================================================================

// TrainingProgress: tracks training metrics
pub struct TrainingProgress {
	pub mut:
	epoch          int
	loss           f64
	val_loss       f64
	training_time  f64
	metrics        map[string]f64
}

// Format progress for logging
pub fn (tp TrainingProgress) format_log() string {
	mut log := "Epoch ${tp.epoch}: loss=${tp.loss:.6f}"
	if tp.val_loss > 0 {
		log += ", val_loss=${tp.val_loss:.6f}"
	}
	if tp.training_time > 0 {
		log += ", time=${tp.training_time:.2f}s"
	}
	for name, value in tp.metrics {
		log += ", ${name}=${value:.4f}"
	}
	return log
}

// Early stopping check: returns true if should stop
// Returns true if loss hasn't improved for 'patience' consecutive epochs
pub fn early_stopping(losses []f64, patience int) bool {
	if losses.len <= patience {
		return false
	}
	
	// Find the best loss in the entire history
	best_loss := arrays.min(losses) or { 0.0 }
	mut best_loss_idx := losses.len - 1
	for i in 0..losses.len {
		if losses[i] == best_loss {
			best_loss_idx = i
		}
	}
	
	// Check if latest loss is within patience epochs of best loss
	epochs_since_best := losses.len - 1 - best_loss_idx
	
	return epochs_since_best > patience
}

// Learning rate scheduler: exponential decay
pub fn decay_learning_rate(initial_lr f64, epoch int, decay_rate f64) f64 {
	return initial_lr * math.pow(decay_rate, f64(epoch))
}
