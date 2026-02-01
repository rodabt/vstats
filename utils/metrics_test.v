module utils

import math

fn test_confusion_matrix() {
	y_true := [1, 1, 0, 0, 1, 0]
	y_pred := [1, 0, 0, 1, 1, 0]
	
	cm := build_confusion_matrix(y_true, y_pred)
	
	// TP: positions 0, 4 = 2
	// TN: positions 2, 5 = 2
	// FP: position 3 = 1
	// FN: position 1 = 1
	assert cm.true_positives == 2, "should have 2 true positives"
	assert cm.true_negatives == 2, "should have 2 true negatives"
	assert cm.false_positives == 1, "should have 1 false positive"
	assert cm.false_negatives == 1, "should have 1 false negative"
}

fn test_confusion_matrix_metrics() {
	y_true := [1, 1, 1, 0, 0, 0]
	y_pred := [1, 1, 0, 0, 0, 1]
	
	cm := build_confusion_matrix(y_true, y_pred)
	
	acc := cm.accuracy()
	prec := cm.precision()
	rec := cm.recall()
	
	// Accuracy: (2+2)/6 = 4/6 = 0.6667
	assert math.abs(acc - 4.0/6.0) < 0.01, "accuracy should be 4/6"
	
	// Precision: TP/(TP+FP) = 2/3 = 0.6667
	assert math.abs(prec - 2.0/3.0) < 0.01, "precision should be 2/3"
	
	// Recall: TP/(TP+FN) = 2/3 = 0.6667
	assert math.abs(rec - 2.0/3.0) < 0.01, "recall should be 2/3"
}

fn test_f1_score() {
	y_true := [1, 1, 1, 0, 0, 0]
	y_pred := [1, 1, 0, 0, 0, 1]
	
	cm := build_confusion_matrix(y_true, y_pred)
	f1 := cm.f1_score()
	
	// Precision and recall both 2/3, so F1 = 2*(2/3)*(2/3) / (4/3) = 2/3
	assert f1 > 0, "F1 score should be positive"
	assert f1 < 1.0, "F1 score should be less than 1"
}

fn test_roc_curve() {
	y_true := [1, 1, 0, 1, 0, 0]
	y_proba := [0.9, 0.8, 0.3, 0.7, 0.2, 0.1]
	
	roc := roc_curve(y_true, y_proba)
	
	// AUC should be between 0 and 1
	assert roc.auc >= 0 && roc.auc <= 1, "AUC should be between 0 and 1"
	
	// TPR and FPR should have same length
	assert roc.tpr.len == roc.fpr.len, "TPR and FPR should have same length"
	
	// Should have at least a few points
	assert roc.tpr.len > 1, "ROC curve should have multiple points"
}

fn test_roc_auc() {
	// Perfect classifier
	y_true := [1, 1, 0, 0]
	y_proba := [0.9, 0.8, 0.2, 0.1]
	
	roc := roc_curve(y_true, y_proba)
	
	// Perfect classifier should have AUC close to 1.0
	assert roc.auc > 0.9, "perfect classifier should have high AUC"
}

fn test_generate_param_grid() {
	param_ranges := {
		'learning_rate': [0.01, 0.1, 1.0]
		'batch_size': [16.0, 32.0]
	}
	
	grid := generate_param_grid(param_ranges)
	
	// Should have 3 * 2 = 6 combinations
	assert grid.len == 6, "should have 6 parameter combinations"
	
	// Each combination should have 2 parameters
	for combo in grid {
		assert combo.len == 2, "each combination should have 2 parameters"
	}
}

fn test_binary_classification_metrics() {
	y_true := [1, 1, 0, 0, 1, 0]
	y_pred := [1, 0, 0, 1, 1, 0]
	
	metrics := binary_classification_metrics(y_true, y_pred)
	
	// Check that all expected metrics are present
	assert 'accuracy' in metrics, "should have accuracy metric"
	assert 'precision' in metrics, "should have precision metric"
	assert 'recall' in metrics, "should have recall metric"
	assert 'f1_score' in metrics, "should have f1_score metric"
	
	// All values should be between 0 and 1
	for _, value in metrics {
		assert value >= 0 && value <= 1, "metric values should be between 0 and 1"
	}
}

fn test_regression_metrics() {
	y_true := [1.0, 2.0, 3.0, 4.0, 5.0]
	y_pred := [1.1, 2.1, 2.9, 4.1, 4.9]
	
	metrics := regression_metrics(y_true, y_pred)
	
	// Check expected metrics
	assert 'mse' in metrics, "should have mse"
	assert 'rmse' in metrics, "should have rmse"
	assert 'mae' in metrics, "should have mae"
	assert 'r2' in metrics, "should have r2"
	
	// MSE and RMSE should be positive
	assert metrics['mse'] > 0, "MSE should be positive"
	assert metrics['rmse'] > 0, "RMSE should be positive"
	
	// RMSE should equal sqrt(MSE)
	expected_rmse := math.sqrt(metrics['mse'])
	assert math.abs(metrics['rmse'] - expected_rmse) < 0.0001, "RMSE should be sqrt(MSE)"
}

fn test_training_progress() {
	tp := TrainingProgress{
		epoch: 1
		loss: 0.5
		val_loss: 0.6
		training_time: 1.5
		metrics: {'acc': 0.8}
	}
	
	log := tp.format_log()
	
	// Should contain epoch number
	assert log.contains('Epoch 1'), "log should contain epoch number"
	// Should contain loss values
	assert log.contains('loss='), "log should contain loss"
	assert log.contains('val_loss='), "log should contain val_loss"
}

fn test_early_stopping() {
	// Losses that improve then plateau
	losses := [0.5, 0.4, 0.35, 0.36, 0.37, 0.38, 0.39]
	
	// With patience=2, after 5 epochs without improvement, should return true
	should_stop := early_stopping(losses, 2)
	assert should_stop, "losses plateauing should trigger early stopping"
	
	// Very short list shouldn't trigger
	short_losses := [0.5, 0.4, 0.35]
	should_not_stop := early_stopping(short_losses, 2)
	assert !should_not_stop, "short loss list should not trigger stop"
}

fn test_decay_learning_rate() {
	initial_lr := 0.1
	decay_rate := 0.9
	
	lr_0 := decay_learning_rate(initial_lr, 0, decay_rate)
	lr_1 := decay_learning_rate(initial_lr, 1, decay_rate)
	lr_10 := decay_learning_rate(initial_lr, 10, decay_rate)
	
	// Learning rate should decrease over epochs
	assert math.abs(lr_0 - 0.1) < 0.001, "initial LR should be 0.1"
	assert lr_1 < lr_0, "LR should decrease after epoch 1"
	assert lr_10 < lr_1, "LR should keep decreasing"
}
