import ml
import utils

// Test Titanic dataset loading
fn test__titanic_dataset_basic() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	assert dataset.features.len > 0, "Titanic dataset should have samples"
	assert dataset.target.len == dataset.features.len, "features and target should match"
	assert dataset.features[0].len == 5, "should have 5 features"
	
	// Real Titanic: 891 samples, ~38% survival rate
	assert dataset.features.len >= 880, "should have at least 880 valid samples"
	
	mut survived := 0
	for label in dataset.target {
		if label == 1 {
			survived++
		}
	}
	survival_rate := f64(survived) / f64(dataset.target.len)
	assert survival_rate > 0.3 && survival_rate < 0.45, "Titanic survival rate should be ~38%"
	
	println("Dataset loaded: ${dataset.features.len} samples, ${survival_rate*100:.1f}% survived")
}

// Test logistic regression on Titanic with expected community benchmarks
// Literature benchmark: Logistic Regression achieves ~70-78% accuracy on Titanic
fn test__logistic_regression_on_titanic() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Convert target to f64
	mut y_f64 := []f64{}
	for label in dataset.target {
		y_f64 << f64(label)
	}
	
	// Standard 80/20 split
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := y_f64[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test_int := dataset.target[split_idx..dataset.target.len]
	
	// Train model with 1000 iterations and 0.01 learning rate
	model := ml.logistic_classifier(x_train, y_train, 1000, 0.01)
	assert model.trained == true, "model should be trained"
	assert model.coefficients.len == 5, "should have 5 coefficients (features)"
	
	// Make predictions with threshold 0.5
	predictions := ml.logistic_classifier_predict(model, x_test, 0.5)
	assert predictions.len == x_test.len, "predictions should match test set size"
	
	// Validate predictions are binary
	for pred in predictions {
		assert pred == 0 || pred == 1, "predictions should be 0 or 1"
	}
	
	// Calculate metrics
	acc := ml.accuracy(y_test_int, predictions)
	prec := ml.precision(y_test_int, predictions, 1)
	rec := ml.recall(y_test_int, predictions, 1)
	f1 := ml.f1_score(y_test_int, predictions, 1)
	
	// Standard benchmark expectation for Logistic Regression on Titanic
	// Community benchmarks show 70-78% accuracy
	println("\n=== Logistic Regression on Titanic ===")
	println("Test set size: ${x_test.len}")
	println("Accuracy:  ${acc:.4f} (expected: 0.70-0.78)")
	println("Precision: ${prec:.4f}")
	println("Recall:    ${rec:.4f}")
	println("F1 Score:  ${f1:.4f}")
	
	// Assert against community benchmarks
	assert acc >= 0.70, "Logistic Regression should achieve ~70%+ accuracy (community benchmark)"
	assert prec >= 0.0, "Precision should be valid"
	assert rec >= 0.0, "Recall should be valid"
	assert f1 >= 0.0, "F1 should be valid"
}

// Test logistic regression probability predictions
fn test__logistic_regression_probabilities() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	mut y_f64 := []f64{}
	for label in dataset.target {
		y_f64 << f64(label)
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := y_f64[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	
	model := ml.logistic_classifier(x_train, y_train, 1000, 0.01)
	proba := ml.logistic_classifier_predict_proba(model, x_test)
	
	assert proba.len == x_test.len, "probability output size should match test set"
	
	for prob in proba {
		assert prob >= 0.0 && prob <= 1.0, "probabilities should be in [0, 1]"
	}
	
	mut min_prob := proba[0]
	mut max_prob := proba[0]
	for prob in proba {
		if prob < min_prob {
			min_prob = prob
		}
		if prob > max_prob {
			max_prob = prob
		}
	}
	
	println("Probability predictions validated: min=${min_prob:.4f}, max=${max_prob:.4f}")
}

// Test confusion matrix from logistic regression predictions
fn test__logistic_regression_confusion_matrix() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	mut y_f64 := []f64{}
	for label in dataset.target {
		y_f64 << f64(label)
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := y_f64[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	model := ml.logistic_classifier(x_train, y_train, 1000, 0.01)
	predictions := ml.logistic_classifier_predict(model, x_test, 0.5)
	
	cm := ml.confusion_matrix(y_test, predictions)
	assert cm.len == 2, "confusion matrix should be 2x2"
	assert cm[0].len == 2, "confusion matrix should be 2x2"
	
	tn := cm[0][0]  // True Negatives
	fp := cm[0][1]  // False Positives
	fn_val := cm[1][0]  // False Negatives
	tp := cm[1][1]  // True Positives
	
	println("\nConfusion Matrix:")
	println("                 Predicted")
	println("              Negative  Positive")
	println("Actual Negative  ${tn:3}      ${fp:3}")
	println("       Positive  ${fn_val:3}      ${tp:3}")
	
	assert (tn + fp + fn_val + tp) == y_test.len, "confusion matrix counts should match test set size"
}

// Test that different thresholds affect predictions
fn test__logistic_regression_threshold_sensitivity() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	mut y_f64 := []f64{}
	for label in dataset.target {
		y_f64 << f64(label)
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := y_f64[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	
	model := ml.logistic_classifier(x_train, y_train, 1000, 0.01)
	
	// Test different thresholds
	pred_03 := ml.logistic_classifier_predict(model, x_test, 0.3)
	pred_05 := ml.logistic_classifier_predict(model, x_test, 0.5)
	pred_07 := ml.logistic_classifier_predict(model, x_test, 0.7)
	
	count_1_03 := pred_03.filter(it == 1).len
	count_1_05 := pred_05.filter(it == 1).len
	count_1_07 := pred_07.filter(it == 1).len
	
	// Lower threshold should predict more positives
	assert count_1_03 >= count_1_05, "lower threshold should produce more positive predictions"
	assert count_1_05 >= count_1_07, "lower threshold should produce more positive predictions"
	
	println("\nThreshold sensitivity:")
	println("Threshold 0.3: ${count_1_03} positive predictions")
	println("Threshold 0.5: ${count_1_05} positive predictions")
	println("Threshold 0.7: ${count_1_07} positive predictions")
}

// Test model coefficient analysis
fn test__logistic_regression_coefficients() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	mut y_f64 := []f64{}
	for label in dataset.target {
		y_f64 << f64(label)
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := y_f64[0..split_idx]
	
	model := ml.logistic_classifier(x_train, y_train, 1000, 0.01)
	
	// Check that coefficients are learned (not all zero)
	mut any_nonzero := false
	for coef in model.coefficients {
		if coef != 0.0 {
			any_nonzero = true
			break
		}
	}
	
	assert any_nonzero, "model should learn non-zero coefficients"
	
	// Feature names: Pclass, Age, SibSp, Parch, Fare
	feature_names := ["Pclass", "Age", "SibSp", "Parch", "Fare"]
	println("\nLogistic Regression Coefficients:")
	println("Intercept: ${model.intercept:.6f}")
	for i, name in feature_names {
		if i < model.coefficients.len {
			println("${name:10}: ${model.coefficients[i]:10.6f}")
		}
	}
}
