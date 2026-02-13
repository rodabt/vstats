import ml
import utils

// Test Titanic dataset loading and class distribution
fn test__titanic_naive_bayes_dataset_loading() {
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
	mut died := 0
	for label in dataset.target {
		if label == 1 {
			survived++
		} else {
			died++
		}
	}
	
	survival_rate := f64(survived) / f64(dataset.target.len)
	assert survival_rate > 0.3 && survival_rate < 0.45, "Titanic survival rate should be ~38%"
	
	println("Dataset loaded: ${dataset.features.len} samples")
	println("  Survived (1): ${survived} (${survival_rate*100:.1f}%)")
	println("  Died (0):     ${died} (${(1.0-survival_rate)*100:.1f}%)")
}

// Test Naive Bayes training on Titanic dataset
// Literature benchmark: Naive Bayes achieves ~70-77% accuracy on Titanic
fn test__naive_bayes_on_titanic() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Standard 80/20 split
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	// Train model
	model := ml.naive_bayes_classifier(x_train, y_train)
	assert model.trained == true, "model should be trained"
	assert model.classes.len == 2, "should have 2 classes"
	
	// Make predictions
	predictions := ml.naive_bayes_predict(model, x_test)
	assert predictions.len == x_test.len, "predictions should match test set size"
	
	// Validate predictions are binary
	for pred in predictions {
		assert pred == 0 || pred == 1, "predictions should be 0 or 1"
	}
	
	// Calculate metrics
	acc := ml.accuracy(y_test, predictions)
	prec := ml.precision(y_test, predictions, 1)
	rec := ml.recall(y_test, predictions, 1)
	f1 := ml.f1_score(y_test, predictions, 1)
	
	// Community benchmark: Naive Bayes on Titanic typically achieves 70-77% accuracy
	println("\n=== Naive Bayes on Titanic ===")
	println("Test set size: ${x_test.len}")
	println("Accuracy:  ${acc:.4f} (expected: 0.70-0.77)")
	println("Precision: ${prec:.4f}")
	println("Recall:    ${rec:.4f}")
	println("F1 Score:  ${f1:.4f}")
	
	// Assert against community benchmarks
	assert acc >= 0.70, "Naive Bayes should achieve ~70%+ accuracy (community benchmark)"
	assert prec >= 0.0, "Precision should be valid"
	assert rec >= 0.0, "Recall should be valid"
	assert f1 >= 0.0, "F1 should be valid"
}

// Test class priors are correctly calculated
fn test__naive_bayes_class_priors() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	
	model := ml.naive_bayes_classifier(x_train, y_train)
	
	// Check class priors exist and sum to ~1.0
	assert model.class_priors.len == 2, "should have priors for 2 classes"

	mut prior_sum := 0.0
	for c in 0 .. model.classes.len {
		class_label := model.classes[c]
		prior := model.class_priors[class_label]
		assert prior > 0.0 && prior < 1.0, "prior should be between 0 and 1"
		prior_sum += prior
	}

	assert prior_sum > 0.99 && prior_sum <= 1.01, "priors should sum to 1.0"

	println("\nClass Priors:")
	for c in 0 .. model.classes.len {
		class_label := model.classes[c]
		println("  Class ${class_label}: ${model.class_priors[class_label]:.4f}")
	}
	println("  Sum:     ${prior_sum:.4f}")
}

// Test feature statistics are learned for each class
fn test__naive_bayes_feature_statistics() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	
	model := ml.naive_bayes_classifier(x_train, y_train)

	// Check feature means are stored for both classes
	assert model.classes.len == 2, "should have 2 classes"
	assert model.feature_means.len == 2, "should have feature means for 2 classes"
	assert model.feature_stds.len == 2, "should have feature stds for 2 classes"

	for c in 0 .. model.classes.len {
		class_label := model.classes[c]
		means := model.feature_means[class_label]
		stds := model.feature_stds[class_label]

		// Should have statistics for all 5 features
		assert means.len == 5, "should have means for 5 features (class ${c})"
		assert stds.len == 5, "should have stds for 5 features (class ${c})"
	}
	
	println("\nFeature Statistics Learned:")
	for c in 0 .. model.classes.len {
		class_label := model.classes[c]
		println("  Class ${class_label}:")
		for feat_idx in 0 .. 5 {
			mean := model.feature_means[class_label][feat_idx][0]
			std := model.feature_stds[class_label][feat_idx][0]
			feature_names := ["Pclass", "Age", "SibSp", "Parch", "Fare"]
			println("    ${feature_names[feat_idx]:8}: μ=${mean:7.2f}, σ=${std:7.2f}")
		}
	}
}

// Test confusion matrix and metrics
fn test__naive_bayes_confusion_matrix() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	model := ml.naive_bayes_classifier(x_train, y_train)
	predictions := ml.naive_bayes_predict(model, x_test)
	
	cm := ml.confusion_matrix(y_test, predictions)
	assert cm.len == 2, "confusion matrix should be 2x2"
	assert cm[0].len == 2, "confusion matrix should be 2x2"
	
	tn := cm[0][0]  // True Negatives (correctly predicted died)
	fp := cm[0][1]  // False Positives (predicted survived, actually died)
	fn_val := cm[1][0]  // False Negatives (predicted died, actually survived)
	tp := cm[1][1]  // True Positives (correctly predicted survived)
	
	println("\nConfusion Matrix:")
	println("                 Predicted")
	println("              Negative  Positive")
	println("Actual Negative  ${tn:3}      ${fp:3}")
	println("       Positive  ${fn_val:3}      ${tp:3}")
	
	assert (tn + fp + fn_val + tp) == y_test.len, "confusion matrix counts should match test set size"
	
	// Calculate specificity and sensitivity from confusion matrix
	if (tn + fp) > 0 {
		specificity := f64(tn) / f64(tn + fp)
		println("  Specificity (TNR): ${specificity:.4f}")
	}
	if (tp + fn_val) > 0 {
		sensitivity := f64(tp) / f64(tp + fn_val)
		println("  Sensitivity (TPR): ${sensitivity:.4f}")
	}
}

// Test model prediction consistency
fn test__naive_bayes_prediction_consistency() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	
	// Train model
	model := ml.naive_bayes_classifier(x_train, y_train)
	
	// Make predictions twice
	pred1 := ml.naive_bayes_predict(model, x_test)
	pred2 := ml.naive_bayes_predict(model, x_test)
	
	// Predictions should be identical
	assert pred1.len == pred2.len, "predictions should have same length"
	for i in 0 .. pred1.len {
		assert pred1[i] == pred2[i], "predictions should be consistent (${i}: ${pred1[i]} != ${pred2[i]})"
	}
	
	println("Prediction consistency: PASS (deterministic)")
}

// Test on different data splits
fn test__naive_bayes_multiple_splits() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	mut accuracies := []f64{}
	
	// Test with different train/test splits
	for split_pct in [0.7, 0.8, 0.9] {
		split_idx := int(f64(dataset.features.len) * split_pct)
		x_train := dataset.features[0..split_idx]
		y_train := dataset.target[0..split_idx]
		x_test := dataset.features[split_idx..dataset.features.len]
		y_test := dataset.target[split_idx..dataset.target.len]
		
		model := ml.naive_bayes_classifier(x_train, y_train)
		predictions := ml.naive_bayes_predict(model, x_test)
		
		acc := ml.accuracy(y_test, predictions)
		accuracies << acc
		
		println("Split ${(split_pct*100):.0f}% train: accuracy = ${acc:.4f}")
	}
	
	// All splits should achieve reasonable accuracy
	for acc in accuracies {
		assert acc >= 0.65, "Naive Bayes should achieve >65% accuracy on Titanic"
	}
}

// Test on training data (check for reasonable fit)
fn test__naive_bayes_on_training_data() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	model := ml.naive_bayes_classifier(x_train, y_train)
	
	// Predict on training and test data
	train_pred := ml.naive_bayes_predict(model, x_train)
	test_pred := ml.naive_bayes_predict(model, x_test)
	
	train_acc := ml.accuracy(y_train, train_pred)
	test_acc := ml.accuracy(y_test, test_pred)
	
	println("Training accuracy: ${train_acc:.4f}")
	println("Test accuracy:     ${test_acc:.4f}")
	println("Gap (should be small due to variance smoothing): ${(train_acc - test_acc):.4f}")
	
	// Training and test should be similar (good generalization due to variance smoothing)
	// Don't expect large gap, but require reasonable overall accuracy
	assert train_acc > 0.60, "training accuracy should be > 60%"
	assert test_acc > 0.70, "test accuracy should be > 70%"
	gap := train_acc - test_acc
	assert gap < 0.15, "gap between train and test should be small (good generalization)"
}

// Test edge cases and robustness
fn test__naive_bayes_robustness() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Use small subset
	subset_size := 100
	x := dataset.features[0..subset_size]
	y := dataset.target[0..subset_size]
	
	// Should not crash on small dataset
	model := ml.naive_bayes_classifier(x, y)
	assert model.trained == true, "should train on small dataset"
	
	// Should make predictions without crashing
	predictions := ml.naive_bayes_predict(model, x)
	assert predictions.len == x.len, "should make predictions"
	
	println("Robustness test: PASS (small dataset handling)")
}

// Test class priors match data distribution
fn test__naive_bayes_prior_accuracy() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	y_train := dataset.target[0..split_idx]
	
	// Count actual distribution
	mut count_0 := 0
	mut count_1 := 0
	for label in y_train {
		if label == 0 {
			count_0++
		} else {
			count_1++
		}
	}
	
	expected_prior_0 := f64(count_0) / f64(y_train.len)
	expected_prior_1 := f64(count_1) / f64(y_train.len)
	
	// Train and check priors
	model := ml.naive_bayes_classifier(dataset.features[0..split_idx], y_train)

	// Find class indices
	mut idx_0 := -1
	mut idx_1 := -1
	for i, class in model.classes {
		if class == 0 { idx_0 = i }
		if class == 1 { idx_1 = i }
	}

	assert idx_0 >= 0, "should have class 0"
	assert idx_1 >= 0, "should have class 1"
	assert model.class_priors[idx_0] > 0.0, "prior for class 0 should be positive"
	assert model.class_priors[idx_1] > 0.0, "prior for class 1 should be positive"

	// Priors should match data distribution closely
	diff_0 := model.class_priors[idx_0] - expected_prior_0
	diff_1 := model.class_priors[idx_1] - expected_prior_1
	assert diff_0 * diff_0 < 0.0001, "class 0 prior should match distribution"
	assert diff_1 * diff_1 < 0.0001, "class 1 prior should match distribution"

	println("Prior Accuracy Test: PASS")
	println("  Expected prior[0]: ${expected_prior_0:.4f}, Actual: ${model.class_priors[idx_0]:.4f}")
	println("  Expected prior[1]: ${expected_prior_1:.4f}, Actual: ${model.class_priors[idx_1]:.4f}")
}

// Comprehensive metrics summary
fn test__naive_bayes_comprehensive_metrics() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	model := ml.naive_bayes_classifier(x_train, y_train)
	predictions := ml.naive_bayes_predict(model, x_test)
	
    // Calculate all metrics
    acc := ml.accuracy(y_test, predictions)
    prec_0 := ml.precision(y_test, predictions, 0)
    prec_1 := ml.precision(y_test, predictions, 1)
    rec_0 := ml.recall(y_test, predictions, 0)
    rec_1 := ml.recall(y_test, predictions, 1)
    f1_0 := ml.f1_score(y_test, predictions, 0)
    f1_1 := ml.f1_score(y_test, predictions, 1)
	
	println("\n=== Naive Bayes Comprehensive Metrics ===")
	println("Overall Accuracy: ${acc:.4f}")
	println("\nClass 0 (Did not survive):")
	println("  Precision: ${prec_0:.4f}")
	println("  Recall:    ${rec_0:.4f}")
	println("  F1 Score:  ${f1_0:.4f}")
	println("\nClass 1 (Survived):")
	println("  Precision: ${prec_1:.4f}")
	println("  Recall:    ${rec_1:.4f}")
	println("  F1 Score:  ${f1_1:.4f}")
	println("\nWeighted F1: ${(f1_0 + f1_1) / 2.0:.4f}")
	
	// All metrics should be valid
	assert acc >= 0.0 && acc <= 1.0, "accuracy should be in [0,1]"
	assert prec_0 >= 0.0 && prec_0 <= 1.0, "precision should be in [0,1]"
	assert rec_0 >= 0.0 && rec_0 <= 1.0, "recall should be in [0,1]"
}
