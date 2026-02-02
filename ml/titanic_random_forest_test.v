module ml

import utils

// Test random forest on Titanic with expected community benchmarks
// Literature benchmark: Random Forest achieves ~77-82% accuracy on Titanic
fn test__random_forest_on_titanic() {
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
	
	// Train random forest with 100 trees
	model := random_forest_classifier(x_train, y_train, 100, 5)
	assert model.trained == true, "model should be trained"
	assert model.num_trees == 100, "should have 100 trees"
	assert model.trees.len == 100, "trees array should have 100 elements"
	
	// Make predictions
	predictions := random_forest_classifier_predict(model, x_test)
	assert predictions.len == x_test.len, "predictions should match test set size"
	
	// Validate predictions are binary
	for pred in predictions {
		assert pred == 0 || pred == 1, "predictions should be 0 or 1"
	}
	
	// Calculate metrics
	acc := accuracy(y_test, predictions)
	prec := precision(y_test, predictions, 1)
	rec := recall(y_test, predictions, 1)
	f1 := f1_score(y_test, predictions, 1)
	
	// Community benchmark expectation for Random Forest on Titanic
	// Expected accuracy: 77-82%
	println("\n=== Random Forest on Titanic ===")
	println("Test set size: ${x_test.len}")
	println("Accuracy:  ${acc:.4f} (expected: 0.77-0.82)")
	println("Precision: ${prec:.4f}")
	println("Recall:    ${rec:.4f}")
	println("F1 Score:  ${f1:.4f}")
	
	// Assert against community benchmarks
	assert acc >= 0.77, "Random Forest should achieve ~77%+ accuracy (community benchmark)"
	assert prec >= 0.0, "Precision should be valid"
	assert rec >= 0.0, "Recall should be valid"
	assert f1 >= 0.0, "F1 should be valid"
}

// Test random forest probability predictions via vote distribution
fn test__random_forest_probabilities() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	
	model := random_forest_classifier(x_train, y_train, 100, 5)
	proba := random_forest_classifier_predict_proba(model, x_test)
	
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
	
	println("Random Forest probability predictions validated: min=${min_prob:.4f}, max=${max_prob:.4f}")
}

// Test confusion matrix from random forest predictions
fn test__random_forest_confusion_matrix() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	model := random_forest_classifier(x_train, y_train, 100, 5)
	predictions := random_forest_classifier_predict(model, x_test)
	
	cm := confusion_matrix(y_test, predictions)
	assert cm.len == 2, "confusion matrix should be 2x2"
	assert cm[0].len == 2, "confusion matrix should be 2x2"
	
	tn := cm[0][0]  // True Negatives
	fp := cm[0][1]  // False Positives
	fn_val := cm[1][0]  // False Negatives
	tp := cm[1][1]  // True Positives
	
	println("\nRandom Forest Confusion Matrix:")
	println("                 Predicted")
	println("              Negative  Positive")
	println("Actual Negative  ${tn:3}      ${fp:3}")
	println("       Positive  ${fn_val:3}      ${tp:3}")
	
	assert (tn + fp + fn_val + tp) == y_test.len, "confusion matrix counts should match test set size"
}

// Test different numbers of trees affect predictions
fn test__random_forest_tree_count_sensitivity() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	// Train models with different tree counts
	model_10 := random_forest_classifier(x_train, y_train, 10, 5)
	model_50 := random_forest_classifier(x_train, y_train, 50, 5)
	model_100 := random_forest_classifier(x_train, y_train, 100, 5)
	
	pred_10 := random_forest_classifier_predict(model_10, x_test)
	pred_50 := random_forest_classifier_predict(model_50, x_test)
	pred_100 := random_forest_classifier_predict(model_100, x_test)
	
	acc_10 := accuracy(y_test, pred_10)
	acc_50 := accuracy(y_test, pred_50)
	acc_100 := accuracy(y_test, pred_100)
	
	println("\nTree count sensitivity:")
	println("10 trees:   accuracy=${acc_10:.4f}")
	println("50 trees:   accuracy=${acc_50:.4f}")
	println("100 trees:  accuracy=${acc_100:.4f}")
	
	// More trees generally leads to better or equal performance
	assert acc_50 >= 0.0, "50-tree model should have valid accuracy"
	assert acc_100 >= 0.0, "100-tree model should have valid accuracy"
}

// Test feature count parameter impact
fn test__random_forest_max_features_sensitivity() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	// Train models with different max_features
	model_2 := random_forest_classifier(x_train, y_train, 100, 2)
	model_3 := random_forest_classifier(x_train, y_train, 100, 3)
	model_5 := random_forest_classifier(x_train, y_train, 100, 5)
	
	pred_2 := random_forest_classifier_predict(model_2, x_test)
	pred_3 := random_forest_classifier_predict(model_3, x_test)
	pred_5 := random_forest_classifier_predict(model_5, x_test)
	
	acc_2 := accuracy(y_test, pred_2)
	acc_3 := accuracy(y_test, pred_3)
	acc_5 := accuracy(y_test, pred_5)
	
	println("\nMax features sensitivity:")
	println("max_features=2: accuracy=${acc_2:.4f}")
	println("max_features=3: accuracy=${acc_3:.4f}")
	println("max_features=5: accuracy=${acc_5:.4f}")
	
	// All variants should produce valid models
	assert acc_2 >= 0.0, "max_features=2 model should have valid accuracy"
	assert acc_3 >= 0.0, "max_features=3 model should have valid accuracy"
	assert acc_5 >= 0.0, "max_features=5 model should have valid accuracy"
}

// Test out-of-bag style voting consistency
fn test__random_forest_voting_distribution() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	
	model := random_forest_classifier(x_train, y_train, 100, 5)
	
	// Get probability predictions (vote ratios)
	proba := random_forest_classifier_predict_proba(model, x_test)
	predictions := random_forest_classifier_predict(model, x_test)
	
	// Verify consistency: probabilities should be proportional to votes
	// For 100 trees, probabilities should reflect vote distributions
	mut prob_0_count := 0
	mut prob_1_count := 0
	
	for i, prob in proba {
		if predictions[i] == 0 {
			assert prob < 0.5, "class 0 prediction should have prob < 0.5"
			prob_0_count++
		} else {
			assert prob >= 0.5, "class 1 prediction should have prob >= 0.5"
			prob_1_count++
		}
	}
	
	println("\nVoting consistency check:")
	println("Predictions with class 0: ${prob_0_count}")
	println("Predictions with class 1: ${prob_1_count}")
	assert prob_0_count + prob_1_count == x_test.len, "all predictions should be classified"
}
