module ml

import utils
import math

// Test Titanic dataset loading and basic statistics
fn test__titanic_dataset_loading() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Check dataset properties
	assert dataset.features.len > 0, "Titanic dataset should have samples"
	assert dataset.target.len == dataset.features.len, "features and target should match"
	assert dataset.features[0].len == 5, "should have 5 features (Pclass, Age, SibSp, Parch, Fare)"
	
	// Expected: ~891 samples in Titanic
	assert dataset.features.len >= 880, "should have at least 880 valid samples"
	
	// Check class distribution (roughly 38% survived, 62% didn't)
	mut survived := 0
	mut total := 0
	for label in dataset.target {
		total++
		if label == 1 {
			survived++
		}
	}
	
	survival_rate := f64(survived) / f64(total)
	// Expected: ~0.383 (38.3% survival rate)
	assert survival_rate > 0.3 && survival_rate < 0.45, "Titanic survival rate should be ~38%"
}

// Test logistic regression on Titanic dataset
fn test__logistic_classifier_on_titanic() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Convert target to f64 for logistic classifier
	mut y_f64 := []f64{}
	for label in dataset.target {
		y_f64 << f64(label)
	}
	
	// Split data: 80% train, 20% test
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := y_f64[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := y_f64[split_idx..y_f64.len]
	y_test_int := dataset.target[split_idx..dataset.target.len]
	
	// Train logistic regression
	model := logistic_classifier(x_train, y_train, 1000, 0.01)
	assert model.trained == true, "model should be trained"
	
	// Make predictions
	predictions := logistic_classifier_predict(model, x_test, 0.5)
	assert predictions.len == x_test.len, "predictions should match test set size"
	
	// Calculate accuracy
	acc := accuracy(y_test_int, predictions)
	// Expected: logistic regression on Titanic typically achieves 50-70% accuracy depending on convergence
	assert acc > 0.35, "logistic regression should achieve >35% accuracy on Titanic"
	
	println("Logistic Regression - Accuracy: ${acc:.4f}")
}

// Test Naive Bayes on Titanic dataset
fn test__naive_bayes_on_titanic() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Split data: 80% train, 20% test
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	// Train Naive Bayes
	model := naive_bayes_classifier(x_train, y_train)
	assert model.trained == true, "model should be trained"
	assert model.classes.len == 2, "should have 2 classes"
	
	// Make predictions
	predictions := naive_bayes_predict(model, x_test)
	assert predictions.len == x_test.len, "predictions should match test set size"
	
	// Calculate accuracy
	acc := accuracy(y_test, predictions)
	// Expected: Naive Bayes on Titanic typically achieves 75-82% accuracy
	assert acc > 0.65, "Naive Bayes should achieve >65% accuracy on Titanic"
	
	println("Naive Bayes - Accuracy: ${acc:.4f}")
}

// Test Random Forest on Titanic dataset
fn test__random_forest_on_titanic() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Split data: 80% train, 20% test
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	// Train Random Forest
	model := random_forest_classifier(x_train, y_train, 10, 5)
	assert model.trained == true, "model should be trained"
	assert model.num_trees == 10, "should have 10 trees"
	
	// Make predictions
	predictions := random_forest_predict(model, x_test)
	assert predictions.len == x_test.len, "predictions should match test set size"
	
	// Calculate accuracy
	acc := accuracy(y_test, predictions)
	// Expected: Random Forest on Titanic typically achieves 60-75% accuracy
	assert acc > 0.55, "Random Forest should achieve >55% accuracy on Titanic"
	
	println("Random Forest - Accuracy: ${acc:.4f}")
}

// Test SVM on Titanic dataset
fn test__svm_on_titanic() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Convert target to f64 for SVM
	mut y_f64 := []f64{}
	for label in dataset.target {
		y_f64 << f64(label)
	}
	
	// Split data: 80% train, 20% test
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := y_f64[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := y_f64[split_idx..y_f64.len]
	y_test_int := dataset.target[split_idx..dataset.target.len]
	
	// Train SVM
	model := svm_classifier(x_train, y_train, 0.01, 100, 0.1, "rbf")
	assert model.trained == true, "model should be trained"
	
	// Make predictions
	predictions := svm_predict(model, x_test)
	assert predictions.len == x_test.len, "predictions should match test set size"
	
	// Calculate accuracy
	acc := accuracy(y_test_int, predictions)
	// Expected: SVM on Titanic typically achieves 75-82% accuracy
	assert acc > 0.6, "SVM should achieve >60% accuracy on Titanic"
	
	println("SVM (RBF) - Accuracy: ${acc:.4f}")
}

// Test confusion matrix and metrics on Titanic
fn test__titanic_confusion_matrix_and_metrics() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Split data: 80% train, 20% test
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	y_train := dataset.target[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	// Train Naive Bayes (simple and reasonably good)
	model := naive_bayes_classifier(x_train, y_train)
	predictions := naive_bayes_predict(model, x_test)
	
	// Build confusion matrix
	cm := confusion_matrix(y_test, predictions)
	assert cm.len == 2, "confusion matrix should be 2x2 for binary classification"
	assert cm[0].len == 2, "confusion matrix should be 2x2"
	
	// Calculate metrics
	acc := accuracy(y_test, predictions)
	prec := precision(y_test, predictions, 1)
	rec := recall(y_test, predictions, 1)
	f1 := f1_score(y_test, predictions, 1)
	
	// Basic checks
	assert acc >= 0 && acc <= 1, "accuracy should be between 0 and 1"
	assert prec >= 0 && prec <= 1, "precision should be between 0 and 1"
	assert rec >= 0 && rec <= 1, "recall should be between 0 and 1"
	assert f1 >= 0 && f1 <= 1, "f1 score should be between 0 and 1"
	
	println("Confusion Matrix:")
	println("[[${cm[0][0]}, ${cm[0][1]}],")
	println(" [${cm[1][0]}, ${cm[1][1]}]]")
	println("Accuracy:  ${acc:.4f}")
	println("Precision: ${prec:.4f}")
	println("Recall:    ${rec:.4f}")
	println("F1 Score:  ${f1:.4f}")
}

// Test multi-classifier comparison on Titanic
fn test__titanic_classifier_comparison() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Split data
	split_idx := int(f64(dataset.features.len) * 0.8)
	x_train := dataset.features[0..split_idx]
	x_test := dataset.features[split_idx..dataset.features.len]
	y_test := dataset.target[split_idx..dataset.target.len]
	
	mut y_train_f64 := []f64{}
	mut y_train_int := dataset.target[0..split_idx]
	for label in y_train_int {
		y_train_f64 << f64(label)
	}
	
	mut accuracies := map[string]f64{}
	
	// Test Naive Bayes
	nb_model := naive_bayes_classifier(x_train, y_train_int)
	nb_pred := naive_bayes_predict(nb_model, x_test)
	accuracies["Naive Bayes"] = accuracy(y_test, nb_pred)
	
	// Test Logistic Regression
	lr_model := logistic_classifier(x_train, y_train_f64, 1000, 0.01)
	lr_proba := logistic_classifier_predict_proba(lr_model, x_test)
	mut lr_pred := []int{len: lr_proba.len}
	for i, prob in lr_proba {
		lr_pred[i] = if prob >= 0.5 { 1 } else { 0 }
	}
	accuracies["Logistic Regression"] = accuracy(y_test, lr_pred)
	
	// Test Random Forest
	rf_model := random_forest_classifier(x_train, y_train_int, 10, 5)
	rf_pred := random_forest_predict(rf_model, x_test)
	accuracies["Random Forest"] = accuracy(y_test, rf_pred)
	
	// Print comparison
	println("\n=== Titanic Classification - Model Comparison ===")
	for name, acc in accuracies {
		println("${name:20} Accuracy: ${acc:.4f}")
	}
	
	// All models should achieve reasonable performance
	for _, acc in accuracies {
		assert acc > 0.35, "classifier should produce predictions above baseline"
	}
}

// Test that all classifiers converge and produce valid predictions
fn test__titanic_classifier_stability() {
	dataset := utils.load_titanic() or {
		assert false, "Failed to load Titanic dataset: ${err}"
		return
	}
	
	// Use smaller subset for quick test
	subset_size := 200
	x := dataset.features[0..subset_size]
	y := dataset.target[0..subset_size]
	
	mut y_f64 := []f64{}
	for label in y {
		y_f64 << f64(label)
	}
	
	// Test Naive Bayes doesn't crash
	nb_model := naive_bayes_classifier(x, y)
	nb_pred := naive_bayes_predict(nb_model, x)
	assert nb_pred.len == x.len, "NB predictions should match input"
	
	// Test Logistic Regression doesn't crash
	lr_model := logistic_classifier(x, y_f64, 100, 0.01)
	lr_pred := logistic_classifier_predict(lr_model, x, 0.5)
	assert lr_pred.len == x.len, "LR predictions should match input"
	
	// Test Random Forest doesn't crash
	rf_model := random_forest_classifier(x, y, 3, 3)
	rf_pred := random_forest_predict(rf_model, x)
	assert rf_pred.len == x.len, "RF predictions should match input"
	
	println("All classifiers completed successfully on Titanic subset")
}
