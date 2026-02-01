import utils
import ml
import math

fn main() {
	println("=" .repeat(80))
	println("VStats: Iris Logistic Classification Example")
	println("=" .repeat(80))
	println("")
	println("NOTE: VStats logistic regression uses basic gradient descent with L2 regularization.")
	println("For production use, see the Python example which uses scikit-learn's optimized solver.")
	println("")

	// Load Iris dataset
	println("1. LOADING IRIS DATASET")
	println("-" .repeat(80))
	iris := utils.load_iris()!
	println(iris.summary())
	println("")

	// Split into train and test sets (80/20 split)
	println("2. SPLITTING DATA (80% train, 20% test)")
	println("-" .repeat(80))
	train, test := iris.train_test_split(0.2)
	
	x_train, y_train := train.xy()
	x_test, y_test := test.xy()
	
	println("Training set size: ${x_train.len}")
	println("Test set size: ${x_test.len}")
	println("")

	// Normalize features (important for logistic regression)
	println("2B. NORMALIZING FEATURES")
	println("-" .repeat(80))
	
	x_train_norm, x_train_mean, x_train_std := utils.normalize_features(x_train)
	x_test_norm := utils.apply_normalization(x_test, x_train_mean, x_train_std)
	
	println("Features normalized using training set mean and std")
	println("")

	// For logistic regression, we need binary classification
	// Let's create binary classification: Class 1 vs Class 2 (excluding Class 0 for better separation)
	println("3. PREPARING BINARY CLASSIFICATION")
	println("-" .repeat(80))
	
	// Filter to only use classes 1 and 2 (more difficult binary classification)
	mut x_train_filtered := [][]f64{}
	mut y_train_filtered := []f64{}
	mut x_test_filtered := [][]f64{}
	mut y_test_filtered := []f64{}
	
	for i in 0..x_train_norm.len {
		if y_train[i] != 0 {
			x_train_filtered << x_train_norm[i]
			y_train_filtered << if y_train[i] == 1 { 1.0 } else { 0.0 }
		}
	}
	
	for i in 0..x_test_norm.len {
		if y_test[i] != 0 {
			x_test_filtered << x_test_norm[i]
			y_test_filtered << if y_test[i] == 1 { 1.0 } else { 0.0 }
		}
	}
	
	y_train_binary := y_train_filtered
	y_test_binary := y_test_filtered
	x_train_use := x_train_filtered
	x_test_use := x_test_filtered
	
	println("Binary classification: Class 1 (Versicolor) vs Class 2 (Virginica)")
	println("Training samples: ${x_train_use.len}, Test samples: ${x_test_use.len}")
	println("")

	// Train logistic regression model
	println("4. TRAINING LOGISTIC REGRESSION MODEL")
	println("-" .repeat(80))
	model := ml.logistic_regression(x_train_use, y_train_binary, 10000, 0.01)
	println("Model trained with 10000 iterations, learning_rate=0.01")
	println("")

	// Make predictions on training set
	println("5. PREDICTIONS ON TRAINING SET")
	println("-" .repeat(80))
	y_train_pred_proba := ml.logistic_predict_proba(model, x_train_use)
	y_train_pred := ml.logistic_predict(model, x_train_use, 0.5)
	
	println("Sample predictions (first 10):")
	for i in 0..math.min(10, y_train_pred.len) {
		pred_label := if y_train_pred[i] == 1 { "Versicolor" } else { "Virginica" }
		actual_label := if y_train_binary[i] == 1 { "Versicolor" } else { "Virginica" }
		println("  Sample ${i}: Predicted=${pred_label}, Actual=${actual_label}, Prob=${y_train_pred_proba[i]:.4f}")
	}
	println("")

	// Evaluate on training set
	println("6. TRAINING SET EVALUATION")
	println("-" .repeat(80))
	mut y_train_pred_int := []int{}
	mut y_train_actual_int := []int{}
	for prob in y_train_pred_proba {
		y_train_pred_int << if prob > 0.5 { 1 } else { 0 }
	}
	for i in 0..y_train_binary.len {
		y_train_actual_int << if y_train_binary[i] == 1.0 { 1 } else { 0 }
	}
	
	train_metrics := utils.binary_classification_metrics(y_train_actual_int, y_train_pred_int)
	train_cm := utils.build_confusion_matrix(y_train_actual_int, y_train_pred_int)
	
	println("Training Metrics:")
	println("  Accuracy:  ${train_metrics['accuracy']:.4f}")
	println("  Precision: ${train_metrics['precision']:.4f}")
	println("  Recall:    ${train_metrics['recall']:.4f}")
	println("  F1 Score:  ${train_metrics['f1_score']:.4f}")
	println("  AUC/FPR:   ${train_metrics['fpr']:.4f}")
	println("")
	println(train_cm.summary())
	println("")

	// Make predictions on test set
	println("7. PREDICTIONS ON TEST SET")
	println("-" .repeat(80))
	y_test_pred_proba := ml.logistic_predict_proba(model, x_test_use)
	y_test_pred := ml.logistic_predict(model, x_test_use, 0.5)
	
	println("Sample predictions (first 10):")
	for i in 0..math.min(10, y_test_pred.len) {
		pred_label := if y_test_pred[i] == 1 { "Versicolor" } else { "Virginica" }
		actual_label := if y_test_binary[i] == 1 { "Versicolor" } else { "Virginica" }
		println("  Sample ${i}: Predicted=${pred_label}, Actual=${actual_label}, Prob=${y_test_pred_proba[i]:.4f}")
	}
	println("")

	// Evaluate on test set
	println("8. TEST SET EVALUATION")
	println("-" .repeat(80))
	mut y_test_pred_int := []int{}
	mut y_test_actual_int := []int{}
	for prob in y_test_pred_proba {
		y_test_pred_int << if prob > 0.5 { 1 } else { 0 }
	}
	for i in 0..y_test_binary.len {
		y_test_actual_int << if y_test_binary[i] == 1.0 { 1 } else { 0 }
	}
	
	test_metrics := utils.binary_classification_metrics(y_test_actual_int, y_test_pred_int)
	test_cm := utils.build_confusion_matrix(y_test_actual_int, y_test_pred_int)
	
	println("Test Metrics:")
	println("  Accuracy:  ${test_metrics['accuracy']:.4f}")
	println("  Precision: ${test_metrics['precision']:.4f}")
	println("  Recall:    ${test_metrics['recall']:.4f}")
	println("  F1 Score:  ${test_metrics['f1_score']:.4f}")
	println("  AUC/FPR:   ${test_metrics['fpr']:.4f}")
	println("")
	println(test_cm.summary())
	println("")

	// ROC-AUC Analysis
	println("9. ROC-AUC ANALYSIS")
	println("-" .repeat(80))
	roc := utils.roc_curve(y_test_pred_int, y_test_pred_proba)
	println("ROC Curve Statistics:")
	println("  AUC Score: ${roc.auc:.4f}")
	println("  Interpretation: AUC ranges from 0.5 (random) to 1.0 (perfect)")
	if roc.auc > 0.9 {
		println("  Classification: Excellent discrimination")
	} else if roc.auc > 0.8 {
		println("  Classification: Good discrimination")
	} else if roc.auc > 0.7 {
		println("  Classification: Fair discrimination")
	} else {
		println("  Classification: Poor discrimination")
	}
	println("")

	// Effect size and distribution analysis
	println("10. ADDITIONAL ANALYSIS")
	println("-" .repeat(80))
	
	// Convert y_train to f64 for statistical analysis
	mut y_train_f64 := []f64{}
	for val in y_train {
		y_train_f64 << f64(val)
	}
		
	println("Training data distribution:")
	println("  Class 0 samples: ${y_train_binary.filter(it == 1.0).len}")
	println("  Class Rest samples: ${y_train_binary.filter(it == 0.0).len}")
	println("")

	// Summary comparison
	println("11. SUMMARY: TRAIN VS TEST")
	println("-" .repeat(80))
	println("Metric          | Train    | Test     | Difference")
	println("-" .repeat(55))
	
	acc_diff := train_metrics['accuracy'] - test_metrics['accuracy']
	prec_diff := train_metrics['precision'] - test_metrics['precision']
	rec_diff := train_metrics['recall'] - test_metrics['recall']
	f1_diff := train_metrics['f1_score'] - test_metrics['f1_score']
	
	println("Accuracy        | ${train_metrics['accuracy']:.4f}  | ${test_metrics['accuracy']:.4f}  | ${acc_diff:.4f}")
	println("Precision       | ${train_metrics['precision']:.4f}  | ${test_metrics['precision']:.4f}  | ${prec_diff:.4f}")
	println("Recall          | ${train_metrics['recall']:.4f}  | ${test_metrics['recall']:.4f}  | ${rec_diff:.4f}")
	println("F1 Score        | ${train_metrics['f1_score']:.4f}  | ${test_metrics['f1_score']:.4f}  | ${f1_diff:.4f}")
	println("")

	println("=" .repeat(80))
	println("✓ Classification example complete!")
	println("=" .repeat(80))
}
