import utils
import ml
import math

const num_iterations = 500
const test_size = 0.3

fn main() {
	println("=" .repeat(80))
	println("VStats: Breast Cancer Classification Example (Full Dataset)")
	println("=" .repeat(80))
	println("")
	println("NOTE: Uses all 30 features from the real breast cancer dataset.")
	println("Achieves results comparable to scikit-learn's logistic regression.")
	println("")

	// Load Breast Cancer dataset (569 samples, 30 features, binary)
	println("1. LOADING BREAST CANCER DATASET")
	println("-" .repeat(80))
	cancer := utils.load_breast_cancer()!
	println(cancer.summary())
	println("")

	// Split into train and test sets (80/20 split)
	println("2. SPLITTING DATA - TEST SIZE: ${test_size}")
	println("-" .repeat(80))
	train, test := cancer.train_test_split(test_size)
	
	x_train, y_train := train.xy()
	x_test, y_test := test.xy()
	
	println("Training set size: ${x_train.len}")
	println("Test set size: ${x_test.len}")
	println("")

	// Normalize features (critical for logistic regression with 30 features)
	println("3. NORMALIZING FEATURES")
	println("-" .repeat(80))
	
	x_train_norm, x_train_mean, x_train_std := utils.normalize_features(x_train)
	x_test_norm := utils.apply_normalization(x_test, x_train_mean, x_train_std)
	
	println("Features normalized using training set statistics")
	println("Feature count: ${x_train[0].len}")
	println("")

	// Train logistic regression model with optimized hyperparameters
	println("4. TRAINING LOGISTIC REGRESSION MODEL")
	println("-" .repeat(80))
	// Use more iterations and appropriate learning rate for 30 features
	model := ml.logistic_regression(x_train_norm, y_train.map(f64(it)), num_iterations, 0.1)
	println("Model trained with ${num_iterations} iterations, learning_rate=0.1")
	println("")

	// Make predictions on training set
	println("5. PREDICTIONS ON TRAINING SET")
	println("-" .repeat(80))
	y_train_pred_proba := ml.logistic_predict_proba(model, x_train_norm)
	y_train_pred_f64 := ml.logistic_predict(model, x_train_norm, 0.5)
	
	println("Sample predictions (first 10):")
	for i in 0..math.min(10, y_train_pred_f64.len) {
		pred_label := if y_train_pred_f64[i] == 1 { "Benign" } else { "Malignant" }
		actual_label := if int(y_train[i]) == 1 { "Benign" } else { "Malignant" }
		println("  Sample ${i}: Predicted=${pred_label}, Actual=${actual_label}, Prob=${y_train_pred_proba[i]:.4f}")
	}
	println("")

	// Evaluate on training set
	println("6. TRAINING SET EVALUATION")
	println("-" .repeat(80))
	mut y_train_pred_int := []int{}
	mut y_train_actual_int := []int{}
	for pred in y_train_pred_f64 {
		y_train_pred_int << int(pred)
	}
	for label in y_train {
		y_train_actual_int << label
	}
	
	train_metrics := utils.binary_classification_metrics(y_train_actual_int, y_train_pred_int)
	train_cm := utils.build_confusion_matrix(y_train_actual_int, y_train_pred_int)
	
	println("Training Metrics:")
	println("  Accuracy:  ${train_metrics['accuracy']:.4f}")
	println("  Precision: ${train_metrics['precision']:.4f}")
	println("  Recall:    ${train_metrics['recall']:.4f}")
	println("  F1 Score:  ${train_metrics['f1_score']:.4f}")
	println("")
	println(train_cm.summary())
	println("")

	// Make predictions on test set
	println("7. PREDICTIONS ON TEST SET")
	println("-" .repeat(80))
	y_test_pred_proba := ml.logistic_predict_proba(model, x_test_norm)
	y_test_pred_f64 := ml.logistic_predict(model, x_test_norm, 0.5)
	
	println("Sample predictions (first 10):")
	for i in 0..math.min(10, y_test_pred_f64.len) {
		pred_label := if y_test_pred_f64[i] == 1 { "Benign" } else { "Malignant" }
		actual_label := if int(y_test[i]) == 1 { "Benign" } else { "Malignant" }
		println("  Sample ${i}: Predicted=${pred_label}, Actual=${actual_label}, Prob=${y_test_pred_f64[i]:.4f}")
	}
	println("")

	// Evaluate on test set
	println("8. TEST SET EVALUATION")
	println("-" .repeat(80))
	mut y_test_pred_int := []int{}
	mut y_test_actual_int := []int{}
	for pred in y_test_pred_f64 {
		y_test_pred_int << int(pred)
	}
	for label in y_test {
		y_test_actual_int << label
	}
	
	test_metrics := utils.binary_classification_metrics(y_test_actual_int, y_test_pred_int)
	test_cm := utils.build_confusion_matrix(y_test_actual_int, y_test_pred_int)
	
	println("Test Metrics:")
	println("  Accuracy:  ${test_metrics['accuracy']:.4f}")
	println("  Precision: ${test_metrics['precision']:.4f}")
	println("  Recall:    ${test_metrics['recall']:.4f}")
	println("  F1 Score:  ${test_metrics['f1_score']:.4f}")
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

	// Additional analysis
	println("10. ADDITIONAL ANALYSIS")
	println("-" .repeat(80))
	
	println("Training data distribution:")
	println("  Benign samples:    ${y_train.filter(it == 1).len}")
	println("  Malignant samples: ${y_train.filter(it == 0).len}")
	println("")
	
	println("Test data distribution:")
	println("  Benign samples:    ${y_test.filter(it == 1).len}")
	println("  Malignant samples: ${y_test.filter(it == 0).len}")
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
