// Scenario: Customer Churn Prediction
// Demonstrates: vstats.ml + vstats.utils — full binary classification pipeline
// Python equivalent: sklearn.ensemble.RandomForestClassifier + classification_report
module main

import vstats.utils
import vstats.ml

fn main() {
	println('=== Customer Churn Prediction ===\n')
	println('Using Breast Cancer dataset (malignant=churned, benign=retained).\n')

	// --- Setup ---
	dataset := utils.load_breast_cancer()!
	train, test := dataset.train_test_split(0.2)
	x_train, y_train := train.xy()
	x_test, y_test   := test.xy()

	// Normalize on train only — no leakage into test
	x_train_norm, feat_mean, feat_std := utils.normalize_features(x_train)
	x_test_norm := utils.apply_normalization(x_test, feat_mean, feat_std)

	println('Train: ${x_train.len} samples  Test: ${x_test.len} samples  Features: ${x_train[0].len}\n')

	// --- Core analysis ---

	// Baseline: logistic regression
	lr := ml.logistic_regression(x_train_norm, y_train.map(f64(it)), 200, 0.1)
	lr_pred := ml.logistic_predict(lr, x_test_norm, 0.5).map(int(it))
	lr_m := utils.binary_classification_metrics(y_test, lr_pred)
	println('Logistic Regression')
	println('  accuracy=${lr_m["accuracy"]:.4f}  precision=${lr_m["precision"]:.4f}  recall=${lr_m["recall"]:.4f}  f1=${lr_m["f1_score"]:.4f}')

	// Random Forest
	rf := ml.random_forest_classifier(x_train_norm, y_train, 20, 5)
	rf_pred := ml.random_forest_predict(rf, x_test_norm)
	rf_m := utils.binary_classification_metrics(y_test, rf_pred)
	println('\nRandom Forest (20 trees, max_depth=5)')
	println('  accuracy=${rf_m["accuracy"]:.4f}  precision=${rf_m["precision"]:.4f}  recall=${rf_m["recall"]:.4f}  f1=${rf_m["f1_score"]:.4f}')

	// ROC / AUC
	rf_proba := ml.random_forest_classifier_predict_proba(rf, x_test_norm)
	roc := utils.roc_curve(y_test, rf_proba)
	println('\nRandom Forest AUC: ${roc.auc:.4f}')

	// --- Interpret output ---
	println('\nConfusion matrix (Random Forest):')
	cm := utils.build_confusion_matrix(y_test, rf_pred)
	println(cm.summary())
}
