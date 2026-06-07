// Scenario: Titanic Survival — Multi-Classifier Comparison
// Demonstrates: vstats.ml + vstats.utils — compare three classifiers on the same split
// Python equivalent: sklearn fit/predict loop + classification_report per model
module main

import vstats.utils
import vstats.ml

fn main() {
	println('=== Titanic Survival: Multi-Classifier Comparison ===\n')
	println('Dataset: Titanic (891 samples, 5 features: Pclass, Age, SibSp, Parch, Fare)')
	println('Target: Survived (0=No, 1=Yes)\n')

	// --- Setup ---
	dataset := utils.load_titanic()!
	train, test := dataset.train_test_split(0.2)
	x_train, y_train := train.xy()
	x_test, y_test   := test.xy()

	x_train_norm, feat_mean, feat_std := utils.normalize_features(x_train)
	x_test_norm := utils.apply_normalization(x_test, feat_mean, feat_std)

	println('Train: ${x_train.len} samples  Test: ${x_test.len} samples\n')

	// --- Core analysis: three classifiers, same train/test split ---

	// 1. Logistic Regression
	lr := ml.logistic_regression(x_train_norm, y_train.map(f64(it)), 300, 0.1)
	lr_pred := ml.logistic_predict(lr, x_test_norm, 0.5).map(int(it))
	lr_m := utils.binary_classification_metrics(y_test, lr_pred)

	// 2. Naive Bayes
	nb := ml.naive_bayes_classifier(x_train_norm, y_train)
	nb_pred := ml.naive_bayes_predict(nb, x_test_norm)
	nb_m := utils.binary_classification_metrics(y_test, nb_pred)

	// 3. Random Forest
	rf := ml.random_forest_classifier(x_train_norm, y_train, 20, 5)
	rf_pred := ml.random_forest_predict(rf, x_test_norm)
	rf_m := utils.binary_classification_metrics(y_test, rf_pred)

	// --- Interpret output ---
	println('Model                Accuracy  Precision  Recall    F1')
	println('-----------------------------------------------------------')
	println('Logistic Regression  ${lr_m["accuracy"]:.4f}    ${lr_m["precision"]:.4f}     ${lr_m["recall"]:.4f}    ${lr_m["f1_score"]:.4f}')
	println('Naive Bayes          ${nb_m["accuracy"]:.4f}    ${nb_m["precision"]:.4f}     ${nb_m["recall"]:.4f}    ${nb_m["f1_score"]:.4f}')
	println('Random Forest        ${rf_m["accuracy"]:.4f}    ${rf_m["precision"]:.4f}     ${rf_m["recall"]:.4f}    ${rf_m["f1_score"]:.4f}')

	// Pick winner by F1
	models := ['Logistic Regression', 'Naive Bayes', 'Random Forest']
	f1s := [lr_m['f1_score'], nb_m['f1_score'], rf_m['f1_score']]
	mut best_idx := 0
	for i in 1 .. f1s.len {
		if f1s[i] > f1s[best_idx] { best_idx = i }
	}
	println('\nBest F1: ${models[best_idx]} (${f1s[best_idx]:.4f})')
}
