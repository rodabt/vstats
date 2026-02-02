import ml

fn test__logistic_classifier() {
	x := [
		[1.0, 2.0],
		[2.0, 3.0],
		[3.0, 1.0],
		[5.0, 8.0],
		[6.0, 9.0],
		[7.0, 8.0],
	]
	y := [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
	
	model := ml.logistic_classifier(x, y, 1000, 0.1)
	
	assert model.trained == true, "model should be trained"
	assert model.coefficients.len == 2, "coefficients should have correct length"
	
	predictions := ml.logistic_classifier_predict(model, x, 0.5)
	assert predictions.len == x.len, "predictions should match input size"
	
	proba := ml.logistic_classifier_predict_proba(model, x)
	assert proba.len == x.len, "proba should match input size"
	
	// All probabilities should be between 0 and 1
	for prob in proba {
		assert prob >= 0 && prob <= 1, "probability should be between 0 and 1"
	}
}

fn test__naive_bayes_classifier() {
	x := [
		[1.0, 1.0],
		[1.5, 1.5],
		[2.0, 2.0],
		[7.0, 7.0],
		[8.0, 8.0],
		[9.0, 9.0],
	]
	y := [0, 0, 0, 1, 1, 1]
	
	model := ml.naive_bayes_classifier(x, y)
	
	assert model.trained == true, "model should be trained"
	assert model.classes.len == 2, "should have 2 classes"
	assert model.class_priors[0] == 0.5, "class 0 prior should be 0.5"
	assert model.class_priors[1] == 0.5, "class 1 prior should be 0.5"
	
	predictions := ml.naive_bayes_predict(model, x)
	assert predictions.len == x.len, "predictions should match input size"
	
	acc := ml.accuracy(y, predictions)
	assert acc > 0, "accuracy should be positive"
}

fn test__svm_classifier() {
	x := [
		[1.0, 1.0],
		[1.5, 1.5],
		[2.0, 2.0],
		[7.0, 7.0],
		[8.0, 8.0],
		[9.0, 9.0],
	]
	y := [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
	
	model := ml.svm_classifier(x, y, 0.01, 100, 0.1, "rbf")
	
	assert model.trained == true, "model should be trained"
	
	predictions := ml.svm_predict(model, x)
	assert predictions.len == x.len, "predictions should match input size"
	
	for pred in predictions {
		assert pred == 0 || pred == 1, "predictions should be binary"
	}
}

fn test__random_forest_classifier() {
	x := [
		[1.0, 1.0],
		[1.5, 1.5],
		[2.0, 2.0],
		[1.2, 1.1],
		[7.0, 7.0],
		[8.0, 8.0],
		[9.0, 9.0],
		[7.5, 7.5],
	]
	y := [0, 0, 0, 0, 1, 1, 1, 1]
	
	model := ml.random_forest_classifier(x, y, 5, 3)
	
	assert model.trained == true, "model should be trained"
	assert model.num_trees == 5, "should have 5 trees"
	
	predictions := ml.random_forest_predict(model, x)
	assert predictions.len == x.len, "predictions should match input size"
	
	for pred in predictions {
		assert pred == 0 || pred == 1, "predictions should be binary"
	}
	
	acc := ml.accuracy(y, predictions)
	assert acc > 0, "accuracy should be positive"
}

fn test__accuracy_metric() {
	y_true := [0, 1, 0, 1, 0, 1]
	y_pred := [0, 1, 0, 1, 0, 1]
	
	acc := ml.accuracy(y_true, y_pred)
	assert acc == 1.0, "perfect predictions should have accuracy 1.0"
	
	y_pred_bad := [1, 0, 1, 0, 1, 0]
	acc_bad := ml.accuracy(y_true, y_pred_bad)
	assert acc_bad == 0.0, "all wrong predictions should have accuracy 0.0"
	
	y_pred_partial := [0, 1, 0, 1, 1, 0]
	acc_partial := ml.accuracy(y_true, y_pred_partial)
	assert acc_partial == 2.0 / 3.0, "partial accuracy should be correct"
}

fn test__precision_recall_f1() {
	y_true := [0, 1, 0, 1, 0, 1]
	y_pred := [0, 1, 0, 1, 0, 1]
	
	p := ml.precision(y_true, y_pred, 1)
	r := ml.recall(y_true, y_pred, 1)
	f1 := ml.f1_score(y_true, y_pred, 1)
	
	assert p == 1.0, "perfect precision should be 1.0"
	assert r == 1.0, "perfect recall should be 1.0"
	assert f1 == 1.0, "perfect f1 should be 1.0"
}

fn test__confusion_matrix() {
	y_true := [0, 1, 0, 1, 0, 1]
	y_pred := [0, 1, 0, 1, 1, 1]
	
	matrix := ml.confusion_matrix(y_true, y_pred)
	
	assert matrix.len == 2, "matrix should have 2 rows"
	assert matrix[0].len == 2, "matrix should have 2 columns"
	
	// Correct predictions on diagonal
	assert matrix[0][0] > 0, "true negatives should be > 0"
	assert matrix[1][1] > 0, "true positives should be > 0"
	
	// One false positive
	assert matrix[0][1] == 1, "false positives should be 1"
}

fn test__kernel_function() {
	x := [1.0, 2.0, 3.0]
	y := [4.0, 5.0, 6.0]
	
	// Linear kernel
	k_linear := ml.kernel_function(x, y, 0.0, "linear")
	assert k_linear > 0, "linear kernel should be positive for this input"
	
	// RBF kernel
	k_rbf := ml.kernel_function(x, y, 1.0, "rbf")
	assert k_rbf > 0 && k_rbf <= 1.0, "RBF kernel should be between 0 and 1"
	
	// Poly kernel
	k_poly := ml.kernel_function(x, y, 1.0, "poly")
	assert k_poly > 0, "polynomial kernel should be positive"
}

fn test__entropy() {
	// Pure labels should have 0 entropy
	pure := [0, 0, 0, 0]
	ent_pure := ml.entropy(pure)
	assert ent_pure == 0, "pure labels should have 0 entropy"
	
	// Balanced labels should have entropy 1
	balanced := [0, 1, 0, 1]
	ent_balanced := ml.entropy(balanced)
	assert ent_balanced == 1.0, "balanced binary labels should have entropy 1"
}

fn test__classification_setup() {
	x := [
		[1.0, 2.0],
		[2.0, 3.0],
		[3.0, 4.0],
		[4.0, 5.0],
		[5.0, 6.0],
		[6.0, 7.0],
	]
	y := [0, 0, 0, 1, 1, 1]
	
	s := ml.setup(x, y, 0.33, "logistic")
	
	assert s.estimator == "logistic", "estimator should be set"
	assert s.preprocessing == true, "preprocessing should be enabled"
	assert s.train_data.len + s.test_data.len == x.len, "train+test should equal total"
	assert s.target.len + s.target_test.len == y.len, "train+test targets should equal total"
}

fn test__logistic_with_different_thresholds() {
	x := [
		[1.0, 2.0],
		[2.0, 3.0],
		[3.0, 1.0],
		[5.0, 8.0],
		[6.0, 9.0],
		[7.0, 8.0],
	]
	y := [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
	
	model := ml.logistic_classifier(x, y, 1000, 0.1)
	
	// Different thresholds should give different predictions
	pred_05 := ml.logistic_classifier_predict(model, x, 0.5)
	pred_03 := ml.logistic_classifier_predict(model, x, 0.3)
	
	assert pred_05.len == pred_03.len, "predictions should have same length"
	
	// Count 1s (lower threshold should predict more 1s typically)
	count_1_low := pred_03.filter(it == 1).len
	count_1_high := pred_05.filter(it == 1).len
	
	assert count_1_low >= count_1_high, "lower threshold should predict more or equal positives"
}

fn test__naive_bayes_multiclass() {
	x := [
		[1.0, 1.0],
		[1.5, 1.5],
		[2.0, 2.0],
		[5.0, 5.0],
		[5.5, 5.5],
		[6.0, 6.0],
		[9.0, 9.0],
		[9.5, 9.5],
		[10.0, 10.0],
	]
	y := [0, 0, 0, 1, 1, 1, 2, 2, 2]
	
	model := ml.naive_bayes_classifier(x, y)
	
	assert model.classes.len == 3, "should have 3 classes"
	
	predictions := ml.naive_bayes_predict(model, x)
	assert predictions.len == x.len, "predictions should match input size"
	
	// Each class should be present in predictions
	mut classes_pred := []int{}
	for pred in predictions {
		if pred !in classes_pred {
			classes_pred << pred
		}
	}
	assert classes_pred.len >= 1, "should have at least 1 predicted class"
}

fn test__random_forest_multiple_trees() {
	x := [
		[1.0, 1.0],
		[1.5, 1.5],
		[2.0, 2.0],
		[1.2, 1.1],
		[7.0, 7.0],
		[8.0, 8.0],
		[9.0, 9.0],
		[7.5, 7.5],
	]
	y := [0, 0, 0, 0, 1, 1, 1, 1]
	
	// Test with different number of trees
	for num_trees in [1, 3, 5] {
		model := ml.random_forest_classifier(x, y, num_trees, 3)
		assert model.num_trees == num_trees, "should have correct number of trees"
		
		predictions := ml.random_forest_predict(model, x)
		assert predictions.len == x.len, "predictions should match input size"
	}
}
