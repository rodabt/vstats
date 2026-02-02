module ml

import math
import linalg
import stats
import rand

// Classification Model Definitions

pub struct LogisticClassifier[T] {
pub mut:
	coefficients  []T
	intercept     T
	classes       []T
	trained       bool
	feature_means []f64
	feature_stds  []f64
}

pub struct NaiveBayesClassifier {
pub mut:	
	class_priors      map[int]f64
	feature_means     map[int][][]f64
	feature_stds      map[int][][]f64
	classes           []int
	trained           bool
	global_means      []f64  // For feature normalization
	global_stds       []f64  // For feature normalization
}

pub struct SVMClassifier {
pub mut:	
	support_vectors [][]f64
	support_labels  []f64
	alphas          []f64
	bias            f64
	kernel_type     string
	gamma           f64
	trained         bool
}

pub struct RandomForestClassifier {
pub mut:
	trees        []TreeNode
	num_trees    int
	num_features int
	trained      bool
}

pub struct TreeNode {
pub mut:	
	feature    int
	threshold  f64
	left       ?&TreeNode
	right      ?&TreeNode
	class_pred ?int
}

// ============================================================================
// Logistic Classifier (Binary Classification)
// ============================================================================

pub fn logistic_classifier(x [][]f64, y []f64, iterations int, learning_rate f64) LogisticClassifier[f64] {
	assert x.len == y.len, "number of samples must match"
	assert x.len > 0, "must have at least one sample"
	
	n := f64(x.len)
	p := x[0].len
	
	// Calculate feature means and standard deviations for normalization
	mut feature_means := []f64{len: p, init: 0.0}
	mut feature_stds := []f64{len: p, init: 0.0}
	
	// Compute means
	for j in 0 .. p {
		mut sum := 0.0
		for i in 0 .. x.len {
			sum += x[i][j]
		}
		feature_means[j] = sum / n
	}
	
	// Compute standard deviations
	for j in 0 .. p {
		mut sum_sq := 0.0
		for i in 0 .. x.len {
			diff := x[i][j] - feature_means[j]
			sum_sq += diff * diff
		}
		feature_stds[j] = math.sqrt(sum_sq / n)
		// Avoid division by zero
		if feature_stds[j] == 0.0 {
			feature_stds[j] = 1.0
		}
	}
	
	// Normalize features (z-score normalization)
	mut x_normalized := [][]f64{len: x.len}
	for i in 0 .. x.len {
		mut row := []f64{len: p}
		for j in 0 .. p {
			row[j] = (x[i][j] - feature_means[j]) / feature_stds[j]
		}
		x_normalized[i] = row
	}
	
	mut coefficients := []f64{len: p, init: 0.0}
	mut intercept := 0.0
	
	// Gradient descent on normalized data
	for _ in 0 .. iterations {
		mut pred := []f64{len: x.len}
		for i in 0 .. x.len {
			mut z := intercept
			for j in 0 .. p {
				z += coefficients[j] * x_normalized[i][j]
			}
			pred[i] = sigmoid_f64(z)
		}
		
		// Calculate gradients
		mut intercept_grad := 0.0
		mut coeff_grad := []f64{len: p, init: 0.0}
		
		for i in 0 .. x.len {
			error := pred[i] - y[i]
			intercept_grad += error
			for j in 0 .. p {
				coeff_grad[j] += error * x_normalized[i][j]
			}
		}
		
		// Update parameters
		intercept -= learning_rate * intercept_grad / n
		for j in 0 .. p {
			coefficients[j] -= learning_rate * coeff_grad[j] / n
		}
	}
	
	return LogisticClassifier[f64]{
		coefficients: coefficients
		intercept: intercept
		classes: [0.0, 1.0]
		trained: true
		feature_means: feature_means
		feature_stds: feature_stds
	}
}

pub fn logistic_classifier_predict(model LogisticClassifier[f64], x [][]f64, threshold f64) []int {
	proba := logistic_classifier_predict_proba(model, x)
	return proba.map(if it >= threshold { 1 } else { 0 })
}

pub fn logistic_classifier_predict_proba(model LogisticClassifier[f64], x [][]f64) []f64 {
	mut predictions := []f64{len: x.len}
	for i in 0 .. x.len {
		mut z := model.intercept
		for j in 0 .. model.coefficients.len {
			if j < x[i].len {
				// Normalize the feature using the training data statistics
				normalized_feature := (x[i][j] - model.feature_means[j]) / model.feature_stds[j]
				z += model.coefficients[j] * normalized_feature
			}
		}
		predictions[i] = sigmoid_f64(z)
	}
	return predictions
}

// ============================================================================
// Naive Bayes Classifier
// ============================================================================

pub fn naive_bayes_classifier(x [][]f64, y []int) NaiveBayesClassifier {
	assert x.len == y.len, "number of samples must match"
	assert x.len > 0, "must have at least one sample"
	
	mut class_priors := map[int]f64{}
	mut feature_means := map[int][][]f64{}
	mut feature_stds := map[int][][]f64{}
	mut classes := []int{}
	
	// Get unique classes
	for label in y {
		if label !in class_priors {
			classes << label
			class_priors[label] = 0.0
			feature_means[label] = [][]f64{}
			feature_stds[label] = [][]f64{}
		}
	}
	
	n := f64(x.len)
	num_features := x[0].len
	
	// Calculate global feature statistics for normalization
	mut global_means := []f64{len: num_features, init: 0.0}
	mut global_stds := []f64{len: num_features, init: 0.0}
	
	// Compute global means
	for j in 0 .. num_features {
		mut sum := 0.0
		for i in 0 .. x.len {
			sum += x[i][j]
		}
		global_means[j] = sum / n
	}
	
	// Compute global standard deviations
	for j in 0 .. num_features {
		mut sum_sq := 0.0
		for i in 0 .. x.len {
			diff := x[i][j] - global_means[j]
			sum_sq += diff * diff
		}
		global_stds[j] = math.sqrt(sum_sq / n)
		if global_stds[j] == 0.0 {
			global_stds[j] = 1.0
		}
	}
	
	// Normalize features for statistics calculation
	mut x_normalized := [][]f64{len: x.len}
	for i in 0 .. x.len {
		mut row := []f64{len: num_features}
		for j in 0 .. num_features {
			row[j] = (x[i][j] - global_means[j]) / global_stds[j]
		}
		x_normalized[i] = row
	}
	
	// Pre-calculate global feature variances for smoothing
	mut global_variances := []f64{len: num_features, init: 0.0}
	for j in 0 .. num_features {
		mut sum_sq := 0.0
		for i in 0 .. x_normalized.len {
			diff := x_normalized[i][j] - global_means[j]
			sum_sq += diff * diff
		}
		global_variances[j] = sum_sq / n
	}
	
	// Calculate class priors and feature statistics on normalized data
	for class in classes {
		mut class_samples := [][]f64{}
		for i in 0 .. x.len {
			if y[i] == class {
				class_samples << x_normalized[i]
			}
		}
		
		class_priors[class] = f64(class_samples.len) / n
		
		// Calculate mean and std for each feature in this class (on normalized data)
		for feature_idx in 0 .. num_features {
			mut feature_values := []f64{}
			for sample in class_samples {
				feature_values << sample[feature_idx]
			}
			
			mean := stats.mean(feature_values)
			variance := stats.variance(feature_values)
			mut std := math.sqrt(variance)
			
			// Variance smoothing: blend class and global variance for stability
			// This regularizes the estimates and improves generalization
			mut global_std := math.sqrt(global_variances[feature_idx])
			if global_std < 0.05 {
				global_std = 0.05
			}
			
			// Weighted average: 85% class variance, 15% global variance
			smoothing_weight := 0.15
			std = (1.0 - smoothing_weight) * std + smoothing_weight * global_std
			
			if std < 0.01 {
				std = 0.01
			}
			
			feature_means[class] << [mean]
			feature_stds[class] << [std]
		}
	}
	
	return NaiveBayesClassifier{
		class_priors: class_priors
		feature_means: feature_means
		feature_stds: feature_stds
		classes: classes
		trained: true
		global_means: global_means
		global_stds: global_stds
	}
}

pub fn naive_bayes_predict(model NaiveBayesClassifier, x [][]f64) []int {
	mut predictions := []int{len: x.len}
	
	for i in 0 .. x.len {
		mut max_prob := f64(-1e10)
		mut predicted_class := model.classes[0]
		
		// Normalize test features using global statistics
		mut x_normalized := []f64{len: x[i].len}
		for j in 0 .. x[i].len {
			x_normalized[j] = (x[i][j] - model.global_means[j]) / model.global_stds[j]
		}
		
		for class in model.classes {
			mut prob := math.log(model.class_priors[class])
			
			for feature_idx in 0 .. x_normalized.len {
				mean := model.feature_means[class][feature_idx][0]
				std := model.feature_stds[class][feature_idx][0]
				
				if std > 0 {
					numerator := -(x_normalized[feature_idx] - mean) * (x_normalized[feature_idx] - mean)
					denominator := 2 * std * std
					prob += numerator / denominator - math.log(std * math.sqrt(2 * math.pi))
				}
			}
			
			if prob > max_prob {
				max_prob = prob
				predicted_class = class
			}
		}
		
		predictions[i] = predicted_class
	}
	
	return predictions
}

// ============================================================================
// Support Vector Machine (SVM) Classifier
// ============================================================================

pub fn svm_classifier(x [][]f64, y []f64, learning_rate f64, iterations int, gamma f64, kernel string) SVMClassifier {
	assert x.len == y.len, "number of samples must match"
	assert x.len > 0, "must have at least one sample"
	
	n := x.len
	mut alphas := []f64{len: n, init: 0.0}
	mut bias := 0.0
	
	c := 1.0  // Regularization parameter
	
	// Simplified sequential minimal optimization (SMO)
	for _ in 0 .. iterations {
		for i in 0 .. n {
			// Compute margin
			mut margin := bias
			for j in 0 .. n {
				if alphas[j] > 0 {
					k := kernel_function(x[i], x[j], gamma, kernel)
					margin += alphas[j] * y[j] * k
				}
			}
			
			// Update alpha
			if y[i] * margin < 1.0 {
				alphas[i] += learning_rate
				if alphas[i] > c {
					alphas[i] = c
				}
			}
		}
		
		// Update bias
		mut bias_sum := 0.0
		mut count := 0
		for i in 0 .. n {
			if alphas[i] > 0 && alphas[i] < c {
				mut margin := 0.0
				for j in 0 .. n {
					if alphas[j] > 0 {
						k := kernel_function(x[i], x[j], gamma, kernel)
						margin += alphas[j] * y[j] * k
					}
				}
				bias_sum += y[i] - margin
				count++
			}
		}
		if count > 0 {
			bias = bias_sum / f64(count)
		}
	}
	
	// Store support vectors (indices where alphas > 0)
	mut support_vectors := [][]f64{}
	mut support_labels := []f64{}
	for i in 0 .. n {
		if alphas[i] > 0 {
			support_vectors << x[i]
			support_labels << y[i]
		}
	}
	
	return SVMClassifier{
		support_vectors: support_vectors
		support_labels: support_labels
		alphas: alphas
		bias: bias
		kernel_type: kernel
		gamma: gamma
		trained: true
	}
}

pub fn svm_predict(model SVMClassifier, x [][]f64) []int {
	mut predictions := []int{len: x.len}
	
	for i in 0 .. x.len {
		mut score := model.bias
		for j in 0 .. model.support_vectors.len {
			k := kernel_function(x[i], model.support_vectors[j], model.gamma, model.kernel_type)
			score += model.alphas[j] * model.support_labels[j] * k
		}
		predictions[i] = if score > 0 { 1 } else { 0 }
	}
	
	return predictions
}

fn kernel_function(x []f64, y []f64, gamma f64, kernel_type string) f64 {
	match kernel_type {
		"linear" {
			mut result := 0.0
			for i in 0 .. x.len {
				result += x[i] * y[i]
			}
			return result
		}
		"rbf" {
			dist := linalg.distance(x, y)
			return math.exp(-gamma * dist * dist)
		}
		"poly" {
			mut result := 0.0
			for i in 0 .. x.len {
				result += x[i] * y[i]
			}
			return math.pow(result + 1.0, 2.0)
		}
		else {
			// Default to linear
			mut result := 0.0
			for i in 0 .. x.len {
				result += x[i] * y[i]
			}
			return result
		}
	}
}

// ============================================================================
// Random Forest Classifier
// ============================================================================

pub fn random_forest_classifier(x [][]f64, y []int, num_trees int, max_depth int) RandomForestClassifier {
	assert x.len == y.len, "number of samples must match"
	assert x.len > 0, "must have at least one sample"
	
	num_features := x[0].len
	mut trees := []TreeNode{}
	
	// Bootstrap aggregating (bagging)
	for _ in 0 .. num_trees {
		// Create bootstrap sample
		mut boot_indices := []int{}
		for _ in 0 .. x.len {
			idx := int(f64(x.len) * rand_f64())
			boot_indices << idx
		}
		
		mut boot_x := [][]f64{}
		mut boot_y := []int{}
		for idx in boot_indices {
			boot_x << x[idx]
			boot_y << y[idx]
		}
		
		// Build decision tree
		tree := build_decision_tree(boot_x, boot_y, 0, max_depth, num_features)
		trees << tree
	}
	
	return RandomForestClassifier{
		trees: trees
		num_trees: num_trees
		num_features: num_features
		trained: true
	}
}

pub fn random_forest_predict(model RandomForestClassifier, x [][]f64) []int {
	mut predictions := []int{len: x.len}
	
	for i in 0 .. x.len {
		mut votes := map[int]int{}
		
		for tree in model.trees {
			pred := predict_tree(tree, x[i])
			if pred in votes {
				votes[pred]++
			} else {
				votes[pred] = 1
			}
		}
		
		mut best_class := 0
		mut best_count := 0
		for class, count in votes {
			if count > best_count {
				best_count = count
				best_class = class
			}
		}
		
		predictions[i] = best_class
	}
	
	return predictions
}

pub fn random_forest_classifier_predict(model RandomForestClassifier, x [][]f64) []int {
	return random_forest_predict(model, x)
}

pub fn random_forest_classifier_predict_proba(model RandomForestClassifier, x [][]f64) []f64 {
	mut probabilities := []f64{len: x.len}
	
	for i in 0 .. x.len {
		mut votes := map[int]int{}
		
		for tree in model.trees {
			pred := predict_tree(tree, x[i])
			if pred in votes {
				votes[pred]++
			} else {
				votes[pred] = 1
			}
		}
		
		// Calculate probability as fraction of votes for class 1
		class_1_votes := votes[1] or { 0 }
		probabilities[i] = f64(class_1_votes) / f64(model.num_trees)
	}
	
	return probabilities
}

fn build_decision_tree(x [][]f64, y []int, depth int, max_depth int, num_features int) TreeNode {
	// Check termination conditions
	if x.len == 0 || depth >= max_depth {
		// Majority class
		return TreeNode{
			feature: -1
			threshold: 0
			left: none
			right: none
			class_pred: get_majority_class(y)
		}
	}
	
	// Check if all labels are the same
	mut all_same := true
	first_label := y[0]
	for label in y {
		if label != first_label {
			all_same = false
			break
		}
	}
	
	if all_same {
		return TreeNode{
			feature: -1
			threshold: 0
			left: none
			right: none
			class_pred: first_label
		}
	}
	
	// Find best split
	mut best_gain := 0.0
	mut best_feature := 0
	mut best_threshold := 0.0
	mut best_left_idx := []int{}
	mut best_right_idx := []int{}
	
	// Try all features (essential for small datasets like Titanic)
	for feature in 0 .. num_features {
		// Get unique values
		mut values := []f64{}
		for sample in x {
			if sample[feature] !in values {
				values << sample[feature]
			}
		}
		if values.len < 2 {
			continue
		}
		values.sort()
		
		// Try all thresholds
		for i in 0 .. (values.len - 1) {
			threshold := (values[i] + values[i + 1]) / 2.0
			mut left_idx := []int{}
			mut right_idx := []int{}
			
			for j in 0 .. x.len {
				if x[j][feature] <= threshold {
					left_idx << j
				} else {
					right_idx << j
				}
			}
			
			if left_idx.len == 0 || right_idx.len == 0 {
				continue
			}
			
			// Calculate information gain
			mut left_labels := []int{}
			mut right_labels := []int{}
			for idx in left_idx {
				left_labels << y[idx]
			}
			for idx in right_idx {
				right_labels << y[idx]
			}
			
			gain := calculate_information_gain(y, left_labels, right_labels)
			
			if gain > best_gain {
				best_gain = gain
				best_feature = feature
				best_threshold = threshold
				best_left_idx = left_idx.clone()
				best_right_idx = right_idx.clone()
			}
		}
	}
	
	// If no good split found, return leaf
	if best_gain == 0 {
		return TreeNode{
			feature: -1
			threshold: 0
			left: none
			right: none
			class_pred: get_majority_class(y)
		}
	}
	
	// Build subtrees
	mut left_x := [][]f64{}
	mut left_y := []int{}
	for idx in best_left_idx {
		left_x << x[idx]
		left_y << y[idx]
	}
	
	mut right_x := [][]f64{}
	mut right_y := []int{}
	for idx in best_right_idx {
		right_x << x[idx]
		right_y << y[idx]
	}
	
	left_tree := build_decision_tree(left_x, left_y, depth + 1, max_depth, num_features)
	right_tree := build_decision_tree(right_x, right_y, depth + 1, max_depth, num_features)
	
	return TreeNode{
		feature: best_feature
		threshold: best_threshold
		left: &left_tree
		right: &right_tree
		class_pred: none
	}
}

fn predict_tree(tree TreeNode, x []f64) int {
	if class_pred := tree.class_pred {
		return class_pred
	}
	
	if x[tree.feature] <= tree.threshold {
		if left := tree.left {
			return predict_tree(left, x)
		}
	} else {
		if right := tree.right {
			return predict_tree(right, x)
		}
	}
	
	return 0
}

fn get_majority_class(labels []int) int {
	mut class_counts := map[int]int{}
	for label in labels {
		if label in class_counts {
			class_counts[label]++
		} else {
			class_counts[label] = 1
		}
	}
	
	mut majority := 0
	mut max_count := 0
	for class, count in class_counts {
		if count > max_count {
			max_count = count
			majority = class
		}
	}
	
	return majority
}

fn calculate_information_gain(parent []int, left []int, right []int) f64 {
	if parent.len == 0 {
		return 0
	}
	
	parent_entropy := entropy(parent)
	
	left_weight := f64(left.len) / f64(parent.len)
	right_weight := f64(right.len) / f64(parent.len)
	
	weighted_child_entropy := left_weight * entropy(left) + right_weight * entropy(right)
	
	return parent_entropy - weighted_child_entropy
}

fn entropy(labels []int) f64 {
	if labels.len == 0 {
		return 0
	}
	
	mut counts := map[int]int{}
	for label in labels {
		if label in counts {
			counts[label]++
		} else {
			counts[label] = 1
		}
	}
	
	mut ent := 0.0
	for _, count in counts {
		p := f64(count) / f64(labels.len)
		if p > 0 {
			ent -= p * math.log2(p)
		}
	}
	
	return ent
}

// ============================================================================
// Classification Metrics
// ============================================================================

pub fn accuracy(y_true []int, y_pred []int) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	if y_true.len == 0 {
		return 0
	}
	
	mut correct := 0
	for i in 0 .. y_true.len {
		if y_true[i] == y_pred[i] {
			correct++
		}
	}
	
	return f64(correct) / f64(y_true.len)
}

pub fn precision(y_true []int, y_pred []int, positive_class int) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut tp := 0
	mut fp := 0
	
	for i in 0 .. y_true.len {
		if y_pred[i] == positive_class {
			if y_true[i] == positive_class {
				tp++
			} else {
				fp++
			}
		}
	}
	
	if tp + fp == 0 {
		return 0
	}
	
	return f64(tp) / f64(tp + fp)
}

pub fn recall(y_true []int, y_pred []int, positive_class int) f64 {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	mut tp := 0
	mut fn_val := 0
	
	for i in 0 .. y_true.len {
		if y_true[i] == positive_class {
			if y_pred[i] == positive_class {
				tp++
			} else {
				fn_val++
			}
		}
	}
	
	if tp + fn_val == 0 {
		return 0
	}
	
	return f64(tp) / f64(tp + fn_val)
}

pub fn f1_score(y_true []int, y_pred []int, positive_class int) f64 {
	p := precision(y_true, y_pred, positive_class)
	r := recall(y_true, y_pred, positive_class)
	
	if p + r == 0 {
		return 0
	}
	
	return 2 * (p * r) / (p + r)
}

pub fn confusion_matrix(y_true []int, y_pred []int) [][]int {
	assert y_true.len == y_pred.len, "arrays must have same length"
	
	// Find unique classes
	mut classes := []int{}
	for label in y_true {
		if label !in classes {
			classes << label
		}
	}
	for label in y_pred {
		if label !in classes {
			classes << label
		}
	}
	classes.sort()
	
	// Initialize matrix
	mut matrix := [][]int{len: classes.len}
	for i in 0 .. classes.len {
		matrix[i] = []int{len: classes.len, init: 0}
	}
	
	// Fill matrix
	for i in 0 .. y_true.len {
		true_idx := get_index(classes, y_true[i])
		pred_idx := get_index(classes, y_pred[i])
		matrix[true_idx][pred_idx]++
	}
	
	return matrix
}

fn get_index(arr []int, val int) int {
	for i, v in arr {
		if v == val {
			return i
		}
	}
	return 0
}

// ============================================================================
// PyCaret-like Setup Function
// ============================================================================

pub struct ClassificationSetup {
pub mut:
	estimator   string
	train_data  [][]f64
	test_data   [][]f64
	target      []int
	target_test []int
	preprocessing bool
}

pub fn setup(x [][]f64, y []int, test_size f64, estimator string) ClassificationSetup {
	// Split data
	split_idx := int(f64(x.len) * (1.0 - test_size))
	
	train_x := x[0..split_idx]
	train_y := y[0..split_idx]
	test_x := x[split_idx..x.len]
	test_y := y[split_idx..y.len]
	
	return ClassificationSetup{
		estimator: estimator
		train_data: train_x
		test_data: test_x
		target: train_y
		target_test: test_y
		preprocessing: true
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

// Sigmoid is defined in regression.v, reuse it
// For classification module, we use the generic version from regression
fn sigmoid_f64(x f64) f64 {
	return sigmoid(x)
}

fn rand_f64() f64 {
	return rand.f64()
}
