module utils

import math

fn test__factorial() {
	assert factorial(0) == 1.0
	assert factorial(1) == 1.0
	assert factorial(5) == 120.0
	assert factorial(10) == 3628800.0
}

fn test__combinations() {
	assert combinations(5, 0) == 1.0
	assert combinations(5, 5) == 1.0
	assert combinations(5, 2) == 10.0
	assert combinations(5, 3) == 10.0
	assert combinations(10, 5) == 252.0
}

fn test__range() {
	assert range(0) == []
	assert range(5) == [0, 1, 2, 3, 4]
	assert range(1) == [0]
}

// Tests for normalize_features
fn test__normalize_features_basic() {
	// Create simple test data: 3 samples, 2 features
	x := [
		[1.0, 2.0],
		[2.0, 4.0],
		[3.0, 6.0],
	]
	
	x_norm, means, stds := normalize_features(x)
	
	// Check dimensions
	assert x_norm.len == 3
	assert x_norm[0].len == 2
	assert means.len == 2
	assert stds.len == 2
	
	// Check mean values (should be [2.0, 4.0])
	assert means[0] > 1.99 && means[0] < 2.01
	assert means[1] > 3.99 && means[1] < 4.01
	
	// Check std values are positive
	assert stds[0] > 0
	assert stds[1] > 0
	
	// Normalized data should have mean ~0 and std ~1
	mut norm_mean_0 := 0.0
	mut norm_mean_1 := 0.0
	for i in 0..x_norm.len {
		norm_mean_0 += x_norm[i][0]
		norm_mean_1 += x_norm[i][1]
	}
	norm_mean_0 /= f64(x_norm.len)
	norm_mean_1 /= f64(x_norm.len)
	
	assert norm_mean_0 > -0.01 && norm_mean_0 < 0.01
	assert norm_mean_1 > -0.01 && norm_mean_1 < 0.01
}

fn test__normalize_features_constant_feature() {
	// Test with one constant feature (std = 0)
	x := [
		[5.0, 1.0],
		[5.0, 2.0],
		[5.0, 3.0],
	]
	
	x_norm, means, stds := normalize_features(x)
	
	// Constant feature should have std = 0
	assert stds[0] == 0.0
	assert stds[1] > 0.0
	
	// Constant feature should remain unchanged (division by 1.0 when std=0)
	for i in 0..x_norm.len {
		assert x_norm[i][0] == 0.0 // (5.0 - 5.0) / 1.0 = 0
	}
}

fn test__normalize_features_two_samples() {
	// Edge case: two samples (minimum for variance calculation)
	x := [
		[1.0, 2.0, 3.0],
		[3.0, 4.0, 5.0],
	]
	
	x_norm, means, stds := normalize_features(x)
	
	assert x_norm.len == 2
	// Means should be [2.0, 3.0, 4.0]
	assert means[0] == 2.0
	assert means[1] == 3.0
	assert means[2] == 4.0
	
	// Standard deviations should be positive
	for std in stds {
		assert std > 0.0
	}
}

// Tests for apply_normalization
fn test__apply_normalization_basic() {
	// Train data
	x_train := [
		[1.0, 2.0],
		[2.0, 4.0],
		[3.0, 6.0],
	]
	
	// Test data
	x_test := [
		[2.0, 4.0],
		[1.5, 3.0],
	]
	
	// Get statistics from training data
	_, train_means, train_stds := normalize_features(x_train)
	
	// Apply to test data
	x_test_norm := apply_normalization(x_test, train_means, train_stds)
	
	// Verify dimensions
	assert x_test_norm.len == 2
	assert x_test_norm[0].len == 2
	
	// Verify normalization using expected values
	// First test sample: [2.0, 4.0]
	// Should be: (2.0 - 2.0) / std[0], (4.0 - 4.0) / std[1] = [0, 0]
	assert x_test_norm[0][0] > -0.01 && x_test_norm[0][0] < 0.01
	assert x_test_norm[0][1] > -0.01 && x_test_norm[0][1] < 0.01
}

fn test__apply_normalization_consistency() {
	// Test that apply_normalization correctly uses precomputed stats
	x_train := [
		[10.0, 20.0, 30.0],
		[20.0, 30.0, 40.0],
		[30.0, 40.0, 50.0],
	]
	
	x_test := [
		[15.0, 25.0, 35.0],
		[25.0, 35.0, 45.0],
	]
	
	// Get statistics from training
	x_train_norm, means, stds := normalize_features(x_train)
	
	// Apply to test set
	x_test_norm := apply_normalization(x_test, means, stds)
	
	// Verify test data is normalized correctly
	// Sample 1 is [15.0, 25.0, 35.0]
	// Expected: [(15.0 - 20.0) / std[0], (25.0 - 30.0) / std[1], (35.0 - 40.0) / std[2]]
	for j in 0..3 {
		expected := (x_test[0][j] - means[j]) / (if stds[j] > 0 { stds[j] } else { 1.0 })
		diff := math.abs(x_test_norm[0][j] - expected)
		assert diff < 0.0001, "apply_normalization should use training statistics correctly"
	}
}

fn test__apply_normalization_with_zero_std() {
	x_train := [
		[5.0, 1.0],
		[5.0, 2.0],
		[5.0, 3.0],
	]
	
	x_test := [
		[5.0, 2.5],
		[5.0, 1.5],
	]
	
	_, means, stds := normalize_features(x_train)
	x_test_norm := apply_normalization(x_test, means, stds)
	
	// First feature is constant (std = 0), should normalize to 0
	for i in 0..x_test_norm.len {
		assert x_test_norm[i][0] == 0.0
	}
}

fn test__normalize_and_apply_roundtrip() {
	// Test data with known properties
	x_train := [
		[0.0, 10.0],
		[1.0, 20.0],
		[2.0, 30.0],
	]
	
	x_test := [
		[1.0, 20.0],
		[0.5, 15.0],
	]
	
	// Normalize training data
	x_train_norm, means, stds := normalize_features(x_train)
	
	// Apply to test data
	x_test_norm := apply_normalization(x_test, means, stds)
	
	// Verify training means are approximately 0
	mut train_mean_0 := 0.0
	mut train_mean_1 := 0.0
	for i in 0..x_train_norm.len {
		train_mean_0 += x_train_norm[i][0]
		train_mean_1 += x_train_norm[i][1]
	}
	train_mean_0 /= f64(x_train_norm.len)
	train_mean_1 /= f64(x_train_norm.len)
	
	assert train_mean_0 > -0.01 && train_mean_0 < 0.01
	assert train_mean_1 > -0.01 && train_mean_1 < 0.01
	
	// Verify test data is normalized using training statistics
	assert x_test_norm.len == 2
	assert x_test_norm[0].len == 2
}

fn test__normalize_features_dimensions() {
	// Test with different dimensions
	// 5 samples, 4 features
	x := [
		[1.0, 2.0, 3.0, 4.0],
		[2.0, 3.0, 4.0, 5.0],
		[3.0, 4.0, 5.0, 6.0],
		[4.0, 5.0, 6.0, 7.0],
		[5.0, 6.0, 7.0, 8.0],
	]
	
	x_norm, means, stds := normalize_features(x)
	
	assert x_norm.len == 5
	assert x_norm[0].len == 4
	assert means.len == 4
	assert stds.len == 4
	
	// All standard deviations should be positive
	for std in stds {
		assert std >= 0.0
	}
}
