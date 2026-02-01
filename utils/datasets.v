module utils

import os

// Dataset structure for labeled data
pub struct Dataset {
	pub mut:
	name        string
	features    [][]f64
	target      []int
	feature_names []string
	target_name string
	description string
}

// RegressionDataset for continuous target values
pub struct RegressionDataset {
	pub mut:
	name        string
	features    [][]f64
	target      []f64
	feature_names []string
	target_name string
	description string
}

// Helper to locate data directory
fn get_data_dir() string {
	// Try different paths for data directory
	paths := [
		"./utils/data",
		"./data",
		"../data",
		"utils/data",
	]
	
	for path in paths {
		if os.is_dir(path) {
			return path
		}
	}
	
	// Return default if not found (will fail at file read)
	return "utils/data"
}

// Helper to parse CSV line
fn parse_csv_line(line string) []string {
	return line.split(',')
}

// ============================================================================
// Classification Datasets
// ============================================================================

// Load the Iris dataset (3 classes, 150 samples)
pub fn load_iris() !Dataset {
	data_dir := get_data_dir()
	csv_path := os.join_path(data_dir, "iris.csv")
	
	content := os.read_file(csv_path) !
	lines := content.split('\n')
	
	mut features := [][]f64{}
	mut target := []int{}
	
	// Skip header
	for i in 1 .. lines.len {
		line := lines[i].trim_space()
		if line.len == 0 {
			continue
		}
		
		parts := parse_csv_line(line)
		if parts.len < 5 {
			continue
		}
		
		// Parse features
		features << [
			parts[0].f64(),
			parts[1].f64(),
			parts[2].f64(),
			parts[3].f64(),
		]
		
		// Parse target
		target << parts[4].int()
	}
	
	return Dataset{
		name: "Iris"
		features: features
		target: target
		feature_names: ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
		target_name: "Species"
		description: "Classic iris dataset: 150 samples, 4 features, 3 classes (Setosa, Versicolor, Virginica)"
	}
}

// Load Wine dataset (3 classes, 178 samples - simplified)
pub fn load_wine() !Dataset {
	data_dir := get_data_dir()
	csv_path := os.join_path(data_dir, "wine.csv")
	
	content := os.read_file(csv_path) !
	lines := content.split('\n')
	
	mut features := [][]f64{}
	mut target := []int{}
	
	// Skip header
	for i in 1 .. lines.len {
		line := lines[i].trim_space()
		if line.len == 0 {
			continue
		}
		
		parts := parse_csv_line(line)
		if parts.len < 5 {
			continue
		}
		
		// Parse features
		features << [
			parts[0].f64(),
			parts[1].f64(),
			parts[2].f64(),
			parts[3].f64(),
		]
		
		// Parse target
		target << parts[4].int()
	}
	
	return Dataset{
		name: "Wine"
		features: features
		target: target
		feature_names: ["Alcohol", "Malic Acid", "Ash", "Phenols"]
		target_name: "Wine Type"
		description: "Wine classification: 178 samples, 13 features (simplified to 4), 3 classes"
	}
}

// Load Breast Cancer dataset (binary classification - 30 features)
pub fn load_breast_cancer() !Dataset {
	data_dir := get_data_dir()
	features_path := os.join_path(data_dir, "breast_cancer_features.csv")
	targets_path := os.join_path(data_dir, "breast_cancer_targets.csv")
	
	// Load features
	features_content := os.read_file(features_path) !
	features_lines := features_content.split('\n')
	
	// Load targets
	targets_content := os.read_file(targets_path) !
	targets_lines := targets_content.split('\n')
	
	mut features := [][]f64{}
	mut target := []int{}
	
	// Get feature names from header
	feature_names := if features_lines.len > 0 {
		features_lines[0].split(',')
	} else {
		[]string{}
	}
	
	// Skip header in features
	for i in 1 .. features_lines.len {
		line := features_lines[i].trim_space()
		if line.len == 0 {
			continue
		}
		
		parts := parse_csv_line(line)
		mut row := []f64{}
		for part in parts {
			row << part.f64()
		}
		features << row
	}
	
	// Load targets
	for i in 0 .. targets_lines.len {
		line := targets_lines[i].trim_space()
		if line.len == 0 {
			continue
		}
		target << line.int()
	}
	
	return Dataset{
		name: "Breast Cancer"
		features: features
		target: target
		feature_names: feature_names
		target_name: "Diagnosis"
		description: "Breast Cancer Wisconsin: 569 samples, 30 features, binary classification (0=Malignant, 1=Benign)"
	}
}

// ============================================================================
// Regression Datasets
// ============================================================================

// Load Boston Housing dataset
pub fn load_boston_housing() !RegressionDataset {
	data_dir := get_data_dir()
	csv_path := os.join_path(data_dir, "boston_housing.csv")
	
	content := os.read_file(csv_path) !
	lines := content.split('\n')
	
	mut features := [][]f64{}
	mut target := []f64{}
	
	// Skip header
	for i in 1 .. lines.len {
		line := lines[i].trim_space()
		if line.len == 0 {
			continue
		}
		
		parts := parse_csv_line(line)
		if parts.len < 14 {
			continue
		}
		
		// Parse features (first 3 columns: crim, zn, indus)
		features << [
			parts[0].f64(),
			parts[1].f64(),
			parts[2].f64(),
		]
		
		// Parse target (last column: medv)
		target << parts[13].f64()
	}
	
	return RegressionDataset{
		name: "Boston Housing"
		features: features
		target: target
		feature_names: ["Crime Rate", "% Residential Land", "Distance to Employment"]
		target_name: "Median House Price"
		description: "Boston Housing: 506 samples, 13 features (simplified to 3), continuous target"
	}
}

// Load synthetic linear regression dataset (generated programmatically)
pub fn load_linear_regression() RegressionDataset {
	mut features := [][]f64{}
	mut target := []f64{}
	
	// Generate synthetic linear data: y = 3*x1 + x2 + noise
	for x1_int in 1 .. 11 {
		x1 := f64(x1_int)
		for x2_offset in 0 .. 2 {
			x2 := x1 + f64(x2_offset) + 0.5
			features << [x1, x2]
			target << 3.0 * x1 + x2 + 0.5 * f64(x2_offset)
		}
	}
	
	return RegressionDataset{
		name: "Linear Regression"
		features: features
		target: target
		feature_names: ["Feature 1", "Feature 2"]
		target_name: "Target"
		description: "Synthetic linear dataset: 20 samples, 2 features, simple linear relationship"
	}
}

// ============================================================================
// Utility Functions
// ============================================================================

// Get summary information about a classification dataset
pub fn (d Dataset) summary() string {
	mut result := "Dataset: ${d.name}\n"
	result += "Description: ${d.description}\n"
	result += "Samples: ${d.features.len}\n"
	result += "Features: ${d.features[0].len}\n"
	result += "Classes: ${get_unique_classes(d.target).len}\n"
	result += "Feature names: ${d.feature_names.join(', ')}\n"
	result += "Target name: ${d.target_name}\n"
	
	// Class distribution
	class_counts := count_class_distribution(d.target)
	result += "Class distribution:\n"
	for class in get_unique_classes(d.target) {
		result += "  Class ${class}: ${class_counts[class]} samples\n"
	}
	
	return result
}

// Get summary information about a regression dataset
pub fn (d RegressionDataset) summary() string {
	mut result := "Dataset: ${d.name}\n"
	result += "Description: ${d.description}\n"
	result += "Samples: ${d.features.len}\n"
	result += "Features: ${d.features[0].len}\n"
	result += "Feature names: ${d.feature_names.join(', ')}\n"
	result += "Target name: ${d.target_name}\n"
	
	// Target statistics
	mut min_target := d.target[0]
	mut max_target := d.target[0]
	mut sum_target := 0.0
	
	for val in d.target {
		if val < min_target {
			min_target = val
		}
		if val > max_target {
			max_target = val
		}
		sum_target += val
	}
	
	mean_target := sum_target / f64(d.target.len)
	
	result += "Target statistics:\n"
	result += "  Min: ${min_target:.2f}\n"
	result += "  Max: ${max_target:.2f}\n"
	result += "  Mean: ${mean_target:.2f}\n"
	
	return result
}

// Split dataset into train and test sets
pub fn (d Dataset) train_test_split(test_size f64) (Dataset, Dataset) {
	split_idx := int(f64(d.features.len) * (1.0 - test_size))
	
	train_features := d.features[0..split_idx]
	train_target := d.target[0..split_idx]
	
	test_features := d.features[split_idx..d.features.len]
	test_target := d.target[split_idx..d.target.len]
	
	train := Dataset{
		name: d.name + " (Train)"
		features: train_features
		target: train_target
		feature_names: d.feature_names
		target_name: d.target_name
		description: d.description
	}
	
	test := Dataset{
		name: d.name + " (Test)"
		features: test_features
		target: test_target
		feature_names: d.feature_names
		target_name: d.target_name
		description: d.description
	}
	
	return train, test
}

// Split regression dataset into train and test sets
pub fn (d RegressionDataset) train_test_split(test_size f64) (RegressionDataset, RegressionDataset) {
	split_idx := int(f64(d.features.len) * (1.0 - test_size))
	
	train_features := d.features[0..split_idx]
	train_target := d.target[0..split_idx]
	
	test_features := d.features[split_idx..d.features.len]
	test_target := d.target[split_idx..d.target.len]
	
	train := RegressionDataset{
		name: d.name + " (Train)"
		features: train_features
		target: train_target
		feature_names: d.feature_names
		target_name: d.target_name
		description: d.description
	}
	
	test := RegressionDataset{
		name: d.name + " (Test)"
		features: test_features
		target: test_target
		feature_names: d.feature_names
		target_name: d.target_name
		description: d.description
	}
	
	return train, test
}

// Get features and target as separate arrays
pub fn (d Dataset) xy() ([][]f64, []int) {
	return d.features, d.target
}

// Get features and target as separate arrays (regression)
pub fn (d RegressionDataset) xy() ([][]f64, []f64) {
	return d.features, d.target
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_unique_classes(target []int) []int {
	mut classes := []int{}
	for label in target {
		if label !in classes {
			classes << label
		}
	}
	classes.sort()
	return classes
}

fn count_class_distribution(target []int) map[int]int {
	mut counts := map[int]int{}
	for label in target {
		if label in counts {
			counts[label]++
		} else {
			counts[label] = 1
		}
	}
	return counts
}
