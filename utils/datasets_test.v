module utils

fn test__load_iris() {
	iris := load_iris() or {
		panic("Failed to load iris dataset: ${err}")
	}
	
	assert iris.name == "Iris", "dataset name should be Iris"
	assert iris.features.len == 150, "Iris should have 150 samples"
	assert iris.features[0].len == 4, "Iris should have 4 features"
	assert iris.target.len == 150, "Iris should have 150 labels"
	assert iris.feature_names.len == 4, "Iris should have 4 feature names"
}

fn test__load_wine() {
	wine := load_wine() or {
		panic("Failed to load wine dataset: ${err}")
	}
	
	assert wine.name == "Wine", "dataset name should be Wine"
	assert wine.features.len == 32, "Wine should have 32 samples"
	assert wine.target.len == wine.features.len, "Wine should have matching target"
	assert wine.feature_names.len == 4, "Wine should have 4 feature names"
}

fn test__load_boston_housing() {
	boston := load_boston_housing() or {
		panic("Failed to load boston housing dataset: ${err}")
	}
	
	assert boston.name == "Boston Housing", "dataset name should be Boston Housing"
	assert boston.features.len == 20, "Boston should have 20 samples"
	assert boston.target.len == boston.features.len, "Boston should have matching target"
	assert boston.features[0].len == 3, "Boston should have 3 features"
}

fn test__load_linear_regression() {
	linreg := load_linear_regression()
	
	assert linreg.name == "Linear Regression", "dataset name should be Linear Regression"
	assert linreg.features.len == 20, "Linear regression should have 20 samples"
	assert linreg.target.len == linreg.features.len, "Linear regression should have matching target"
	assert linreg.features[0].len == 2, "Linear regression should have 2 features"
}

fn test__dataset_summary() {
	iris := load_iris() or {
		panic("Failed to load iris")
	}
	
	summary := iris.summary()
	
	assert summary.contains("Iris"), "summary should contain dataset name"
	assert summary.contains("150"), "summary should contain sample count"
	assert summary.contains("Sepal Length"), "summary should contain feature names"
}

fn test__dataset_train_test_split() {
	iris := load_iris() or {
		panic("Failed to load iris")
	}
	
	train, test := iris.train_test_split(0.2)
	
	assert train.features.len + test.features.len == iris.features.len, "split should preserve all samples"
	assert train.target.len == train.features.len, "train should have matching target"
	assert test.target.len == test.features.len, "test should have matching target"
	assert train.name.contains("Train"), "train dataset name should contain Train"
	assert test.name.contains("Test"), "test dataset name should contain Test"
}

fn test__dataset_xy() {
	iris := load_iris() or {
		panic("Failed to load iris")
	}
	
	x, y := iris.xy()
	
	assert x.len == iris.features.len, "x should match features length"
	assert y.len == iris.target.len, "y should match target length"
}

fn test__regression_dataset_summary() {
	boston := load_boston_housing() or {
		panic("Failed to load boston")
	}
	
	summary := boston.summary()
	
	assert summary.contains("Boston Housing"), "summary should contain dataset name"
	assert summary.contains("20"), "summary should contain sample count"
	assert summary.contains("Mean"), "summary should contain statistics"
}

fn test__regression_dataset_train_test_split() {
	boston := load_boston_housing() or {
		panic("Failed to load boston")
	}
	
	train, test := boston.train_test_split(0.2)
	
	assert train.features.len + test.features.len == boston.features.len, "split should preserve all samples"
	assert train.target.len == train.features.len, "train should have matching target"
	assert test.target.len == test.features.len, "test should have matching target"
}

fn test__regression_dataset_xy() {
	boston := load_boston_housing() or {
		panic("Failed to load boston")
	}
	
	x, y := boston.xy()
	
	assert x.len == boston.features.len, "x should match features length"
	assert y.len == boston.target.len, "y should match target length"
}

fn test__iris_data_quality() {
	iris := load_iris() or {
		panic("Failed to load iris")
	}
	
	// Check that features are reasonable
	for row in iris.features {
		for val in row {
			assert val > 0, "feature values should be positive"
			assert val < 100, "feature values should be reasonable (< 100)"
		}
	}
	
	// Check target classes
	mut classes := []int{}
	for label in iris.target {
		if label !in classes {
			classes << label
		}
	}
	classes.sort()
	
	assert classes.len == 3, "iris should have 3 classes"
	assert classes[0] == 0 && classes[1] == 1 && classes[2] == 2, "iris classes should be 0, 1, 2"
}
