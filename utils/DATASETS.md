# Datasets Module

The datasets module provides convenient access to sample datasets for classification and regression tasks, similar to PyCaret and scikit-learn. All datasets are stored as external CSV files for modularity and ease of updates.

## Data Location

Dataset CSV files are stored in `utils/data/`:
- `iris.csv` - Iris flower dataset
- `wine.csv` - Wine classification dataset
- `boston_housing.csv` - Boston Housing regression dataset

## Classification Datasets

### Iris Dataset

Classic dataset with three iris flower species (Setosa, Versicolor, Virginica).

**Properties:**
- Samples: 150
- Features: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- Classes: 3
- Type: Multi-class classification

**Usage:**

```v
iris := load_iris()!

// Get summary
println(iris.summary())

// Split into train/test
train, test := iris.train_test_split(0.2)

// Extract features and targets
x, y := iris.xy()
```

### Wine Dataset

Wine classification based on chemical properties.

**Properties:**
- Samples: 32 (simplified subset)
- Features: 4 (Alcohol, Malic Acid, Ash, Phenols)
- Classes: 3
- Type: Multi-class classification

**Usage:**

```v
wine := load_wine()!

train, test := wine.train_test_split(0.3)
predictions := ml.random_forest_classifier(train.features, train.target, 5, 3)
```

### Breast Cancer Dataset

Binary classification for cancer diagnosis (placeholder).

**Properties:**
- Status: Placeholder (empty by default)
- Samples: 569 (when complete)
- Features: 30 (simplified to 4)
- Classes: 2
- Type: Binary classification

## Regression Datasets

### Boston Housing Dataset

House price prediction based on neighborhood features.

**Properties:**
- Samples: 20 (simplified subset)
- Features: 13 (simplified to 3: Crime Rate, % Residential Land, Distance to Employment)
- Target: Median house price
- Type: Continuous regression

**Usage:**

```v
boston := load_boston_housing()!

train, test := boston.train_test_split(0.25)

x_train, y_train := train.xy()
model := ml.linear_regression(x_train, y_train)
```

### Linear Regression Dataset

Synthetic linear dataset generated programmatically.

**Properties:**
- Samples: 20
- Features: 2
- Target: Linear combination with noise
- Relationship: `y = 3*x1 + x2 + noise`
- Type: Continuous regression

**Usage:**

```v
linreg := load_linear_regression()

// No error handling needed - generated, not loaded from file
train, test := linreg.train_test_split(0.2)

x, y := linreg.xy()
```

## API Reference

### Classification Dataset Operations

#### Loading

```v
pub fn load_iris() !Dataset
pub fn load_wine() !Dataset
pub fn load_breast_cancer() !Dataset
```

All return errors if CSV files cannot be found or parsed.

#### Methods

**Summary Information:**
```v
pub fn (d Dataset) summary() string
```

Provides:
- Dataset name and description
- Number of samples and features
- Number of classes
- Feature names
- Class distribution

**Train/Test Split:**
```v
pub fn (d Dataset) train_test_split(test_size f64) (Dataset, Dataset)
```

Splits data into training and test sets. `test_size` should be between 0.0 and 1.0.

**Extract Data:**
```v
pub fn (d Dataset) xy() ([][]f64, []int)
```

Returns features and targets as separate arrays.

### Regression Dataset Operations

#### Loading

```v
pub fn load_boston_housing() !RegressionDataset
pub fn load_linear_regression() RegressionDataset
```

Only Boston Housing requires error handling (file-based). Linear Regression is generated programmatically.

#### Methods

**Summary Information:**
```v
pub fn (d RegressionDataset) summary() string
```

Provides:
- Dataset name and description
- Number of samples and features
- Feature names
- Target statistics (min, max, mean)

**Train/Test Split:**
```v
pub fn (d RegressionDataset) train_test_split(test_size f64) (RegressionDataset, RegressionDataset)
```

**Extract Data:**
```v
pub fn (d RegressionDataset) xy() ([][]f64, []f64)
```

## Example Workflow

```v
module main

import utils
import ml

fn main() {
    // Load classification dataset
    iris := utils.load_iris()!
    
    // View dataset information
    println(iris.summary())
    
    // Split data
    train, test := iris.train_test_split(0.2)
    
    // Train model
    model := ml.random_forest_classifier(train.features, train.target, 10, 5)
    
    // Make predictions
    predictions := ml.random_forest_predict(model, test.features)
    
    // Evaluate
    acc := ml.accuracy(test.target, predictions)
    println("Accuracy: ${acc}")
}
```

## CSV File Format

### Classification Datasets

Header row with comma-separated columns:

```
feature1,feature2,feature3,feature4,target
value,value,value,value,class_label
...
```

Example (Iris):
```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
...
```

### Regression Datasets

Header row with all feature columns followed by target:

```
feature1,feature2,feature3,...,target
value,value,value,...,target_value
...
```

Example (Boston Housing):
```
crim,zn,indus,...,medv
0.00632,18.0,2.31,...,24.0
0.02731,0.0,7.07,...,21.6
...
```

## Data Directory Resolution

The module automatically searches for the `data/` directory in these locations (in order):

1. `./utils/data`
2. `./data`
3. `../data`
4. `utils/data`

This allows the module to work from different working directories.

## Extending with Custom Datasets

To add a new dataset:

1. Create a CSV file in `utils/data/your_dataset.csv`
2. Add a loading function in `datasets.v`:

```v
pub fn load_your_dataset() !YourDataset {
    data_dir := get_data_dir()
    csv_path := os.join_path(data_dir, "your_dataset.csv")
    
    content := os.read_file(csv_path) !
    lines := content.split('\n')
    
    // Parse CSV...
    
    return Dataset{
        name: "Your Dataset"
        features: features
        target: target
        feature_names: [...]
        target_name: "Target"
        description: "..."
    }
}
```

3. Add comprehensive tests in `datasets_test.v`

## Notes

- All external data is in the `utils/data/` directory as CSV files
- Features and targets are loaded as `f64` (float64) for features and `int` for classification targets
- Regression targets use `f64`
- Dataset classes are 0-indexed integers
- The module follows zero-dependency design - no external file format libraries needed
