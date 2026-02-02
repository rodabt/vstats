# VStats 0.1.0

A dependency-free Linear Algebra, Statistics, and Machine Learning library written from scratch in V.

## Installation

```bash
v install https://github.com/rodabt/vstats
```

## Quick Start

```v
import vstats.stats
import vstats.utils
import vstats.linalg

// Statistics with generic types (int or f64)
mean_val := stats.mean([1, 2, 3, 4, 5])  // Works with int or f64
variance := stats.variance([1.0, 2.0, 3.0])

// Advanced statistics
f_stat, p_val := stats.anova_one_way([group1, group2, group3])
lower, upper := stats.confidence_interval_mean(data, 0.95)

// Metrics
cm := utils.build_confusion_matrix(y_true, y_pred)
println("F1: ${cm.f1_score():.4f}")

// Linear Algebra (also supports generics)
v1 := [1, 2, 3]
v2 := [4, 5, 6]
v_sum := linalg.add(v1, v2)  // Works with int or f64
result := linalg.matmul(matrix_a, matrix_b)
distance := linalg.distance(vector_a, vector_b)
```

## Features Overview

| Module         | Purpose                           | Status      |
| -------------- | --------------------------------- | ----------- |
| **linalg**     | Vector & Matrix operations        | ✓ Complete |
| **stats**      | Descriptive & Advanced Statistics | ✓ Complete |
| **prob**       | Probability Distributions         | ✓ Complete |
| **optim**      | Numerical Optimization            | ✓ Complete |
| **utils**      | Utilities, Metrics, Datasets      | ✓ Complete |
| **ml**         | Machine Learning Algorithms       | ✓ Complete |
| **nn**         | Neural Networks                   | ✓ Complete |
| **hypothesis** | Hypothesis Testing                | ✓ Complete |
| **symbol**     | Symbolic Computation              | 🚧 WIP     |

## Generic Type Support

Many functions across VStats support **generic numeric types** (`int`, `f64`):

- **Linear Algebra (linalg)**: All vector and matrix operations
  - Examples: `add[T]`, `subtract[T]`, `dot[T]`, `matmul[T]`, `distance[T]`
  - Vector operations return the same type: `add[int]` returns `[]int`
  - Matrix operations work seamlessly with both types
  
- **Statistics & Optimization**: Accept generic input, return `f64` for precision
  - Examples: `mean[T]`, `variance[T]`, `correlation[T]`, `mse_loss[T]`
  - Design: Generic input `[]T` → F64 output (maintains mathematical precision)
  
- **Type-Specific Functions**: Require `[]f64` due to algorithmic constraints
  - `median`, `quantile`, `mode` - require sorting/hashing
  - Functions depending on `[][]f64` matrices for feature operations

## Modules implemented

The following is the list of modules and functions implemented

### Linear Algebra (linalg)

####  Vectors 

- `add[T](v []T, w []T) []T`: Adds two vectors `a` and `b`: `(a + b)`
- `subtract[T](v []T, w []T) []T`: Subtracts two vectors `a` and `b`: `(a - b)`
- `vector_sum[T](vector_list [][]T) []T`: Sums a list of vectors, example: `vector_sum([[1,2],[3,4]]) => [4, 6]`
- `scalar_multiply[T](c f64, v []T) []T`: Multiplies a scalar value `c` to each element of a vector `v`
- `vector_mean[T](vector_list [][]T) []T`: Calculates 1/n sum_j (v[j])
- `dot[T](v []T, w []T) T`: Dot product of `v` and `w`
- `sum_of_squares[T](v []T) T`: Squares each term of a vector, example: [1,2,3]^2 = [1^2, 2^2, 3^2]
- `magnitude[T](v []T) T`: Module of a vector, example: || [3,4] || = 5
- `squared_distance[T](v []T, w []T) T`: Calculates sqrt[(v1-w1)^2 + (v2-w2)^2...]
- `distance[T](v []T, w []T) T`: Calculates the distance between `v` and `w`

#### Matrices

- `shape[T](a [][]T) (int, int)`: Returns the shape of a matrix (rows, columns)
- `get_row[T](a [][]T, i int) []T`: Gets the i-th row of a matrix as a vector
- `get_column[T](a [][]T, j int) []T`: Gets the j-th column of a matrix as a vector
- `flatten[T](m [][]T) []T`: Flattens a matrix to a 1D array
- `make_matrix[T](num_rows int, num_cols int, op fn (int, int) T) [][]T`: Makes a matrix using a formula given by function `op(i,j)` 
- `identity_matrix[T](n int) [][]T`: Returns an n-identity matrix
- `matmul[T](a [][]T, b [][]T) [][]T`: Multiplies matrix `a` with `b`

### Probabilites (prob)

#### Distributions (CDF and PDF)

- `beta_function(x f64, y f64) f64`
- `normal_cdf(x f64, mu f64, sigma f64) f64`
- `inverse_normal_cdf(p f64, mu f64, sigma f64, dp DistribParams) f64`
- `bernoulli_pdf(x f64, p f64) f64`
- `bernoulli_cdf(x f64, p f64) f64`
- `binomial_pdf(k int, n int, p f64) f64`
- `poisson_pdf(k int, lambda f64) f64`
- `poisson_cdf(k int, lambda f64) f64`
- `exponential_pdf(x f64, lambda f64) f64`
- `exponential_cdf(x f64, lambda f64) f64`
- `gamma_pdf(x f64, k f64, theta f64) f64`
- `chi_squared_pdf(x f64, df int) f64`
- `students_t_pdf(x f64, df int) f64`
- `f_distribution_pdf(x f64, d1 int, d2 int) f64`
- `beta_pdf(x f64, alpha f64, beta f64) f64`
- `uniform_pdf(x f64, a f64, b f64) f64`
- `uniform_cdf(x f64, a f64, b f64) f64`
- `negative_binomial_pdf(k int, r int, p f64) f64`
- `negative_binomial_cdf(k int, r int, p f64) f64`
- `multinomial_pdf(x []int, p []f64) f64`
- `expectation[T](x []T, p []T) T` - Generic expectation calculation

### Statistics (stats)

#### Descriptive Statistics
- `sum[T](x []T) f64` - Accepts generic numeric input, returns f64
- `mean[T](x []T) f64` - Accepts generic numeric input, returns f64
- `median(x []f64) f64` - Requires f64 (requires sorting)
- `quantile(x []f64, p f64) f64` - Requires f64 (requires sorting)
- `mode(x []f64) []f64` - Requires f64 (requires hashing)
- `range[T](x []T) T` - Accepts generic numeric input, returns same type
- `dev_mean[T](x []T) []f64` - Accepts generic numeric input, returns f64
- `variance[T](x []T) f64` - Accepts generic numeric input, returns f64
- `standard_deviation[T](x []T) f64` - Accepts generic numeric input, returns f64
- `interquartile_range(x []f64) f64` - Requires f64
- `covariance[T](x []T, y []T) f64` - Accepts generic numeric input, returns f64
- `correlation[T](x []T, y []T) f64` - Accepts generic numeric input, returns f64

#### Advanced Statistical Tests
- `anova_one_way(groups [][]f64) (f64, f64)` - One-way ANOVA test
- `confidence_interval_mean(x []f64, confidence_level f64) (f64, f64)` - CI for mean
- `cohens_d(group1 []f64, group2 []f64) f64` - Effect size between two groups
- `cramers_v(contingency [][]int) f64` - Effect size for categorical association
- `skewness(x []f64) f64` - Distribution asymmetry
- `kurtosis(x []f64) f64` - Distribution tailedness (excess kurtosis)

### Optimization (optim)

- `difference_quotient(f fn (f64) f64, x f64, h f64) f64`
- `partial_difference_quotient(f fn([]f64) f64, v []f64, i int, h f64) f64`
- `gradient(f fn([]f64) f64, v []f64, h f64) []f64`
- `gradient_step[T](v []T, gradient_vector []T, step_size T) []T` - Generic gradient descent step
- `sum_of_squares_gradient[T](v []T) []T` - Generic sum of squares gradient

## Statistics (stats) - Advanced Functions

### Hypothesis Testing
- `anova_one_way(groups [][]f64) (f64, f64)` - One-way ANOVA test comparing group means
  - Returns: F-statistic and p-value
  - Tests if 3+ groups have significantly different means

### Confidence Intervals
- `confidence_interval_mean(x []f64, confidence_level f64) (f64, f64)` - CI for population mean
  - Supports: 90%, 95%, 99% confidence levels
  - Returns: (lower_bound, upper_bound)

### Effect Sizes
- `cohens_d(group1 []f64, group2 []f64) f64` - Cohen's d effect size
  - Standardized difference between two group means
  - Interpretation: |d| > 0.8 = large effect
  
- `cramers_v(contingency [][]int) f64` - Cramér's V effect size
  - Measures association between categorical variables
  - Range: 0 (no association) to 1 (perfect association)

### Distribution Shape
- `skewness(x []f64) f64` - Measure of distribution asymmetry
  - Positive = right-skewed, Negative = left-skewed
  
- `kurtosis(x []f64) f64` - Measure of tail heaviness
  - Returns excess kurtosis (normal distribution = 0)

## Utilities (utils)

### Classification Metrics
- `build_confusion_matrix(y_true []int, y_pred []int) ConfusionMatrix` - Build confusion matrix from predictions
- `(ConfusionMatrix).accuracy() f64` - (TP+TN)/Total
- `(ConfusionMatrix).precision() f64` - TP/(TP+FP)
- `(ConfusionMatrix).recall() f64` - TP/(TP+FN) [Sensitivity]
- `(ConfusionMatrix).specificity() f64` - TN/(TN+FP)
- `(ConfusionMatrix).f1_score() f64` - Harmonic mean of precision & recall
- `(ConfusionMatrix).false_positive_rate() f64` - FP/(FP+TN)
- `(ConfusionMatrix).summary() string` - Formatted metrics summary

### ROC & AUC
- `roc_curve(y_true []int, y_proba []f64) ROC_Curve` - Generate ROC curve with AUC
  - Calculates TPR and FPR at different thresholds
  - Returns ROC_Curve with auc value (0-1 scale)
- `(ROC_Curve).auc_value() f64` - Extract Area Under Curve

### Quick Metrics Calculation
- `binary_classification_metrics(y_true []int, y_pred []int) map[string]f64`
  - Returns all 6 metrics in one call: accuracy, precision, recall, specificity, f1_score, fpr
  
- `regression_metrics(y_true []f64, y_pred []f64) map[string]f64`
  - Returns: mse, rmse, mae, r2

### Hyperparameter Tuning
- `generate_param_grid(param_ranges map[string][]f64) []map[string]f64`
  - Generates all parameter combinations for grid search
  - Supports 1-3 parameters

### Training Utilities
- `(TrainingProgress).format_log() string` - Pretty-print training progress with metrics
- `early_stopping(losses []f64, patience int) bool` - Early stopping based on loss plateau
  - Prevents overfitting by checking if loss hasn't improved for N epochs
  
- `decay_learning_rate(initial_lr f64, epoch int, decay_rate f64) f64`
  - Exponential learning rate decay: `lr = initial_lr * (decay_rate)^epoch`

### Feature Normalization
- `normalize_features(x [][]f64) ([][]f64, []f64, []f64)` - Standardize features using z-score normalization
  - Returns: (normalized_data, feature_means, feature_stds)
  - Computes mean and std for each feature, then applies (x - mean) / std
  - Essential for ML algorithms sensitive to feature scaling
  
- `apply_normalization(x [][]f64, means []f64, stds []f64) [][]f64` - Apply pre-computed normalization
  - Uses statistics from training data on new data
  - Prevents data leakage in train/test split scenarios
  - Handles zero standard deviation gracefully

### Basic Utilities
- `factorial(n int) f64` - Compute factorial
- `combinations(n int, k int) f64` - Binomial coefficient C(n,k)
- `range(n int) []int` - Generate range [0, 1, ..., n-1]

### Dataset Functions
- `load_iris() !Dataset` - Iris dataset (150 samples, 4 features, 3 classes)
- `load_wine() !Dataset` - Wine dataset (178 samples, 13 features→4, 3 classes)
- `load_breast_cancer() !Dataset` - Breast cancer dataset (binary classification)
- `load_boston_housing() !RegressionDataset` - Boston housing (506 samples, regression)
- `load_linear_regression() RegressionDataset` - Synthetic linear data (20 samples)
- `(Dataset).summary() string` - Dataset information and class distribution
- `(Dataset).train_test_split(test_size f64) (Dataset, Dataset)` - Split dataset
- `(Dataset).xy() ([][]f64, []int)` - Get features and targets separately
- Similar methods for `RegressionDataset` with continuous targets

## Neural Networks (nn)

### Loss Functions

#### Generic Loss Functions (Accept `int` or `f64`)
- `mse_loss[T](y_true []T, y_pred []T) f64` - Mean Squared Error
- `mse_loss_gradient[T](y_true []T, y_pred []T) []f64` - MSE gradient
- `mae_loss[T](y_true []T, y_pred []T) f64` - Mean Absolute Error
- `mae_loss_gradient[T](y_true []T, y_pred []T) []f64` - MAE gradient
- `huber_loss[T](y_true []T, y_pred []T, delta f64) f64` - Robust loss function
- `hinge_loss[T](y_true []T, y_pred []T) f64` - SVM-style loss
- `squared_hinge_loss[T](y_true []T, y_pred []T) f64` - Squared hinge loss
- `cosine_similarity_loss[T](y_true []T, y_pred []T) f64` - Cosine similarity-based loss
- `triplet_loss[T](anchor []T, positive []T, negative []T, margin f64) f64` - Metric learning loss

#### Fixed-Type Loss Functions (Require `f64`)
- `binary_crossentropy_loss(y_true []f64, y_pred []f64) f64` - Binary classification loss
- `binary_crossentropy_loss_gradient(y_true []f64, y_pred []f64) []f64` - BCE gradient
- `categorical_crossentropy_loss(y_true [][]f64, y_pred [][]f64) f64` - Multi-class loss
- `sparse_categorical_crossentropy_loss(y_true []int, y_pred [][]f64) f64` - Sparse multi-class loss
- `kl_divergence_loss(y_true []f64, y_pred []f64) f64` - KL divergence
- `contrastive_loss(y_true f64, distance f64, margin f64) f64` - Siamese network loss

## Machine Learning (ml)

### Regression (Generic Support)
All regression functions support generic numeric types with automatic conversion to f64 for precision.

**Model Training & Prediction:**
- `linear_regression[T](x [][]T, y []T) LinearModel[T]` - Ordinary Least Squares regression
- `linear_predict[T](model LinearModel[T], x [][]T) []T` - Predictions using linear model
- `logistic_regression[T](x [][]T, y []T, iterations int, learning_rate T) LogisticModel[T]` - Binary classification with gradient descent
- `logistic_predict_proba[T](model LogisticModel[T], x [][]T) []T` - Probability predictions
- `logistic_predict[T](model LogisticModel[T], x [][]T, threshold T) []T` - Class predictions

**Error Metrics (Generic input, f64 output for precision):**
- `mse[T](y_true []T, y_pred []T) f64` - Mean Squared Error
- `rmse[T](y_true []T, y_pred []T) f64` - Root Mean Squared Error
- `mae[T](y_true []T, y_pred []T) f64` - Mean Absolute Error
- `r_squared[T](y_true []T, y_pred []T) f64` - R² coefficient of determination

### Classification
- `logistic_classifier(x [][]f64, y []f64, iterations int, learning_rate f64) LogisticClassifier[f64]`
- `logistic_classifier_predict(model LogisticClassifier[f64], x [][]f64, threshold f64) []int`
- `logistic_classifier_predict_proba(model LogisticClassifier[f64], x [][]f64) []f64`
- `naive_bayes_classifier(x [][]f64, y []int) NaiveBayesClassifier` - Probabilistic classifier
- `naive_bayes_predict(model NaiveBayesClassifier, x [][]f64) []int`
- `svm_classifier(x [][]f64, y []f64, learning_rate f64, iterations int, gamma f64, kernel string) SVMClassifier`
- `svm_predict(model SVMClassifier, x [][]f64) []int`
- `random_forest_classifier(x [][]f64, y []int, num_trees int, max_depth int) RandomForestClassifier`
- `random_forest_predict(model RandomForestClassifier, x [][]f64) []int`
- `accuracy(y_true []int, y_pred []int) f64` - Classification accuracy

### Clustering
- `kmeans(data [][]f64, k int, max_iterations int) KMeansModel` - K-means clustering
- `kmeans_predict(model KMeansModel, data [][]f64) []int` - Predict cluster assignments
- `kmeans_inertia(model KMeansModel, data [][]f64) f64` - Sum of squared distances
- `silhouette_coefficient(data [][]f64, labels []int) f64` - Cluster quality metric
- `hierarchical_clustering(data [][]f64, num_clusters int) HierarchicalClustering` - Agglomerative clustering
- `dbscan(data [][]f64, eps f64, min_points int) []int` - Density-based clustering

## Hypothesis Testing (hypothesis)

### Statistical Tests
- `t_test_one_sample(x []f64, mu f64, tp TestParams) (f64, f64)` - One-sample t-test
- `t_test_two_sample(x []f64, y []f64, tp TestParams) (f64, f64)` - Two-sample t-test
- `chi_squared_test(observed []f64, expected []f64) (f64, f64)` - Chi-squared goodness of fit
- `correlation_test(x []f64, y []f64, tp TestParams) (f64, f64)` - Pearson correlation significance
- `wilcoxon_signed_rank_test(x []f64, y []f64) (f64, f64)` - Non-parametric paired test
- `mann_whitney_u_test(x []f64, y []f64) (f64, f64)` - Non-parametric independent samples test

## Neural Network Layers (nn)

### Layer Operations
- `dense_layer(input_size int, output_size int) DenseLayer` - Fully connected layer
- `(layer DenseLayer) forward(input []f64) []f64` - Forward pass
- `(mut layer DenseLayer) backward(grad_output []f64, input []f64, learning_rate f64) []f64` - Backward pass

### Activation Functions
- `relu(x f64) f64` - ReLU activation
- `relu_derivative(x f64) f64` - ReLU derivative
- `sigmoid(x f64) f64` - Sigmoid activation
- `sigmoid_derivative(x f64) f64` - Sigmoid derivative
- `tanh(x f64) f64` - Hyperbolic tangent
- `tanh_derivative(x f64) f64` - Tanh derivative
- `softmax(x []f64) []f64` - Softmax activation

### Sequential Network
- `sequential(layer_sizes []int, activation_fn string) NeuralNetwork` - Create sequential model
- `(net NeuralNetwork) forward(input []f64) []f64` - Forward propagation
- `(mut net NeuralNetwork) backward(grad_output []f64, input []f64, learning_rate f64) []f64` - Backpropagation
- `(mut net NeuralNetwork) train(x_train [][]f64, y_train []f64, config TrainingConfig)` - Train network
- `(net NeuralNetwork) predict(x [][]f64) [][]f64` - Batch predictions
- `(net NeuralNetwork) predict_single(x []f64) []f64` - Single prediction
- `(net NeuralNetwork) evaluate(x_test [][]f64, y_test []f64) f64` - Evaluate accuracy

## Usage Examples

### Generic Types Support
```v
import vstats.stats
import vstats.nn

// Statistical functions accept both int and f64
int_mean := stats.mean([1, 2, 3, 4, 5])  // Returns f64
f64_mean := stats.mean([1.0, 2.0, 3.0])  // Also returns f64

// Neural network loss functions are also generic
y_true := [1, 2, 3]
y_pred := [1, 2, 2]
loss := nn.mse_loss(y_true, y_pred)  // Works with int arrays

// See examples/generic_types_example.v for comprehensive demo
```

### ANOVA Test
```v
import vstats.stats

// Compare three treatment groups
control := [1.0, 2.0, 3.0, 2.5]
treatment_a := [4.0, 5.0, 4.5, 5.5]
treatment_b := [7.0, 8.0, 7.5, 8.5]

f_stat, p_val := stats.anova_one_way([control, treatment_a, treatment_b])
if p_val < 0.05 {
    println("Groups have significantly different means")
}
```

### Model Evaluation
```v
import vstats.utils

y_true := [1, 1, 0, 0, 1, 0]
y_pred := [1, 0, 0, 1, 1, 0]

// Method 1: Build confusion matrix manually
cm := utils.build_confusion_matrix(y_true, y_pred)
println("Accuracy: ${cm.accuracy():.4f}")
println("F1 Score: ${cm.f1_score():.4f}")
println(cm.summary())

// Method 2: Get all metrics at once
metrics := utils.binary_classification_metrics(y_true, y_pred)
for name, value in metrics {
    println("${name}: ${value:.4f}")
}
```

### ROC-AUC Score
```v
import vstats.utils

y_true := [1, 1, 0, 1, 0, 0]
y_proba := [0.9, 0.8, 0.3, 0.7, 0.2, 0.1]

roc := utils.roc_curve(y_true, y_proba)
println("AUC: ${roc.auc:.4f}")  // Closer to 1.0 is better
```

### Training with Early Stopping and LR Decay
```v
import vstats.utils

mut losses := []f64{}
for epoch in 0..100 {
    // Train model, compute loss
    loss := compute_loss()
    losses << loss
    
    // Check if should stop
    if utils.early_stopping(losses, patience: 10) {
        println("Early stopping at epoch ${epoch}")
        break
    }
    
    // Decay learning rate
    lr := utils.decay_learning_rate(0.1, epoch, 0.95)
    optimizer.set_learning_rate(lr)
}
```

### Grid Search for Hyperparameters
```v
import vstats.utils

param_ranges := {
    'learning_rate': [0.001, 0.01, 0.1]
    'batch_size': [16.0, 32.0, 64.0]
}

grid := utils.generate_param_grid(param_ranges)
for combo in grid {
    lr := combo['learning_rate']
    batch := combo['batch_size']
    // Train model with these parameters
}
```

### Feature Normalization for ML
```v
import vstats.utils

// Load dataset
iris := utils.load_iris()!
train, test := iris.train_test_split(0.2)
x_train, y_train := train.xy()
x_test, y_test := test.xy()

// Normalize using training set statistics
x_train_norm, means, stds := utils.normalize_features(x_train)

// Apply same normalization to test set (prevents data leakage)
x_test_norm := utils.apply_normalization(x_test, means, stds)

// Now train model on normalized data
model := ml.logistic_regression(x_train_norm, y_train_float, 1000, 0.01)

// Predict and evaluate on normalized test data
predictions := ml.logistic_predict(model, x_test_norm, 0.5)
metrics := utils.binary_classification_metrics(y_test, predictions)
```

### Dataset Loading
```v
import vstats.utils

// Load iris dataset
iris := utils.load_iris()!
println(iris.summary())

// Split into train/test
train, test := iris.train_test_split(0.2)
x_train, y_train := train.xy()

// Evaluate
predictions := model.predict(x_test)
metrics := utils.regression_metrics(y_test, predictions)
```

## Roadmap

- Add more optimization algorithms
- Dimensionality reduction (PCA, t-SNE)
- Time series forecasting (ARIMA, exponential smoothing)
- Convolutional and Recurrent neural network layers
- Learning rate scheduling variants (warmup, cosine annealing)
- Model checkpointing and serialization
- GPU acceleration support
- Complete symbolic computation module

## Disclaimer

- This was written as an exercise to get V closer to Data Analytics and Machine Learning tasks
- Heavily inspired by the book from Joel Grus "Data Science from Scratch: First principles with Python"
- It is **not** optimized for performance (current focus is correctness and API design)
- Documentation is an ongoing effort

## Contributing

Contributions are welcome! The library structure is modular and easy to extend. See `AGENTS.md` for development guidelines.

## References

- [V Language Documentation](https://vlang.io)
- [Data Science from Scratch](https://github.com/joelgrus/data-science-from-scratch) by Joel Grus
- Statistical methods from standard textbooks

**Pull requests are welcome!**