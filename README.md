# VStats 0.0.2

A dependency-free Linear Algebra, Statistics, and Machine Learning library written from scratch in V.

## Quick Start

```v
import stats
import utils
import linalg

// Statistics
f_stat, p_val := stats.anova_one_way([group1, group2, group3])
lower, upper := stats.confidence_interval_mean(data, 0.95)

// Metrics
cm := utils.build_confusion_matrix(y_true, y_pred)
println("F1: ${cm.f1_score():.4f}")

// Linear Algebra
result := linalg.matmul(matrix_a, matrix_b)
distance := linalg.distance(vector_a, vector_b)
```

## Features Overview

| Module | Purpose | Status |
|--------|---------|--------|
| **linalg** | Vector & Matrix operations | ✓ Complete |
| **stats** | Descriptive & Advanced Statistics | ✓ Complete |
| **prob** | Probability Distributions | ✓ Complete |
| **optim** | Numerical Optimization | ✓ Complete |
| **utils** | Utilities, Metrics, Datasets | ✓ Complete |
| **ml** | Machine Learning Algorithms | ✓ Complete |
| **nn** | Neural Networks | ✓ Complete |
| **hypothesis** | Hypothesis Testing | ✓ Complete |
| **symbol** | Symbolic Computation | 🚧 WIP |

## Modules implemented

The following is the list of modules and functions implemented

### Linear Algebra (linalg)

####  Vectors 

- `add(v []f64, w []f64) []f64`: Adds two vectors `a` and `b`: `(a + b)`
- `subtract(v []f64, w []f64) []f64`: Subtracts two vectors `a` and `b`: `(a - b)`
- `vector_sum(vector_list [][]f64) []f64`: Sums a list of vectors, example: `vector_sum([[f64(1),2],[3,4]]) => [4.0, 6.0]`
- `scalar_multiply(c f64, v []f64) []f64`: Multiplies an scalar value `c` to each element of a vector `v`
- `vector_mean(vector_list [][]f64) []f64`: Calculates 1/n sum_j (v[j])
- `dot(v []f64, w []f64) f64`: Dot product of `v` and `w`
- `sum_of_squares(v []f64) f64`: Squares each term of a vector, example: [1,2,3]^2 = [1^2, 2^2, 3^2]
- `magnitude(v []f64) f64`: Module of a vector, example: || [3,4] || = 5
- `squared_distance(v []f64, w []f64) f64`: Calculates sqrt[(v1-w1)^2 + (v2-w2)^2...]
- `distance(v []f64, w []f64) f64`: Calculates the distance between `v` and `w`

#### Matrices

- `shape(a [][]f64) (int, int)`: Returns the shape of a matrix (rows, columns)
- `get_row(a [][]f64, i int) []f64`: Gets the i-th row of a matrix as a vector
- `get_column(a [][]f64, j int) []f64`: Gets the j-th column of a matrix as a vector
- `make_matrix(num_rows int, num_cols int, op fn (int, int) f64) [][]f64`: Makes a matrix using a formula given by function `op(i,j)` 
- `identity_matrix(n int) [][]f64`: Returns a n-identity matrix
- `matmul(a [][]f64, b [][]f64) [][]f64`: Multuplies matrix `a` with `b`

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
- `expectation(x []f64, p []f64) f64`

### Statistics (stats)

#### Descriptive Statistics
- `sum(x []f64) f64`
- `mean(x []f64) f64`
- `median(x []f64) f64`
- `quantile(x []f64, p f64) f64`
- `mode(x []f64) []f64`
- `range(x []f64) f64`
- `dev_mean(x []f64) []f64`
- `variance(x []f64) f64`
- `standard_deviation(x []f64) f64`
- `interquartile_range(x []f64) f64`
- `covariance(x []f64, y []f64) f64`
- `correlation(x []f64, y []f64) f64`

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
- `gradient_step(v []f64, gradient_vector []f64, step_size f64) []f64`
- `sum_of_squares_gradient(v []f64) []f64`

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

## Usage Examples

### ANOVA Test
```v
import stats

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
import utils

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
import utils

y_true := [1, 1, 0, 1, 0, 0]
y_proba := [0.9, 0.8, 0.3, 0.7, 0.2, 0.1]

roc := utils.roc_curve(y_true, y_proba)
println("AUC: ${roc.auc:.4f}")  // Closer to 1.0 is better
```

### Training with Early Stopping and LR Decay
```v
import utils

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
import utils

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

### Dataset Loading
```v
import utils

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