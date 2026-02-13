# VStats Tutorial - Complete Guide

A hands-on tutorial for the VStats library covering all modules with practical examples.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Linear Algebra (linalg)](#linear-algebra-linalg)
3. [Statistics (stats)](#statistics-stats)
4. [Probability Distributions (prob)](#probability-distributions-prob)
5. [Optimization (optim)](#optimization-optim)
6. [Machine Learning (ml)](#machine-learning-ml)
7. [Neural Networks (nn)](#neural-networks-nn)
8. [Utilities (utils)](#utilities-utils)
9. [Hypothesis Testing (hypothesis)](#hypothesis-testing-hypothesis)
10. [Complete Example: End-to-End Workflow](#complete-example-end-to-end-workflow)

---

## Getting Started

### Installation

```bash
v install https://github.com/rodabt/vstats
```

### Basic Import

```v
import vstats.linalg
import vstats.stats
import vstats.prob
import vstats.ml
import vstats.nn
import vstats.utils
```

---

## Linear Algebra (linalg)

The `linalg` module provides vector and matrix operations with generic type support.

### Vector Operations

```v
import vstats.linalg

fn main() {
    // Basic vector operations
    v1 := [1.0, 2.0, 3.0]
    v2 := [4.0, 5.0, 6.0]
    
    // Vector addition
    sum := linalg.add(v1, v2)
    println("Sum: ${sum}")  // [5.0, 7.0, 9.0]
    
    // Vector subtraction
    diff := linalg.subtract(v1, v2)
    println("Difference: ${diff}")  // [-3.0, -3.0, -3.0]
    
    // Dot product
    dot := linalg.dot(v1, v2)
    println("Dot product: ${dot}")  // 32.0
    
    // Scalar multiplication
    scaled := linalg.scalar_multiply(2.0, v1)
    println("Scaled: ${scaled}")  // [2.0, 4.0, 6.0]
    
    // Vector magnitude (length)
    mag := linalg.magnitude([3.0, 4.0])
    println("Magnitude: ${mag}")  // 5.0
    
    // Distance between vectors
    dist := linalg.distance([1.0, 2.0], [4.0, 6.0])
    println("Distance: ${dist}")  // 5.0
    
    // Sum of squares
    sos := linalg.sum_of_squares([1.0, 2.0, 3.0])
    println("Sum of squares: ${sos}")  // 14.0
}
```

### Vector Aggregation

```v
import vstats.linalg

fn main() {
    // Sum multiple vectors
    vectors := [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    total := linalg.vector_sum(vectors)
    println("Vector sum: ${total}")  // [9.0, 12.0]
    
    // Mean of vectors
    mean_vec := linalg.vector_mean(vectors)
    println("Vector mean: ${mean_vec}")  // [3.0, 4.0]
}
```

### Matrix Operations

```v
import vstats.linalg

fn main() {
    // Matrix creation
    a := [[1.0, 2.0], [3.0, 4.0]]
    b := [[5.0, 6.0], [7.0, 8.0]]
    
    // Matrix shape
    rows, cols := linalg.shape(a)
    println("Matrix shape: ${rows}x${cols}")  // 2x2
    
    // Matrix multiplication
    c := linalg.matmul(a, b)
    println("Matrix product: ${c}")  // [[19.0, 22.0], [43.0, 50.0]]
    
    // Identity matrix
    identity := linalg.identity_matrix[f64](3)
    println("Identity matrix: ${identity}")
    // [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    
    // Matrix transpose
    m := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    mt := linalg.transpose_f64(m)
    println("Transposed: ${mt}")
    // [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    
    // Flatten matrix to vector
    flat := linalg.flatten(m)
    println("Flattened: ${flat}")  // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    // Reshape vector to matrix
    v := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    reshaped := linalg.reshape(v, 2, 3)
    println("Reshaped: ${reshaped}")
    // [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}
```

### Rotation Matrices

```v
import vstats.linalg
import math

fn main() {
    // Create rotation matrices (for 3D graphics)
    angle := math.pi / 4  // 45 degrees
    
    rot_z := linalg.rotation_z[f64](angle)
    rot_y := linalg.rotation_y[f64](angle)
    rot_x := linalg.rotation_x[f64](angle)
    
    // Combined rotation
    rot_all := linalg.rotation[f64](angle, angle, angle)
    println("Rotation matrix: ${rot_all}")
}
```

---

## Statistics (stats)

The `stats` module provides descriptive and inferential statistics.

### Descriptive Statistics

```v
import vstats.stats

fn main() {
    data := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    // Central tendency
    mean_val := stats.mean(data)
    println("Mean: ${mean_val}")  // 5.5
    
    med := stats.median(data)
    println("Median: ${med}")  // 5.5
    
    modes := stats.mode(data)
    println("Mode: ${modes}")
    
    // Spread
    variance := stats.variance(data)
    println("Variance: ${variance}")  // 9.166...
    
    std_dev := stats.standard_deviation(data)
    println("Standard Deviation: ${std_dev}")  // 3.027...
    
    data_range := stats.range(data)
    println("Range: ${data_range}")  // 9.0
    
    iqr := stats.interquartile_range(data)
    println("IQR: ${iqr}")
    
    // Quantiles
    q1 := stats.quantile(data, 0.25)
    q3 := stats.quantile(data, 0.75)
    println("Q1: ${q1}, Q3: ${q3}")
}
```

### Correlation and Covariance

```v
import vstats.stats

fn main() {
    x := [1.0, 2.0, 3.0, 4.0, 5.0]
    y := [2.0, 4.0, 5.0, 4.0, 5.0]
    
    // Covariance
    cov := stats.covariance(x, y)
    println("Covariance: ${cov}")  // 1.5
    
    // Correlation coefficient
    corr := stats.correlation(x, y)
    println("Correlation: ${corr}")  // ~0.832
}
```

### Advanced Statistical Tests

```v
import vstats.stats

fn main() {
    // ANOVA - Analysis of Variance
    group1 := [23.0, 25.0, 28.0, 30.0, 26.0]
    group2 := [31.0, 33.0, 35.0, 32.0, 34.0]
    group3 := [40.0, 42.0, 38.0, 41.0, 39.0]
    
    groups := [group1, group2, group3]
    f_stat, p_value := stats.anova_one_way(groups)
    println("ANOVA F-statistic: ${f_stat}")
    println("ANOVA p-value: ${p_value}")
    
    // Confidence Interval for mean
    data := [50.0, 52.0, 48.0, 51.0, 49.0, 53.0, 50.0, 51.0]
    lower, upper := stats.confidence_interval_mean(data, 0.95)
    println("95% Confidence Interval: [${lower}, ${upper}]")
    
    // Effect size: Cohen's d
    group_a := [100.0, 102.0, 98.0, 101.0, 99.0]
    group_b := [110.0, 112.0, 108.0, 111.0, 109.0]
    d := stats.cohens_d(group_a, group_b)
    println("Cohen's d: ${d}")  // Effect size between groups
    
    // Skewness and Kurtosis
    skew := stats.skewness(data)
    kurt := stats.kurtosis(data)
    println("Skewness: ${skew}")
    println("Kurtosis: ${kurt}")
}
```

### Cramer's V (Effect Size for Categorical Data)

```v
import vstats.stats

fn main() {
    // Contingency table: rows = Gender, cols = Preference
    //               Preference A    Preference B
    // Male              30             20
    // Female            15             35
    
    contingency := [
        [30, 20],
        [15, 35]
    ]
    
    cramers_v := stats.cramers_v(contingency)
    println("Cramer's V: ${cramers_v}")  // Measures association strength
}
```

---

## Probability Distributions (prob)

The `prob` module provides probability density functions (PDF) and cumulative distribution functions (CDF).

### Normal Distribution

```v
import vstats.prob

fn main() {
    mu := 0.0    // mean
    sigma := 1.0 // standard deviation
    
    // Cumulative Distribution Function (CDF)
    p1 := prob.normal_cdf(0.0, mu, sigma)  // P(X <= 0)
    println("P(X <= 0): ${p1}")  // 0.5
    
    p2 := prob.normal_cdf(1.96, mu, sigma)
    println("P(X <= 1.96): ${p2}")  // ~0.975
    
    // Inverse CDF (quantile function)
    x := prob.inverse_normal_cdf(0.95, mu, sigma, prob.DistribParams{})
    println("95th percentile: ${x}")  // ~1.645
}
```

### Discrete Distributions

```v
import vstats.prob

fn main() {
    // Bernoulli distribution
    p_bernoulli := prob.bernoulli_pdf(1, 0.7)  // P(X=1) with p=0.7
    println("Bernoulli P(X=1): ${p_bernoulli}")  // 0.7
    
    // Binomial distribution
    p_binom := prob.binomial_pdf(3, 10, 0.5)  // P(X=3) in 10 trials
    println("Binomial P(X=3): ${p_binom}")
    
    // Poisson distribution
    p_poisson := prob.poisson_pdf(5, 3.0)  // P(X=5) with lambda=3
    println("Poisson P(X=5): ${p_poisson}")
    
    // Poisson CDF
    p_poisson_cdf := prob.poisson_cdf(5, 3.0)  // P(X <= 5)
    println("Poisson P(X <= 5): ${p_poisson_cdf}")
}
```

### Continuous Distributions

```v
import vstats.prob

fn main() {
    // Exponential distribution
    p_exp := prob.exponential_pdf(2.0, 0.5)  // P(X=2) with lambda=0.5
    println("Exponential PDF: ${p_exp}")
    
    p_exp_cdf := prob.exponential_cdf(2.0, 0.5)  // P(X <= 2)
    println("Exponential CDF: ${p_exp_cdf}")
    
    // Gamma distribution
    p_gamma := prob.gamma_pdf(2.0, 2.0, 1.0)  // x=2, k=2, theta=1
    println("Gamma PDF: ${p_gamma}")
    
    // Chi-squared distribution
    p_chi2 := prob.chi_squared_pdf(3.0, 5)  // x=3, df=5
    println("Chi-squared PDF: ${p_chi2}")
    
    // Student's t-distribution
    p_t := prob.students_t_pdf(2.0, 10)  // x=2, df=10
    println("t-distribution PDF: ${p_t}")
    
    // F-distribution
    p_f := prob.f_distribution_pdf(2.0, 5, 10)  // x=2, d1=5, d2=10
    println("F-distribution PDF: ${p_f}")
    
    // Beta distribution
    p_beta := prob.beta_pdf(0.5, 2.0, 3.0)  // x=0.5, alpha=2, beta=3
    println("Beta PDF: ${p_beta}")
    
    // Uniform distribution
    p_unif := prob.uniform_pdf(0.5, 0.0, 1.0)  // x=0.5, a=0, b=1
    println("Uniform PDF: ${p_unif}")  // 1.0
    
    p_unif_cdf := prob.uniform_cdf(0.5, 0.0, 1.0)
    println("Uniform CDF: ${p_unif_cdf}")  // 0.5
}
```

### Multinomial Distribution

```v
import vstats.prob

fn main() {
    // Multinomial: 10 trials, 3 categories with probabilities
    counts := [5, 3, 2]  // Observed counts
    probs := [0.5, 0.3, 0.2]  // Probabilities for each category
    
    p_multi := prob.multinomial_pdf(counts, probs)
    println("Multinomial probability: ${p_multi}")
}
```

### Expected Value

```v
import vstats.prob

fn main() {
    // Expected value of a discrete random variable
    values := [1.0, 2.0, 3.0, 4.0, 5.0]
    probabilities := [0.1, 0.2, 0.4, 0.2, 0.1]
    
    expected := prob.expectation(values, probabilities)
    println("Expected value: ${expected}")  // 3.0
}
```

---

## Optimization (optim)

The `optim` module provides numerical optimization utilities.

### Gradient Computation

```v
import vstats.optim
import math

fn main() {
    // Function to optimize: f(x) = x^2
    f := fn (x f64) f64 {
        return x * x
    }
    
    // Compute derivative at x=3 using difference quotient
    h := 0.0001
    derivative := optim.difference_quotient(f, 3.0, h)
    println("Derivative of x^2 at x=3: ${derivative}")  // ~6.0
    
    // Gradient of sum of squares: f(v) = sum(v[i]^2)
    sum_of_squares := fn (v []f64) f64 {
        mut sum := 0.0
        for x in v {
            sum += x * x
        }
        return sum
    }
    
    v := [1.0, 2.0, 3.0]
    grad := optim.gradient(sum_of_squares, v, h)
    println("Gradient: ${grad}")  // [2.0, 4.0, 6.0]
}
```

### Gradient Descent Step

```v
import vstats.optim
import vstats.linalg

fn main() {
    // Current position
    v := [3.0, 4.0]
    
    // Gradient at current position (for f(x,y) = x^2 + y^2)
    gradient := [6.0, 8.0]
    
    // Take gradient descent step with step size 0.1
    step_size := 0.1
    new_v := optim.gradient_step(v, gradient, -step_size)
    println("New position: ${new_v}")  // [2.4, 3.2]
    
    // Sum of squares gradient (analytical)
    v2 := [1.0, 2.0, 3.0]
    sos_grad := optim.sum_of_squares_gradient(v2)
    println("Sum of squares gradient: ${sos_grad}")  // [2.0, 4.0, 6.0]
}
```

---

## Machine Learning (ml)

The `ml` module provides classification and regression algorithms.

### Linear Regression

```v
import vstats.ml

fn main() {
    // Training data: predict y from x
    // Simple linear relationship: y = 2x + 1 + noise
    x_train := [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y_train := [3.0, 5.0, 7.0, 9.0, 11.0]
    
    // Train linear regression model
    model := ml.linear_regression(x_train, y_train)
    
    println("Intercept: ${model.intercept}")  // ~1.0
    println("Coefficients: ${model.coefficients}")  // ~[2.0]
    
    // Make predictions
    x_test := [[6.0], [7.0], [8.0]]
    predictions := ml.linear_predict(model, x_test)
    println("Predictions: ${predictions}")  // [13.0, 15.0, 17.0]
    
    // Evaluate model
    y_test := [13.0, 15.0, 17.0]
    r2 := ml.r_squared(y_test, predictions)
    println("R-squared: ${r2}")  // 1.0 (perfect fit)
    
    mse := ml.mse(y_test, predictions)
    rmse := ml.rmse(y_test, predictions)
    mae := ml.mae(y_test, predictions)
    println("MSE: ${mse}, RMSE: ${rmse}, MAE: ${mae}")
}
```

### Logistic Regression (Binary Classification)

```v
import vstats.ml

fn main() {
    // Binary classification data
    // Features: [feature1, feature2]
    x_train := [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [6.0, 7.0],
        [7.0, 8.0],
    ]
    // Labels: 0 or 1
    y_train := [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    
    // Train logistic regression model
    iterations := 1000
    learning_rate := 0.1
    model := ml.logistic_regression(x_train, y_train, iterations, learning_rate)
    
    println("Trained: ${model.trained}")
    println("Intercept: ${model.intercept}")
    println("Coefficients: ${model.coefficients}")
    
    // Predict probabilities
    x_test := [[4.0, 5.0], [8.0, 9.0]]
    proba := ml.logistic_predict_proba(model, x_test)
    println("Probabilities: ${proba}")
    
    // Predict classes with threshold 0.5
    predictions := ml.logistic_predict(model, x_test, 0.5)
    println("Predictions: ${predictions}")
}
```

### Logistic Classifier with Feature Normalization

```v
import vstats.ml

fn main() {
    // Data with different feature scales
    x_train := [
        [1.0, 1000.0],
        [2.0, 2000.0],
        [3.0, 3000.0],
        [5.0, 5000.0],
    ]
    y_train := [0.0, 0.0, 1.0, 1.0]
    
    // Train with automatic feature normalization
    model := ml.logistic_classifier(x_train, y_train, 1000, 0.01)
    
    // Make predictions (normalization applied automatically)
    x_test := [[4.0, 4000.0]]
    predictions := ml.logistic_classifier_predict(model, x_test, 0.5)
    proba := ml.logistic_classifier_predict_proba(model, x_test)
    
    println("Prediction: ${predictions[0]}")
    println("Probability: ${proba[0]}")
}
```

### Naive Bayes Classifier

```v
import vstats.ml
import vstats.utils

fn main() {
    // Load Titanic dataset
    dataset := utils.load_titanic() or {
        println("Error loading dataset: ${err}")
        return
    }
    
    // Split data
    split_idx := int(f64(dataset.features.len) * 0.8)
    x_train := dataset.features[0..split_idx]
    y_train := dataset.target[0..split_idx]
    x_test := dataset.features[split_idx..dataset.features.len]
    y_test := dataset.target[split_idx..dataset.target.len]
    
    // Train Naive Bayes classifier
    model := ml.naive_bayes_classifier(x_train, y_train)
    println("Classes: ${model.classes}")
    println("Trained: ${model.trained}")
    
    // Make predictions
    predictions := ml.naive_bayes_predict(model, x_test)
    
    // Evaluate
    accuracy := ml.accuracy(y_test, predictions)
    precision := ml.precision(y_test, predictions, 1)
    recall := ml.recall(y_test, predictions, 1)
    f1 := ml.f1_score(y_test, predictions, 1)
    
    println("Accuracy: ${accuracy:.4f}")
    println("Precision: ${precision:.4f}")
    println("Recall: ${recall:.4f}")
    println("F1 Score: ${f1:.4f}")
    
    // Confusion matrix
    cm := ml.confusion_matrix(y_test, predictions)
    println("Confusion Matrix: ${cm}")
}
```

### Support Vector Machine (SVM)

```v
import vstats.ml

fn main() {
    // Training data
    x_train := [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
    ]
    // Labels must be -1 or 1 for SVM
    y_train := [-1.0, -1.0, 1.0, 1.0]
    
    // Train SVM with RBF kernel
    learning_rate := 0.01
    iterations := 100
    gamma := 0.1
    kernel := "rbf"
    
    model := ml.svm_classifier(x_train, y_train, learning_rate, iterations, gamma, kernel)
    println("Trained: ${model.trained}")
    println("Number of support vectors: ${model.support_vectors.len}")
    
    // Make predictions
    x_test := [[2.5, 2.5], [3.5, 3.5]]
    predictions := ml.svm_predict(model, x_test)
    println("Predictions: ${predictions}")
}
```

### Random Forest Classifier

```v
import vstats.ml
import vstats.utils

fn main() {
    // Load dataset
    dataset := utils.load_titanic() or {
        println("Error: ${err}")
        return
    }
    
    // Split data
    split_idx := int(f64(dataset.features.len) * 0.8)
    x_train := dataset.features[0..split_idx]
    y_train := dataset.target[0..split_idx]
    x_test := dataset.features[split_idx..dataset.features.len]
    y_test := dataset.target[split_idx..dataset.target.len]
    
    // Train Random Forest
    num_trees := 50
    max_depth := 5
    model := ml.random_forest_classifier(x_train, y_train, num_trees, max_depth)
    println("Number of trees: ${model.num_trees}")
    println("Trained: ${model.trained}")
    
    // Make predictions
    predictions := ml.random_forest_classifier_predict(model, x_test)
    
    // Get probability estimates (vote proportions)
    proba := ml.random_forest_classifier_predict_proba(model, x_test)
    
    // Evaluate
    accuracy := ml.accuracy(y_test, predictions)
    println("Accuracy: ${accuracy:.4f}")
}
```

---

## Neural Networks (nn)

The `nn` module provides neural network layers, loss functions, and training utilities.

### Creating a Neural Network

```v
import vstats.nn

fn main() {
    // Create a sequential neural network
    // Layer sizes: [input, hidden1, hidden2, output]
    layer_sizes := [5, 10, 10, 2]
    activation_fn := "relu"  // or "sigmoid", "tanh"
    
    mut net := nn.sequential(layer_sizes, activation_fn)
    println("Network created with ${net.num_layers} layers")
    
    // Forward pass
    input := [0.5, 0.3, 0.2, 0.1, 0.4]
    output := net.forward(input)
    println("Output: ${output}")
}
```

### Training a Neural Network

```v
import vstats.nn

fn main() {
    // Create network for binary classification
    // Input: 3 features, Hidden: 8 neurons, Output: 1 neuron
    mut layer_sizes := [3, 8, 1]
    mut net := nn.sequential(layer_sizes, "relu")
    
    // Training data
    x_train := [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4],
    ]
    y_train := [0.0, 0.0, 1.0, 0.0]
    
    // Training configuration
    config := nn.training_config(0.01, 100, 2)  // lr=0.01, epochs=100, batch_size=2
    
    // Train
    net.train(x_train, y_train, config)
    
    // Evaluate
    loss := net.evaluate(x_train, y_train)
    println("Training loss: ${loss}")
    
    // Make predictions
    predictions := net.predict(x_train)
    println("Predictions: ${predictions}")
}
```

### Loss Functions

```v
import vstats.nn

fn main() {
    y_true := [1.0, 0.0, 1.0, 0.0]
    y_pred := [0.9, 0.1, 0.8, 0.2]
    
    // Mean Squared Error
    mse := nn.mse_loss(y_true, y_pred)
    println("MSE: ${mse}")
    
    // Mean Absolute Error
    mae := nn.mae_loss(y_true, y_pred)
    println("MAE: ${mae}")
    
    // Binary Cross-Entropy
    bce := nn.binary_crossentropy_loss(y_true, y_pred)
    println("Binary Cross-Entropy: ${bce}")
    
    // Huber Loss (robust to outliers)
    huber := nn.huber_loss(y_true, y_pred, 1.0)
    println("Huber Loss: ${huber}")
    
    // Hinge Loss (for SVM-like training)
    hinge := nn.hinge_loss(y_true, y_pred)
    println("Hinge Loss: ${hinge}")
    
    // Cosine Similarity Loss
    cosine := nn.cosine_similarity_loss(y_true, y_pred)
    println("Cosine Similarity Loss: ${cosine}")
}
```

### Categorical Cross-Entropy for Multi-class

```v
import vstats.nn

fn main() {
    // For multi-class classification (one-hot encoded)
    y_true := [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    y_pred := [
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8],
    ]
    
    loss := nn.categorical_crossentropy_loss(y_true, y_pred)
    println("Categorical Cross-Entropy: ${loss}")
    
    // Sparse categorical (class indices instead of one-hot)
    y_true_sparse := [0, 1, 2]
    loss_sparse := nn.sparse_categorical_crossentropy_loss(y_true_sparse, y_pred)
    println("Sparse Categorical Cross-Entropy: ${loss_sparse}")
}
```

### Activation Functions

```v
import vstats.nn

fn main() {
    x := -2.0
    
    // ReLU
    relu_out := nn.relu(x)
    println("ReLU(-2): ${relu_out}")  // 0
    
    // Sigmoid
    sigmoid_out := nn.sigmoid(x)
    println("Sigmoid(-2): ${sigmoid_out}")  // ~0.119
    
    // Tanh
    tanh_out := nn.tanh(x)
    println("Tanh(-2): ${tanh_out}")  // ~-0.964
    
    // Softmax (for multi-class output)
    logits := [1.0, 2.0, 3.0]
    softmax_out := nn.softmax(logits)
    println("Softmax: ${softmax_out}")  // [0.090, 0.245, 0.665]
}
```

### Advanced Layers

```v
import vstats.nn

fn main() {
    // Batch Normalization
    batch_norm := nn.batch_norm_layer(10)
    input := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    normalized := batch_norm.forward(input)
    println("Batch normalized: ${normalized}")
    
    // Dropout (during training)
    training_input := [1.0, 2.0, 3.0, 4.0, 5.0]
    dropout_rate := 0.2  // 20% dropout
    dropped := nn.dropout(training_input, dropout_rate)
    println("After dropout: ${dropped}")
    
    // 1D Convolution
    signal := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    kernel := [0.5, 1.0, 0.5]
    convolved := nn.conv1d(signal, kernel, 1)
    println("Convolved: ${convolved}")
    
    // 1D Max Pooling
    pooled := nn.max_pool1d(signal, 2, 2)
    println("Max pooled: ${pooled}")
    
    // 1D Average Pooling
    avg_pooled := nn.avg_pool1d(signal, 2, 2)
    println("Average pooled: ${avg_pooled}")
}
```

### Network Weights Management

```v
import vstats.nn

fn main() {
    mut net := nn.sequential([3, 5, 2], "relu")
    
    // Get current weights
    weights := net.get_weights()
    biases := net.get_biases()
    println("Number of weight matrices: ${weights.len}")
    
    // Set new weights (for transfer learning or initialization)
    // new_weights := [[...], [...]]  // Must match layer dimensions
    // net.set_weights(new_weights)
}
```

---

## Utilities (utils)

The `utils` module provides helper functions, metrics, and dataset utilities.

### Mathematical Functions

```v
import vstats.utils
import math

fn main() {
    // Sigmoid and derivatives
    x := 0.0
    sig := utils.sigmoid(x)
    println("Sigmoid(0): ${sig}")  // 0.5
    
    sig_deriv := utils.sigmoid_derivative(x)
    println("Sigmoid derivative: ${sig_deriv}")  // 0.25
    
    // Factorial
    fact := utils.factorial(5)
    println("5! = ${fact}")  // 120
    
    // Combinations
    combo := utils.combinations(10, 3)
    println("C(10,3) = ${combo}")  // 120
    
    // Permutations
    perm := utils.permutations(10, 3)
    println("P(10,3) = ${perm}")  // 720
    
    // Power
    pow := utils.ipow(2, 10)
    println("2^10 = ${pow}")  // 1024
}
```

### Regression Metrics

```v
import vstats.utils

fn main() {
    y_true := [3.0, 5.0, 7.0, 9.0]
    y_pred := [2.8, 5.2, 6.9, 9.1]
    
    // Mean Squared Error
    mse := utils.mse(y_true, y_pred)
    println("MSE: ${mse}")
    
    // Root Mean Squared Error
    rmse := utils.rmse(y_true, y_pred)
    println("RMSE: ${rmse}")
    
    // Mean Absolute Error
    mae := utils.mae(y_true, y_pred)
    println("MAE: ${mae}")
}
```

### Classification Metrics

```v
import vstats.utils

fn main() {
    y_true := [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred := [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    
    // Build confusion matrix
    cm := utils.build_confusion_matrix(y_true, y_pred)
    println("Confusion Matrix:")
    println("  TP: ${cm.true_positives}")
    println("  TN: ${cm.true_negatives}")
    println("  FP: ${cm.false_positives}")
    println("  FN: ${cm.false_negatives}")
    
    // Calculate metrics
    println("Accuracy: ${cm.accuracy():.4f}")
    println("Precision: ${cm.precision():.4f}")
    println("Recall: ${cm.recall():.4f}")
    println("Specificity: ${cm.specificity():.4f}")
    println("F1 Score: ${cm.f1_score():.4f}")
    println("FPR: ${cm.false_positive_rate():.4f}")
    
    // Summary string
    println(cm.summary())
}
```

### ROC Curve and AUC

```v
import vstats.utils

fn main() {
    y_true := [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    // Probability predictions
    y_proba := [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85, 0.95, 0.15]
    
    // Calculate ROC curve
    roc := utils.roc_curve(y_true, y_proba)
    println("AUC: ${roc.auc_value():.4f}")
    println("Number of thresholds: ${roc.thresholds.len}")
    
    // Access TPR and FPR arrays
    for i in 0..roc.thresholds.len {
        println("Threshold ${roc.thresholds[i]:.2f}: TPR=${roc.tpr[i]:.4f}, FPR=${roc.fpr[i]:.4f}")
    }
}
```

### Batch Metrics Calculation

```v
import vstats.utils

fn main() {
    y_true := [0, 1, 1, 0, 1]
    y_pred := [0, 1, 0, 0, 1]
    
    // Calculate all classification metrics at once
    metrics := utils.binary_classification_metrics(y_true, y_pred)
    println("All metrics: ${metrics}")
    // Access: metrics['accuracy'], metrics['precision'], etc.
    
    // Regression metrics
    y_true_reg := [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred_reg := [1.1, 1.9, 3.2, 3.8, 5.1]
    reg_metrics := utils.regression_metrics(y_true_reg, y_pred_reg)
    println("Regression metrics: ${reg_metrics}")
    // Access: reg_metrics['mse'], reg_metrics['r2'], etc.
}
```

### Datasets

```v
import vstats.utils

fn main() {
    // Load built-in Titanic dataset
    dataset := utils.load_titanic() or {
        println("Error: ${err}")
        return
    }
    
    println("Samples: ${dataset.features.len}")
    println("Features per sample: ${dataset.features[0].len}")
    println("Classes: ${dataset.target}")
    
    // Load Iris dataset
    iris := utils.load_iris() or {
        println("Error: ${err}")
        return
    }
    
    println("Iris samples: ${iris.features.len}")
    
    // Load Wine dataset
    wine := utils.load_wine() or {
        println("Error: ${err}")
        return
    }
    
    println("Wine samples: ${wine.features.len}")
    
    // Load Breast Cancer dataset
    cancer := utils.load_breast_cancer() or {
        println("Error: ${err}")
        return
    }
    
    println("Breast Cancer samples: ${cancer.features.len}")
}
```

### Data Splitting

```v
import vstats.utils

fn main() {
    // Load dataset
    dataset := utils.load_titanic() or { return }
    
    // Manual split
    split_idx := int(f64(dataset.features.len) * 0.8)
    x_train := dataset.features[0..split_idx]
    y_train := dataset.target[0..split_idx]
    x_test := dataset.features[split_idx..dataset.features.len]
    y_test := dataset.target[split_idx..dataset.target.len]
    
    println("Train: ${x_train.len}, Test: ${x_test.len}")
}
```

### Training Utilities

```v
import vstats.utils

fn main() {
    // Track training progress
    progress := utils.TrainingProgress{
        epoch: 10
        loss: 0.2345
        val_loss: 0.2567
        training_time: 45.5
        metrics: {'accuracy': 0.85, 'f1': 0.83}
    }
    
    println(progress.format_log())
    // Output: Epoch 10: loss=0.234500, val_loss=0.256700, time=45.50s, accuracy=0.8500, f1=0.8300
    
    // Early stopping check
    losses := [0.5, 0.45, 0.42, 0.40, 0.41, 0.42, 0.43, 0.44]
    should_stop := utils.early_stopping(losses, 3)  // patience=3
    println("Should stop: ${should_stop}")  // true (no improvement for 3 epochs)
    
    // Learning rate decay
    initial_lr := 0.01
    for epoch in 0..10 {
        lr := utils.decay_learning_rate(initial_lr, epoch, 0.95)
        println("Epoch ${epoch}: LR=${lr:.6f}")
    }
}
```

### Hyperparameter Grid Search

```v
import vstats.utils

fn main() {
    // Define parameter grid
    param_ranges := {
        'learning_rate': [0.001, 0.01, 0.1],
        'regularization': [0.0, 0.01, 0.1]
    }
    
    // Generate all combinations
    combinations := utils.generate_param_grid(param_ranges)
    println("Number of combinations: ${combinations.len}")
    
    for combo in combinations {
        println("Params: LR=${combo['learning_rate']}, Reg=${combo['regularization']}")
    }
}
```

---

## Hypothesis Testing (hypothesis)

The `hypothesis` module provides statistical hypothesis testing.

```v
import vstats.hypothesis

fn main() {
    // One-sample t-test
    sample := [52.0, 48.0, 50.0, 51.0, 49.0, 53.0, 47.0, 50.0]
    mu0 := 50.0  // Null hypothesis: mean = 50
    
    t_stat, p_value := hypothesis.one_sample_t_test(sample, mu0)
    println("One-sample t-test:")
    println("  t-statistic: ${t_stat:.4f}")
    println("  p-value: ${p_value:.4f}")
    
    if p_value < 0.05 {
        println("  Result: Reject null hypothesis")
    } else {
        println("  Result: Fail to reject null hypothesis")
    }
    
    // Two-sample t-test (independent)
    group1 := [52.0, 48.0, 50.0, 51.0, 49.0]
    group2 := [58.0, 55.0, 57.0, 56.0, 59.0]
    
    t_stat2, p_value2 := hypothesis.two_sample_t_test(group1, group2)
    println("\nTwo-sample t-test:")
    println("  t-statistic: ${t_stat2:.4f}")
    println("  p-value: ${p_value2:.4f}")
    
    // Chi-squared test for independence
    // Contingency table
    observed := [
        [30, 20],
        [15, 35]
    ]
    
    chi2, p_val_chi2 := hypothesis.chi_squared_test(observed)
    println("\nChi-squared test:")
    println("  Chi-squared: ${chi2:.4f}")
    println("  p-value: ${p_val_chi2:.4f}")
    
    // Kolmogorov-Smirnov test
    sample1 := [0.1, 0.2, 0.3, 0.4, 0.5]
    sample2 := [0.15, 0.25, 0.35, 0.45, 0.55]
    
    ks_stat, p_val_ks := hypothesis.ks_test(sample1, sample2)
    println("\nKolmogorov-Smirnov test:")
    println("  KS statistic: ${ks_stat:.4f}")
    println("  p-value: ${p_val_ks:.4f}")
}
```

---

## Complete Example: End-to-End Workflow

Here's a complete example demonstrating a full machine learning workflow:

```v
module main

import vstats.utils
import vstats.ml
import vstats.linalg
import vstats.stats

fn main() {
    println("=== VStats Complete Workflow Demo ===\n")
    
    // 1. Load Data
    println("1. Loading Titanic Dataset")
    dataset := utils.load_titanic() or {
        println("Error: ${err}")
        return
    }
    println("   Loaded ${dataset.features.len} samples")
    println("   Features: ${dataset.features[0].len}")
    
    // 2. Exploratory Data Analysis
    println("\n2. Exploratory Data Analysis")
    
    // Calculate feature statistics
    mut ages := []f64{}
    mut fares := []f64{}
    for sample in dataset.features {
        ages << sample[1]   // Age is feature 1
        fares << sample[4]  // Fare is feature 4
    }
    
    println("   Age statistics:")
    println("     Mean: ${stats.mean(ages):.2f}")
    println("     Std:  ${stats.standard_deviation(ages):.2f}")
    println("     Median: ${stats.median(ages):.2f}")
    
    println("   Fare statistics:")
    println("     Mean: ${stats.mean(fares):.2f}")
    println("     Std:  ${stats.standard_deviation(fares):.2f}")
    
    // Survival rate
    mut survivors := 0
    for label in dataset.target {
        if label == 1 {
            survivors++
        }
    }
    survival_rate := f64(survivors) / f64(dataset.target.len)
    println("   Survival rate: ${survival_rate*100:.1f}%")
    
    // 3. Data Preprocessing
    println("\n3. Data Preprocessing")
    
    // Split data
    split_idx := int(f64(dataset.features.len) * 0.8)
    x_train := dataset.features[0..split_idx]
    y_train := dataset.target[0..split_idx]
    x_test := dataset.features[split_idx..dataset.features.len]
    y_test := dataset.target[split_idx..dataset.target.len]
    println("   Train set: ${x_train.len} samples")
    println("   Test set:  ${x_test.len} samples")
    
    // 4. Train Multiple Models
    println("\n4. Training Models")
    
    // Model 1: Naive Bayes
    println("   Training Naive Bayes...")
    nb_model := ml.naive_bayes_classifier(x_train, y_train)
    nb_pred := ml.naive_bayes_predict(nb_model, x_test)
    nb_acc := ml.accuracy(y_test, nb_pred)
    println("     Accuracy: ${nb_acc:.4f}")
    
    // Model 2: Logistic Classifier
    println("   Training Logistic Regression...")
    lr_model := ml.logistic_classifier(x_train, y_train, 1000, 0.01)
    lr_pred := ml.logistic_classifier_predict(lr_model, x_test, 0.5)
    lr_acc := ml.accuracy(y_test, lr_pred)
    println("     Accuracy: ${lr_acc:.4f}")
    
    // Model 3: Random Forest (smaller for demo)
    println("   Training Random Forest...")
    rf_model := ml.random_forest_classifier(x_train, y_train, 20, 5)
    rf_pred := ml.random_forest_classifier_predict(rf_model, x_test)
    rf_acc := ml.accuracy(y_test, rf_pred)
    println("     Accuracy: ${rf_acc:.4f}")
    
    // 5. Detailed Evaluation
    println("\n5. Detailed Evaluation (Best Model)")
    
    // Choose best model
    best_pred := if rf_acc >= lr_acc && rf_acc >= nb_acc { rf_pred } 
                 else if lr_acc >= nb_acc { lr_pred } 
                 else { nb_pred }
    
    // Confusion Matrix
    cm := ml.confusion_matrix(y_test, best_pred)
    println("   Confusion Matrix:")
    println("                 Predicted")
    println("              0      1")
    println("   Actual 0   ${cm[0][0]:3}    ${cm[0][1]:3}")
    println("          1   ${cm[1][0]:3}    ${cm[1][1]:3}")
    
    // Classification Report
    println("\n   Classification Report:")
    for class in [0, 1] {
        prec := ml.precision(y_test, best_pred, class)
        rec := ml.recall(y_test, best_pred, class)
        f1 := ml.f1_score(y_test, best_pred, class)
        println("   Class ${class}: Precision=${prec:.4f}, Recall=${rec:.4f}, F1=${f1:.4f}")
    }
    
    // 6. Feature Analysis
    println("\n6. Feature Analysis")
    
    // Calculate correlation between features and target
    for feature_idx in 0..5 {
        mut feature_vals := []f64{}
        for sample in dataset.features {
            feature_vals << sample[feature_idx]
        }
        
        target_vals := dataset.target.map(f64(it))
        corr := stats.correlation(feature_vals, target_vals)
        
        feature_names := ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        println("   ${feature_names[feature_idx]:8} correlation: ${corr:+.4f}")
    }
    
    // 7. Save/Load Model (weights extraction example)
    println("\n7. Model Inspection")
    println("   Logistic Regression Coefficients:")
    feature_names := ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    for i, coef in lr_model.coefficients {
        println("     ${feature_names[i]:8}: ${coef:+.4f}")
    }
    println("   Intercept: ${lr_model.intercept:+.4f}")
    
    println("\n=== Demo Complete ===")
}
```

---

## Tips and Best Practices

### 1. Type Safety with Generics

VStats uses V's generic system for type flexibility:

```v
// Both int and f64 work with most functions
int_vec := [1, 2, 3]
f64_vec := [1.0, 2.0, 3.0]

sum_int := linalg.sum(int_vec)  // Returns int
sum_f64 := linalg.sum(f64_vec)  // Returns f64
```

### 2. Error Handling

Always handle potential errors, especially with dataset loading:

```v
dataset := utils.load_titanic() or {
    eprintln("Failed to load dataset: ${err}")
    return
}
```

### 3. Data Normalization

For best results with gradient-based methods, normalize your features:

```v
// The logistic_classifier automatically normalizes features
// For manual normalization:
mean := stats.mean(feature_column)
std := stats.standard_deviation(feature_column)
normalized := feature_column.map((it - mean) / std)
```

### 4. Train/Test Split

Always split your data to evaluate generalization:

```v
split_idx := int(f64(data.len) * 0.8)  // 80/20 split
x_train := x[0..split_idx]
x_test := x[split_idx..x.len]
```

### 5. Hyperparameter Tuning

Use grid search for finding optimal parameters:

```v
param_grid := {
    'learning_rate': [0.001, 0.01, 0.1],
    'iterations': [500.0, 1000.0, 2000.0]
}
combinations := utils.generate_param_grid(param_grid)
```

---

## Additional Resources

- **Source Code**: https://github.com/rodabt/vstats
- **API Reference**: See inline documentation in each module
- **Examples**: Check the `examples/` directory for more complete examples
- **Tests**: Review `tests/` directory for usage patterns

---

## Contributing

Contributions are welcome! Areas for improvement:

- Additional ML algorithms (K-Means, KNN, etc.)
- More probability distributions
- Advanced neural network architectures
- GPU acceleration
- Performance optimizations

---

*Happy computing with VStats!*
