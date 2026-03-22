# vstats - Statistical & Machine Learning Library for V

Complete module documentation for the vstats library.

## Module Structure

### Core Modules

#### `linalg/` - Linear Algebra
Core vector and matrix operations.

**Files:**
- `vectors.v` - Vector operations (add, subtract, dot product, distance, magnitude)
- `matrix.v` - Matrix operations
- `util.v` - Utility functions
- Tests: `vectors_test.v`, `matrix_test.v`

**Key Functions:**
- `add(v, w)` - Vector addition
- `subtract(v, w)` - Vector subtraction
- `dot(v, w)` - Dot product
- `magnitude(v)` - Vector magnitude
- `distance(v, w)` - Euclidean distance
- `sum_of_squares(v)` - Sum of squares
- `flatten(m)` - Flatten matrix to vector

---

#### `stats/` - Descriptive Statistics & Hypothesis Testing
Statistical measures, aggregations, and advanced statistical tests.

**Files:**
- `descriptive.v` - Descriptive statistics and hypothesis tests
- `descriptive_test.v` - Unit tests
- `advanced_tests_test.v` - Advanced statistics test suite

**Descriptive Functions:**
- `mean(x)` - Arithmetic mean
- `median(x)` - Median value
- `mode(x)` - Mode (most frequent values)
- `variance(x)` - Sample variance
- `standard_deviation(x)` - Sample standard deviation
- `correlation(x, y)` - Pearson correlation
- `covariance(x, y)` - Covariance between two variables
- `quantile(x, p)` - Quantile at probability p
- `interquartile_range(x)` - IQR (Q3 - Q1)
- `skewness(x)` - Distribution asymmetry (3rd moment)
- `kurtosis(x)` - Distribution tailedness (4th moment, excess)

**Advanced Statistical Tests:**
- `anova_one_way(groups)` - One-way ANOVA F-test for comparing group means
- `confidence_interval_mean(x, confidence_level)` - CI for population mean
- `cohens_d(group1, group2)` - Cohen's d effect size for mean differences
- `cramers_v(contingency)` - Cramér's V effect size for categorical association

---

#### `prob/` - Probability Distributions
Probability density functions (PDF), cumulative distribution functions (CDF), and distribution utilities.

**Files:**
- `distributions.v` - All probability distributions

**Continuous Distributions:**
- Normal: `normal_cdf()`, `inverse_normal_cdf()`
- Exponential: `exponential_pdf()`, `exponential_cdf()`
- Uniform: `uniform_pdf()`, `uniform_cdf()`
- Gamma: `gamma_pdf()`
- Chi-squared: `chi_squared_pdf()`
- Student's t: `students_t_pdf()`
- F-Distribution: `f_distribution_pdf()`
- Beta: `beta_pdf()`

**Discrete Distributions:**
- Bernoulli: `bernoulli_pdf()`, `bernoulli_cdf()`
- Binomial: `binomial_pdf()`
- Poisson: `poisson_pdf()`, `poisson_cdf()`
- Negative Binomial: `negative_binomial_pdf()`, `negative_binomial_cdf()`
- Multinomial: `multinomial_pdf()`

**Utilities:**
- `expectation(x, p)` - Expected value
- `beta_function(x, y)` - Beta function

---

#### `optim/` - Optimization
Numerical optimization algorithms for finding gradients and performing gradient descent.

**Files:**
- `algorithms.v` - Optimization algorithms

**Key Functions:**
- `difference_quotient(f, x, h)` - Numerical derivative
- `partial_difference_quotient(f, v, i, h)` - Partial derivative
- `gradient(f, v, h)` - Compute full gradient vector
- `gradient_step(v, gradient, step_size)` - Update parameters via gradient descent
- `sum_of_squares_gradient(v)` - Gradient of sum of squares function

---

#### `symbol/` - Symbolic Computation
Symbolic algebra and expression manipulation.

**Files:**
- `symbol.v` - Symbolic operations

---

#### `utils/` - Utilities
General utility functions, evaluation metrics, and training helpers.

**Files:**
- `utils.v` - Basic utility functions
- `metrics.v` - Classification/regression metrics and training utilities
- `datasets.v` - Dataset loading and splitting
- Test files: `*_test.v`

**Basic Functions:**
- `factorial(n)` - Factorial computation
- `combinations(n, k)` - Binomial coefficient
- `range(n)` - Generate range of integers

**Evaluation Metrics (metrics.v):**
- `build_confusion_matrix(y_true, y_pred)` - Build confusion matrix structure
- `(ConfusionMatrix).accuracy()` - Accuracy metric
- `(ConfusionMatrix).precision()` - Precision metric
- `(ConfusionMatrix).recall()` - Recall/sensitivity metric
- `(ConfusionMatrix).specificity()` - Specificity metric
- `(ConfusionMatrix).f1_score()` - F1 score
- `(ConfusionMatrix).false_positive_rate()` - FPR metric
- `(ConfusionMatrix).summary()` - Formatted summary of all metrics
- `roc_curve(y_true, y_proba)` - Calculate ROC curve and AUC
- `(ROC_Curve).auc_value()` - Extract AUC value

**Utility Metrics (metrics.v):**
- `binary_classification_metrics(y_true, y_pred)` - All binary metrics in one call
- `regression_metrics(y_true, y_pred)` - All regression metrics (MSE, RMSE, MAE, R²)
- `generate_param_grid(param_ranges)` - Generate parameter combinations for grid search

**Training Utilities (metrics.v):**
- `(TrainingProgress).format_log()` - Format progress for logging
- `early_stopping(losses, patience)` - Check early stopping criterion
- `decay_learning_rate(initial_lr, epoch, decay_rate)` - Exponential LR scheduler

**Dataset Functions (datasets.v):**
- `load_iris()` - Iris classification dataset (150 samples, 4 features, 3 classes)
- `load_wine()` - Wine classification dataset (178 samples, 13 features→4, 3 classes)
- `load_breast_cancer()` - Breast cancer classification dataset
- `load_boston_housing()` - Boston housing regression dataset (506 samples, 13 features→3)
- `load_linear_regression()` - Synthetic linear regression data
- `(Dataset).summary()` - Dataset summary statistics
- `(Dataset).train_test_split(test_size)` - Split dataset
- `(Dataset).xy()` - Get features and targets as separate arrays
- Similar methods for `RegressionDataset`

---

### Machine Learning Modules (NEW)

#### `ml/` - Machine Learning
Supervised and unsupervised learning algorithms.

**Files:**

##### `regression.v`
Regression models and evaluation metrics.

**Models:**
- `LinearModel` - Linear regression model
- `LogisticModel` - Logistic regression model

**Key Functions:**
- `linear_regression(x, y)` - Fit OLS linear regression
- `linear_predict(model, x)` - Predict with linear model
- `logistic_regression(x, y, iterations, lr)` - Fit logistic regression
- `logistic_predict(model, x, threshold)` - Binary classification
- `logistic_predict_proba(model, x)` - Prediction probabilities

**Evaluation Metrics:**
- `mse(y_true, y_pred)` - Mean Squared Error
- `rmse(y_true, y_pred)` - Root Mean Squared Error
- `mae(y_true, y_pred)` - Mean Absolute Error
- `r_squared(y_true, y_pred)` - R² coefficient of determination

---

##### `clustering.v`
Unsupervised clustering algorithms.

**Models:**
- `KMeansModel` - K-means cluster model
- `HierarchicalClustering` - Hierarchical cluster result

**Key Functions:**
- `kmeans(data, k, max_iterations)` - K-means clustering
- `kmeans_predict(model, data)` - Cluster assignment for new data
- `kmeans_inertia(model, data)` - Inertia (sum of squared distances)
- `silhouette_coefficient(data, labels)` - Cluster quality measure

- `hierarchical_clustering(data, num_clusters)` - Agglomerative clustering (single linkage)

- `dbscan(data, eps, min_points)` - Density-based clustering
  - Returns labels (0 = noise, >0 = cluster ID)

---

### Neural Network Module (NEW)

#### `nn/` - Neural Networks
Deep learning components for building neural networks.

**Files:**

##### `layers.v`
Neural network layers and activation functions.

**Layer Types:**
- `DenseLayer` - Fully connected layer
- `ActivationLayer` - Non-linear activation
- `BatchNormLayer` - Batch normalization

**Activation Functions:**
- `relu(x)` - ReLU activation
- `sigmoid(x)` - Sigmoid activation
- `tanh(x)` - Hyperbolic tangent
- `softmax(x)` - Softmax (multi-class output)

**Key Functions:**
- `dense_layer(input_size, output_size)` - Create dense layer
- `(layer).forward(input)` - Forward pass
- `(layer).backward(grad, input, lr)` - Backward pass (backpropagation)
- `activation_layer(fn_name)` - Create activation layer
- `dropout(input, rate)` - Dropout regularization
- `flatten(data)` - Reshape 2D to 1D
- `reshape(data, rows, cols)` - Reshape 1D to 2D

**Convolution & Pooling:**
- `conv1d(input, kernel, stride)` - 1D convolution
- `max_pool1d(input, pool_size, stride)` - Max pooling
- `avg_pool1d(input, pool_size, stride)` - Average pooling

---

##### `loss.v`
Loss functions for training neural networks.

**Key Functions:**
- `mse_loss(y_true, y_pred)` - Mean Squared Error
- `mae_loss(y_true, y_pred)` - Mean Absolute Error
- `binary_crossentropy_loss(y_true, y_pred)` - Binary classification
- `categorical_crossentropy_loss(y_true, y_pred)` - Multi-class classification
- `sparse_categorical_crossentropy_loss(y_true, y_pred)` - Multi-class (integer labels)
- `hinge_loss(y_true, y_pred)` - SVM-like loss
- `huber_loss(y_true, y_pred, delta)` - Robust to outliers
- `kl_divergence_loss(y_true, y_pred)` - KL divergence
- `cosine_similarity_loss(y_true, y_pred)` - Cosine distance
- `contrastive_loss(y_true, distance, margin)` - Siamese networks
- `triplet_loss(anchor, positive, negative, margin)` - Metric learning

**Gradient Functions:**
- `mse_loss_gradient()` - MSE gradient
- `mae_loss_gradient()` - MAE gradient
- `binary_crossentropy_loss_gradient()` - BCE gradient

---

##### `network.v`
High-level neural network construction and training.

**Main Class:**
- `NeuralNetwork` - Sequential neural network

**Key Functions:**
- `sequential(layer_sizes, activation_fn)` - Create sequential network
- `(nn).forward(input)` - Forward pass
- `(nn).backward(grad, input, lr)` - Backward pass
- `(nn).train(x_train, y_train, config)` - Train network
- `(nn).predict(x)` - Predict on batch
- `(nn).predict_single(x)` - Predict on single sample
- `(nn).evaluate(x_test, y_test)` - Evaluate on test set
- `(nn).get_weights()` - Extract weights
- `(nn).get_biases()` - Extract biases
- `(nn).set_weights(weights)` - Set weights

**Configuration:**
- `TrainingConfig` - Training parameters (learning_rate, epochs, batch_size, verbose)
- `default_training_config()` - Default config
- `training_config(lr, epochs, batch_size)` - Custom config

---

### Experimentation Module

#### `experiment/` - Causal Inference & Experimentation
Industry-standard experimentation workflows: A/B testing, propensity score matching, and difference-in-differences.

**Dependencies:** `ml`, `hypothesis`, `stats`, `prob`, `linalg`

**Files:**

##### `abtest.v`
A/B testing, power analysis, and CUPED variance reduction.

**Structs:**
- `ABTestConfig` — `alpha` (default 0.05), `equal_variance`
- `ABTestResult` — means, SDs, lift, Cohen's d, t-stat, df, p-value, CI, significance flag
- `PowerAnalysisResult` — `n_per_group`, `power`, `alpha`, `effect_size`
- `CUPEDResult` — `theta`, `variance_reduction`, `adjusted_result`

**Key Functions:**
- `abtest(control, treatment, cfg)` — Welch's t-test with effect size, relative lift, and CI
- `power_analysis(effect_size, alpha, power)` — Required n per group via normal approximation
- `cuped_test(y_ctrl, y_treat, pre_ctrl, pre_treat, cfg)` — CUPED-adjusted A/B test using pre-experiment covariates

---

##### `sample_size.v`
Sample size calculation for experiments before data collection.

**Structs:**
- `SampleSizeResult` — `n_per_group`, `total_n`, `alpha`, `power`, `mde`, `baseline`, `effect_size`, `baseline_std`, `method`

**Key Functions:**
- `sample_size_proportions(baseline_rate, mde, alpha, power)` — n per group for conversion/proportion metrics; `mde` is absolute rate change (e.g. 0.01 for +1pp)
- `sample_size_means(baseline_mean, baseline_std, mde_absolute, alpha, power)` — n per group for continuous metrics; effect size field = Cohen's d

---

##### `proportion_ztest.v`
Two-proportion z-test for comparing conversion rates.

**Structs:**
- `ProportionTestConfig` — `alpha` (default 0.05); `@[params]`
- `ProportionTestResult` — `rate_a`, `rate_b`, `diff`, `relative_lift`, `z_statistic`, `p_value`, `significant`, `ci_lower`, `ci_upper`, `pooled_se`, `n_a`, `n_b`

**Key Functions:**
- `proportion_test(successes_a, n_a, successes_b, n_b, cfg)` — Pooled z-test under H₀; CI uses unpooled SE and `alpha` from config

---

##### `sequential.v`
Sequential Probability Ratio Test (SPRT) for safe interim analysis.

**Types:**
- `SPRTDecision` — enum: `continue_testing`, `reject_null`, `accept_null`
- `SPRTConfig` — `alpha` (0.05), `beta` (0.20), `mde` (required, no default); NOT `@[params]`
- `SPRTResult` — `log_likelihood_ratio`, `decision`, `upper_boundary`, `lower_boundary`, `rate_a`, `rate_b`, `n_a`, `n_b`

**Key Functions:**
- `sprt_test(successes_a, n_a, successes_b, n_b, cfg)` — One-shot Bernoulli SPRT over cumulative totals; call repeatedly at each interim check

---

##### `bayesian.v`
Bayesian A/B test using Beta-Binomial conjugate model.

**Structs:**
- `BayesianConfig` — `alpha_prior` (1.0), `beta_prior` (1.0), `n_samples` (10000); `@[params]`
- `BayesianResult` — `posterior_mean_a/b`, `prob_b_beats_a`, `expected_loss_a/b`, `ci_lower/upper_a/b`, `successes_a/b`, `n_a/b`

**Key Functions:**
- `bayesian_ab_test(successes_a, n_a, successes_b, n_b, cfg)` — Beta posteriors via Marsaglia-Tsang sampler; Monte Carlo estimates for P(B>A), expected loss, and 95% credible intervals

---

##### `psm.v`
Propensity score matching and covariate balance checking.

**Structs:**
- `PropensityModel` — fitted logistic model, scores, treatment vector
- `PropensityConfig` — `iterations`, `learning_rate`, `trim`
- `MatchingConfig` — `caliper`, `replacement`
- `MatchedPair` — `treated_idx`, `control_idx`, `ps_distance`
- `MatchingResult` — `pairs`, matched/unmatched counts, average distance
- `BalanceResult` — SMDs before/after matching, `mean_abs_smd_*`, `balanced` flag
- `ATEResult` — `ate`, `se`, CI, t-stat, p-value, group sizes

**Key Functions:**
- `estimate_propensity_scores(x, treatment, cfg)` — Logistic regression for p(T=1|X); optional common-support trimming
- `match_nearest_neighbor(model, cfg)` — Greedy O(n_T × n_C) nearest-neighbor matching
- `check_balance(x, treatment, result)` — Standardised mean differences before and after matching
- `ate_matched(y, treatment, result)` — ATE from matched pairs with two-sample t-test

---

##### `did.v`
Difference-in-Differences estimation, regression DiD, parallel trends testing, and event studies.

**Structs:**
- `DiDConfig` — `alpha`
- `DiDResult` — DiD effect, SE, t-stat, p-value, CI, group changes, cell sizes
- `DiDRegressionResult` — OLS interaction coefficient, SE, CI, R², n
- `ParallelTrendsResult` — slopes per group, slope difference, t-stat, p-value, `parallel_trends_hold`
- `EventStudyResult` — `relative_times`, `effects`, `std_errors`, `t_statistics`, `p_values`, CIs

**Key Functions:**
- `did_2x2(y_treat_pre, y_treat_post, y_ctrl_pre, y_ctrl_post, cfg)` — Classic 2×2 DiD with delta-method SE
- `did_regression(y, x, group, time, cfg)` — OLS with treat×post interaction; OLS standard errors via (X'X)⁻¹
- `test_parallel_trends(y_treated_pre, y_control_pre, time_pre, cfg)` — Tests slope equality in pre-period via pooled OLS
- `event_study(y, group, relative_time, cfg)` — Period-by-period DiD using period -1 as reference

---

### Hypothesis Testing Module (NEW)

#### `hypothesis/` - Hypothesis Testing
Statistical tests and hypothesis testing functions.

**Files:**
- `tests.v` - Statistical hypothesis tests

**Parametric Tests:**
- `t_test_one_sample(x, mu, params)` - One-sample t-test
- `t_test_two_sample(x, y, params)` - Two-sample t-test (equal variances)
- `correlation_test(x, y, params)` - Test significance of correlation
- `chi_squared_test(observed, expected)` - Goodness of fit test

**Non-Parametric Tests:**
- `wilcoxon_signed_rank_test(x, y)` - Paired samples test
- `mann_whitney_u_test(x, y)` - Independent samples test
- `shapiro_wilk_test(x)` - Normality test

**Return Values:**
All tests return `(test_statistic, p_value)` tuple.

**Parameters:**
- `TestParams` struct with `alpha` (significance level, default 0.05)

---

### Growth & Product Analytics Module (NEW)

#### `growth/` - Growth & Product Analytics
Industry-standard product and marketing metrics, funnel analysis, cohort analysis, and attribution modeling.

**Files:**

##### `metrics.v`
Revenue, customer, and retention metrics.

**Revenue Metrics:**
- `arpa(revenue, accounts)` - Average Revenue Per Account
- `arpu(revenue, users)` - Average Revenue Per User
- `monthly_recurring_revenue(plan_revenues)` - MRR calculation
- `annual_recurring_revenue(mrr)` - ARR from MRR

**Customer Metrics:**
- `cac(acquisition_spend, new_customers)` - Customer Acquisition Cost
- `ltv(revenue, users, lifespan)` - Lifetime Value
- `ltv_cac_ratio(...)` - LTV:CAC ratio (healthy: 3:1+)
- `payback_period(cac, monthly_arpu)` - Payback period in months
- `magic_number(net_new_arr, gross_margin, sales_marketing_spend)` - SaaS efficiency

**Retention Metrics:**
- `churn_rate(customers_lost, total_customers)` - Customer churn rate
- `retention_rate(customers_lost, total_customers)` - 1 - Churn Rate
- `net_revenue_retention(mrr_start, mrr_end, churn_mrr, expansion_mrr)` - NRR
- `gross_revenue_retention(mrr_start, churn_mrr)` - GRR

**Financial Metrics:**
- `burn_rate(starting_cash, ending_cash, months)` - Monthly burn rate
- `runway_months(current_cash, monthly_burn)` - Months of runway

##### `funnel.v`
Conversion funnel analysis and optimization.

**Structs:**
- `FunnelStage` — name, users, conversions, dropouts
- `FunnelResult` — stages, conversion_rate, total_conversion
- `FunnelConversion` — from/to, rate, drop_off_rate

**Key Functions:**
- `create_funnel(stage_names, stage_users)` - Create funnel from stage data
- `stage_conversion_rate(from, to)` - Conversion between stages
- `(FunnelResult).get_conversions()` - Detailed conversion data
- `(FunnelResult).highest_drop_off()` - Stage with most leakage
- `funnel_leakage(funnel)` - Users lost at each stage
- `projected_conversions(funnel, additional_users)` - Project with more traffic
- `segment_funnel(segment_data)` - Compare funnels across segments

##### `cohort.v`
Cohort analysis and retention matrix computation.

**Structs:**
- `CohortPeriod` — period_index, cohort_size, retained, revenue, retention
- `Cohort` — name, periods
- `CohortAnalysis` — cohorts, retention_matrix, avg_retention, ltv_by_period

**Key Functions:**
- `create_cohort_analysis(cohort_names, initial_sizes, retention_data)` - Build cohort analysis
- `(CohortAnalysis).retention_at_period(cohort, period)` - Retention at specific point
- `(CohortAnalysis).avg_retention_at_period(period)` - Average across cohorts
- `(CohortAnalysis).churn_by_period()` - Monthly churn rates
- `(CohortAnalysis).compare_cohorts(name_a, name_b)` - Compare two cohorts
- `(CohortAnalysis).ltv_projection(periods, avg_revenue)` - Project LTV

##### `attribution.v`
Marketing channel attribution modeling.

**Structs:**
- `AttributionResult` — channel, conversions, revenue, attribution_score

**Attribution Models:**
- `first_touch_attributes(channels, conversions, revenue)` - 100% to first touch
- `last_touch_attributes(channels, conversions, revenue)` - 100% to last touch
- `linear_attributes(touchpoints, conversions, revenue)` - Equal credit
- `time_decay_attributes(touchpoints, days, conversions, revenue, half_life)` - Recent bias
- `position_based_attributes(touchpoints, conversions, revenue)` - 40/20/40 split

**Channel Analytics:**
- `channel_roi(attribution_results, channel_costs)` - ROI per channel
- `optimal_channel_mix(channel_performance, total_budget)` - Budget allocation
- `roas(revenue, ad_spend)` - Return on Ad Spend
- `blended_roas(total_revenue, total_ad_spend)` - Blended ROAS

---

## Usage Examples

### Linear Regression
```v
import ml

// Sample data: 5 samples, 2 features
x := [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
y := [3.0, 5.0, 7.0, 9.0, 11.0]

// Fit model
model := ml.linear_regression(x, y)

// Predict
predictions := ml.linear_predict(model, x)

// Evaluate
mse_score := ml.mse(y, predictions)
```

### K-Means Clustering
```v
import ml

data := [[1.0, 1.0], [1.5, 1.5], [10.0, 10.0], [10.5, 10.5]]
model := ml.kmeans(data, 2, 100)

// Evaluate clustering quality
silhouette := ml.silhouette_coefficient(data, model.labels)
```

### Neural Network
```v
import nn

// Create network: input(10) -> hidden(5) -> output(1)
mut network := nn.sequential([10, 5, 1], 'relu')

// Prepare data
x_train := [...] // 100 samples, 10 features each
y_train := [...] // 100 targets

// Train
config := nn.training_config(0.01, 100, 32)
network.train(x_train, y_train, config)

// Predict
predictions := network.predict(x_test)
```

### Hypothesis Testing
```v
import hypothesis

// One-sample t-test
data := [1.0, 2.0, 3.0, 4.0, 5.0]
t_stat, p_val := hypothesis.t_test_one_sample(data, 3.0, hypothesis.TestParams{})

if p_val < 0.05 {
    println("Reject null hypothesis")
}
```

---

## Dependencies

- **Built-in V modules**: `math`, `arrays`, `rand`
- **Cross-module dependencies**:
  - `ml` depends on `linalg`, `stats`
  - `nn` depends on `linalg`, `math`
  - `hypothesis` depends on `stats`, `prob`
  - `prob` depends on `linalg`, `math`, `utils`
  - `experiment` depends on `ml`, `hypothesis`, `stats`, `prob`, `linalg`
  - `growth` depends on `math` (standalone module)

---

## Module Statistics

| Module     | Files | Functions | Purpose                                     |
| ---------- | ----- | --------- | ------------------------------------------- |
| linalg     | 4     | 20+       | Vector/matrix operations                    |
| stats      | 3     | 18+       | Descriptive & advanced statistics           |
| prob       | 1     | 20+       | Probability distributions                   |
| optim      | 1     | 5+        | Optimization algorithms                     |
| ml         | 3     | 25+       | Machine learning algorithms                 |
| nn         | 3     | 40+       | Neural networks & layers                    |
| hypothesis | 1     | 7+        | Statistical hypothesis tests                |
| experiment | 7     | 20+       | A/B testing, sample size, proportion z-test, SPRT, Bayesian, PSM, DiD |
| growth     | 4     | 30+       | Growth metrics, funnel, cohort, attribution |
| symbol     | 1     | ?         | Symbolic computation                        |
| utils      | 5     | 35+       | Metrics, utilities, datasets                |

---

## Architecture Notes

1. **Layered Design**: Each module has clear dependencies, with lower-level modules (linalg, utils) supporting higher-level ones (ml, nn, experiment)

2. **Functional Style**: Emphasizes pure functions with minimal state mutation

3. **Type Safety**: Uses V's strong type system with structs for models and configuration

4. **Separation of Concerns**: 
   - Algorithms in one file, loss functions in another
   - Models and training in separate files
   - Tests in dedicated test files

5. **Numerical Stability**: Includes safeguards (e.g., clipping in log operations, max subtraction in softmax)

6. **Extensibility**: Modular activation functions, loss functions, and optimization algorithms can be easily extended
