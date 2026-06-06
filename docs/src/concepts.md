# Core Concepts

## Linear Algebra Basics

Linear algebra provides the mathematical foundation for machine learning. Vectors represent
points in n-dimensional space, while matrices represent linear transformations.

### Vectors

A vector is an ordered collection of numbers. Key operations include:

- **Addition:** Element-wise sum of two vectors
- **Dot Product:** a · b = sum of element-wise products
- **Magnitude:** Length of vector (L2 norm)

### Matrices

A matrix is a 2D array with rows and columns. Key operations:

- **Multiplication:** Row-column dot products
- **Transpose:** Swap rows and columns
- **Identity:** Matrix that does nothing (1s on diagonal)

## Statistics Fundamentals

### Descriptive Statistics

- **Mean:** Arithmetic average of all values
- **Median:** Middle value (50th percentile)
- **Variance:** Average squared deviation from mean (vstats uses *sample* variance, ÷n-1)
- **Standard Deviation:** Square root of variance

### Hypothesis Testing

Statistical tests determine if observed differences are significant:

- **t-test:** Compare means of two groups
- **ANOVA:** Compare means of 3+ groups
- **p-value:** Probability of observing results by chance under the null hypothesis

## Machine Learning Essentials

### Supervised vs Unsupervised

- **Supervised:** Learn from labeled data (regression, classification)
- **Unsupervised:** Find patterns in unlabeled data (clustering)

### Model Evaluation

- **MSE:** Mean squared error (regression)
- **Accuracy:** Correct predictions / Total
- **R²:** Proportion of variance explained

## Growth Metrics

| Metric | Formula |
|--------|---------|
| ARPA | Revenue / Accounts |
| CAC | Acquisition Spend / New Customers |
| LTV | ARPU × Customer Lifespan |
| Churn Rate | Customers Lost / Total Customers |
| NRR | (MRR_end − Churn_MRR) / MRR_start |

## Experimentation

### Statistical Power and Sample Size

Before running an experiment, you must decide how many observations you need. Four numbers
determine this:

- **Alpha (α):** The false positive rate — how often you'll declare a winner when there's no
  real effect. Typically 0.05 (5%).
- **Power (1 − β):** The probability of detecting a real effect when one exists. Typically
  0.80 (80%). Setting power too low means your experiment may end with no result even when
  the treatment works.
- **Minimum Detectable Effect (MDE):** The smallest improvement worth detecting. A tighter
  MDE requires more data. Be honest about what effect size would change a business decision
  — don't chase tiny effects that aren't actionable.
- **Baseline variance:** Higher variance in your metric means more noise, requiring more
  data to see signal. Use historical data to estimate this.

The formula for continuous metrics: `n = 2 × ((z_α/2 + z_β) × σ / Δ)²` per group, where
σ is the standard deviation and Δ is the MDE.

### Frequentist vs. Bayesian

Both are valid frameworks with different outputs:

- **Frequentist (p-values):** Answers "if there were no effect, how unlikely is this data?"
  You reject the null hypothesis when p < α. The result is binary: significant or not. Does
  not give the probability that B is better.
- **Bayesian (posteriors):** Answers "given this data, what is the probability that B beats
  A?" More intuitive for business decisions. Requires specifying a prior — use Beta(1,1)
  (uniform) when you have no prior knowledge.

Bayesian is preferable when you need to communicate results to non-statisticians ("94%
chance B is better"), when you want to incorporate prior knowledge, or when you need to
make a decision before collecting enough data for a frequentist test.

### The Peeking Problem

A common mistake: checking an experiment's p-value every day and stopping when it first
crosses 0.05. This *inflates the false positive rate dramatically* — you may declare
significance by chance, especially early in an experiment.

Two solutions:

- **Sequential testing (SPRT):** Uses Wald's likelihood ratio test with boundaries that
  account for repeated looks. You can check at any time; the false positive rate stays at α.
  Use `experiment.sprt_test`.
- **Pre-commit to a fixed end date:** Calculate your sample size, run the experiment until
  you have it, then do one final test. Never look at results before the end.

### Effect Sizes

A p-value tells you whether an effect exists; effect size tells you how big it is. Always
report both.

- **Cohen's d:** For continuous metrics. d = (mean_B − mean_A) / pooled_std. Small: 0.2,
  Medium: 0.5, Large: 0.8.
- **Absolute lift:** For proportions. "The signup rate increased by 1.2 percentage points
  (from 5.0% to 6.2%)." Always prefer absolute over relative for communication.
- **Relative lift:** Percentage change relative to control. Useful for comparing across
  metrics with different baselines, but can be misleading for small baselines.

### Variance Reduction with CUPED

CUPED (Controlled-experiment Using Pre-Experiment Data) exploits the correlation between a
user's pre-experiment behavior and their in-experiment behavior. If last week's revenue
predicts this week's revenue, you can "remove" that predictable component from both groups,
reducing noise without introducing bias. A correlation of ρ = 0.7 reduces required sample
size by approximately 1 − ρ² = 51%.
