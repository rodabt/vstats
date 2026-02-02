# VStats Classification Testing Guide

## Overview
This document describes the comprehensive testing suite for machine learning classification algorithms using the real Titanic dataset.

## Dataset: Titanic Survival Prediction

### Dataset Details
- **Source**: UCI Machine Learning Repository
- **Samples**: 891 passengers
- **Target**: Survived (0=Did not survive, 1=Survived)
- **Class Distribution**: 38.4% survived, 61.6% did not survive
- **Features** (5 normalized features):
  - `Pclass`: Passenger class (1, 2, 3)
  - `Age`: Age in years (missing values filled with median ~29)
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Fare`: Ticket fare

### Loading the Dataset
```v
import utils

dataset := utils.load_titanic()!
x, y := dataset.xy()
```

## Community Benchmarks (Expected Performance)

These are established benchmarks from published literature and Kaggle competitions:

| Algorithm           | Accuracy | Precision | Recall    | F1-Score  | Source     |
| ------------------- | -------- | --------- | --------- | --------- | ---------- |
| Logistic Regression | 70-78%   | 0.70-0.75 | 0.65-0.75 | 0.68-0.75 | UCI/Kaggle |
| Naive Bayes         | 70-77%   | 0.68-0.76 | 0.68-0.74 | 0.69-0.74 | UCI/Kaggle |
| Random Forest       | 77-82%   | 0.78-0.85 | 0.72-0.80 | 0.75-0.82 | UCI/Kaggle |
| SVM (RBF)           | 72-78%   | 0.72-0.78 | 0.68-0.76 | 0.70-0.77 | UCI/Kaggle |

## Test Suite: `ml/titanic_logistic_test.v`

### Test 1: Dataset Loading
```v
test__titanic_dataset_basic()
```
- Verifies 891 samples loaded
- Validates 5 features per sample
- Checks class distribution (~38% survival)

### Test 2: Logistic Regression Main Test
```v
test__logistic_regression_on_titanic()
```
- **Expected Accuracy**: ≥70% (community benchmark)
- Trains on 80% of data (712 samples)
- Tests on 20% of data (179 samples)
- Hyperparameters: 1000 iterations, lr=0.01
- **Status**: FAILING - Currently achieving only 38% accuracy

**Problem Identified**: 
- Feature scaling not applied
- Fare feature coefficient: 210.2 (too large)
- Age feature coefficient: 0.0 (not contributing)
- Model predicts almost all positive (bias towards class 1)

### Test 3: Probability Predictions
```v
test__logistic_regression_probabilities()
```
- Validates probability output in [0, 1]
- Checks distribution of predicted probabilities

### Test 4: Confusion Matrix Analysis
```v
test__logistic_regression_confusion_matrix()
```
- Builds 2x2 confusion matrix
- Reports TN, FP, FN, TP counts
- Current output: Model strongly biased towards positive class

### Test 5: Threshold Sensitivity
```v
test__logistic_regression_threshold_sensitivity()
```
- Tests prediction changes at thresholds: 0.3, 0.5, 0.7
- Validates lower thresholds produce more positive predictions

### Test 6: Coefficient Analysis
```v
test__logistic_regression_coefficients()
```
- Displays learned coefficients for each feature
- **Issue**: Shows need for feature normalization
  - Fare: 210.2 (dominates)
  - Pclass: -0.41 (reasonable)
  - SibSp: 0.12 (reasonable)
  - Parch: 0.03 (reasonable)
  - Age: 0.0 (not used)

## Running Tests

```bash
# Run Titanic logistic regression tests
v test ml/titanic_logistic_test.v

# Run all ML tests
v test ml/
```

## Current Issues & Recommendations

### Issue 1: Feature Scaling
The logistic regression implementation does not normalize/standardize features, leading to:
- Fare (range 0-512) dominates the decision
- Pclass (range 1-3) and other features have minimal impact

**Recommendation**: Implement feature normalization (StandardScaler) before training

### Issue 2: Missing Value Handling
Age feature has ~177 missing values. Currently filled with median (29.0):
- May lose predictive information
- Consider: mean imputation, model-based imputation, or separate missing indicator

**Recommendation**: Create separate feature for "age_missing" (binary) + impute

### Issue 3: Hyperparameter Tuning
Current hyperparameters (1000 iterations, lr=0.01) may not be optimal:
- No learning rate scheduling
- Fixed iteration count
- No validation for convergence

**Recommendation**: Implement:
- Cross-validation
- Learning rate scheduling
- Early stopping

## Expected Improvements

After addressing the issues above, expected results:

```
=== Logistic Regression on Titanic (Fixed) ===
Test set size: 179
Accuracy:  0.75 (target: 0.70-0.78)
Precision: 0.75
Recall:    0.72
F1 Score:  0.73
```

## References

- [Titanic Dataset - Kaggle](https://www.kaggle.com/c/titanic)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Titanic)
- Titanic Survivor Prediction Benchmark Results:
  - Various published accuracy comparisons: 70-82% depending on features and preprocessing
