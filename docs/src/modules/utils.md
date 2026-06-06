# utils

`import vstats.utils`

Built-in datasets, feature scaling, classification metrics, and utility functions.

> **vs Python:** replaces `sklearn.datasets`, `sklearn.preprocessing.StandardScaler`,
> and `sklearn.metrics` for the common binary classification workflow.

## Datasets

All classification datasets return `Dataset`; regression datasets return `RegressionDataset`.

```v
load_iris() !Dataset
load_wine() !Dataset
load_breast_cancer() !Dataset
load_titanic() !Dataset
load_boston_housing() !RegressionDataset
load_linear_regression() RegressionDataset

// Dataset methods
dataset.xy() ([][]f64, []int)
dataset.train_test_split(test_ratio f64) (Dataset, Dataset)
dataset.summary() string
```

## Feature Scaling

> **vs Python:** replaces `sklearn.preprocessing.StandardScaler`. Fit on train,
> apply to test — `normalize_features` returns the fit parameters for reuse.

```v
normalize_features(x [][]f64) ([][]f64, []f64, []f64)   // returns (normalized, mean, std)
apply_normalization(x [][]f64, mean []f64, std []f64) [][]f64
standardize(x []f64) []f64
min_max_scale(x []f64) []f64
```

## Metrics

> **vs Python:** replaces `sklearn.metrics.classification_report` and `roc_auc_score`.

```v
binary_classification_metrics(y_true []int, y_pred []int) map[string]f64
// keys: "accuracy", "precision", "recall", "f1_score"

build_confusion_matrix(y_true []int, y_pred []int) ConfusionMatrix
// ConfusionMatrix.summary() string

roc_curve(y_true []int, y_pred_proba []f64) ROCResult
// ROCResult.auc f64
```

## Activation Functions

```v
sigmoid[T](x T) T
relu[T](x T) T
tanh_activation[T](x T) T
```

## Hyperparameter Search

```v
generate_param_grid(params map[string][]f64) []map[string]f64
// Returns cartesian product of all parameter combinations
```

## See Also

- [churn-prediction example](../examples.html#churn-prediction)
