# ml

`import vstats.ml`

Regression, classification, and clustering. Pair with `utils` for datasets and
normalization.

> **vs Python:** replaces `sklearn.linear_model`, `sklearn.ensemble`,
> `sklearn.svm`, `sklearn.cluster` — with no install required.

## Regression

```v
linear_regression[T](x [][]T, y []T) LinearModel[T]
linear_predict[T](model LinearModel[T], x [][]T) []T

logistic_regression[T](x [][]T, y []T, iterations int, learning_rate T) LogisticModel[T]
logistic_predict[T](model LogisticModel[T], x [][]T, threshold T) []T
logistic_predict_proba[T](model LogisticModel[T], x [][]T) []T

// Regression metrics
mse[T](y_true []T, y_pred []T) f64
rmse[T](y_true []T, y_pred []T) f64
mae[T](y_true []T, y_pred []T) f64
r_squared[T](y_true []T, y_pred []T) f64
```

## Classification

> **vs Python:** `random_forest_classifier` + `random_forest_predict` mirrors
> `sklearn.ensemble.RandomForestClassifier().fit().predict()`.
> `y` must be `[]int` for Naive Bayes and Random Forest; `[]f64` for logistic.

```v
// Naive Bayes
naive_bayes_classifier(x [][]f64, y []int) NaiveBayesClassifier
naive_bayes_predict(model NaiveBayesClassifier, x [][]f64) []int

// SVM
svm_classifier(x [][]f64, y []f64, learning_rate f64, iterations int, gamma f64, kernel string) SVMClassifier
svm_predict(model SVMClassifier, x [][]f64) []int

// Random Forest
random_forest_classifier(x [][]f64, y []int, num_trees int, max_depth int) RandomForestClassifier
random_forest_predict(model RandomForestClassifier, x [][]f64) []int
random_forest_classifier_predict_proba(model RandomForestClassifier, x [][]f64) []f64

// Classification metrics
accuracy(y_true []int, y_pred []int) f64
precision(y_true []int, y_pred []int, positive_class int) f64
recall(y_true []int, y_pred []int, positive_class int) f64
f1_score(y_true []int, y_pred []int, positive_class int) f64
```

## Clustering

> **vs Python:** replaces `sklearn.cluster.KMeans`, `DBSCAN`,
> `AgglomerativeClustering`.

```v
kmeans(data [][]f64, k int, max_iterations int) KMeansModel
kmeans_predict(model KMeansModel, data [][]f64) []int
kmeans_inertia(model KMeansModel, data [][]f64) f64
silhouette_coefficient(data [][]f64, labels []int) f64   // quality score −1..1

hierarchical_clustering(data [][]f64, num_clusters int) HierarchicalClustering
dbscan(data [][]f64, eps f64, min_points int) []int      // −1 = noise
```

## See Also

- [churn-prediction example](../examples.html#churn-prediction)
