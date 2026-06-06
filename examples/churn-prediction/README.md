# Customer Churn Prediction

Full binary classification pipeline: load data, normalize, train logistic regression
and random forest, evaluate with precision/recall/F1 and AUC.

**Verify (do not run — slow):** `v -check examples/churn-prediction/main.v`

**Modules used:** `vstats.ml`, `vstats.utils`

**Python equivalent:** `sklearn` pipeline with `LogisticRegression`,
`RandomForestClassifier`, `classification_report`, `roc_auc_score`.
