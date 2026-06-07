# Titanic Survival — Multi-Classifier Comparison

Trains three classifiers (logistic regression, Naive Bayes, random forest) on the
same train/test split and compares accuracy, precision, recall, and F1. Shows how
to evaluate classifier choice rather than assuming the best model upfront.

**Verify (do not run — slow):** `v -check examples/titanic-survival/main.v`

**Modules used:** `vstats.ml`, `vstats.utils`

**Python equivalent:** `sklearn` loop over `[LogisticRegression(), GaussianNB(),
RandomForestClassifier()]` with `classification_report` per model.
