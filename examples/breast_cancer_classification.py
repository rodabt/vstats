#!/usr/bin/env python3
"""
Breast Cancer Classification Example - Python
Binary classification using scikit-learn's breast cancer dataset

Uses scikit-learn for comparison with VStats library
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, roc_auc_score
)
import numpy as np

def print_section(title):
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    print()

def print_subsection(title):
    print(title)
    print("-" * 80)

def main():
    print_section("Python: Breast Cancer Classification Example")
    
    # 1. Load Breast Cancer dataset
    print_subsection("1. LOADING BREAST CANCER DATASET")
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    print(f"Dataset: Breast Cancer")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Feature names (first 10): {list(cancer.feature_names[:10])}")
    print(f"Target names: {list(cancer.target_names)}")
    print(f"Class distribution: Malignant={sum(y==0)}, Benign={sum(y==1)}")
    print()
    
    # 2. Split data
    print_subsection("2. SPLITTING DATA (80% train, 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print()
    
    # 3. Data normalization (important for logistic regression)
    print_subsection("3. NORMALIZING FEATURES")
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_norm = (X_train - X_train_mean) / (X_train_std + 1e-8)
    X_test_norm = (X_test - X_train_mean) / (X_train_std + 1e-8)
    print("Features normalized using training set mean and std")
    print()
    
    # 4. Train logistic regression
    print_subsection("4. TRAINING LOGISTIC REGRESSION MODEL")
    model = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)
    model.fit(X_train_norm, y_train)
    print(f"Model trained with max_iter=10000")
    print(f"Model weights shape: {model.coef_[0].shape}")
    print(f"Top 5 feature weights: {sorted([(i, w) for i, w in enumerate(model.coef_[0])], key=lambda x: abs(x[1]), reverse=True)[:5]}")
    print(f"Model intercept: {model.intercept_[0]:.6f}")
    print()
    
    # 5. Predictions on training set
    print_subsection("5. PREDICTIONS ON TRAINING SET")
    y_train_pred_proba = model.predict_proba(X_train_norm)[:, 1]
    y_train_pred = model.predict(X_train_norm)
    
    print("Sample predictions (first 10):")
    for i in range(min(10, len(y_train_pred))):
        pred_label = "Benign" if y_train_pred[i] == 1 else "Malignant"
        actual_label = "Benign" if y_train[i] == 1 else "Malignant"
        print(f"  Sample {i}: Predicted={pred_label}, Actual={actual_label}, "
              f"Prob={y_train_pred_proba[i]:.4f}")
    print()
    
    # 6. Training set evaluation
    print_subsection("6. TRAINING SET EVALUATION")
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    
    print("Training Metrics:")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  F1 Score:  {train_f1:.4f}")
    
    # Confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    tn, fp, fn, tp = cm_train.ravel()
    print()
    print("Confusion Matrix Summary")
    print("========================")
    print(f"TP: {tp}, TN: {tn}")
    print(f"FP: {fp}, FN: {fn}")
    print()
    print(f"Accuracy:    {(tp+tn)/(tp+tn+fp+fn):.4f}")
    print(f"Precision:   {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}")
    print(f"Recall:      {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}")
    print(f"Specificity: {tn/(tn+fp) if (tn+fp) > 0 else 0:.4f}")
    if (tp+fp) > 0 and (tp+fn) > 0:
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        print(f"F1 Score:    {2*(p*r)/(p+r):.4f}")
    print()
    
    # 7. Predictions on test set
    print_subsection("7. PREDICTIONS ON TEST SET")
    y_test_pred_proba = model.predict_proba(X_test_norm)[:, 1]
    y_test_pred = model.predict(X_test_norm)
    
    print("Sample predictions (first 10):")
    for i in range(min(10, len(y_test_pred))):
        pred_label = "Benign" if y_test_pred[i] == 1 else "Malignant"
        actual_label = "Benign" if y_test[i] == 1 else "Malignant"
        print(f"  Sample {i}: Predicted={pred_label}, Actual={actual_label}, "
              f"Prob={y_test_pred_proba[i]:.4f}")
    print()
    
    # 8. Test set evaluation
    print_subsection("8. TEST SET EVALUATION")
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    print("Test Metrics:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    # Confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm_test.ravel()
    print()
    print("Confusion Matrix Summary")
    print("========================")
    print(f"TP: {tp}, TN: {tn}")
    print(f"FP: {fp}, FN: {fn}")
    print()
    print(f"Accuracy:    {(tp+tn)/(tp+tn+fp+fn):.4f}")
    print(f"Precision:   {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}")
    print(f"Recall:      {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}")
    print(f"Specificity: {tn/(tn+fp) if (tn+fp) > 0 else 0:.4f}")
    if (tp+fp) > 0 and (tp+fn) > 0:
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        print(f"F1 Score:    {2*(p*r)/(p+r):.4f}")
    print()
    
    # 9. ROC-AUC Analysis
    print_subsection("9. ROC-AUC ANALYSIS")
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    print("ROC Curve Statistics:")
    print(f"  AUC Score: {roc_auc:.4f}")
    print("  Interpretation: AUC ranges from 0.5 (random) to 1.0 (perfect)")
    
    if roc_auc > 0.9:
        print("  Classification: Excellent discrimination")
    elif roc_auc > 0.8:
        print("  Classification: Good discrimination")
    elif roc_auc > 0.7:
        print("  Classification: Fair discrimination")
    else:
        print("  Classification: Poor discrimination")
    print()
    
    # 10. Additional analysis
    print_subsection("10. ADDITIONAL ANALYSIS")
    print("Training data distribution:")
    print(f"  Benign samples:    {sum(y_train == 1)}")
    print(f"  Malignant samples: {sum(y_train == 0)}")
    print()
    print("Test data distribution:")
    print(f"  Benign samples:    {sum(y_test == 1)}")
    print(f"  Malignant samples: {sum(y_test == 0)}")
    print()
    
    # 11. Summary comparison
    print_subsection("11. SUMMARY: TRAIN VS TEST")
    print("Metric          | Train    | Test     | Difference")
    print("-" * 55)
    print(f"Accuracy        | {train_acc:.4f}  | {test_acc:.4f}  | {(train_acc - test_acc):.4f}")
    print(f"Precision       | {train_prec:.4f}  | {test_prec:.4f}  | {(train_prec - test_prec):.4f}")
    print(f"Recall          | {train_rec:.4f}  | {test_rec:.4f}  | {(train_rec - test_rec):.4f}")
    print(f"F1 Score        | {train_f1:.4f}  | {test_f1:.4f}  | {(train_f1 - test_f1):.4f}")
    print()
    
    print_section("✓ Classification example complete!")

if __name__ == "__main__":
    main()
