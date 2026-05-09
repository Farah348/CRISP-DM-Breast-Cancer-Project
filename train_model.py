"""
CRISP-DM Breast Cancer Data Mining Project
Author: Student
Dataset: Breast Cancer Wisconsin Diagnostic dataset from scikit-learn/UCI
Task: Binary classification - malignant vs benign tumour diagnosis
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


def load_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    # In sklearn: 0 = malignant, 1 = benign. This project uses 1 = malignant, 0 = benign.
    y = (data.target == 0).astype(int)
    df = X.copy()
    df["target"] = y
    df["diagnosis_label"] = df["target"].map({1: "Malignant", 0: "Benign"})
    return data, X, y, df


def save_dataset_samples(df):
    df.head(10).to_csv("artifacts/dataset_sample.csv", index=False)
    pd.DataFrame({
        "rows": [df.shape[0]],
        "columns_including_target_and_label": [df.shape[1]],
        "missing_values": [int(df.isna().sum().sum())],
        "duplicates": [int(df.duplicated().sum())]
    }).to_csv("artifacts/dataset_summary.csv", index=False)


def perform_eda(df):
    plt.figure(figsize=(7, 5))
    counts = df["diagnosis_label"].value_counts().reindex(["Benign", "Malignant"])
    ax = counts.plot(kind="bar")
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center")
    plt.title("Class Distribution: Benign vs Malignant")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig("figures/class_distribution.png", dpi=200)
    plt.close()

    corr = df.drop(columns=["target", "diagnosis_label"]).corr()
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Correlation Heatmap of Breast Cancer Features")
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="diagnosis_label", y="mean radius", order=["Benign", "Malignant"], color="lightgray")
    plt.title("Boxplot: Mean Radius by Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Mean Radius")
    plt.tight_layout()
    plt.savefig("figures/boxplot_mean_radius.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x="mean radius", y="target", logistic=True, y_jitter=0.03, scatter_kws={"alpha": 0.5})
    plt.title("Logistic Regression Plot: Mean Radius vs Malignancy")
    plt.xlabel("Mean Radius")
    plt.ylabel("Probability of Malignant Class")
    plt.tight_layout()
    plt.savefig("figures/logistic_plot_mean_radius.png", dpi=200)
    plt.close()

    rows = []
    for col in df.drop(columns=["target", "diagnosis_label"]).columns:
        r, p = pointbiserialr(df["target"], df[col])
        rows.append({"feature": col, "point_biserial_r": r, "p_value": p, "abs_r": abs(r)})
    corr_pvals = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    corr_pvals.to_csv("artifacts/correlations_with_pvalues.csv", index=False)

    top10 = corr_pvals.head(10).sort_values("point_biserial_r")
    plt.figure(figsize=(8, 6))
    plt.barh(top10["feature"], top10["point_biserial_r"])
    plt.title("Top 10 Features Associated with Malignancy")
    plt.xlabel("Point-Biserial Correlation with Target")
    plt.tight_layout()
    plt.savefig("figures/top_correlations.png", dpi=200)
    plt.close()


def preprocess_features(X):
    missing_summary = X.isna().sum().reset_index()
    missing_summary.columns = ["feature", "missing_values"]
    missing_summary.to_csv("artifacts/missing_values_before.csv", index=False)

    outlier_rows = []
    for col in X.columns:
        q1, q3 = X[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((X[col] < lower) | (X[col] > upper)).sum())
        outlier_rows.append({"feature": col, "outlier_count_iqr": count, "lower_bound": lower, "upper_bound": upper})
    pd.DataFrame(outlier_rows).to_csv("artifacts/outlier_summary.csv", index=False)

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_reduced = X.drop(columns=to_drop)
    pd.DataFrame({"dropped_feature_due_to_high_correlation": to_drop}).to_csv("artifacts/dropped_features.csv", index=False)
    pd.DataFrame({
        "stage": ["before_feature_removal", "after_feature_removal"],
        "feature_count": [X.shape[1], X_reduced.shape[1]]
    }).to_csv("artifacts/preprocessing_before_after.csv", index=False)
    return X_reduced, to_drop


def evaluate_model(model_name, estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    else:
        y_score = estimator.decision_function(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score)
    }
    return metrics, y_pred, y_score


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    pd.DataFrame({
        "split": ["train", "test"],
        "rows": [len(X_train), len(X_test)],
        "malignant_count": [int(y_train.sum()), int(y_test.sum())],
        "benign_count": [int((y_train == 0).sum()), int((y_test == 0).sum())]
    }).to_csv("artifacts/train_test_split_summary.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    lr_grid = {
        "model__C": [0.1, 1, 10],
        "model__solver": ["liblinear"],
        "model__penalty": ["l2"]
    }
    lr_search = GridSearchCV(lr_pipeline, lr_grid, cv=cv, scoring="f1", n_jobs=1)
    lr_search.fit(X_train, y_train)

    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    rf_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=cv, scoring="f1", n_jobs=1)
    rf_search.fit(X_train, y_train)

    models = {
        "Logistic Regression": lr_search.best_estimator_,
        "Random Forest": rf_search.best_estimator_
    }
    best_params = {
        "Logistic Regression": lr_search.best_params_,
        "Random Forest": rf_search.best_params_
    }
    with open("artifacts/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    metrics_list = []
    for name, estimator in models.items():
        metrics, y_pred, y_score = evaluate_model(name, estimator, X_test, y_test)
        metrics_list.append(metrics)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        filename = name.lower().replace(" ", "_")
        plt.savefig(f"figures/confusion_matrix_{filename}.png", dpi=200)
        plt.close()

    metrics_df = pd.DataFrame(metrics_list)
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        metrics_df[col] = metrics_df[col].round(4)
    metrics_df.to_csv("artifacts/metrics.csv", index=False)

    plt.figure(figsize=(8, 6))
    for name, estimator in models.items():
        RocCurveDisplay.from_estimator(estimator, X_test, y_test, name=name, ax=plt.gca())
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig("figures/roc_curve_comparison.png", dpi=200)
    plt.close()

    metrics_melt = metrics_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_melt, x="Metric", y="Score", hue="Model", palette="deep")
    plt.ylim(0.85, 1.01)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("figures/model_comparison.png", dpi=200)
    plt.close()

    metrics_df_sorted = metrics_df.sort_values(["F1-score", "Recall", "ROC-AUC"], ascending=False)
    best_model_name = metrics_df_sorted.iloc[0]["Model"]
    best_model = models[best_model_name]
    joblib.dump(best_model, "artifacts/best_model.joblib")
    joblib.dump(list(X.columns), "artifacts/feature_columns.joblib")
    with open("artifacts/best_model_name.txt", "w") as f:
        f.write(best_model_name)
    return metrics_df, best_model_name, best_params


def main():
    data, X, y, df = load_dataset()
    save_dataset_samples(df)
    perform_eda(df)
    X_reduced, dropped = preprocess_features(X)
    metrics_df, best_model_name, best_params = train_and_evaluate(X_reduced, y)
    print("Dataset shape:", df.shape)
    print("Dropped features:", dropped)
    print("Best hyperparameters:", best_params)
    print(metrics_df)
    print("Best model:", best_model_name)


if __name__ == "__main__":
    main()
