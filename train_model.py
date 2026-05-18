"""
CRISP-DM Data Mining Project - Larger Breast Cancer Dataset Version
Dataset: SEER/Kaggle Breast Cancer Survival Dataset
Task: Binary classification of patient survival status (Alive vs Dead)

IMPORTANT:
Download the SEER Breast Cancer CSV from Kaggle and place it at:
    data/seer_breast_cancer.csv
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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

RANDOM_STATE = 42
DATA_PATH = "data/seer_breast_cancer.csv"
os.makedirs("figures", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


def load_seer_dataset(path: str = DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the SEER Breast Cancer CSV from Kaggle "
            "and save it as data/seer_breast_cancer.csv"
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Status" not in df.columns:
        raise ValueError("Target column 'Status' was not found. Please check the dataset columns.")

    df["Status"] = df["Status"].astype(str).str.strip()
    y = df["Status"].str.lower().map({"alive": 0, "dead": 1})

    if y.isna().any():
        raise ValueError("The Status column contains unexpected values. Expected values: Alive and Dead.")

    X = df.drop(columns=["Status"])
    return df, X, y.astype(int)


def create_basic_artifacts(df, X, y):
    df.head(10).to_csv("artifacts/dataset_sample.csv", index=False)
    pd.DataFrame({
        "rows": [df.shape[0]],
        "columns": [df.shape[1]],
        "target_column": ["Status"],
        "positive_class": ["Dead"],
        "negative_class": ["Alive"],
        "missing_values_total": [int(df.isna().sum().sum())],
        "duplicate_rows": [int(df.duplicated().sum())]
    }).to_csv("artifacts/dataset_summary.csv", index=False)

    y.value_counts().rename(index={0: "Alive", 1: "Dead"}).reset_index().to_csv(
        "artifacts/class_distribution.csv", index=False
    )


def perform_eda(df, y):
    plot_df = df.copy()
    plot_df["target_numeric"] = y
    plot_df["Status_Label"] = y.map({0: "Alive", 1: "Dead"})

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=plot_df, x="Status_Label", order=["Alive", "Dead"])
    for container in ax.containers:
        ax.bar_label(container)
    plt.title("Class Distribution: Alive vs Dead")
    plt.xlabel("Patient Status")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig("figures/status_distribution.png", dpi=200)
    plt.close()

    if "Age" in plot_df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=plot_df, x="Age", hue="Status_Label", bins=25, kde=True)
        plt.title("Age Distribution by Patient Status")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("figures/age_distribution.png", dpi=200)
        plt.close()

    if "Tumor Size" in plot_df.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=plot_df, x="Status_Label", y="Tumor Size", order=["Alive", "Dead"], color="lightgray")
        plt.title("Tumor Size by Patient Status")
        plt.xlabel("Patient Status")
        plt.ylabel("Tumor Size")
        plt.tight_layout()
        plt.savefig("figures/tumor_size_boxplot.png", dpi=200)
        plt.close()

    if "Survival Months" in plot_df.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=plot_df, x="Status_Label", y="Survival Months", order=["Alive", "Dead"], color="lightgray")
        plt.title("Survival Months by Patient Status")
        plt.xlabel("Patient Status")
        plt.ylabel("Survival Months")
        plt.tight_layout()
        plt.savefig("figures/survival_months_boxplot.png", dpi=200)
        plt.close()

    numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 2:
        corr = plot_df[numeric_cols].corr()
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation Matrix for Numeric Features")
        plt.tight_layout()
        plt.savefig("figures/numeric_correlation_heatmap.png", dpi=200)
        plt.close()

    rows = []
    for col in numeric_cols:
        if col == "target_numeric":
            continue
        temp = plot_df[[col, "target_numeric"]].dropna()
        if temp[col].nunique() > 1:
            r, p = pointbiserialr(temp["target_numeric"], temp[col])
            rows.append({"feature": col, "point_biserial_r": r, "p_value": p, "abs_r": abs(r)})
    if rows:
        corr_df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
        corr_df.to_csv("artifacts/correlations_with_pvalues.csv", index=False)

        top = corr_df.head(8).sort_values("point_biserial_r")
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["point_biserial_r"])
        plt.title("Top Numeric Feature Correlations with Death Status")
        plt.xlabel("Point-Biserial Correlation")
        plt.tight_layout()
        plt.savefig("figures/top_numeric_correlations.png", dpi=200)
        plt.close()


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    pd.DataFrame({
        "numeric_features": pd.Series(numeric_features),
        "categorical_features": pd.Series(categorical_features)
    }).to_csv("artifacts/feature_types.csv", index=False)

    return preprocessor


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    pd.DataFrame({
        "split": ["train", "test"],
        "rows": [len(X_train), len(X_test)],
        "dead_count": [int(y_train.sum()), int(y_test.sum())],
        "alive_count": [int((y_train == 0).sum()), int((y_test == 0).sum())]
    }).to_csv("artifacts/train_test_split_summary.csv", index=False)

    preprocessor = build_preprocessor(X)

    lr_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
    ])

    lr_grid = {
        "model__C": [0.1, 1, 10],
        "model__solver": ["liblinear"],
        "model__penalty": ["l2"]
    }

    rf_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    searches = {
        "Logistic Regression": GridSearchCV(lr_pipeline, lr_grid, cv=cv, scoring="f1", n_jobs=-1),
        "Random Forest": GridSearchCV(rf_pipeline, rf_grid, cv=cv, scoring="f1", n_jobs=-1)
    }

    fitted = {}
    best_params = {}
    metrics_rows = []

    for name, search in searches.items():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        fitted[name] = best_model
        best_params[name] = search.best_params_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alive", "Dead"])
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"figures/confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=200)
        plt.close()

    metrics_df = pd.DataFrame(metrics_rows)
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        metrics_df[col] = metrics_df[col].round(4)
    metrics_df.to_csv("artifacts/metrics.csv", index=False)

    plt.figure(figsize=(8, 6))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=plt.gca())
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig("figures/roc_curve_comparison.png", dpi=200)
    plt.close()

    metrics_melt = metrics_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_melt, x="Metric", y="Score", hue="Model")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/model_comparison.png", dpi=200)
    plt.close()

    with open("artifacts/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    best_model_name = metrics_df.sort_values(["F1-score", "Recall", "ROC-AUC"], ascending=False).iloc[0]["Model"]
    joblib.dump(fitted[best_model_name], "artifacts/best_model.joblib")
    joblib.dump(list(X.columns), "artifacts/feature_columns.joblib")
    with open("artifacts/best_model_name.txt", "w") as f:
        f.write(best_model_name)

    return metrics_df, best_model_name


def main():
    df, X, y = load_seer_dataset()
    create_basic_artifacts(df, X, y)
    perform_eda(df, y)
    metrics_df, best_model_name = train_models(X, y)
    print("Dataset shape:", df.shape)
    print(metrics_df)
    print("Best model:", best_model_name)


if __name__ == "__main__":
    main()
