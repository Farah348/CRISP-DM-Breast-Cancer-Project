"""
Simple Streamlit App for Breast Cancer Prediction
Run with: streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Prediction App", page_icon="🩺", layout="wide")
st.title("Breast Cancer Prediction App")
st.write("Upload a CSV file containing the required breast cancer feature columns. The app will predict whether each record is benign or malignant.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/best_model.joblib")
    feature_columns = joblib.load("artifacts/feature_columns.joblib")
    with open("artifacts/best_model_name.txt", "r") as f:
        model_name = f.read().strip()
    return model, feature_columns, model_name

try:
    model, feature_columns, model_name = load_artifacts()
    st.success(f"Loaded best model: {model_name}")
except Exception as e:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error("The uploaded CSV is missing required columns:")
        st.write(missing_cols)
    else:
        X_new = df[feature_columns]
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)[:, 1]

        result_df = df.copy()
        result_df["Predicted_Class"] = predictions
        result_df["Predicted_Diagnosis"] = result_df["Predicted_Class"].map({1: "Malignant", 0: "Benign"})
        result_df["Malignancy_Probability"] = probabilities.round(4)

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        malignant_count = int((result_df["Predicted_Class"] == 1).sum())
        benign_count = int((result_df["Predicted_Class"] == 0).sum())

        col1, col2 = st.columns(2)
        col1.metric("Predicted Benign", benign_count)
        col2.metric("Predicted Malignant", malignant_count)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
else:
    st.info("Upload sample_input.csv to test the app.")
