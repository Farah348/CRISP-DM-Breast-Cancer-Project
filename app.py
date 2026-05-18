"""
Streamlit app for the SEER Breast Cancer Survival Classification project.
Run after executing train_model.py:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SEER Breast Cancer Prediction App", page_icon="🩺", layout="wide")
st.title("SEER Breast Cancer Survival Prediction App")
st.write("Upload a CSV file with the same feature columns used during training. The app predicts Alive or Dead status.")

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
except Exception:
    st.error("Model artifacts were not found. Please run train_model.py first.")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error("Missing required columns:")
        st.write(missing_cols)
    else:
        X_new = df[feature_columns]
        preds = model.predict(X_new)
        probs = model.predict_proba(X_new)[:, 1]

        result_df = df.copy()
        result_df["Predicted_Class"] = preds
        result_df["Predicted_Status"] = result_df["Predicted_Class"].map({0: "Alive", 1: "Dead"})
        result_df["Probability_of_Death"] = probs.round(4)

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        col1, col2 = st.columns(2)
        col1.metric("Predicted Alive", int((result_df["Predicted_Class"] == 0).sum()))
        col2.metric("Predicted Dead", int((result_df["Predicted_Class"] == 1).sum()))

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "seer_prediction_results.csv", "text/csv")
else:
    st.info("Upload a SEER-format CSV file to generate predictions.")
