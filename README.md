# CRISP-DM Breast Cancer Classification Project

This repository contains the complete Python code package for the Mining and Predictive Analytics (COMP 30044) CW2 assignment.

The project follows the CRISP-DM methodology to perform a breast cancer diagnostic classification task using the Breast Cancer Wisconsin Diagnostic dataset.

## Project Files

- `train_model.py` - Main Python script for dataset loading, EDA, preprocessing, model training, hyperparameter tuning, evaluation, graph creation, and artifact saving.
- `app.py` - Streamlit app for uploading CSV files and generating predictions.
- `requirements.txt` - Required Python libraries.
- `sample_input.csv` - Sample input file for testing the Streamlit app.
- `artifacts/metrics.csv` - Final model comparison results.
- `artifacts/best_hyperparameters.json` - Best hyperparameters selected through GridSearchCV.
- `artifacts/dataset_summary.csv` - Dataset structure summary.
- `artifacts/train_test_split_summary.csv` - Training/testing split summary.

## How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Models Used

1. Logistic Regression with StandardScaler pipeline and GridSearchCV tuning.
2. Random Forest Classifier with GridSearchCV tuning.

## Final Result

The best performing model in this project was **Logistic Regression**.

Final test results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9950 |
| Random Forest | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9940 |

## GitHub Repository Link

https://github.com/Farah348/CRISP-DM-Breast-Cancer-Project
