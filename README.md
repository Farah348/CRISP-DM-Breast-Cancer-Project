# CRISP-DM Breast Cancer Project - Larger SEER Dataset Version

This repository contains the updated larger-dataset version of the Mining and Predictive Analytics (COMP 30044) CW2 assignment.

The project follows the CRISP-DM methodology to perform a healthcare classification task using a SEER/Kaggle breast cancer survival dataset.

## Dataset

The project uses the SEER/Kaggle Breast Cancer Survival Dataset. This is larger than the earlier Wisconsin Diagnostic Breast Cancer dataset and is more suitable for showing categorical encoding, numerical scaling, model tuning, and survival-status prediction.

Expected target column:

- `Status` with values such as `Alive` and `Dead`

Expected common feature columns include:

- Age
- Race
- Marital Status
- T Stage
- N Stage
- 6th Stage
- differentiate
- Grade
- A Stage
- Tumor Size
- Estrogen Status
- Progesterone Status
- Regional Node Examined
- Reginol Node Positive
- Survival Months

## How to Use

1. Download the SEER Breast Cancer dataset CSV from Kaggle.
2. Rename the file and place it here:

```text
data/seer_breast_cancer.csv
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the full CRISP-DM pipeline:

```bash
python train_model.py
```

5. Run the prediction app:

```bash
streamlit run app.py
```

## Project Files

- `train_model.py` - EDA, preprocessing, model training, hyperparameter tuning, evaluation, and artifact saving.
- `app.py` - Streamlit app for uploading CSV files and generating predictions.
- `requirements.txt` - Required Python libraries.
- `sample_input.csv` - Small example input file showing expected columns.
- `data/README.md` - Explains where to place the downloaded dataset.

## Models Used

1. Logistic Regression with preprocessing pipeline and GridSearchCV tuning.
2. Random Forest Classifier with preprocessing pipeline and GridSearchCV tuning.

## Evaluation Metrics

The script automatically generates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices
- ROC curve comparison
- Model comparison graph

## GitHub Repository Link

https://github.com/Farah348/CRISP-DM-Breast-Cancer-Project
