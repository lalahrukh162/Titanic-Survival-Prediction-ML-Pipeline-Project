# Titanic Survival Prediction – Machine Learning Classification

## Project Overview
This project builds an end-to-end **machine learning classification pipeline** to predict passenger survival on the **Titanic dataset**.  
It demonstrates **professional ML workflow practices**, including data preprocessing, feature engineering, model pipelines, cross-validation, and hyperparameter tuning.

The goal is not just accuracy, but **clean, reproducible, and production-ready modeling code**.

---

## Dataset
- **Source:** Titanic dataset (Kaggle / public ML benchmark)
- **Target Variable:** `Survived` (0 = No, 1 = Yes)
- **Features:**  
  - Numerical: `Age`, `Fare`, `SibSp`, `Parch`
  - Categorical: `Sex`, `Embarked`, `Pclass`, etc.

---

## Key Concepts Demonstrated
- Feature type separation (numerical vs categorical)
- Missing value handling using `SimpleImputer`
- One-Hot Encoding for categorical variables
- `ColumnTransformer` for clean preprocessing
- `Pipeline` to prevent data leakage
- Model comparison using cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Clear evaluation using multiple metrics

---

## Models Implemented
The notebook compares multiple classifiers using the **same preprocessing pipeline**:

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

Each model is:
- Wrapped in a `Pipeline`
- Tuned using `GridSearchCV`
- Evaluated using cross-validation and hold-out validation data

---

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Cross-Validation Mean Accuracy

This ensures the selected model generalizes well rather than overfitting.

