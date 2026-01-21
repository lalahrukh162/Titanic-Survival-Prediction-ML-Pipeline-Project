# Titanic-Survival-Prediction-ML-Pipeline-Project
Overview

This project builds an end-to-end binary classification pipeline to predict passenger survival on the Titanic.
The emphasis is on clean preprocessing, feature engineering, reproducible pipelines, and model comparison using cross-validation and hyperparameter tuning.

Dataset used: Kaggle Titanic – Machine Learning from Disaster

Objectives

Clean and preprocess structured data

Engineer meaningful features

Build reusable ML pipelines

Compare multiple classification models

Tune hyperparameters with GridSearchCV

Evaluate models using cross-validation and classification metrics

Dataset

Target: Survived (0 = No, 1 = Yes)

Features: Age, Sex, Fare, Pclass, Embarked, SibSp, Parch, and engineered features such as passenger titles and child indicators

Feature Engineering

Title extraction from names

Age imputation

Child indicator feature

Explicit handling of missing categorical values

Numerical and categorical preprocessing via ColumnTransformer

Models Used

Logistic Regression

Random Forest

XGBoost

All models are trained using scikit-learn Pipelines to prevent data leakage.

Evaluation

5-fold cross-validation

GridSearchCV for hyperparameter tuning

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Project Structure
titanic-classification/
├── titanic-classification.ipynb
├── README.md

Tech Stack

Python, pandas, numpy, scikit-learn, XGBoost, Jupyter Notebook
