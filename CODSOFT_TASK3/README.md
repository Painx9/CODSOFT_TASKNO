# Task 3: Customer Churn Prediction

## Overview

This project is part of the CodSoft Machine Learning virtual internship. The goal is to predict customer churn for a bank using a machine learning classification algorithm. By identifying customers likely to leave, businesses can take proactive retention measures.

## Dataset

The dataset contains 10,000 customer records with the following characteristics:

- **Demographics:** Age, Gender, Geography
- **Financials:** Credit Score, Balance, Estimated Salary
- **Engagement:** Tenure, Number of Products, Has Credit Card, Is Active Member
- **Target Variable:** `Exited` (1 = Churned, 0 = Retained)

## Methodology

1. **Data Preprocessing:** Dropped irrelevant identifier columns (`RowNumber`, `CustomerId`, `Surname`).
2. **Feature Engineering:** Converted categorical variables (`Geography`, `Gender`) into numerical features using One-Hot Encoding.
3. **Data Splitting & Scaling:** Split the data into an 80/20 train-test split. Standardized the numerical features using Scikit-Learn's `StandardScaler`.
4. **Model Training:** Trained a **Random Forest Classifier** (`n_estimators=100`) configured with `class_weight='balanced'` to handle the slight class imbalance (roughly 20% of customers churned).

## Results

- **Overall Accuracy:** 85.75%
- **Precision (Churn):** 0.77 (When the model predicts a churn, it is correct 77% of the time).
- The final predictions on the test set are saved in `dataset/predicted_churn.csv`.

## Technologies Used

- Python
- Pandas
- Scikit-Learn (StandardScaler, RandomForestClassifier)
