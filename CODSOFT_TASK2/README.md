# Task 2: Credit Card Fraud Detection

## Overview

This project is part of the CodSoft Machine Learning virtual internship. The objective is to build a classification model capable of identifying fraudulent credit card transactions while handling a highly imbalanced dataset.

## Dataset

The project uses `fraudTrain.csv` and `fraudTest.csv`, containing nearly 1.3 million training transactions and over 550,000 testing transactions.

- **Legitimate Transactions:** ~99.42%
- **Fraudulent Transactions:** ~0.57%

## Methodology

1. **Feature Engineering:** Filtered out personally identifiable information (PII) and retained behavioral/geographical features (`category`, `amt`, `gender`, `city_pop`, `lat`, `long`).
2. **Data Preprocessing:** Converted categorical variables into numerical format using One-Hot Encoding (`pd.get_dummies`) and aligned the training and testing datasets.
3. **Feature Scaling:** Standardized the feature ranges using Scikit-Learn's `StandardScaler` to ensure optimal model performance.
4. **Model Training & Imbalance Handling:** Trained a **Logistic Regression** classifier. To combat the extreme class imbalance (accuracy paradox), the model was configured with `class_weight='balanced'`, heavily penalizing misclassifications of the minority fraud class.

## Results

- **Overall Accuracy:** 88.92%
- **Fraud Recall:** 0.74 (The model successfully identifies 74% of actual fraudulent transactions)
- The final test set predictions are saved in the `dataset/predicted_fraud.csv` file.

## Technologies Used

- Python
- Pandas
- Scikit-Learn (StandardScaler, LogisticRegression)
