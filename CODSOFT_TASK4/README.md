# Task 4: Spam SMS Detection

## Overview

This project is part of the CodSoft Machine Learning virtual internship. The objective is to build a Natural Language Processing (NLP) classification model to accurately identify whether an SMS text message is "spam" or "ham" (legitimate).

## Dataset

The dataset consists of 5,572 SMS messages labeled as either spam or ham.

## Methodology

1. **Data Cleaning:** Removed empty auxiliary columns and standardized column names. Mapped the text labels ('ham', 'spam') to binary numerical values (0, 1).
2. **Text Vectorization:** Utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform the raw text messages into a matrix of 3,000 numerical features, removing standard English stop words in the process.
3. **Model Training:** Split the vectorized data into an 80/20 train-test split. Trained a **Multinomial Naive Bayes** classifier, which is highly optimized for discrete feature counts in text classification tasks.

## Results

- **Test Accuracy:** 97.85%
- **Spam Precision:** 1.00 (Zero false positives; every message flagged as spam was actually spam).
- **Spam Recall:** 0.84 (Successfully identified 84% of all actual spam messages).
- The final test set predictions are saved in the `dataset/predicted_spam.csv` file.

## Technologies Used

- Python
- Pandas
- Scikit-Learn (TfidfVectorizer, MultinomialNB)
