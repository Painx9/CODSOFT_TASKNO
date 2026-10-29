# Task 1: Movie Genre Classification

## Overview

This project is part of the CodSoft Machine Learning virtual internship. The objective of this task is to build a machine learning model capable of predicting the genre of a movie based on its plot summary.

## Dataset

The dataset consists of movie records containing the following features:

- **ID:** Unique identifier for each movie.
- **Title:** The name and release year of the movie.
- **Genre:** The target variable (e.g., drama, documentary, action).
- **Description:** A text summary of the movie's plot.

## Methodology

The project follows a standard Natural Language Processing (NLP) pipeline:

1. **Data Preprocessing:** Cleaned the raw text descriptions by converting them to lowercase, removing punctuation, stripping numbers, and eliminating extra spaces to isolate the meaningful vocabulary.
2. **Feature Extraction:** Utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the cleaned text data into numerical feature vectors, limiting to the top 10,000 most frequent words.
3. **Model Training:** Split the data into an 80/20 training and validation set and trained a **Multinomial Naive Bayes** classifier, which is highly efficient for text classification tasks.
4. **Prediction:** Generated predictions on the unseen test dataset and mapped the numeric outputs back to their respective genre labels.

## Results

- **Validation Accuracy:** ~50.49%
- _Note: The dataset is highly imbalanced, with genres like "drama" and "documentary" dominating the data, which influences the baseline accuracy score._
- The final test set predictions are saved in the `dataset/predicted_genres.csv` file.

## Technologies Used

- Python
- Pandas
- Scikit-Learn (TF-IDF, MultinomialNB, LabelEncoder)
- Regular Expressions (re)
