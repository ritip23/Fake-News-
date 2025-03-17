# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:13:01 2025

@author: RITI
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the file path
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Split features and labels
X = data["title"]
y = data["label"]

# Convert text data into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Encode labels (Fake = 1, Real = 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest model
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = random_forest_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict if news is fake using the trained model
def fakenewsprediction(title):
    title_tfidf = tfidf_vectorizer.transform([title])
    prediction = random_forest_classifier.predict(title_tfidf)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Example test case
title_input = "Few reasons for optimism after Antony Blinken's diplomatic dash"
prediction = fakenewsprediction(title_input)
print(f"Prediction: {prediction}")
