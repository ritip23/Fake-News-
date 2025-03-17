# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:41:10 2025

@author: RITI
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Define the file path
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
missing_data = data[["text", "label"]].isnull().any(axis=1)
if missing_data.any():
    print("Missing Values Found in the Dataset. Handle Missing Data Before Proceeding")
else:
    # Encode labels (Fake = 1, Real = 0)
    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])
    
    # Split data
    X = data["text"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create TF-IDF and Logistic Regression pipeline
    text_feature_extraction = TfidfVectorizer(max_features=5000, stop_words="english")
    model = LogisticRegression()
    pipeline = Pipeline([
        ('tfidf', text_feature_extraction),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Function to predict if news is fake using the trained model
    def fakenewsprediction(text):
        input_data = [text]
        prediction = pipeline.predict(input_data)
        return "Real News" if prediction[0] == 0 else "Fake News"

    # Example test case
    article_input = "Stocks rallied sharply after the Labor Department said nonfarm payrolls rose by 150,000 in October — 20,000 fewer than expected but a difference attributable pretty much completely to the auto strikes, which appear to be over."
    prediction = fakenewsprediction(article_input)
    print(f"Prediction: {prediction}")
