# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:20:25 2025

@author: RITI
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Define the file path (Update this path accordingly)
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score["compound"] >= 0.05:
        return "Positive"
    elif sentiment_score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
data["Sentiment"] = data["text"].apply(analyze_sentiment)

# Display results
print("Sentiment Analysis Results:")
print(data[['text', 'Sentiment']].head())

# Save updated dataset
data.to_csv("C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/sentiment_analysis_results.csv", index=False)
print("Sentiment analysis completed and saved.")