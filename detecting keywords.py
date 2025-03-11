# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:08:14 2025

@author: RITI
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Define the file path (Update this path accordingly)
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))

# Initialize word counters
title_counter = Counter()
text_counter = Counter()

# Iterate over dataset to extract keywords for Fake news
for index, row in data.iterrows():
    if pd.isna(row["title"]) or pd.isna(row["text"]):
        continue
    
    title_words = word_tokenize(row["title"])
    text_words = word_tokenize(row["text"])
    
    title_words = [word.lower() for word in title_words if word.isalpha() and word.lower() not in stop_words]
    text_words = [word.lower() for word in text_words if word.isalpha() and word.lower() not in stop_words]
    
    if row["label"] == "Fake":
        title_counter.update(title_words)
        text_counter.update(text_words)

# Get top 5 keywords associated with Fake News
top_keywords_title = title_counter.most_common(5)
top_keywords_text = text_counter.most_common(5)

print("Top 5 Keywords Associated with Fake News Titles:")
for keyword, count in top_keywords_title:
    print(f"{keyword}: {count} times")

print("Top 5 Keywords Associated with Fake News Texts:")
for keyword, count in top_keywords_text:
    print(f"{keyword}: {count} times") 