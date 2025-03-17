# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:01:59 2025

@author: RITI
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import os

# Ensure NLTK resources are available
nltk.download("stopwords")
nltk.download("punkt")

# Define the file path
file_path = "cleaned_dataset.csv"

# Step 1: Data Cleaning (fakenews.py)
def clean_dataset():
    print("Cleaning dataset...")
    data = pd.read_csv("news_articles.csv")
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.to_csv(file_path, index=False)
    print("Dataset cleaned and saved as:", file_path)

# Step 2: Keyword Detection (detecting_keywords.py)
def detect_keywords():
    print("\nDetecting top fake news keywords...")
    data = pd.read_csv(file_path)
    stop_words = set(stopwords.words("english"))
    title_counter = Counter()
    text_counter = Counter()

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

    top_keywords_title = title_counter.most_common(10)
    top_keywords_text = text_counter.most_common(10)

    print("Top Keywords in Fake News Titles:", top_keywords_title)
    print("Top Keywords in Fake News Texts:", top_keywords_text)

# Step 3: News Source Credibility Analysis (news_credibility.py)
def analyze_news_credibility():
    print("\nAnalyzing news source credibility...")
    data = pd.read_csv(file_path)
    source_counts = data.groupby(["site_url", "label"]).size().unstack(fill_value=0)

    source_counts["Percentage Real (%)"] = (source_counts.get("Real", 0) / (source_counts.get("Real", 0) + source_counts.get("Fake", 0))) * 100
    source_counts["Percentage Fake (%)"] = (source_counts.get("Fake", 0) / (source_counts.get("Real", 0) + source_counts.get("Fake", 0))) * 100

    sorted_sources = source_counts.sort_values(by="Percentage Real (%)", ascending=False)
    
    print("Top 5 Most Credible News Sources:")
    print(sorted_sources.head(5))
    
    print("\nTop 5 Least Credible News Sources:")
    print(sorted_sources.tail(5))

# Step 4: News Length Analysis (news_length.py)
def analyze_news_length():
    print("\nAnalyzing news title and text length...")
    data = pd.read_csv(file_path)
    
    data["title_length"] = data["title"].apply(lambda x: len(str(x)))
    data["text_length"] = data["text"].apply(lambda x: len(str(x)))

    real_news = data[data["label"] == "Real"]
    fake_news = data[data["label"] == "Fake"]

    avg_real_title_length = real_news["title_length"].mean()
    avg_fake_title_length = fake_news["title_length"].mean()
    avg_real_text_length = real_news["text_length"].mean()
    avg_fake_text_length = fake_news["text_length"].mean()

    print(f"Average Title Length for Real News: {avg_real_title_length:.2f} characters")
    print(f"Average Title Length for Fake News: {avg_fake_title_length:.2f} characters")
    print(f"Average Text Length for Real News: {avg_real_text_length:.2f} characters")
    print(f"Average Text Length for Fake News: {avg_fake_text_length:.2f} characters")

    # Plot
    labels = ["Real Title", "Fake Title", "Real Text", "Fake Text"]
    lengths = [avg_real_title_length, avg_fake_title_length, avg_real_text_length, avg_fake_text_length]

    plt.figure(figsize=(10,6))
    plt.bar(labels, lengths, color=["green", "red", "green", "red"])
    plt.title("Average Title & Text Lengths for Real & Fake News")
    plt.ylabel("Average Length (characters)")
    plt.xticks(rotation=45)
    plt.show()

# Step 5: Sensationalism Analysis (sensationalism.py)
def analyze_sensationalism():
    print("\nAnalyzing sensationalism in news...")
    data = pd.read_csv(file_path)

    def detect_sensationalism(text):
        sensational_keywords = ["shocking", "outrageous", "unbelievable", "mind-blowing", "explosive","must-watch", "terrifying", "secret reveal"]
        if pd.isna(text):
            return False
        return any(re.search(r'\b' + kw + r'\b', text, re.IGNORECASE) for kw in sensational_keywords)

    data["Sensationalism"] = data["text"].apply(detect_sensationalism)

    contingency_table = pd.crosstab(data["Sensationalism"], data["label"])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"Chi-squared statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")

    alpha = 0.05
    if p < alpha:
        print("Significant association between sensationalism and fake news.")
    else:
        print("No significant association between sensationalism and fake news.")

# Run all functions
if __name__ == "__main__":
    clean_dataset()
    detect_keywords()
    analyze_news_credibility()
    analyze_news_length()
    analyze_sensationalism()
