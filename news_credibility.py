# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:05:46 2025

@author: RITI
"""

import pandas as pd

# Define the file path (Update this path accordingly)
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Group by site URL and label, then count occurrences
source_counts = data.groupby(["site_url", "label"]).size().unstack(fill_value=0)

# Calculate credibility percentages
source_counts["Percentage Real (%)"] = (source_counts.get("Real", 0) / (source_counts.get("Real", 0) + source_counts.get("Fake", 0))) * 100
source_counts["Percentage Fake (%)"] = (source_counts.get("Fake", 0) / (source_counts.get("Real", 0) + source_counts.get("Fake", 0))) * 100

# Sort sources by credibility
sorted_sources = source_counts.sort_values(by="Percentage Real (%)", ascending=False)

# Display Top 10 Most Credible News Sources
print("Top 10 Most Credible News Sources:")
for source, row in sorted_sources.head(10).iterrows():
    print(f"News {source}, fake news = {row['Percentage Fake (%)']:.1f}%")

# Display Top 10 Least Credible News Sources
print("Top 10 Least Credible News Sources:")
for source, row in sorted_sources.tail(10).iterrows():
    print(f"News {source}, fake news = {row['Percentage Fake (%)']:.1f}%")
