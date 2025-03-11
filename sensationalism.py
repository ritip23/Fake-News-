# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:59:09 2025

@author: RITI
"""

import pandas as pd
import re
from scipy.stats import chi2_contingency

# Define the file path (Update this path accordingly)
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Function to detect sensationalism in news text
def detect_sensationalism(text):
    sensational_keywords = ["shocking", "outrageous", "unbelievable", "mind-blowing", "explosive"]
    if pd.isna(text):
        return False
    for keyword in sensational_keywords:
        if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE):
            return True
    return False

# Apply sensationalism detection
data["Sensationalism"] = data["text"].apply(detect_sensationalism)

# Create a contingency table
contingency_table = pd.crosstab(data["Sensationalism"], data["label"])
print("Contingency Table:")
print(contingency_table)

# Perform Chi-Squared Test
chi2, p, _, _ = chi2_contingency(contingency_table)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")

# Set significance level
alpha = 0.05
if p < alpha:
    print("There is a significant association between sensationalism and credibility of the news.")
else:
    print("There is no significant association between sensationalism and credibility of the news.")