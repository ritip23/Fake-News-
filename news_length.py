# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:37:53 2025

@author: RITI
"""

import pandas as pd
import matplotlib.pyplot as plt

# Define the file path (Update this path accordingly)
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Calculate title and text length
data["title_length"] = data["title"].apply(lambda x: len(str(x)))
data["text_length"] = data["text"].apply(lambda x: len(str(x)))

# Separate Real and Fake news
real_news = data[data["label"] == "Real"]
fake_news = data[data["label"] == "Fake"]

# Compute average lengths
avg_real_title_length = real_news["title_length"].mean()
avg_fake_title_length = fake_news["title_length"].mean()
avg_real_text_length = real_news["text_length"].mean()
avg_fake_text_length = fake_news["text_length"].mean()

print(f"Average Title Length for Real News: {avg_real_title_length:.2f} characters")
print(f"Average Title Length for Fake News: {avg_fake_title_length:.2f} characters")
print(f"Average Text Length for Real News: {avg_real_text_length:.2f} characters")
print(f"Average Text Length for Fake News: {avg_fake_text_length:.2f} characters")

# Bar Chart Visualization
labels = ["Real Title", "Fake Title", "Real Text", "Fake Text"]
lengths = [avg_real_title_length, avg_fake_title_length, avg_real_text_length, avg_fake_text_length]

plt.figure(figsize=(10,6))
plt.bar(labels, lengths, color=["green", "red", "green", "red"])
plt.title("Average Title & Text Lengths for Real & Fake News")
plt.ylabel("Average Length (characters)")
plt.xticks(rotation=45)
plt.show()