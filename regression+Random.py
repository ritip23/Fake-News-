import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Define the file path
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Display first few rows
print("First few rows of the dataset:")
print(data.head())

# Balance the dataset (equal Real & Fake news)
real_news = data[data["label"] == "Real"]
fake_news = data[data["label"] == "Fake"]
min_samples = min(len(real_news), len(fake_news))
balanced_data = pd.concat([real_news.sample(min_samples, random_state=42), fake_news.sample(min_samples, random_state=42)])

# Encode labels (Fake = 1, Real = 0)
label_encoder = LabelEncoder()
balanced_data["label"] = label_encoder.fit_transform(balanced_data["label"])

# Split data
X = balanced_data["text"]
y = balanced_data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer with bigrams/trigrams
vectorizer = TfidfVectorizer(max_features=7000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Train Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=200, random_state=42)
random_forest_model.fit(X_train_tfidf, y_train)

# Train Naïve Bayes Model (New addition)
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_tfidf, y_train)

# Evaluate models
y_pred_logistic = logistic_model.predict(X_test_tfidf)
y_pred_rf = random_forest_model.predict(X_test_tfidf)
y_pred_nb = naive_bayes_model.predict(X_test_tfidf)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"Logistic Regression Model Accuracy: {accuracy_logistic:.2f}")
print(f"Random Forest Model Accuracy: {accuracy_rf:.2f}")
print(f"Naïve Bayes Model Accuracy: {accuracy_nb:.2f}")

# Function to predict if news is fake using all models
def fakenewsprediction(text):
    text_tfidf = vectorizer.transform([text])
    pred_logistic = logistic_model.predict(text_tfidf)[0]
    pred_rf = random_forest_model.predict(text_tfidf)[0]
    pred_nb = naive_bayes_model.predict(text_tfidf)[0]
    
    # Majority vote (if at least 2 models predict fake, classify as fake)
    fake_votes = sum([pred_logistic, pred_rf, pred_nb])
    
    if fake_votes >= 2:
        return "Fake News"
    else:
        return "Real News"

# Expanded test cases (Includes clear fake news examples)
news_samples = [
    "Scientists discover new species of deep-sea fish",
    "World War III may start soon, experts warn",
    "NASA confirms water on Mars",
    "Celebrity admits to using time machine",
    "Bitcoin reaches record high of $100,000",
    "Aliens have landed on Earth and are living among us",
    "Government to ban all mobile phones next year",
    "Scientists prove that the Earth is actually flat",
    "Secret underground city discovered beneath New York",
    "Time travel is now possible, claims scientist",
    "COVID-19 vaccine turns people into zombies",
    "Elon Musk announces first human trip to Mars next year",
    "Politician caught shapeshifting into reptilian alien",
    "Scientists confirm the existence of a parallel universe",
    "Government to introduce new law banning all electronic devices"
]

for news_text in news_samples:
    prediction = fakenewsprediction(news_text)
    print(f"News: {news_text} | Prediction: {prediction}")

