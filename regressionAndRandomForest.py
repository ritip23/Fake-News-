import pandas as pd
import nltk
from flask import Flask, request, jsonify, render_template
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Ensure NLTK downloads are available
nltk.download('punkt')

# Define the file path
file_path = "C:/Users/RITI/Desktop/COEP study/SEM 6/Dsci/PROJECT/cleaned_dataset.csv"

# Load dataset
data = pd.read_csv(file_path)

# Apply word stemming to improve feature extraction
ps = PorterStemmer()

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())  # Tokenization
    stemmed_words = [ps.stem(word) for word in words if word.isalnum()]  # Stemming
    return " ".join(stemmed_words)

# Apply stemming to text
data["text"] = data["text"].astype(str).apply(preprocess_text)

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

# Create TF-IDF vectorizer with extended n-gram range
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Train Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=300, random_state=42)
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
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    
    pred_logistic = logistic_model.predict(text_tfidf)[0]
    pred_rf = random_forest_model.predict(text_tfidf)[0]
    pred_nb = naive_bayes_model.predict(text_tfidf)[0]
    
    # Majority vote (if at least 2 models predict fake, classify as fake)
    fake_votes = sum([pred_logistic, pred_rf, pred_nb])
    
    if fake_votes >= 2:
        return "Fake News"
    else:
        return "Real News"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data
    if not data or 'news' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    news_text = data['news']
    prediction = fakenewsprediction(news_text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
