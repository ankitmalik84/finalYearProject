from flask import Flask, request, jsonify
import pickle
import spacy
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the pre-trained model and vectorizer
with open('countvectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def normalize_text(text):
    doc = nlp(text)
    normalized_words = [token.lemma_ for token in doc]
    normalized_text = ' '.join(normalized_words)
    return normalized_text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Combine all text fields
    combined_text = ' '.join([
        data.get('title', ''),
        data.get('location', ''),
        data.get('salary_range', ''),
        data.get('company_profile', ''),
        data.get('description', ''),
        data.get('requirements', ''),
        data.get('benefits', ''),
        data.get('employment_type', ''),
        data.get('required_experience', ''),
        data.get('required_education', ''),
        data.get('industry', ''),
        data.get('function', ''),
        data.get('department', '')
    ])

    cleaned_text = clean_text(combined_text)
    normalized_text = normalize_text(cleaned_text)

    # Create feature matrix
    text_matrix = vectorizer.transform([normalized_text])
    combined_matrix = hstack([text_matrix])

    # Make prediction
    prediction = model.predict(combined_matrix)
    result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

    return jsonify({'message': f'The job posting is {result}'})

if __name__ == '__main__':
    app.run(debug=True)
