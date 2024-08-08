import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

# Load Multinomial Naive Bayes model
with open('naive_bayes_model.pkl', 'rb') as file:
    naive_bayes_model = pickle.load(file)

# Load SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Load Decision Tree model
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

# Load CountVectorizer
with open('countvectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


import re
import nltk
import spacy
from nltk.tokenize import word_tokenize

# Load spaCy model
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


def predict(fixed_text):
    cleaned_text = clean_text(fixed_text)
    normalized_text = normalize_text(cleaned_text)
    
    # Vectorize the text
    text_matrix = vectorizer.transform([normalized_text])
    
    # Example: Predict using Logistic Regression
    prediction = logistic_model.predict(text_matrix)
    
    return prediction

# Example usage
input_text = "Your input text goes here"
prediction = predict(input_text)
print("Prediction:", prediction)



def preprocess_input(title, location, salary_range, company_profile, description, requirements, benefits, employment_type, required_experience, required_education, industry, function, department):
    # Combine all fields into a single text
    combined_text = f"{title} {location} {salary_range} {company_profile} {description} {requirements} {benefits} {employment_type} {required_experience} {required_education} {industry} {function} {department}"
    
    # Clean and normalize the text
    cleaned_text = clean_text(combined_text)
    normalized_text = normalize_text(cleaned_text)
    
    return normalized_text



def vectorize_input(normalized_text, vectorizer):
    text_matrix = vectorizer.transform([normalized_text])
    return text_matrix


def predict(fixed_text, model, vectorizer):
    # Preprocess and vectorize the input text
    normalized_text = preprocess_input(*fixed_text)
    text_matrix = vectorize_input(normalized_text, vectorizer)
    
    # Make the prediction
    prediction = model.predict(text_matrix)
    
    return prediction


# Example input
input_data = (
    "Director of Engineering",
    "San Francisco, CA",
    "$120,000 - $150,000",
    "Innovative tech company",
    "Lead the engineering team in developing cutting-edge solutions.",
    "10+ years of experience, Bachelor's degree in Computer Science, proficiency in Python and Java.",
    "Health insurance, 401(k), flexible hours.",
    "Full-time",
    "10 years",
    "Bachelor's Degree",
    "Technology",
    "Engineering",
    "Development"
)

# Example usage
prediction = predict(input_data, logistic_model, vectorizer)
print("Prediction:", "Fraudulent" if prediction[0] == 1 else "Not Fraudulent")