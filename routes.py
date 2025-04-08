from flask import Blueprint, request, jsonify, render_template
import joblib
import os
from app.model import clean_text  # Import from model.py

# Initialize Blueprint
routes = Blueprint("routes", __name__)

# Load the saved model and vectorizers
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model")

tfidf_vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
chi2_selector = joblib.load(os.path.join(MODEL_PATH, "chi2_selector.pkl"))
spam_classifier = joblib.load(os.path.join(MODEL_PATH, "spam_classifier.pkl"))

# Home Route
@routes.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@routes.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    text = data["message"]
    cleaned_text = clean_text(text)

    # Transform input
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    text_chi2 = chi2_selector.transform(text_tfidf)

    # Prediction
    prediction = spam_classifier.predict(text_chi2)[0]
    label = "Spam" if prediction == 1 else "Ham"

    return jsonify({"message": text, "prediction": label})
