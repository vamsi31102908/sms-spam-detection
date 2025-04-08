import nltk
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Load and Preprocess Data
def load_data():
    file_path = "dataset/super_sms_dataset.csv"  # Ensure dataset is in the correct folder
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['Labels', 'SMSes']].rename(columns={'Labels': 'label', 'SMSes': 'text'})
    df['text'] = df['text'].astype(str)
    return df

# Step 2: Text Preprocessing
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Step 3: Feature Engineering
def feature_engineering(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    chi2_selector = SelectKBest(chi2, k=1000)
    X_train_chi2 = chi2_selector.fit_transform(X_train_tfidf, y_train)
    X_test_chi2 = chi2_selector.transform(X_test_tfidf)

    return X_train_chi2, X_test_chi2, tfidf, chi2_selector

# Step 4: Handle Class Imbalance
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

# Step 5: Build Hybrid Model
def build_hybrid_model():
    nb = MultinomialNB(alpha=0.1)
    svm = SVC(kernel='linear', probability=True, C=1, random_state=42)
    return VotingClassifier(estimators=[('nb', nb), ('svm', svm)], voting='soft', weights=[1, 1.5])

# Main Execution
if __name__ == "__main__":
    df = load_data()
    df['text'] = df['text'].apply(clean_text)
    df.dropna(subset=['label'], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    X_train_chi2, X_test_chi2, tfidf, chi2_selector = feature_engineering(X_train, X_test)
    X_train_res, y_train_res = balance_data(X_train_chi2, y_train)

    model = build_hybrid_model()
    model.fit(X_train_res, y_train_res)

    # Evaluate model
    y_pred = model.predict(X_test_chi2)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

    # Save model components
    joblib.dump(tfidf, "saved_model/tfidf_vectorizer.pkl")
    joblib.dump(chi2_selector, "saved_model/chi2_selector.pkl")
    joblib.dump(model, "saved_model/spam_classifier.pkl")

    print("✅ Model training complete. Files saved in 'saved_model/' folder.")
