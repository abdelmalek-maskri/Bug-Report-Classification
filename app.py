import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# Load Pretrained Model & TF-IDF Vectorizer
MODEL_PATH = "xgboost_bug_report_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)

clf = load_model()
tfidf = load_vectorizer()

# Initialize NLTK
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Define Text Preprocessing Functions
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Keep only alphanumeric characters
    text = text.lower().strip()  # Convert to lowercase
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def apply_stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Performance Keywords Boosting
performance_terms = ['slow', 'speed', 'fast', 'memory', 'cpu', 'gpu', 'performance',
                     'latency', 'throughput', 'bottleneck', 'optimization', 'efficient',
                     'regression', 'benchmark', 'overhead', 'usage']

def boost_keywords(text):
    for term in performance_terms:
        if term in text:
            text += f' {term} {term}'  # Boost TF-IDF weight
    return text

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = apply_stemming(text)
    text = boost_keywords(text)
    return text

# Streamlit UI
st.title("Bug Report Classifier")
st.write("Upload a bug report or type text to classify whether it's performance-related.")

# User Input Options
input_method = st.radio("Choose Input Method", ("Manual Entry", "Upload Text File", "Upload CSV"))

if input_method == "Manual Entry":
    user_text = st.text_area("Enter bug report text:")
elif input_method == "Upload Text File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        user_text = uploaded_file.read().decode("utf-8")
    else:
        user_text = ""
elif input_method == "Upload CSV":
    csv_file = st.file_uploader("Upload a CSV file (must contain a 'text' column)", type=["csv"])
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column!")
        else:
            df['processed_text'] = df['text'].apply(preprocess_text)
            X_input = tfidf.transform(df['processed_text']).toarray()
            df['Prediction'] = clf.predict(X_input)
            df['Probability'] = clf.predict_proba(X_input)[:, 1]
            df['Performance Bug?'] = df['Prediction'].apply(lambda x: 'Yes' if x == 1 else 'No')
            st.write("### CSV Classification Results")
            st.dataframe(df[['text', 'Performance Bug?', 'Probability']])
            df.hist(column='Probability', bins=10, figsize=(6, 4), grid=False)
            plt.xlabel("Probability of Performance Bug")
            plt.ylabel("Frequency")
            st.pyplot(plt)

if user_text:
    st.subheader("Preprocessing... ✨")
    processed_text = preprocess_text(user_text)
    st.write(f"**Processed Text:** {processed_text}")
    
    # Convert to TF-IDF features
    X_input = tfidf.transform([processed_text]).toarray()
    
    # Prediction
    prediction = clf.predict(X_input)[0]
    prob = clf.predict_proba(X_input)[0][1]
    
    # Display Result
    st.subheader("Classification Result 🔍")
    if prediction == 1:
        st.success(f"**Performance Bug Detected!** (Confidence: {prob:.2f})")
    else:
        st.error(f"**Not a Performance Bug.** (Confidence: {prob:.2f})")
    
    # Highlight Important Keywords
    st.subheader("Key Influencing Words")
    feature_names = tfidf.get_feature_names_out()
    word_importances = X_input[0] * clf.feature_importances_
    important_words = sorted(zip(feature_names, word_importances), key=lambda x: x[1], reverse=True)[:10]
    st.write(pd.DataFrame(important_words, columns=["Word", "Importance"]))
    
    # Display Probability Distribution
    plt.figure(figsize=(6, 4))
    plt.barh([w[0] for w in important_words], [w[1] for w in important_words])
    plt.xlabel("TF-IDF Weight * Feature Importance")
    plt.ylabel("Top Influencing Words")
    plt.title("Top Words Influencing Prediction")
    st.pyplot(plt)
