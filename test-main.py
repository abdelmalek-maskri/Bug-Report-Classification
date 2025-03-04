import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Load the saved model and vectorizer
clf = joblib.load('xgboost_bug_report_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Define the same preprocessing functions
def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english') + ['...'])
    return " ".join([word for word in text.split() if word not in stop_words])

def clean_str(string):
    return re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string).strip().lower()

# Function to preprocess and predict
def predict_sentiment(user_input):
    # Preprocess the input
    user_input = remove_html(user_input)
    user_input = remove_emoji(user_input)
    user_input = remove_stopwords(user_input)
    user_input = clean_str(user_input)

    # Transform the input using the saved TF-IDF vectorizer
    X_user = tfidf.transform([user_input]).toarray()

    # Make a prediction
    prediction = clf.predict(X_user)
    prediction_proba = clf.predict_proba(X_user)

    # Map prediction to sentiment (assuming 0 = negative, 1 = positive)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    confidence = prediction_proba[0][prediction[0]]

    return sentiment, confidence

# Test the classifier with a file
if __name__ == "__main__":
    import sys

    # Check if a filename is provided
    if len(sys.argv) != 2:
        print("Usage: python test_classifier_file.py <filename>")
        sys.exit(1)

    # Get the filename from command-line arguments
    filename = sys.argv[1]

    # Read the file
    try:
        if filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(filename)

            # Merge 'Title' and 'Body' into a single 'text' column
            df['text'] = df.apply(
                lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
                axis=1
            )
            
            # Get the actual class (ground truth)
            df['actual_class'] = df['class'].map({0: 'Negative', 1: 'Positive'})

            bug_reports = df[['text', 'actual_class']].values.tolist()
        elif filename.endswith('.txt'):
            # Read text file (one bug report per line)
            with open(filename, 'r') as file:
                bug_reports = [[line.strip(), 'Unknown'] for line in file]
        else:
            print("Unsupported file format. Please provide a CSV or TXT file.")
            sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)

    # Print header
    print("\n=== Actual vs. Predicted Labels ===")
    print(f"{'Index':<6} {'Actual Class':<15} {'Predicted Class':<15} {'Confidence':<10}")
    print("-" * 60)

    # Predict sentiment for each bug report
    for i, (bug_report, actual_class) in enumerate(bug_reports):
        if bug_report:
            sentiment, confidence = predict_sentiment(bug_report)
            print(f"{i+1:<6} {actual_class:<15} {sentiment:<15} {confidence:.2f}")
