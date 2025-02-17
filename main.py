########## 1. Import Required Libraries ##########
import pandas as pd
import numpy as np
import re
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define Text Preprocessing Methods ##########

def remove_html(text):
    """Remove HTML tags using regex."""
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    """Remove emojis using regex."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

NLTK_stop_words_list = stopwords.words('english')
final_stop_words_list = NLTK_stop_words_list + ['...']

def remove_stopwords(text):
    """Remove stopwords from text."""
    return " ".join([word for word in text.split() if word not in final_stop_words_list])

def clean_str(string):
    """Remove non-alphanumeric characters, normalize text."""
    return re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string).strip().lower()

########## 3. Train on a Single Dataset Over 10 Runs ##########

# Choose the dataset
project = 'tensorflow'  # Change this to another dataset if needed
path = f'datasets/{project}.csv'
REPEAT = 10  # Number of runs

if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found at {path}")

# Load dataset
pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"})
data = pd_tplusb.fillna('')

# Text cleaning
text_col = 'text'
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])

# Convert text to TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X = tfidf.fit_transform(data[text_col]).toarray()
y = data['sentiment']

# Store metrics across 10 runs
accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

for repeated_time in range(REPEAT):
    # Train-test split (Fixed for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=repeated_time, stratify=y
    )

    # Train XGBoost model
    clf = XGBClassifier(
        learning_rate=0.1, 
        max_depth=3, 
        n_estimators=100, 
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_probs = clf.predict_proba(X_test)[:, 1]

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Fix AUC calculation
    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_probs)
    else:
        auc = 0.5  # Set AUC to 0.5 if only one class exists

    # Store results for this run
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    auc_values.append(auc)

# Compute average results across 10 runs
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1_scores)
avg_auc = np.mean(auc_values)

# Print final averaged results
print(f"\n=== XGBoost + TF-IDF Results on {project} Dataset ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {avg_accuracy:.4f}")
print(f"Average Precision:     {avg_precision:.4f}")
print(f"Average Recall:        {avg_recall:.4f}")
print(f"Average F1 Score:      {avg_f1:.4f}")
print(f"Average AUC:           {avg_auc:.4f}")
