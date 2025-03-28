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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def apply_lemmatization(text):
    """Lemmatize each word in the text."""
    return ' '.join([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in text.split()])

performance_terms = ['slow', 'speed', 'fast', 'memory', 'cpu', 'gpu', 'performance',
                     'latency', 'throughput', 'bottleneck', 'optimization', 'efficient',
                     'regression', 'benchmark', 'overhead', 'usage']

def boost_performance_keywords(text):
    """Repeat performance-related keywords to boost their importance"""
    for term in performance_terms:
        if term in text.lower():
            # Repeat the term to increase its TF-IDF weight
            text = text + " " + term + " " + term
    return text

########## 3. Train on a Single Dataset Over 10 Runs ##########

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'tensorflow'
path = f'datasets/{project}.csv'
REPEAT = 10

if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found at {path}")

# Load dataset
pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

# Merge Title and Body into a single column
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns
pd_tplusb = pd_all.rename(columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"})

pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

datafile = 'Title+Body.csv'
data = pd.read_csv(datafile).fillna('')

original_data = data.copy()

# Text cleaning pipeline
text_col = 'text'
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)
data[text_col] = data[text_col].apply(apply_lemmatization)  # Added lemmatization
data[text_col] = data[text_col].apply(boost_performance_keywords)

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])

# 3) Output CSV file name
out_csv_name = f'./{project}_NB.csv'

# Store metrics across 10 runs
accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

for repeated_time in range(REPEAT):
    # Train-test split
    train_index, test_index = train_test_split(
    np.arange(data.shape[0]), test_size=0.2, random_state=repeated_time, stratify=data['sentiment'])
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
    X_train = tfidf.fit_transform(data[text_col].iloc[train_index]).toarray()
    X_test  = tfidf.transform(data[text_col].iloc[test_index]).toarray()
    
    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]

    # Model training
    # 1. Add class weighting to balance the classes
    clf = XGBClassifier(
        learning_rate=0.1, 
        max_depth=3, 
        n_estimators=100,
        scale_pos_weight=5,  # Give more weight to positive class
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = clf.predict(X_test)
    y_pred_probs = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_probs)
    else:
        auc = 0.5

    # Store results
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    auc_values.append(auc)

# Compute averages
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1_scores)
avg_auc = np.mean(auc_values)

# Print results
print(f"\n=== XGBoost + TF-IDF Results on {project} Dataset ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {avg_accuracy:.4f}")
print(f"Average Precision:     {avg_precision:.4f}")
print(f"Average Recall:        {avg_recall:.4f}")
print(f"Average F1 Score:      {avg_f1:.4f}")
print(f"Average AUC:           {avg_auc:.4f}")

# Save final results to CSV (append mode)
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'repeated_times': [REPEAT],
        'Accuracy': [avg_accuracy],
        'Precision': [avg_precision],
        'Recall': [avg_recall],
        'F1': [avg_f1],
        'AUC': [avg_auc],
        'CV_list(AUC)': [str(auc_values)]
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have also been saved to: {out_csv_name}")


# Save artifacts 
#Uncomment this if you want to use the classifier later ( for example build a GUI)
# import joblib
# joblib.dump(clf, 'xgboost_bug_report_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')