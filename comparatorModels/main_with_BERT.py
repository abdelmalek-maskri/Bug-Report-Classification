import pandas as pd
import numpy as np
import re
import os
import nltk
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, models

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

#1. Define Text Preprocessing Methods

def remove_html(text):
    # remove HTML tags using regex
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    # remove emojis using regex
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
    # remove stopwords from text
    return " ".join([word for word in text.split() if word not in final_stop_words_list])

def clean_str(string):
    # remove non-alphanumeric characters, normalize text
    return re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string).strip().lower()

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def apply_lemmatization(text):
    # lemmatize each word in the text
    return ' '.join([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in text.split()])

performance_terms = ['slow', 'speed', 'fast', 'memory', 'cpu', 'gpu', 'performance',
                     'latency', 'throughput', 'bottleneck', 'optimization', 'efficient',
                     'regression', 'benchmark', 'overhead', 'usage']

def boost_performance_keywords(text):
    # repeat performance-related keywords to boost their importance
    for term in performance_terms:
        if term in text.lower():
            # Repeat the term to increase its TF-IDF weight
            text = text + " " + term + " " + term
    return text

# bert-base-uncased mean-pooled is plain BERT. 'sentence-transformers/all-mpnet-base-v2'
# swaps in a model tuned to produce sentence vectors, which is the stronger comparison.
MODEL_NAME = 'bert-base-uncased'

# the cleaning above is built for TF-IDF. Stopword removal, lemmatization and keyword
# repetition strip the word order and context BERT depends on, so RAW_TEXT skips them
# and embeds the report as written.
RAW_TEXT = False

def build_encoder(name):
    # wrap a plain HF checkpoint with mean pooling; sentence-transformer checkpoints
    # already carry their own pooling layer
    if name.startswith('sentence-transformers/'):
        return SentenceTransformer(name)
    word_embedding = models.Transformer(name, max_seq_length=512)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode='mean')
    return SentenceTransformer(modules=[word_embedding, pooling])

#2. Train on a Single Dataset Over 10 Runs
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'tensorflow'
path = f'datasets/{project}.csv'
REPEAT = 10

if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found at {path}")

# load dataset
pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

# merge Title and Body into a single column
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# keep only necessary columns
pd_tplusb = pd_all.rename(columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"})

pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

datafile = 'Title+Body.csv'
data = pd.read_csv(datafile).fillna('')

original_data = data.copy()

# text cleaning pipeline
text_col = 'text'
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
if not RAW_TEXT:
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)
    data[text_col] = data[text_col].apply(apply_lemmatization)  # added lemmatization
    data[text_col] = data[text_col].apply(boost_performance_keywords)

# convert labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])

# 3) output CSV file name
variant = 'BERT_raw' if RAW_TEXT else 'BERT'
out_csv_name = f'./{project}_{variant}.csv'

# BERT Vectorization
# the encoder is frozen, so embedding every report once up front cannot leak the test
# split into training: nothing is fitted here. Reports longer than 512 tokens are cut.
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
encoder = build_encoder(MODEL_NAME)
X = encoder.encode(data[text_col].tolist(), batch_size=32, device=device,
                   show_progress_bar=True, convert_to_numpy=True)
y = data['sentiment'].values

# store metrics across 10 runs
accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

for repeated_time in range(REPEAT):
    # train-test split
    train_index, test_index = train_test_split(
    np.arange(data.shape[0]), test_size=0.2, random_state=repeated_time, stratify=data['sentiment'])

    X_train = X[train_index]
    X_test  = X[test_index]

    y_train = y[train_index]
    y_test  = y[test_index]

    # model training
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

    # predictions and metrics
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

    # store results
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

# print results
print(f"\n=== XGBoost + {variant} ({MODEL_NAME}) Results on {project} Dataset ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {avg_accuracy:.4f}")
print(f"Average Precision:     {avg_precision:.4f}")
print(f"Average Recall:        {avg_recall:.4f}")
print(f"Average F1 Score:      {avg_f1:.4f}")
print(f"Average AUC:           {avg_auc:.4f}")

# save final results to CSV
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except FileNotFoundError:
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
