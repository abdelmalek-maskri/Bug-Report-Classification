########## 1. Import Required Libraries ##########
import pandas as pd
import numpy as np
import re
import os
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec

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

# Initialize stemmer
stemmer = PorterStemmer()

def apply_stemming(text):
    """Apply stemming to each word in the text."""
    return ' '.join([stemmer.stem(word) for word in text.split()])

########## 3. Load and Preprocess Data ##########

project = 'incubator-mxnet'  # Change dataset if needed
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

# Keep only necessary columns
pd_tplusb = pd_all.rename(columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"})

data = pd_tplusb[['id', 'sentiment', 'text']].fillna('')

# Text cleaning
text_col = 'text'
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)
data[text_col] = data[text_col].apply(apply_stemming)  # Comment this out if you want to try without stemming

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])

# Tokenize text
tokenized_texts = [text.split() for text in data[text_col]]

########## 4. Train Word2Vec Model ##########

# Initialize and train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_texts, 
                    vector_size=100,  # Dimension of word vectors
                    window=5,         # Context window size
                    min_count=1,      # Ignore words that appear less than this
                    workers=4,        # Number of threads
                    sg=1)             # Use skip-gram (1) instead of CBOW (0)

print(f"Word2Vec model trained on {len(w2v_model.wv.key_to_index)} words")

# Function to convert sentence to Word2Vec vector
def sentence_to_vec(sentence, model, vector_size=100):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(vector_size)  # Return zero vector if no words found
    return np.mean(word_vectors, axis=0)  # Average word vectors

# Convert all texts to Word2Vec embeddings
X = np.array([sentence_to_vec(sentence, w2v_model) for sentence in data[text_col]])
y = data['sentiment'].values  # Convert to NumPy array

# Store metrics across 10 runs
accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

for repeated_time in range(REPEAT):
    # Train-test split
    train_index, test_index = train_test_split(
        np.arange(data.shape[0]), test_size=0.2, random_state=repeated_time
    )

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train Gaussian Naive Bayes model (suitable for continuous features like word embeddings)
    clf = GaussianNB()
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

    # Store results
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
print(f"\n=== Naive Bayes + Word2Vec Results on {project} Dataset ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {avg_accuracy:.4f}")
print(f"Average Precision:     {avg_precision:.4f}")
print(f"Average Recall:        {avg_recall:.4f}")
print(f"Average F1 Score:      {avg_f1:.4f}")
print(f"Average AUC:           {avg_auc:.4f}")

import joblib

# Save the model and Word2Vec embeddings
joblib.dump(clf, 'nb_word2vec_model.pkl')
w2v_model.save("word2vec.model")