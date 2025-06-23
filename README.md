# Bug Report Classification using XGBoost & TF-IDF

## 📌 Project Overview
The goal of this project is to **build an intelligent tool** that classifies bug reports as either:
- **Performance bug-related** (e.g., accuracy/inference speed issues).
- **Non-performance bug-related** (other issues).

### **Approach**
- I use **TF-IDF** to convert bug report text into numerical features.
- A **XGBoost classifier** is trained to predict whether a bug report is performance-related or not.
- The classifier is evaluated using **precision, recall, F1-score, and AUC**.

## 🔧 Implemented Features
✔️ **Preprocessing of bug report text** (removal of HTML, emojis, stopwords, lemmatization)  
✔️ **Feature extraction using TF-IDF**  
✔️ **XGBoost classifier for bug report classification**  
✔️ **Evaluation with precision, recall, F1-score, and AUC**  
✔️ **Performance keyword boosting for improved classification**  
✔️ **Statistical analysis for performance comparison**  

## 📊 Running the Code
1. **Ensure dependencies are installed**:
   ```bash
   pip install pandas numpy nltk scikit-learn xgboost
   ```
2. **Run the classification tool**:
   ```bash
   python main.py
   ```

3. **Modify dataset selection** (optional):
   Change the dataset being used in `main.py`:
   ```python
   project = 'tensorflow'  # Options: pytorch, tensorflow, keras, incubator-mxnet, caffe
   ```

4. **Check the saved CSV file**:
   - After running the script, results are saved in a CSV file named after the selected project.
   - Example: If the project is `tensorflow`, the results will be saved as `tensorflow_NB.csv`.

## 📂 Output Files
- **CSV files with classification results** (saved in the root directory)
- **All raw result CSVs** are stored under the `outputs/` folder for easy access and analysis.
- **Model artifacts (if enabled)**: Uncomment the following in `main.py` to save models:
   ```python
   import joblib
   joblib.dump(clf, 'xgboost_bug_report_model.pkl')
   joblib.dump(tfidf, 'tfidf_vectorizer.pkl')


## 📖 Documentation
- **[Requirements](requirements.pdf)**: Dependencies and setup instructions.
- **[Manual](manual.pdf)**: Guide on running and modifying the tool.
- **[Replication Guide](replication.pdf)**: Steps to reproduce the results.

---
This project enhances bug report classification using **XGBoost & TF-IDF**, improving accuracy over Naive Bayes. 🚀

