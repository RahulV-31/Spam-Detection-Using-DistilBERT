# Spam-Detection-Using-DistilBERT

This project focuses on detecting spam messages using **DistilBERT** for feature extraction and **Logistic Regression** for classification. The goal is to utilize a lightweight yet powerful transformer model to accurately distinguish between spam and non-spam (ham) emails.

---

## 🚀 Project Highlights

- 🔍 Uses **DistilBERT** from Hugging Face Transformers to convert raw email text into high-dimensional embeddings.
- 🧠 Applies **Logistic Regression** as a lightweight yet effective classifier.
- 📊 Achieves high accuracy and balanced classification metrics on a large real-world email dataset.
- ⚡ Efficient pipeline with minimal preprocessing — handles raw text intelligently.

---

## 🗂️ Dataset Information

**Dataset Source:**  
[Email Spam Classification Dataset on Kaggle](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)

### 📄 Overview:

- **File Name:** `emails.csv`
- **Total Records:** 83,446 emails
- **Columns:**
  - `text`: The full content of the email.
  - `spam`: Binary label for classification:
    - `1` = Spam  
    - `0` = Ham (not spam)

### 📊 Class Distribution:
- Approximately **48% spam**, **52% ham**.
- Balanced enough for reliable binary classification.

---

## 🧪 Workflow

### 1. **Library Setup**
Required Python libraries are installed: `transformers`, `torch`, `scikit-learn`, `pandas`, `tqdm`.

### 2. **Data Loading & Exploration**
- The dataset is read using `pandas`.
- Distribution of spam vs. ham is analyzed.
- Optional exploratory visualization can be added.

### 3. **Tokenization with DistilBERT**
- Uses Hugging Face’s `DistilBERTTokenizerFast` to tokenize emails.
- Extracts `[CLS]` token embeddings which represent the entire email.

### 4. **Feature Extraction**
- Passes tokenized text through **DistilBERT** model.
- Extracts hidden state of the `[CLS]` token as a 768-dimensional feature vector.

### 5. **Train-Test Split**
- The dataset is split into 80% training and 20% testing using `train_test_split`.

### 6. **Model Training**
- A **Logistic Regression** classifier is trained on the DistilBERT features.

### 7. **Model Evaluation**
- Evaluated using `accuracy`, `precision`, `recall`, and `f1-score`.

---

## 📈 Model Performance

```
Logistic Regression Model Performance:

              precision    recall  f1-score   support

           0       0.96      0.96      0.96      7938
           1       0.96      0.97      0.96      8752

    accuracy                           0.96     16690
   macro avg       0.96      0.96      0.96     16690
weighted avg       0.96      0.96      0.96     16690
```

### ✅ Summary:
- **Overall Accuracy:** 96%
- Balanced performance across both classes.
- Very strong recall for spam emails (97%) ensures low false negatives.

---

## 📌 Key Takeaways

- **DistilBERT** effectively captures context and semantics in raw email text without complex preprocessing.
- Combining it with a traditional classifier like **Logistic Regression** yields high accuracy with low computational overhead.
- Suitable for real-world deployment in email filtering and security systems.

---

## 💡 Future Improvements

- Deploy using **Flask**, **FastAPI**, or **Streamlit**.
- Extend model to support multi-language spam detection.
- Compare performance with other Transformer models (e.g., BERT, RoBERTa, ALBERT).
- Add visualization of confusion matrix, ROC-AUC, etc.

---


## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository, create a new branch, and open a pull request.

---
