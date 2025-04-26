# Spam Detection Model

## Project Overview

This project focuses on building a machine learning model that can accurately classify emails as **spam** or **not spam**. It involves data preprocessing, feature extraction using NLP techniques, and training various classifiers to achieve the best performance.

---

## Problem Statement

Spam emails are a major annoyance and pose security risks such as phishing and malware. The goal is to develop an automated system that detects spam messages based on their content.

---

## Techniques Used

- **Natural Language Processing (NLP)**
- **Machine Learning Classification**
- **Text Vectorization (TF-IDF, CountVectorizer)**

---

## Dataset

- **Source**: [Specify source if any, e.g., UCI Spam Dataset]
- **Features**: Text content of emails
- **Labels**: `1` for spam, `0` for not spam

---

## Workflow

1. **Data Cleaning**  
   - Lowercasing, removing punctuation, stopwords, and special characters.

2. **Tokenization & Lemmatization**

3. **Feature Extraction**  
   - CountVectorizer or TF-IDF

4. **Model Training**  
   - Naive Bayes  
   - Logistic Regression  
   - Decision Tree
   - Random forest 

5. **Evaluation**  
   - Accuracy  
   - Precision, Recall, F1 Score  
   - Confusion Matrix

---

## Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 96.7%    | 98.4%     | 86.0%  | 95.7%    |


