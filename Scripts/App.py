import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load model and vectorizer
lr = pickle.load(open('../Artifacts/lr.pkl', 'rb'))
cv = pickle.load(open('../Artifacts/vectorizer.pkl', 'rb'))

# Initialize stemmer
stemmer = PorterStemmer()

# Page configuration
st.set_page_config(page_title="Spam Mail Detector", page_icon="📩", layout="centered")

# Title and subtitle
st.title("📩 Spam Mail Detector")
st.markdown("""
Welcome to the **Spam Mail Detector**!  
Enter any message below, and our AI model will tell you if it's **SPAM** or **NOT SPAM**.  
Made with ❤️ using Machine Learning and Streamlit.
""")

# Input field
msg = st.text_input("✍️ Enter your message here:")

# Preprocessing function
def preprocessing(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove symbols & numbers
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Prediction
if msg:
    processed_msg = preprocessing(msg)
    vectorized_msg = cv.transform([processed_msg]).toarray()
    prediction = lr.predict(vectorized_msg)

    st.markdown("---")  # Divider

    if prediction[0] == True:
        st.error("🚨 Warning: This message is **SPAM**.")
    else:
        st.success("✅ Good news: This message is **NOT SPAM**.")

    # Optional explanation
    with st.expander("🔍 What just happened?"):
        st.markdown("""
        Your message was cleaned and transformed using NLP, then passed into a trained machine learning model (Logistic Regression) that learned from thousands of spam messages.
        """)
