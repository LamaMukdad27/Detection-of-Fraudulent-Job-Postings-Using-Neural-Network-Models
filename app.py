import streamlit as st
import numpy as np
import re
import nltk
import joblib
import tensorflow as tf
import gdown
import os

from nltk.corpus import stopwords

# DOWNLOAD MODEL
MODEL_PATH = "ann_model.keras"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=16HH3agQO5Y5Sbi5u40LqNsY4AjqBEKT3"
    with st.spinner("Downloading model..."):
        gdown.download(url, MODEL_PATH, quiet=False)

# LOAD MODEL & VECTORIZER=
model = tf.keras.models.load_model(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# NLP SETUP
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# STREAMLIT UI
st.set_page_config(page_title="Fraud Job Detector", layout="centered")

st.title("🧠 Job Fraud Detection System")
st.write("AI-powered detection of fake job postings")

mode = st.radio("Select Input Mode:", ["Full Text", "Structured Input"])

# PREDICTION FUNCTION
def predict(text):
    cleaned = clean_text(text)

    X = vectorizer.transform([cleaned]).toarray().astype(np.float32)

    prob = model.predict(X, verbose=0)[0][0]
    pred = int(prob > 0.5)

    return pred, prob

# FULL TEXT MODE
if mode == "Full Text":
    job_text = st.text_area("Paste Job Description")

    if st.button("Predict"):
        if job_text.strip() == "":
            st.warning("Please enter job description")
        else:
            pred, prob = predict(job_text)

            st.subheader("Result")

            if pred == 1:
                st.error(f"⚠ Fraudulent Job (Confidence: {prob:.2f})")
            else:
                st.success(f"✔ Legitimate Job (Confidence: {prob:.2f})")

# STRUCTURED INPUT MODE
else:
    title = st.text_input("Job Title")
    company = st.text_input("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")

    if st.button("Predict"):
        combined_text = f"{title} {company} {description} {requirements} {benefits}"

        if combined_text.strip() == "":
            st.warning("Please fill at least some fields")
        else:
            pred, prob = predict(combined_text)

            st.subheader("Result")

            if pred == 1:
                st.error(f"⚠ Fraudulent Job (Confidence: {prob:.2f})")
            else:
                st.success(f"✔ Legitimate Job (Confidence: {prob:.2f})")

# FOOTER
st.markdown("---")
st.markdown("Built with TensorFlow + Streamlit + TF-IDF")
