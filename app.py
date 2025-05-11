# ✅ app.py — Fake News Classifier + Exploratory Analysis

import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.sparse import hstack


# --------------------------
# Load pre-trained components
# --------------------------
model = joblib.load("models/xgb_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------------
# Custom preprocessing
# --------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

def extract_title_features(title):
    if not title:
        return [0, 0, 0.0, 0]
    exclam = title.count("!")
    upper = sum(1 for c in title if c.isupper())
    ratio = upper / len(title) if len(title) > 0 else 0
    length = len(title)
    return [exclam, upper, ratio, length]

# --------------------------
# App Layout
# --------------------------
st.set_page_config(page_title="Fake News App", layout="wide")
st.title("📰 Fake News Analyzer")

menu = st.sidebar.radio("Navigation", ["🔍 Classify an Article", "📊 LDA Analysis", "📁 Upload CSV File"])

if menu == "🔍 Classify an Article":
    with st.form("news_form"):
        title = st.text_input("Title (optional)")
        text = st.text_area("Article content")
        submitted = st.form_submit_button("Analyze")

    if submitted:
        cleaned_title = clean_text(title)
        cleaned_text = clean_text(text)
        combined = (cleaned_title + " " + cleaned_text).strip() if title else cleaned_text

        tfidf_vector = vectorizer.transform([combined])
        extra_features = scaler.transform([extract_title_features(title)])
        full_input = hstack([tfidf_vector, extra_features])

        prediction = model.predict(full_input)[0]
        proba = model.predict_proba(full_input)[0]
        label = "🟡 UNDETERMINED"
        if proba[0] > 0.55:
            label = "🔴 FAKE"
        elif proba[1] > 0.55:
            label = "🟢 REAL"
        st.subheader(f"Prediction Result: {label}")
        st.caption(f"Probabilities → FAKE: {proba[0]:.2%} | REAL: {proba[1]:.2%}")

        st.markdown("### 🔤 Word Cloud")
        wc = WordCloud(width=800, height=400, background_color='white').generate(combined)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

elif menu == "📊 LDA Analysis":
    st.subheader("Exploring Latent Themes (LDA)")
    example_texts = [
        "the president met with ministers at the white house",
        "hillary clinton is a liar watch this shocking video",
        "breaking news terrorist attack in london",
        "scientists discover new cure for cancer",
        "donald trump exposes massive election fraud"
    ]
    X_tfidf = vectorizer.transform(example_texts)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X_tfidf)

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            st.markdown(f"**🟦 Topic {topic_idx+1}**: {', '.join(top_features)}")

    print_top_words(lda, vectorizer.get_feature_names_out(), 10)

elif menu == "📁 Upload CSV File":
    st.subheader("Predict News from CSV File")
    uploaded_file = st.file_uploader("Upload CSV with 'title' and 'text' columns", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["title"] = df["title"].fillna("")
        df["text"] = df["text"].fillna("")

        df["combined"] = df["title"].apply(clean_text) + " " + df["text"].apply(clean_text)
        tfidf = vectorizer.transform(df["combined"])
        title_feats = np.vstack([extract_title_features(t) for t in df["title"]])
        scaled = scaler.transform(title_feats)
        final_input = hstack([tfidf, scaled])

        df["prediction_proba"] = model.predict_proba(final_input).max(axis=1)
        df["prediction"] = model.predict(final_input)
        df["prediction_label"] = "UNDETERMINED"
        df.loc[(df["prediction"] == 0) & (df["prediction_proba"] > 0.55), "prediction_label"] = "FAKE"
        df.loc[(df["prediction"] == 1) & (df["prediction_proba"] > 0.55), "prediction_label"] = "REAL"

        if "expected" in df.columns:
            df["correct"] = df["prediction_label"].str.upper() == df["expected"].str.upper()
            st.dataframe(df[["title", "prediction_label", "expected", "correct"]])
        else:
            st.dataframe(df[["title", "prediction_label"]])

        st.markdown("### 🔤 Word Cloud from All Articles")
        all_text = " ".join(df["combined"])
        wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
