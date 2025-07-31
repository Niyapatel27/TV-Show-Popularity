import streamlit as st
import pandas as pd
import numpy as np
import ast
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("IMDd Top Indian TV Shows.csv")

    for col in ['Genre', 'Creators', 'Stars']:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    df['Duration'] = df['Duration'].str.extract('(\\d+)').astype(float)

    numeric_cols = ['Rating', 'No. of Ratings', 'No. of Episodes', 
                    'Reviews (Users)', 'Reviews (Critics)', 'Seasons']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Storyline'] = df['Storyline'].fillna('')
    df['Language'] = df['Language'].fillna('Unknown')

    df['Polarity'] = df['Storyline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Subjectivity'] = df['Storyline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    return df

# Train popularity model
@st.cache_resource
def train_model(df):
    df['Popularity'] = df['Rating'].apply(lambda x: 'High' if x >= 8.5 else 'Low')

    X = df[['No. of Ratings', 'No. of Episodes', 'Seasons', 'Polarity', 'Subjectivity']]
    y = df['Popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return model, train_acc, test_acc, report_df

# Build recommender
def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Storyline'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['Name']).drop_duplicates()

    def recommend(title, num=5):
        if title not in indices:
            return ["TV Show not found."]
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num+1]
        show_indices = [i[0] for i in sim_scores]
        return df['Name'].iloc[show_indices].tolist()

    return recommend

# ---------- Streamlit UI ----------

st.set_page_config(page_title="TV Show Analyzer", layout="wide")
st.title("📺 Indian TV Show Analyzer & Recommender")

# Objective Section
st.markdown("""
### Objective
This project analyzes Indian TV shows using **AI techniques** like Natural Language Processing (NLP), sentiment analysis, and machine learning. It includes a **recommendation engine**, **popularity prediction**, **data visualizations**, and **chatbot-style interactions**.
""")

df = load_data()
model, train_acc, test_acc, report_df = train_model(df)
recommend = build_recommender(df)

menu = st.sidebar.radio("Choose Section", ["📊 Popularity Analysis", "🎯 Recommendation", "🤖 Chatbot", "📈 Model Accuracy", "📷 Visualizations & NLP"])

if menu == "📊 Popularity Analysis":
    st.subheader("Top Rated Indian TV Shows")
    top_rated = df.sort_values(by="Rating", ascending=False).head(10)
    st.dataframe(top_rated[['Name', 'Rating', 'Genre', 'Language', 'Polarity']])

    show = st.selectbox("Choose a show to check sentiment:", df['Name'].tolist())
    selected = df[df['Name'] == show].iloc[0]
    sentiment = TextBlob(selected['Storyline']).sentiment
    st.markdown(f"**Polarity**: {sentiment.polarity:.2f}")
    st.markdown(f"**Subjectivity**: {sentiment.subjectivity:.2f}")
    st.markdown("**Sentiment Category**: " +
                ("Positive" if sentiment.polarity > 0.2 else "Neutral" if sentiment.polarity > -0.2 else "Negative"))

elif menu == "🎯 Recommendation":
    st.subheader("TV Show Recommendation Based on Title")
    selected_title = st.selectbox("Select a TV Show", df['Name'].tolist())

    if selected_title:
        recommendations = recommend(selected_title)
        st.markdown(f"**Because you liked:** `{selected_title}`")
        st.write("You may also enjoy:")
        for r in recommendations:
            st.markdown(f"• {r}")

elif menu == "🤖 Chatbot":
    st.subheader("TV Show Chatbot (Describe What You Want)")
    user_input = st.text_input("Enter your ideal show description:")

    if user_input.strip():
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Storyline'])
        user_vec = tfidf.transform([user_input])
        similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
        idx = similarity.argsort()[-1]
        recommendation = df.iloc[idx]

        st.write("### 🎯 Recommended Show")
        st.markdown(f"**Name**: {recommendation['Name']}")
        st.markdown(f"**Genre**: {recommendation['Genre']}")
        st.markdown(f"**Rating**: {recommendation['Rating']}")
        st.markdown(f"**Storyline**: {recommendation['Storyline']}")

        sentiment = TextBlob(recommendation['Storyline']).sentiment
        st.markdown(f"**Sentiment**: {'Positive' if sentiment.polarity > 0.2 else 'Neutral' if sentiment.polarity > -0.2 else 'Negative'}")

elif menu == "📈 Model Accuracy":
    st.subheader("Random Forest Classifier Accuracy")
    st.markdown(f"**Training Accuracy:** `{train_acc*100:.2f}%`")
    st.markdown(f"**Testing Accuracy:** `{test_acc*100:.2f}%`")
    st.markdown("### Classification Report")
    st.dataframe(report_df)

elif menu == "📷 Visualizations & NLP":
    st.subheader("Data Visualizations")

    st.markdown("####  Language Distribution")
    lang_counts = df['Language'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    ax1.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%')
    st.pyplot(fig1)

    st.markdown("####  Rating Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Rating'], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.markdown("####  Word Cloud of Storylines")
    text = " ".join(df['Storyline'].tolist())
    wordcloud = WordCloud(background_color='white', max_words=200).generate(text)
    fig3, ax3 = plt.subplots()
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis("off")
    st.pyplot(fig3)

    st.markdown("####  Overfitting Check")
    st.markdown(f"Training Accuracy: `{train_acc*100:.2f}%`")
    st.markdown(f"Testing Accuracy: `{test_acc*100:.2f}%`")
    st.markdown("If training accuracy is too high compared to testing, overfitting may be present.")
