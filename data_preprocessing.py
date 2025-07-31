import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from textblob import TextBlob

def preprocess_tv_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert columns from string to list
    df['Genre'] = df['Genre'].apply(literal_eval)
    df['Creators'] = df['Creators'].apply(literal_eval)
    df['Stars'] = df['Stars'].apply(literal_eval)
    
    # Sentiment features from storyline
    df['Polarity'] = df['Storyline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Subjectivity'] = df['Storyline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # Genre One-hot encoding
    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(mlb.fit_transform(df['Genre']), columns=mlb.classes_)
    df = df.join(genre_df)
    
    return df