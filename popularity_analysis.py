from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def train_popularity_model(df):
    df['Popularity'] = df['Rating'].apply(lambda x: 'High' if x >= 8.5 else 'Low')
    
    X = df[['No. of Ratings', 'No. of Episodes', 'Seasons', 'Polarity', 'Subjectivity']]
    y = df['Popularity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    return clf,acc

