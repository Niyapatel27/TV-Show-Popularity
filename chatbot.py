import gradio as gr
import pandas as pd
from data_preprocessing import preprocess_tv_data
from recommendation import build_recommender

df = preprocess_tv_data("tvshows.csv")
recommender = build_recommender(df)

def chat(input_text):
    input_text = input_text.lower()
    
    if "drama" in input_text:
        filtered = df[df['Genre'].apply(lambda x: 'Drama' in x)]
    elif "romance" in input_text:
        filtered = df[df['Genre'].apply(lambda x: 'Romance' in x)]
    elif "crime" in input_text:
        filtered = df[df['Genre'].apply(lambda x: 'Crime' in x)]
    elif "recommend" in input_text:
        title = input_text.replace("recommend", "").strip().title()
        return f"You may also like: {', '.join(recommender(title))}"
    else:
        return "Tell me what genre you like (e.g., Drama, Romance, Crime) or ask: recommend [TV Show Name]"

    top = filtered.sort_values(by="Rating", ascending=False).head(3)
    return f"Top {filtered.iloc[0]['Genre'][0]} shows: {', '.join(top['Name'].tolist())}"

iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="TV Show Recommender Chatbot")
iface.launch()