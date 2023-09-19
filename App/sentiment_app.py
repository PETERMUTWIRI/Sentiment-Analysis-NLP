import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax

# Load your model and tokenizer
model_path = "Enyonam/distilbert-base-uncased-Distilbert-Model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess text (username and link placeholders)
#In summary, this preprocessing function helps ensure that usernames and links in the input text do not interfere with the sentiment analysis performed by the model. It replaces them with placeholder tokens to maintain the integrity of the text's structure while anonymizing or standardizing specific elements.

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment_analysis(text):
    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    # Format output dict of scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l: float(s) for (l, s) in zip(labels, scores_)}

    return scores

# Streamlit app layout with two columns
st.title("Sentiment Analysis App")
st.write(" Sentiment analysis, also known as opinion mining, is the process of determining the emotional tone or sentiment expressed in text data, whether it's positive,negative, or neutral")
st.image("Assets/sent_emoji.jpg", caption="Sentiments examples", use_column_width=True)

# Input text area for user to enter a tweet in the left column
input_text = st.text_area("Write your tweet here...")

# Output area for displaying sentiment in the right column
if st.button("Analyze Sentiment"):
    if input_text:
        # Perform sentiment analysis using the loaded model
        scores = sentiment_analysis(input_text)

        # Display sentiment scores in the right column
        st.text("Sentiment Scores:")
        for label, score in scores.items():
            st.text(f"{label}: {score:.2f}")

        # Determine the overall sentiment label
        sentiment_label = max(scores, key=scores.get)

        # Map sentiment labels to human-readable forms
        sentiment_mapping = {
            "Negative": "Negative",
            "Neutral": "Neutral",
            "Positive": "Positive"
        }
        sentiment_readable = sentiment_mapping.get(sentiment_label, "Unknown")

        # Display the sentiment label in the right column
        st.text(f"Sentiment: {sentiment_readable}")

# Button to Clear the input text
if st.button("Clear Input"):
    input_text = ""
