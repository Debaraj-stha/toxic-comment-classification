import emoji
import re
from constants import categories, bad_words
from functools import cache
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
import pickle
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

def convert_emoji(text):
    return emoji.demojize(text, delimiters=(" ", " "))


def has_emoji(text):
    if not isinstance(text, str):
        return 0
    return int(any(char in emoji.EMOJI_DATA for char in text))

def normalize_leetspeak(text):
    patterns = {
        r'[4@]': 'a',
        r'[8]': 'b',
        r'[\(\{\[]': 'c',
        r'[3]': 'e',
        r'[6|9]': 'g',
        r'[1!|]': 'i',
        r'[0]': 'o',
        r'[$5]': 's',
        r'[7+]': 't',
        r'[2]': 'z'
    }

    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)

    return text




def emoji_sentiment_multi(text):
  scores = {
       "negative": 0,
        "sarcastic": 0,
        "threat": 0,
        "positive": 0,
        "sad": 0
    }
  if not isinstance(text,str):
       return scores

  text = text.lower()



  for label, keywords in categories.items():
        for word in keywords:
            if word in text:
                scores[label] += 1

    # convert to binary (one-hot style)
  return {label: int(score > 0) for label, score in scores.items()}





def bad_word_features(text):
    text = text.lower()
    features = {}

    for category, words in bad_words.items():
        count = 0
        for word in words:
            count += len(re.findall(rf"\b{re.escape(word)}\b", text))
        features[category] = count


    total = sum(features.values())
    features["total_bad_words"] = total
    features["bad_word_ratio"] = total/len(text) if len(text)>0 else 0

    features["has_bad_words"] = 1 if total > 0 else 0

    return features

def repetition_features(text):
    repeats = re.findall(r'(.)\1{2,}', text)

    return {
        "repetition_count": len(repeats),
        "has_repetition": int(len(repeats) > 0)
    }



def translate(text):
    if len(str(text)) > 500:
        return text
    return GoogleTranslator(source="auto", target="en").translate(text)


def preprocess(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # clean spam
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    #  URLs & emails
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)

    # numbers
    text = re.sub(r'\d+', ' NUM ', text)

    #  leetspeak normalization
    text = normalize_leetspeak(text)

    # emoji conversion
    text = convert_emoji(text)

 
    #  clean punctuation
    text = re.sub(r"[^\w\s!?\.]", " ", text)

    return text



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@cache
def load_my_model():
    try:
        model_path = os.path.join(BASE_DIR, "model/toxicity_model.keras")
        config_path = os.path.join(BASE_DIR, "model/config.pkl")
        tokenizer_path = os.path.join(BASE_DIR, "model/tokenizer.pkl")

        model = load_model(model_path)

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        return model, tokenizer, config

    except FileNotFoundError:
        raise RuntimeError(" Model files not found. Check paths.")
    

labels=["","",""]

def make_predict(comments):
    if not len(comments)>0:
        return []
    # df=pd.DataFrame(comments,columns=["comment_text"])

    # rep_features = df["comment_text"].apply(repetition_features)
    # df["comment_text"]=df["comment_text"].apply(translate)
    # df["comment_text"] = df["comment_text"].apply(preprocess)
    # df["char_len"] = df["comment_text"].str.len()
    # df["word_len"] = df["comment_text"].str.split().str.len()

    # bad_features = df["comment_text"].apply(bad_word_features)
    # bad_df = pd.DataFrame(bad_features.tolist())
    # rep_df = pd.DataFrame(rep_features.tolist())


    # df["has_emoji"] = df["comment_text"].apply(has_emoji)
    # emoji_sentiment=df["comment_text"].apply(emoji_sentiment_multi)
    # emoji_sentiment_df=pd.DataFrame(emoji_sentiment.to_list())

    # final_df = pd.concat([
    #     df.reset_index(drop=True),
    #     bad_df.reset_index(drop=True),
    #     rep_df.reset_index(drop=True),
    #     emoji_sentiment_df.reset_index(drop=True)
    # ], axis=1)
    model, tokenizer, config = load_my_model()

    max_len = config.get("max_len")
    labels = config.get("labels")

    
    X_text = tokenizer.texts_to_sequences(comments)
    X_text = pad_sequences(X_text, maxlen=max_len, padding="post")

    y_pred = model.predict(X_text)

    thresholds = [0.5, 0.4, 0.4]

    # Convert probabilities → binary
    y_pred_bin = (y_pred > thresholds).astype(int)

    # Convert binary → labels
    labeled_predictions = []

    for i in range(len(y_pred_bin)):
        current_labels = []

        for j in range(len(labels)):
            if y_pred_bin[i][j] == 1:
                current_labels.append(labels[j])

        if not current_labels:
            current_labels.append("clean")

        labeled_predictions.append(current_labels)

    return labeled_predictions
                
  
