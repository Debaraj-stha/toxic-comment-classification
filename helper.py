import emoji
import re
import spacy
from constants import categories, bad_words
from functools import cache
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator

@cache
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )

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

nlp = load_spacy()

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

    #  spaCy processing
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_space
    ]

    return " ".join(tokens)

@cache
def load_model():
    pass

labels=["","",""]

def make_predict(comments):
    df=pd.DataFrame(comments,columns=["comment_text"])

    rep_features = df["comment_text"].apply(repetition_features)
    df["comment_text"]=df["comment_text"].apply(translate)
    df["comment_text"] = df["comment_text"].apply(preprocess)
    df["char_len"] = df["comment_text"].str.len()
    df["word_len"] = df["comment_text"].str.split().str.len()

    bad_features = df["comment_text"].apply(bad_word_features)
    bad_df = pd.DataFrame(bad_features.tolist())
    rep_df = pd.DataFrame(rep_features.tolist())


    df["has_emoji"] = df["comment_text"].apply(has_emoji)
    emoji_sentiment=df["comment_text"].apply(emoji_sentiment_multi)
    emoji_sentiment_df=pd.DataFrame(emoji_sentiment.to_list())

    final_df = pd.concat([
        df.reset_index(drop=True),
        bad_df.reset_index(drop=True),
        rep_df.reset_index(drop=True),
        emoji_sentiment_df.reset_index(drop=True)
    ], axis=1)

    vectorizer,model=load_model()
    comment_text=final_df.loc[:,["comment_text"]]
    x=final_df.drop(columns=[labels]+["comment_text"])
    # y=final_test_df[labels]
    vector=vectorizer.transform(comment_text)
    vectorized_df=np.hstack([vector,x])
    y_pred=model.predict(vectorized_df)
    return y_pred

  
