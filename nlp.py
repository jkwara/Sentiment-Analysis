import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import pickle
from pickle import load
from pickle import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import joblib
import string
import nltk



data = pd.read_csv("financial_sentiment_data.csv")

st.title('Model Deployment: Sentiment Analysis')

# #image 
# from PIL import Image
# image = Image.open(r"C:\Users\Hp/image1.PNG")
# st.image(image, caption="Your Image Caption", use_column_width=True)


def input_features():
    Input = st.text_area('Type your input sentence here:', '')  # Use st.text_area for user input
    return Input
    
df = input_features()

#data cleaning
import re
import string

def clean_sentence(sentence):
    if sentence is None:
        return ""  # Return an empty string if input is None
    sentence = sentence.lower()
    sentence = re.sub('\[.*?\]', '', sentence)
    sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
    sentence = re.sub('\w*\d\w*', '', sentence)
    sentence = re.sub("[0-9" "]+", " ", sentence)
    sentence = re.sub('[‘’“”…]', '', sentence)
    sentence = re.sub('@\w+', '', sentence)
    return sentence



df = clean_sentence(df)

#removing stopwords

nltk.download("stopwords")
from nltk.corpus import stopwords
stop = stopwords.words("english")

df = " ".join(x for x in df.split() if x.lower() not in stop)


# CountVectorizer
loaded_vectorizer = joblib.load("cv.sav")
df = loaded_vectorizer.transform([df]).toarray()

#model load

loaded_model = load(open('model.sav', 'rb'))


prediction = loaded_model.predict(df)

# output
submit = st.button("Predict")
if submit:
    st.subheader('Predicted Result')
    st.write(prediction)
