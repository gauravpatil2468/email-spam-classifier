import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from streamlit import header

nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()


tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
model = pickle.load(open('./model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


st.title('Email/SMS Spam Classifier')
input_msg = st.text_area('Enter the message')
if st.button('Predict'):
    #preprocess
    transformed_msg = transform_text(input_msg)

    #vectorize
    vectorized_msg = tfidf.transform([transformed_msg])
    #predict
    result = model.predict(vectorized_msg)[0]
    #display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')




