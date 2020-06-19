

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
# Use English stemmer.
word_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

from gensim.models.keyedvectors import KeyedVectors

fake_news_df = pd.read_csv('https://raw.githubusercontent.com/harishSA05/datasets/master/Fake.csv')
real_news_df = pd.read_csv('https://raw.githubusercontent.com/harishSA05/datasets/master/True.csv')

"""**Remove missing text rows from the data whose length is less than 3 characters (cells with only white spaces for ex.) and also rows with NAN**"""

real_news_df['text'] = real_news_df['title'] + " " + real_news_df['text'] + " " + real_news_df['subject']
fake_news_df['text'] = fake_news_df['title'] + " " + fake_news_df['text'] + " " + real_news_df['subject']

print(real_news_df.shape)
print(fake_news_df.shape)
real_news_df = real_news_df[real_news_df['text'].str.len() >= 3]
fake_news_df = fake_news_df[fake_news_df['text'].str.len() >=3]
real_news_df['real_fact'] = 1
fake_news_df['real_fact'] = 0
print(real_news_df.shape)
print(fake_news_df.shape)

print(real_news_df.head(3))

print(fake_news_df.head(3))


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def get_cleaned_data(input_data, mode='df'):
    stop = stopwords.words('english')
    
    input_df = ''
    
    if mode != 'df':
        input_df = pd.DataFrame([input_data], columns=['text'])
    else:
        input_df = input_data
        
    #lowercase the text
    input_df['text'] = input_df['text'].str.lower()
    
    input_df['text'] = input_df['text'].apply(lambda elem: decontracted(elem))
    
    #remove special characters
    input_df['text'] = input_df['text'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    
    # remove numbers
    input_df['text'] = input_df['text'].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    #remove stopwords
    input_df['text'] = input_df['text'].apply(lambda x: ' '.join([word.strip() for word in x.split() if word not in (stop)]))
    
    #stemming, changes the word to root form
#     input_df['text'] = input_df['text'].apply(lambda words: [word_stemmer.stem(word) for word in words])
    
    #lemmatization, same as stemmer, but language corpus is used to fetch the root form, so resulting words make sense
#     more description @ https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    input_df['text'] = input_df['text'].apply(lambda words: (wordnet_lemmatizer.lemmatize(words)))
#     print(input_df.head(3))
    
    return input_df


fake_news_df = get_cleaned_data(fake_news_df)

real_news_df = get_cleaned_data(real_news_df)

news_data_df = pd.concat([real_news_df, fake_news_df], ignore_index = True)
print(news_data_df.shape)

news_data_df.head(2)

MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.3

x_train,x_test,y_train,y_test = train_test_split(news_data_df.text,news_data_df.real_fact,random_state = 42, test_size=VALIDATION_SPLIT, shuffle=True)

"""Vectorize the text samples into a 2D integer tensor"""

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

# Updates internal vocabulary based on a list of texts. 
# This method creates the vocabulary index based on word frequency. So if you give it something like, "The cat sat on the mat." 
# It will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value. 
# So lower integer means more frequent word (often the first few are stop words because they appear a lot).
tokenizer.fit_on_texts(x_train)

# Transforms each text in texts to a sequence of integers. 
# So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary.
# sequences = tokenizer.texts_to_sequences(news_data_df.text)
tokenized_train = tokenizer.texts_to_sequences(x_train)
X_train = pad_sequences(tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print('Found {} unique tokens. and {} lines '.format(len(word_index), len(X_train)))

tokenized_test = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(tokenized_test, maxlen=MAX_SEQUENCE_LENGTH)

"""Fetch pre-trained embedding index from GoogleNews-vectors-negative300:
* GoogleNews vectors are in most to least frequent order, so the first N are usually the N-sized subset. 
* So use limit=500000 to get the most-frequent 500,000 words' vectors – saving 5/6ths of the memory/load-time.
"""

def get_embeddings(path):
  # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300', binary=True, limit=500000)
  wv_from_bin = KeyedVectors.load_word2vec_format(path, binary=True, limit=500000) 
  #extracting word vectors from google news vector
  embeddings_index = {}
  for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
      coefs = np.asarray(vector, dtype='float32')
      embeddings_index[word] = coefs
  return embeddings_index

embeddings_index = {}
embeddings_index = get_embeddings("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz")
print('Found %s word vectors.' % len(embeddings_index))

"""Preparing embedding matrix."""

# # prepare embedding matrix format - 1
# num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
# embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i >= MAX_NUM_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

"""Free up the memory"""

del embeddings_index

"""Verify if the dataframe has real vs fact values, real news has real_fact = 1 and fake news has real_fact = 0"""

news_data_df[news_data_df['real_fact'] == 0]

news_data_df[news_data_df['real_fact'] == 1]

"""**Prepare the CNN model with GlobalMaxPooling for classification**"""

def cnn_net1():
    model = Sequential()

    #Non-trainable embeddidng layer
    model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=128, kernel_size=4, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(units = 250 , activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

"""**Prepare the CNN model with Deep network for classification**"""



unseen_data_fake2 = """
Americans to fund killing babies in abortion that she has been caught trying to add taxpayer financing of abortions to the bill to combat the Coronavirus and provide economic stimulus to the nation as it deals with the COVD-19 outbreak.
Nancy Pelosi has a long history of promoting abortion and her first act after becoming Speaker in 2019 was pushing legislation to use tax money for abortions. So it’s no surprise she is trying to exploit the Coronavirus pandemic to push abortion funding again.
As The Daily Caller reports: House Speaker Nancy Pelosi sought to include a potential way to guarantee federal funding for abortion into the coronavirus economic stimulus plan, according to multiple senior White House officials.
Speaking to the Daily Caller, those officials alleged that while negotiating the stimulus with U.S. Treasury Secretary Steve Mnuchin, Pelosi tried to lobby for “several” provisions that stalled bipartisan commitment to the effort. One was a mandate for up to $1 billion to reimburse laboratory claims, which White House officials say would set a precedent of health spending without protections outlined in the Hyde Amendment.
LifeNews depends on the support of readers like you to combat the pro-abortion media. Please donate now.
“A New mandatory funding stream that does not have Hyde protections would be unprecedented,” one White House official explained. “Under the guise of protecting people, Speaker Pelosi is working to make sure taxpayer dollars are spent covering abortion—which is not only backwards, but goes against historical norms.”
A second White House official referred to the provision as a “slush fund” and yet another questioned “what the Hyde Amendment and abortion have to do with protecting Americans from coronavirus?”
Americans should insist to their members of Congress that we need a clean bill that provides aggressive action to help patients and spur the economy. Killing babies with our tax dollars is not the answer to the coronavirus and the situation should not be exploited for political gain.
"""
unseen_data_fake = """The death of Kim Jong-il was reported by North Korean state television news on 19 December 2011. The presenter Ri Chun-hee announced that he had died on 17 December at 8:32 am of a massive heart attack while travelling by train to an area outside Pyongyang"""
unseen_data_real = """
India Coronavirus (Covid-19) Cases: While the number of novel Coronavirus cases in the country has been rising steadily, between eight and nine thousand a day these days, the growth rate has been coming down, nationally as well as in most of the states with major caseloads.
At the start of May, the compounded daily growth rate of cases in India was around 6.2 per cent. It rose to about 7 per cent before starting a decline that has continued since then. On Tuesday, the national growth rate was 4.67 per cent.
One of the main reasons for the decline in growth at the national level is the fact that Maharashtra, which accounts for more than a third of all cases in India, has been slowing down for more than two weeks now. And that has happened at a rate much faster than that what is observed at the national level. Till the middle of May, Maharashtra’s growth was about one per cent higher than the country as a whole. As that started to decline, it dragged down the national growth as well. On June 2, Maharashtra’s growth rate was 4.05 per cent, well below the national rate."""

def get_pred_output(text_to_check):
    sequences = tokenizer.texts_to_sequences([text_to_check])
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predicted_val = model.predict_classes(data)
#     predicted_val = model.predict(data)    
#     if predicted_val.max() > 0.7:
#         output = 1
#     else:
#         output = 0
    return predicted_val

# train a 1D convnet with global maxpooling
model = cnn_net1()

batch_size = 256
epochs = 1

model.summary()

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

history = model.fit(X_train, y_train, batch_size = batch_size , validation_data = (X_test,y_test) , epochs = epochs)

accr_train = model.evaluate(X_train,y_train)
print('Accuracy Train: {}'.format(accr_train[1]*100))
accr_test = model.evaluate(X_test,y_test)
print('Accuracy Test: {}'.format(accr_test[1]*100))

pred = model.predict_classes(X_test)

text_to_check = unseen_data_real
pred = get_pred_output(text_to_check)
print('Unseen real data prediction {} '.format(pred[0]))

text_to_check = unseen_data_fake
pred = get_pred_output(text_to_check)
print('Unseen fake data prediction {} '.format(pred[0]))

text_to_check = news_data_df.text[1000]
pred = get_pred_output(text_to_check)
print('Seen real data prediction {} '.format(pred[0]))

text_to_check = news_data_df.text[31000]
pred = get_pred_output(text_to_check)
print('Seen fake data prediction {} '.format(pred[0]))



from flask import Flask,request,jsonify
app = Flask(__name__)

from flask_cors import CORS, cross_origin
import spacy
from collections import Counter
from string import punctuation
import subprocess
import os
import sys
base_dir = os.path.abspath(os.path.dirname('__file__'))
print(base_dir)
import requests
from datetime import datetime,timedelta
import json
from pydub import AudioSegment
import pydub
from PIL import Image
import pytesseract
import re 
from textblob import TextBlob
import speech_recognition as sr

CORS(app)     
app.config["IMAGE_UPLOADS"] = os.path.join(base_dir+"/uploads/")
app.config["AUDIO_UPLOADS"] = os.path.join(base_dir+"/uploads/")


print(app.config["IMAGE_UPLOADS"])
def getrelevantNews(text):
        #transformed test reviews
        print(get_pred_output(text))
        return 1


@app.route('/getnounsfromtext_spacy',methods=['POST'])
def getspacy():
        somejsonfile = request.get_json()
        text = somejsonfile['data']
        text = str(text)
        relevantNews = getrelevantNews(text)
        return jsonify({"success":True,"result":relevantNews})


@app.route('/gettextfromimage',methods=['POST'])
def gettextfromimage():
        file = request.files['image']
        file.save(os.path.join(app.config["IMAGE_UPLOADS"], file.filename))
        if file.filename.split('.')[1] == 'webp':
             image = Image.open(os.path.join('./uploads/',file.filename)).convert("RGB")
        else:
            image = Image.open(os.path.join('./uploads/',file.filename))
        textfromimage = pytesseract.image_to_string(image, lang='eng')
        relevantNews = getrelevantNews(textfromimage)
        return jsonify({"success":True,"result":relevantNews})


@app.route('/gettextfromvoice',methods=['POST'])
def gettextfromvoice():
        file = request.files['file']
        r = sr.Recognizer()
        ext = file.filename.split('.')[1]
        if ext == 'mp3':
            file.save(os.path.join(app.config["AUDIO_UPLOADS"], "oldfile.mp3"))
            oldfile = os.path.join(app.config["AUDIO_UPLOADS"],"oldfile.mp3")
            newfile= os.path.join(app.config["AUDIO_UPLOADS"],"newfile.wav")
            subprocess.call(['ffmpeg','-y','-i',oldfile,newfile])
            harvard = sr.AudioFile(newfile)
        elif ext == 'mp4' or ext == 'avi':
            file.save(os.path.join(app.config["AUDIO_UPLOADS"],"oldfile."+ext))
            oldfile = os.path.join(app.config["AUDIO_UPLOADS"],"oldfile."+ext)
            newfile= os.path.join(app.config["AUDIO_UPLOADS"],"newfile.wav")
            clip = mp.VideoFileClip(oldfile)
            clip.audio.write_audiofile(newfile)
            harvard = sr.AudioFile(newfile)
        elif ext == 'wav':
            file.save(os.path.join(app.config["AUDIO_UPLOADS"], "oldfile.wav"))
            harvard = sr.AudioFile(os.path.join('./uploads/',"oldfile.wav"))
        else:
            return jsonify({"status":"failed","message":"invalid file format"})
        with harvard as source:
            audio = r.record(source)
        textfromaudio = r.recognize_sphinx(audio)
        relevantNews = getrelevantNews(textfromaudio)
        return jsonify({"success":True,"result":relevantNews})

@app.route('/login',methods=['POST'])
def login():
    return jsonify({"status":"success","username":user.username}) 
        

@app.route('/signup',methods=['POST'])
def signup():
    return jsonify({
            "status":"success",
            "message": "User  created!",
    })

threaded=False




