from flask import Flask,request,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS, cross_origin
from textblob import TextBlob
import nltk
import speech_recognition as sr
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
from string import punctuation
from GoogleNews import GoogleNews
googlenews = GoogleNews()
from newsapi import NewsApiClient
import subprocess
import os
import pandas as pd
nlp = spacy.load("en_core_web_sm")
import requests
app = Flask(__name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
import hashlib
import moviepy.editor as mp
# jwt = JWT()
app.config['SECRET_KEY']="raand"
from datetime import datetime,timedelta
from pydub import AudioSegment
import pydub
from PIL import Image
import pytesseract
import json
import re 
import gensim

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

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://postgres:password@localhost:5432/fakenews'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
newsapi = NewsApiClient(api_key='2dfd81e0287b440abb85d7bd4da4a9d6')




#create a model
class Users(db.Model):
    __tablename__ = 'users'
    id=db.Column(db.Integer,primary_key=True)
    username = db.Column(db.String)
    phoneno = db.Column(db.String)
    email=db.Column(db.String)
    password=db.Column(db.String)
    status=db.Column(db.Integer)

    def __init__(self,email,password,phoneno,username,status):
        self.email = email
        self.password = password
        self.status = status
        self.username = username
        self.phoneno = phoneno



CORS(app)       
app.config["IMAGE_UPLOADS"] = os.path.join(base_dir+"\\uploads\\")
app.config["AUDIO_UPLOADS"] = os.path.join(base_dir+"\\uploads\\")




def nltk(text):
        tokens = nltk.word_tokenize(text)
        nouns = [] 
        for token in tokens:
            for word,pos in nltk.pos_tag(nltk.word_tokenize(str(token))):
                if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                    nouns.append(word)
        news = ' '.join(nouns)
        return news


def textblob(text):
        paragraph = TextBlob(text)
        nouns = paragraph.noun_phrases
        news = ' '.join(nouns)
        return news


# num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
#              6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
#             11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
#             15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
#             19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
#             50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
#             90: 'Ninety',100: 'One Hundered',200: 'Two Hundered',\
#             300: 'Three Hundered',400: 'Four Hundered',500: 'Five Hundered',\
#             600: 'Six Hundered',700: 'Seven Hundered',800: 'Eight Hundered',\
#             900: 'Nine Hundered', 0: 'Zero'}

# def n2w(n)
# :
#         try:
#             print (num2words[n])
#         except KeyError:
#             try:
#                 print (num2words[n-n%10] + num2words[n%10].lower())
#             except KeyError:
#                 print ('Number out of range')


def getrelevantNews(nouns,text):
        # googlenews.clear()
        # googlenews.search(nouns)
        # relevantNews = googlenews.result()
        # print(relevantNews)
        news = newsapi.get_everything(q=nouns,language='en')
        # news = requests.get("https://gnews.io/api/v3/search?q=nouns&token=911faa3568a07ec3606a958da028ee59")
        # if(text in news):
            # print(news["articles"])
        relevantNews = json.dumps(news["articles"])
        newsArticles = news["articles"]
        print(newsArticles)
        res = []
        userNews = nlp(spacy(text))
        for i in range(0,len(newsArticles)):
            articleNews = nlp(spacy(newsArticles[i]["description"]))
            # print(text)
            # print(newsArticles[i]["description"])
            # print(userNews.similarity(articleNews))
            # print("************")
            if userNews.similarity(articleNews) > 0.50:
                    res.append(newsArticles[i])
        print(res)
        return json.dumps(res)


def spacy(text):
        res = re.findall(r'\d+', text)
        for num in res:
            number = int(num)
            numintxt = int_to_en(number)
            text = text.replace(num,numintxt)
        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN']
        doc = nlp(text.lower())
        for token in doc:
            if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                result.append(token.text)
        newsSubject = ' '.join(result)
        return newsSubject

@app.route('/getnounsfromtext_textblob',methods=['POST'])
def gettextblob():
        somejsonfile = request.get_json()
        subject = textblob(somejsonfile['data'])
        relevantNews = getrelevantNews(subject,somejsonfile['data'])
        return jsonify({"data":relevantNews,"subject":subject})


@app.route('/getnounsfromtext_spacy',methods=['POST'])
def getspacy():
        somejsonfile = request.get_json()
        text = somejsonfile['data']
        subject = spacy(somejsonfile['data'])
        relevantNews = getrelevantNews(subject,text)
        return jsonify({"data":relevantNews,"subject":subject})



@app.route('/getnounsfromtext_nltk',methods=['POST'])
def getnltk():
        somejsonfile = request.get_json()
        subject = nltk(somejsonfile['data'])
        relevantNews = getrelevantNews(subject,somejsonfile['data'])
        return jsonify({"data":relevantNews,"subject":subject})


@app.route('/gettextfromimage',methods=['POST'])
def gettextfromimage():
        file = request.files['image']
        file.save(os.path.join(app.config["IMAGE_UPLOADS"], file.filename))
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if file.filename.split('.')[1] == 'webp':
             image = Image.open(os.path.join('./uploads/',file.filename)).convert("RGB")
        else:
            image = Image.open(os.path.join('./uploads/',file.filename))
        textfromimage = pytesseract.image_to_string(image, lang='eng')
        subject = spacy(textfromimage)
        relevantNews = getrelevantNews(subject,textfromimage)
        return jsonify({"data":relevantNews,"subject":subject})



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
        subject = spacy(textfromaudio)
        relevantNews = getrelevantNews(subject,textfromaudio)
        return jsonify({"data":relevantNews,"subject":subject})



@app.route('/login',methods=['POST'])
def login():
    data = request.get_json()
    print(data)
    email = data['email']
    password = data['password']
    print(email,password)
    try:
        user = Users.query.filter_by(email=email).first()
        print(user and user.password == hashlib.sha256(password.encode("utf-8")).hexdigest())
        if user and user.password == hashlib.sha256(data["password"].encode("utf-8")).hexdigest():
            return jsonify({"status":"success","username":user.username}) 
        else:
           return jsonify({"status":"failed","message":"Incorrect password!"}) 
    except:
        return jsonify({"status":"failed","message":"login Failed!"})


@app.route('/signup',methods=['POST'])
def signup():
    data = request.get_json()
    print(data)
    email = data['email']
    password = data['password']
    username = data['name']
    phoneno = data['mobile']
    print(email,password,username,phoneno)
    data = db.session.query(Users).filter(Users.email == email).first()
    if data and data.email == email: 
        return jsonify({
                "status":"failed",
              "message": "email exists!"
         })
    user = Users(email, hashlib.sha256(password.encode("utf-8")).hexdigest(),phoneno,username,1)
    db.session.add(user)
    db.session.commit()
    return jsonify({
            "status":"success",
            "message": "User  created!",
    })




# def similarity():
#     text_sentences = []
#     text = "Mars is approximately half the diameter of Earth."
#     for line in text:
#         text_sentences.append(line)
#     gen_docs = [[w.lower() for w in word_tokenize(text)] 
#             for text in text_sentences]
#     dictionary = gensim.corpora.Dictionary(gen_docs)
#     corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#     print(corpus)

# similarity()

