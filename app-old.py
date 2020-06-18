from flask import Flask,request,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS, cross_origin
from textblob import TextBlob
import nltk
import speech_recognition as sr
import spacy
from collections import Counter
from string import punctuation
from GoogleNews import GoogleNews
googlenews = GoogleNews()
import subprocess
import os
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

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://postgres:password@localhost:5432/fakenews'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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
        news = '+'.join(nouns)
        print("-----------")
        print(news)
        print("-----------")
        return news


def textblob(text):
        paragraph = TextBlob(text)
        nouns = paragraph.noun_phrases
        news = '+'.join(nouns)
        print("++++++++++")
        print(news)
        print("++++++++++++++")
        return news


def getrelevantNews(nouns):
        # googlenews.clear()
        # googlenews.search(nouns)
        # relevantNews = googlenews.result()
        # print(relevantNews)
        relevantNews = requests.get("https://gnews.io/api/v3/search?q="+nouns+"&token=911faa3568a07ec3606a958da028ee59").json()
        relevantNews = json.dumps(news)
        # print(relevantNews.articles)
        return relevantNews


def spacy(text):
        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN']
        doc = nlp(text.lower())
        for token in doc:
            if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                result.append(token.text)
        newsSubject = '+'.join(result)
        print("************")
        print(newsSubject)
        print("************")
        return newsSubject

@app.route('/getnounsfromtext_textblob',methods=['POST'])
def gettextblob():
        somejsonfile = request.get_json()
        subject = textblob(somejsonfile['data'])
        relevantNews = getrelevantNews(subject)
        return jsonify({"data":relevantNews,"subject":subject})


@app.route('/getnounsfromtext_spacy',methods=['POST'])
def getspacy():
        somejsonfile = request.get_json() 
        subject = spacy(somejsonfile['data'])
        # textblob(somejsonfile['data'])
        relevantNews = getrelevantNews(subject)
        print(relevantNews)
        return jsonify({"data":relevantNews,"subject":subject})


@app.route('/getnounsfromtext_nltk',methods=['POST'])
def getnltk():
        somejsonfile = request.get_json()
        subject = nltk(somejsonfile['data'])
        relevantNews = getrelevantNews(subject)
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
        relevantNews = getrelevantNews(subject)
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
            print(newfile,oldfile)
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
        relevantNews = getrelevantNews("coronavirus+cases")
        return jsonify({"data":relevantNews,"subject":subject})




@app.route('/login',methods=['POST'])
def login():
    data = request.get_json()
    print(data)
    email = data['email']
    password = data['password']
    print(email,password)
    print(hashlib.sha256(password.encode("utf-8")).hexdigest() == "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8")

    try:
        user = Users.query.filter_by(email=email).first()
        print(user)
        if user and user.password == hashlib.sha256(data["password"].encode("utf-8")).hexdigest():
            return jsonify({"status":"success","username":user.username}) 
        else:
           return jsonify({"status":"failed","message":"Incorrect password!"}) 
    except:
        print("exception")
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




