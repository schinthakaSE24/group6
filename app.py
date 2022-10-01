from ast import Str
from asyncio.windows_events import NULL
from collections import UserList
from contextlib import nullcontext
from distutils.command.upload import upload
import email
from email.headerregistry import Address
from email.policy import default
from enum import unique
from fileinput import filename
from importlib.machinery import FileFinder
from importlib.metadata import files
import json
from logging import PlaceHolder
from queue import Empty
import secrets
import sqlite3
from tkinter.tix import Form
from tokenize import String
from turtle import width
from unittest import result
from flask import Flask, render_template, redirect, url_for, request,send_file,session
from flask_bootstrap import Bootstrap
from flask import flash
import unicodedata
import pandas as pd
import string
import os
from flask_wtf import FlaskForm
from sqlalchemy import false, true
from wtforms import StringField, PasswordField, BooleanField,FileField,SubmitField,RadioField
from wtforms.validators import InputRequired, Email, Length,DataRequired, ValidationError
from flask_wtf.file import FileField, FileAllowed
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from pyresparser import ResumeParser
from docx import Document
from PIL import Image
import re
import joblib
from werkzeug.utils import secure_filename
import spacy
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user,login_required, logout_user, current_user
import uuid
from extract_txt import read_files
from txt_processing import preprocess
from txt_to_features import txt_features, feats_reduce
from extract_entities import get_number, get_email, rm_email, rm_number, get_name, get_skills
from model import simil


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Load the spacy library for text cleaning
nlp = spacy.load('en_core_web_sm')
# Loading the saved model
rf_clf = joblib.load('rf_clf.pkl')


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15),unique=True)
    email = db.Column(db.String(50),unique=True)
    password = db.Column(db.String(80))
    image_file = db.Column(db.String(20), nullable=False, default="download.jpg")
    phonenumber = db.Column(db.String(50),unique=True)
    address = db.Column(db.String(50),unique=True)
    usertype = db.Column(db.String(50),unique=True)
    messages_sent = db.relationship(
        'Message',
        foreign_keys='Message.sender_id',
        backref='author',
        lazy='dynamic')
    messages_received = db.relationship(
        'Message',
        foreign_keys='Message.recipient_id',
        backref='recipient',
        lazy='dynamic')
    last_message_read_time = db.Column(db.DateTime)
   

    def new_messages(self):
        last_read_time = self.last_message_read_time or datetime(1900, 1, 1)
        return Message.query.filter_by(recipient=self).filter(
            Message.timestamp > last_read_time).count()

   
class Cv(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username= db.Column(db.String(150),unique=True)
    cvemail = db.Column(db.String(150),unique=True)
    cvname = db.Column(db.String(50),unique=True)
    phonenumber = db.Column(db.String(14),unique=True)
    skills = db.Column(db.String(),unique=True)
    degree = db.Column(db.String(),unique=True)
    experience =db.Column(db.String(),unique=True)
    
class Company(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    companyname = db.Column(db.String(50),unique=True)
    companyemail = db.Column(db.String(50),unique=True)
    password = db.Column(db.String(80))
   

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    recipient_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    def __repr__(self):
        return f'Message: {self.body}'  

    
def __repr__(self):
    return f"User('{self.username}', '{self.password}')"  


    
       
@app.before_first_request
def create_tables():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id)) 


class LoginForm(FlaskForm):
    username = StringField('Username:', validators=[InputRequired()])
    password = PasswordField('Password:', validators=[InputRequired()])
    remember = BooleanField('Remember me')

class LoginFormcom(FlaskForm):
    
    companyemail = StringField('Companyemail:', validators=[InputRequired()])
    companyname = StringField('Companyemail:', validators=[InputRequired()])
    password = PasswordField('Password:', validators=[InputRequired()])
   

class RegisterForm(FlaskForm):
    email = StringField('Email' ,validators=[InputRequired()])
    username = StringField('Username:', validators=[InputRequired()])
    usertype = RadioField('usertype', choices=[('user as applicant'),('user as recruiter')] ,validators=[InputRequired()])
    password = PasswordField('Password:')
   
class RegisterFormc(FlaskForm):   
    companyname = StringField("companyname:",validators=[InputRequired()])
    companyemail = StringField('companyemail:',validators=[InputRequired()])
    password = PasswordField('Password:')

@app.route('/')
def index():
   
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('profiles'))

    
        flash("Invalid username or password", 'danger')
        return redirect(url_for('login'))
      

    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():   
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(
            form.password.data, method='sha256')
        email = request.form.get("email")
        username = request.form.get("username")
        checked = request.form.get("usertype")
    
        user = User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first()
        if not user:
            new_user = User(username=form.username.data,email=form.email.data,usertype=checked,password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash("User Account created successfully")
            return redirect(url_for('login'))
      
        else:
            flash('The email or username already exists')
        
       
            

    return render_template('signup.html', form=form)



@app.route('/profilecom', methods=['GET', 'POST'])
def profilecom():
    cusers = Company.query.all()
    return render_template('cprofile.html',cusers=cusers)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)

UPLOAD_CV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files/cv/')
@app.route('/uploader', methods=['GET', 'POST'])
def dashboards():
     if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
       
      
    

   
        try:
            doc = Document()
            with open(f.filename, 'r') as file:
                doc.add_paragraph(file.read())
                doc.save("text.docx")
                data = ResumeParser('text.docx').get_extracted_data()

        except:
            data = ResumeParser(f.filename).get_extracted_data()

        cleaned_data = {x.replace('_', ' '): v
                        for x, v in data.items()}

        def clean_content(text):
            text = text.replace("uf0b7", "").replace(
                "'", "").replace("[", "").replace("]", "")
            return text

        df = pd.DataFrame()
        df["key"] = cleaned_data.keys()
        df["content"] = cleaned_data.values()
        df["content"] = df.content.apply(lambda x: clean_content(str(x)))
        final_data = df.set_index('key')['content'].to_dict()

        datas = []
        datas.append(final_data)

        
      #  return redirect(url_for('dashboard'))

      # Turn a Unicode string to plain ASCII, --> https://stackoverflow.com/a/518232/2809427
        def unicode_to_ascii(s):
            all_letters = string.ascii_letters + " .,;'-"
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
                and c in all_letters
            )

        # Remove Stop Words
        def remove_stopwords(text):
            doc = nlp(text)
            return " ".join([token.text for token in doc if not token.is_stop])

        def clean_text(text):
            #print(f'Text before Cleaning: {text}')
            # Text to lowercase
            text = text.lower()
            # Remove URL from text
            text = re.sub(r"http\S+", "", text)
            # Remove Numbers from text
            text = re.sub(r'\d+', '', text)
            # Convert the unicode string to plain ASCII
            text = unicode_to_ascii(text)
            # Remove Punctuations
            text = re.sub(r'[^\w\s]', '', text)
            #text = remove_punct(text)
            # Remove StopWords
            text = remove_stopwords(text)
            # Remove empty spaces
            text = text.strip()
            # \s+ to match all whitespaces
            # replace them using single space " "
            text = re.sub(r"\s+", " ", text)
            #print(f'Text after Cleaning: {text}')
            return text
        name = current_user.username
        cvmail = data.get("email")
        cvapplicant_name = data.get("name")
        phonenumber = data.get("mobile_number")
        skills = data.get("skills")
        degree = data.get("degree")
        experience = data.get("total_experience")
        skillss = ','.join(map(str,skills))
        
        ccuser = Cv.query.filter_by(cvemail=cvmail).first() or Cv.query.filter_by(username=name).first()
        if not ccuser:
            cvuser = Cv( username=name,cvemail=cvmail,cvname=cvapplicant_name,phonenumber=phonenumber,degree=degree, skills=skillss,experience=experience)
            db.session.add(cvuser)
            db.session.commit()
            flash('File uploaded Successfully')
        else:
            flash("File Already uploaded one,Try again later")
        
        data = str(data)
        cleaned = clean_text(data)
        prediction = rf_clf.predict([cleaned])
        result = prediction[0]

        return render_template('Myprofile.html', name=current_user.username, res_content=datas, pred=result)
   
    

       


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile/<username>')
def pro(username):
    username="nuwan"
   
    return render_template('search.html', username=username)

@app.route('/rec',methods=['GET', 'POST'])
def cprofile():
    return render_template('cprofile.html')

ROWS_PER_PAGE = 4
@app.route('/profiles')
def profiles():
    users = User.query.all()
 


   
    page = request.args.get('page', 1, type=int)

    users = User.query.paginate(page=page, per_page=ROWS_PER_PAGE) 
    
    return render_template('profile.html',users=users)


@app.route('/Myprofile',methods=['GET', 'POST'])
@login_required
def Myprofile():
    
    
    imagefile = url_for('static', filename='profilepic/' + str(current_user.image_file))
    return render_template('Myprofile.html', imagefile=imagefile)

class UpdateAccountForm(FlaskForm):
    username = StringField('Username',validators=[DataRequired()])
    email = StringField("email",validators=[DataRequired()])
    picture = FileField("update profile picture",validators=[FileAllowed(['jpg','png'])])
    address = StringField('address',validators=[DataRequired()])
    phonenumber = StringField('Phonenumbers',validators=[DataRequired()])
    submit = SubmitField('update')

def validate_username(self, username):
        if username.data != current_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('That username is taken. Please choose a different one.')

def validate_email(self, email):
    if email.data != current_user.email:
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')
def validate_phonenumber(self, phonenumber):
    if phonenumber.data != current_user.phonenumber:
        user = User.query.filter_by(phonenumber=phonenumber.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')
def validate_address(self, address):
    if address.data != current_user.address:
        user = User.query.filter_by(address=address.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')
def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static\\profilepic', picture_fn)
    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
   

    return picture_fn


@app.route('/profileupdate',methods=['GET', 'POST'])
@login_required
def profileupdate():
   
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.phonenumber = form.phonenumber.data
        current_user.address = form.address.data

        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('Myprofile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data =current_user.email
        form.phonenumber.data = current_user.phonenumber
        form.address.data = current_user.address

    image_file = url_for('static', filename='profilepic/' + str(current_user.image_file))
    return render_template('update.html',
                           image_file=image_file, form=form)
   
class SearchForm(FlaskForm):
	searched = StringField("Searched", validators=[DataRequired()])
	submit = SubmitField("send")
@app.route('/search', methods=["POST"])
@login_required
def search():
 
        return render_template("search.html")


@app.route('/profiles/<username>',methods=["GET", "POST"])
@login_required
def view(username):
    user= User.query.filter_by(username=username).first()
    cuser= Cv.query.filter_by(username=username).first()
   

  
    return render_template("pdf.html",user=user,cuser=cuser)
@app.route('/info/<username>')
def info(username):
    user= User.query.filter_by(username=username).first()
    cuser= Cv.query.filter_by(username=username).first()
    return render_template('info.html', user=user,cuser=cuser)
    
@app.route('/info')
def infov():
    users = User.query.all()
    return render_template('name.html', users=users)   
   

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files/resumes/')
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files/outputs/')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data/')



# Make directory if UPLOAD_FOLDER does not exist
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Make directory if DOWNLOAD_FOLDER does not exist
if not os.path.isdir(DOWNLOAD_FOLDER):
    os.mkdir(DOWNLOAD_FOLDER)
#Flask app config 
app.config['UPLOAD_CV'] = UPLOAD_CV
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['SECRET_KEY'] = 'nani?!'

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'doc','docx'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/n', methods=['GET'])
def cv():
    return _show_page()

@app.route('/n', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    app.logger.info(request.files)
    upload_files = request.files.getlist('file')
    app.logger.info(upload_files)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if not upload_files:
        flash('No selected file')
        return redirect(request.url)
    for file in upload_files:
        original_filename = file.filename
        if allowed_file(original_filename):
            extension = original_filename.rsplit('.', 1)[1].lower()
            filename = str(uuid.uuid1()) + '.' + extension
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
            files = _get_files()
            files[filename] = original_filename
            with open(file_list, 'w') as fh:
                json.dump(files, fh)
 
    flash('Upload succeeded')
    return redirect(url_for('upload_file'))
 
 
@app.route('/download/<code>', methods=['GET'])
def download(code):
    files = _get_files()
    if code in files:
        path = os.path.join(UPLOAD_FOLDER, code)
        if os.path.exists(path):
            return os.sendfile(path)
    
 
def _show_page():
    files = _get_files()
    return render_template('cv.html', files=files)
 
def _get_files():
    file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
    if os.path.exists(file_list):
        with open(file_list) as fh:
            return json.load(fh)
    return {}

@app.route('/process',methods=["POST"])
def process():
    if request.method == 'POST':

        rawtext = request.form['rawtext']
        jdtxt=[rawtext]
        resumetxt=read_files(UPLOAD_FOLDER)
        p_resumetxt = preprocess(resumetxt)
        p_jdtxt = preprocess(jdtxt)

        feats = txt_features(p_resumetxt, p_jdtxt)
        feats_red = feats_reduce(feats)

        df = simil(feats_red, p_resumetxt, p_jdtxt)

        t = pd.DataFrame({'Original Resume':resumetxt})
        dt = pd.concat([df,t],axis=1)

        dt['Phone No.']=dt['Original Resume'].apply(lambda x: get_number(x))
        
        dt['E-Mail ID']=dt['Original Resume'].apply(lambda x: get_email(x))

        dt['Original']=dt['Original Resume'].apply(lambda x: rm_number(x))
        dt['Original']=dt['Original'].apply(lambda x: rm_email(x))
        dt['Candidate\'s Name']=dt['Original'].apply(lambda x: get_name(x))

        skills = pd.read_csv(DATA_FOLDER+'skill_red.csv')
        skills = skills.values.flatten().tolist()
        skill = []
        for z in skills:
            r = z.lower()
            skill.append(r)

        dt['Skills']=dt['Original'].apply(lambda x: get_skills(x,skill))
        dt = dt.drop(columns=['Original','Original Resume'])
        sorted_dt = dt.sort_values(by=['JD 1'], ascending=False)

        out_path = DOWNLOAD_FOLDER+"Candidates.csv"
        sorted_dt.to_csv(out_path,index=False)

        return send_file(out_path, as_attachment=True)


class MessageForm(FlaskForm):
    message = StringField(
        'Message',
        validators=[DataRequired(), Length(min=0, max=140)],)
    submit = SubmitField('send')


@app.route('/send_message/<recipient>', methods=['GET', 'POST'])
@login_required
def send_message(recipient):
    user = User.query.filter_by(username=recipient).first_or_404()
    form = MessageForm()
    if form.validate_on_submit():
        msg = Message(
            sender_id=current_user.id,
            recipient_id=user.id,
            body=form.message.data)
        db.session.add(msg)
       
        db.session.commit()
        flash('Your message has been sent.')
        return redirect(url_for('profiles', username=recipient))
    return render_template(
        'send_message.html',
        title='Send Message',
        form=form,
        recipient=recipient)

@app.route('/messages')
@login_required
def messages():
    current_user.last_message_read_time = datetime.utcnow()
    
    db.session.commit()
    page = request.args.get('page', 1, type=int)
    messages = current_user.messages_received.order_by(
        Message.timestamp.desc())
    
    return render_template(
        'messages.html',
        messages=messages
        )

    
if __name__ == '__main__':
    #app.run('0.0.0.0', port=(os.environ.get("PORT", 5000)))
     app.run(debug=True)
     