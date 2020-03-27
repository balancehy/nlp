# This file contains an example Flask-User application.
# To keep the example simple, we are applying some unusual techniques:
# - Placing everything in one file
# - Using class-based configuration (instead of file-based configuration)
# - Using string-based templates (instead of file-based templates)
import os, sys

from flask import Flask, render_template_string, url_for, request, render_template, json
from flask_sqlalchemy import SQLAlchemy
from flask_user import login_required, UserManager, UserMixin, SQLAlchemyAdapter

# Class-based application configuration
from sqlalchemy import engine
from werkzeug.utils import redirect
from werkzeug.exceptions import HTTPException
from wtforms import StringField
from wtforms.validators import DataRequired, ValidationError, InputRequired

from util import *
import pickle
import numpy as np
import nltk

# Global variables
with open("./model_logi.pkl", 'rb') as f:
    model = pickle.load(f)

with open("./feature_name.pkl", 'rb') as f:
    feature_name = pickle.load(f)

with open("./idf.pkl", 'rb') as f:
    feat_idf = pickle.load(f)

# Global variables
text_label = ['talk.politics.mideast', 'rec.autos', 'comp.sys.mac.hardware', 'alt.atheism', 
        'rec.sport.baseball', 'comp.os.ms-windows.misc', 'rec.sport.hockey', 'sci.crypt', 
        'sci.med', 'talk.politics.misc', 'rec.motorcycles', 'comp.windows.x', 'comp.graphics', 
        'comp.sys.ibm.pc.hardware', 'sci.electronics', 'talk.politics.guns', 'sci.space', 
        'soc.religion.christian', 'misc.forsale', 'talk.religion.misc']

RESULT_LABEL = None # store predicted label
ALLOWED_EXTENSIONS = set(['txt']) # optionally filter the input file type

nltk.download("stopwords")
nltk.download("stopwords")

# Config class
class ConfigClass(object):
    """ Flask application config """

    # Flask settings
    SECRET_KEY = 'This is an INSECURE secret!! DO NOT use this in production!!'

    # Flask-SQLAlchemy settings
    SQLALCHEMY_DATABASE_URI = 'sqlite:///quickstart_app.sqlite'  # File-based SQL database
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Avoids SQLAlchemy warning

    # Flask-User settings
    USER_APP_NAME = "Email Classification System"  # Shown in and email templates and page footers
    USER_ENABLE_EMAIL = False  # Disable email authentication
    USER_ENABLE_USERNAME = True  # Enable username authentication
    USER_REQUIRE_RETYPE_PASSWORD = False  # Simplify register form
    USER_USER_SESSION_EXPIRATION = 120 # sign out after 2 mins idle
    USER_COPYRIGHT_YEAR = 2020

def create_app():
    """ Flask application factory """

    # Create Flask app load app.config
    app = Flask(__name__)
    app.config.from_object(__name__ + '.ConfigClass')

    # Initialize Flask-SQLAlchemy
    db = SQLAlchemy(app)

    # Define the User data-model.
    # NB: Make sure to add flask_user UserMixin !!!
    class User(db.Model, UserMixin):
        __tablename__ = 'users'
        id = db.Column(db.Integer, primary_key=True)
        active = db.Column('is_active', db.Boolean(), nullable=False, server_default='1')

        # User authentication information. The collation='NOCASE' is required
        # to search case insensitively when USER_IFIND_MODE is 'nocase_collation'.
        username = db.Column(db.String(100, collation='NOCASE'), nullable=False, unique=True)
        password = db.Column(db.String(255), nullable=False, server_default='')
        email_confirmed_at = db.Column(db.DateTime())

        # User information
        first_name = db.Column(db.String(100, collation='NOCASE'), nullable=False, server_default='')
        last_name = db.Column(db.String(100, collation='NOCASE'), nullable=False, server_default='')
        middle_name = db.Column(db.String(50), nullable=True, default='')
        email = db.Column(db.String(50), nullable=True, default='')
        phone = db.Column(db.String(50), nullable=True, default='')
        address = db.Column(db.String(50), nullable=True, default='')
        occupation = db.Column(db.String(50), nullable=True, default='')

    # Create all database tables

    db.create_all()


    from flask_user.forms import RegisterForm
    class CustomRegisterForm(RegisterForm):

        first_name = StringField('first_name', validators=[DataRequired("First name is required")])
        middle_name = StringField('middle_name',validators=[DataRequired('Middle name is required')])
        last_name = StringField('last_name',validators=[DataRequired('Last name is required')])
        email = StringField('email',validators=[DataRequired('Email is required')])
        phone = StringField('phone',validators=[DataRequired('Phone number is required')])
        address = StringField('address',validators=[DataRequired('Address is required')])
        occupation = StringField('occupation',validators=[DataRequired('Occupation is required')])


    # Add a field to the UserProfile form
    # Customize Flask-User
    class CustomUserManager(UserManager):

        def customize(self, app):
            # Configure customized forms
            self.RegisterFormClass = CustomRegisterForm
        def password_validator(self, form, field): # disable password validation
            pass
        def username_validator(self, form, field):
            """
            Override the default username validator.
            """            
            # Handling error for the uniqueness of registered username
            if self.db.session.query(self.db.func.count(User.id)).filter_by(username=field.data).scalar() > 0:
                raise ValidationError('this username is already taken')

    # Setup Flask-User
    user_manager = CustomUserManager(app, db, User)

    # The Home page is accessible to anyone
    @app.route('/')
    def home_page():

        return render_template_string("""
            {% extends "flask_user_layout.html" %}
            {% block content %}
                <h2>Home page</h2>
                <p><a href={{ url_for('user.register') }}>Register</a></p>
                <p><a href={{ url_for('user.login') }}>Sign in</a></p>
                <p><a href={{ url_for('member_page') }}>Run Model</a> (login required)</p>
                <p><a href={{ url_for('user.logout') }}>Sign out</a></p>
            {% endblock %}
            """)

    # The Members page is only accessible to authenticated users via the @login_required decorator
    @app.route('/members', methods=['GET', 'POST'])
    @login_required  # User must be authenticated
    def member_page():
        # String-based templates
        global RESULT_LABEL
        res = -1
        if RESULT_LABEL is not None:
            res = RESULT_LABEL.copy()
            RESULT_LABEL = None
        
        return render_template_string("""
            {% extends "flask_user_layout.html" %}
            {% block content %}
                <p><a href={{ url_for('user.register') }}>Register</a></p>
                <p><a href={{ url_for('user.login') }}>Sign in</a></p>
                <!-- <p><a href={{ url_for('home_page') }}>Home page</a> (accessible to anyone)</p> -->
                <p><a href={{ url_for('member_page') }}>Run Model</a> (login required)</p>
                <p><a href={{ url_for('user.logout') }}>Sign out</a></p>
                
                <form action="/file" method="post" enctype="multipart/form-data">
                    Choose the file: <input type="file" name="file"/><BR>
                <input type="submit" value="Upload"/>
                </form>
                
                {% if result!=-1 %}
                    <p> Predicted class label: </p>
                    {% for value in result %}
                        <p> {{value}} </p>
                    {% endfor %}
                {% endif %}

            {% endblock %}
            """, result=res)

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

    def predict_logi():
        file_paths = ["./temp"]
        data = []
        for path in file_paths:
            data_raw = load_file(path)
            data.append(data_clean(data_raw))
        
        X = get_model_inputs(data, feature_name, feat_idf, norm_by_doclen=False)
        labels = model.predict(X)

        res = [text_label[int(x)] for x in labels]
        
        return res

    @app.route('/file', methods=['GET', 'POST'])
    def upload_file():
        global RESULT_LABEL
        if request.method == 'POST':
            file = request.files['file']
            if file and file.filename:
                filename = file.filename
                file.save("temp")
                RESULT_LABEL = predict_logi();
                
                return redirect(url_for('member_page'))
                # return member_page()
                # return render_template(url_for('member_page'))
    
    
    return app




# Start development web server
if __name__ == '__main__':
    debug = False
    print(sys.argv)
    if len(sys.argv) > 1:
        if sys.argv[1] == '-d' and sys.argv[2] == "1":
            debug = True
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=debug)
