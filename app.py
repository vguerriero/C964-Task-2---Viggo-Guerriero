import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, login_required, hash_password, login_user, current_user
import os
import pickle
import numpy as np

# Verify template folder path
template_dir = os.path.abspath('templates')
app = Flask(__name__, template_folder=template_dir)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'super-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECURITY_PASSWORD_SALT'] = 'some_salt'
app.config['SECURITY_REGISTERABLE'] = True
app.config['SECURITY_SEND_REGISTER_EMAIL'] = False

db = SQLAlchemy(app)

roles_users = db.Table('roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id')))

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    fs_uniquifier = db.Column(db.String(64), unique=True, nullable=False)
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))

user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)

# Set up logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

def initialize_database():
    with app.app_context():
        db.create_all()
        if not user_datastore.find_user(email='admin@example.com'):
            user_datastore.create_user(email='admin@example.com', password=hash_password('password'))
        db.session.commit()

@app.route('/')
@login_required
def home():
    app.logger.info('Home page accessed')
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = user_datastore.find_user(email=email)
        if user and user.verify_and_update_password(password):
            login_user(user)
            return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = hash_password(request.form['password'])
        user_datastore.create_user(email=email, password=password)
        db.session.commit()
        return redirect(url_for('login'))
    app.logger.info('Rendering register.html')
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    app.logger.info('Prediction requested')
    # Ensure model is loaded
    model_path = 'models/house_price_model.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Extract features from request
    features = [
        request.form.get('bedrooms'),
        request.form.get('bathrooms'),
        request.form.get('sqft_living'),
        request.form.get('sqft_lot'),
        request.form.get('floors'),
        request.form.get('waterfront'),
        request.form.get('view')
    ]
    
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    app.logger.info(f'Prediction made: {prediction[0]}')
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.logger.info(f'Template folder: {template_dir}')
    initialize_database()
    app.run()
