import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, login_required
from joblib import load
import numpy as np
import os
import uuid

# Specify the template folder
app = Flask(__name__, template_folder='app/templates')
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'super-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECURITY_PASSWORD_SALT'] = 'some_salt'

db = SQLAlchemy(app)

roles_users = db.Table('roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id')))

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    fs_uniquifier = db.Column(db.String(64), unique=True, nullable=False)
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))

user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)

# Load the model
model = load('models/house_price_xgb_model.pkl')

# Set up logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

def create_user():
    with app.app_context():  # Ensure this runs within the application context
        db.create_all()
        if not user_datastore.find_user(email='admin@example.com'):
            user_datastore.create_user(email='admin@example.com', password='password', fs_uniquifier=str(uuid.uuid4()))
        db.session.commit()

@app.route('/')
@login_required
def home():
    app.logger.info('Home page accessed')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Log the form data received
        app.logger.debug(f"Form data received: {request.form}")

        # Extract features from request form
        features = [float(request.form['bedrooms']),
                    float(request.form['bathrooms']),
                    float(request.form['sqft_living']),
                    float(request.form['sqft_lot']),
                    float(request.form['floors'])]

        app.logger.debug(f"Features extracted: {features}")

        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        app.logger.debug(f"Prediction result: {prediction}")

        # Return the result
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    create_user()  # Ensure user is created before starting the app
    app.run(debug=True)
