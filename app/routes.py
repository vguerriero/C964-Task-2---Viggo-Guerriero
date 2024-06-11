from flask import render_template, request, jsonify
from app import app
import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/house_price_model.pkl')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['features']])
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})
 
