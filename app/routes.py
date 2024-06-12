from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import logging

app = Flask(__name__, template_folder='app/templates')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
model = load('models/house_price_xgb_model.pkl')

@app.route('/')
@login_required
def home():
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
                    float(request.form['floors']),
                    float(request.form['waterfront']),
                    float(request.form['view'])]

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
