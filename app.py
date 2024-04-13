# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.joblib')

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # The method of extracting JSON data from an HTTP request. In this code, it is used to retrieve JSON data from a POST request sent by the user, which contains the feature values used for prediction.
    data = request.get_json()
    # Extract features and convert to NumPy array
    features = [
        data['Dew_Point_Temp_C'],
        data['Press_kPa'],
        data['Rel_Hum_%'],
        data['Wind_Speed_km/h']
    ]

    # Converts the list of features into a NumPy array and reshapes it into a 2D array with one row and as many columns as there are features.
    data_array = np.array(features).reshape(1, -1)

    # Make predictions using the model
    prediction = model.predict(data_array)

    # Convert the prediction result to JSON format and return it
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
