from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the trained model
try:
    with open("pcos_stacking_svm_lr.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise Exception("Model file 'pcos_stacking_svm_lr.pkl' not found.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Expected input keys
        expected_keys = [
            "Age", "Weight", "Height", "Blood Group",
            "Menstrual Cycle Interval", "Recent Weight Gain", "Skin Darkening",
            "Hair Loss", "Acne", "Regular Fast Food Consumption", "Regular Exercise",
            "Mood Swings", "Regular Periods", "Excessive Body/Facial Hair",
            "Menstrual Duration (Days)"
        ]

        # Check for missing fields
        for key in expected_keys:
            if key not in data:
                return jsonify({'error': f"Missing field: {key}"}), 400

        # Extract and order features
        features = [data[key] for key in expected_keys]

        # Convert to numpy array and reshape
        features = np.array(features, dtype=np.float64).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        return jsonify({'pcos_diagnosis': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
