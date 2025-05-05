from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd # Import pandas

# Define the expected feature names in the correct order
# This list MUST match the column order of the DataFrame 'X' used during training
expected_feature_order = [
    "Age", "Weight", "Height", "Blood Group",
    "Menstrual Cycle Interval", "Recent Weight Gain", "Skin Darkening",
    "Hair Loss", "Acne", "Regular Fast Food Consumption", "Regular Exercise",
    "Mood Swings", "Regular Periods", "Excessive Body/Facial Hair",
    "Menstrual Duration (Days)"
]

# Load the trained model
# This model was trained on the raw DataFrame X in your ML script
try:
    with open("pcos_stacking_svm_lr.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'pcos_stacking_svm_lr.pkl' not found.")
    print("Please ensure the pickled model file is in the correct location.")
    # In a production environment, you might want to exit or handle this more gracefully
    model = None # Set to None if loading fails

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model was loaded successfully
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Expected input keys (should match the keys in your JSON request)
        expected_keys_in_request = [
            "Age", "Weight", "Height", "Blood Group",
            "Menstrual Cycle Interval", "Recent Weight Gain", "Skin Darkening",
            "Hair Loss", "Acne", "Regular Fast Food Consumption", "Regular Exercise",
            "Mood Swings", "Regular Periods", "Excessive Body/Facial Hair",
            "Menstrual Duration (Days)"
        ]

        # Check for missing fields in the request data
        for key in expected_keys_in_request:
            if key not in data:
                return jsonify({'error': f"Missing field in input data: {key}"}), 400

        # Create a pandas DataFrame from the input data
        # It is CRUCIAL to maintain the correct column order as defined by expected_feature_order
        # The model was trained on a DataFrame with this specific column order
        input_df = pd.DataFrame([data], columns=expected_feature_order)

        # Ensure data types in the DataFrame match the types in your training DataFrame X
        # This is especially important for categorical features like 'Blood Group'
        # If 'Blood Group' was an object/string in training, it should be here too.
        # Based on your ML code, it seems you might be treating these as numbers,
        # but explicitly setting types can prevent issues.
        # Example (adjust types based on your actual data in CLEAN- PCOS SURVEY SPREADSHEET.csv):
        # input_df['Blood Group'] = input_df['Blood Group'].astype(str)
        # input_df['Menstrual Cycle Interval'] = input_df['Menstrual Cycle Interval'].astype(int)
        # ... other discrete features ...


        # Make prediction using the loaded model
        # The model expects input in the DataFrame format it was trained on
        prediction = model.predict(input_df)
        # Get prediction probabilities as well for confidence
        prediction_proba = model.predict_proba(input_df)


        # Return the prediction as an integer
        return jsonify({
            'pcos_diagnosis': int(prediction[0]),
            'probability_no_pcos': float(prediction_proba[0][0]),
            'probability_pcos': float(prediction_proba[0][1])
            })

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Ensure debug=True is only used during development
    app.run(debug=True, port=5000)

