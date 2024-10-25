from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model from a .pkl file
model_filename = 'optimized_logistic_regression_model.pkl'
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
model_path = os.path.join(base_dir, model_filename)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Helper function to process input data
def preprocess_input(data):
    # Convert categorical data into numerical format as done during training
    # For Gender: Male -> 1, Female -> 0
    # For Own_Car and Own_Housing: Yes -> 1, No -> 0
    gender_map = {"Male": 1, "Female": 0}
    own_car_map = {"Yes": 1, "No": 0}
    own_housing_map = {"Yes": 1, "No": 0}
    
    # Convert the features to match the format used during model training
    processed_data = []
    for i in range(len(data['Num_Children'])):
        processed_data.append([
            data['Num_Children'][i],
            gender_map[data['Gender'][i]],
            data['Income'][i],
            own_car_map[data['Own_Car'][i]],
            own_housing_map[data['Own_Housing'][i]]
        ])

    # Convert the processed data to a DataFrame with correct column names
    column_names = ["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]
    return pd.DataFrame(processed_data, columns=column_names)

# API route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request (expects JSON)
    input_data = request.json

    # Ensure input features match the expected feature names and values
    expected_features = ["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]
    for feature in expected_features:
        if feature not in input_data:
            return jsonify({"error": f"Missing required feature: {feature}"}), 400

    # Preprocess the input data to match the model's expected format
    processed_data = preprocess_input(input_data)
    
    # Perform prediction
    predictions = model.predict(processed_data)

    # Convert numpy array to list for JSON response
    predictions = predictions.tolist()

    # Return predictions as JSON response
    return jsonify({'predictions': predictions})

# Run the API
if __name__ == '__main__':
    app.run(debug=True)
