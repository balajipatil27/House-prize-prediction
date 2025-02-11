from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and data columns
model = pickle.load(open("finalized_model.sav", "rb"))
data_columns = pickle.load(open("training_columns.pickle", "rb"))  # Column names used during training

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/get_locations', methods=['GET'])
def get_locations():
    """Fetches the location names dynamically from the pickle file."""
    locations = data_columns[3:]  # Assuming first 3 columns are numeric and rest are locations
    return jsonify({'locations': locations})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Create feature array of zeros
        features = np.zeros(len(data_columns))

        # Fill numerical features
        features[0] = float(data['total_sqft'])
        features[1] = int(data['bhk'])
        features[2] = float(data['bath'])

        # Handle categorical 'location' feature
        if data['location'] in data_columns:
            loc_index = data_columns.index(data['location'])
            features[loc_index] = 1  # One-hot encoding

        # Make prediction
        prediction = model.predict([features])[0]

        return jsonify({'estimated_price': abs(round(prediction, 2))})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
