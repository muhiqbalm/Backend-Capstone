from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the Random Forest model
model = joblib.load("rf_classifier_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON input data
        data = request.get_json()
        # Create a DataFrame from the input data
        df = pd.DataFrame(data)

        # Make predictions using the loaded model
        predictions = model.predict(df)

        # Return the predictions as JSON
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
