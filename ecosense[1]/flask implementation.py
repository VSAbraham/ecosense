from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load("your_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json

    # Perform prediction using the loaded model
    prediction = model.predict([data['input']])

    # Return prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
