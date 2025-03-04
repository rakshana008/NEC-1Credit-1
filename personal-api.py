# personal-api.py
from flask import Flask, jsonify
import joblib

# Create an instance of Flask
app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('iris_model.pkl')

# Define the routes and their corresponding functions
@app.route('/')
def personal_details():
    return 'Hello, World!'

@app.route('/name')
def get_name():
    return 'Rakshana'

@app.route('/regno')
def get_regno():
    return '22IT037'

@app.route('/department')
def get_dept():
    return 'Information Technology'

@app.route('/predict', methods=['GET'])
def predict():
    # Example input for prediction: [5.1, 3.5, 1.4, 0.2] (Iris-setosa)
    input_data = [5.1, 3.5, 1.4, 0.2]
    prediction = model.predict([input_data])
    return jsonify({'prediction': str(prediction[0])})

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

