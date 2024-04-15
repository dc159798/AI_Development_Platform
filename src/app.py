from flask import Flask, request, jsonify
import pandas as pd
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['ai_platform_db']

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def hello_world():
    return 'Hello, World connected to MongoDB!'


UPLOAD_FOLDER = '/home/haxck/Desktop/AI_Development_Platform/data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return jsonify({'message': 'CSV file uploaded successfully', 'filename': filename})
    
    
@app.route('/preprocess_csv', methods=['POST'])
def preprocess_csv():
    data = request.json
    filename = data['filename']
    # Read the CSV file
    df = pd.read_csv(filename)
    # Perform basic preprocessing (e.g., remove duplicates, handle missing values)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    # Save the preprocessed data
    preprocessed_filename = filename.replace('.csv', '_preprocessed.csv')
    df.to_csv(preprocessed_filename, index=False)
    return jsonify({'message': 'Data preprocessed successfully', 'preprocessed_filename': preprocessed_filename})


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.json
    preprocessed_filename = data['preprocessed_filename']
    # Read the preprocessed CSV file
    df = pd.read_csv(preprocessed_filename)
    X = df.drop('target', axis=1)  # Assuming 'target' is the target variable
    y = df['target']
    # Train the model
    model.fit(X, y)
    # Make predictions
    predictions = model.predict(X)
    return jsonify({'message': 'Model trained and predictions made successfully', 'predictions': predictions.tolist()})


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        # Save the file to the server
        file.save(os.path.join('/home/haxck/Desktop/AI_Development_Platform/data', filename))
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})

@app.route('/predict', methods=['GET'])
def predict():
    # Just using the first sample for prediction
    prediction = model.predict([X[0]])
    return jsonify({'prediction': str(prediction)})

@app.route('/predict_iris', methods=['POST'])
def predict_iris():
    data = request.json
    features = data['features']
    prediction = model.predict([features])
    predicted_class = iris.target_names[prediction[0]]
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
