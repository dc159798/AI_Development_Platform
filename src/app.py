from flask import Flask, request, jsonify
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

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
        file.save(os.path.join('/path/to/save', filename))
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})

@app.route('/predict', methods=['GET'])
def predict():
    # Just using the first sample for prediction
    prediction = model.predict([X[0]])
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
