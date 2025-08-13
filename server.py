from sklearn.datasets import load_iris
import pandas as pd
from sklearn.datasets import load_wine
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans 
from transformers import pipeline
from flask import request
import transformers
import ssl
from io import BytesIO
from PIL import Image
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
import random
from ultralytics import YOLO
import cv2
import io

ssl._create_default_https_context = ssl._create_unverified_context

def load_iris_data():
    iris = load_iris()
    return iris

def load_wine_data():
    wine = load_wine()
    return wine

app = Flask(__name__)
CORS(app) 

@app.route('/api/kmeans/<int:k>', methods=['GET'])
def kmeans(k):
    iris = load_iris_data()
    kmeans = KMeans(n_clusters = k, random_state = 0)
    y_kmeans = kmeans.fit_predict(iris.data)
    data = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'data': iris.data.tolist(),
        'target': iris.target.tolist(),
        'cluster_centers_': kmeans.cluster_centers_.tolist(),
        'y_kmeans': y_kmeans.tolist()
    }
    return jsonify(data)

@app.route('/api/sa', methods=['GET'])
def sa():
    text = request.args.get('text')
    if text is None or text.strip() == '':
        return jsonify({"error": "Resource not found"}), 404
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier(text)
    data = {
        'label': result[0]['label'],
        'score': result[0]['score']
    }
    return jsonify(data)

@app.route('/api/image-classification', methods=['POST'])
def imageClassification():
    if 'file' not in request.files:
        return jsonify({"error": "Resource not found"}), 404
    file = request.files['file']
    if file:
        try:
            model = ResNet50(weights='imagenet')
            img_bytes = file.read()
            img_stream = BytesIO(img_bytes)
            img = Image.open(img_stream)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))

            # img_path = 'https://live.staticflickr.com/41/114587853_cdabf6568c_o.jpg'
            # img_path = 'http://images.cocodataset.org/train2017/000000391898.jpg'
            # local_path = get_file(origin=img_path)
            # img = image.load_img(local_path, color_mode='rgb', target_size=(224, 224))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            predictions = decode_predictions(preds, top=3)[0]
            result = [
                {'label': pred[1], 'probability': float(pred[2])} 
                for pred in predictions
            ]
            return jsonify({'predictions': result})
        except Exception as e:
            return jsonify({"error": "Resource not found"}), 404

@app.route('/api/objectdetect', methods=['POST'])
def od():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    model = YOLO('yolov8n.pt')  # yolov8n.pt is the nano version (smallest)

    try:

        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        results = model(img)
        
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy.tolist()[0]  # Convert tensor to list
                })
        
        data = {
            # 'class_names': model.names,
            'detections': detections
        }
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)