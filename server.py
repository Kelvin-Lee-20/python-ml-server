from sklearn.datasets import load_iris
import pandas as pd
from sklearn.datasets import load_wine
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans 

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)