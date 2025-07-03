# Import Library
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Inisialisasi Flask App
app = Flask(__name__)

# Global variables
model = None
vectorizer = None
model_type = None

# class RBF Neural Network
class RBFNN:
    def __init__(self, num_centroids=10):
        self.num_centroids = num_centroids
        self.centroids = None
        self.beta = None
        self.model = LogisticRegression()

# Fungsi Aktivasi RBF (Gaussian)
    def _rbf(self, X, centroid):
        return np.exp(-self.beta * np.linalg.norm(X - centroid, axis=1)**2)

# Training RBFNN
    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.num_centroids, random_state=42)
        kmeans.fit(X.toarray())
        self.centroids = kmeans.cluster_centers_

        max_distance = np.max([np.linalg.norm(a - b) for a in self.centroids for b in self.centroids])
        self.beta = 1 / (2 * (max_distance**2))

        transformed_X = np.array([self._rbf(X.toarray(), c) for c in self.centroids]).T
        self.model.fit(transformed_X, y)

# Klasifikasi RBFNN
    def predict(self, X):
        transformed_X = np.array([self._rbf(X.toarray(), c) for c in self.centroids]).T
        return self.model.predict(transformed_X)

    def predict_proba(self, X):
        transformed_X = np.array([self._rbf(X.toarray(), c) for c in self.centroids]).T
        return self.model.predict_proba(transformed_X)

# Flask Route (Halaman Utama) 
@app.route('/')
def index():
    return render_template('index.html')

# Flask Route (training Model)
@app.route('/train', methods=['POST'])
def train():
    global model, vectorizer, model_type
    file = request.files['file']
    model_type = request.form.get('model_type')

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

# Validasi CSV
    data = pd.read_csv(file, sep=';')
    if 'Email Text' not in data.columns or 'Label' not in data.columns:
        return jsonify({"error": "Invalid file format. Must contain 'Email Text' and 'Label' columns"}), 400

# Proses Data dan Vectorization
    data['Label'] = data['Label'].str.strip()
    X = data['Email Text']
    y = data['Label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_val_vect = vectorizer.transform(X_val)

# Pemilihan Model (Naive Bayes atau RBFNN)
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'rbf':
        model = RBFNN(num_centroids=10)
    else:
        return jsonify({"error": "Invalid model type selected"}), 400

    model.fit(X_train_vect, y_train)

# Evaluasi Model
    y_val_pred = model.predict(X_val_vect)
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, pos_label='Spam', zero_division=0)
    recall = recall_score(y_val, y_val_pred, pos_label='Spam', zero_division=0)
    f1 = f1_score(y_val, y_val_pred, pos_label='Spam', zero_division=0)

    return jsonify({
        "model": model_type,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

# Klasifikasi Email Baru
@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer, model_type
    if not model or not vectorizer:
        return jsonify({"error": "Model not trained yet"}), 400

    text = request.form.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vect)[0]
        spam_proba = proba[1]
        accuracy_percentage = spam_proba * 100 if prediction == "Spam" else (1 - spam_proba) * 100
    else:
        accuracy_percentage = 100.0

    return jsonify({
        "prediction": prediction,
        "accuracy_percentage": accuracy_percentage,
        "model": model_type
    })

if __name__ == '__main__':
    app.run(debug=True)
