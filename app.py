# file: task_priority_api.py
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)

# Load training data from CSV file
def load_training_data(filename="task_priority_samples.csv"):
    df = pd.read_csv(filename, encoding='utf-8-sig')
    return df['text'].tolist(), df['label'].tolist()

train_texts, train_labels = load_training_data()

# Vectorizer and model (character analyzer for Arabic + English)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
model = MultinomialNB()

def train_model():
    X_train = vectorizer.fit_transform(train_texts)
    model.fit(X_train, train_labels)

# Initial training
train_model()

@app.route('/predict_priority', methods=['POST'])
def predict_priority():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]

    return jsonify({"priority": prediction})

@app.route('/train', methods=['POST'])
def train_new_sample():
    data = request.get_json()
    text = data.get("text")
    label = data.get("label")

    if not text or not label:
        return jsonify({"error": "Text and label are required"}), 400

    train_texts.append(text)
    train_labels.append(label)
    train_model()

    return jsonify({"message": "Model updated with new training sample."})

if __name__ == '__main__':
    app.run(port=5000)
