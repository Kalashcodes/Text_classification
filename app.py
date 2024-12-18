import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import os

app = Flask(__name__)
CORS(app)  

model = load_model("C:\\Users\\aabha\\Downloads\\NLC Assignment\\toxic model\\lstm_model.h5")
with open("C:\\Users\\aabha\\Downloads\\NLC Assignment\\toxic model\\tokenize.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100 

emotion_model = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\aabha\\Downloads\\NLC Assignment\\emotion_model")
emotion_tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\aabha\\Downloads\\NLC Assignment\\emotion_model")

emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer)

def preprocess_comment(comment):
    """Preprocess the comment for prediction."""
    sequence = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

def classify_comment(comment):
    """Classify the comment and return the toxicity score."""
    processed_comment = preprocess_comment(comment)
    score = model.predict(processed_comment)
    return float(score[0])  

def get_emotion(comment):
    """Detect emotion from the comment using the custom emotion model."""
    emotion_prediction = emotion_pipeline(comment)
    return {
        'emotion': emotion_prediction[0]['label'],
        'emotion_score': emotion_prediction[0]['score']
    }

@app.route('/score_comment', methods=['POST'])
def score_comment():
    data = request.get_json()
    comment_body = data['body']
    comment_id = data['commentID']

    score = classify_comment(comment_body)
    toxicity_level = 'High Toxic' if score > 0.5 else 'Low Toxic' if score > 0 else 'Non-Toxic'

    return jsonify({
        'commentID': comment_id,
        'score': score,
        'classification': 'Toxic' if score > 0 else 'Non-Toxic',
        'toxicity_level': toxicity_level
    })

@app.route('/emotion_classification', methods=['POST'])
def emotion_classification():
    data = request.get_json()
    comment_body = data['body']
    comment_id = data['commentID']

    emotion_data = get_emotion(comment_body)
    return jsonify({
        'commentID': comment_id,
        'emotion': emotion_data['emotion'],
        'emotion_score': emotion_data['emotion_score']
    })

@app.route('/process_csv', methods=['POST'])
def process_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        df = pd.read_csv(file)
        if 'commentID' not in df.columns or 'body' not in df.columns:
            return jsonify({"error": "CSV must contain 'commentID' and 'body' columns."}), 400

        results = []
        for index, row in df.iterrows():
            comment_id = row['commentID']
            comment_body = row['body']
            score = classify_comment(comment_body)
            classification = 'Toxic' if score > 0 else 'Non-Toxic'
            toxicity_level = 'High Toxic' if score > 0.5 else 'Low Toxic' if score > 0 else 'Non-Toxic'
            emotion_data = get_emotion(comment_body)
            results.append({
                'commentID': comment_id,
                'body': comment_body,
                'score': score,
                'classification': classification,
                'toxicity_level': toxicity_level,
                'emotion': emotion_data['emotion'],
                'emotion_score': emotion_data['emotion_score']
            })

        result_df = pd.DataFrame(results)
        output_csv_path = "processed_comments.csv"
        result_df.to_csv(output_csv_path, index=False)
        return send_file(output_csv_path, as_attachment=True)

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
