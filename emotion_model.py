import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')  

file_path = "C:\\Users\\aabha\\Downloads\\NLC Assignment\\updated_reddit_comments_with_names.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please check the file path.")
    exit()

model_name = "nateraw/bert-base-uncased-emotion"
emotion_pipeline = pipeline("text-classification", model=model_name, return_all_scores=False)

def classify_comments(dataframe, toxic_threshold=0.5):
    def classify(score):
        return 'Toxic' if score > 0 else 'Non-Toxic'
    
    def toxic_level(score):
        if score > toxic_threshold:
            return 'High Toxic'
        elif score > 0:
            return 'Low Toxic'
        else:
            return ''
    
    dataframe['classification'] = dataframe['score'].apply(classify)
    dataframe['toxicity_level'] = dataframe['score'].apply(toxic_level)
    return dataframe

def display_classification_by_comment_id(comment_id, dataframe):
    classified_df = classify_comments(dataframe)
    comment_data = classified_df[classified_df['comment_id'] == comment_id]

    if not comment_data.empty:
        results = []
        for _, row in comment_data.iterrows():
            comment_body = row['body']
            emotion_prediction = emotion_pipeline(comment_body)
            emotion_label = emotion_prediction[0]['label']
            emotion_score = emotion_prediction[0]['score']

            result = {
                'comment_id': row['comment_id'],
                'body': row['body'],
                'score': row['score'],
                'classification': row['classification'],
                'toxicity_level': row['toxicity_level'],
                'emotion': emotion_label,
                'emotion_score': emotion_score
            }
            results.append(result)

        return results
    else:
        return f"No comments found for the comment_id '{comment_id}'"

comment_id_input = input("Enter the comment_id to search for: ")
results = display_classification_by_comment_id(comment_id_input, df)

if isinstance(results, list):
    for result in results:
        print("\nComment ID:", result['comment_id'])
        print("Comment Body:", result['body'])
        print("Score:", result['score'])
        print("Toxicity Classification:", result['classification'])
        print("Toxicity Level:", result['toxicity_level'])
        print("Detected Emotion:", result['emotion'])
        print("Emotion Score:", result['emotion_score'])
else:
    print(results)

model_save_path = "C:\\Users\\aabha\\Downloads\\NLC Assignment\\emotion_model"
emotion_pipeline.model.save_pretrained(model_save_path)
emotion_pipeline.tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")
