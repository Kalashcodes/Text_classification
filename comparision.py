import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

file_path = "C:\\Users\\aabha\\Downloads\\NLC Assignment\\cleaned_reddit_comments_score.csv"
df = pd.read_csv(file_path)

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['body'])

vocab_size = min(max_words, len(tokenizer.word_index) + 1)

sequences = tokenizer.texts_to_sequences(df['body'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['score'], test_size=0.2, random_state=42)


# Model 1: LSTM
def create_lstm_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Model 2: GRU
def create_gru_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        GRU(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Model 3: Bidirectional LSTM
def create_bilstm_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(32, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

models = {
    "LSTM": create_lstm_model(),
    "GRU": create_gru_model(),
    "Bidirectional_LSTM": create_bilstm_model()
}

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name} model...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    results[model_name] = {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }
    
    print(f"{model_name} Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R2 Score: {r2}")
