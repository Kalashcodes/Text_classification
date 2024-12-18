import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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

embedding_dim = 64

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='tanh')
])

model.compile(optimizer='adam', loss='mean_squared_error')

epochs = 5 
history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test), verbose=1)

model.save('C:\\Users\\aabha\\Downloads\\lstm_model.h5')  

with open('C:\\Users\\aabha\\Downloads\\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save_weights('C:\\Users\\aabha\\Downloads\\lstm_model_weights.weights.h5')

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")
