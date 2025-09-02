import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load IMDB Dataset (only top 10,000 most frequent words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# 2. Pad sequences to same length (like standardizing input shape)
maxlen = 200
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# 3. Build the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=maxlen),  # turns word indexes into vectors
    GlobalAveragePooling1D(),  # averages over the word vectors
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

# 4. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=512, validation_split=0.2)

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_acc)

#####################################################################
word_index = imdb.get_word_index()
def encode_review(text):
    tokens = [1]  # 1 is start token
    for word in text.lower().split():
        index = word_index.get(word, 2)  # 2 is "unknown"
        tokens.append(index)
    return pad_sequences([tokens], maxlen=maxlen)

sample = "The movie was absolutely wonderful and amazing!"
pred = model.predict(encode_review(sample))
print("Prediction:", pred[0][0])

