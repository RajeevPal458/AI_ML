import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Load Audio Dataset
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print("Error loading:", file_path)
        return None

# 2. Load sample data
labels = ['dog_bark', 'siren', 'engine_idling']
audio_folder = 'UrbanSound8K_sample/'
files = os.listdir(audio_folder)

data = []
targets = []

for file in files:
    label = file.split('-')[-1].replace('.wav', '')  # e.g., 0 means dog_bark
    if label in ['0', '1', '2']:  # restrict to small subset
        path = os.path.join(audio_folder, file)
        features = extract_features(path)
        if features is not None:
            data.append(features)
            targets.append(label)

X = np.array(data)
X = X[..., np.newaxis]  # Add channel dimension
y = np.array(targets)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(set(y)), activation='softmax')
])

# 4. Compile, Fit, Evaluate
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# 5. Predict from new audio file
def predict_new_audio(path):
    feat = extract_features(path)
    if feat is not None:
        feat = feat[np.newaxis, ..., np.newaxis]
        prediction = model.predict(feat)
        class_idx = np.argmax(prediction)
        return encoder.inverse_transform([class_idx])[0]
    return None

# 6. Test
test_audio = 'audio/dog_bark.wav'
predicted = predict_new_audio(test_audio)
print("Predicted sound class:", predicted)
