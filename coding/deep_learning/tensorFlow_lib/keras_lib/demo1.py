import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset (32x32 RGB images, 10 classes)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize image data to 0-1 range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model definition
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
residual = x = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, padding="same")(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(64, 3, padding="same")(x)
x = layers.Activation("relu")(x)

# Residual connection
x = x + residual

x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="mini_resnet")
model.summary()

# Save model architecture diagram
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", test_acc)

# Predict on one image
import numpy as np
sample_image = x_test[0:1]  # shape: (1, 32, 32, 3)
prediction = model.predict(sample_image)
predicted_label = np.argmax(prediction)

print("\nPredicted label:", predicted_label)

# Show the image and predicted label
plt.imshow(x_test[0])
plt.title(f"Predicted Label: {predicted_label}")
plt.axis('off')
plt.show()
