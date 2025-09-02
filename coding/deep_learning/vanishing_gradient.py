import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.utils import to_categorical

X,y = make_moons(n_samples=500,noise=0.02,random_state=42)

y = to_categorical(y) # pne hot encoding for softmax

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

def build_model(activation='sigmoid',hidden_layes=10,hidden_units=32,lr=0.01):
    model= Sequential()
    
    # Input + hidden layers
    for _ in range(hidden_layes):
        model.add(Dense(hidden_units,activation=activation,input_dim=2 if _ == 0 else hidden_units))

    #Output layer (binary classification)
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer=SGD(learning_rate=lr),loss = 'categorical_crossentropy',metrics=['accuracy'])

    return model


# train models with different activations
histories = {}
activations = ['sigmoid','tanh','relu']

for act in activations:
    print(f'training with activations :{act}')
    model = build_model(activation=act,hidden_layes=10,hidden_units=32,lr=0.01)
    history = model.fit(x_train,y_train,validation_data =(x_test,y_test),epochs=50,batch_size=32,verbose=1)
    histories[act] = history

# plot loss curve

# plot loss curve
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
for act in activations:
    plt.plot(histories[act].history['loss'], label='training loss')
    plt.plot(histories[act].history['val_loss'], label='validation loss')

plt.title("loss curve - vanishing gradient effect")
plt.xlabel("epoch number")
plt.ylabel("loss value")
plt.legend()
plt.show()


# plot accuracy curve

# plot loss curve
plt.subplot(1,2,2)
for act in activations:
    plt.plot(histories[act].history['accuracy'], label='training loss')
    plt.plot(histories[act].history['val_accuracy'], label='validation loss')

plt.title("loss curve - vanishing gradient effect")
plt.xlabel("epoch number")
plt.ylabel("loss value")
plt.legend()
plt.show()