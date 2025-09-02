import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(500,2)
y = ((X[:,0] * X[:,1]) > 0).astype('int').reshape(-1,1)
#print(X)
#print(y)

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def sigmoind_derivative(z):
    s = sigmoid(z)
    return s*(1-s)

def relu(z):
    return np.maximum(0,z)

def relu_derivatives(z):
    return (z>0).astype('int')

#. initialization parameters
input_dim: int =2
hidden_dim: int =4
output_dim: int =1

lr = 0.01 # learning rate
epoachs = 200

w1 = np.random.randn(input_dim,hidden_dim)*0.01
b1 = np.zeros(hidden_dim).reshape(1,hidden_dim)
w2 = np.random.randn(hidden_dim,output_dim) *0.01
b2 = np.zeros(output_dim).reshape(1,output_dim)

losses = []
accuracies = []

#. Training loop SGD Style batch updates

for epoach in range(epoachs):
    # Farword propagation
    z1 = X @ w1 +b1
    A1 = relu(z1)
    z2 = A1 @ w2 +b2
    A2 = sigmoid(z2)

    #print(A2.shape, z2.shape , A1.shape, w2.shape)

    # Compute loss binary cross entroppy

    m = X.shape[0]
    loss = -(1/m)*np.mean(y*np.log(A2+1e-8) + (1-y)*np.log(1-A2 + 1e-8))
    losses.append(loss)

    # Accuracy
    pred = (A2>0.5).astype('int')
    accuracy = np.mean(pred == y)
    accuracies.append(accuracy)

    # Backword propagation
    dz2 = A2 - y
    dw2 = (A1.T @ dz2)/m
    db2 = np.mean(dz2,axis=0,keepdims=True)

    #print(dz2.shape , w2.shape, y.shape)

    dA1 = dz2 @ w2.T
    dz1 = dA1 * relu_derivatives(z1)
    dw1 = (X.T @ dz1)/m
    db1 = np.mean(dz1,axis=0,keepdims=True)

    # Parameter update SGD
    w1 -= lr*dw1
    b1 -= lr*db1
    w2 -= lr*dw2
    b2 -= lr*db2

    if(epoach % 20 == 0):
        print(f"epoach: {epoach} , loss: {loss:.2f} , accuracy: {accuracy:.2f}")

#. plot loss and accuracy curve
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses,label='training loss',color='red')
plt.xlabel('epoach')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(accuracies,label='training accuracy',color='green')
plt.xlabel('epoach')
plt.ylabel('accuracy')
plt.legend()

plt.show()