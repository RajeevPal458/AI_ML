"""
Scikit-learn's SGDRegressor uses Stochastic Gradient Descent, but it allows you to control the batch behavior via:

learning_rate='constant'

eta0=... (learning rate)

average=False (no model averaging)

penalty=None (no regularization)

max_iter and tol (for convergence)

batch_size is not exposed, but:

If you use SGDRegressor with batch_size = len(X), it mimics Batch GD.

However, SGDRegressor in scikit-learn does not support full-batch mode explicitly.
To simulate batch gradient descent, you must set **learning_rate='constant'** and a low number of iterations, and ensure the data is passed all at once.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load and split the data
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Standardize features (important for SGD convergence)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_gd = SGDRegressor(
    loss='squared_error',
    penalty=None,
    learning_rate='constant',
    eta0=0.01,
    max_iter=1000,
    tol=None,
    random_state=42
)

model_gd.fit(X_train, y_train)
pred_gd = model_gd.predict(X_test)
mse_gd = mean_squared_error(y_test, pred_gd)
print(f"1. Batch GD (1 batch per epoch) MSE: {mse_gd:.4f}")


####

model_sgd = SGDRegressor(
    loss='squared_error',
    penalty=None,
    learning_rate='constant',
    eta0=0.01,
    max_iter=1,  # we'll control epochs manually
    tol=None,
    shuffle=False,
    random_state=42
)

n_epochs = 1000
for epoch in range(n_epochs):
    i = np.random.randint(0, len(X_train))  # random row index
    xi = X_train[i].reshape(1, -1)
    yi = y_train[i].reshape(-1)
    model_sgd.partial_fit(xi, yi)

pred_sgd = model_sgd.predict(X_test)
mse_sgd = mean_squared_error(y_test, pred_sgd)
print(f"2. SGD (random one sample per update) MSE: {mse_sgd:.4f}")


####

model_mbgd = SGDRegressor(
    loss='squared_error',
    penalty=None,
    learning_rate='constant',
    eta0=0.01,
    max_iter=1,
    tol=None,
    shuffle=False,
    random_state=42
)

batch_size = 32
n_epochs = 1000
n_samples = len(X_train)

for epoch in range(n_epochs):
    idx = np.random.randint(0, n_samples - batch_size)
    X_batch = X_train[idx:idx + batch_size]
    y_batch = y_train[idx:idx + batch_size]
    model_mbgd.partial_fit(X_batch, y_batch)

pred_mbgd = model_mbgd.predict(X_test)
mse_mbgd = mean_squared_error(y_test, pred_mbgd)
print(f"3. Mini-Batch GD (batch size = {batch_size}) MSE: {mse_mbgd:.4f}")


###


plt.figure(figsize=(10, 6))

plt.scatter(y_test, pred_gd, alpha=0.5, label="Batch GD", marker='o')
plt.scatter(y_test, pred_sgd, alpha=0.5, label="SGD", marker='x')
plt.scatter(y_test, pred_mbgd, alpha=0.5, label="Mini-Batch GD", marker='^')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted: GD vs SGD vs Mini-Batch GD")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
