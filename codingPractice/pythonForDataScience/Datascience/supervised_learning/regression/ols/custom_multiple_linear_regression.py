import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class MyMultiLinearRegressor:
    def __init__(self):
        self.weights = None  # includes bias as the first element

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values  # convert to NumPy
        if isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)

        # Add bias column (intercept term): shape becomes (m, n+1)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))  # X_b = [1, x1, x2, ..., xn]

        # Normal Equation: (X^T X)^-1 X^T y
        XtX = X_b.T @ X_b
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_b.T @ y
        self.weights = XtX_inv @ Xty  # (n+1, 1)

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X = X.values

        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))  # Add bias term
        return (X_b @ self.weights).flatten()


# Load dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = MyMultiLinearRegressor()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Show sample predictions
print("Predicted:", predictions[:5])
print("Actual   :", y_test.values[:5])


# ===========
import matplotlib.pyplot as plt

# Scatter: predicted vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')

# Line: perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Multiple Linear Regression: Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.show()





"""
Example with Intercept Column
If you want to add a bias term (column of 1s):

X = np.array([[2], [4], [6]])
ones = np.ones((X.shape[0], 1))

# All of these are valid:
X_b1 = np.hstack((ones, X))
X_b2 = np.concatenate((ones, X), axis=1)
X_b3 = np.column_stack((ones.ravel(), X.ravel()))
X_b4 = np.c_[ones, X]

print(X_b1)
All will give:

[[1. 2.]
 [1. 4.]
 [1. 6.]]
 
 =============================
 @ — Matrix Multiplication (dot product)
Introduced in Python 3.5 as a dedicated operator for matrix multiplication.
Equivalent to:
A @ B  ≡  np.dot(A, B)


So in my case (Linear Regression):
X_b.T @ X_b    # Correct → Matrix multiplication for (X^T X)
X_b.T * X_b    # ❌ Wrong → Tries to do element-wise, will break or give wrong results

"""

import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
print("A @ B =\n", A @ B)

# Element-wise multiplication
print("A * B =\n", A * B)

