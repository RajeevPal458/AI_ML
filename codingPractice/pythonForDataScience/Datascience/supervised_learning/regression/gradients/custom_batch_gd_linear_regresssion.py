import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score
import matplotlib.pyplot as plt

class BatchGDRegressor:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.B = 0
        self.losses = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        m, n = X.shape
        self.W = np.zeros(n)
        self.B = 0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.W) + self.B
            error = y - y_pred

            # Compute gradients over full batch
            dw = (-2 / m) * np.dot(X.T, error)
            db = (-2 / m) * np.sum(error)

            # Update weights and bias
            self.W -= self.lr * dw
            self.B -= self.lr * db

            # Compute loss (MSE)
            loss = (1 / m) * np.sum(error ** 2)
            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | W: {self.W[:3]}... | B: {self.B:.4f}")

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.dot(X, self.W) + self.B

# Load dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

X = df.drop('target', axis=1)
y = df['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train using Batch Gradient Descent
model = BatchGDRegressor(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f"\nTest MSE: {mse:.4f}")

rmse = root_mean_squared_error(y_test,predictions)
print(f"rmse : ",rmse)

r2score = r2_score(y_test,predictions)
print(f"r2 score : ",r2score)


# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, color='purple', alpha=0.6, edgecolors='k', label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Batch GD: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
