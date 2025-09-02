# STOCHASTIC GRADIENT DESCENT
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class SGDMultipleLinearRegressor:
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
            # Shuffle the data
            indices = np.arange(m)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(m):
                x_i = X[i]
                y_i = y[i]

                y_pred = np.dot(x_i, self.W) + self.B
                error = y_i - y_pred

                # Gradients
                dw = -2 * x_i * error
                db = -2 * error

                # Update weights
                self.W -= self.lr * dw
                self.B -= self.lr * db

            # Loss for current epoch (MSE)
            y_pred_all = np.dot(X, self.W) + self.B
            loss = (1 / m) * np.sum((y - y_pred_all) ** 2)
            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | W: {self.W[:3]}... | B: {self.B:.4f}")

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.dot(X, self.W) + self.B

np.random.seed(42)
# Load and prepare data
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using SGD
model = SGDMultipleLinearRegressor(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test,predictions)
print(f"rmse : ",np.sqrt(mse))

r2score = r2_score(y_test,predictions)
print(f"r2 score : ",r2score)



import matplotlib.pyplot as plt

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.6, edgecolors='k', label="Predicted")
plt.plot([y_test.min(), y_test.max()], [predictions.min(), predictions.max()], 'r--', lw=2, label="Perfect Prediction")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("SGD Regressor: Actual vs Predicted on Test Set")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
