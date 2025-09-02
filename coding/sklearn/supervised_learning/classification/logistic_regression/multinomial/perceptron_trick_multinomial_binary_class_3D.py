import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :3]  # First 3 features
y = (iris.target != 0).astype(int)  # Binary classification: class 0 vs. classes 1 & 2

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Custom Logistic Regression class using sigmoid
class LogisticRegressionCustom:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.X = np.insert(X, 0, 1, axis=1)  # Add bias term
        self.y = y.reshape(-1, 1)
        self.weights = np.zeros((self.X.shape[1], 1))
        m,n = self.X.shape
        for _ in range(self.epochs):
            z = np.dot(self.X, self.weights)
            pred = self.sigmoid(z)
            error = pred - self.y
            #print(f" error type : {type(error)} , size : {error.shape}")
            dw = (1/m) * np.dot(self.X.T, error)
            self.weights -= self.lr * dw   # dw is gradient

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Train model
model = LogisticRegressionCustom(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

# Generate meshgrid for decision surface
x_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 50)
y_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 50)
x_grid, y_grid = np.meshgrid(x_range, y_range)
z_grid = -(model.weights[0] + model.weights[1]*x_grid + model.weights[2]*y_grid) / model.weights[3]

# Plot with Plotly
fig = go.Figure()

# Class 0 points
fig.add_trace(go.Scatter3d(
    x=X_scaled[y == 0, 0],
    y=X_scaled[y == 0, 1],
    z=X_scaled[y == 0, 2],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Class 0'
))

# Class 1 points
fig.add_trace(go.Scatter3d(
    x=X_scaled[y == 1, 0],
    y=X_scaled[y == 1, 1],
    z=X_scaled[y == 1, 2],
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Class 1'
))

# Decision boundary
fig.add_trace(go.Surface(
    x=x_range, y=y_range, z=z_grid,
    showscale=False, opacity=0.5,
    name='Decision Surface'
))

fig.update_layout(
    title="3D Logistic Regression Decision Surface",
    scene=dict(
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        zaxis_title="Feature 3"
    ),
    width=600,
    height=500,
    margin=dict(l=10, r=0, b=10, t=0)
)

fig.show()
