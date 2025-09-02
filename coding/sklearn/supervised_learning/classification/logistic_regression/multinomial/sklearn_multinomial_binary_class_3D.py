import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Use only class 0 (Setosa) and class 1 (Versicolor) for binary classification
binary_mask = y < 2
X = X[binary_mask]
y = y[binary_mask]

# Select 3 features for 3D visualization
X = X[:, :3]  # Using sepal length, sepal width, petal length

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression
model = LogisticRegression()
model.fit(X_scaled, y)

# Prepare scatter plot
scatter = go.Scatter3d(
    x=X_scaled[:, 0], y=X_scaled[:, 1], z=X_scaled[:, 2],
    mode='markers',
    marker=dict(size=6, color=y, colorscale='Viridis', opacity=0.85),
    name="Data Points"
)

# Create meshgrid for decision hyperplane
xx, yy = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-2, 2, 40))
coef = model.coef_[0]
intercept = model.intercept_[0]

# Solve for z using plane equation: w1*x + w2*y + w3*z + b = 0
zz = -(coef[0]*xx + coef[1]*yy + intercept) / coef[2]

# Create decision surface
plane = go.Surface(
    x=xx, y=yy, z=zz,
    showscale=False,
    opacity=0.5,
    name="Decision Hyperplane",
    colorscale='Blues'
)

# Plot with enhanced layout
fig = go.Figure(data=[scatter, plane])
fig.update_layout(
    title="3D Logistic Regression Hyperplane (Iris: Setosa vs Versicolor)",
    scene=dict(
        xaxis_title=feature_names[0],  # sepal length
        yaxis_title=feature_names[1],  # sepal width
        zaxis_title=feature_names[2]   # petal length
    ),
    width=600,
    height=500,
    margin=dict(l=10, r=0, b=10, t=0)
)

fig.show()
