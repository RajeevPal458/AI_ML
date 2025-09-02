import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize and reduce to 3D using PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Train logistic regression model
model = LogisticRegression(multi_class='ovr', solver='lbfgs')
model.fit(X_pca, y)

# Create 3D grid of points
x_range = np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 30)
y_range = np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 30)
z_range = np.linspace(X_pca[:, 2].min() - 1, X_pca[:, 2].max() + 1, 30)
xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# Predict class for each grid point
predictions = model.predict(grid_points)
pred_3d = predictions.reshape(xx.shape)

# Plot original points
fig = go.Figure()

colors = ['red', 'green', 'blue']
for i, name in enumerate(target_names):
    mask = y == i
    fig.add_trace(go.Scatter3d(
        x=X_pca[mask, 0], y=X_pca[mask, 1], z=X_pca[mask, 2],
        mode='markers',
        name=name,
        marker=dict(size=5, color=colors[i], line=dict(width=0.5, color='black'))
    ))

# Add the predicted decision volume as scatter points
for i in range(3):
    class_mask = pred_3d == i
    fig.add_trace(go.Scatter3d(
        x=xx[class_mask],
        y=yy[class_mask],
        z=zz[class_mask],
        mode='markers',
        marker=dict(size=1.5, color=colors[i], opacity=0.06),
        name=f'Class {target_names[i]} region',
        showlegend=False
    ))

fig.update_layout(
    title="3D PCA + Logistic Regression with Predicted Class Regions",
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    margin=dict(l=20, r=0, b=10, t=30)
)

fig.show()
