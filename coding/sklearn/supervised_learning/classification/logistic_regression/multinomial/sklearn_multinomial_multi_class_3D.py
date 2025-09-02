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

# Fit logistic regression model on 3D PCA data
model = LogisticRegression(multi_class='ovr', solver='lbfgs')
model.fit(X_pca, y)

# Create a dataframe for easier plotting
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df['target'] = y
df['class'] = df['target'].apply(lambda i: target_names[i])

# Plot using Plotly 3D scatter
fig = go.Figure()

colors = ['red', 'green', 'blue']
for i, name in enumerate(target_names):
    class_df = df[df['target'] == i]
    fig.add_trace(go.Scatter3d(
        x=class_df['PC1'],
        y=class_df['PC2'],
        z=class_df['PC3'],
        mode='markers',
        name=name,
        marker=dict(size=6, color=colors[i], opacity=0.8, line=dict(width=0.5, color='black'))
    ))

fig.update_layout(
    title="3D PCA + Logistic Regression Classification of Iris Dataset",
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3"
    ),
    margin=dict(l=20, r=0, b=10, t=30)
)

fig.show()
