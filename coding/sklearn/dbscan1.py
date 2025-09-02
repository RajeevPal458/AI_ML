import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans
from sklearn.datasets import make_moons

# 1. Prepare the data and run DBSCAN
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
dbscan = DBSCAN(eps=0.15, min_samples=5) # Set parameters
dbscan_clusters = dbscan.fit_predict(X)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_clusters = kmeans.fit_predict(X)

# 2. Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_clusters, cmap='viridis') # Color by cluster
plt.colorbar()  # Add colorbar to show cluster assignments
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 2. Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_clusters, cmap='viridis') # Color by cluster
plt.colorbar()  # Add colorbar to show cluster assignments
plt.title('KMeans Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()