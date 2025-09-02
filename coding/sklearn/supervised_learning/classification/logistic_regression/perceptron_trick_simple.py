import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Step function for classification
def step_function(z):
    return np.where(z >= 0, 1, 0)


# Outer PerceptronTrick class
class PerceptronTrick:
    class Model:
        def __init__(self, n_features, lr=0.1):
            # +1 for bias
            self.W = np.zeros(n_features + 1)
            self.lr = lr

        def predict(self, X):
            # Augment feature with 1 for bias
            X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
            z = np.dot(X_aug, self.W)
            return step_function(z)

        def train(self, X, y, epochs=1000):
            # Augment feature with 1 for bias
            X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
            for _ in range(epochs):
                for xi, yi in zip(X_aug, y):
                    y_hat = step_function(np.dot(xi, self.W))
                    error = yi - y_hat
                    self.W += self.lr * error * xi

# Create 2D sparse, noisy classification dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=1, 
                           n_redundant=0, n_clusters_per_class=1, flip_y=0.1, class_sep=10, random_state=42)

# Train the model
model = PerceptronTrick.Model(n_features=2, lr=0.01)
model.train(X, y, epochs=1000)

# Predict for visualization
y_pred = model.predict(X)

# Plot original data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', label='Data')

# Plot decision boundary
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
# Augmented line equation: w1*x1 + w2*x2 + w0 = 0 → x2 = -(w1*x1 + w0) / w2
w1, w2, b = model.W
x2_boundary = -(w1 * x1_range + b) / w2
plt.plot(x1_range, x2_boundary, 'k--', label='Decision Boundary')

plt.title("Perceptron Trick with Step Function")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
