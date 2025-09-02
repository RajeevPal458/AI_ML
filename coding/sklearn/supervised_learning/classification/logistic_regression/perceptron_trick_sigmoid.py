import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class PerceptronTrick:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = self.Model()

    class Model:
        def __init__(self):
            self.w = None
            self.b = 0

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        def train(self, X, y, learning_rate, epochs):
            n_samples, n_features = X.shape
            self.w = np.zeros(n_features)
            self.b = 0

            for epoch in range(epochs):
                for i in range(n_samples):
                    z = np.dot(self.w, X[i]) + self.b
                    y_hat = self.sigmoid(z)
                    error = y_hat - y[i]
                    self.w -= learning_rate * error * X[i]
                    self.b -= learning_rate * error

        def predict(self, X):
            z = np.dot(X, self.w) + self.b
            return (self.sigmoid(z) >= 0.5).astype(int)

        def decision_boundary(self, X):
            x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
            x2 = -(self.w[0] * x1 + self.b) / self.w[1]
            return x1, x2

    def fit(self, X, y):
        self.model.train(X, y, self.learning_rate, self.epochs)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        print(f"Training accuracy: {accuracy:.2f}")
        return y_pred

    def plot_decision_boundary(self, X, y):
        x1, x2 = self.model.decision_boundary(X)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
        plt.plot(x1, x2, 'k--', label='Decision Boundary')
        plt.title("Logistic Regression using Perceptron Trick")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

# Step 1: Create sparse and noisy data
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, flip_y=0.1, class_sep=10, random_state=42
)

# Step 2: Create instance and train
clf = PerceptronTrick(learning_rate=0.1, epochs=1000)
clf.fit(X, y)

# Step 3: Evaluate
clf.evaluate(X, y)

# Step 4: Plot decision boundary
clf.plot_decision_boundary(X, y)
