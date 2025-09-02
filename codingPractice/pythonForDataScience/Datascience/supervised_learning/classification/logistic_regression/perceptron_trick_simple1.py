import numpy as np
import matplotlib.pyplot as plt

class PerceptronTrick:
    def __init__(self, use_sigmoid=False):
        self.model = self.Model(use_sigmoid)

    class Model:
        def __init__(self, use_sigmoid=False, lr=0.1, epochs=1000):
            self.W = None
            self.lr = lr
            self.epochs = epochs
            self.use_sigmoid = use_sigmoid

        def _step_function(self, z):
            return 1 if z >= 0 else 0

        def _sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
            #return (val >= 0.5).astype(int)

        def _activation(self, z):
            if self.use_sigmoid:
                return self._sigmoid(z)
            else:
                return self._step_function(z)

        def fit(self, X, y):
            # Add bias (1) as first column
            X = np.insert(X, 0, 1, axis=1)
            self.W = np.zeros(X.shape[1])

            for _ in range(self.epochs):
                for xi, target in zip(X, y):
                    z = np.dot(xi, self.W)
                    y_hat = self._activation(z)
                    #print(f"yhat :{y_hat}")
                    error = target - y_hat
                    self.W += self.lr * error * xi

        def predict(self, X):
            X = np.insert(X, 0, 1, axis=1)
            z = np.dot(X, self.W)
            if self.use_sigmoid:
                return (self._sigmoid(z) >= 0.5).astype(int)
            else:
                return np.array([self._step_function(val) for val in z])

        def decision_boundary(self, x1_range):
            # Line: w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1)/w2
            w0, w1, w2 = self.W
            return -(w0 + w1 * x1_range) / w2

# --- Create toy noisy data ---
np.random.seed(42)
num_samples = 100

# Features (sparse-ish and noisy)
X = np.random.randn(num_samples, 2) * 2
X[:50] += 2   # Class 1 centered at (2,2)
X[50:] -= 2   # Class 0 centered at (-2,-2)

# Labels
y = np.array([1]*50 + [0]*50)

# --- Train Perceptron ---
perceptron = PerceptronTrick(use_sigmoid=True)  # Change to True to use sigmoid
model = perceptron.model
model.fit(X, y)

# --- Predict ---
y_pred = model.predict(X)

# --- Plot true labels (not predictions!) ---
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=60, label='True Labels')

# --- Plot decision boundary ---
x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_vals = model.decision_boundary(x1_vals)
plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Classification (Step Function)' if not model.use_sigmoid else 'Perceptron (Sigmoid Function)')
plt.legend()
plt.grid(True)
plt.show()
