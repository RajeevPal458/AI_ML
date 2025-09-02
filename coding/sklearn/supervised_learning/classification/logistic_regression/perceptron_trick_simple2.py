import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def step_function(z):
    return np.where(z >= 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class PerceptronTrick:
    class Model:
        def __init__(self, num_features, use_sigmoid=False):
            self.W = np.zeros(num_features + 1)  # +1 for bias term
            self.use_sigmoid = use_sigmoid

        def predict(self, X):
            X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
            z = np.dot(X_bias, self.W)
            return sigmoid(z) if self.use_sigmoid else step_function(z)
        
        def train(self, X, y, lr=0.1, epochs=1000):
            flag = True
            X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
            print(f"X_bias - {X_bias.shape}  , self.W : {self.W.shape}")
            for epoch in range(epochs):
                for xi, yi in zip(X_bias, y):
                    z = np.dot(xi, self.W)
                    if flag :
                        print(f"z - {z.size} , type {type(z)}")
                    pred = sigmoid(z) if self.use_sigmoid else step_function(z)
                    error = yi - pred
                    
                    self.W += lr * error * xi
                    
                    if flag :
                        print(f"yi - {yi} , type {type(yi)}")
                        print(f"error - {error} , type {type(error)}")
                        print(f"W - {self.W} , type {type(self.W)}")
                        flag = False

        def decision_boundary(self, X):
            # W0 + W1*x1 + W2*x2 = 0 → x2 = -(W0 + W1*x1)/W2
            x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
            x2_vals = -(self.W[0] + self.W[1] * x1_vals) / self.W[2]
            return x1_vals, x2_vals

# --- Generate Sparse & Noisy Data ---
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           class_sep=2, flip_y=0.1, random_state=42)

# --- Train Model ---
model_step = PerceptronTrick.Model(num_features=2, use_sigmoid=False)  # Change to True to use sigmoid
model_step.train(X, y, lr=0.1, epochs=1000)

model_sigmoid = PerceptronTrick.Model(num_features=2, use_sigmoid=True)  # Change to True to use sigmoid
model_sigmoid.train(X, y, lr=0.1, epochs=1000)

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=60, alpha=0.7)

x1_s, x2_s = model_sigmoid.decision_boundary(X)
x1_st, x2_st = model_step.decision_boundary(X)

plt.plot(x1_s, x2_s, 'g--', label='Sigmoid Boundary')
plt.plot(x1_st, x2_st, 'b-.', label='Step Function Boundary')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Classifier ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

