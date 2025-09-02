import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only first two features (for 2D visualization)
y = iris.target

# Binary classification: Setosa vs Versicolor
binary_mask = y < 2
X = X[binary_mask]
y = y[binary_mask]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction & evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot data points and decision boundary line
def plot_decision_boundary_line(model, X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=60)

    # Extract weights and intercept
    w1, w2 = model.coef_[0]
    b = model.intercept_[0]

    # Calculate x2 (y-axis) for two points along x1 (x-axis)
    x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_vals = -(b + w1 * x1_vals) / w2

    plt.plot(x1_vals, x2_vals, color='green', linestyle='--', label='Decision Boundary')

    plt.title(title)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot
plot_decision_boundary_line(model, X, y, title="Logistic Regression (Line Plot - Iris Setosa vs Versicolor)")
