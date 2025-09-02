import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Generate 1D regression dataset
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # Sine function with noise

# Fit Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# Predict over a smooth input space
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='blue', label='Training data', s=20)
plt.plot(X_test, y_pred, color='red', label='Random Forest Prediction', linewidth=2)
plt.title("Random Forest Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
