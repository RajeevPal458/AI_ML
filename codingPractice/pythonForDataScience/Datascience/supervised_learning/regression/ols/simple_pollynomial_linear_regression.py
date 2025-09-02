import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic 1D data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X**3 - X**2 + X + 5 + np.random.randn(100, 1) * 20

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Polynomial transformation
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Fit linear regression on polynomial features
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict
y_pred = model.predict(X_poly_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE (1D polynomial): {mse:.2f}")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label='Actual', color='blue')
plt.scatter(X_test, y_pred, label='Predicted', color='red')
plt.title("Polynomial Regression (1 Feature, Degree=3)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
