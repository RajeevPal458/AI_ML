import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate synthetic data (200 samples, 2 features)
def generate_polynomial_data(n_samples=200, noise_std=10):
    np.random.seed(42)
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = 3 * x1**2 + 2 * x2 + 5 + noise  # True polynomial with noise
    return x1.reshape(-1, 1), x2.reshape(-1, 1), y.reshape(-1, 1)

# Generate data
x1, x2, y = generate_polynomial_data()
X = np.hstack([x1, x2])  # Combine features

# Step 2: Preprocess and train-test split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Step 5: Predict
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print("Learned coefficients:", model.coef_)
print("Polynomial feature names:", poly.get_feature_names_out(['x1', 'x2']))

# Step 6: Plot predicted surface and actual points
# Build meshgrid to visualize surface
x1_range = np.linspace(x1.min(), x1.max(), 50)
x2_range = np.linspace(x2.min(), x2.max(), 50)
X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
X1X2_combined = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
X1X2_scaled = scaler.transform(X1X2_combined)
X1X2_poly = poly.transform(X1X2_scaled)
y_pred_grid = model.predict(X1X2_poly).reshape(X1_mesh.shape)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the regression surface
ax.plot_surface(X1_mesh, X2_mesh, y_pred_grid, alpha=0.6, cmap='viridis', edgecolor='none', label='Predicted Surface')

# Overlay actual test data points
x1_test_unscaled, x2_test_unscaled = scaler.inverse_transform(X_test)[:, 0], scaler.inverse_transform(X_test)[:, 1]
ax.scatter(x1_test_unscaled, x2_test_unscaled, y_test, color='red', label='Actual Test', s=40)
ax.scatter(x1_test_unscaled, x2_test_unscaled, y_pred, color='blue', label='Predicted Test', s=40, marker='x')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Polynomial Regression Surface with Actual vs Predicted Points')
plt.tight_layout()
plt.show()
