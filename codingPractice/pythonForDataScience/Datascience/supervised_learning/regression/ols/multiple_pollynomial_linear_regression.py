import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
diabetes = load_diabetes()
X_full = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Select 3 features for polynomial regression
selected_features = ['bmi', 'bp', 's1']
X = X_full[selected_features].copy()

# Keep unscaled X for plotting
X_unscaled = X.copy()

# Standardize features for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (fix unpacking 6 variables)
X_train_scaled, X_test_scaled, y_train, y_test, X_train_unscaled, X_test_unscaled = train_test_split(
    X_scaled, y, X_unscaled, test_size=0.2, random_state=42
)

# Polynomial transformation (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train_scaled)
X_poly_test = poly.transform(X_test_scaled)

# Train model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_poly_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nTest MSE: {mse:.2f}")

# --- Predict and draw surface (for visualization) ---
# Create grid for bmi and bp, fix s1 to its mean
bmi_vals = np.linspace(X_unscaled['bmi'].min(), X_unscaled['bmi'].max(), 50)
bp_vals = np.linspace(X_unscaled['bp'].min(), X_unscaled['bp'].max(), 50)
bmi_grid, bp_grid = np.meshgrid(bmi_vals, bp_vals)

# Fix s1 to mean for visualization
s1_fixed = X_unscaled['s1'].mean()
s1_grid = np.full_like(bmi_grid, s1_fixed)

# Flatten and build [bmi, bp, s1] input matrix
X_grid = np.stack([bmi_grid.ravel(), bp_grid.ravel(), s1_grid.ravel()], axis=1)

# Standardize the grid input using same scaler
X_grid_scaled = scaler.transform(X_grid)

# Generate polynomial features and predict
X_grid_poly = poly.transform(X_grid_scaled)
y_grid_pred = model.predict(X_grid_poly)
y_grid_pred = y_grid_pred.reshape(bmi_grid.shape)

# --- Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot prediction surface
ax.plot_surface(bmi_grid, bp_grid, y_grid_pred, cmap='viridis', alpha=0.6, edgecolor='none')

# Overlay actual test points
ax.scatter(X_test_unscaled['bmi'], X_test_unscaled['bp'], y_test, color='red', label='Actual Test', s=30)
ax.scatter(X_test_unscaled['bmi'], X_test_unscaled['bp'], y_pred, color='blue', label='Predicted Test', s=30)

ax.set_xlabel('BMI')
ax.set_ylabel('BP')
ax.set_zlabel('Target')
ax.set_title('Polynomial Regression Surface (3D) with Test Data')
ax.legend()
plt.tight_layout()
plt.show()
