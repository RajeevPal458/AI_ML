import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Generate Sample Data
np.random.seed(42)
X = np.random.rand(100, 2) * 10  # 100 samples, 2 features
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(100) * 0.1  # non-linear target

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Fit RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Predict on a Grid for 3D Plot
x1 = np.linspace(0, 10, 50)
x2 = np.linspace(0, 10, 50)
x1_grid, x2_grid = np.meshgrid(x1, x2)
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
y_pred_grid = model.predict(X_grid).reshape(x1_grid.shape)

# 5. 3D Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, cmap='viridis', alpha=0.8)
ax.scatter(X[:, 0], X[:, 1], y, color='r', s=20, label='Data points')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Random Forest Regression Surface')
plt.legend()
plt.tight_layout()
plt.show()
