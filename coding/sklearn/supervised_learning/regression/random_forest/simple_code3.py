# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

# # Generate synthetic nonlinear data
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# # Test data for prediction
# X_test = np.linspace(0, 5, 500).reshape(-1, 1)

# # Models
# dt = DecisionTreeRegressor(max_depth=3)
# br = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, random_state=1)
# rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=1)

# # Fit models
# dt.fit(X, y)
# br.fit(X, y)
# rf.fit(X, y)

# # Predict
# y_dt = dt.predict(X_test)
# y_br = br.predict(X_test)
# y_rf = rf.predict(X_test)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, c='gray', label='Training Data', s=20)
# plt.plot(X_test, y_dt, label='Decision Tree', color='blue', linewidth=2)
# plt.plot(X_test, y_br, label='Bagging Regressor', color='#00ff00', linewidth=6)
# plt.plot(X_test, y_rf, label='Random Forest', color='red', linewidth=2)
# plt.title("Comparison of Regression Models")
# plt.xlabel("X")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a larger, realistic regression dataset
np.random.seed(42)
X = np.sort(5 * np.random.rand(500, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# Create test set for smoother prediction curve
X_test = np.linspace(0, 5, 1000).reshape(-1, 1)

# Define models
dt = DecisionTreeRegressor(max_depth=5)
bagging = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=5), n_estimators=50, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Fit models
dt.fit(X, y)
bagging.fit(X, y)
rf.fit(X, y)

# Predict
y_pred_dt = dt.predict(X_test)
y_pred_bag = bagging.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(X, y, c='lightgrey', label='Training Data', s=10)
plt.plot(X_test, y_pred_dt, label='Decision Tree', color='blue', linewidth=2)
plt.plot(X_test, y_pred_bag, label='Bagging Regressor', color='green', linewidth=4)
plt.plot(X_test, y_pred_rf, label='Random Forest', color='red', linewidth=2)
plt.title("Comparison of Regression Models")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
