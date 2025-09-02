import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Generate synthetic data
def generate_polynomial_data(n_samples=200, noise_std=10):
    np.random.seed(42)
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = 3 * x1**2 + 2 * x2 + 5 + noise
    return x1.reshape(-1, 1), x2.reshape(-1, 1), y.reshape(-1, 1)

# Generate and preprocess data
x1, x2, y = generate_polynomial_data()
X = np.hstack([x1, x2])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")

# Create grid for surface
x1_range = np.linspace(x1.min(), x1.max(), 50)
x2_range = np.linspace(x2.min(), x2.max(), 50)
X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
X1X2_combined = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
X1X2_scaled = scaler.transform(X1X2_combined)
X1X2_poly = poly.transform(X1X2_scaled)
y_pred_grid = model.predict(X1X2_poly).reshape(X1_mesh.shape)

# Unscale test points for plotting
x1_test_unscaled, x2_test_unscaled = scaler.inverse_transform(X_test)[:, 0], scaler.inverse_transform(X_test)[:, 1]

# --- Plotly Interactive Chart ---
fig = go.Figure()

# Add surface
fig.add_trace(go.Surface(
    x=X1_mesh,
    y=X2_mesh,
    z=y_pred_grid,
    colorscale='Viridis',
    opacity=0.7,
    name='Predicted Surface',
    showscale=False
))

# Actual test points
fig.add_trace(go.Scatter3d(
    x=x1_test_unscaled,
    y=x2_test_unscaled,
    z=y_test.ravel(),
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Actual Test'
))

# Predicted test points
fig.add_trace(go.Scatter3d(
    x=x1_test_unscaled,
    y=x2_test_unscaled,
    z=y_pred.ravel(),
    mode='markers',
    marker=dict(size=5, color='blue', symbol='cross'),
    name='Predicted Test'
))

# Layout
fig.update_layout(
    title='Interactive 3D Polynomial Regression Surface',
    scene=dict(
        xaxis_title='x1',
        yaxis_title='x2',
        zaxis_title='Target y',
    ),
    legend=dict(x=0.02, y=0.98),
    margin=dict(l=0, r=0, b=0, t=30)
)

# Set Plotly renderer
import plotly.io as pio
pio.renderers.default = 'browser'  # Opens in default browser window

fig.show()

