import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

# Load the iris dataset
iris = load_iris()
X = iris.data  # 4 features
y = iris.target  # 3 classes

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensions to 2 for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train logistic regression
model = LogisticRegression(multi_class='ovr', solver='lbfgs')
model.fit(X_pca, y)

# Plot decision boundaries
disp = DecisionBoundaryDisplay.from_estimator(
    model,
    X_pca,
    response_method="predict",
    alpha=0.3,
    cmap=plt.cm.coolwarm
)

# Plot the data points
for i, target_name in enumerate(iris.target_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=target_name,
        edgecolor='k'
    )

plt.title("Logistic Regression with Iris Dataset\n(4 features reduced to 2D using PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
