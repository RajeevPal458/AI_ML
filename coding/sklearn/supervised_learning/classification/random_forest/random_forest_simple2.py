import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Create 2D classification dataset with 3 classes
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_informative=2, n_classes=3, n_clusters_per_class=1,
                           random_state=42)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate
print("Classification Report:\n", classification_report(y_test, clf.predict(X_test)))

print(f"X shape : {X.shape}")
# Step 5: Visualize decision boundaries
# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, X.shape[0]),
                     np.linspace(y_min, y_max, X.shape[0]))

#xx,yy = np.meshgrid(X[:,0],X[:,1])
print(f"xx shape : {xx.shape} , yy shape : {yy.shape}")
# Predict over mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print(f"z shape : {Z.shape}")
# Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='Set1')
plt.title("Random Forest Decision Boundaries (3 Classes)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
