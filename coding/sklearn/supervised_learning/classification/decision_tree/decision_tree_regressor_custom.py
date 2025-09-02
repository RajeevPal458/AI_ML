import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress deprecation warning (load_boston is deprecated in newer sklearn versions)
import warnings
warnings.filterwarnings("ignore")

# Load dataset (for simplicity, use only a few features)
data = load_boston()
X = data.data[:, :3]  # Use only 3 features for clarity
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Decision Tree Regressor (basic version)
class SimpleDecisionTreeRegressor:
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  # Leaf value

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            return self.Node(value=np.mean(y))

        best_feature, best_thresh, best_mse = None, None, float('inf')
        best_left_idx, best_right_idx = None, None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idx = X[:, feature_index] <= threshold
                right_idx = ~left_idx

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                mse = self._calculate_mse(y[left_idx], y[right_idx])

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_thresh = threshold
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        if best_feature is None:
            return self.Node(value=np.mean(y))

        left_subtree = self._build_tree(X[best_left_idx], y[best_left_idx], depth + 1)
        right_subtree = self._build_tree(X[best_right_idx], y[best_right_idx], depth + 1)
        return self.Node(feature_index=best_feature, threshold=best_thresh,
                         left=left_subtree, right=right_subtree)

    def _calculate_mse(self, left_y, right_y):
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        return (left_mse + right_mse) / (len(left_y) + len(right_y))

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

# Train custom regressor
tree = SimpleDecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

# Predict
predictions = tree.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
predictions[:5], y_test[:5], mse
