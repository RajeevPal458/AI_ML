import numpy as np
from sklearn.datasets import load_iris
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from dtreeviz.trees import dtreeviz

class DecisionTreeClassifierCustom:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _gini(self, y):
        counts = Counter(y)
        impurity = 1.0
        for lbl in counts:
            prob_of_lbl = counts[lbl] / len(y)
            impurity -= prob_of_lbl ** 2
        return impurity

    def _best_split(self, X, y):
        best_gain = -1
        best_feat, best_thresh = None, None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature_index
                    best_thresh = threshold
        return best_feat, best_thresh

    def _information_gain(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        return self._gini(parent) - (weight_left * self._gini(left) + weight_right * self._gini(right))

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return self.Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return self.Node(value=leaf_value)

        left_idxs = X[:, feature] <= threshold
        right_idxs = ~left_idxs

        left_node = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_node = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(feature, threshold, left_node, right_node)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# ---- Testing the Custom Classifier ----
if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifierCustom(max_depth=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("Predicted:", predictions)
    print("Actual:   ", y_test)
    
    print(f"y_test : {y_test.shape}  , predictions : {predictions.shape} ")
  
    print("Accuracy:", accuracy_score(y_test, predictions))
    
    
    # 4. Visualize with dtreeviz
    # viz = dtreeviz(
    #     regressor,
    #     X_train,
    #     y_train,
    #     target_name="Target",
    #     feature_names=["Feature1"],
    #     title="Decision Tree Regression",
    #     show_node_labels=True
    # )

    # # 5. Display
    # viz.show()
    



