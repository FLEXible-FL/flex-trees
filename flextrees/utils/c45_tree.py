import math
from collections import deque

import pandas as pd
import numpy as np
from scipy.stats import entropy

from dtfl.utils.utils_trees import Node

from sklearn.exceptions import NotFittedError

class C45Tree:
    """
    Class used to build a C45 decision tree for centralized learning, i.e., traditional machine learning.
    This method is called C4.5, which is an extension of ID3 algorithm, and it
    can handle both continuous and discrete attributes.
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_gain_ratio=1e-3, random_state=42, improve_speed=False, **kwargs):
        self.__max_depth = max_depth
        self.__node = None
        self.__is_fitted = False
        self.__features = None
        self.__feature_types = None
        self.__leaf_nodes = []
        self.__min_sample_split = min_samples_split
        self.__min_gain_ratio = min_gain_ratio
        self.__min_gain_improvement = 1e-8
        self.__improve_speed = improve_speed

    @property
    def is_fitted(self):
        return self.__is_fitted

    @property
    def max_depth(self):
        return self.__max_depth

    @property
    def node(self):
        return self.__node

    @property
    def min_sample_split(self):
        return self.__min_sample_split

    @property
    def leaf_nodes(self):
        return self.__leaf_nodes

    @property
    def classes_(self):
        return self.__classes

    @property
    def n_features_in_(self):
        return len(self.__feature_names)

    def __entropy(self, y, x_ids):
        """Calculates the entropy for the given set of labels.

        Args:
            y (Array-like object): List containing the labels
            x_ids (Array-like object): IDs of the remaining data

        Returns:
            float: Entropy of the node
        """
        labels = [y[id] for id in x_ids]
        total_samples = len(labels)
        label_counts = {label: labels.count(label) for label in set(labels)}  # Count for each label

        entropy = 0
        for count in label_counts.values():
            probability = count / total_samples
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def __new_entropy(self, y, x_ids, base=2):
        """Función para crear la entropía usando scipy

        Args:
            y (ArrayLike): labels
            x_ids (ArrayLike): Ids del nodo hijo a calcular la entropía
        """
        labels = [y[id] for id in x_ids]
        value, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)

    def __entropy_binary(self, y, x_ids):
        """Calculates entropy using vectorized operations."""
        labels = np.array([y[id] for id in x_ids])
        label_counts = np.bincount(labels)
        probabilities = label_counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add a small constant to avoid log(0)


    def __gain_ratio(self, X, y, x_ids, feature_id):
        """Calculates the gain ratio for a feature.

        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): List containing the remaining ids
            feature_id (int): ID of the feature to evaluate

        Returns:
            float: Gain ratio for the given feature
        """
        # breakpoint()
        # Calculate information gain for the feature
        feature_values = [X[x][feature_id] for x in x_ids]
        feature_set_values = list(set(feature_values))
        print("Antes subset")
        subsets = {val: [x for x in x_ids if X[x][feature_id] == val] for val in feature_set_values}
        print("Después subset")
        # Information gain calculation
        entropy_before = self.__entropy(y, x_ids)
        weighted_entropy_after = sum(
            (len(subset) / len(x_ids)) * self.__entropy(y, subset) for subset in subsets.values()
        )

        info_gain = entropy_before - weighted_entropy_after

        # Split information calculation__ (split entropy)
        feature_val_count = [len(subset) for subset in subsets.values()]
        split_info = -sum(
            (count / len(x_ids)) * math.log(count / len(x_ids), 2) if count > 0 else 0
            for count in feature_val_count
        )

        return info_gain / split_info if split_info > 0 else 0

    def __is_continue(self, X, feature_id):
        """
        Check is a feature is continuous or not
        Arguments:
            X {Array-like object} -- Training data
            feature_id {int} -- ID of the feature to evaluate
        Returns:
            bool -- True if the feature is continuous, False otherwise
        """
        return isinstance(X[0][feature_id], (int, float))

    def __check_feature_type(self, X, feature_id):
        """
        Check the type of a feature.
        First, it checks the feature's type in the self.__feature_types atrribute
        of the class. If it is not defined, it checks the type of the feature in the
        dataset. If you want to force the type of the feature, you can pass a list
        in the fit method with the types of the features, or in the constructor of the
        class.

        Arguments:
            X {Array-like object} -- Training data
            feature_id {int} -- ID of the feature to evaluate
        Returns:
            str -- Type of the feature. True if the feature is continuous, False otherwise
        """
        if self.__feature_types:
            if self.__feature_types[feature_id] == "continuous":
                return True
        else:
            return self.__is_continue(X, feature_id)
        return False

    def __get_majority_class(self, y, x_ids):
        freq = [0] * len(set(y))
        for i in x_ids:
            freq[y[i]] += 1
        max_freq = max(freq)
        return freq.index(max_freq)

    def __best_feature(self, X, y, x_ids, features):
        """Finds the best feature to split the data
        
        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): List containing the remaining ids
            features (List[int]): List containing the feature IDs

        Returns:
            int: ID of the best feature
            float: Information gain for the best feature
            tuple: Best split for the best feature
        """
        gains = []
        splits = []

        # Precompute sorted values for continuous features
        if not hasattr(self, 'sorted_indices'):
            self.sorted_indices = self.__precompute_sorted_continuous_features(X, features)

        # Iterate over all the features to find the best split
        for feature in features:
            info_gain, threshold = self.__information_gain(X, y, x_ids, feature)
            gains.append(info_gain)
            splits.append(threshold)
        
        # Find the feature with the maximum information gain
        best_feature_id = np.argmax(gains)
        best_gain = gains[best_feature_id]
        best_split = splits[best_feature_id]
        # breakpoint()
        # Return the best feature index, the gain, and the split criterion (threshold for continuous)
        return features[best_feature_id], best_gain, best_split

    def __get_thresholds(self, unique_values, step_size=1):
        """Returns a reduced number of thresholds based on step size."""
        return unique_values[::step_size]  # Take every `step_size`th value

    def __information_gain(self, X, y, x_ids, feature_id):
        """Calculates the information gain for a specific feature.
        
        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): List of sample indices to consider
            feature_id (int): The feature ID (column index) to calculate gain for
            
        Returns:
            float: The information gain for the given feature
        """
        # Calculate entropy before the split (parent node entropy)
        entropy_before = self.__entropy(y, x_ids)
        
        # Check if the feature is continuous
        # if self.__is_continue(X, feature_id): # Old method of checking.
        if self.__check_feature_type(X, feature_id):
            # For continuous features, split based on a threshold
            best_gain = 0
            best_threshold = None
            # breakpoint()
            feature_values = [(X[x][feature_id], y[x]) for x in x_ids]
            feature_values.sort(key=lambda x: x[0])
            feature_values = np.unique(feature_values, axis=0)
            # breakpoint()
            # feature_values = sorted(set(feature_values))
            previous_gain = -np.inf
            feature_values = self.__get_thresholds(unique_values=feature_values, step_size=1)
            # gain_ratio_before = self.__gain_ratio(X, y, x_ids, feature_id)
            # breakpoint()
            #  self.__gain_ratio(self, X, y, x_ids, feature_id)
            for i in range(len(feature_values) - 1):
                if feature_values[i][1] != feature_values[i + 1][1]:
                    # Compute threshold
                    #         threshold = (feature_values[i][0] + feature_values[i + 1][0]) / 2
                    #         x_left = [x for x in x_ids if X[x][feature_id] <= threshold]
                    #         x_right = [x for x in x_ids if X[x][feature_id] > threshold]
                    #         gain_ratio_left = self.__gain_ratio(X, y, x_left, feature_id)
                    #         gain_ratio_right = self.__gain_ratio(X, y, x_right, feature_id)
                            
                    #         gain_ratio_after = gain_ratio_left + gain_ratio_right
                    #         gain_ratio = gain_ratio_before - gain_ratio_after
                    #         if gain_ratio - previous_gain < self.__min_gain_ratio:
                    #             break
                    #         if gain_ratio > best_gain:
                    #             best_gain = gain_ratio
                    #             best_threshold = threshold
                    #         if self.__improve_speed:
                    #             previous_gain = gain_ratio
                    # return best_gain, best_threshold
                    threshold = (feature_values[i][0] + feature_values[i + 1][0]) / 2
                    # breakpoint()
                    # Split data based on threshold
                    x_left = [x for x in x_ids if X[x][feature_id] <= threshold]
                    # y_left = [y[x] for x in x_left]
                    x_right = [x for x in x_ids if X[x][feature_id] > threshold]
                    # y_right = [y[x] for x in x_right]
                    
                    # Calculate the weighted entropy after the split
                    weight_left = len(x_left) / len(x_ids)
                    weight_right = len(x_right) / len(x_ids)
                    
                    # entropy_left = self.__entropy(y, x_left)
                    entropy_left = self.__entropy(y, x_left)
                    entropy_right = self.__entropy(y, x_right)
                    # entropy_right = self.__entropy(y, x_right)
                    
                    entropy_after = (weight_left * entropy_left) + (weight_right * entropy_right)
                    
                    # Calculate information gain
                    info_gain = entropy_before - entropy_after

                    if info_gain - previous_gain < self.__min_gain_improvement:
                        break
                    
                    if info_gain > best_gain:
                        # breakpoint()
                        best_gain = info_gain
                        best_threshold = threshold

                    if self.__improve_speed:
                        previous_gain = info_gain
            return best_gain, best_threshold
        else:
            # For discrete features, split based on unique values
            feature_values = [X[x][feature_id] for x in x_ids]
            unique_values = set(feature_values)
            
            # Calculate entropy for each subset
            entropy_after = 0
            for value in unique_values:
                subset_x_ids = [x for x in x_ids if X[x][feature_id] == value]
                weight = len(subset_x_ids) / len(x_ids)
                subset_entropy = self.__entropy(y, subset_x_ids)
                entropy_after += weight * subset_entropy
            
            # Information gain is the reduction in entropy
            info_gain = entropy_before - entropy_after
            
            return info_gain, unique_values  # No threshold for discrete features

    def fit(self, X, y, feature_names=None, feature_types=None):
        x_ids = list(range(len(X)))
        features_ids = list(np.arange(X.shape[1]))
        self.__classes = list(set(y))
        self.__feature_names = feature_names if feature_names else list(range(X.shape[1]))
        if feature_types:
            self.__feature_types = feature_types
        print(self.__feature_types)
        self.__node = self.__build_tree(self.__node, X, y, x_ids, features_ids, 0)
        # self.__build_tree(self.__node, X, y, x_ids, features_ids, 1)
        self.__is_fitted = True

    def __build_tree(self, node, X, y, x_ids, features, depth):
        """Builds the decision tree
        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): List containing the remaining ids
            features (List[int]): List containing the feature IDs
            depth (int): Current depth of the tree

        Returns:
            Node: Root node of the decision tree
        """
        if node is None:
            node = Node()
        
        node.depth = depth
        node.available_features = features
        labels_in_feature = [y[i] for i in x_ids]
        node.n_node_samples = len(x_ids)
        node.n_classes = len(set(labels_in_feature))
        node.label = max(set(labels_in_feature), key=labels_in_feature.count)
        node.class_idx = np.array(
            list(
                {
                    c: list(labels_in_feature).count(c)
                    if c in set(labels_in_feature)
                    else 0
                    for c in self.__classes
                }.values()
            )
        )
        # Case 1: If the node contains only one class, make it a leaf
        if node.n_classes == 1:
            node.value = labels_in_feature[0]
            node.is_leaf = True
            self.__leaf_nodes.append(node)
            return node

        # Case 2: Stopping criteria - max depth, no features left, or min samples
        if len(features) == 0 or depth >= self.__max_depth:
            return self.__build_leaf_node(node, labels_in_feature, depth)

        # Find the best feature and split
        best_feature_id, best_gain_ratio, best_split = self.__best_feature(X, y, x_ids, features)
        
        # If no meaningful split can be found, make a leaf
        if best_feature_id is None or best_gain_ratio == 0:
            return self.__build_leaf_node(node, labels_in_feature, depth)

        if len(x_ids) < self.min_sample_split or best_gain_ratio < self.__min_gain_ratio:
            return self.__build_leaf_node(node, labels_in_feature, depth)

        # Set feature info for the current node
        node.feature = best_feature_id
        node.criterion_value = best_gain_ratio
        # breakpoint()
        new_available_features = list(node.available_features)
        new_available_features.remove(best_feature_id)

        if self.__check_feature_type(X, best_feature_id):  # Handle continuous features
            node.feature_type = "continuous"
            node.threshold = best_split
            
            child_x_ids_left = [x for x in x_ids if X[x][best_feature_id] <= node.threshold]
            child_x_ids_right = [x for x in x_ids if X[x][best_feature_id] > node.threshold]
            
            # Pre-pruning if split is not informative
            if len(child_x_ids_left) == 0 or len(child_x_ids_right) == 0:
                return self.__build_leaf_node(node, labels_in_feature, depth)
            
            node.left = Node()
            node.left.dad = node
            self.__build_tree(node.left, X, y, child_x_ids_left, new_available_features, depth + 1)
            
            node.right = Node()
            node.right.dad = node
            self.__build_tree(node.right, X, y, child_x_ids_right, new_available_features, depth + 1)
        
        else:  # Handle discrete features
            node.feature_type = "discrete"
            node.childs = []
            # breakpoint()
            for value in best_split:
                child_x_ids = [x for x in x_ids if X[x][best_feature_id] == value]
                if len(child_x_ids) == 0:
                    continue  # Skip empty splits
                child = Node()
                child.dad = node
                child.value = value
                child.feature_type = "discrete"
                child.next = Node()
                child.next.dad = child
                node.childs.append(child)
                self.__build_tree(child.next, X, y, child_x_ids, new_available_features, depth + 1)

        return node

    def predict(self, X):
        """Predicts the class for a given instance
        """	
        if not self.is_fitted:
            raise NotFittedError("Model not fitted")
        return np.array([self.__predict_row(self.__node, x) for x in X])

    def __build_leaf_node(self, node, labels_in_feature, depth):
        """Builds a leaf node if it is not possible to split the data

        Args:
            node (Node): Node to build
            labels_in_feature (Array-like object): List containing the labels
            depth (int): Actual depth of the tree

        Returns:
            _type_: _description_
        """
        node.value = max(set(labels_in_feature), key=labels_in_feature.count)
        node.is_leaf = True
        node.depth = depth
        self.__leaf_nodes.append(node)
        return node

    def __precompute_sorted_continuous_features(self, X, features):
        """Precomputes sorted indices for continuous features."""
        sorted_indices = {}
        for feature in features:
            if self.__is_continue(X, feature):
                sorted_indices[feature] = np.argsort([X[x][feature] for x in range(len(X))])
        return sorted_indices


    def __predict_row(self, node, row):
        """Predicts the class for a given instance
        """
        # breakpoint()
        if node.is_leaf:
            return node.value
        if node.feature_type == "continuous":
            # breakpoint()
            if row[node.feature] <= node.threshold:
                return self.__predict_row(node.left, row)
            else:
                return self.__predict_row(node.right, row)
        else:
            for child in node.childs:
                if row[node.feature] == child.value:
                    return self.__predict_row(child.next, row)
            return node.label

    def reach_root_node(self, node):
        """Returns the root node of the tree."""
        stack = deque()
        # breakpoint()
        last_node = None
        while node.dad:
            if node.is_leaf:
                stack.append([-1, node.value, 'label', 'leaf'])
                last_node = node
            else:
                if node.feature_type == "continuous":
                    # breakpoint()
                    # try:
                    if node.left and node.left == last_node:
                        stack.append([node.feature, node.threshold, 'continuous', 'left'])
                    elif node.right and node.right == last_node:
                        stack.append([node.feature, node.threshold, 'continuous', 'right'])
                    # if node.dad.left and node.dad.left == node:
                    #     stack.append([node.feature, node.threshold, 'continuous', 'left'])
                    # elif node.dad.right and node.dad.right == node:
                    #     stack.append([node.feature, node.threshold, 'continuous', 'right'])
                    #except:
                        # breakpoint()
                    # stack.append([node.feature, node.threshold, 'continuous', ])
                    last_node = node
                else: # Feature is discrete
                    value = node.value
                    node = node.dad
                    feature = node.feature
                    stack.append([feature, value, 'discrete', 'child'])
            if node.dad:
                node = node.dad
        # Root node now
        # breakpoint()
        if node.feature_type == "continuous":
            # stack.append([node.feature, node.threshold, 'continuous', 'root'])
            if node.left == last_node:
                stack.append([node.feature, node.threshold, 'continuous', 'left'])
            else:
                stack.append([node.feature, node.threshold, 'continuous', 'right'])
        # Reverse the stack to get the path from the root to the leaf
        stack.reverse()
        # print(stack)
        return stack

    def decision_path(self, X_test):
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet.")
        return [self.__get_decision_path(self.__node, row) for row in X_test]

    def __get_decision_path(self, node, row):
        if not node.childs and not node.left and not node.right:
            return "."
        if node.feature_type == "continuous":
            # print(self.__feature_names)
            index = self.__feature_names.index(node.feature)
            if row[index] <= node.threshold:
                return f"x{node.feature} <= {node.threshold}, {self.__get_decision_path(node.left, row)}"
            else:
                return f"x{node.feature} > {node.threshold}, {self.__get_decision_path(node.right, row)}"
        else:
            index = self.__feature_names.index(node.feature)
            value = row[index]
            for child in node.childs:
                if child.next and value == child.value:
                    next_child = child.next
                    return f"x{node.feature} = {child.value}, {self.__get_decision_path(next_child, row)}"

    def print_tree(self):
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet.")
        self.__print_tree(self.__node)

    def __print_tree(self, node, depth=0):
        if node.is_leaf:
            print(f"{depth * '  '}Leaf node: {node.value}")
        else:
            if node.feature_type == "continuous":
                print(f"{depth * '  '}x{node.feature} <= {node.threshold}")
                self.__print_tree(node.left, depth + 1)
                print(f"{depth * '  '}x{node.feature} > {node.threshold}")
                self.__print_tree(node.right, depth + 1)
            else:
                print(f"{depth * '  '}x{node.feature}")
                for child in node.childs:
                    print(f"{depth * '  '}  x{node.feature} = {child.value}")
                    self.__print_tree(child.next, depth + 1)

"""
if __name__ == "__main__":
    import time
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    # Load the Iris dataset
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    X = df.drop(columns="species").to_numpy()
    y = df["species"]
    # Transform the labels to integers
    y, labels_names = pd.factorize(y)[0], pd.Series(pd.factorize(y)[1])
    nursery = pd.read_csv('../../nursery.csv')
    X_columns = nursery.columns
    y = nursery["label"].to_numpy()
    X = nursery.drop(columns="label", axis=1).to_numpy()
    path_to_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    x_columns = ['x' + str(i) for i in range(14)]
    y_column = 'label'
    train_data = pd.read_csv(path_to_train, names=x_columns + [y_column])
    y = train_data.apply(lambda row: 1 if '>50K' in row['label'] else 0, axis=1).to_numpy()
    X = train_data.drop(columns=[y_column], axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a C45 decision tree
    tree = C45Tree(max_depth=5, improve_speed=True)
    # tree.fit(X.values, y.values)
    print("Fitting the model")
    start_time = time.time()
    tree.fit(X_train, y_train, feature_names=list(X_columns))
    elapsed_time = time.time() - start_time
    print("Model fitted:", tree.is_fitted)
    print("Max depth:", tree.max_depth)
    print("Elapsed time:", elapsed_time)
    print("Predict on test data")
    y_pred = tree.predict(X_test)
    # print("Accuracy:", accuracy_score([y_test.values[0]], y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred, average="macro"))
"""