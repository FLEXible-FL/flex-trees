import time
from collections import deque
import numpy as np
from numpy.lib import math
from sklearn.exceptions import NotFittedError


class Node:
    """Basic node for the ID3/CART model.
    # Arguments:
        value: value of the node. If the node is leaf, then the value is the class predicted.
        Otherwise, the value indicates the feature selected in the node.
        childs: List of possible childs of the node.
        next: Next node.
        depth: Depth of the node.
        feature_type: feature type of the value, if the node is not leaf. Here we differenciate
        categorical features than numerical/discrete features.
    """

    def __init__(self):
        self.value = None
        self.childs = None
        self.next = None
        self.depth = None
        # New data for adding more information
        # Feature type for deal difference continuous vars from others.
        self.feature_type = None
        # Dad of the node. Root won't have one
        self.dad = None
        self.left = None
        self.right = None
        self.available_features = None
        # criterion value
        self.criterion_value = None # Used in ID3Classifier
        self.value_proba = None
        # To simulate sklearn atributes
        self.n_classes = None
        self.n_node_samples = None
        self.class_idx = None
        self.is_leave = False
        # Updating ID3 -> Use really one node insted of two when adding a node at the tree.
        self.label = None
        self.feature = None

class ID3:
    """Class containing the tree structure of the ID3 model.

    Actually, this model only supports categorical data, as we have used it only with Nursery.
    In the future it will support discrete and continuous data too.
    
    This model is the one used in the Federated ID3 method.

    # Arguments:
        max_depth: Maximum depth of the tree
        node: root node
        feature_names: Names of the features used in the training stage
        feature_types: Types of the features used in the training stage
    """

    def __init__(self, max_depth, feature_names):
        self._max_depth = max_depth
        self._node = None
        self.feature_names = feature_names
        self.feature_types = None
        # Variable for aggregating params
        self._aux_params = None

    @property
    def max_depth_(self):
        """Property to get the maximum depth of the tree
        """
        return self._max_depth

    @property
    def tree(self):
        """Property that returns the root from the tree
        """
        return self._node

    def set_root(self, node):
        """Set the root node for the ID3 tree
        """
        self._node = node

    def predict(self, data):
        """Runs the predict for this method. It calls recursively the predict_node function.
        Arguments:
            data: test data to predict
        Returns:
            np array with the predictions.
        """

        if not self._node:
            raise NotFittedError('No se ha entrenado el modelo.')
        return np.array([self._predict_node(self._node, row) for row in data])

    def _predict_node(self, node, row):
        """Predict the row if node is leaf, else keep moving up the tree
        """
        if not node.childs:
            return node.value
        index = self.feature_names.index(node.value)
        value = row[index]
        for child in node.childs:
            if child.next and value == child.value:
                next_child = child.next
                return self._predict_node(next_child, row)

    def print_tree_(self):
        """Function to print an ID3 tree. This functions add depth times a '\t'.
        """
        if not self._node:
            return
        nodes = deque()
        nodes.append(self._node)
        while len(nodes) > 0:
            node = nodes.popleft()
            # print(f'The value of the node is: {node.value}')
            # print(f'This node has depth: {node.depth}')
            if node:
                times = '\t'*node.depth
                print(times, node.value, ' parent: ', node.dad.dad.value) if (node.dad and node) else print(times, node.value)
                if node.childs:
                    for child in node.childs:
                        nodes.append(child.next)

class CART:
    """Class containing the tree structure of the CART model.
    
    This model is used in the Federated Extra-Trees method, where it builds
    multiple CARTs.

    # Arguments:
        node: root node
        feature_names: Names of the features used in the training stage
    """
    def __init__(self, feature_names):
        self._node = None
        self.feature_names = feature_names

    @property
    def tree(self):
        """Property that returns the root from the tree
        """
        return self._node

    def set_root(self, node):
        """Set the root node for the ID3 tree
        """
        self._node = node

    def predict(self, data):
        """Runs the predict for this method. It calls recursively the predict_node function.
        Arguments:
            data: test data to predict
        Returns:
            np array with the predictions.
        """

        if not self._node:
            raise NotFittedError('No se ha entrenado el modelo.')
        return np.array([self._predict_node(self._node, row) for row in data])

    def _predict_node(self, node, row):  # sourcery skip: merge-else-if-into-elif
        """Predict the row if node is leaf, else keep moving up the tree
        Args:
            node (Node): Actual node that will be tested.
            row (np.array): row to be predicted
        Returns:
            Node or Value: If node is leaf returns prediction else returns the next 
            child based on the feature.
        """
        if not node.childs:
            return node.value
        index = self.feature_names.index(node.value)
        test_value = row[index]
        for child in node.childs:
            sign, value = child.value.split(',')
            value = float(value)            
            if child.next and ((sign == '<=' and test_value <= value) or sign == '>'):
                return self._predict_node(child.next, row)

    def print_tree_(self):
        """Function to print a Cart tree. This functions add depth times a '\t'. 
        """
        if not self._node:
            return
        nodes = deque()
        nodes.append(self._node)
        while len(nodes) > 0:
            node = nodes.popleft()
            times = '\t'*node.depth
            print(times, node.value, ' parent: ', node.dad.dad.value) if node.dad else print(times, node.value)
            if node.childs:
                for child in node.childs:
                    nodes.append(child.next)


class TreeBoosting:
    """
    Class used to build a CART deccision tree model with the boosting method. This method
    is called at level client at the Federated Gradient Boosting Decision Trees.
    """
    def __init__(self, subsample_cols=0.8, min_leaf=5, min_child_weight=1,
                max_deph=8, lambda_=1, gamma=1, eps=0.1):
        self.max_depth = max_deph
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.row_count = None
        self.col_count = None
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.val = None
        self.score = float('-inf')
        self.root = None
        self.split_value = None
        self.depth = -1
        self.var_id = None

    def compute_gamma(self, gradient, hessian):
        '''
        Calculates the optimal leaf value equation (5) in "XGBoost: A Scalable Tree Boosting System"
        '''
        return -np.sum(gradient) / (np.sum(hessian) + self.lambda_)

    def gain(self, lhs, rhs, gradient, hessian, x_ids):
        """Calculates the gain at a particular split point.
        Args:
            lhs (list): index of the left child
            rhs (list): index of the right child
            gradient (np.array): array with the gradients of the instances
            hessian (np.array): array with the hessians of the instances
            x_ids (list): ids of the instances in this node
        Returns:
            gain (float): Gain index for the node.
        """
        gradient_ = gradient[x_ids]
        hessian_ = hessian[x_ids]

        lhs_gradient = gradient_[lhs].sum()
        lhs_hessian = hessian_[lhs].sum()

        rhs_gradient = gradient_[rhs].sum()
        rhs_hessian = hessian_[rhs].sum()

        gain = 0.5 * (
            (lhs_gradient**2/(lhs_hessian+self.lambda_)) +
            (rhs_gradient**2/(rhs_hessian+self.lambda_)) -
            (
                    (lhs_gradient + rhs_gradient)**2/(lhs_hessian + rhs_hessian + self.lambda_)
            )
        ) - self.gamma

        return gain

    def fit(self, x, gradient, hessian, x_ids, depth=1):
        self.col_count = x.shape[1]
        self.row_count = len(x_ids)
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols*self.col_count)]
        self.val = self.compute_gamma(gradient[x_ids], hessian[x_ids])
        self.depth=depth
        # self.root = self.split(x, gradient, hessian, x_ids)
        self.split(x, gradient, hessian, x_ids)
        return self

    def find_greedy_split(self, x, x_ids, var_id, gradient, hessian):
        x_ = x[x_ids,var_id]
        
        for r in range(self.row_count):
            lhs = x_ <= x_[r]
            rhs = x_ > x_[r]

            lhs_indices = np.nonzero(x_ <= x_[r])[0]
            rhs_indices = np.nonzero(x_ > x_[r])[0]
            if lhs.sum() < self.min_leaf or rhs.sum() < self.min_leaf or hessian[lhs_indices].sum() < self.min_child_weight or hessian[rhs_indices].sum() < self.min_child_weight:
                continue

            curr_score = self.gain(lhs=lhs, rhs=rhs, gradient=gradient, hessian=hessian, x_ids=x_ids)
            if curr_score > self.score:
                self.var_id = var_id
                self.score = curr_score
                self.split_value = x_[r]

    def find_greedy_split_improved(self, x, x_ids, var_id, gradient, hessian):
        x_ = x[x_ids,var_id]

        # Get unique values for the actual node
        values = np.unique(x_)
        
        for val in values:
            lhs = x_ <= val
            rhs = x_ > val
            
            lhs_indices = np.nonzero(x_ <= val)[0]
            rhs_indices = np.nonzero(x_ > val)[0]
            if lhs.sum() < self.min_leaf or rhs.sum() < self.min_leaf or hessian[lhs_indices].sum() < self.min_child_weight or hessian[rhs_indices].sum() < self.min_child_weight:
                continue

            curr_score = self.gain(lhs=lhs, rhs=rhs, gradient=gradient, hessian=hessian, x_ids=x_ids)
            if curr_score > self.score:
                self.var_id = var_id
                self.score = curr_score
                self.split_value = val

    def split(self, x, gradient, hessian, x_ids): # , depth=1):
        """Builds the decision tree
        """
        for c in self.column_subsample: self.find_greedy_split_improved(x=x, x_ids=x_ids, var_id=c, gradient=gradient,
                                                                        hessian=hessian)
        if self.is_leaf: return
        x_ = self.split_col(x, x_ids)

        
        lhs = np.nonzero(x_ <= self.split_value)[0]
        rhs = np.nonzero(x_ > self.split_value)[0]
        self.lhs = TreeBoosting(subsample_cols=self.subsample_cols, min_leaf=self.min_leaf, min_child_weight=self.min_child_weight,
                                max_deph=self.max_depth, lambda_=self.lambda_, gamma=self.gamma,
                                eps=self.eps).fit(x=x, gradient=gradient, hessian=hessian, x_ids=x_ids[lhs], depth=self.depth+1)
        self.rhs = TreeBoosting(subsample_cols=self.subsample_cols, min_leaf=self.min_leaf, min_child_weight=self.min_child_weight,
                                max_deph=self.max_depth, lambda_=self.lambda_, gamma=self.gamma,
                                eps=self.eps).fit(x=x, gradient=gradient, hessian=hessian, x_ids=x_ids[rhs], depth=self.depth+1)

    def split_col(self, x, x_ids):
        """Function that splits a column
        Args:
            x (np.array): dataset to split
            x_ids (list): list of instances to select
            var_id (int): column to split
        """
        return x[x_ids, self.var_id]

    @property
    def is_leaf(self):
        """Function that checks if a node is a leaf
        Args:
            depth (int): actual depth of the tree
        """
        return self.score == float('-inf') or self.depth >= self.max_depth

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        node = self.lhs if xi[self.var_id] <= self.split_value else self.rhs
        return node.predict_row(xi)

    def predict(self, X_test):
        return np.array([self.predict_row(xi) for xi in X_test])


## ID3 CLASSIFIER ##
class ID3Classifier:
    """Class that builds an ID3 at the client level. Not used in Federated ID3.
    random_state=self._seed,
                                     min_samples_split=max(1.0, int(0.02 * len(X_train))),
                                     **self._tree_params
    """
    def __init__(self, max_depth=5, min_size=10, random_state=42, **kwargs):
        self.__max_depth = kwargs["max_depth"] if "max_depth" in kwargs.keys() else max_depth
        self.__min_sinze = kwargs["min_size"] if "min_size" in kwargs.keys() else min_size
        self.__node = None
        self.__is_fitted = None
        self.__feature_names = None
        self.__random_state = random_state
        self.__classes = None
        self.__leave_nodes = []

    @property
    def leaf_nodes(self):
        assert self.__is_fitted
        return self.__leave_nodes

    @property
    def max_depth_(self):
        return self.__max_depth

    @max_depth_.setter
    def max_depth_(self, max_depth):
        if not self.__is_fitted:
            self.__max_depth = max_depth
        else:
            print("The model is fitted, the max depth can't be changed.")

    @property
    def classes_(self):
        if not self.__is_fitted:
            raise NotFittedError("The model is not fitted yet.")
        return self.__classes

    @property
    def min_size_(self):
        return self.__min_sinze

    @min_size_.setter
    def min_size_(self, min_size):
        if not self.__is_fitted:
            self.__min_sinze = min_size
        else:
            print("The model is fitted, the min size can't be changed.")

    @property
    def criterion_(self):
        return "The criterion used for ID3 is information gain."

    @property
    def tree_(self):
        return self.__node

    @property
    def is_fitted_(self):
        return self.__is_fitted

    @property
    def features_names_(self):
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet. Fit the model to access to this property.")
        return self.__feature_names

    @property
    def params_(self):
        return ""

    @property
    def n_features_in_(self):
        return len(self.__feature_names)

    def fit(self, X, y, feature_names=None):
        x_ids = list(np.arange(len(y)))
        features_ids = list(np.arange(X.shape[1]))
        self.__classes = list(set(y))
        self.__feature_names = feature_names or list(np.array(np.arange(X.shape[1]), dtype=np.str_))
        self.__features_types = [X[0:,i].dtype for i in range(X.shape[1])]
        self.__node = self.__split(self.__node, X, y, x_ids, features_ids, 1)
        self.__is_fitted = True

    def __split(self, node, X, y, x_ids, features_ids, depth):
        """Builds the ID3 classifier

        Args:
            node (Node): Node to be splitted
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (list): List containing the ids of the available instances
            features_ids (list): List containing the available features ids
            depth (int): Actual depth of the tree

        Returns:
            Node: returns a Node.
        """
        if not node:
            node = Node()
        node.depth = depth
        node.available_features = list(features_ids)
        labels_in_feature = [y[x] for x in x_ids]
        node.n_node_samples = len(x_ids)
        node.n_classes = len(set(labels_in_feature))
        node.class_idx = np.array(list({c:list(labels_in_feature).count(c) if c in set(labels_in_feature) else 0 for c in self.__classes}.values()))
        if node.n_classes == 1:
            node.value = int(y[x_ids[0]])
            node.is_leave = True
            self.__leave_nodes.append(node)
            # print(node.class_idx)
            return node
        if len(features_ids) == 0 or depth >= self.__max_depth:
            node.value = int(max(set(labels_in_feature), key=labels_in_feature.count))
            node.is_leave = True
            self.__leave_nodes.append(node)
            return node
        best_feature, best_feature_id, best_feature_information_gain = self.__get_feature_max_information_gain(X, y, x_ids, features_ids)
        node.value = best_feature
        node.feature_type = self.__features_types[best_feature_id]
        node.criterion_value = best_feature_information_gain
        new_available_features = list(node.available_features)
        new_available_features.remove(best_feature_id)
        node.childs = []
        feature_values = list({X[x][best_feature_id] for x in x_ids})
        for value in feature_values:
            child = Node()
            child.dad = node
            child.value = value
            child_x_ids = [x for x in x_ids if X[x][best_feature_id] == value]
            child.next = Node()
            child.next.dad = child
            node.childs.append(child)
            child.next = self.__split(child.next, X, y, child_x_ids, new_available_features, depth+1)
        return node

    def predict(self, X_test):
        """Predict the class for the X_test.
        The model must be fitted to use this function.
        Args:
            X_test (np object): data to be predicted.

        Raises:
            NotFittedError: If the model is not fitted raise and error.

        Returns:
            Array object: Array containing the predictions made by the tree
        """
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet.")
        return np.array([self.__predict_row(self.__node, row) for row in X_test], dtype=int)

    def __predict_row(self, node, row):  # sourcery skip: merge-else-if-into-elif
        """Predict the row if node is leaf, else keep moving up the tree
        Args:
            node (Node): Actual node that will be tested.
            row (np.array): row to be predicted
        Returns:
            Node or Value: If node is leaf returns prediction else returns the next 
            child based on the feature.
        """
        if not node.childs:
            return node.value
        index = self.__feature_names.index(node.value)
        value = row[index]
        for child in node.childs:
            if child.next and value == child.value:
                next_child = child.next
                return self.__predict_row(next_child, row)
        return list(node.class_idx).index(max(node.class_idx))

    def __entropy(self, y, x_ids):
        """Calculats the entropy

        Args:
            y (Array-like object): List containing the labels
            x_ids (Array-like object): IDs of the remaining data

        Returns:
            float: Entropy of the node
        """
        labels = [y[id] for id in x_ids]
        label_count = [labels.count(x) for x in set(y)]
        return sum(
            -count / len(x_ids) * math.log(count / len(x_ids), 2)
            if count else 0
            for count in label_count
        )

    def __information_gain(self, X, y, x_ids, feature_id):
        """Calculates the information gain for a feature

        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): List containing the remaining ids
            feature_id (int): ID of the feature to evaluate

        Returns:
            float: Information gain for the given feature
        """
        info_gain = self.__entropy(y, x_ids)
        feature_values = [X[x][feature_id] for x in x_ids]
        feature_set_values = list(set(feature_values))
        feature_val_count = [feature_values.count(x) for x in feature_set_values]
        feature_val_id = [
            [x_ids[i]
             for i, x in enumerate(feature_values)
             if x == feat]
             for feat in feature_set_values
        ]
        info_gain_feature = sum(
            v_counts / len(x_ids) * self.__entropy(y, v_ids)
            for v_counts, v_ids in zip(feature_val_count, feature_val_id)
        )
        info_gain = info_gain - info_gain_feature
        return info_gain

    def __get_feature_max_information_gain(self, X, y, x_ids, feature_ids):
        """Get the feature that maximices the information gain for the
        remaining data.

        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): ID's of the remaining data
            feature_ids (List[int]): ID's of the remaining features.

        Returns:
            str, int, float: Feature that maximices the information gain, the id and it's value
        """
        features_info_gain = [self.__information_gain(X, y, x_ids, feature_id)
                                for feature_id in feature_ids]
        best_feature_id = feature_ids[features_info_gain.index(max(features_info_gain))]
        return self.__feature_names[best_feature_id], best_feature_id, 0.0 # , features_info_gain[best_feature_id]

    def print_tree(self):
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet.")
        self.__print_node(self.__node, 1)

    def __print_node(self, node, depth=1):
        times = '\t'*depth
        if node.childs:
            print(times, "Node: ", node.value)
            for child in node.childs:
                print(times,"Child node: ",  child.value)
                if child.next:
                    self.__print_node(child.next, depth+2)
        else:
            print(times, "Leaf node: ", node.value)

    def reach_root_node(self, node):
        """Function to reach root node in a tree.
        # Arguments:
            node (Node): Actual node.
        """
        stack = deque()
        while node:
            if node.value:
                stack.append(node.value)
                # stack.append([node.feature, node.value])
            node = node.dad
        # Reverse the stack
        stack.reverse()
        return stack

class NewID3Classifier:
    """Class that builds an ID3 at the client level. Not used in Federated ID3.
    random_state=self._seed,
                                     min_samples_split=max(1.0, int(0.02 * len(X_train))),
                                     **self._tree_params
    """
    def __init__(self, max_depth=5, min_size=10, random_state=42, **kwargs):
        self.__max_depth = kwargs["max_depth"] if "max_depth" in kwargs.keys() else max_depth
        self.__min_sinze = kwargs["min_size"] if "min_size" in kwargs.keys() else min_size
        self.__node = None
        self.__is_fitted = None
        self.__feature_names = None
        self.__random_state = random_state
        self.__classes = None
        self.__leave_nodes = []

    @property
    def leaf_nodes(self):
        assert self.__is_fitted
        return self.__leave_nodes

    @property
    def max_depth_(self):
        return self.__max_depth

    @max_depth_.setter
    def max_depth_(self, max_depth):
        if not self.__is_fitted:
            self.__max_depth = max_depth
        else:
            print("The model is fitted, the max depth can't be changed.")

    @property
    def classes_(self):
        if not self.__is_fitted:
            raise NotFittedError("The model is not fitted yet.")
        return self.__classes

    @property
    def min_size_(self):
        return self.__min_sinze

    @min_size_.setter
    def min_size_(self, min_size):
        if not self.__is_fitted:
            self.__min_sinze = min_size
        else:
            print("The model is fitted, the min size can't be changed.")

    @property
    def criterion_(self):
        return "The criterion used for ID3 is information gain."

    @property
    def tree_(self):
        return self.__node

    @property
    def is_fitted_(self):
        return self.__is_fitted

    @property
    def features_names_(self):
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet. Fit the model to access to this property.")
        return self.__feature_names

    @property
    def params_(self):
        return ""

    @property
    def n_features_in_(self):
        return len(self.__feature_names)

    def fit(self, X, y, feature_names=None):
        x_ids = list(np.arange(len(y)))
        features_ids = list(np.arange(X.shape[1]))
        self.__classes = list(set(y))
        self.__feature_names = feature_names or list(np.array(np.arange(X.shape[1]), 
                                                            dtype=np.str_))
        self.__features_types = [X[0:,i].dtype for i in range(X.shape[1])]
        self.__node = self.__split(self.__node, X, y, x_ids, features_ids, 0)
        self.__is_fitted = True

    def __split(self, node, X, y, x_ids, features_ids, depth):
        """Builds the ID3 classifier

        Args:
            node (Node): Node to be splitted
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (list): List containing the ids of the available instances
            features_ids (list): List containing the available features ids
            depth (int): Actual depth of the tree

        Returns:
            Node: returns a Node.
        """
        if not node:
            node = Node()
        node.depth = depth
        node.available_features = list(features_ids)
        labels_in_feature = [y[x] for x in x_ids]
        node.n_node_samples = len(x_ids)
        node.n_classes = len(set(labels_in_feature))
        node.class_idx = np.array(list({c:list(labels_in_feature).count(c) if c in set(labels_in_feature) else 0 for c in self.__classes}.values()))
        if node.n_classes == 1:
            node.value = int(y[x_ids[0]])
            node.is_leave = True
            self.__leave_nodes.append(node)
            # print(node.class_idx)
            return node
        if len(features_ids) == 0 or depth >= self.__max_depth:
            node.value = int(max(set(labels_in_feature), key=labels_in_feature.count))
            node.is_leave = True
            self.__leave_nodes.append(node)
            return node
        best_feature, best_feature_id, best_feature_information_gain = self.__get_feature_max_information_gain(X, y, x_ids, features_ids)
        # node.value = best_feature
        node.feature = best_feature
        node.feature_type = self.__features_types[best_feature_id]
        node.criterion_value = best_feature_information_gain
        new_available_features = list(node.available_features)
        new_available_features.remove(best_feature_id)
        node.childs = []
        feature_values = list({X[x][best_feature_id] for x in x_ids})
        for value in feature_values:
            child = Node()
            child.dad = node
            child.feature = best_feature
            child.value = value
            # time.sleep(5)
            child_x_ids = [x for x in x_ids if X[x][best_feature_id] == value]
            child.next = Node()
            child.next.dad = child
            node.childs.append(child)
            self.__split(child.next, X, y, child_x_ids, new_available_features, depth+1)
        return node

    def predict(self, X_test):
        """Predict the class for the X_test.
        The model must be fitted to use this function.
        Args:
            X_test (np object): data to be predicted.

        Raises:
            NotFittedError: If the model is not fitted raise and error.

        Returns:
            Array object: Array containing the predictions made by the tree
        """
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet.")
        return np.array([self.__predict_row(self.__node, row) for row in X_test], dtype=int)

    def __predict_row(self, node, row):  # sourcery skip: merge-else-if-into-elif
        """Predict the row if node is leaf, else keep moving up the tree
        Args:
            node (Node): Actual node that will be tested.
            row (np.array): row to be predicted
        Returns:
            Node or Value: If node is leaf returns prediction else returns the next 
            child based on the feature.
        """
        if not node.childs:
            return node.value
        index = self.__feature_names.index(node.feature)
        value = row[index]
        for child in node.childs:
            if value == child.value:
                return self.__predict_row(child, row)
        return list(node.class_idx).index(max(node.class_idx))

    def __entropy(self, y, x_ids):
        """Calculats the entropy

        Args:
            y (Array-like object): List containing the labels
            x_ids (Array-like object): IDs of the remaining data

        Returns:
            float: Entropy of the node
        """
        labels = [y[id] for id in x_ids]
        label_count = [labels.count(x) for x in set(y)]
        return sum(
            -count / len(x_ids) * math.log(count / len(x_ids), 2)
            if count else 0
            for count in label_count
        )

    def __information_gain(self, X, y, x_ids, feature_id):
        """Calculates the information gain for a feature

        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): List containing the remaining ids
            feature_id (int): ID of the feature to evaluate

        Returns:
            float: Information gain for the given feature
        """
        info_gain = self.__entropy(y, x_ids)
        feature_values = [X[x][feature_id] for x in x_ids]
        feature_set_values = list(set(feature_values))
        feature_val_count = [feature_values.count(x) for x in feature_set_values]
        feature_val_id = [
            [x_ids[i]
                for i, x in enumerate(feature_values)
                if x == feat]
            for feat in feature_set_values
        ]
        info_gain_feature = sum(
            v_counts / len(x_ids) * self.__entropy(y, v_ids)
            for v_counts, v_ids in zip(feature_val_count, feature_val_id)
        )
        info_gain = info_gain - info_gain_feature
        return info_gain

    def __get_feature_max_information_gain(self, X, y, x_ids, feature_ids):
        """Get the feature that maximices the information gain for the
        remaining data.

        Args:
            X (Array-like object): Training data
            y (Array-like object): Training labels
            x_ids (List[int]): ID's of the remaining data
            feature_ids (List[int]): ID's of the remaining features.

        Returns:
            str, int, float: Feature that maximices the information gain, the id and it's value
        """
        features_info_gain = [self.__information_gain(X, y, x_ids, feature_id)
                                for feature_id in feature_ids]
        best_feature_id = feature_ids[features_info_gain.index(max(features_info_gain))]
        return self.__feature_names[best_feature_id], best_feature_id, 0.0 # , features_info_gain[best_feature_id]

    def print_tree(self):
        if not self.__is_fitted:
            raise NotFittedError("Model is not fitted yet.")
        self.__print_node(self.__node, 0)

    def __print_node(self, node, depth=1):
        times = '\t'*depth
        if node.childs:
            for child in node.childs:
                print(f"{times}Child node: Feature= {child.feature}. Value={child.value}")
                self.__print_node(child, depth+1)
        else:
            print(times, "Leaf node: ", node.label)

    def reach_root_node(self, node):
        """Function to reach root node in a tree.
        # Arguments:
            node (Node): Actual node.
        """
        stack = deque()
        while node:
            if node.value:
                # stack.append(node.value)
                stack.append([node.feature, node.value])
            else:
                stack.append([node.feature, list(node.class_idx).index(max(node.class_idx))])
            node = node.dad
        # Reverse the stack
        stack.reverse()
        return stack