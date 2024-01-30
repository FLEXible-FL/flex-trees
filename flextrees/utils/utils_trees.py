import time
from copy import deepcopy
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
    multiple CARTs. The Federated Extra-Trees method is based on the Extra-Trees
    method, and will be added in the future.

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
