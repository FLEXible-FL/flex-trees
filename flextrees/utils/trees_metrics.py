import os
import csv
import math
from collections import deque

def get_feature_with_max_information_gain(X, y, x_ids, feature_ids, _features):
    """Function that calculate the information gian for all the features available
    and return that feature.
    # Arguments:
        X: data
        y: labels
        x_ids: ids that were not used yet in the training
        feature_ids: ids of the available features
    """
    # print('Get feature max info gain')
    features_info_gain = [information_gain(X, y, x_ids, feature_id)
                            for feature_id in feature_ids]
    split = {
        _features[feature_id]:features_info_gain[i] 
        for i, feature_id in enumerate(feature_ids)
    }
    return split

def information_gain(X, y, x_ids, feature_id):
    """Calculate information gain for the remaining data for a feature

    # Arguments:
        X (np.array): Data
        x_ids (np.array/list): ids of the remaining data
        feature_id (int): Feature ID
    Returns:
        info_gain (float): Information gin
    """

    info_gain = entropy(x_ids, y)
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
        v_counts / len(x_ids) * entropy(v_ids, y)
        for v_counts, v_ids in zip(feature_val_count, feature_val_id)
    )
    info_gain -= info_gain_feature

    return info_gain

def entropy(x_ids, y):
    """Calculates the entropy

    Args:
        x_ids (np.array/list): ids of the remaining data

    Returns:
        entrpy (float): Entropy
    """
    # Sort labels by id
    labels = [y[xid] for xid in x_ids]
    # Count the number of instances of each category
    label_count = [labels.count(x) for x in set(y)]
    # Calculate the entropy of each category and sum them
    entropy = sum(
        -count / len(x_ids) * math.log(count / len(x_ids), 2)
        if count else 0
        for count in label_count
    )
    return entropy

def reach_root_node(node):
    """Function to reach root node in a tree.
    # Arguments:
        node (Node): Actual node.
    """

    stack = deque()
    while node:
        if node.value:
            stack.append(node.value)
        node = node.dad
    # Reverse the stack
    stack.reverse()
    return stack

def get_df_cut(df_, stack):
    """Function that receive a stack and get the dataframe cut to the values
    of the features.
    This function helps to get the x_ids to calculate the information_gain
    or the class_counts.
    # Arguments:
        stack: stack with the path from the node to the root path.
    """
    # Transform the stack into a list that contains tuples (feature, value).
    root_path = [(stack[i], stack[i+1]) for i in range(0, len(stack), 2)]
    # Query method can be faster then the for loop.
    # query = ' and '.join(feature+"=="+'"'+str(value)+'"' for feature, value in root_path)
    # df = self._df.query(query) if root_path else self._df
    # Once transformed, we have to update the df to the values.
    # df_ = df.copy()
    for feature, value in root_path:
        df_ = df_.loc[df_[feature] == value]
    x_ids = list(df_.index)
    return x_ids, df_

def client_write_results(filename, client_id, acc_local, f1_local,
                        tam_test_data):
    if not os.path.exists(filename):
        header = ['client_id', 'local_model_acc', 'local_model_f1', 'tam_test_data']
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(header)
    results = [client_id, acc_local, f1_local, tam_test_data]
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(results)

def server_write_results(filename, client_id, acc_local, f1_local,
                        tam_test_data, etime):
    if not os.path.exists(filename):
        header = ['client_id', 'local_model_acc', 'local_model_f1', 'tam_test_data', 'time']
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(header)
    results = [client_id, acc_local, f1_local, tam_test_data, etime]
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(results)
