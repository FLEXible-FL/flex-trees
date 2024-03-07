"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import random
from copy import deepcopy

import numpy as np
import pandas as pd
from flex.pool import FlexPool
from flex.model import FlexModel

from flextrees.utils import (
    ID3, Node, reach_root_node, get_df_cut,
    get_feature_with_max_information_gain,
)

from flex.pool.decorators import (
    collect_clients_weights,
    deploy_server_model,
    evaluate_server_model,
    init_server_model,
    set_aggregated_weights,
)

from flextrees.pool.aggregators_fedid3 import id3_aggregate_counts, id3_aggegate_class_counts, id3_aggregate_class_counts_sum

@init_server_model
def init_server_model_id3(dataset_features: list, config=None, *args, **kwargs):
    """Function to initialize the server model

    Args:
        dataset_features (list): List that contains the name of the features
        config (dict, optional): Dict that contains the configuration of the
        server model. Defaults to None.
    """
    from flex.model import FlexModel

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': len(dataset_features) // 2,
                'available_features': dataset_features,
                'dataset_features': dataset_features[:-1],
                'used_features': [],
            },
            'clients_params': {
                'available_features': dataset_features,
                'dataset_features': dataset_features[:-1],
                'features_ids': list(range(len(dataset_features)))
            }
        }

    server_flex_model['model'] = ID3(max_depth=config['server_params']['max_depth'], 
                                    feature_names=dataset_features)

    server_flex_model.update(config)

    return server_flex_model

@deploy_server_model
def deploy_server_config_id3(server_flex_model, *args, **kwargs):
    client_flex_model = FlexModel()

    client_flex_model['clients_params'] = deepcopy(server_flex_model['clients_params'])
    return client_flex_model

@deploy_server_model
def deploy_server_model_id3(server_flex_model, *args, **kwargs):
    client_flex_model = FlexModel()

    client_flex_model['model'] = deepcopy(server_flex_model['model'])
    return client_flex_model

@deploy_server_model
def deploy_node_id3(server_flex_model, *args, **kwargs):
    client_flex_model = FlexModel()

    client_flex_model['node'] = deepcopy(kwargs['node'])
    return client_flex_model

@collect_clients_weights
def collect_clients_counts_id3(client_flex_model, *args, **kwargs):
    return client_flex_model['counts']

@collect_clients_weights
def collect_clients_class_counts_id3(client_flex_model, *args, **kwargs):
    return client_flex_model['class_count']

@set_aggregated_weights
def set_aggregated_id3(server_flex_model, aggregated_weights, *args, **kwargs):
    max_depth = server_flex_model['server_params']['max_depth']
    feature_names = server_flex_model['server_params']['dataset_features']
    tree = ID3(max_depth=max_depth, feature_names=feature_names)
    print(f"Aggregated weights: {aggregated_weights}")
    tree.set_root(aggregated_weights)
    server_flex_model['model'] = deepcopy(tree)

@evaluate_server_model
def evaluate_id3_model(server_flex_model, test_data, *args, **kwargs):
    test_data, test_labels = test_data.to_numpy()
    clf = server_flex_model['model']
    y_pred = clf.predict(test_data)
    from sklearn import metrics
    acc, f1, report = metrics.accuracy_score(test_labels, y_pred), metrics.f1_score(test_labels, y_pred, average='macro'), metrics.classification_report(test_labels, y_pred)  # noqa: E501
    print(report)
    print(f"Accuracy: {acc}")
    print(f"F1-Macro: {f1}")

# Function that calculate the counts in the clients.
def calculate_counts(client_flex_model, client_data, *args, **kwargs):
    node = kwargs['node']
    features = kwargs['features']

    # Trying to improve recursive functions. Save data in the Clients FlexModel.
    if 'X_data' not in client_flex_model:
        X_data, y_data = client_data.to_numpy()
        client_flex_model['X_data'] = X_data
        client_flex_model['y_data'] = y_data
    else:
        X_data = client_flex_model['X_data']
        y_data = client_flex_model['y_data']

    if 'df' not in client_flex_model:
        df_features = client_flex_model['clients_params']['dataset_features']
        df = pd.DataFrame(X_data, columns=df_features)
    else:
        df = client_flex_model['df']

    if isinstance(node, Node):
        client_flex_model['clients_params']['available_features'] = features
        stack = reach_root_node(node)
        x_ids, _ = get_df_cut(df, stack)
        feature_ids = [features.index(feature) for feature in features]
        counts = get_feature_with_max_information_gain(X_data,
                                                    y_data,
                                                    x_ids, feature_ids,
                                                    features
                                                    )
    else:
        # Features 0 have the node y Features 1 have the {feature: value} to test.
        stack = reach_root_node(node[0])
        test_feature = [list(node[1].keys())[0], list(node[1].values())[0]]
        stack.extend(test_feature)
        x_ids, _ = get_df_cut(df, stack)
        counts = len(x_ids)
    client_flex_model['counts'] = counts

def calculate_class_counts(client_flex_model, client_data, *args, **kwargs):
    """Get class counts in client.
    When getting the class_counts, we must receive a node that will be the leaf node.
    This implies that it will have the following structure when plugged into a stack:
    stack = [value, feature, value, feature, value, feature, ... , value, feature]
    As a stack, the last item in the list will be the root node, while the first
    item in the list will be the leaf node.
    # Arguments:
        node (Node): leaf node.
    # Returns:
        max class based on the local dataset.
    """
    node = kwargs['node']
    # features = client_flex_model['clients_params']['dataset_features']
    # X_data, y_data = client_data.to_numpy()
    # X_data, y_data = client_flex_model['X_data'], client_flex_model['y_data']
    # y_data = client_flex_model['y_data']
    if 'df' not in client_flex_model:
        X_data, y_data = client_flex_model['X_data'], client_flex_model['y_data']
        df_features = client_flex_model['clients_params']['dataset_features']
        df = pd.DataFrame(X_data, columns=df_features)
        client_flex_model['df'] = df
    else:
        y_data = client_flex_model['y_data']
        df = client_flex_model['df']
    # df = pd.DataFrame(X_data, columns=features)
    if isinstance(node, Node):
        # Get the node path in a stack.
        stack = reach_root_node(node)
    else:
        stack = reach_root_node(node[0])
        test_feature = [list(node[1].keys())[0], list(node[1].values())[0]]
        stack.extend(test_feature)
    x_ids, _ = get_df_cut(df, stack)
    labels_in_feature = [y_data[x] for x in x_ids]
    classes, counts = np.unique(labels_in_feature, return_counts=True)
    # class_count = {label: count for label, count in zip(classes, counts)}
    class_count = dict(zip(classes, counts))
    client_flex_model['class_count'] = class_count

def build_id3(node: Node, depth: int, available_features: list, pool: FlexPool, 
            max_depth: int, values_features: list):
    """Function that creates a Federated ID3 with the FLEXible framework.
    This functions integrates multiple funcions

    Args:
        node (Node): Actual node that is being evaluated.
        depth (int): Depth of the actual node.
        available_features (list): Remaining features.
        pool (FlexPool): FlexPool containing the actors that are building the tree.
        max_depth (int): Max depth of the tree.
        values_features (list): List with the possible feature values.
    """
    if node is None:
        node = Node()
    node.depth = depth
    node.available_features = list(available_features) # Make a copy of the list

    # Aggregate counts
    pool.servers.map(deploy_node_id3, pool.clients, node=node)
    # Aggergate counts
    feature = aggregate_counts(node=node, pool=pool, features=available_features)
    if depth >= max_depth or node.available_features is None or feature == -1:
        # Aggregate class counts
        class_counts = aggregate_class_counts(node=node, pool=pool)
        if class_counts in [-1, '-1']:
            # breakpoint()
            class_counts = aggregate_class_counts(node.dad.dad.dad, pool=pool)
            # class_counts = aggregate_class_counts(node.dad.dad, pool=pool)
        # print(f"Class_counts: {class_counts}")
        node.value = class_counts
        return node
    else:
        # print(f"else node.available_features: {node.available_features}")
        feature_value = aggregate_feature_counts(node, pool, node.available_features, values_features)
        feature = list(feature_value.keys())[0]
        node.available_features.remove(feature)
        new_available_features = list(node.available_features)
        node.value = feature
        # child_values = ['a', 'b']# self._value_features[feature]
        # breakpoint()
        child_values = values_features[feature]
        node.childs = []
        for value in child_values:
            child = Node()
            child.value = str(value)
            child.dad = node
            child.next = Node()
            child.next.dad = child
            node.childs.append(child)
            child.next = build_id3(child.next, depth+1, new_available_features,
                                pool, max_depth, values_features)
    return node

def aggregate_counts(node, pool, features):
    pool.clients.map(calculate_counts, node=node, features=features)
    pool.aggregators.map(collect_clients_counts_id3, pool.clients)
    pool.aggregators.map(id3_aggregate_counts)
    return pool.aggregators._models['server']['aggregated_weights']

def aggregate_class_counts(node, pool):
    pool.clients.map(calculate_class_counts, node=node)
    pool.aggregators.map(collect_clients_class_counts_id3, pool.clients)
    pool.aggregators.map(id3_aggegate_class_counts)
    return pool.aggregators._models['server']['aggregated_weights']

def aggregate_class_counts_sum(node, pool):
    pool.clients.map(calculate_class_counts, node=node)
    pool.aggregators.map(collect_clients_class_counts_id3, pool.clients)
    pool.aggregators.map(id3_aggregate_class_counts_sum)
    return pool.aggregators._models['server']['aggregated_weights']

def aggregate_feature_counts(node, pool, features, values_features):
    """Function to aggregate the information gain for the available features
    at the client to select the best that will be chosen to split.
    # Arguments:
        node: Actual ID3 Model.
        features: Remaining features
    # Returns:
        info_gain: Information gain aggregated from the clients
        for the remaining features.
    """
    value_feature = {}
    for i, feature in enumerate(features):
        class_counts_values = []
        for value in values_features[feature]:
            node_feature = [node, {feature: value}]
            class_params = aggregate_class_counts_sum(node=node_feature, pool=pool)
            # class_params = self._nodes_federation.query_model_params(node=node_feature, class_counts=True, counts=False)
            # class_params = self._server._aggregator.aggregate_class_counts_sum(class_params)
            class_counts_values.append(class_params)
        vf = 0.0
        for class_counts in class_counts_values:
            all_label_counts = np.array(list(class_counts.values()))
            all_counts = np.sum(all_label_counts)
            all_scores = all_label_counts * np.log2(
                np.divide(all_label_counts, all_counts,
                            out=np.zeros_like(all_label_counts, dtype=float),
                            where=all_counts != 0),
                out=np.zeros_like(all_label_counts, dtype=float),
                where=all_label_counts != 0)
            vf += np.sum(all_scores, axis=0)
        value_feature[feature] = vf
    # print(f"value_feature: {value_feature}")
    # print(f"value_feature: {type(value_feature)}")
    best_feature = max(value_feature.keys(), key=lambda x:value_feature[x])
    return {best_feature: value_feature[best_feature]}

def evaluate_global_model_clients(
    client_flex_model, client_data, *args, **kwargs
):
    """Function to evaluate the global model on clients local data

    Args:
        client_flex_model (FlexModel): Clients Flex Model
        client_data (Dataset): Flex Dataset object with clients data
    """
    from sklearn import metrics

    X_test, y_test = client_data.to_numpy()
    clf = client_flex_model['model']

    y_pred = clf.predict(X_test)

    client_id = f"cliend_{random.randint(a=10, b=10000)}" # Create a random ID

    acc, f1, report = metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='macro'), metrics.classification_report(y_test, y_pred)  # noqa: E501

    print("Results on test data at client level.")
    print(f"Accuracy: {acc}")
    print(f"Macro F1: {f1}")
    print(f"Classificarion report: \n {report}")
