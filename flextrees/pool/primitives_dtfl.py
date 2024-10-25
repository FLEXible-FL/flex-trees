import sys
import random
from copy import deepcopy

from dtfl.utils.save_results import client_write_results, print_results_client

from flex.pool.decorators import (
    collect_clients_weights,
    deploy_server_model,
    evaluate_server_model,
    init_server_model,
    set_aggregated_weights
)


@init_server_model
def init_server_model_dtfl(
    config=None, *args, **kwargs
):
    """Function to initialize the server model (rules_threshold, etc.)

    Args:
        model (dict, optional): _description_. Defaults to None.
    Returns:
        FlexModel: A FlexModel with the config that will be deployed 
        to the clients to train the local model.
    """
    from flex.model.model import FlexModel

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'local_model_params': {
                'max_depth': 5,
                'criterion': 'gini',
                'splitter': 'best',
                'model_type': 'cart',
            },
            'global_model_params': {
                'rules_threshold': 3000
            }
        }

    server_flex_model.update(config)

    return server_flex_model

@deploy_server_model
def deploy_local_model_config_dtfl(
    server_flex_model, *args, **kwargs
):
    """Function to deploy the configuration of the local model to the clients1
    Args:
        server_flex_model (flex.model.model.FlexModel): The server's FlexModel
    Returns:
        A FlexModel created for the client with the 
    """
    from flex.model.model import FlexModel

    client_flex_model = FlexModel()
    client_flex_model['local_model_params'] = deepcopy(server_flex_model['local_model_params'])
    return client_flex_model

def train_dtfl(client_flex_model, server_flex_model, *args, **kwargs):
    """Function to train the global model at the server

    Args:
        client_flex_model (_type_): _description_
        client_data (_type_): _description_
    """

def train_local_model(client_flex_model, client_data, *args, **kwargs):
    """Function to train a model (DT from Sklearn or ID3 from utils) at client level

    Args:
        client_flex_model (_type_): _description_
        client_data (_type_): _description_
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from dtfl.utils.utils_trees import ID3Classifier
    from dtfl.utils.c45_tree import C45Tree
    if client_flex_model['local_model_params']['model_type'] == 'c45':
        model = C45Tree
    elif client_flex_model['local_model_params']['model_type'] == 'id3':
        model = ID3Classifier
    else:
        model = DecisionTreeClassifier
    # model = ID3Classifier if client_flex_model['local_model_params']['model_type'] == 'id3' else DecisionTreeClassifier  # noqa: E501

    clf = model(
        random_state=42,
        min_samples_split=max(1.0, int(0.02 * len(client_data.X_data))),
        max_depth=client_flex_model['local_model_params']['max_depth'],
        criterion=client_flex_model['local_model_params']['criterion'],
        splitter=client_flex_model['local_model_params']['splitter'],
        improve_speed=False, # This is for the C45Tree
    )
    X_data, y_data = client_data.X_data.to_numpy(), client_data.y_data.to_numpy()
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size=0.2)  # noqa: E501
    # feature_types = ['str'] * len(X_data[0])
    feature_types = None
    # clf.fit(X=client_data.X_data, y=client_data.y_data)
    if isinstance(clf, ID3Classifier) or isinstance(clf, C45Tree):
        # feature_types = [type(f) for f in X_data[0]]
        clf.fit(X=X_data, y=y_data, feature_types=feature_types)
    else:
        clf.fit(X_data, y_data)
    y_pred = clf.predict(X_test)
    acc, f1 = accuracy_score(y_pred, y_test), f1_score(y_pred, y_test, average='macro')
    print(f"Results on client. Acc: {acc}. F1_macro: {f1}. Len of data: {len(X_data)}, Len of test data: {len(X_test)}")
    # Once the model is trained, each client create the ruleset
    from sklearn import metrics
    print(metrics.classification_report(y_pred, y_test))
    # print(metrics.classification_report(y_pred_cart, y_test))
    from dtfl.utils.ConjunctionSet import ConjunctionSet
    feature_names = [f'x{i}' for i in range(X_data.shape[1])]

    validation_data = X_data
    validation_labels = y_data
    client_params = [clf]
    min_forest_size = 1
    feature_types = ['str'] * len(feature_names) if client_flex_model['local_model_params']['model_type'] == 'id3' else ['int'] * len(feature_names)
    if client_flex_model['local_model_params']['model_type'] == 'c45':
        feature_types = [type(f) for f in X_data[0]]
    print(f"Feature types: {feature_types}")
    local_cs = ConjunctionSet(feature_names, validation_data, validation_data, validation_labels,
                            client_params, feature_types=feature_types,
                            amount_of_branches_threshold=3000,
                            minimal_forest_size=min_forest_size, estimators=clf,
                            filter_approach='probability', personalized=False)
    client_flex_model['local_cs'] = local_cs
    # print(clf.decision_path(X_data))
    client_flex_model['local_branches'] = local_cs.get_branches_list()
    client_flex_model['local_branches_df'] = local_cs.get_conjunction_set_df().round(decimals=5)
    client_flex_model['local_classes'] = clf.classes_
    client_flex_model['local_tree'] = clf
    client_flex_model['X_test'] = X_test
    client_flex_model['y_test'] = y_test
    client_flex_model['local_acc'] = acc
    client_flex_model['local_f1'] = f1


@collect_clients_weights
def collect_clients_weights_dtfl(
    client_flex_model, *args, **kwargs
):
    """Function that collect the rules generated from the tree in a client

    Args:
        client_flex_model (_type_): _description_
    """
    return client_flex_model['local_branches'], client_flex_model['local_classes'], client_flex_model['local_branches_df'], client_flex_model['local_model_params']['model_type']  # noqa: E501

@set_aggregated_weights
def set_aggregated_weights_dtfl(
    server_flex_model, aggregated_weights, *args, **kwargs
):
    """Function to set the aggregated weights (rules in this case) to the server

    Args:
        server_flex_model (_type_): _description_
        aggregated_weights (_type_): _description_
    """
    server_flex_model['global_model'] = aggregated_weights

@set_aggregated_weights
def set_local_trees_to_server(
    server_flex_model, aggregated_weights, *args, **kwargs
):
    """
    Function that set the local trees to the server
    """
    server_flex_model['list_of_trees'] = aggregated_weights

@evaluate_server_model
def evaluate_server_model_dtfl(
    server_flex_model, test_data, test_labels, *args, **kwargs
):
    """Function to evaluate the global model on the test data

    Args:
        server_flex_model (_type_): _description_
        test_data (_type_): _description_
        test_labels (_type_): _description_
    """
    classes_tree = get_classes_branches(server_flex_model['global_model'][2])
    
    # predictions = perso_dt.predict(X_test, self.get_classes_branches(perso_branches_df), perso_branches_df)
    y_pred, y_explainations = server_flex_model['global_model'][1].predict(test_data, classes_tree, server_flex_model['global_model'][2])  # noqa: E501
    from sklearn import metrics
    acc, f1, report = metrics.accuracy_score(test_labels, y_pred), metrics.f1_score(test_labels, y_pred, average='macro'), metrics.classification_report(test_labels, y_pred)  # noqa: E501
    # print(y_pred)
    print("Results on test data at server level.")
    print(f"Accuracy: {acc}")
    print(f"Macro F1: {f1}")
    print(f"Classificarion report: \n {report}")
    client_write_results(kwargs['filename'], client_id='server', 
                        acc_local_model=len(server_flex_model['selected_trees']),
                        f1_local_model=0.0,
                        acc_global_model=acc,
                        f1_global_model=f1,
                        tam_test_data=len(test_labels)
                        )

def get_classes_branches(branches):
    """Function to get the classes from a branch DataFrame
    Args:
        branches (pd.DataFrame): DataFrame with the branches
    Returns:
        list: Classes in the branch DataFrame
    """
    assert branches is not None
    # return [c for c in range(len(branches['probas'].iloc[0]))]
    return list(range(len(branches['probas'].iloc[0])))

# FUNCIONES NUEVA IDEA

@deploy_server_model
def send_all_trees_to_client(
    server_flex_model, *args, **kwargs
):
    """Function to deploy the configuration of the local model to the clients1
    Args:
        server_flex_model (flex.model.model.FlexModel): The server's FlexModel
    Returns:
        A FlexModel created for the client with the 
    """
    from flex.model.model import FlexModel

    client_flex_model = FlexModel()
    client_flex_model['client_trees'] = deepcopy(server_flex_model['list_of_trees'])
    return client_flex_model

@deploy_server_model
def deploy_global_model(
    server_flex_model, *args, **kwargs
):
    """Function to deploy the configuration of the local model to the clients1
    Args:
        server_flex_model (flex.model.model.FlexModel): The server's FlexModel
    Returns:
        A FlexModel created for the client with the 
    """
    from flex.model.model import FlexModel

    client_flex_model = FlexModel()
    client_flex_model['global_model'] = deepcopy(server_flex_model['global_model'])
    return client_flex_model

@collect_clients_weights
def collect_clients_trees(
    client_flex_model, *args, **kwargs
):
    """
    Function that send the local tree from the client to the server
    """
    return client_flex_model['local_tree']

def evaluate_global_model(
    client_flex_model, client_data, *args, **kwargs
):
    """
    Function to evaluate the global model with the test data of each client.
    """
    classes_tree = get_classes_branches(client_flex_model['global_model'][2])
    test_data = client_flex_model['X_test']
    test_labels = client_flex_model['y_test']
    # Local model:
    local_clf = client_flex_model['local_tree']
    local_y_preds = local_clf.predict(test_data)
    # TODO: Add function decision_path to the ID3Classifier to get the rules.
    local_y_decision_path = local_clf.decision_path(test_data)
    # local_y_decision_path = "TODO For C45"
    # Global model:
    y_pred, y_explanations = client_flex_model['global_model'][1].predict(test_data, classes_tree, client_flex_model['global_model'][2])  # noqa: E501
    from sklearn import metrics
    if len(y_pred) != len(test_labels):
        breakpoint()
    acc, f1, _ = metrics.accuracy_score(test_labels, y_pred), metrics.f1_score(test_labels, y_pred, average='macro'), metrics.classification_report(test_labels, y_pred)  # noqa: E501
    client_id = f"cliend_{random.randint(a=10, b=10000)}" # Create a random ID
    client_write_results(kwargs['filename'], client_id=client_id, 
                        acc_local_model=client_flex_model['local_acc'],
                        f1_local_model=client_flex_model['local_f1'],
                        acc_global_model=acc,
                        f1_global_model=f1,
                        tam_test_data=len(test_labels)
                        )
    if acc > client_flex_model['local_acc'] or f1 > client_flex_model['local_f1']:
        print_results_client(tam_test_data=len(test_labels), acc_global_model=acc,
                            acc_local_model=client_flex_model['local_acc'],
                            f1_global_model=f1,
                            f1_local_model=client_flex_model['local_f1']
                            )
    # Write explainable output.
    import pandas as pd
    explanations_output = []
    # df_explanations = pd.DataFrame(columns=["instance", "label", "LocalModel", "GlobalModel", "LocalExplanation", "GlobalExplanation"])
    rows_list = []
    for i in range(len(test_data)):
        local_explanation = local_y_decision_path[i]
        # local_explanation = "TODO For C45"
        instance_i_text = ""
        instance_i_text += f"The instance {test_data[i]} is associated to the class {test_labels[i]}.\n"
        instance_i_text += f"The instance has been classified as {local_y_preds[i]} by the local model because: {local_explanation}.\n"
        instance_i_text += f"The instance has been classified as {y_pred[i]} by the global model because: {y_explanations[i]}.\n"
        explanations_output.append(instance_i_text)
        # TODO: Uncomment the next line when TreeBanchMixed is finished.
        # if test_labels[i] == y_pred[i] and test_labels[i] != local_y_preds[i]:
        #     print(instance_i_text)
        dict_explanations = {
            "instance": test_data[i],
            "label": test_labels[i],
            "LocalModel": local_y_preds[i],
            "Globalmodel": y_pred[i],
            "LocalExplanation": local_explanation,
            "GlobalExplanation": y_explanations[i],
        }
        rows_list.append(dict_explanations)
    df_explanations = pd.DataFrame(rows_list)
    # Save the explanations to a file. MacOs for my laptop, Windows for my desktop.
    if sys.platform == "darwin":
        # MacOS
        df_explanations_folder = "~/Documents/UGR-Work/ArticuloTesis/DTFL/explanations_folder"
    elif sys.platform == "win32":
        # Windows
        df_explanations_folder = "C:/Users/Cris/Documents/Alberto/Tesis/DTFL/explanations_folder"
    # df_explanations_folder = "C:/Users/Cris/Documents/Alberto/Tesis/DTFL/explanations_folder"
    df_explanations.to_csv(f"{df_explanations_folder}/explanation_{client_id}.csv", index=False, index_label=False)


def evaluate_global_trees(client_flex_model, client_data, *args, **kwargs):
    """
    Function that evaluate the global trees in each client to test them and check
    if they pass and threshold to be used to generate the global model.
    """
    trees = client_flex_model['client_trees']
    trees_test_acc = []
    trees_test_f1 = []
    from sklearn.metrics import accuracy_score, f1_score
    X_test, y_test = client_flex_model['X_test'], client_flex_model['y_test']
    trees_test_acc = [accuracy_score(y_test, clf.predict(X_test)) for clf in trees]
    trees_test_f1 = [f1_score(y_test, clf.predict(X_test), average='macro') for clf in trees]
    client_flex_model['trees_test_acc'] = trees_test_acc
    client_flex_model['trees_test_f1'] = trees_test_f1

@collect_clients_weights
def collect_local_evaluations_from_clients(
    client_flex_model, *args, **kwargs
):
    """Function that collect the rules generated from the tree in a client

    Args:
        client_flex_model (_type_): _description_
    """
    return client_flex_model['trees_test_acc'], client_flex_model['trees_test_f1']

@set_aggregated_weights
def set_selected_trees_to_server(
    server_flex_model, aggregated_weights, *args, **kwargs
):
    """
    Function that set the local trees to the server
    """
    server_flex_model['selected_trees'] = aggregated_weights
