import random
from copy import deepcopy

import numpy as np

from flex.pool import FlexPool
from flex.model import FlexModel
from flex.data import Dataset
from flex.pool.decorators import (
    collect_clients_weights,
    deploy_server_model,
    evaluate_server_model,
    init_server_model,
    set_aggregated_weights,
)
from flextrees.pool.pool_functions import(
    select_client_by_id_from_pool,
    select_client_neq_id_from_pool,
)
from flextrees.pool.aggregators_fegbdt import aggregate_transition_step
from flextrees.utils import first_grad, first_hess, update_local_gradient_hessian
from flextrees.utils import LSHash
from flextrees.utils.utils_trees import TreeBoosting
from flextrees.utils.utils_gbdt import softmax


@init_server_model
def init_server_model_gbdt(config=None, *args, **kwargs):
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
                'max_depth': 8,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'estimators': [],
            },
            'clients_params': {
                'max_depth': 8,
                'n_estimators': 100,
                'n_hashtables': 0,
                'gradients': None,
                'hessians': None,
                'global_gradients': None,
                'global_hessians': None,
                'learning_rate': 0.1,
                'estimators': []
            }
        }

    server_flex_model['model'] = []
    dataset_dim = kwargs['dataset_dim']
    n_hash_tables = min(40, dataset_dim - 1)
    server_flex_model['lsh'] = LSHash(hash_size=8, input_dim=dataset_dim, num_hashtables=n_hash_tables)
    server_flex_model.update(config)

    return server_flex_model

@deploy_server_model
def deploy_server_config_gbdt(server_flex_model, *args, **kwargs):
    client_flex_model = FlexModel()

    client_flex_model['clients_params'] = deepcopy(server_flex_model['clients_params'])
    client_flex_model['client_id'] = f"client_{random.randint(a=10, b=10000)}" # Create a random ID
    client_flex_model['lsh'] = deepcopy(server_flex_model['lsh'])
    return client_flex_model

# Functions at client level for calculating their own hash values.
def init_hash_tables(client_flex_model: FlexModel, client_data: Dataset, *args, **kwargs):
    """Function deprecated. Moved to the server
    """
    client_flex_model['base_pred'] = np.full((client_data.X_data.to_numpy().shape[0], 1), 1).flatten().astype('float64')
    client_flex_model['gradients'] = first_grad(client_flex_model['base_pred'], client_data.y_data.to_numpy().flatten())
    client_flex_model['hessians'] = first_hess(client_flex_model['base_pred'])
    client_flex_model['idx'] = np.arange(len(client_data.X_data.to_numpy()))
    client_flex_model['global_hash_table'] = []
    client_flex_model['global_gradients'] = {idx:0 for idx in client_flex_model['idx']}
    client_flex_model['global_hessians'] = {idx:0 for idx in client_flex_model['idx']}

def compute_hash_values(client_flex_model: FlexModel, client_data: Dataset, *args, **kwargs):
    if 'client_id' not in client_flex_model.keys():
        client_flex_model['client_id'] = f"client_{random.randint(a=10, b=10000)}" # Create a random ID

    client_id = client_flex_model['client_id']

    client_flex_model['idx_hash_values'] = []
    for idx, row in enumerate(client_data.X_data.to_numpy()):
        extra_data = f"{client_id},{str(idx)}"
        client_flex_model['lsh'].index(row, extra_data=extra_data)
        d = {}
        for j, plane in enumerate(client_flex_model['lsh'].uniform_planes):
            plane_id = f"plane_{str(j)}"
            d[plane_id] = client_flex_model['lsh'].hash(plane, row)
        client_flex_model['idx_hash_values'].append((extra_data, d))

@collect_clients_weights
def collect_hash_tables(client_flex_model, *args, **kwargs):
    """Function to get the hash tables from the clients to the aggregators
    so they can 'map_reduce' them.

    Args:
        client_flex_model (FlexModel): Client FlexModel with the hash tables

    Returns:
        list: List with the hash tables from the client
    """
    return client_flex_model['idx_hash_values']

@collect_clients_weights
def collect_clients_ids(client_flex_model, *args, **kwargs):
    return client_flex_model['client_id']

@collect_clients_weights
def collect_client_gradients_hessians_by_idx(client_flex_model, *args, **kwargs):
    idx = kwargs['idx']
    gradients_ = client_flex_model['gradients'][idx]
    hessians_ = client_flex_model['hessians'][idx]
    return (gradients_, hessians_)

@set_aggregated_weights
def send_hash_table_to_server(server_flex_model: FlexModel, hash_table_j, *args, **kwargs):
    server_flex_model['client_hash_table'] = deepcopy(hash_table_j)

@set_aggregated_weights
def set_ids_clients_into_server(server_flex_model: FlexModel, clients_ids, *args, **kwargs):
    server_flex_model['clients_ids'] = deepcopy(clients_ids)

@set_aggregated_weights
def set_hash_tables_to_server(server_flex_model: FlexModel, global_hash_table, *args, **kwargs):
    server_flex_model['global_hash_table'] = deepcopy(global_hash_table)

@set_aggregated_weights
def set_last_tree_trained_to_server(server_flex_model: FlexModel, last_clf, *args, **kwargs):
    server_flex_model['last_tree_trained'] = deepcopy(last_clf[0])
    server_flex_model['server_params']['estimators'].append(deepcopy(last_clf[0]))

@deploy_server_model
def deploy_hash_table_to_clients(server_flex_model: FlexModel, *args, **kwargs):
    client_flex_model = FlexModel()
    client_flex_model['global_hash_table'] = server_flex_model['global_hash_table']
    # Once the client has the hash table, we can initialize the global gradients
    return client_flex_model

@deploy_server_model
def deploy_client_j_has_table_to_client(server_flex_model: FlexModel, *args, **kwargs):
    client_flex_model = FlexModel()
    client_flex_model['hash_table_j'] = deepcopy(server_flex_model['client_hash_table'])
    return client_flex_model

@deploy_server_model
def deploy_last_clf(server_flex_model: FlexModel, *args, **kwargs):
    client_flex_model = FlexModel()
    client_flex_model['last_tree_trained'] = server_flex_model['last_tree_trained']
    return client_flex_model

# Primitive for updating gradients and hessians of instance with local values
def update_gradients_hessians_local_values(client_flex_model: FlexModel, client_data: Dataset, *args, **kwargs):
    idx = kwargs['idx']
    if idx == 'all':
        # print(f"Global_gradients b4  updating: {client_flex_model['global_gradients']}")
        for idx in client_flex_model['idx']:
            client_flex_model['global_gradients'][idx] += client_flex_model['gradients'][idx]
            client_flex_model['global_hessians'][idx] += client_flex_model['hessians'][idx]
        # print(f"Global_gradients after  updating: {client_flex_model['global_gradients']}")
    elif isinstance(idx, int) and idx in client_flex_model['idx']:
        print(f"Global_gradients b4  updating idx: {client_flex_model['global_gradients'][idx]}")
        client_flex_model['global_gradients'][idx] += client_flex_model['gradients'][idx]
        client_flex_model['global_hessians'][idx] += client_flex_model['hessians'][idx]
        print(f"Global_gradients b4  updating idx: {client_flex_model['global_gradients'][idx]}")
        import time
        time.sleep(5)
    else:
        raise ValueError(f"The {idx} is not in the client's data.")

def client_global_gh_update(client_flex_model: FlexModel, client_data:Dataset, 
                            gradients_ = None, hessians_ = None, *args, **kwargs
                            ):
    """Function that update the gradient and the hessian of the local client,
    using the aggregated gradients and hessians from other clients.
    """
    for idx, key in enumerate(client_flex_model['global_hash_table']):
        # key = list(row.items())[0][0]
        for client_id_idx, val in gradients_.items():
            if client_id_idx in gradients_.items():
                client_flex_model['global_gradients'][idx] += val
                client_flex_model['global_hessians'][idx] += hessians_[client_id_idx]

def get_client_hash_tables(client_flex_model: FlexModel, client_data: Dataset, *args, **kwargs):
    # return client_flex_model['idx_hash_values']
    return client_flex_model['global_hash_table']

def get_client_gradients_hessians_by_idx(client_flex_model: FlexModel, client_data: Dataset, *args, **kwargs):
    idx = kwargs['idx']
    if isinstance(idx, int) and idx in client_flex_model['idx']:
        return client_flex_model['gradients'][idx], client_flex_model['hessians'][idx]
    else:
        raise ValueError(f"The {idx} is not in the client's data.")

def map_reduce_hash_tables(clients: FlexPool, server: FlexPool, aggregator: FlexPool):
    for m, client_m_id in enumerate(clients):
        # Compute on client_m
        client_m = clients.select(lambda a, b: a==client_m_id)
        for j, client_j_id in enumerate(clients):
            client_j = clients.select(lambda a, b: a==client_j_id)
            if m != j:
                aggregator.map(func=collect_hash_tables, dst_pool=client_j)
                aggregator.map(func=aggregate_transition_step)
                aggregator.map(func=send_hash_table_to_server, dst_pool=server)
                server.map(func=deploy_client_j_has_table_to_client, dst_pool=client_m)
                # Once the client_m has the hash_table, he can find the better instances
                client_m.map(func=find_instance_highest_count_hash_values)
            else:
                # Instances of client m are equal to instances of client m
                # We don't need to store them, just continue
                ...
            

def find_instance_highest_count_hash_values(client_flex_model: FlexModel, client_data: Dataset, *args, **kwargs):
    """Function that computes on client the operations to find the similar instances
    from other clients in order to complete the global hash table.
    Args:
        client_flex_model (FlexModel): Client that computes the operations
        client_data (Dataset): Client's data
    """
    assert 'hash_table_j' in client_flex_model.keys()
    hash_table_j = client_flex_model['hash_table_j'][0]
    # Iterate over the local hash table
    client_id_i = client_flex_model['client_id']
    id_options = {}
    similar_instances_ids = {}
    for instance_i in client_flex_model['idx_hash_values']:
        client_id_i = instance_i[0]
        dict_i = instance_i[1]
        similar_instances_ids[client_id_i] = []
        for instance_j in hash_table_j:
            client_id_j, dict_j = instance_j[0], instance_j[1]
            counts = sum(
                {k: 1 if dict_i.get(k) == dict_j.get(k) else 0 for k in dict_i.keys()}.values()
            )
            if id_options.get(counts):
                id_options[counts].append(client_id_j)
            else:
                id_options[counts] = [client_id_j]
            # Once all the has values have been processed at client, select one random
            max_count = max(id_options.keys())
            similar_id = id_options[max_count][random.randint(0, len(id_options[max_count])-1)] if len(
                id_options[max_count]) > 0 else id_options[max_count]
            similar_instances_ids[client_id_i].append(similar_id)
    # Add the similar_instances_ids to the clients global_hash_table
    client_flex_model['global_hash_table'].append(similar_instances_ids)

def train_n_estimators(clients: FlexPool, server: FlexPool,
                    aggregator: FlexPool, total_estimators: int):
    """Function that make the training of whole training phase.

    Args:
        clients (FlexPool): Pool of clients that will train the
        server (FlexPool): Pool of servers that will orchestate the training phase
        aggregator (FlexPool): Pool of aggregators that will aggregate make the
        aggregation process
        total_estimators (int): Total estimators that the clients will build
    """
    estimators_built = 0
    for _ in range(total_estimators):
        if estimators_built >= total_estimators:
            print(f"Model number: {estimators_built} has been trained. Ending training phase.")
            break
        # Each client have to build a tree, but before the gradients have to be updated
        for client_id_i in clients:
            client_i = clients.select(select_client_by_id_from_pool, other_actor_id=client_id_i)
            # Get the hash_table from the client
            hash_vector_i = client_i.map(func=get_client_hash_tables)[0][0]
            # Iterate through clients to update the gradients and hessians
            # Update the gradients and hessians on client_i
            print("Updating the client_i's global gradients with other clients gradients")
            for client_id_j in clients:
                if client_id_i != client_id_j:
                    gradients_ = {}
                    hessians_ = {}
                    # Get the hash_table from client_id_j
                    client_j = clients.select(select_client_by_id_from_pool, other_actor_id=client_id_j)
                    hash_vector_j = client_j.map(func=get_client_hash_tables)[0][0]
                    for key_j, _ in hash_vector_j.items():
                        # for key, _ in instance_vector.items():
                        idx_j = int(key_j.split(',')[1])
                        for key_i in hash_vector_i.keys():
                            if key_j in hash_vector_i[key_i]:
                                gradients_j, hessians_j = client_j.map(func=get_client_gradients_hessians_by_idx, idx=idx_j)[0]
                                gradients_[key_j] = gradients_j
                                hessians_[key_j] = hessians_j
                    if len(gradients_):
                        client_i.map(client_global_gh_update, gradients_=gradients_, hessians_=hessians_)
            # Conduct on client_i
            print("Updating local gradients on client_i")
            client_i.map(update_gradients_hessians_local_values, idx="all")
            # Train the model at the client_i
            print(f"Training model number: {estimators_built}")
            client_i.map(train_single_tree_at_client)
            # Send the model to the server
            aggregator.map(func=collect_last_tree_trained, dst_pool=client_i)
            aggregator.map(func=aggregate_transition_step)
            aggregator.map(func=set_last_tree_trained_to_server, dst_pool=server)
            # Deploy the model to all the clients, excect the one that trained the model
            # as she already has it
            clients_not_client_i = clients.select(select_client_neq_id_from_pool, neq_actor_id=client_id_i)
            server.map(func=deploy_last_clf, dst_pool=clients_not_client_i)
            # Make all the clients, except client_i, to add the last_tree_trained
            # to their estimators
            clients_not_client_i.map(func=clients_add_last_tree_trained_to_estimators)
            estimators_built += 1

def train_single_tree_at_client(client_flex_model, client_data, *args, **kwargs):
    clf = TreeBoosting(max_deph=client_flex_model['clients_params']['max_depth'])
    clf.fit(x=client_data.X_data.to_numpy(),
            gradient=np.array(list(client_flex_model['global_gradients'].values())),
            hessian=np.array(list(client_flex_model['global_hessians'].values())),
            x_ids=client_flex_model['idx']
            )
    client_flex_model['last_tree_trained'] = clf
    client_flex_model['clients_params']['estimators'].append(clf)
    # Update the base_preds after building the tree.
    client_flex_model['base_pred'] += client_flex_model['clients_params']['learning_rate'] * clf.predict(client_data.X_data.to_numpy())
    train_labels = client_data.y_data.to_numpy().flatten()
    # Update local gradient_hessian
    gradients_, hessians_ = update_local_gradient_hessian(
        base_pred=client_flex_model['base_pred'],
        train_labels=train_labels # client_data.y_data.to_numpy()
        )
    client_flex_model['gradients'] = gradients_
    client_flex_model['hessians'] = hessians_

@collect_clients_weights
def collect_last_tree_trained(client_flex_model: FlexModel, *args, **kwargs):
    return client_flex_model['last_tree_trained']

def clients_add_last_tree_trained_to_estimators(client_flex_model, client_data, *args, **kwargs):
    """Function that ensures that the last tree trained is added to the
    estimators in the client's FlexModel. After adding the last tree trained,
    it is deleted from the key, 'last_tree_trained'.

    Args:
        client_flex_model (FlexModel): client's FlexModel
        client_data (Dataset): Client's Dataset
    """
    client_flex_model['clients_params']['estimators'].append(deepcopy(client_flex_model['last_tree_trained']))
    del client_flex_model['last_tree_trained']

@evaluate_server_model
def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset, *args, **kwargs):
    test_data, test_labels = test_data.to_numpy()
    test_labels = test_labels.flatten()
    preds = np.zeros(test_data.shape[0])
    estimators = server_flex_model['server_params']['estimators']
    learning_rate = server_flex_model['server_params']['learning_rate']

    for clf in estimators:
        preds += learning_rate * clf.predict(test_data)
    predicted_probas = softmax(np.full((test_data.shape[0], 1), 1).flatten().astype('int64') + preds)
    y_pred = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
    y_pred = np.array([int(pred) for pred in y_pred])
    from sklearn import metrics
    acc, f1, report = metrics.accuracy_score(test_labels, y_pred), metrics.f1_score(test_labels, y_pred, average='macro'), metrics.classification_report(test_labels, y_pred)  # noqa: E501
    auc = metrics.roc_auc_score(test_labels, y_pred, average='weighted')
    print(f"Accuracy: {acc}")
    print(f"F1_Macro: {f1}")
    print(f"AUC: {auc}")
    print(report)

def evaluate_global_model_clients_gbdt(
    client_flex_model: FlexModel(),
    client_data: Dataset,
    *args, **kwargs
):
    from sklearn import metrics

    X_test, y_test = client_data.to_numpy()
    y_test = y_test.flatten()
    preds = np.zeros(X_test.shape[0])
    estimators = client_flex_model['clients_params']['estimators']
    learning_rate = client_flex_model['clients_params']['learning_rate']

    for clf in estimators:
        preds += learning_rate * clf.predict(X_test)
    predicted_probas = softmax(np.full((X_test.shape[0], 1), 1).flatten().astype('int64') + preds)
    # breakpoint()
    y_pred = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
    y_pred = np.array([int(pred) for pred in y_pred])
    # y_pred = [1 for _ in y_pred]
    client_id = client_flex_model['client_id']
    acc, f1, report = metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='macro'), metrics.classification_report(y_test, y_pred)  # noqa: E501
    auc = metrics.roc_auc_score(y_test, y_pred, average='weighted')
    print("Results on test data at client level.")
    print(f"Accuracy: {acc}")
    print(f"Macro F1: {f1}")
    print(f"AUC: {auc}")
    print(f"Classificarion report: \n {report}")
