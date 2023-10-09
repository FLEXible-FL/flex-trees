import random
from copy import deepcopy

import numpy as np

from flex.model import FlexModel
from flex.data import Dataset
from flex.pool.decorators import (
    collect_clients_weights,
    deploy_server_model,
    evaluate_server_model,
    init_server_model,
    set_aggregated_weights,
)

from flextrees.utils.utils_gbdt import LSHash

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
                'max_depth': 5,
                'n_estimators': 100,
            },
            'clients_params': {
                'max_depth': 5,
                'n_estimators': 100,
                'n_hashtables': 0
            }
        }

    server_flex_model['model'] = []

    server_flex_model.update(config)

    return server_flex_model

def deploy_server_config_gbdt(server_flex_model, *args, **kwargs):
    client_flex_model = FlexModel()

    client_flex_model['clients_params'] = deepcopy(server_flex_model['clients_params'])
    client_flex_model['client_id'] = f"client_{random.randint(a=10, b=10000)}" # Create a random ID
    return client_flex_model

# Functions at client level for calculating their own hash values.
def init_hash_tables(client_flex_model: FlexModel, client_data: Dataset, *args, **kwargs):
    """Function deprecated. Moved to the server
    """
    n_hash_tables = min(40, client_data.X_data.to_numpy().shape[1] - 1)
    client_flex_model['n_hashtables'] = n_hash_tables
    lsh = LSHash(hash_size=8, input_dim=client_data.X_data.to_numpy().shape[1], num_hashtables=n_hash_tables)
    client_flex_model['lsh'] = lsh

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
