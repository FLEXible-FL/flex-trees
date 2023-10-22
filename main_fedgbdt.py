import os
import time
from datetime import datetime
import argparse

import numpy as np

from flex.data import FedDataDistribution, FedDatasetConfig, one_hot_encoding
from flex.pool import FlexPool

from flextrees.datasets.tabular_datasets import ildp, adult, bank, credit2

from flextrees.pool import (
    init_server_model_gbdt,
    init_hash_tables,
    compute_hash_values,
    deploy_server_config_gbdt,
    collect_hash_tables,
    collect_last_tree_trained,
    aggregate_transition_step,
    aggregate_hash_tables,
    collect_clients_ids,
    set_ids_clients_into_server,
    set_hash_tables_to_server,
    deploy_hash_table_to_clients,
    map_reduce_hash_tables,
    select_client_by_id_from_pool,
    select_client_neq_id_from_pool,
    get_client_hash_tables,
    update_gradients_hessians_local_values,
    train_single_tree_at_client,
    set_last_tree_trained_to_server,
    deploy_last_clf,
    clients_add_last_tree_trained_to_estimators,
    get_client_gradients_hessians_by_idx,
    evaluate_global_model,
    client_global_gh_update,
    evaluate_global_model_clients_gbdt,
    train_n_estimators,
)


def main():  # sourcery skip: extract-duplicate-method
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nclients', type=int, default=2, help='Number of clients')
    parser.add_argument('-d', '--dist', type=str, default='iid', help='Distribution (iid/niid)')
    parser.add_argument('-ne', '--exec', type=int, default=0, help='Number of execution')
    parser.add_argument('-db', '--database', type=int, default=0, help='Database to use: 0: Nursery, 1:Adult, 2: Car, 3:Credit2')
    parser.add_argument('-e', '--estimators', type=int, default=10, help='Number of trees to train.')

    args = parser.parse_args()
    n_clients = args.nclients
    dist = args.dist
    exec = args.exec
    db = args.database
    estimators = args.estimators

    available_datasets = [ildp, adult, bank, credit2]

    dataset = available_datasets[db]
    print(f"Dataset: {dataset}")
    start_time = time.time()

    train_data, test_data = dataset(ret_feature_names=False, categorical=False)
    n_labels = len(np.unique(train_data.y_data.to_numpy()))
    # breakpoint()
    dataset_dim = train_data.to_numpy()[0].shape[1] # We need the dimension to create the LSH hyper planes
    n_clients = n_clients
    if dist == 'iid':
        federated_data = FedDataDistribution.iid_distribution(centralized_data=train_data,
                                                            n_nodes=n_clients)
    else:
        config_nidd = FedDatasetConfig(seed=0, n_nodes=n_clients, replacement=False)

        federated_data = FedDataDistribution.from_config(centralized_data=train_data,
                                                            config=config_nidd)
    # One hot encode the labels for using softmax
    def one_hot_encoding_(node_dataset, *args, **kwargs):
        """Function that apply one hot encoding to the labels of a node_dataset.

        Args:
            node_dataset (Dataset): node_dataset to which apply one hot encode to her labels.

        Raises:
            ValueError: Raises value error if n_labels is not given in the kwargs argument.

        Returns:
            Dataset: Returns the node_dataset with the y_data property updated.
        """
        from copy import deepcopy
        from flex.data import LazyIndexable, Dataset
        if "n_labels" not in kwargs:
            raise ValueError(
                "No number of labels given. The parameter n_labels must be given through kwargs."
            )
        # breakpoint()
        y_data = node_dataset.y_data.to_numpy().flatten()
        n_labels = int(kwargs["n_labels"])
        one_hot_labels = np.zeros((y_data.size, n_labels))
        one_hot_labels[np.arange(y_data.size), y_data] = 1
        new__y_data = one_hot_labels
        return Dataset(
            X_data=deepcopy(node_dataset.X_data),
            y_data=LazyIndexable(new__y_data, len(new__y_data)),
        )
    federated_data.apply(one_hot_encoding_, n_labels=n_labels)
    # Set server config
    pool = FlexPool.client_server_architecture(federated_data, init_server_model_gbdt, dataset_dim=dataset_dim)
    clients = pool.clients
    aggregator = pool.aggregators
    server = pool.servers

    # Total number of estimators
    total_estimators = estimators
    print(f"Number of trees to build: {total_estimators}")
    estimators_built = 0
    # Deploy clients config
    pool.servers.map(func=deploy_server_config_gbdt, dst_pool=pool.clients)
    # Preprocessing Stage
    pool.clients.map(func=init_hash_tables) # Init hash tables
    pool.clients.map(func=compute_hash_values) # Calculate the hash tables on the clients
    # Aggregate ids first
    pool.aggregators.map(func=collect_clients_ids, dst_pool=pool.clients)
    pool.aggregators.map(func=aggregate_transition_step)
    pool.aggregators.map(func=set_ids_clients_into_server, dst_pool=pool.servers)
    # As client_ids are randomized, we need to collect them before sending
    # the hash tables.
    client_ids = server._models['server']['aggregated_weights'] # Generated at client lvl
    # BEGINNING OF THE PREPROCESSING STAGE
    map_reduce_hash_tables(clients=pool.clients, server=pool.servers, aggregator=pool.aggregators)
    # print(server._models['server'].keys())
    print("END OF PREPROCESSING STAGE")
    # END OF PREPROCESSING STAGE
    print("STARTING THE TRAINIG STAGE")
    # BEGINNING OF THE TRAININ STAGE
    train_n_estimators(clients=pool.clients, server=pool.servers,
                    aggregator=pool.aggregators, total_estimators=total_estimators,
                    client_ids=client_ids)
    # END OF TRAINING STAGE
    # EVALUATE THE GLOBAL MODEL
    # On server's side
    server.map(evaluate_global_model, test_data=test_data)
    # On clients side
    clients.map(evaluate_global_model_clients_gbdt)
    # server.map(evaluate_server_global_model, test_data=test_data) # Future names aprox
    print("Finish script")

    elapsed_time = time.time()
    total_time = elapsed_time - start_time
    print("Total time: ", total_time)

if __name__ == '__main__':
    main()