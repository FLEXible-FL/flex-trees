import os
import time
from datetime import datetime
import argparse

from flex.data import FedDataDistribution, FedDatasetConfig
from flex.pool import FlexPool

from flextrees.datasets.tabular_datasets import nursery, adult, car, credit2

from flextrees.pool import (
    init_server_model_rf,
    deploy_server_config_rf,
    deploy_server_model_rf,
    aggregate_trees_from_rf,
    evaluate_global_rf_model,
    evaluate_global_rf_model_at_clients,
    evaluate_local_rf_model_at_clients,
    train_rf,
    collect_clients_trees_rf,
    set_aggregated_trees_rf,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nclients', type=int, default=2, help='Number of clients')
    parser.add_argument('-d', '--dist', type=str, default='iid', help='Distribution (iid/niid)')
    parser.add_argument('-ne', '--exec', type=int, default=0, help='Number of execution')
    parser.add_argument('-db', '--database', type=int, default=0, help='Database to use: 0: Nursery, 1:Adult, 2: Car, 3:Credit2')

    args = parser.parse_args()
    n_clients = args.nclients
    dist = args.dist
    exec = args.exec
    db = args.database

    available_datasets = [nursery, adult, car, credit2]

    dataset = available_datasets[db]

    start_time = time.time()

    train_data, test_data = dataset(ret_feature_names=False, categorical=False)
    n_clients = n_clients
    if dist == 'iid':
        federated_nursery = FedDataDistribution.iid_distribution(centralized_data=train_data,
                                                            n_clients=n_clients)
    else:
        config_nidd = FedDatasetConfig(seed=0, n_clients=n_clients, replacement=False)

        federated_nursery = FedDataDistribution.from_config(centralized_data=train_data,
                                                            config=config_nidd)
    # Set server config
    pool = FlexPool.client_server_architecture(federated_nursery, init_server_model_rf)

    clients = pool.clients
    aggregator = pool.aggregators
    server = pool.servers

    # Deploy clients config
    pool.servers.map(func=deploy_server_config_rf, dst_pool=pool.clients)
    pool.clients.map(func=train_rf)
    pool.clients.map(func=evaluate_local_rf_model_at_clients)
    pool.aggregators.map(func=collect_clients_trees_rf, dst_pool=pool.clients)
    pool.aggregators.map(func=aggregate_trees_from_rf)
    pool.aggregators.map(func=set_aggregated_trees_rf, dst_pool=pool.servers)
    pool.servers.map(func=deploy_server_model_rf, dst_pool=pool.clients)
    pool.servers.map(func=evaluate_global_rf_model, test_data=test_data)
    pool.clients.map(func=evaluate_global_rf_model_at_clients)

    elapsed_time = time.time()
    total_time = elapsed_time - start_time
    print("Total time: ", total_time)

if __name__ == '__main__':
    main()