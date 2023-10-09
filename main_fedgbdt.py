import os
import time
from datetime import datetime
import argparse

from flex.data import FedDataDistribution, FedDatasetConfig
from flex.pool import FlexPool

from flextrees.datasets.tabular_datasets import ildp, adult, bank, credit2

from flextrees.pool import (
    init_server_model_gbdt,
    init_hash_tables,
    compute_hash_values,
    deploy_server_config_gbdt,
    collect_hash_tables,
    aggregate_hash_tables,
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

    available_datasets = [ildp, adult, bank, credit2]

    dataset = available_datasets[db]

    start_time = time.time()

    train_data, test_data = dataset(ret_feature_names=False, categorical=False)
    n_clients = n_clients
    if dist == 'iid':
        federated_data = FedDataDistribution.iid_distribution(centralized_data=train_data,
                                                            n_clients=n_clients)
    else:
        config_nidd = FedDatasetConfig(seed=0, n_clients=n_clients, replacement=False)

        federated_data = FedDataDistribution.from_config(centralized_data=train_data,
                                                            config=config_nidd)
    # Set server config
    # TODO: Add init_server_model_gbdt
    pool = FlexPool.client_server_architecture(federated_data, init_server_model_gbdt)

    clients = pool.clients
    aggregator = pool.aggregators
    server = pool.servers

    # Total number of estimators
    total_estimators = 2


    # Deploy clients config
    pool.servers.map(func=deploy_server_config_gbdt, dst_pool=pool.clients)
    # Preprocessing Stage
    pool.clients.map(func=init_hash_tables) # Init hash tables
    pool.clients.map(func=compute_hash_values) # Calculate the hash tables on the clients
    pool.aggregators.map(func=collect_hash_tables, dst_pool=pool.clients)
    pool.aggregators.map(func=aggregate_hash_tables)
    # TODO: Finish preprocessing stage
    print("Finish script")
    # Training Stage

    elapsed_time = time.time()
    total_time = elapsed_time - start_time
    print("Total time: ", total_time)

if __name__ == '__main__':
    main()