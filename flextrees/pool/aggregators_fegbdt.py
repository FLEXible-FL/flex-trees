import random
import numpy as np
from flex.pool.decorators import aggregate_weights


@aggregate_weights
def aggregate_hash_tables(aggregated_hash_tables, *args, **kwargs):
    """Function that make the all reduce operation of the hash tables.
    With this function we create a matrix for each client with the similar instances ids from other
    clients to their instances. This function correspond to the lines 4-12 from the Algorithm 1 from
    the paper "Practical Federated Gradient Boosting Decision Trees".
    Args:
        global_hash_values (list): List containing the ids and the hash values of each instance of
        every client.
    """
    client_global_hash = []
    for i, client_hash in enumerate(aggregated_hash_tables):
        for client_id_i, dict_i in client_hash:
            similar_instances_ids = {client_id_i:[]}
            for j, client_hash_j in enumerate(aggregated_hash_tables):
                if i == j:
                    continue
                id_options = {}
                for client_id_j, dict_j in client_hash_j:
                    counts = sum(
                        {k: 1 if dict_i.get(k) == dict_j.get(k) else 0 for k in dict_i.keys()}.values()
                    )
                    if id_options.get(counts):
                        id_options[counts].append(client_id_j)
                    else:
                        id_options[counts] = [client_id_j]
                    # Once all hash values have been processed, select one random
                    max_count = max(id_options.keys())
                    similar_id = id_options[max_count][random.randint(0, len(id_options[max_count])-1)] if len(
                        id_options[max_count]) > 0 else id_options[max_count]
                    similar_instances_ids[client_id_i].append(similar_id)
                # Append the similar instances ids to the client global hash table
                client_global_hash.append((client_id_i, similar_instances_ids))
    # Keep client_global_hash to send each one to the corresponding client
    return client_global_hash
