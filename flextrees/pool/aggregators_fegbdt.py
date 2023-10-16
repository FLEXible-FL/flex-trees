import random
from flex.pool.decorators import aggregate_weights


@aggregate_weights
def aggregate_transition_step(aggregated_weights, *args, **kwargs):
    """Function that add into the 'aggregated_weights' from the
    aggregator the variable `aggregated_weights`.
    This variable may change depending when using it. For example:
    - Can "aggregate" the client_ids
    - Can "aggregate" multiple hash_tables
    - Can "aggregate" one hash table.
    This function doesn't really works as a aggregation function,
    it act's more as a transitional step needed for the communications
    for the preprocessing and the training stages.

    Args:
        aggregated_weights (list): List with collected weights

    Returns:
        list: List with the unchanged weights
    """
    return aggregated_weights

@aggregate_weights
def aggregate_hash_tables(aggregated_weights, *args, **kwargs):
    """Function that aggregate the hash tables onto one global
    hash table.

    Args:
        aggregated_weights (list): List containing the hash tables
        from the clients
    """
    global_hash_table = []
    for hash_table in aggregated_weights:
        global_hash_table.extend(hash_table)
    return global_hash_table

@aggregate_weights
def aggregate_hash_tables_deprecated(aggregated_hash_tables, *args, **kwargs):
    """Function that make the all reduce operation of the hash tables.
    With this function we create a matrix for each client with the similar instances ids from other
    clients to their instances. This function correspond to the lines 4-12 from the Algorithm 1 from
    the paper "Practical Federated Gradient Boosting Decision Trees".
    Args:
        global_hash_values (list): List containing the ids and the hash values of each instance of
        every client.
    """
    # aggregated_hash_tables = lista de tamaño n_clientes.
    # Cada elemento de la lista es la tabla de cada cliente.
    import time
    aggregated_weights = {}
    for i, client_hash in enumerate(aggregated_hash_tables):
        client_global_hash = []
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
        # Se guardan en un dict la tabla hash de cada cliente
        client_id_str = str(client_hash[0][0].split(',')[0])
        aggregated_weights[client_id_str] = client_global_hash
    # Keep client_global_hash to send each one to the corresponding client
    return aggregated_weights
