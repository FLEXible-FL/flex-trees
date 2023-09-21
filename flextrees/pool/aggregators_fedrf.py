import numpy as np
from flex.pool.decorators import aggregate_weights


@aggregate_weights
def aggregate_trees_from_rf(aggregated_trees, *args, **kwargs):
    """Function to aggregate the trees from the clients.

    Args:
        aggregated_trees (List): List of trees to aggregate.

    Returns:
        List: List with the aggregated trees.
    """
    # Make the aggregator to append all the trees in a list
    aggregated_trees = [tree for trees in aggregated_trees for tree in trees]
    return aggregated_trees