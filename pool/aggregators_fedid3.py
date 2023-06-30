from flex.pool.decorators import aggregate_weights


@aggregate_weights
def id3_agg(aggregated_weights_as_list, *args, **kwargs):
    """Function that implements the Federated ID3 aggregator

    Args:
        aggregated_weights_as_list (List): List of weights to
        aggregate.
    """
    ...