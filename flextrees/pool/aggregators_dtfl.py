from flex.pool.decorators import aggregate_weights


@aggregate_weights
def aggregate_dtfl(list_of_weights: list, *args, **kwargs):
    """Function that aggregate the rules from the clients
    """
    from dtfl.utils.utils_function_aggregator import generate_cs_dt_branches_from_list
    from dtfl.utils.branch_tree import TreeBranch
    from dtfl.utils.branch_tree_categorical import TreeBranchCategorical

    classes_ = set()
    features_ = set()
    classes_ |= {c for client in list_of_weights for c in client[1]}
    features_ |= {fe for client in list_of_weights for fe in client[2].columns if 'upper' or 'lower' in fe}
    classes_ = list(classes_)
    features_.remove('probas')
    features_ = list(set(features_)) # This param will be used in a future for VFL. Still need some coding to match the features correctly.
    client_cs = [cs[0] for cs in list_of_weights]
    tree_model_ = {client[3] for client in list_of_weights}
    assert len(tree_model_) == 1
    tree_model_ = TreeBranch if 'cart' in tree_model_ else TreeBranchCategorical
    return generate_cs_dt_branches_from_list(client_cs, classes_, tree_model_)

@aggregate_weights
def aggregate_dtfl_prunning(list_of_weights: list, *args, **kwargs):
    """Function that aggregate the rules from the clients
    """
    list_of_weights = [client for i, client in enumerate(list_of_weights) if i in kwargs['selected_indexes']]
    from dtfl.utils.utils_function_aggregator import generate_cs_dt_branches_from_list
    from dtfl.utils.branch_tree import TreeBranch
    from dtfl.utils.branch_tree_categorical import TreeBranchCategorical
    from dtfl.utils.branch_tree_mixed import TreeBranchMixed

    classes_ = set()
    features_ = set()
    classes_ |= {c for client in list_of_weights for c in client[1]}
    features_ |= {fe for client in list_of_weights for fe in client[2].columns if 'upper' or 'lower' in fe}
    classes_ = list(classes_)
    # breakpoint()
    features_.remove('probas')
    features_ = list(set(features_)) # This param will be used in a future for VFL. Still need some coding to match the features correctly.
    client_cs = [cs[0] for cs in list_of_weights]
    # breakpoint()
    tree_model_ = {client[3] for client in list_of_weights}
    
    try:
        assert len(tree_model_) == 1
    except AssertionError:
        print(f"Tree model: {tree_model_}")
        print(f"List of weights: {list_of_weights}")
        print(f"Selected indexes: {kwargs['selected_indexes']}")
        raise AssertionError
    # tree_model_ = TreeBranch if 'cart' in tree_model_ else TreeBranchCategorical # OLD
    if 'cart' in tree_model_:
        tree_model_ = TreeBranch
    elif 'id3' in tree_model_:
        tree_model_ = TreeBranchCategorical
    elif 'c45' in tree_model_:
        print("Using TreeBranchMixed")
        tree_model_ = TreeBranchMixed
    else:
        raise NotImplementedError(f"Tree model {tree_model_} not implemented.")
    return generate_cs_dt_branches_from_list(client_cs, classes_, tree_model_)

@aggregate_weights
def aggregate_client_dts(list_of_weights: list, *args, **kwargs):
    """Function that aggregate all the client trees to send them to the clients
    """
    return list_of_weights

@aggregate_weights
def aggregate_thresholds_and_select(list_of_weights: list, *args, **kwargs):
    """
    Function that select those trees that pass the threshold in both accuracy and f1.
    This function recieves a list with all the f1 and acc for each tree with the predictions
    for each test dataset for each client, and returns the indices of those that surpass
    the threshold given for both acc and macro f1.
    """
    acc_threshold = kwargs['acc_threshold']
    f1_threshold = kwargs['f1_threshold']
    func_str = kwargs['func_str']
    func_kwval = kwargs['func_kwargs']
    # print(f"Metrics at client level: {list_of_weights}")
    import numpy as np
    sum_list_of_weights = np.sum(np.array(list_of_weights), axis=0)/len(list_of_weights)
    acc_array = sum_list_of_weights[0]
    f1_array = sum_list_of_weights[1]
    def select_func_aggregation(func_str='percentile'):
        func_opts = {
            'percentile': (np.percentile, 'q'),
            'quantile': (np.quantile, 'q'),
            'mean': (np.mean, None),
            'median': (np.median, None)
        }
        return func_opts[func_str]
    func, func_kwargs = select_func_aggregation(func_str=func_str)
    print(f"Using {func_str} as threshold.")
    func_kwargs = {func_kwargs:func_kwval} if func_kwargs is not None else {}
    acc_threshold, f1_threshold = func(np.mean(np.array(list_of_weights), axis=0),
                                       axis=1, **func_kwargs)
    # acc_threshold, f1_threshold = np.percentile(np.mean(np.array(list_of_weights), axis=0), q=75, axis=1)  # noqa: E501
    # END FOR TESTING PURPOSES #
    selected_trees = np.where((acc_array >= acc_threshold) & (f1_array >= f1_threshold))[0]
    if len(selected_trees) < 1:
        """
        If no tree is selected, we select the best tree according to the accuracy threshold
        and the best tree according to the f1 threshold. Instead of using the last
        thresholds, we use a 98.9% of the original thresholds. This way, we can be sure
        that at least one tree will be selected.
        """
        # selected_trees = np.where((acc_array >= acc_threshold) | (f1_array >= f1_threshold))[0]
        f1_threshold = f1_threshold * 0.989
        acc_threshold = acc_threshold * 0.989
        selected_trees = np.where((acc_array >= acc_threshold) & (f1_array >= f1_threshold))[0]
    print(f"Number of selected trees: {len(selected_trees)}")
    return list(selected_trees)

@aggregate_weights
def aggregate_transfer_learning(list_of_weights: list, *args, **kwargs):
    """Function that select the best models to aggregate them into one
    Right now return all of them as the final model must be built first in
    order to optimize the build of the global tree.

    Args:
        list_of_weights (list): _description_

    Returns:
        _type_: _description_
    """
    import numpy as np
    print("transfer_agg")
    raise NotImplementedError("This function is not implemented yet.")
    return list(np.arange(len(list_of_weights)))
