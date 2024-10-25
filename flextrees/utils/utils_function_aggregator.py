import numpy as np
import pandas as pd


def generate_cs_dt_branches_from_list(client_cs, classes_, tree_model, threshold=3000):
    """Function that generate a global ConjuctionSet, a GlobalTree and the branches
    associated to the tree in the server node.
    """
    from dtfl.utils.ConjunctionSet import ConjunctionSet

    cs = ConjunctionSet(
        filter_approach="entropy",
        amount_of_branches_threshold=threshold,
        feature_names=[],
        personalized=False,
    )
    cs.aggregate_branches(client_cs, classes_)
    cs.buildConjunctionSet()
    print(f"Conjunction set length: {len(cs.conjunctionSet)}")
    cs.conjunctionSet = delete_duplicated_rules(cs.conjunctionSet)
    print(f"Conjunction set length after removing duplicates: {len(cs.conjunctionSet)}")
    branches_df_aggregator = cs.get_conjunction_set_df().round(decimals=5)

    probabilities = branches_df_aggregator["branch_probability"].to_list()
    # new_probas = [x for x in probabilities]
    new_probas = list(probabilities)
    total_probas = sum(new_probas)
    # self._save_rules(client_cs, cs, round_number)
    branches_df_aggregator["branch_probability"] = branches_df_aggregator[
        "branch_probability"
    ].map(lambda x: x / total_probas)
    if pd.isna(branches_df_aggregator).any().any():
        import time
        time.sleep(5)
        print("Before fillna")
        print(branches_df_aggregator)
        branches_df_aggregator = branches_df_aggregator.fillna(np.inf)
        print("After fillna")
        print(branches_df_aggregator)
        time.sleep(6)
    else:
        print(f"branches df aggreagator is not null: {branches_df_aggregator}")
        branches_df_aggregator.to_csv("branches_df_aggregator.csv")
    new_df_dict = {
        col: branches_df_aggregator[col].values
        for col in branches_df_aggregator.columns
    }
    new_dt_model = tree_model([True] * len(branches_df_aggregator), classes_)
    new_dt_model.split(new_df_dict)
    return [cs, new_dt_model, branches_df_aggregator]


def delete_duplicated_rules(rules_dataset):
    rules = {str(rule): rule for rule in rules_dataset}
    return list(rules.values())
