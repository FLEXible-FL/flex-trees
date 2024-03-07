"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import collections

from flex.pool.decorators import aggregate_weights


@aggregate_weights
def id3_aggegate_class_counts(aggregated_weights_as_list, *args, **kwargs):
    """Function to aggregate the class probabilities of a leaf node.
    # Arguments:
        params: Client class counts.
    # Returns:
        class_probs: Returns the class probabilities for the node.
    """
    agg_as_list = False
    if isinstance(aggregated_weights_as_list, list):
        aggregated_weights_as_list = aggregated_weights_as_list[0]
        agg_as_list = True
    # breakpoint()
    res = collections.Counter()
    for k, weights in aggregated_weights_as_list.items():
        res.update({k: weights})
        # res |= weights
    if agg_as_list:
        return -1 if sum(res.values()) == 0 else max(res.keys(), key=lambda x: res[x])

    key1 = max(res.keys(), key=lambda x: res[x])
    return {key1: res[key1]}


@aggregate_weights
def id3_aggregate_counts(aggregated_weights_as_list, *args, **kwargs):
    """Function to aggregate the information gain for the available features
    at the client to select the best that will be chosen to split.
    # Arguments:
        params: Client info gain for the remaining features.
    # Returns:
        feature: Returns the feature with the maximum information gain.
    """
    res = collections.Counter()
    for weights in aggregated_weights_as_list:
        res.update(weights)  # Keep .update for Python 3.8
    return -1 if sum(res.values()) == 0 else max(res.keys(), key=lambda x: res[x])


@aggregate_weights
def id3_aggregate_class_counts_sum(aggregated_wegiths_as_list, *args, **kwargs):
    """Function to aggregate the information gain for the available features
    at the client to select the best that will be chosen to split.
    # Arguments:
        params: Client info gain for the remaining features.
    # Returns:
        info_gain: Returns the info_gain for the feature indicated
    """
    res = collections.Counter()
    if isinstance(aggregated_wegiths_as_list, list):
        aggregated_wegiths_as_list = aggregated_wegiths_as_list[0]
    for k, v in aggregated_wegiths_as_list.items():
        res.update({k: v})
    return res
