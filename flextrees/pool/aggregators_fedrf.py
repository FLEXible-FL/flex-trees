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
