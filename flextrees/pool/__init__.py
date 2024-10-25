"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

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
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Primitive for ID3
from flextrees.pool.aggregators_fedid3 import id3_aggregate_counts
from flextrees.pool.aggregators_fedid3 import id3_aggegate_class_counts
from flextrees.pool.aggregators_fedid3 import id3_aggregate_class_counts_sum
from flextrees.pool.primitives_fedid3 import build_id3
from flextrees.pool.primitives_fedid3 import init_server_model_id3
from flextrees.pool.primitives_fedid3 import deploy_server_config_id3
from flextrees.pool.primitives_fedid3 import deploy_server_model_id3
from flextrees.pool.primitives_fedid3 import deploy_node_id3
from flextrees.pool.primitives_fedid3 import collect_clients_counts_id3
from flextrees.pool.primitives_fedid3 import collect_clients_class_counts_id3
from flextrees.pool.primitives_fedid3 import calculate_counts
from flextrees.pool.primitives_fedid3 import set_aggregated_id3
from flextrees.pool.primitives_fedid3 import evaluate_id3_model
from flextrees.pool.primitives_fedid3 import evaluate_global_model_clients
# Primitive for RF
from flextrees.pool.aggregators_fedrf import aggregate_trees_from_rf
from flextrees.pool.primitives_fedrf import init_server_model_rf
from flextrees.pool.primitives_fedrf import deploy_server_config_rf
from flextrees.pool.primitives_fedrf import deploy_server_model_rf
from flextrees.pool.primitives_fedrf import collect_clients_trees_rf
from flextrees.pool.primitives_fedrf import set_aggregated_trees_rf
from flextrees.pool.primitives_fedrf import evaluate_global_rf_model
from flextrees.pool.primitives_fedrf import evaluate_global_rf_model_at_clients
from flextrees.pool.primitives_fedrf import evaluate_local_rf_model_at_clients
from flextrees.pool.primitives_fedrf import train_rf
# Primitive for FBDT
from flextrees.pool.aggregators_fegbdt import aggregate_transition_step
from flextrees.pool.aggregators_fegbdt import aggregate_hash_tables
from flextrees.pool.primitives_fedgbdt import init_server_model_gbdt
from flextrees.pool.primitives_fedgbdt import init_hash_tables
from flextrees.pool.primitives_fedgbdt import compute_hash_values
from flextrees.pool.primitives_fedgbdt import deploy_server_config_gbdt
from flextrees.pool.primitives_fedgbdt import collect_hash_tables
from flextrees.pool.primitives_fedgbdt import collect_client_gradients_hessians_by_idx
from flextrees.pool.primitives_fedgbdt import collect_clients_ids
from flextrees.pool.primitives_fedgbdt import collect_last_tree_trained
from flextrees.pool.primitives_fedgbdt import set_ids_clients_into_server
from flextrees.pool.primitives_fedgbdt import set_hash_tables_to_server
from flextrees.pool.primitives_fedgbdt import send_hash_table_to_server
from flextrees.pool.primitives_fedgbdt import set_last_tree_trained_to_server
from flextrees.pool.primitives_fedgbdt import deploy_hash_table_to_clients
from flextrees.pool.primitives_fedgbdt import deploy_last_clf
from flextrees.pool.primitives_fedgbdt import get_client_hash_tables
from flextrees.pool.primitives_fedgbdt import map_reduce_hash_tables
from flextrees.pool.primitives_fedgbdt import find_instance_highest_count_hash_values
from flextrees.pool.primitives_fedgbdt import deploy_client_j_has_table_to_client
from flextrees.pool.primitives_fedgbdt import update_gradients_hessians_local_values
from flextrees.pool.primitives_fedgbdt import train_single_tree_at_client
from flextrees.pool.primitives_fedgbdt import clients_add_last_tree_trained_to_estimators
from flextrees.pool.primitives_fedgbdt import get_client_gradients_hessians_by_idx
from flextrees.pool.primitives_fedgbdt import client_global_gh_update
from flextrees.pool.primitives_fedgbdt import evaluate_global_model
from flextrees.pool.primitives_fedgbdt import evaluate_global_model_clients_gbdt
from flextrees.pool.primitives_fedgbdt import train_n_estimators

# Primitives and aggregation functions for DTFL
from flextrees.pool.primitives_dtfl import init_server_model_dtfl
from flextrees.pool.primitives_dtfl import deploy_server_model_dtfl
from flextrees.pool.primitives_dtfl import train_dtfl
from flextrees.pool.primitives_dtfl import collect_clients_weights_dtfl
from flextrees.pool.primitives_dtfl import set_aggregated_weights_dtfl
from flextrees.pool.primitives_dtfl import set_local_trees_to_server
from flextrees.pool.primitives_dtfl import evaluate_server_model_dtfl
from flextrees.pool.primitives_dtfl import get_classes_branches
from flextrees.pool.primitives_dtfl import send_all_trees_to_client
from flextrees.pool.primitives_dtfl import deploy_global_model_dtfl
from flextrees.pool.primitives_dtfl import collect_clients_trees_dtfl
from flextrees.pool.primitives_dtfl import evaluate_global_model_dtfl_on_client
from flextrees.pool.primitives_dtfl import evaluate_global_trees
from flextrees.pool.primitives_dtfl import collect_local_evaluations_from_clients_dtfl
from flextrees.pool.primitives_dtfl import set_selected_trees_to_server_dtfl
from flextrees.pool.aggregators_dtfl import aggregate_dtfl
from flextrees.pool.aggregators_dtfl import aggregate_dtfl_prunning
from flextrees.pool.aggregators_dtfl import aggregate_client_dts
from flextrees.pool.aggregators_dtfl import aggregate_thresholds_and_select
from flextrees.pool.aggregators_dtfl import aggregate_transfer_learning


from flextrees.pool.primitives_fedgbdt import preprocessing_stage
# Functions from pool_functions
from flextrees.pool.pool_functions import select_client_by_id_from_pool
from flextrees.pool.pool_functions import select_client_neq_id_from_pool
