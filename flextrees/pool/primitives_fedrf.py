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
import random
from copy import deepcopy

import numpy as np
from flex.model import FlexModel

from flextrees.utils import GlobalRandomForest

from flex.pool.decorators import (
    collect_clients_weights,
    deploy_server_model,
    evaluate_server_model,
    init_server_model,
    set_aggregated_weights,
)


@init_server_model
def init_server_model_rf(config=None, *args, **kwargs):
    """Function to initialize the server model

    Args:
        dataset_features (list): List that contains the name of the features
        config (dict, optional): Dict that contains the configuration of the
        server model. Defaults to None.
    """
    from flex.model import FlexModel

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': 5,
                'n_estimators': 100,
            },
            'clients_params': {
                'max_depth': 5,
                'n_estimators': 100,
            }
        }

    server_flex_model['model'] = GlobalRandomForest(max_depth=config['server_params']['max_depth'], 
                                                    n_estimators=config['server_params']['n_estimators'])

    server_flex_model.update(config)

    return server_flex_model

@deploy_server_model
def deploy_server_config_rf(server_flex_model, *args, **kwargs):
    client_flex_model = FlexModel()

    client_flex_model['clients_params'] = deepcopy(server_flex_model['clients_params'])
    return client_flex_model

@deploy_server_model
def deploy_server_model_rf(server_flex_model, *args, **kwargs):
    client_flex_model = FlexModel()

    client_flex_model['global_model'] = deepcopy(server_flex_model['model'])
    return client_flex_model

@collect_clients_weights
def collect_clients_trees_rf(client_flex_model, *args, **kwargs):
    # Select random trees given the number of estimators to select
    # If the number of estimators is 0, then select all the trees
    nr_estimators = kwargs['nr_estimators']
    return (
        np.random.choice(
            client_flex_model['model'].estimators_,
            nr_estimators,
            replace=False,
        )
        if nr_estimators > 0
        else client_flex_model['model'].estimators_
    )

@set_aggregated_weights
def set_aggregated_trees_rf(server_flex_model, aggregated_weights, *args, **kwargs):
    server_flex_model['model'].estimators_ = deepcopy(aggregated_weights)
    return server_flex_model

def train_rf(client_flex_model, client_data, *args, **kwargs):
    from sklearn.ensemble import RandomForestClassifier
    X_train, y_train = client_data.to_numpy()
    clf = RandomForestClassifier(max_depth=client_flex_model['clients_params']['max_depth'], 
                                n_estimators=client_flex_model['clients_params']['n_estimators'])
    clf.fit(X_train, y_train)
    rf_classes_ = clf.classes_
    # Set the classes to the estimators as single trees will have [0, 1] as classes
    # in a binary problem, but the global model will have the original classes.
    # This is needed to be able to evaluate the global model on the client, becuase
    # if not, the prediction with the global model will predict [0, 1] and the classes
    # could be [1, 2] for example, or [2, 4]. This may be a bug in sklearn.
    for estimator in clf.estimators_: # Set the classes to the estimators
        estimator.classes_ = rf_classes_
    client_flex_model['model'] = clf
    return client_flex_model

@evaluate_server_model
def evaluate_global_rf_model(server_flex_model, test_data, *args, **kwargs):
    """Evaluate global model on the server with a global test set.

    Args:
        server_flex_model (FlexModel): Server Flex Model.
        X (ArrayLike): Array with the data to evaluate.
        y (ArrayLike): Labels of the data to evaluate.
    """
    X, y = test_data.to_numpy()
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    preds_rf = server_flex_model['model'].predict(X)
    acc, f1, report = accuracy_score(y, preds_rf), f1_score(y, preds_rf, average='macro'), classification_report(y, preds_rf)
    print("Results on server: ")
    print(f"Accuracy: {acc}, F1: {f1}")
    print(report)

def evaluate_global_rf_model_at_clients(
    client_flex_model, client_data, *args, **kwargs
):
    print("Evaluating global model on client.")
    from sklearn import metrics
    X_test, y_test = client_data.to_numpy()
    clf = client_flex_model['global_model']
    y_pred = clf.predict(X_test)
    if 'client_id' not in client_flex_model.keys():
        client_flex_model['client_id'] = f"client_{random.randint(a=10, b=10000)}" # Create a random ID

    client_id = client_flex_model['client_id']

    acc, f1, report = metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='macro'), metrics.classification_report(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    print("Results on client: ", client_id)
    print(f"Accuracy: {acc}, F1: {f1}")
    print(f"Auc: {auc}")
    print(f"Classification report: {report}")

def evaluate_local_rf_model_at_clients(
    client_flex_model, client_data, *args, **kwargs
):
    from sklearn import metrics

    X_test, y_test = client_data.to_numpy()
    clf = client_flex_model['model']
    y_pred = clf.predict(X_test)

    if 'client_id' not in client_flex_model.keys():
        client_flex_model['client_id'] = f"client_{random.randint(a=10, b=10000)}" # Create a random ID

    client_id = client_flex_model['client_id']

    acc, f1, report = metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='macro'), metrics.classification_report(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    print("Local results on client: ", client_id)
    print(f"Accuracy: {acc}, F1: {f1}")
    print(f"Auc: {auc}")
    print(f"Classification report: {report}")
    
