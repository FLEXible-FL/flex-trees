import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.tree import DecisionTreeClassifier

from dtfl.utils.utils_trees import ID3Classifier, NewID3Classifier
from dtfl.utils.c45_tree import C45Tree

"""
Funciones para elegir aquellos nodos del árbol que obtienen mejor auc sobre los datos para elegir
esos nodos para generar las mejores reglas de cada cliente.
"""


def get_auc(Y, y_score, classes):
    """
    Función para calcular el auc de la rama sobre los datos.
    """
    y_test_binarize = np.array([[1 if i == c else 0 for c in classes] for i in Y])
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)


# def predict_with_included_trees(model,included_indexes,X): # OLD
def predict_with_included_trees(model, included_indexes, X, classes_):  # NEW
    """Function to predict all the X instances with a Sklearn Tree.
    Args:
        model (sklearn tree_ (DT)/ sklearn estimators_ (RF)): Tree/Forest to do the predictions with
        included_indexes (list of ints): If model is a RF, this mean those trees from RF that are included and must not
            be testes.
        X (np.array): Instances to predict
        classes_ (list): Classes avaiable in the problem. (Must be kept for the aggregation in the server).
    """
    predictions = []
    X = X[0] if isinstance(X, tuple) else X
    for inst in X:
        # predictions.append(predict_instance_with_included_tree(model,included_indexes,inst)) # OLD
        predictions.append(
            predict_instance_with_included_tree(model, included_indexes, inst, classes_)
        )  # NEW
    return np.array(predictions)


# def predict_instance_with_included_tree(model,included_indexes,inst): # OLD DEFINITION
def predict_instance_with_included_tree(model, included_indexes, inst, classes_):
    # v=np.array([0]*model.n_classes_) # OLD
    v = np.array([0] * len(classes_))  # NEW
    for i in range(len(model)):
        if model[i] in included_indexes:
            d = {c: 0 for c in classes_}
            local_classes = model[i].classes_
            local_pred = model[i].predict_proba(inst.reshape(1, -1))[0]
            # local_pred = model[i].predict_proba(inst)[0]
            for cl, j in zip(local_classes, range(len(local_classes))):
                d[cl] = local_pred[j]
            probas = np.array(
                list(d.values())
            )  # Al tratar con valores no-iid es necesario tener en el servidor
            v = v + probas
    return v / np.sum(v)


# def select_index(rf,current_indexes,validation_x,validation_y): # OLD
def select_index(rf, current_indexes, validation_x, validation_y, classes_):  # NEW
    options_auc = {}
    for i, tree in enumerate(rf):  # NEW
        if rf[i] in current_indexes:  # NEW AUX. Keeping this line until above line work
            continue
        predictions = predict_with_included_trees(
            rf, current_indexes + [tree], validation_x, classes_
        )  # NEW
        options_auc[i] = get_auc(validation_y, predictions, classes_)
    # In case options_auc is null, we get the first tree. Doing this while try to figure out why options_auc is empty.
    # if len(options_auc) == 0: # NEW. Commented till is solved
    # options_auc[0] = 0.1
    best_index = max(options_auc, key=options_auc.get)
    best_auc = options_auc[best_index]
    return best_auc, current_indexes + [best_index]


# def reduce_error_pruning(model,validation_x,validation_y,min_size): # OLD
def reduce_error_pruning(model, validation_x, validation_y, min_size, classes_):  # NEW
    """
    Función que coge todos los árboles de los clientes y elimina aquellos redundantes.
    """
    if isinstance(model[0], DecisionTreeClassifier):
        return reduce_error_pruning_sklearn(
            model, validation_x, validation_y, min_size, classes_
        )
    elif isinstance(model[0], ID3Classifier) or isinstance(model[0], NewID3Classifier):
        return reduce_error_pruning_id3(
            model, validation_x, validation_y, min_size, classes_
        )
    elif isinstance(model[0], C45Tree):
        return reduce_error_pruning_id3(
            model, validation_x, validation_y, min_size, classes_
        )
    else:
        raise NotImplementedError("No other tree model is available right now.")


def reduce_error_pruning_sklearn(model, validation_x, validation_y, min_size, classes_):
    """Function that deletes the redudant trees from a sklearn DecissionTree model.
    This function delete trees while the AUC is not decreasing.
    Args:
        model (list): List containing all the models to prune
        validation_x (np object): Training data
        validation_y (np object): Training labels
        min_size (int): Minimal size of the forest
        classes_ (list): List of all the classes in the problem. Important to
        use because of the non-IID distribution.

    Returns:
        List[int]: List containing the trees that will be kept
    """
    best_auc, current_indexes = select_index(
        model, [], validation_x, validation_y, classes_
    )
    assert min_size <= len(model)
    while len(current_indexes) <= len(model):
        new_auc, new_current_indexes = select_index(
            model, current_indexes, validation_x, validation_y, classes_
        )
        if new_auc <= best_auc and len(new_current_indexes) > min_size:
            break
        best_auc, current_indexes = new_auc, new_current_indexes
    return current_indexes


def reduce_error_pruning_id3(model, validation_x, validation_y, min_size, classes_):
    """Function that deletes the redudant trees.
    This function delete trees while the AUC is not decreasing.
    Args:
        model (list): List containing all the models to prune
        validation_x (np object): Training data
        validation_y (np object): Training labels
        min_size (int): Minimal size of the forest
        classes_ (list): List of all the classes in the problem. Important to
        use because of the non-IID distribution.

    Returns:
        List[int]: List containing the trees that will be kept
    """
    """
    best_auc,current_indexes = select_index(model,[],validation_x,validation_y, classes_)
    # while len(current_indexes) <= model.n_estimators: # TODO: Modificar para adaptar a len(models) o models[i]
    assert min_size <= len(model)
    while len(current_indexes) <= len(model):
        new_auc, new_current_indexes = select_index(model, current_indexes,
                                                    validation_x,validation_y, classes_)
        if new_auc <= best_auc and len(new_current_indexes) > min_size:
            break
        best_auc, current_indexes = new_auc, new_current_indexes
        # print(best_auc, current_indexes)
    # print(len(current_indexes))
    # print('Finish pruning')
    return current_indexes
    """
    return list(np.arange(len(model)))
