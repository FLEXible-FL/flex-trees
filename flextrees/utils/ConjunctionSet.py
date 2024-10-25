import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.distributions.empirical_distribution import ECDF

from dtfl.utils.Branch import Branch
from dtfl.utils.branch_categorical import BranchCategorical
from dtfl.utils.branch_mixed import BranchMixed
from dtfl.utils.pruningFunctions import *
from dtfl.utils.utils_trees import ID3Classifier
from dtfl.utils.c45_tree import C45Tree


class ConjunctionSet:
    def __init__(
        self,
        feature_names=None,
        original_data=None,
        pruning_x=None,
        pruning_y=None,
        model=None,
        feature_types=None,
        amount_of_branches_threshold=None,
        estimators=None,
        filter_approach="entropy",
        exclusion_starting_point=2,
        minimal_forest_size=3,
        exclusion_threshold=0.9,
        personalized=True,
    ):
        print(f"Estimators: {estimators}")
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.model = model  # Lista de árboles
        self.model_estimators = estimators  # Solo sirve para mantenerla el total de árboles que se utilizan aquí.
        self.feature_names = feature_names
        self.exclusion_threshold = exclusion_threshold
        self.filter_approach = filter_approach
        self.number_of_branches_per_iteration = []
        self.exclusion_starting_point = exclusion_starting_point
        all_clases = []
        self.ecdf_dict = None
        self._personalized = personalized
        self.conjunctionSet = None
        self.branches_lists = []
        # Variable para mantener las reglas individuales de todos los clientes.
        # Se añadirán estas reglas a las agregadas.
        self.__global_CS = []
        if (
            self.model is not None
            and pruning_x is not None
            and pruning_y is not None
            and estimators is not None
        ):
            for clf in model:
                all_clases += list(clf.classes_)
            self.classes_ = list(set(all_clases))
            self.relevant_indexes = reduce_error_pruning(
                self.model, pruning_x, pruning_y, minimal_forest_size, self.classes_
            )
            # print('SE MUESTRA RELEVANT_INDEXES: ')
            # print(self.relevant_indexes)
            self.feature_types = feature_types
            self.feature_model = self._get_features_tree_to_features_name(
                feature_names, model[0]
            )
            # print(self.feature_model)
            self.set_ecdf(
                original_data
            )  # ARREGLAR A PARTIR DE AQUÍ YA QUE USA LOS DATOS ORIGINALES (¿¿USAR MEJOR LOS
            # DATOS DE VALIDACIÓN??)
            # Se necesitan los datos originales para generar los Branches, así que una opción es generar en cada cliente
            # los branches para así generar en el servidor el árbol final.
            # self.get_ranges(original_data)
            self.generateBranches()
            self.calculate_branches_probability()
            # self.print_branches_probability()
            # self.get_associative_leaves(np.concatenate((original_data,pruning_x)))
            self.buildConjunctionSet()

    def generateBranches(self):
        # trees = [estimator.tree_ for estimator in self.model]
        trees = [estimator for estimator in self.model]
        self.branches_lists = [
            self.get_tree_branches(estimator, i, type(estimator))
            for i, estimator in enumerate(trees)
            if i in self.relevant_indexes
        ]

        for list_indx, branch_list in enumerate(self.branches_lists):
            for leaf_index, branch in enumerate(branch_list):
                # print('here')
                branch.calculate_branch_probability_by_ecdf_new(self.ecdf_dict)
                # print(branch.get_branch_probability())
                branch.leaves_indexes = [str(list_indx) + "_" + str(leaf_index)]

    def get_tree_branches(self, tree_, tree_index, model_type):
        """Function to generatre rules from a tree

        Args:
            tree_ (sklearn.tree): Tree from a ID3Classifier
            tree_index (int): tree
            model_type (str): type of the tree_, to process the tree_ correctly.

        Returns:
            list: list with the branches generated
        """
        if model_type == DecisionTreeClassifier:
            print("AQUÍr")
            return self.get_tree_branches_cart(tree_.tree_, tree_index)
        elif model_type == ID3Classifier:
            return self.get_tree_branches_id3(tree_, tree_index)
        elif model_type == C45Tree:
            print("C45Tree")
            return self.get_tree_branches_c45(tree_, tree_index)
        else:
            raise ValueError("This model is not available yet.")

    def get_tree_branches_cart(self, tree_, tree_index):
        """Function to generatre rules from a DecisionTreeClassifier

        Args:
            tree_ (sklearn.tree): Tree from a DecisionTreeClassifier
            tree_index (int): tree

        Returns:
            list: list with the branches generated
        """
        
        leaf_indexes = [
            i
            for i in range(tree_.node_count)
            if tree_.children_left[i] == -1 and tree_.children_right[i] == -1
        ]
        branches = [
            self.get_branch_from_leaf_index(tree_, leaf_index)
            for leaf_index in leaf_indexes
        ]
        return branches

    def get_tree_branches_id3(self, tree_, tree_index=0):
        """Function to generatre rules from a ID3Classifier

        Args:
            tree_ (sklearn.tree): Tree from a ID3Classifier
            tree_index (int): tree

        Returns:
            list: list with the branches generated
        """
        # First step: Search for leaf nodes.
        leaf_nodes = tree_.leaf_nodes
        return [self.get_branch_from_leaf_node(node, tree_) for node in leaf_nodes]
        # return branches

    def get_tree_branches_c45(self, tree_, tree_index=0):
        """Function to generatre rules from a C45Tree

        Args:
            tree_ (sklearn.tree): Tree from a C45Tree
            tree_index (int): tree

        Returns:
            list: list with the branches generated
        """
        # First step: Search for leaf nodes.
        leaf_nodes = tree_.leaf_nodes
        branches = [self.get_branch_from_leaf_node_c45(node, tree_) for node in leaf_nodes]
        branches = [branch for branch in branches if branch is not None]
        return branches

    def get_branch_from_leaf_node_c45(self, leaf_node, root):
        sum_of_probas = np.sum(
            leaf_node.class_idx
        ) # Equal to tree_.value[leaf_index][0]
        label_probas = [
            i / sum_of_probas for i in leaf_node.class_idx
        ] # Equal to label probas in get_branch_from_leaf_index
        new_branch = BranchMixed(
            self.feature_names,
            self.feature_types,
            self.classes_,
            label_probas=label_probas,
            number_of_samples=leaf_node.n_node_samples,
        ) # Initialize branch
        # Get path to root node
        stack = root.reach_root_node(leaf_node)
        # Get the root path, which is the path from the root to the leaf node.
        # In the root path we don't take the leaf node, that's why we use len(stack) - 1
        # In the condicion of the stack we add the label probabilities for each node.
        # breakpoint()
        if len(stack) > 1:
            root_path = [stack[i] for i in range(0, len(stack) - 1)]
            # breakpoint()
            for feature, value, feature_type, child_type in root_path:
                # print(f"Adding branch condition. Feature: {feature}, value: {value}")
                if feature_type == 'discrete':
                    new_branch.add_condition(int(feature), value)
                else:
                    if child_type == 'left':
                        new_branch.add_condition(int(feature), value, 'upper')
                    elif child_type == 'right':
                        new_branch.add_condition(int(feature), value, 'lower')
                    elif child_type == 'root':
                        new_branch.add_condition(int(feature), value, 'upper')
            print(new_branch)
            return new_branch
        else:
            return None

    def get_branch_from_leaf_node(self, leaf_node, root):
        sum_of_probas = np.sum(
            leaf_node.class_idx
        )  # Equal to tree_.value[leaf_index][0]
        label_probas = [
            i / sum_of_probas for i in leaf_node.class_idx
        ]  # Equal to label probas in get_branch_from_leaf_index
        new_branch = BranchCategorical(
            self.feature_names,
            self.feature_types,
            self.classes_,
            label_probas=label_probas,
            number_of_samples=leaf_node.n_node_samples,
        )  # Initialize branch
        # Get path to root node
        stack = root.reach_root_node(leaf_node)
        # root.print_tree()
        root_path = [(stack[i], stack[i + 1]) for i in range(0, len(stack) - 1, 2)]
        # root_path = list(stack)
        # print(root_path)
        # print(stack)
        # print(f"Label probas: {label_probas}")
        for feature, value in root_path:
            # print(f"Adding branch condition. Feature: {feature}, value: {value}")
            new_branch.add_condition(int(feature), value)
        # print(f"Node first node: {leaf_node}")
        # print(f"Branch generated: {new_branch}")
        return new_branch

    def get_branch_from_leaf_index(self, tree_, leaf_index):
        sum_of_probas = np.sum(tree_.value[leaf_index][0])
        label_probas = [i / sum_of_probas for i in tree_.value[leaf_index][0]]
        new_branch = Branch(
            self.feature_names,
            self.feature_types,
            self.classes_,
            label_probas=label_probas,
            number_of_samples=tree_.n_node_samples[leaf_index],
        )  # initialize branch
        # breakpoint()
        node_id = leaf_index
        while node_id:  # iterate over all nodes in branch
            ancesor_index = np.where(tree_.children_left == node_id)[
                0
            ]  # assuming left is the default for efficiency purposes
            bound = "upper"
            if len(ancesor_index) == 0:
                bound = "lower"
                ancesor_index = np.where(tree_.children_right == node_id)[0]
            feature = tree_.feature[ancesor_index[0]]
            threshold = tree_.threshold[ancesor_index[0]]
            # new_branch.addCondition(tree_.feature[ancesor_index[0]], tree_.threshold[ancesor_index[0]], bound)
            new_branch.add_condition(feature, threshold, bound)
            node_id = ancesor_index[0]
        # print(f"Branch generated: {new_branch.toString()}")
        return new_branch

    def buildConjunctionSet(self):
        conjunctionSet = self.branches_lists[0]
        excluded_branches = []
        self.__global_CS.extend(conjunctionSet)
        # Si se utiliza un Random Forest entrará en el for, buscando ampliar el conjunto de reglas. Por ahora
        # solo se obtiene del árbol de decisión local del cliente.
        if isinstance(self.model_estimators, RandomForestClassifier):
            print("Aggregating rules from forest in client")
            for i, branch_list in enumerate(self.branches_lists[1:]):
                print(
                    "Iteration "
                    + str(i + 1)
                    + ": "
                    + str(len(conjunctionSet))
                    + " conjunctions"
                )
                filter = False if i == len(self.branches_lists[i + 1 :]) else True
                # filter = False
                if len(conjunctionSet) > 0:
                    conjunctionSet = self.merge_branch_with_conjunctionSet(
                        branch_list, conjunctionSet, filter=filter
                    )
                else:
                    conjunctionSet = branch_list
                if (
                    i >= self.exclusion_starting_point
                    and len(conjunctionSet) > 0.8 * self.amount_of_branches_threshold
                ):
                    (
                        conjunctionSet,
                        this_iteration_exclusions,
                    ) = self.exclude_branches_from_cs(
                        conjunctionSet, self.exclusion_threshold
                    )
                    excluded_branches.extend(this_iteration_exclusions)
            # print(conjunctionSet)
            self.conjunctionSet = conjunctionSet  # + excluded_branches
        else:  #  isinstance(self.model_estimators, DecisionTreeClassifier): # DecisionTreeClassifier & ID3Classifier & C45Tree
            for i, branch_list in enumerate(self.branches_lists[1:]):
                print(
                    "Iteration "
                    + str(i + 1)
                    + ": "
                    + str(len(conjunctionSet))
                    + " conjunctions"
                )
                print('\nLas reglas actuales son: ')
                # breakpoint()
                # for branch in conjunctionSet:
                #     print(branch.toString())
                # print('\nLas reglas que se van a juntar son las siguientes: \n')
                # for branch in branch_list:
                #     print(branch.toString())
                filter = False if i == len(self.branches_lists[i + 1 :]) else True
                # filter = False
                if len(conjunctionSet) > 0:
                    # print('Entro aquí')
                    conjunctionSet = self.merge_branch_with_conjunctionSet(
                        branch_list, conjunctionSet, filter=filter
                    )
                else:
                    conjunctionSet = branch_list
                # conjunctionSet = self.remove_duplicate_branches(conjunctionSet)
                # print('\n\nEn la ronda ' + str(i) + ' hay un total de ' + str(len(conjunctionSet)) + ' reglas.')
                # print('Las reglas en la iteración son: ')
                # for branch in conjunctionSet:
                #     print(branch.toString())
                # print('\n')
                # print('i='+str(i))
                if (
                    i >= self.exclusion_starting_point
                    and len(conjunctionSet) > 0.8 * self.amount_of_branches_threshold
                ):
                    (
                        conjunctionSet,
                        this_iteration_exclusions,
                    ) = self.exclude_branches_from_cs(
                        conjunctionSet, self.exclusion_threshold
                    )
                    excluded_branches.extend(this_iteration_exclusions)
        # print('Excluded_Branches: ')
        # print(excluded_branches)
        if self.model_estimators is not None and (
            isinstance(self.model_estimators, DecisionTreeClassifier)
            or isinstance(self.model_estimators, ID3Classifier)
            or isinstance(self.model_estimators, C45Tree)
        ):
            self.conjunctionSet = conjunctionSet + excluded_branches
            # self.conjunctionSet = conjunctionSet + excluded_branches + self.__global_CS
            # self.conjunctionSet = conjunctionSet + self.__global_CS
            self.conjunctionSet = self.remove_duplicate_branches(self.conjunctionSet)
        else:
            self.conjunctionSet = conjunctionSet + excluded_branches
            # self.conjunctionSet = conjunctionSet + excluded_branches + self.__global_CS
            # self.conjunctionSet = conjunctionSet + self.__global_CS
            self.conjunctionSet = self.remove_duplicate_branches(self.conjunctionSet)
        # print('Final CS size: ' + str(len(self.conjunctionSet)))

    def exclude_branches_from_cs(self, cs, threshold):
        filtered_cs = []
        excludable_branches = []
        for branch in cs:
            if branch.is_excludable_branch(threshold):
                excludable_branches.append(branch)
            else:
                filtered_cs.append(branch)
        return filtered_cs, excludable_branches

    def filter_conjunction_set(self, cs):
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        if self.filter_approach == "probability":
            branches_metrics = [
                b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs
            ]
        elif self.filter_approach == "number_of_samples":
            branches_metrics = [b.number_of_samples for b in cs]
        elif self.filter_approach == "probability_entropy":
            branches_metrics = [
                b.calculate_branch_probability_by_ecdf(self.ecdf_dict)
                * (1 - entropy(b.label_probas))
                for b in cs
            ]
        elif self.filter_approach == "entropy":
            branches_metrics = [-entropy(b.label_probas) for b in cs]
        elif self.filter_approach == "range":
            branches_metrics = [
                b.calculate_branch_probability_by_range(self.ranges) for b in cs
            ]
        # breakpoint()
        threshold = sorted(branches_metrics, reverse=True)[
            self.amount_of_branches_threshold - 1
        ]
        return [b for b, metric in zip(cs, branches_metrics) if metric >= threshold][
            : self.amount_of_branches_threshold
        ]

    def merge_branch_with_conjunctionSet(
        self, branch_list, conjunctionSet, filter=True
    ):
        new_conjunction_set = []
        for b1 in conjunctionSet:
            new_conjunction_set.extend(
                [
                    b1.merge_branch(b2, self._personalized)
                    for b2 in branch_list
                    if b1.contradict_branch(b2) is False
                ]
            )
        # print(branch_list)
        # print(conjunctionSet)
        # print('Nuveo conjuctionset')
        # print(new_conjunction_set)
        if filter:
            # print('number of branches before filterring: '+str(len(new_conjunction_set)))
            new_conjunction_set = self.filter_conjunction_set_aggregator(
                new_conjunction_set
            )
            # print('number of branches after filterring: ' + str(len(new_conjunction_set)))
        self.number_of_branches_per_iteration.append(len(new_conjunction_set))
        return new_conjunction_set

    def merge_branch_with_conjunctionSet_old(
        self, branch_list, conjunctionSet, filter=True
    ):
        new_conjunction_set = []
        for b1 in conjunctionSet:
            # self.__global_CS.append(b1)
            for b2 in branch_list:
                self.__global_CS.append(b2)
                if b1.contradict_branch(b2) is False:
                    new_conjunction_set.append(b1.merge_branch(b2, self._personalized))
            # new_conjunction_set.extend([b1.merge_branch(b2, self._personalized) for b2 in branch_list if b1.contradict_branch(b2) == False])
        if filter:
            # print('number of branches before filterring: '+str(len(new_conjunction_set)))
            new_conjunction_set = self.filter_conjunction_set_aggregator(
                new_conjunction_set
            )
            # print('number of branches after filterring: ' + str(len(new_conjunction_set)))
        self.number_of_branches_per_iteration.append(len(new_conjunction_set))
        return new_conjunction_set

    def get_conjunction_set_df(self):
        return (
            pd.DataFrame(
                [b.get_branch_dict(self.ecdf_dict) for b in self.conjunctionSet]
            )
            if self.ecdf_dict is not None
            else pd.DataFrame(
                [b.get_branch_dict_new(None) for b in self.conjunctionSet]
            )
        )

    """
    def predict(self, X):
        predictions = []
        for inst in X:
            for conjunction in self.conjunctionSet:
                if conjunction.containsInstance(inst):
                    predictions.append(self.classes_[conjunction.getLabel()])
        return predictions
    

    def get_instance_branch(self, inst):
        for conjunction in self.conjunctionSet:
            if conjunction.containsInstance(inst):
                return conjunction
    """

    def set_ecdf(self, data):
        # self.ecdf_dict={i:ECDF(data.transpose().T[i])for i in range(len(self.feature_names))} # OLD
        self.ecdf_dict = {
            i: ECDF(data.transpose()[i]) for i in range(len(self.feature_names))
        }  # NEW
        # El cambio se hace porque con data.transpose().T se obtiene de nuevo data, y por tanto no se podría
        # realizar el for que se está aplicando, a no ser que se tuvieran el mismo número de características
        # que de instancias para poder recorrer correctamente el for.

    """
    def group_by_label_probas(self, conjunctionSet):
        probas_hashes = {}
        for i, b in enumerate(conjunctionSet):
            probas_hash = hash(tuple(b.label_probas))
            if probas_hash not in probas_hashes:
                probas_hashes[probas_hash] = []
            probas_hashes[probas_hash].append(i)
        return probas_hashes
    """

    def get_ranges(self, original_data):
        self.ranges = [max(v) - min(v) for v in original_data.transpose()]

    #################################### MÉTODOS HECHOS POR MÍ ####################################
    """
    Dado que se hará la agregación totalmente en el servidor, y no se pueden disponer de datos, las funciones que se
    utilizarán sólo podrán ser utilizadas por el agregador sin tener en cuenta ninguna función de las ya declaradas
    que en el trazo de ejecución utilice cualquier dato de entrenamiento.
    """

    def remove_duplicate_branches(self, conjunctionSet):
        if not len(conjunctionSet):
            return []
        # print('Estoy aquí eliminando duplicados.')
        # print(f"Longitud de cs antes de entrar en la función: {len(conjunctionSet)}")
        cs = {}
        for branch in conjunctionSet:
            if branch.str_branch() not in cs:
                cs[branch.str_branch()] = branch
            else:
                # Get branch probabilities and number of samples affected
                probas_cs1 = np.array(branch.label_probas)
                # probas_cs1 = branch.get_branch_probability()
                ns_cs1 = branch.number_of_samples
                probas_cs2 = np.array(cs[branch.str_branch()].label_probas)
                # probas_cs2 = cs[branch.str_branch()].get_branch_probability()
                ns_cs2 = cs[branch.str_branch()].number_of_samples
                total_ns_cs = ns_cs1 + ns_cs2
                # Calculate new probas
                new_probas = list(
                    probas_cs1 * (ns_cs1 / total_ns_cs)
                    + probas_cs2 * (ns_cs2 / total_ns_cs)
                )
                # new_probas = probas_cs1 + probas_cs2 / 2
                new_number_of_samples = (
                    branch.number_of_samples + cs[branch.str_branch()].number_of_samples
                )
                cs[branch.str_branch()].number_of_samples = new_number_of_samples
                # Set new branch
                cs[branch.str_branch()].label_probas = new_probas
        cs_to_list = list(cs.values())
        # breakpoint()
        # print(f"Longitud de cs después de entrar en la función: {len(cs_to_list)}")
        return cs_to_list

    def get_branches_list(self):
        """
        Nuevo método. Se devuelve el atributo self.branches_lists de cada cliente para poder utilizar las funciones
        de las que se disponen en la clase.
        """
        return self.branches_lists

    def calculate_branches_probability(self):
        assert self.ecdf_dict is not None
        if self.conjunctionSet is None:
            for branch in self.branches_lists[0]:
                branch.calculate_branch_probability_by_ecdf(ecdf=self.ecdf_dict)
        else:
            for branch in self.conjunctionSet:
                branch.calculate_branch_probability_by_ecdf(ecdf=self.ecdf_dict)

    def aggregate_branches(self, list_cs, global_classes):
        """
        Método que recibe de los clientes todos los branches_lists y hace un merge en 1 solo.
        """
        self.branches_lists = []
        self.classes_ = global_classes
        for cs in list_cs:
            if len(cs) == 1:
                self.branches_lists.append(cs[0])
            else:
                for branches in cs:
                    self.branches_lists.append(branches)
        # Update global_classes
        # print('Se actualizan las clases de los clientes.')
        for branch_list in self.branches_lists:
            for branch in branch_list:
                branch.set_global_classes(self.classes_)

    def filter_conjunction_set_aggregator(self, cs):
        """
        Misma funcionalidad que filter_conjunction_set, pero esta utiliza menos probabilidades de entre las disponibles.
        """
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        if self.filter_approach == "number_of_samples":
            branches_metrics = [b.number_of_samples for b in cs]
        elif self.filter_approach == "entropy":
            branches_metrics = [-entropy(b.label_probas) for b in cs]
        elif self.filter_approach == "range":
            branches_metrics = [
                b.calculate_branch_probability_by_range(self.ranges) for b in cs
            ]
        threshold = sorted(branches_metrics, reverse=True)[
            self.amount_of_branches_threshold - 1
        ]
        return [b for b, metric in zip(cs, branches_metrics) if metric >= threshold][
            : self.amount_of_branches_threshold
        ]

    def set_parameters_model(
        self, features_names, original_data, feature_types, amount_of_branches_threshold
    ):
        # print(self.feature_names)
        self.feature_names = features_names
        original_data = original_data
        self.feature_types = feature_types
        self.exclusion_threshold = amount_of_branches_threshold
        if original_data is not None:
            self.set_ecdf(original_data)

    def print_branches_probability(self):
        print("MOSTRANDO BRANCHES PROBABILITY EN LOS CLIENTES.")
        if isinstance(self.conjunctionSet, list):
            for branch in self.conjunctionSet:
                print(branch.get_branch_probability())
        else:
            for branch_list in self.branches_lists:
                for branch in branch_list:
                    print(branch.get_branch_probability())

    def to_string(self):
        s = ""
        for branch in self.conjunctionSet:
            s += str(branch)
            s += "\n"
        return s

    def __len__(self):
        if self.conjunctionSet is None or self.conjunctionSet == 0:
            if len(self.branches_lists) > 0:
                total_branches = 0
                for branch_list in self.branches_lists:
                    total_branches += len(branch_list)
                return total_branches
            else:
                return 0
        else:
            return len(self.conjunctionSet)

    def _get_features_tree_to_features_name(self, features_names, model):
        # features_model = np.arange(model.n_features_)
        features_model = np.arange(model.n_features_in_)
        return {
            features_model[i]: features_names[i] for i in range(len(features_names))
        }
