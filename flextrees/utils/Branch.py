import numpy as np

EPSILON = 0.001


def get_prob(i, features_upper, features_lower, ecdf):
    return ecdf[i]([features_lower[i], features_upper[i]])


class Branch:
    def __init__(
        self,
        feature_names,
        feature_types,
        label_names,
        label_probas=None,
        number_of_samples=None,
    ):
        """Branch inatance can be initialized in 2 ways. One option is to initialize an empty branch
        (only with a global number of features and number of class labels) and gradually add
        conditions - this option is relevant for the merge implementation.
        Second option is to get the number of samples in branch and the labels
        probability vector - relevant for creating a branch out of an existing tree leaf.
        """
        self.feature_types = feature_types
        self.label_names = label_names
        self.number_of_features = len(feature_names)
        self.feature_names = feature_names
        # print(f"Feature_names: {feature_names}")
        # print(f"Feature_tupes: {feature_types}")
        self.features_upper = [
            np.inf
        ] * self.number_of_features  # upper bound of the feature for the given rule
        self.features_lower = [
            -np.inf
        ] * self.number_of_features  # lower bound of the feature for the given rule
        self.label_probas = label_probas
        self.number_of_samples = number_of_samples  # save number of samples in leaf (not relevant for the current model)
        self.categorical_features_dict = {}
        self._branch_probability = 0
        # Nuevo
        self._global_classes = None
        self._local_classes = None

    def add_condition(self, feature, threshold, bound):
        """
        This function gets feature index, its threshold for the condition and whether
        it is upper or lower bound. It updates the features thresholds for the given rule.
        """
        # print("Adding conditions to a branch.")
        # print(f"Threshold: {threshold}")
        if bound == "lower":
            self.features_lower[feature] = max(self.features_lower[feature], threshold)
        else:
            self.features_upper[feature] = min(self.features_upper[feature], threshold)
        if "=" in self.feature_names[feature] and threshold >= 0:
            splitted = self.feature_names[feature].split("=")
            self.categorical_features_dict[splitted[0]] = splitted[1]

    def contradict_branch(self, other_branch):
        """
        check wether Branch b can be merged with the "self" Branch. Returns Boolean answer.
        """
        for categorical_feature in self.categorical_features_dict:
            if (
                categorical_feature in other_branch.categorical_features_dict
                and self.categorical_features_dict[categorical_feature]
                != other_branch.categorical_features_dict[categorical_feature]
            ):
                return True
        for i in range(self.number_of_features):
            if (
                self.features_upper[i] <= other_branch.features_lower[i] + EPSILON
                or self.features_lower[i] + EPSILON >= other_branch.features_upper[i]
            ):
                return True
            if (
                self.feature_types[i] == "int"
                and min(self.features_upper[i], other_branch.features_upper[i]) % 1 > 0
                and min(self.features_upper[i], other_branch.features_upper[i])
                - max(self.features_lower[i], other_branch.features_lower[i])
                < 1
            ):
                return True

        return False

    def merge_branch(self, other_branch, personalized=True):
        """
        This method gets Branch b and create a new branch which is a merge of the "self" object
        with b. As describe in the algorithm.
        """
        # print('Entro aquí')
        new_label_probas = [
            k + v for k, v in zip(self.label_probas, other_branch.label_probas)
        ]
        new_label_probas = list(np.array(new_label_probas) / 2)
        new_number_of_samples = int(
            np.round(np.sqrt(self.number_of_samples * other_branch.number_of_samples))
        )
        # new_branch_probability = np.sqrt(self._branch_probability + other_branch.get_branch_probability())
        if personalized is True or personalized == "max" or personalized != "mean":
            new_branch_probability = max(
                self._branch_probability, other_branch.get_branch_probability()
            )
        else:
            new_branch_probability = np.mean(
                self._branch_probability + other_branch.get_branch_probability()
            )
        # new_branch_probability = 0
        # print('Número de muestras de otra regla: {}'.format(other_branch.number_of_samples))
        # print('Número de muestras es esta regla: {}'.format(self.number_of_samples))
        # print('Número de muestras de la nueva regla: {}'.format(np.round(new_number_of_samples)))
        new_b = Branch(
            self.feature_names,
            self.feature_types,
            self.label_names,
            new_label_probas,
            new_number_of_samples,
        )
        new_b.features_upper, new_b.features_lower = list(self.features_upper), list(
            self.features_lower
        )
        for feature in range(self.number_of_features):
            new_b.add_condition(feature, other_branch.features_upper[feature], "upper")
            new_b.add_condition(feature, other_branch.features_lower[feature], "lower")
        new_b.categorical_features_dict = dict(self.categorical_features_dict)
        new_b.categorical_features_dict.update(
            dict(other_branch.categorical_features_dict)
        )
        new_b.leaves_indexes = self.leaves_indexes + other_branch.leaves_indexes
        new_b.set_branch_probability(new_branch_probability)
        return new_b

    def toString(self):
        """
        This function creates a string representation of the branch (only for demonstration purposes)
        """
        s = ""
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                # s +=  self.feature_names[feature] + ' > ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + " > " + str(np.round(threshold, 3)) + ", "
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                # s +=  self.feature_names[feature] + ' <= ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + " <= " + str(np.round(threshold, 3)) + ", "
        s += "labels: ["
        for k in range(len(self.label_probas)):
            s += str(self.label_names[k]) + " : " + str(self.label_probas[k]) + " "
        s += "]"
        s += " Number of samples: " + str(self.number_of_samples)
        return s

    def str_branch(self):
        s = "".join(
            f"{str(feature)} > {str(np.round(threshold, 3))}, "
            for feature, threshold in enumerate(self.features_lower)
            if threshold != (-np.inf)
        )

        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                # s +=  self.feature_names[feature] + ' <= ' + str(np.round(threshold,3)) + ", "
                s += f"{str(feature)} <= {str(np.round(threshold, 3))}, "
        return s

    def printBranch(self):
        # print the branch by using tostring()
        print(self.toString())

    def containsInstance(self, instance):
        """This function gets an ibservation as an input. It returns True if the set of rules
        that represented by the branch matches the instance and false otherwise.
        """
        if np.sum(self.features_upper >= instance) == len(instance) and np.sum(
            self.features_lower < instance
        ) == len(instance):
            return True
        return False

    def getLabel(self):
        # Return the predicted label accordint to the branch
        return np.argmax(self.label_probas)

    # def containsInstance(self, v):
    #     for i,lower,upper in zip(range(len(v)),self.features_lower,self.features_upper):
    #         if v[i]>upper or v[i]<=lower:
    #             return False
    #     return True

    def get_branch_dict(self, ecdf):
        features = {}
        for feature, upper_value, lower_value in zip(
            range(len(self.features_upper)), self.features_upper, self.features_lower
        ):
            features[str(feature) + "_upper"] = upper_value
            features[str(feature) + "_lower"] = lower_value
        features["number_of_samples"] = self.number_of_samples
        features["branch_probability"] = self.calculate_branch_probability_by_ecdf(ecdf)
        # features['branch_probability'] = self._branch_probability
        # print('Mostrando label probas en BRANCH.')
        # print(self.label_probas)
        # print('Mostrando label names en BRANCH.')
        # print(self.label_names)
        features["probas"] = np.array(self.label_probas)
        return features

    def calculate_branch_probability_by_ecdf(self, ecdf):
        features_probabilities = []
        delta = 0.000000001
        for i in range(len(ecdf)):
            probs = ecdf[i]([self.features_lower[i], self.features_upper[i]])
            features_probabilities.append((probs[1] - probs[0] + delta))
        return np.product(features_probabilities)

    def calculate_branch_probability_by_range(self, ranges):
        features_probabilities = 1
        for range, lower, upper in zip(
            ranges, self.features_lower, self.features_upper
        ):
            probs = min(1, (upper - lower) / range)
        features_probabilities = features_probabilities * probs
        return features_probabilities

    def is_excludable_branch(self, threshold):
        if max(self.label_probas) / np.sum(self.label_probas) > threshold:
            return True
        return False

    def is_addable(self, other):
        for feature in range(self.number_of_features):
            if (
                self.features_upper[feature] + EPSILON < other.features_lower[feature]
                or other.features_upper[feature] + EPSILON
                < self.features_lower[feature]
            ):
                return False
        return True

    def is_valid_association(self, associative_leaves):
        for leaf1 in self.leaves_indexes:
            for leaf2 in self.leaves_indexes:
                if leaf1 == leaf2:
                    continue
                if associative_leaves[leaf1 + "|" + leaf2] == 0:
                    return False
        return True

    def number_of_unseen_pairs(self, associative_leaves):
        count = 0
        for leaf1 in self.leaves_indexes:
            for leaf2 in self.leaves_indexes:
                if leaf1 == leaf2:
                    continue
                if associative_leaves[leaf1 + "|" + leaf2] == 0:
                    count += 1
        return count * (-1)

    ############################ MÉTODOS NUEVOS ################################
    def calculate_branch_probability_by_ecdf_new(self, ecdf):
        features_probabilities = []
        delta = 0.000000001
        for i in range(len(ecdf)):
            probs = ecdf[i]([self.features_lower[i], self.features_upper[i]])
            features_probabilities.append((probs[1] - probs[0] + delta))
        # print('Aquí en ecdf new')
        self._branch_probability = np.product(features_probabilities)
        # print(self._branch_probability)

    def get_branch_probability(self):
        return self._branch_probability

    def set_branch_probability(self, bp):
        self._branch_probability = bp

    def get_branch_dict_new(self, ecdf=None):
        features = {}
        for feature, upper_value, lower_value in zip(
            range(len(self.features_upper)), self.features_upper, self.features_lower
        ):
            features[str(feature) + "_upper"] = upper_value
            features[str(feature) + "_lower"] = lower_value
        features["number_of_samples"] = self.number_of_samples
        features["branch_probability"] = self.get_branch_probability()
        features["probas"] = np.array(self.label_probas)
        return features

    def set_global_classes(self, gc):
        self._global_classes = (
            gc  # Nombres de las clases ej: [1, 2, 3, 4, 5, 6, 7] o [0, 1, 2]
        )
        if len(gc) > len(self.label_names):
            # print('Entro SET_GLOBAL_CLASSES')
            aux_probas = {c: 0 for c in gc}
            for i in range(len(self.label_names)):
                aux_probas[self.label_names[i]] = self.label_probas[i]
            self.label_names = self._global_classes
            # print(gc)
            self.label_probas = np.array(list(aux_probas.values()))
            # print(self.label_probas)

    def __lt__(self, other_branch):
        raise NotImplementedError("Comparation < not implemented yet.")

    def __gt__(self, other_branch):
        raise NotImplementedError("Comparation > not implemented yet.")

    def __eq__(self, other_branch):
        return (
            (self.feature_names == other_branch.feature_names)
            and (self.features_upper == other_branch.features_upper)
            and (self.features_lower == other_branch.features_lower)
            and (self.label_probas == other_branch.label_probas)
            and (self._branch_probability == other_branch._branch_probability)
            and (self.number_of_samples == other_branch.number_of_samples)
            and (self.feature_types == other_branch.feature_types)
        )

    def __ge__(self, other_branch):
        raise NotImplementedError("Comparation >= not implemented yet.")

    def __le__(self, other_branch):
        raise NotImplementedError("Comparation <= not implemented yet.")

    def __ne__(self, other_branch):
        return not self == other_branch

    def __str__(self) -> str:
        """
        This function creates a string representation of the branch (only for demonstration purposes)
        """
        s = ""
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                # s +=  self.feature_names[feature] + ' > ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + " > " + str(np.round(threshold, 3)) + ", "
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                # s +=  self.feature_names[feature] + ' <= ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + " <= " + str(np.round(threshold, 3)) + ", "
        s += "labels: ["
        for k in range(len(self.label_probas)):
            s += str(self.label_names[k]) + " : " + str(self.label_probas[k]) + " "
        s += "]"
        s += " Number of samples: " + str(self.number_of_samples)
        return s
