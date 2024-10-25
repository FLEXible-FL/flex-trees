import numpy as np


class BranchCategorical:
    """Class for categorical branches"""

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
        self.feature_values = [
            -np.inf
        ] * self.number_of_features  # bound of the feature for the given rule
        self.label_probas = label_probas
        self.number_of_samples = number_of_samples  # save number of samples in leaf (not relevant for the current model)
        self.categorical_features_dict = {}
        self._branch_probability = 0
        # New
        self._global_classes = None
        # New for C4.5
        self.bound = None

    def add_condition(self, feature, value):
        """Function to add a condition to a feature

        Args:
            feature (str): Feature to add the condition to
            value (str): Value of the feature
        """
        self.feature_values[feature] = value  # Replace -np.inf for the value on the branch
        self.categorical_features_dict[feature] = value

    def contradict_branch(self, other_branch):
        """Function that checks whether two branch crontradicts each other or not

        Args:
            other_branch (BranchCategorical): other branch to compare with

        Returns:
            bool: True in the branches contradict each other
        """
        return any(
            feature in other_branch.categorical_features_dict
            and self.categorical_features_dict[feature]
            != other_branch.categorical_features_dict[feature]
            for feature in self.categorical_features_dict
        )

    def merge_branch(self, other_branch, personalized=True):
        """Function to merge two branches

        Args:
            other_branch (BranchCategorical): Other branch to merge with
            personalized (bool, optional): Deprecated for categorical branch. Defaults to True.

        Returns:
            BranchCategorical: New branch generated after merging.
        """
        new_label_probas = [
            k + v for k, v in zip(self.label_probas, other_branch.label_probas)
        ]
        new_label_probas = list(np.array(new_label_probas) / 2)

        new_label_probas = self.calculate_new_branch_probas(other_branch)

        new_number_of_samples = int(
            np.round(np.sqrt(self.number_of_samples * other_branch.number_of_samples))
        )
        new_branch_probability = max(
            self._branch_probability, other_branch.get_branch_probability()
        )
        # new_branch_probability = self.calculate_new_branch_probability(other_branch)
        new_b = BranchCategorical(
            self.feature_names,
            self.feature_types,
            self.label_names,
            new_label_probas,
            new_number_of_samples,
        )
        new_b.features = list(self.feature_values)
        for feature in range(self.number_of_features):
            new_b.add_condition(feature, other_branch.feature_values[feature])
        new_b.categorical_features_dict = dict(self.categorical_features_dict)
        new_b.categorical_features_dict.update(
            dict(other_branch.categorical_features_dict)
        )
        new_b.set_branch_probability(new_branch_probability)
        return new_b

    def calculate_new_branch_probas(self, other_branch):
        """Function to calculate the branch probability

        Args:
            other_branch_probability (_type_): _description_
        """
        self_probas = np.array(self.label_probas)
        self_ns = self.number_of_samples
        other_ns = other_branch.number_of_samples
        other_probas = np.array(other_branch.label_probas)
        total_ns = self_ns + other_ns
        return list(
            self_probas * (self_ns / total_ns) + other_probas * (other_ns / total_ns)
        )

    def toString(self):
        """
        This function creates a string representation of the branch (only for demonstration purposes)
        """
        s = "".join(
            str(feature) + " == " + str(threshold) + ", "
            for feature, threshold in enumerate(self.feature_values)
            if threshold != (-np.inf)
        )
        # s = ""
        # for feature, threshold in enumerate(self.features):
        #     if threshold != (-np.inf):
        #         #s +=  self.feature_names[feature] + ' > ' + str(np.round(threshold,3)) + ", "
        #         s += str(feature) + ' == ' + str(np.round(threshold, 3)) + ", "
        s += "labels: ["
        for k in range(len(self.label_probas)):
            s += str(self.label_names[k]) + " : " + str(self.label_probas[k]) + " "
        s += "]"
        s += " Number of samples: " + str(self.number_of_samples)
        return s

    def print_branch(self):
        # print the branch by using __str__()
        print(str(self))

    def get_label(self):
        # Return the predicted label accordint to the branch
        return np.argmax(self.label_probas)

    def get_branch_query(self) -> str:
        return " and ".join(
            str(feature) + "==" + str(threshold)
            for feature, threshold in enumerate(self.feature_values)
            if threshold != (-np.inf)
        )

    def __str__(self) -> str:
        s = "".join(
            str(feature) + " == " + str(threshold) + ", "
            for feature, threshold in enumerate(self.feature_values)
            if threshold != (-np.inf)
        )
        # s = ""
        # for feature, threshold in enumerate(self.features):
        #     if threshold != (-np.inf):
        #         #s +=  self.feature_names[feature] + ' > ' + str(np.round(threshold,3)) + ", "
        #         s += str(feature) + ' == ' + str(np.round(threshold, 3)) + ", "
        s += "Label probas: ["
        for k in range(len(self.label_probas)):
            s += str(self.label_names[k]) + " : " + str(self.label_probas[k]) + " "
        s += "]"
        s += " Number of samples: " + str(self.number_of_samples)
        return s

    def str_branch(self):
        return "".join(
            str(feature) + " == " + str(threshold) + ", "
            for feature, threshold in enumerate(self.feature_values)
            if threshold != (-np.inf)
        )

    def get_branch_dict(self, ecdf):
        """Generate a dictionaty with the information from the branch.

        Args:
            ecdf (Array-like Object): ECDF

        Returns:
            dict: Dictionary containing the information about the branch.
        """
        features = {
            self.feature_names[feature] + "_bound": value
            for feature, value in zip(
                range(len(self.feature_names)), self.feature_values
            )
        }
        features["number_of_samples"] = self.number_of_samples
        features["branch_probability"] = self.calculate_branch_probability_by_ecdf(ecdf)
        features["probas"] = np.array(self.label_probas)
        return features

    def get_branch_dict_new(self, ecdf=None):
        features = {
            self.feature_names[feature] + "_bound": value
            for feature, value in zip(
                range(len(self.feature_names)), self.feature_values
            )
        }
        features["number_of_samples"] = self.number_of_samples
        features["branch_probability"] = self.get_branch_probability()
        features["probas"] = np.array(self.label_probas)
        return features

    def calculate_branch_probability_by_ecdf(self, ecdf):
        # for i in range(len(ecdf)):
        #     probs = ecdf[i]([self.feature_names[i]])
        #     feature_probabilities.append((probs+delta))
        # return np.product(feature_probabilities)
        return 1.0

    def calculate_branch_probability_by_ecdf_new(self, ecdf):
        # for i in range(len(ecdf)):
        #     probs = ecdf[i]([self.feature_names[i]])
        #     feature_probabilities.append((probs+delta))
        # self._branch_probability = np.product(feature_probabilities)
        self._branch_probability = 1.0

    def is_excludable_branch(self, threshold):
        """Function to check if the branch is excludable of if it's neccesary
        for the ConjuctionSet.

        This function check check if the probability of the predicted class
        by the branch is greater than the threshold given by parameter.

        Args:
            threshold (float): Threshold value

        Returns:
            bool: True if the branch is excludable.
        """
        return max(self.label_probas) / np.sum(self.label_probas) > threshold

    def set_global_classes(self, global_classes):
        """Set the global classes of the problem, so it won't produce error
        when merging multiple clients

        Args:
            gc (list): List containing the name of the global classes
        """
        self._global_classes = global_classes  # Nombres de las clases ej: [1, 2, 3, 4, 5, 6, 7] o [0, 1, 2]
        if len(global_classes) > len(self.label_names):
            # print('Entro SET_GLOBAL_CLASSES')
            aux_probas = {c: 0 for c in global_classes}
            for i in range(len(self.label_names)):
                aux_probas[self.label_names[i]] = self.label_probas[i]
            self.label_names = self._global_classes
            # print(global_classes)
            self.label_probas = np.array(list(aux_probas.values()))
            # print(self.label_probas)

    def get_branch_probability(self):
        return self._branch_probability

    def set_branch_probability(self, branch_probability):
        self._branch_probability = branch_probability

    def __eq__(self, other_branch):
        return (
            (self.feature_names == other_branch.feature_names)
            and (self.feature_values == other_branch.feature_values)
            and (
                self.categorical_features_dict == other_branch.categorical_features_dict
            )
            and (self.label_probas == other_branch.label_probas)
            and (self._branch_probability == other_branch._branch_probability)
            and (self.number_of_samples == other_branch.number_of_samples)
            and (self.feature_types == other_branch.feature_types)
        )
