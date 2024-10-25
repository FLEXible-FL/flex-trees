import numpy as np
import pandas as pd
from scipy.stats import entropy

EPSILON = 0.000001

#################################### NEW ####################################
#############################################################################


class TreeBranchMixed:
    def __init__(self, mask, classes=None, depth=0):
        self.mask = mask
        # print(mask)
        # print(self.mask)
        self.classes_ = classes
        self.childs = None
        self.split_feature = None
        self.split_values = None
        self.feature_mask = None
        self.depth = depth
        self.feature_type = None # Feature type of the node
        # Add options for continuous features
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None

    def split(self, df, available_features=None):
        """Function that splits a node.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            available_features (list, optional): List of the available features to
                split the node. Defaults to None.
        """
        # print(df)
        # TODO: Cambiar el split para que funcione con el lower/upper y el ==.
        # Ahora mismo sólo funciona con el ==.
        # breakpoint()
        if np.sum(self.mask) == 1:
            # print("Leaf node")
            self.childs = None
            self.left = None
            self.right = None
            return
        self.features = [str(i.split('_')[0]) for i in df.keys() if 'upper' in str(i)]
        self.features += [
            str(col.split("_")[0]) for col in df.keys() if "bound" in str(col)
        ]
        self.available_features = (
            list(available_features) if available_features else list(self.features)
        )
        # Select the best feature with its value to split the node, the mask to split the node and the type of the feature
        (
            self.split_feature,
            self.split_values,
            self.feature_mask,
            self.feature_type,
        ) = self.select_split_feature(df)
        new_available_features = list(self.available_features)
        new_available_features.remove(self.split_feature)
        self.create_mask(df)
        is_splitable = self.is_splitable()
        if is_splitable is False:
            self.childs = None
            self.left = None
            self.right = None
            return
        # breakpoint()
        if self.feature_type == 'continuous':
            self.left = TreeBranchMixed(
                list(
                    np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))
                ),
                self.classes_,
                depth=self.depth + 1,
            )
            self.right = TreeBranchMixed(
                list(
                    np.logical_and(
                        self.mask, np.logical_or(self.right_mask, self.both_mask)
                    )
                ),
                self.classes_,
                depth=self.depth + 1,
            )
            self.left.split(df)
            self.right.split(df)
        else:
            self.childs = []
            for i, child_mask in enumerate(self.childs_mask):
                child = TreeBranchMixed(
                    list(
                        np.logical_and(
                            self.mask, np.logical_and(child_mask, self.feature_mask)
                        )
                    ),
                    self.classes_,
                    depth=self.depth + 1,
                )
                self.childs.append((child, self.split_values[i]))
                child.split(df, new_available_features)
        # self.left=TreeBranchCategorical(list(np.logical_and(self.mask,np.logical_or(self.left_mask,self.both_mask))), self.classes_)
        # self.right = TreeBranchCategorical(list(np.logical_and(self.mask,np.logical_or(self.right_mask,self.both_mask))), self.classes_)
        # self.left.split(df)
        # self.right.split(df)

    def is_splitable_cont(self):
        """Function that checks if a node is splittable.

        Returns:
            bool: Returns True if the node is splittable, False otherwise.
        """
        if (
            np.sum(
                np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))
            )
            == 0
            or np.sum(
                np.logical_and(
                    self.mask, np.logical_or(self.right_mask, self.both_mask)
                )
            )
            == 0
        ):
            return False
        if np.sum(
            np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))
        ) == np.sum(self.mask) or np.sum(
            np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))
        ) == np.sum(
            self.mask
        ):
            return False
        return True

    def is_splitable(self):
        """Function that checks if a node is splittable.

        Returns:
            bool: Returns True if the node is splittable, False otherwise.
        """
        if self.feature_type == 'continuous':
            return self.is_splitable_cont()
        else:
            return self.is_splitable_cat()

    def is_splitable_cat(self):
        for child_mask in self.childs_mask:
            if (
                np.sum(
                    np.logical_and(
                        self.mask, np.logical_and(self.feature_mask, child_mask)
                    )
                )
                == 0
            ):
                return False
            if np.sum(
                np.logical_and(self.mask, np.logical_and(self.feature_mask, child_mask))
            ) == np.sum(self.mask):
                return False
        return True

    def create_mask(self, df):
        """Function that creates the mask for the childs of a node.

        Args:
            df (Pandas.Dataframe): Dataframe with the instances of the node.
        """
        if self.feature_type == 'continuous':
            # breakpoint()
            self.left_mask = df[str(self.split_feature) + "_upper"] <= self.split_values
            self.right_mask = df[str(self.split_feature) + "_lower"] >= self.split_values
            self.both_mask = (df[str(self.split_feature) + "_lower"] < self.split_values) & (
                df[str(self.split_feature) + "_upper"] > self.split_values
            )
        else:
            self.childs_mask = [
                df[str(self.split_feature) + "_bound"] == value
                for value in self.split_values
            ]
        # self.childs_mask = df.query(query)

    def select_split_feature(self, df):
        """Function that select the feature to split the node. It calculates the
        metric for each feature and returns the feature with the lowest metric.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.

        Returns:
            tuple: Tuple containing the feature, the value of the feature and the
                mask for the feature.
        """
        feature_mask = {}
        feature_to_metric = {}
        feature_values = {}
        feature_type = {}
        # breakpoint()
        df_cols = list(df.keys())
        for feature in self.available_features:
            cont = True if f'{feature}_upper' in df_cols or f'{feature}_lower' in df_cols else False
            if cont:
                # Feature is continuous
                value, metric = self.check_feature_split_value_cont(df, feature)
                feature_values[feature] = value
                feature_to_metric[feature] = metric
                feature_type[feature] = 'continuous'
                feature_mask[feature] = None # It will be created in create_mask
            else:
                # Feature is categorical
                metric, mask, values = self.check_feature_split_value(df, feature)
                feature_mask[feature] = mask # It will be used in create_mask
                feature_to_metric[feature] = metric
                feature_values[feature] = values
                feature_type[feature] = 'categorical'
        # breakpoint()
        # feature_to_metric |= {
        #     feature: feature_to_metric[feature] / len(feature_values[feature])
        #     for feature in feature_to_metric if f"{feature}_bound" in df_cols
        # }
        # for feature in feature_to_metric:
        #     # print(f"Feature: {feature_to_metric}")
        #     if feature_values[feature] == np.inf:
        #         feature_to_metric[feature] = -np.inf
        # breakpoint()
        feature = min(feature_to_metric, key=feature_to_metric.get)
        # feature = min(feature_to_metric, key=lambda k: feature_to_metric[k] + (1 / len(feature_values[k])))
        # print(f"Best feature: {feature}")
        # print(f"Best feature value: {feature_values[feature]}")
        # print(f"Best feature metric: {feature_to_metric[feature]}")
        # print(f"Best feature type: {feature_type[feature]}")
        return (
            feature,
            feature_values[feature],
            feature_mask[feature],
            feature_type[feature],
        )  # feature_to_value[feature]

    def check_feature_split_value_cont(self, df, feature):
        """Function that calculate the metric for a given feature.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            feature (str): Feature to calculate the metric.

        Returns:
            tuple: Two values, the first one is the value of the feature that
                minimizes the metric, and the second one is the metric for that
                value.
        """
        # breakpoint()
        value_to_metric = {}
        values = list(
            set(
                list(df[str(feature) + "_upper"][self.mask])
                + list(df[str(feature) + "_lower"][self.mask])
            )
        )
        # values = [val for val in values if val != -np.inf or val != np.inf]
        # breakpoint()
        np.random.shuffle(values)
        # values = values[:3]
        # print(values)
        for value in values:
            left_mask = [
                True if upper <= value else False
                for upper in df[str(feature) + "_upper"]
            ]
            right_mask = [
                True if lower >= value else False
                for lower in df[str(feature) + "_lower"]
            ]
            both_mask = [
                True if value < upper and value > lower else False
                for lower, upper in zip(
                    df[str(feature) + "_lower"], df[str(feature) + "_upper"]
                )
            ]
            value_to_metric[value] = self.get_value_metric_cont(
                df, left_mask, right_mask, both_mask
            )
        # print('CHECK FEATURE SPLIT VALUE')
        # print(value_to_metric)
        val = min(value_to_metric, key=value_to_metric.get)
        # val = min(value_to_metric, key=lambda k: value_to_metric[k] + (1 / len(values)))
        # print(f"Feature: {feature}")
        # print(f"Value: {val}")
        # print(f"Metric: {value_to_metric[val]}")
        # print(f"Values: {values}")
        # print(f"Metrics: {value_to_metric}")
        # breakpoint()
        return val, value_to_metric[val]

    def get_value_metric_cont(self, df, left_mask, right_mask, both_mask):
        """Function that calculates the metric for a given value of a feature.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            value_mask (Pandas.DataFrame): Masked dataframe with the instances of the
                node for a given value of a feature.
            feature_mask (Pandas.DataFrame): Masked dataframe with the instances of the
                node for a given feature.

        Returns:
            _type_: _description_
        """
        # breakpoint()
        l_df_mask = np.logical_and(np.logical_or(left_mask, both_mask), self.mask)
        r_df_mask = np.logical_and(np.logical_or(right_mask, both_mask), self.mask)
        if np.sum(l_df_mask) == 0 or np.sum(r_df_mask) == 0:
            return np.inf
        l_entropy, r_entropy = self.calculate_entropy(
            df, l_df_mask
        ), self.calculate_entropy(df, r_df_mask)
        l_prop = np.sum(l_df_mask) / len(l_df_mask)
        r_prop = np.sum(r_df_mask) / len(l_df_mask)
        return l_entropy * l_prop + r_entropy * r_prop

    def check_feature_split_value(self, df, feature):
        """Function that calculate the metric for a given feature.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            feature (str): Feature to calculate the metric.

        Returns:
            tuple: Three values, the metric, the mask for the feature and the values.
        """
        # breakpoint()
        value_to_metric = {}
        values = list(set(list(df[str(feature) + "_bound"][self.mask])))
        if -np.inf in values:
            values.remove(-np.inf)
        feature_mask = [
            val != -np.inf and val != np.inf for val in df[str(feature) + "_bound"]
        ]  # [self.mask]]
        for value in values:
            value_mask = [val == value for val in df[str(feature) + "_bound"]]
            value_to_metric[value] = self.get_value_metric(df, value_mask, feature_mask)
            # time.sleep(10)
        val = sum(value_to_metric.values())
        # val = min(value_to_metric, key=lambda k: value_to_metric[k] + (1 / len(values)))

        if val == 0 and len(value_to_metric) == 0:
            val = 9999999999
        return val, feature_mask, values  # , value_to_metric[val]

    def get_value_metric(self, df, value_mask, feature_mask):
        """Function that calculates the metric for a given value of a feature.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            value_mask (Pandas.DataFrame): Masked dataframe with the instances of the
                node for a given value of a feature.
            feature_mask (Pandas.DataFrame): Masked dataframe with the instances of the
                node for a given feature.

        Returns:
            _type_: _description_
        """
        # print(f"Value mask: {value_mask}")
        # print(f"Len value mask: {len(value_mask)}")
        child_df_mask = np.logical_and(
            np.logical_and(value_mask, feature_mask), self.mask
        )
        if np.sum(child_df_mask) == 0:
            return np.inf
        child_entropy = self.calculate_entropy(df, child_df_mask)
        child_prop = np.sum(child_df_mask) / len(child_df_mask)
        return child_prop * child_entropy

    def calculate_entropy(self, df, df_mask):
        """Function that calculates the entropy for a given node.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            df_mask (Pandas.DataFrame): Masked dataframe with the instances of the node.

        Returns:
            float: The entropy of the node.
        """
        x = df["probas"][df_mask].mean()
        return entropy(x / x.sum())

    def predict(self, X, classes_, branches_df):
        """Function that predicts the class for a given instance.

        Args:
            X (array): instance to predict
            classes_ (array): classes of the dataset
            branches_df (Pandas.DataFrame): Dataframe with the branches of the tree.

        Returns:
            tuple: Predictions and explanations for the instances predicted.
        """
        probas, depths = [], []
        explanations = []
        classes_ = self.classes_ if self.classes_ is not None else classes_
        for inst in X:
            # prob, depth = self.predict_probas_and_depth(inst, branches_df)
            try:
                prob, depth, explanation = self.predict_probas_and_depth(
                    inst, branches_df
                )
            except TypeError as er:
                # breakpoint()
                # print(er)
                pass
            try:
                probas.append(prob)
                depths.append(depth)
                explanations.append(explanation)
            except UnboundLocalError as er:
                breakpoint()
                print(er)
        predictions = [
            classes_[i] for i in np.array([np.argmax(prob) for prob in probas])
        ]
        explanations = [
            self.generate_explanation(pred, expl)
            for pred, expl in zip(predictions, explanations)
        ]
        # breakpoint()
        return predictions, explanations

    def predict_probas(self, X, classes_, branches_df):
        """Function that predicts the class for a given instance.

        Args:
            X (array): instance to predict.
            classes_ (array): classes of the dataset.
            branches_df (Pandas.DataFrame): Dataframe with the branches of the tree.

        Returns:
            array: Returns an array containing the probabilities for each class.
        """
        probas, depths = [], []
        explanations = []
        classes_ = self.classes_ if self.classes_ is not None else classes_
        for inst in X:
            prob, depth, explanation = self.predict_probas_and_depth(inst, branches_df)
            probas.append(prob)
            depths.append(depth)
            explanations.append(explanation)
        return np.array(probas), explanations

    def node_probas(self, df):
        """Function that get the probabilities for a node. Those probabilities are
        the label of the node.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.

        Returns:
            array: Array with the probabilities for each class.
        """
        x = df["probas"][self.mask].mean()
        return x / x.sum()

    def predict_probas_and_depth(self, inst, training_df, explanation=None):
        """This function returns the prediction of the instance and the depth of the
        tree that has been used to predict the instance. Also returns the explanation,
        that is calculated as the feature and the value of the feature that has been
        used to predict the instance.

        Args:
            inst (np.array): Instance to be predicted
            training_df (Pandas.DataFrame): Pandas dataframe with the training data.

        Returns:
            prediction: The prediction for a node
        """
        # TODO: Queda cambiar el predict para que funcione con el lower/upper y el ==.
        if explanation is None:
            explanation = {}
        if self.childs is None and self.left is None and self.right is None:  # self.left is None and self.right is None:
            # breakpoint()
            return self.node_probas(training_df), 1, {}
        if self.feature_type == 'continuous':
            feature = self.features.index(self.split_feature)
            feature = int(self.split_feature)
            if inst[feature] <= self.split_values:
                prediction, depth, aux_explanation = self.left.predict_probas_and_depth(
                    inst, training_df
                )
                explanation.update(aux_explanation)
                explanation[f"x{self.split_feature}"] = "<=" + str(self.split_values)
                return prediction, depth + 1, explanation
            else:
                prediction, depth, aux_explanation = self.right.predict_probas_and_depth(
                    inst, training_df
                )
                explanation.update(aux_explanation)
                explanation[f"x{self.split_feature}"] = ">" + str(self.split_values)
                return prediction, depth + 1, explanation
        else:
            for child, value in self.childs:
                # print(self.split_feature+"_bound")
                feature = self.features.index(self.split_feature)
                feature = int(self.split_feature)
                if inst[feature] == value:
                    prediction, depth, aux_explanation = child.predict_probas_and_depth(
                        inst, training_df
                    )
                    # explanation[self.split_feature] = value
                    explanation[self.split_feature] = inst[feature]
                    explanation.update(aux_explanation)
                    return prediction, depth + 1, explanation
            aux_df = training_df.loc[training_df[f'{self.split_feature}_bound'] == -np.inf]
            return aux_df["probas"].mean(), 1, {}
        return self.node_probas(training_df), 1, 'a'

    def generate_explanation(self, target, explanation):
        """Function that generates the explanation for a given instance.

        Args:
            target (int): Label predicted for an instance
            explanation (Dict): Dict containing the explanation for the instance
            in format feature: value.
        """
        ret = ""
        ret += f"The instance has been classified as: {str(target)}. "
        ret += "Because: "
        if self.feature_type == 'continuous':
            ret += "".join(
                [f" {feature}{value}," for feature, value in explanation.items()]
            )
        else:
            ret += "".join(
                f"{feature} = {value}, " for feature, value in explanation.items()
            )
        return f"{ret[:-2]}."
