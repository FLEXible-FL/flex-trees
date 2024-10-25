import numpy as np
from scipy.stats import entropy

EPSILON = 0.000001

#################################### NEW ####################################
#############################################################################


class TreeBranchCategorical:
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

    def split(self, df, available_features=None):
        """Function that splits a node.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            available_features (list, optional): List of the available features to
                split the node. Defaults to None.
        """
        if np.sum(self.mask) == 1:
            self.childs = None
            return
        # self.features = [int(i.split('_')[0]) for i in df.keys() if 'upper' in str(i)]
        self.features = [
            str(col.split("_")[0]) for col in df.keys() if "bound" in str(col)
        ]
        self.available_features = (
            list(available_features) if available_features else list(self.features)
        )
        # print(self.features)
        # print(df.keys())
        # print(f"Printing self.mask: {self.mask}")
        # print(f"Printing len self.mask: {len(self.mask)}")
        (
            self.split_feature,
            self.split_values,
            self.feature_mask,
        ) = self.select_split_feature(df)
        new_available_features = list(self.available_features)
        new_available_features.remove(self.split_feature)
        self.create_mask(df)
        is_splitable = self.is_splitable()
        if is_splitable is False:
            self.childs = None
            return
        self.childs = []
        for i, child_mask in enumerate(self.childs_mask):
            # print(f"Child mask: {list(np.logical_and(self.mask, np.logical_or(child_mask, self.feature_mask)))}")
            # print(f"Len child mask: {len(list(np.logical_and(self.mask, np.logical_or(child_mask, self.feature_mask))))}")
            # print(f"Len child mask: {len(child_mask)}")
            # print(f"Child mask: {child_mask}")
            # print(f"Feature mask: {self.feature_mask}")
            # print(f"Logical or entre hijo y característica: {np.logical_or(child_mask, self.feature_mask)}")
            # print(f"Logical and entre hijo y característica: {np.logical_and(child_mask, self.feature_mask)}")
            # print("CREO NODO HIJO.")
            # print(f"Hijo con característica {self.split_feature} y valor de característica: {self.split_values[i]}")
            # child = TreeBranchCategorical(list(np.logical_and(self.mask, np.logical_or(child_mask, self.feature_mask))), self.classes_)
            child = TreeBranchCategorical(
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

    def is_splitable(self):
        """Function that checks if a node is splittable.

        Returns:
            bool: Returns True if the node is splittable, False otherwise.
        """
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
        # query = ' and '.join(str(self.split_feature)+"_bound"+"=="+'"'+str(value)+'"' for value in self.split_values)
        # print(df[self.split_feature+"_bound"])
        # print(type(df[self.split_feature+"_bound"]))
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
        for feature in self.available_features:
            metric, mask, values = self.check_feature_split_value(df, feature)
            feature_mask[feature] = mask
            feature_to_metric[feature] = metric
            feature_values[feature] = values
        feature_to_metric = {
            feature: feature_to_metric[feature] / len(feature_values[feature])
            for feature in feature_to_metric
        }
        feature = min(feature_to_metric, key=feature_to_metric.get)
        # print(f"Best feature: {feature}")
        # print(f"Best feature value: {feature_to_value[feature]}")
        # print(f"Best feature metric: {feature_to_metric[feature]}")
        values = list(set(list(df[f"{str(feature)}_bound"][self.mask])))
        # values=list(set(list(df[str(feature)+'_bound'][self.mask])))
        # print(f"Best feature values: {values}")
        # print(f"Feature to metric: {feature_to_metric}")
        return (
            feature,
            feature_values[feature],
            feature_mask[feature],
        )  # feature_to_value[feature]

    def check_feature_split_value(self, df, feature):
        """Function that calculate the metric for a given feature.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            feature (str): Feature to calculate the metric.

        Returns:
            tuple: Three values, the metric, the mask for the feature and the values.
        """
        value_to_metric = {}
        values = list(set(list(df[str(feature) + "_bound"][self.mask])))
        # print(f"Feature {feature} with values: {values}")
        # print(f"Feature values: {df[str(feature)+'_bound']}")
        # print(f"Any: {any(df[str(feature)+'_bound']!=-np.inf)}")
        # np.random.shuffle(values) # Not needed in TreeBranchCategorical?
        # values = values[:3] # Not needed in TreeBranchCategorical?
        feature_mask = [
            val != -np.inf for val in df[str(feature) + "_bound"]
        ]  # [self.mask]]
        # print(f"self.mask: {self.mask}")
        # print(f"Len self.mask: {len(self.mask)}")
        # print(f"df de feature{df[str(feature)+'_bound']}")
        # print(f"df feature con mask: {df[str(feature)+'_bound'][self.mask]}")
        # print(f"Feature mask: {feature_mask}")
        # print(f"Len feature mask: {len(feature_mask)}")
        # print(f"Evaluating feature: {feature}")
        for value in values:
            value_mask = [val == value for val in df[str(feature) + "_bound"]]
            value_to_metric[value] = self.get_value_metric(df, value_mask, feature_mask)
            # time.sleep(10)
        val = sum(value_to_metric.values())
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
            except TypeError:
                # breakpoint()
                # print(er)
                pass
            try:
                probas.append(prob)
                depths.append(depth)
                explanations.append(explanation)
                print("Appending")
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
        if explanation is None:
            explanation = {}
        if self.childs is None:  # self.left is None and self.right is None:
            # breakpoint()
            return self.node_probas(training_df), 1, {}
        for child, value in self.childs:
            # print(self.split_feature+"_bound")
            feature = self.features.index(self.split_feature)
            if inst[feature] == value:
                prediction, depth, aux_explanation = child.predict_probas_and_depth(
                    inst, training_df
                )
                # explanation[self.split_feature] = value
                explanation[self.split_feature] = inst[feature]
                explanation.update(aux_explanation)
                return prediction, depth + 1, explanation
        # return self.node_probas(training_df), 1, 'a'

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
        ret += "".join(
            f"{feature} = {value}, " for feature, value in explanation.items()
        )
        return f"{ret[:-2]}."
