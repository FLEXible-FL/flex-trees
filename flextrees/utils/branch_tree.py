import numpy as np
from scipy.stats import entropy

EPSILON = 0.000001

#################################### NEW ####################################
from sklearn.metrics import auc, cohen_kappa_score, roc_curve

#############################################################################


class TreeBranch:
    def __init__(self, mask, classes=None, depth=0):
        self.mask = mask
        # print(mask)
        # print(self.mask)
        self.classes_ = classes
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.detph = depth

    def split(self, df):
        """Function that splits a node into two childs.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
        """
        # print(df)
        # if np.sum(self.mask)==1 or self.has_same_class(df):
        if np.sum(self.mask) == 1:
            self.left = None
            self.right = None
            return
        self.features = [int(i.split("_")[0]) for i in df.keys() if "upper" in str(i)]
        # print(self.features)
        # print(f"Printing self.mask: {self.mask}")
        # print(f"Printing len self.mask: {len(self.mask)}")
        self.split_feature, self.split_value = self.select_split_feature(df)
        self.create_mask(df)
        is_splitable = self.is_splitable()
        if is_splitable is False:
            self.left = None
            self.right = None
            return
        # print(f"Left tree mask: {list(np.logical_and(self.mask,np.logical_or(self.left_mask,self.both_mask)))}")
        # print(f"Left len tree mask: {len(list(np.logical_and(self.mask,np.logical_or(self.left_mask,self.both_mask))))}")
        # print(f"Right tree mask: {list(np.logical_and(self.mask,np.logical_or(self.right_mask,self.both_mask)))}")
        # print(f"Right len tree mask: {len(list(np.logical_and(self.mask,np.logical_or(self.right_mask,self.both_mask))))}")
        # print(f"Both mask: {self.both_mask}")
        # print(f"Len de both mask: {len(self.both_mask)}")
        # print(f"Logical or entre right mask y both mask: {np.logical_or(self.right_mask,self.both_mask)}")
        # print(f"True right mask: {self.right_mask}")
        self.left = TreeBranch(
            list(
                np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))
            ),
            self.classes_,
            depth=self.detph + 1,
        )
        self.right = TreeBranch(
            list(
                np.logical_and(
                    self.mask, np.logical_or(self.right_mask, self.both_mask)
                )
            ),
            self.classes_,
            depth=self.detph + 1,
        )
        self.left.split(df)
        self.right.split(df)

    def is_splitable(self):
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

    def create_mask(self, df):
        """Function that creates the mask for the childs of a node.

        Args:
            df (Pandas.Dataframe): Dataframe with the instances of the node.
        """
        self.left_mask = df[str(self.split_feature) + "_upper"] <= self.split_value
        self.right_mask = df[str(self.split_feature) + "_lower"] >= self.split_value
        self.both_mask = (df[str(self.split_feature) + "_lower"] < self.split_value) & (
            df[str(self.split_feature) + "_upper"] > self.split_value
        )
        # self.both_mask = [True if self.split_value < upper and self.split_value > lower else False for lower, upper in
        #             zip(df[str(self.split_feature) + '_lower'], df[str(self.split_feature) + "_upper"])]

    def select_split_feature(self, df):
        """Function that select the feature to split the node. It calculates the
        metric for each feature and returns the feature with the lowest metric.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.

        Returns:
            tuple: Tuple containing the feature and the value of the feature that
                minimizes the metric.
        """
        feature_to_value = {}
        feature_to_metric = {}
        for feature in self.features:
            value, metric = self.check_feature_split_value(df, feature)
            feature_to_value[feature] = value
            feature_to_metric[feature] = metric
        # print('SELECT_SPLIT_FEATURE')
        # print(feature_to_value)
        # print(feature_to_metric)
        feature = min(feature_to_metric, key=feature_to_metric.get)
        return feature, feature_to_value[feature]

    def check_feature_split_value(self, df, feature):
        """Function that calculate the metric for a given feature.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            feature (str): Feature to calculate the metric.

        Returns:
            tuple: Two values, the first one is the value of the feature that
                minimizes the metric, and the second one is the metric for that
                value.
        """
        value_to_metric = {}
        values = list(
            set(
                list(df[str(feature) + "_upper"][self.mask])
                + list(df[str(feature) + "_lower"][self.mask])
            )
        )
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
            value_to_metric[value] = self.get_value_metric(
                df, left_mask, right_mask, both_mask
            )
        # print('CHECK FEATURE SPLIT VALUE')
        # print(value_to_metric)
        val = min(value_to_metric, key=value_to_metric.get)
        return val, value_to_metric[val]

    def get_value_metric(self, df, left_mask, right_mask, both_mask):
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
        if self.left is None and self.right is None:
            return self.node_probas(training_df), 1, {}
        if inst[self.split_feature] <= self.split_value:
            prediction, depth, aux_explanation = self.left.predict_probas_and_depth(
                inst, training_df
            )
            explanation.update(aux_explanation)
            explanation[f"x{self.split_feature}"] = "<=" + str(self.split_value)
            return prediction, depth + 1, explanation
        else:
            prediction, depth, aux_explanation = self.right.predict_probas_and_depth(
                inst, training_df
            )
            explanation.update(aux_explanation)
            explanation[f"x{self.split_feature}"] = ">" + str(self.split_value)
            return prediction, depth + 1, explanation

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

    """
    def get_node_prediction(self,training_df):
        v=training_df['probas'][self.mask][0]
        v=[i/np.sum(v) for i in v]
        return np.array(v)
    
    
    def opposite_col(self,s):
        if 'upper' in s:
            return s.replace('upper','lower')
        else:
            return s.replace('lower', 'upper')
    """

    def calculate_entropy(self, test_df, test_df_mask):
        """Function that calculates the entropy for a given node.

        Args:
            df (Pandas.DataFrame): Dataframe with the instances of the node.
            df_mask (Pandas.DataFrame): Masked dataframe with the instances of the node.

        Returns:
            float: The entropy of the node.
        """
        x = test_df["probas"][test_df_mask].mean()
        return entropy(x / x.sum())

    def count_depth(self):
        """Function that counts the depth of a node.

        Returns:
            int: The depth of the node.
        """
        if self.right is None:
            return 1
        return max(self.left.count_depth(), self.right.count_depth()) + 1

    def number_of_children(self):
        """
        Function that returns the number of children of a node.
        Returns:
            int: The number of children of a node.
        """
        if self.right is None:
            return 1
        return 1 + self.right.number_of_children() + self.left.number_of_children()

    """
    def has_same_class(self,df):
        labels=set([np.argmax(l) for l in df['probas'][self.mask]])
        if len(labels)>1:
            return False
        return True
    """

    ################################## NEW ####################################
    def new_model_measures(self, X, Y, branches_df, classes_p=None):
        # DEPRECATED
        # NO LAS ESTOY UTILIZANDO
        result_dict = {}
        probas, depths = [], []
        self.classes_ = classes_p if classes_p is not None else self.classes_
        for inst in X:
            prob, depth = self.predict_probas_and_depth(inst, branches_df)
            probas.append(prob)
            depths.append(depth)
        print(self.classes_)
        # Modificar la predicción, para que se haga sobre la clase más probable sobre las que tiene el cliente
        # Modificar para que funcione correctamente sobre multiclase
        predictions = [
            self.classes_[i] for i in np.array([np.argmax(prob) for prob in probas])
        ]
        result_dict["new_model_average_depth"] = np.mean(depths)
        result_dict["new_model_min_depth"] = np.min(depths)
        result_dict["new_model_max_depth"] = np.max(depths)
        result_dict["new_model_accuracy"] = np.sum(predictions == Y) / len(Y)
        result_dict["new_model_auc"] = self.get_auc(Y, np.array(probas), self.classes_)
        result_dict["new_model_kappa"] = cohen_kappa_score(Y, predictions)
        result_dict["new_model_number_of_nodes"] = self.number_of_children()
        result_dict["new_model_probas"] = probas
        result_dict["predictions"] = predictions
        return result_dict

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
            prob, depth, explanation = self.predict_probas_and_depth(inst, branches_df)
            probas.append(prob)
            depths.append(depth)
            explanations.append(explanation)
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
            # prob, depth = self.predict_probas_and_depth(inst, branches_df)
            prob, depth, explanation = self.predict_probas_and_depth(inst, branches_df)
            probas.append(prob)
            depths.append(depth)
            explanations.append(explanation)
        return np.array(probas), explanations

    def get_auc(self, Y, y_score, classes):
        """Function to calculate the auc for a given set of predictions.

        Args:
            Y (array): Labels of the instances.
            y_score (array): Array with the predictions for each instance.
            classes (array): Array with the classes of the dataset.

        Returns:
            _type_: _description_
        """
        y_test_binarize = np.array([[1 if i == c else 0 for c in classes] for i in Y])
        fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
        return auc(fpr, tpr)

    def generate_explanation(self, target, explanation):
        """Function that generates an explanation for a given target instance.

        Args:
            target (int): Label predicted for an instance
            explanation (Dict): Dict containing the explanation for the instance
            in format feature: '<=value' or feature: '>value'.
        """
        ret = ""
        ret += f"The instance was classified as {target}. "
        ret += "Because:"
        ret += "".join(
            [f" {feature}{value}," for feature, value in explanation.items()]
        )
        return f"{ret[:-1]}."
