from copy import deepcopy


class GlobalRandomForest:
    def __init__(self, max_depth=5, n_estimators=10, estimators_ = None) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.estimators_ = [] if estimators_ is None else estimators_

    def predict(self, X):
        """Function to predict the class of the input data. It calculates the
        majority vote of the estimators.

        Args:
            X (ArrayLike): Array with the data to predict.

        Returns:
            list: List with the predictions.
        """
        print(f"Number of estimators: {len(self.estimators_)}")
        predictions = {}
        for estimator in self.estimators_:
            prediction = estimator.predict(X)
            import time
            print(prediction)
            print(f"Estimator classes: {estimator.classes_}")
            time.sleep(10)
            for i, p in enumerate(prediction):
                if i not in predictions:
                    predictions[i] = []
                predictions[i].append(p)
        print(predictions)
        def most_common(lst):
            return max(set(lst), key=lst.count)
        predictions = [most_common(predictions[i]) for i in range(len(predictions))]
        # import time
        # print(predictions)
        # time.sleep(10)
        return predictions

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
