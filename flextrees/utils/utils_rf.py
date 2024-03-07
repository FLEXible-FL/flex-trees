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
        predictions = {}
        for estimator in self.estimators_:
            prediction = estimator.predict(X)
            for i, p in enumerate(prediction):
                if i not in predictions:
                    predictions[i] = []
                predictions[i].append(p)
        def most_common(lst):
            return max(set(lst), key=lst.count)
        predictions = [most_common(predictions[i]) for i in range(len(predictions))]
        return predictions

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
