from typing import Union

import numba
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from lib.common.logging import get_logger

_LOGGER = get_logger(__name__)


class StepwiseFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: Union[DecisionTreeClassifier, LogisticRegression],
        n_jobs: int = 1,
        cv: int = 5,
        min_improvement_r: float = 0.01,
    ):
        """
        Forward feature selection.
        The algorithm is as follows:

        ```
        * begin with no features and minimum model score (0.5 for AUC) equivalent to guessing at random.
        * while there are remaining features:
            * for each remaining feature:
                * create a new model including the feature
                * evaluate unseen performance of new model with cross-validation
            * find feature providing best new model
            * if adding this feature improved out of sample performance on average by more than 1%, we include it in the model and remove it from remaining features
            * if no improvement in the model from adding any feature, break
        ```
        """
        self.estimator = estimator
        self.scoring = roc_auc_score
        self.n_jobs = n_jobs
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        self.min_improvement_r = min_improvement_r

    def _calculate_score(self, X: numba.float32[:, :], y: numba.float32[:], feature_set: list[int]) -> numba.float32:
        """
        Fit the model and calculate score on unseen data via K-fold sampling.
        """
        scores = []
        for train_index, val_index in self.cv.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            self.estimator.fit(X_train[:, feature_set], y_train)
            y_pred = self.estimator.predict_proba(X_val[:, feature_set])[:, 1]
            scores.append(self.scoring(y_val, y_pred))
        return np.median(scores)

    def _stepwise_selection(self, X: numba.float32[:, :], y: numba.float32[:]) -> list[int]:
        """
        Perform forward stepwise feature selection algorithm
        """
        best_features = []
        best_score = 0.5

        remaining_features = list(range(X.shape[1]))

        while remaining_features:
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._calculate_score)(X, y, best_features + [feature]) for feature in remaining_features
            )
            best_idx = np.argmax(scores)
            if best_score * (1 + self.min_improvement_r) >= scores[best_idx]:
                _LOGGER.info(
                    f"Finishing as selection best score {scores[best_idx]} was not better than existing model {best_score}"
                )
                break
            selected_feature = remaining_features.pop(best_idx)
            best_features.append(selected_feature)
            best_score = max(scores)
            _LOGGER.info(f"Selected {selected_feature} with best score of {best_score}")
        return best_features

    def fit(self, X: numba.float32[:, :], y: numba.float32[:]) -> "StepwiseFeatureSelector":
        """
        Select features using forward stepwise algorithm.
        """
        self.selected_features_ = self._stepwise_selection(X, y)
        _LOGGER.info(f"Selected features: {self.selected_features_}")
        return self

    def transform(self, X: numba.float32[:, :]) -> numba.float32[:, :]:
        """
        Filter input data to only selected features.
        """
        return X[:, self.selected_features_]
