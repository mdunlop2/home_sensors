from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


class StepwiseFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, scoring=roc_auc_score, n_jobs=1, cv=5):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    def _calculate_score(self, X, y, feature_set):
        # Fit the model and calculate score
        scores = []
        for train_index, val_index in self.cv.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            self.estimator.fit(X_train[:, feature_set], y_train)
            y_pred = self.estimator.predict(X_val[:, feature_set])
            scores.append(self.scoring(y_val, y_pred))
        return np.mean(scores)

    def _stepwise_selection(self, X, y):
        best_score = -np.inf
        best_features = []
        remaining_features = list(range(X.shape[1]))

        while remaining_features:
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._calculate_score)(X, y, best_features + [feature]) for feature in remaining_features
            )
            best_idx = np.argmax(scores)
            if best_score >= scores[best_idx]:
                print(
                    f"Finishing as selection best score {scores[best_idx]} was not better than existing model {best_score}"
                )
                break
            selected_feature = remaining_features.pop(best_idx)
            best_features.append(selected_feature)
            best_score = max(scores)
            print(f"Selected {selected_feature} with best score of {best_score}")
        return best_features

    def fit(self, X, y):
        self.selected_features_ = self._stepwise_selection(X, y)
        print(f"Selected features: {self.selected_features_}")
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


def main():
    # Generate a synthetic classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # Check the shape of features and labels
    print("Shape of features:", X.shape)
    print("Shape of labels:", y.shape)

    # Create the pipeline
    pipeline = Pipeline(
        [
            ("feature_selector", StepwiseFeatureSelector(estimator=LogisticRegression())),
            ("classifier", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)
    pred_skl = pipeline.predict(X.astype(np.float64))
    accuracy = roc_auc_score(y, pred_skl)
    print(accuracy)


if __name__ == "__main__":
    main()
