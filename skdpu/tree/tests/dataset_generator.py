# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification

n_dpu = 10
n_points_per_dpu = 1000
n_features = 16


def get_classification_problem():
    X, y = make_classification(
        n_samples=n_dpu * n_points_per_dpu,
        n_features=n_features,
        n_informative=4,
        n_classes=3,
        random_state=0,
    )

    return X, y
