# -*- coding: utf-8 -*-
from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


def test_validity():
    n_dpu = 50
    n_points_per_dpu = 2000
    n_features = 16

    acceptable_delta = 0.1

    X, y = make_classification(n_samples=n_dpu * n_points_per_dpu,
                               n_features=n_features,
                               n_informative=4,
                               n_classes=3,
                               random_state=0)

    clf_dpu = DecisionTreeClassifierDpu(random_state=1,
                                        criterion='gini_dpu',
                                        splitter='random_dpu',
                                        ndpu=n_dpu,
                                        max_depth=10)

    clf_dpu.fit(X, y)
    y_pred_dpu = clf_dpu.predict(X)
    accuracy_gpu = accuracy_score(y, y_pred_dpu)

    clf_cpu = DecisionTreeClassifier(random_state=1,
                                     criterion='gini',
                                     splitter='random',
                                     max_depth=10)

    clf_cpu.fit(X, y)
    y_pred_cpu = clf_cpu.predict(X)
    accuracy_cpu = accuracy_score(y, y_pred_cpu)

    assert abs(accuracy_cpu - accuracy_gpu) <= acceptable_delta
