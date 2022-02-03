# -*- coding: utf-8 -*-
from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import export_text
from sklearn.datasets import make_classification


def test_consistency():
    n_dpu = 50
    n_points_per_dpu = 2000
    n_features = 16

    X, y = make_classification(n_samples=n_dpu * n_points_per_dpu,
                               n_features=n_features,
                               n_informative=4,
                               n_classes=3,
                               random_state=0)

    clf1 = DecisionTreeClassifierDpu(random_state=1,
                                     criterion='gini_dpu',
                                     splitter='random_dpu',
                                     ndpu=n_dpu,
                                     max_depth=10)

    clf1.fit(X, y)

    clf2 = DecisionTreeClassifierDpu(random_state=1,
                                     criterion='gini_dpu',
                                     splitter='random_dpu',
                                     ndpu=n_dpu,
                                     max_depth=10)

    clf2.fit(X, y)

    r1 = export_text(clf1)
    r2 = export_text(clf2)

    assert r1 == r2
