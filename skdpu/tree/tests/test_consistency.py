# -*- coding: utf-8 -*-
from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import export_text
from sklearn.datasets import make_classification


def test_consistency_new_classifier():
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


def test_consistency_same_classifier():
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
    r1 = export_text(clf1)

    clf1.fit(X, y)
    r2 = export_text(clf1)

    assert r1 == r2


def test_consistency_new_array():
    n_dpu = 50
    n_points_per_dpu = 2000
    n_features = 16

    X1, y1 = make_classification(n_samples=n_dpu * n_points_per_dpu,
                               n_features=n_features,
                               n_informative=4,
                               n_classes=3,
                               random_state=0)

    clf1 = DecisionTreeClassifierDpu(random_state=1,
                                     criterion='gini_dpu',
                                     splitter='random_dpu',
                                     ndpu=n_dpu,
                                     max_depth=10)

    clf1.fit(X1, y1)

    X2, y2 = make_classification(n_samples=n_dpu * n_points_per_dpu,
                               n_features=n_features,
                               n_informative=4,
                               n_classes=3,
                               random_state=0)

    clf2 = DecisionTreeClassifierDpu(random_state=1,
                                     criterion='gini_dpu',
                                     splitter='random_dpu',
                                     ndpu=n_dpu,
                                     max_depth=10)

    clf2.fit(X2, y2)

    r1 = export_text(clf1)
    r2 = export_text(clf2)

    assert r1 == r2