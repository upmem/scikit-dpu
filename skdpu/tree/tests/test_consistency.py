# -*- coding: utf-8 -*-
from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import export_text
from . import dataset_generator
from .dataset_generator import get_classification_problem

n_dpu = dataset_generator.n_dpu


def test_consistency_new_classifier():
    X, y = get_classification_problem()

    clf1 = DecisionTreeClassifierDpu(
        random_state=1,
        criterion="gini_dpu",
        splitter="random_dpu",
        ndpu=n_dpu,
        max_depth=10,
    )

    clf1.fit(X, y)

    clf2 = DecisionTreeClassifierDpu(
        random_state=1,
        criterion="gini_dpu",
        splitter="random_dpu",
        ndpu=n_dpu,
        max_depth=10,
    )

    clf2.fit(X, y)

    r1 = export_text(clf1)
    r2 = export_text(clf2)

    assert r1 == r2


def test_consistency_same_classifier():
    X, y = get_classification_problem()

    clf1 = DecisionTreeClassifierDpu(
        random_state=1,
        criterion="gini_dpu",
        splitter="random_dpu",
        ndpu=n_dpu,
        max_depth=10,
    )

    clf1.fit(X, y)
    r1 = export_text(clf1)

    clf1.fit(X, y)
    r2 = export_text(clf1)

    assert r1 == r2


def test_consistency_new_array():
    X1, y1 = get_classification_problem()

    clf1 = DecisionTreeClassifierDpu(
        random_state=1,
        criterion="gini_dpu",
        splitter="random_dpu",
        ndpu=n_dpu,
        max_depth=10,
    )

    clf1.fit(X1, y1)

    X2, y2 = get_classification_problem()

    clf2 = DecisionTreeClassifierDpu(
        random_state=1,
        criterion="gini_dpu",
        splitter="random_dpu",
        ndpu=n_dpu,
        max_depth=10,
    )

    clf2.fit(X2, y2)

    r1 = export_text(clf1)
    r2 = export_text(clf2)

    assert r1 == r2
