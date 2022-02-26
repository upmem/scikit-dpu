# -*- coding: utf-8 -*-
from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from . import dataset_generator
from .dataset_generator import get_classification_problem

n_dpu = dataset_generator.n_dpu


def test_validity():
    acceptable_delta = 0.1

    X, y = get_classification_problem()

    clf_dpu = DecisionTreeClassifierDpu(
        random_state=1,
        criterion="gini_dpu",
        splitter="random_dpu",
        ndpu=n_dpu,
        max_depth=10,
    )

    clf_dpu.fit(X, y)
    y_pred_dpu = clf_dpu.predict(X)
    accuracy_gpu = accuracy_score(y, y_pred_dpu)

    clf_cpu = DecisionTreeClassifier(
        random_state=1, criterion="gini", splitter="random", max_depth=10
    )

    clf_cpu.fit(X, y)
    y_pred_cpu = clf_cpu.predict(X)
    accuracy_cpu = accuracy_score(y, y_pred_cpu)

    assert abs(accuracy_cpu - accuracy_gpu) <= acceptable_delta
