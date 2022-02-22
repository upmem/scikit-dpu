from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import export_text
from sklearn.datasets import make_classification
from contextlib import redirect_stdout
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter


n_dpu = 2
n_points_per_dpu = 200
n_features = 4

X, y = make_classification(n_samples=n_dpu * n_points_per_dpu,
                           n_features=n_features,
                           n_informative=2,
                           n_classes=3,
                           random_state=0,
                           n_clusters_per_class=1)

clf1 = DecisionTreeClassifierDpu(random_state=1,
                                 criterion='gini_dpu',
                                 splitter='random_dpu',
                                 ndpu=n_dpu,
                                 max_depth=3)
with open("out1.log", "w") as f:
    with redirect_stdout(f):
        clf1.fit(X, y)

clf2 = DecisionTreeClassifierDpu(random_state=1,
                                 criterion='gini_dpu',
                                 splitter='random_dpu',
                                 ndpu=n_dpu,
                                 max_depth=3)
with open("out2.log", "w") as f:
    with redirect_stdout(f):
        clf2.fit(X, y)