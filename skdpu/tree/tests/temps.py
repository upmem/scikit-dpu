from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import export_text
from sklearn.datasets import make_classification
from contextlib import redirect_stdout
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter


n_dpu = 2
n_points_per_dpu = 100
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

r1 = export_text(clf1, decimals=5, show_weights=True)
r2 = export_text(clf2, decimals=5, show_weights=True)

assert r1 == r2


with open("tree1.txt", "w") as f:
    f.write(r1)

with open("tree2.txt", "w") as f:
    f.write(r2)

y_pred_1, y_pred_2 = clf1.predict(X), clf2.predict(X)
acc1, acc2 = accuracy_score(y, y_pred_1), accuracy_score(y, y_pred_2)
acc1, acc2

np.where(y_pred_1 != y_pred_2)
y_pred_1[1292], y_pred_2[1292], y[1292]
y_pred_1[1940], y_pred_2[1940], y[1940]
X[1292]
X[1940]

mask_0=X[:,0] <= 0.427067
Counter(y[mask_0])
X_0 = X[mask_0]
mask_1 = X_0[:,0] <= 1.320848
y_0 = y[mask_0]
Counter(y_0[mask_1])
len(y_0)



n_dpu = 2
n_points_per_dpu = 100
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

clf1.fit(X, y)
r1 = export_text(clf1)

clf1.fit(X, y)
r2 = export_text(clf1)

assert r1 == r2

