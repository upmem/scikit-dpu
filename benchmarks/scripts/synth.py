import os
from pathlib import Path

from skdpu.tree import DecisionTreeClassifierDpu
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz, export_text
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from time import perf_counter


dirname = os.path.dirname(__file__)
graph_dir = os.path.join(dirname, '../graphs')
Path(graph_dir).mkdir(parents=True, exist_ok=True)

ndpu = 50
npoints_per_dpu = 2000
nfeatures = 16

data_size = npoints_per_dpu * (nfeatures + 1) * 4

X, y = make_classification(n_samples=ndpu * npoints_per_dpu,
                           n_features=nfeatures,
                           n_informative=4,
                           n_classes=3,
                           random_state=0)

clf = DecisionTreeClassifierDpu(random_state=1, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu, max_depth=10)

tic = perf_counter()
clf.fit(X, y)
toc = perf_counter()
export_graphviz(clf, out_file=os.path.join(graph_dir, "tree_dpu.dot"))
r = export_text(clf)
with open(os.path.join(graph_dir, 'dpu_tree.txt'), 'w') as f:
    f.write(r)
y_pred = clf.predict(X)
dpu_accuracy = accuracy_score(y, y_pred)

print(f"Accuracy for DPUs: {dpu_accuracy}")
print(f"total time for DPUs: {toc - tic} s")

clf2 = DecisionTreeClassifier(random_state=1, criterion='gini', splitter='random', max_depth=10)

tic = perf_counter()
clf2.fit(X, y)
toc = perf_counter()
export_graphviz(clf2, out_file=os.path.join(graph_dir, "tree_cpu.dot"))
y_pred2 = clf2.predict(X)
accuracy = accuracy_score(y, y_pred2)

print(f"Accuracy for CPU: {accuracy}")
print(f"total time for CPUs: {toc - tic} s")
