from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from time import perf_counter
from sklearn.tree import _perfcounter as _percounter_cpu
from skdpu.tree import _perfcounter as _perfcounter_dpu
from hurry.filesize import size

ndpu = 500
npoints_per_dpu = 10000
nfeatures = 67

data_size = npoints_per_dpu * (nfeatures + 1) * 4
print(f"data size per dpu= {size(data_size)}")

X, y = make_blobs(n_samples=ndpu * npoints_per_dpu, n_features=nfeatures, centers=3, random_state=0, center_box=(-1, 1))

clf = DecisionTreeClassifierDpu(random_state=1, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu, max_depth=10)

tic = perf_counter()
clf.fit(X, y)
toc = perf_counter()
export_graphviz(clf, out_file="tree_dpu.dot")
y_pred = clf.predict(X)
dpu_accuracy = accuracy_score(y, y_pred)

print(f"Accuracy for DPUs: {dpu_accuracy}")
print(f"build time for CPUs : {_perfcounter_dpu.time_taken} s")
print(f"total time for DPUs: {toc - tic} s")

clf2 = DecisionTreeClassifier(random_state=1, criterion='gini', splitter='random', max_depth=10)

tic = perf_counter()
clf2.fit(X,y)
toc = perf_counter()
export_graphviz(clf2, out_file="tree_cpu.dot")
y_pred2 = clf2.predict(X)
accuracy = accuracy_score(y, y_pred2)

print(f"Accuracy for CPU: {accuracy}")
print(f"build time for CPUs : {_percounter_cpu.time_taken} s")
print(f"total time for CPUs: {toc - tic} s")
