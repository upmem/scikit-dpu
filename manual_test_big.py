from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

ndpu = 64
npoints = 8

X, y = make_blobs(n_samples=ndpu * npoints, n_features=4, centers=3, random_state=0)

clf = DecisionTreeClassifierDpu(random_state=0, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu, max_depth=9)

clf.fit(X, y)
# export_graphviz(clf, out_file="tree.dot")
y_pred = clf.predict(X)
dpu_accuracy = accuracy_score(y, y_pred)

print(f"Accuracy for DPUs: {dpu_accuracy}")

clf2 = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='random', max_depth=9)

clf2.fit(X,y)
y_pred2 = clf2.predict(X)
accuracy = accuracy_score(y, y_pred2)

print(f"Accuracy for CPU: {accuracy}")
