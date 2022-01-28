from sklearn.datasets import load_iris
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.tree import export_graphviz

iris = load_iris()

clf = DecisionTreeClassifierDpu(random_state=0, criterion='gini_dpu', splitter='random_dpu', ndpu=1)

X = iris.data[:140]
y = iris.target[:140]
clf.fit(X, y)
export_graphviz(clf, out_file="tree.dot")
print("predict")
print(clf.predict(X))
