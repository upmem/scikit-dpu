from sklearn.datasets import load_iris
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.tree import export_graphviz

iris = load_iris()

clf = DecisionTreeClassifierDpu(random_state=0, criterion='gini_dpu', splitter='random_dpu', ndpu=4)

X = iris.data[:128]
y = iris.target[:128]
clf.fit(X, y)
export_graphviz(clf, out_file="tree.dot")
print("predict")
print(clf.predict(X))
if (clf.predict(X) == y).all():
    print("success!")
else:
    print("failure...")
