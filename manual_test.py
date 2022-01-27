from sklearn.datasets import load_iris
from skdpu.tree._classes import DecisionTreeClassifierDpu

iris = load_iris()

clf = DecisionTreeClassifierDpu(random_state=0, criterion='gini_dpu', splitter='random_dpu', ndpu=2)

X = iris.data[:140]
y = iris.target[:140]
clf.fit(X, y)
print("predict")
print(clf.predict(iris.data))
