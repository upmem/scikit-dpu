from sklearn.datasets import load_iris
from skdpu.tree._classes import DecisionTreeClassifierDpu

iris = load_iris()

clf = DecisionTreeClassifierDpu(random_state=0, criterion='gini_dpu', splitter='random_dpu')

clf.fit(iris.data, iris.target)
print(clf.predict(iris.data))
