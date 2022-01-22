from sklearn.datasets import load_iris
from skdpu.tree._classes import DecisionTreeClassifier

iris = load_iris()

clf = DecisionTreeClassifier(random_state=0, criterion='gini_dpu', splitter='random_dpu')

clf.fit(iris.data, iris.target)
clf.predict(iris.data)
