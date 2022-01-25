from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

clf = DecisionTreeClassifier(random_state=0, splitter='random')

clf.fit(iris.data, iris.target)
print(clf.predict(iris.data))
