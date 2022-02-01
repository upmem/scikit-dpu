from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

clf = DecisionTreeClassifier(random_state=0, splitter='random')

X = iris.data[:140]
y = iris.target[:140]
clf.fit(X, y)
print(clf.predict(iris.data))