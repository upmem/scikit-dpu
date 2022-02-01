from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

X = iris.data[:140]
y = iris.target[:140]

leaf_filter = np.logical_and(X[:, 2] >= 2.209,
              np.logical_and(X[:, 2] >= 5.029,
              np.logical_and(X[:, 3] <= 2.121,
                             X[:, 0] <= 6.303)))
