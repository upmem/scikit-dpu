from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ndpu_list = [256, 512, 1024, 2048, 2524]
train_size = int(6e5) * 256
test_size = int(1e5) * 256
npoints = train_size + test_size
nfeatures = 16
random_state = 42

accuracies_cpu = []

build_times_cpu = []

total_times_cpu = []

print(f"number of points : {train_size}")
data_size = train_size * (nfeatures + 1) * 4
print(f"data size = {data_size}")

X, y = make_classification(n_samples=npoints, n_features=nfeatures, n_informative=4, n_redundant=4,
                           random_state=random_state)
X = X.astype(np.float32)
y = y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)

clf2 = DecisionTreeClassifier(random_state=random_state, criterion='gini', splitter='random', max_depth=10)

tic = perf_counter()
clf2.fit(X_train, y_train)
toc = perf_counter()
y_pred2 = clf2.predict(X_test)
cpu_accuracy = accuracy_score(y_test, y_pred2)

cpu_total_time = toc - tic
print(f"Accuracy for CPU: {cpu_accuracy}")
print(f"total time for CPUs: {cpu_total_time} s")

# read CPU times
accuracies_cpu.append(cpu_accuracy)
total_times_cpu.append(cpu_total_time)

df = pd.DataFrame(
    {
        "CPU accuracy": accuracies_cpu,
        "Total time on CPU": total_times_cpu,
        "Build time on CPU": build_times_cpu,
    },
)

df.to_csv("strong_scaling_cpu.csv")
