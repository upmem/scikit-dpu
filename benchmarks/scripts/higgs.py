import pandas as pd
import numpy as np
from hurry.filesize import size

from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from time import perf_counter
from sklearn.tree import _perfcounter as _perfcounter_cpu
from skdpu.tree import _perfcounter as _perfcounter_dpu

MAX_FEATURE_DPU = 5000000

ndpu_list = [10, 30, 100, 300, 1000, 2524]

# df = pd.read_csv("data/HIGGS.csv", dtype=np.float32, header=None, sep=",")
# df.to_parquet("./data/higgs.pq", index=False, compression=None)

df = pd.read_parquet("data/higgs.pq")

X, y = np.require(df.iloc[:, 1:].to_numpy(), requirements=['C', 'A', 'O']), np.require(df.iloc[:, 0].to_numpy(),
                                                                                       requirements=['C', 'A', 'O'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500000)

npoints, nfeatures = X_train.shape

del df

X_train, y_train = shuffle(X_train, y_train, random_state=0)

accuracies_dpu = []
accuracies_cpu = []
build_times_dpu = []
build_times_cpu = []
total_times_dpu = []
total_times_cpu = []
ndpu_effective = []

clf2 = DecisionTreeClassifier(random_state=None, criterion='gini', splitter='random', max_depth=10)

tic = perf_counter()
clf2.fit(X_train, y_train)
toc = perf_counter()
# export_graphviz(clf2, out_file="tree_cpu.dot")
y_pred2 = clf2.predict(X_test)
cpu_accuracy = accuracy_score(y_test, y_pred2)
cpu_total_time = toc - tic

print(f"Accuracy for CPU: {cpu_accuracy}")
print(f"build time for CPUs : {_perfcounter_cpu.time_taken} s")
print(f"total time for CPUs: {toc - tic} s")

for i_ndpu, ndpu in enumerate(ndpu_list):
    print(f"\nnumber of dpus : {ndpu}")
    npoints_per_dpu = 2 * (npoints // (2 * ndpu))
    if npoints_per_dpu * nfeatures > MAX_FEATURE_DPU:
        print("not enough DPUs, skipping")
        continue
    ndpu_effective.append(ndpu)
    data_size = npoints_per_dpu * (nfeatures + 1) * 4
    print(f"data size per dpu= {size(data_size)}")

    clf = DecisionTreeClassifierDpu(random_state=None, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu,
                                    max_depth=10)

    tic = perf_counter()
    npoints_rounded = npoints_per_dpu * ndpu
    print(f"npoints_rounded : {npoints_rounded}")
    X_rounded, y_rounded = X_train[:npoints_rounded], y_train[:npoints_rounded]
    clf.fit(X_rounded, y_rounded)
    toc = perf_counter()
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X_test)
    dpu_accuracy = accuracy_score(y_test, y_pred)

    accuracies_dpu.append(dpu_accuracy)
    build_times_dpu.append(_perfcounter_dpu.time_taken)
    total_times_dpu.append(toc - tic)
    print(f"Accuracy for DPUs: {dpu_accuracy}")
    print(f"build time for DPUs : {_perfcounter_dpu.time_taken} s")
    print(f"total time for DPUs: {toc - tic} s")

    accuracies_cpu.append(cpu_accuracy)
    build_times_cpu.append(_perfcounter_cpu.time_taken)
    total_times_cpu.append(cpu_total_time)

    df = pd.DataFrame(
        {"DPU accuracy": accuracies_dpu, "CPU accuracy": accuracies_cpu, "Build time on DPU": build_times_dpu,
         "Build time on CPU": build_times_cpu, "Total time on DPU": total_times_dpu,
         "Total time on CPU": total_times_cpu},
        index=ndpu_effective)

    df.to_csv("../higgs_results.csv")
