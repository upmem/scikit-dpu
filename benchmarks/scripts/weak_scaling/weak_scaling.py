from time import perf_counter

import pandas as pd
from hurry.filesize import size

from skdpu.tree import _perfcounter as _perfcounter_dpu
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _perfcounter as _perfcounter_cpu

ndpu_list = [1, 4, 16, 64]
npoints_per_dpu = int(1e5)
nfeatures = 16

accuracies_dpu = []
accuracies_cpu = []
build_times_dpu = []
dpu_times_dpu = []

build_times_cpu = []
total_times_dpu = []
total_times_cpu = []

for i_ndpu, ndpu in enumerate(ndpu_list):
    print(f"number of points per dpu : {npoints_per_dpu}")
    data_size = npoints_per_dpu * (nfeatures + 1) * 4
    print(f"data size per dpu= {size(data_size)}")

    X, y = make_blobs(n_samples=ndpu * npoints_per_dpu, n_features=nfeatures, centers=3, random_state=0,
                      center_box=(-1, 1))

    clf = DecisionTreeClassifierDpu(random_state=0, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu,
                                    max_depth=10)

    tic = perf_counter()
    clf.fit(X, y)
    toc = perf_counter()
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X)
    dpu_accuracy = accuracy_score(y, y_pred)

    accuracies_dpu.append(dpu_accuracy)
    build_times_dpu.append(_perfcounter_dpu.time_taken)
    total_times_dpu.append(toc - tic)
    dpu_times_dpu.append(_perfcounter_dpu.dpu_time)
    print(f"Accuracy for DPUs: {dpu_accuracy}")
    print(f"build time for DPUs : {_perfcounter_dpu.time_taken} s")
    print(f"total time for DPUs: {toc - tic} s")
    print(f"DPU runtime for DPUs: {_perfcounter_dpu.dpu_time} s")

    clf2 = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='random', max_depth=10)

    tic = perf_counter()
    clf2.fit(X, y)
    toc = perf_counter()
    # export_graphviz(clf2, out_file="tree_cpu.dot")
    y_pred2 = clf2.predict(X)
    cpu_accuracy = accuracy_score(y, y_pred2)

    accuracies_cpu.append(cpu_accuracy)
    build_times_cpu.append(_perfcounter_cpu.time_taken)
    total_times_cpu.append(toc - tic)
    print(f"Accuracy for CPU: {cpu_accuracy}")
    print(f"build time for CPUs : {_perfcounter_cpu.time_taken} s")
    print(f"total time for CPUs: {toc - tic} s")

    df = pd.DataFrame(
        {"DPU accuracy": accuracies_dpu, "CPU accuracy": accuracies_cpu, "Build time on DPU": build_times_dpu,
         "DPU runtime": dpu_times_dpu, "Build time on CPU": build_times_cpu, "Total time on DPU": total_times_dpu,
         "Total time on CPU": total_times_cpu},
        index=ndpu_list[:i_ndpu + 1])

    df.to_csv("weak_scaling.csv")
