from time import perf_counter

import pandas as pd
from hurry.filesize import size

from skdpu.tree import _perfcounter as _perfcounter_dpu
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _perfcounter as _perfcounter_cpu

ndpu = 1
train_size_per_dpu = int(6e5)
test_size_per_dpu = int(1e5)
npoints_per_dpu = train_size_per_dpu + test_size_per_dpu
nfeatures = 16
random_state_set = list(range(42))

accuracies_dpu = []
accuracies_cpu = []

build_times_dpu = []
build_times_cpu = []

total_times_dpu = []
total_times_cpu = []

init_times_dpu = []
pim_cpu_times = []
inter_pim_core_times = []
cpu_pim_times = []
dpu_kernel_times = []

for i_random_state, random_state in enumerate(random_state_set):
    print(f"number of points per dpu : {train_size_per_dpu}")
    data_size = train_size_per_dpu * (nfeatures + 1) * 4
    print(f"data size per dpu= {size(data_size)}")

    X, y = make_classification(n_samples=npoints_per_dpu * ndpu, n_features=nfeatures, n_informative=4, n_redundant=4,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size_per_dpu * ndpu, shuffle=False)

    clf = DecisionTreeClassifierDpu(random_state=0, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu,
                                    max_depth=10)

    tic = perf_counter()
    clf.fit(X_train, y_train)
    toc = perf_counter()
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X_train)
    dpu_accuracy = accuracy_score(y_train, y_pred)

    # read DPU times
    accuracies_dpu.append(dpu_accuracy)
    build_times_dpu.append(_perfcounter_dpu.time_taken)
    total_times_dpu.append(toc - tic)
    init_times_dpu.append(_perfcounter_dpu.dpu_init_time)
    pim_cpu_times.append(0.0)  # there is no final transfer for trees
    inter_pim_core_times.append(_perfcounter_dpu.time_taken - _perfcounter_dpu.dpu_time)
    cpu_pim_times.append(_perfcounter_dpu.cpu_pim_time)
    dpu_kernel_times.append(_perfcounter_dpu.dpu_time)
    print(f"Accuracy for DPUs: {dpu_accuracy}")
    print(f"build time for DPUs : {_perfcounter_dpu.time_taken} s")
    print(f"total time for DPUs: {toc - tic} s")

    clf2 = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='random', max_depth=10)

    tic = perf_counter()
    clf2.fit(X_train, y_train)
    toc = perf_counter()
    # export_graphviz(clf2, out_file="tree_cpu.dot")
    y_pred2 = clf2.predict(X_train)
    cpu_accuracy = accuracy_score(y_train, y_pred2)

    # read CPU times
    accuracies_cpu.append(cpu_accuracy)
    build_times_cpu.append(_perfcounter_cpu.time_taken)
    total_times_cpu.append(toc - tic)
    print(f"Accuracy for CPU: {cpu_accuracy}")
    print(f"build time for CPUs : {_perfcounter_cpu.time_taken} s")
    print(f"total time for CPUs: {toc - tic} s")

    df = pd.DataFrame(
        {
            "DPU accuracy": accuracies_dpu,
            "Total time on DPU": total_times_dpu,
            "Build time on DPU": build_times_dpu,
            "DPU kernel time": dpu_kernel_times,
            "DPU allocation time": init_times_dpu,
            "PIM-CPU time": pim_cpu_times,
            "Inter PIM core time": inter_pim_core_times,
            "CPU-PIM time": cpu_pim_times,
            "CPU accuracy": accuracies_cpu,
            "Total time on CPU": total_times_cpu,
            "Build time on CPU": build_times_cpu,
        },
        index=random_state_set[:i_random_state + 1])

    df.to_csv("precision_train.csv")
