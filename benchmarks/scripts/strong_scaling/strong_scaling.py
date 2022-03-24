from time import perf_counter

import pandas as pd
from hurry.filesize import size

from skdpu.tree import _perfcounter as _perfcounter_dpu
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _perfcounter as _perfcounter_cpu

ndpu_list = [256, 512, 1024, 2048]
npoints = int(6e5) * 256
nfeatures = 16
random_state = 42

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

print(f"number of points : {npoints}")
data_size = npoints * (nfeatures + 1) * 4
print(f"data size = {size(data_size)}")

# X, y = make_classification(n_samples=npoints, n_features=nfeatures, random_state=random_state)
X, y = make_blobs(n_samples=npoints, n_features=nfeatures, random_state=random_state, centers=3)

clf2 = DecisionTreeClassifier(random_state=random_state, criterion='gini', splitter='random', max_depth=10)

tic = perf_counter()
# clf2.fit(X, y)
toc = perf_counter()
# export_graphviz(clf2, out_file="tree_cpu.dot")
# y_pred2 = clf2.predict(X)
cpu_accuracy = 0  # accuracy_score(y, y_pred2)

cpu_total_time = toc - tic
print(f"Accuracy for CPU: {cpu_accuracy}")
print(f"build time for CPUs : {_perfcounter_cpu.time_taken} s")
print(f"total time for CPUs: {cpu_total_time} s")

for i_ndpu, ndpu in enumerate(ndpu_list):
    clf = DecisionTreeClassifierDpu(random_state=random_state, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu,
                                    max_depth=10)

    tic = perf_counter()
    clf.fit(X, y)
    toc = perf_counter()
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X)
    dpu_accuracy = accuracy_score(y, y_pred)

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

    # read CPU times
    accuracies_cpu.append(cpu_accuracy)
    build_times_cpu.append(_perfcounter_cpu.time_taken)
    total_times_cpu.append(cpu_total_time)

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
        index=ndpu_list[:i_ndpu + 1])

    df.to_csv("strong_scaling.csv")
