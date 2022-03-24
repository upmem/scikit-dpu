import sys
from time import perf_counter

import numpy as np
import pandas as pd
from hurry.filesize import size

from skdpu.tree import _perfcounter as _perfcounter_dpu
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _perfcounter as _perfcounter_cpu
from sklearn.utils import shuffle

MAX_FEATURE_DPU = 10000000

ndpu_list = [64, 128, 256, 512, 1024, 2048, 2524]
random_state = 42

if len(sys.argv) >= 2:
    higgs_file = sys.argv[1]
else:
    higgs_file = "data/higgs.pq"
df = pd.read_parquet(higgs_file)

X, y = np.require(df.iloc[:, 1:].to_numpy(), requirements=['C', 'A', 'O']), np.require(df.iloc[:, 0].to_numpy(),
                                                                                       requirements=['C', 'A', 'O'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500000, shuffle=False)

train_size, nfeatures = X_train.shape

del df

X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

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

ndpu_effective = []

print(f"number of points : {train_size}")
data_size = train_size * (nfeatures + 1) * 4
print(f"data size = {size(data_size)}")

clf2 = DecisionTreeClassifier(random_state=random_state, criterion='gini', splitter='random', max_depth=10)

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
    npoints_per_dpu = 2 * (train_size // (2 * ndpu))
    if npoints_per_dpu * nfeatures > MAX_FEATURE_DPU:
        print("not enough DPUs, skipping")
        continue
    ndpu_effective.append(ndpu)
    data_size = npoints_per_dpu * (nfeatures + 1) * 4
    print(f"data size per dpu= {size(data_size)}")

    clf = DecisionTreeClassifierDpu(random_state=None, criterion='gini_dpu', splitter='random_dpu', ndpu=ndpu,
                                    max_depth=10)

    npoints_rounded = npoints_per_dpu * ndpu
    print(f"npoints_rounded : {npoints_rounded}")
    X_rounded, y_rounded = X_train[:npoints_rounded], y_train[:npoints_rounded]
    tic = perf_counter()
    clf.fit(X_rounded, y_rounded)
    toc = perf_counter()
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X_test)
    dpu_accuracy = accuracy_score(y_test, y_pred)

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
        index=ndpu_effective)

    df.to_csv("higgs_results.csv")
