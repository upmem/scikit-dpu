import sys
import os
from time import perf_counter

import numpy as np
import pandas as pd
import dask.dataframe as dd
from hurry.filesize import size

from sklearnex import patch_sklearn
patch_sklearn()

from skdpu.tree import _perfcounter as _perfcounter_dpu
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _perfcounter as _perfcounter_cpu
from sklearn.utils import shuffle

MAX_FEATURE_DPU = 10000000

ndpu_list = [2048]
random_state = 42

NR_DAYS_IN_TRAIN_SET = 2
NR_DAYS_IN_TEST_SET = NR_DAYS_IN_TRAIN_SET

if len(sys.argv) >= 2:
    criteo_folder = sys.argv[1]
else:
    criteo_folder = "data"
print("reading data from ", criteo_folder)

df_train = dd.read_parquet(
    os.path.join(criteo_folder, f"train_day_0_to_{NR_DAYS_IN_TRAIN_SET-1}.pq")
)

print("read train dataframe")

# X = np.require(df.iloc[:, 1:].to_numpy(), dtype=np.float32, requirements=['C', 'A', 'O'])
# X_train = df_train.iloc[:, 1:].to_numpy(dtype=np.float32)
# y = np.require(df.iloc[:, 0].to_numpy(), dtype=np.float32, requirements=['C', 'A', 'O'])
# y_train = df_train.iloc[:, 0].to_numpy(dtype=np.float32)

train_size = df_train.shape[0].compute()
nfeatures = df_train.shape[1]
print(f"train_size: {train_size}, nfeatures: {nfeatures}")

# X_train = np.require(X_train, dtype=np.float32, requirements=["C", "A", "O"])
X_train = np.require(df_train.iloc[:, 1:], dtype=np.float32, requirements=["C", "A", "O"])
# y_train = np.require(y_train, dtype=np.float32, requirements=["C", "A", "O"])
y_train = np.require(df_train.iloc[:, 0], dtype=np.float32, requirements=["C", "A", "O"])

print("built train numpy arrays")

del df_train

df_test = dd.read_parquet(
    os.path.join(
        criteo_folder,
        f"test_day_{NR_DAYS_IN_TRAIN_SET}_to_{NR_DAYS_IN_TRAIN_SET+NR_DAYS_IN_TEST_SET-1}.pq",
    )
)

print("read test dataframe")

# X_test = df_test.iloc[:, 1:].to_numpy(dtype=np.float32)
# y_test = df_test.iloc[:, 0].to_numpy(dtype=np.float32)

X_test = np.require(df_test.iloc[:, 1:], dtype=np.float32, requirements=["O"])
y_test = np.require(df_test.iloc[:, 0], dtype=np.float32, requirements=["O"])

print("built test numpy arrays")

del df_test

accuracies_dpu = []
accuracies_cpu = []

balanced_accuracy_score_dpu = []
balanced_accuracy_score_cpu = []

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

clf2 = DecisionTreeClassifier(
    random_state=random_state, criterion="gini", splitter="random", max_depth=10
)

tic = perf_counter()
clf2.fit(X_train, y_train)
toc = perf_counter()
print(f"CPU build time: {toc - tic}")
# export_graphviz(clf2, out_file="tree_cpu.dot")
y_pred2 = clf2.predict(X_test)
cpu_accuracy = accuracy_score(y_test, y_pred2)
cpu_balanced_accuracy_score = balanced_accuracy_score(y_test, y_pred2)

cpu_total_time = toc - tic
print(f"Accuracy for CPU: {cpu_accuracy}")
print(f"Balanced accuracy score for CPU: {cpu_balanced_accuracy_score}")
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

    clf = DecisionTreeClassifierDpu(
        random_state=None,
        criterion="gini_dpu",
        splitter="random_dpu",
        ndpu=ndpu,
        max_depth=10,
    )

    npoints_rounded = npoints_per_dpu * ndpu
    print(f"npoints_rounded : {npoints_rounded}")
    X_rounded, y_rounded = X_train[:npoints_rounded], y_train[:npoints_rounded]
    tic = perf_counter()
    clf.fit(X_rounded, y_rounded)
    toc = perf_counter()
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X_test)
    dpu_accuracy = accuracy_score(y_test, y_pred)
    dpu_balanced_accuracy_score = balanced_accuracy_score(y_test, y_pred)

    # read DPU times
    accuracies_dpu.append(dpu_accuracy)
    balanced_accuracy_score_dpu.append(dpu_balanced_accuracy_score)
    build_times_dpu.append(_perfcounter_dpu.time_taken)
    total_times_dpu.append(toc - tic)
    init_times_dpu.append(_perfcounter_dpu.dpu_init_time)
    pim_cpu_times.append(0.0)  # there is no final transfer for trees
    inter_pim_core_times.append(_perfcounter_dpu.time_taken - _perfcounter_dpu.dpu_time)
    cpu_pim_times.append(_perfcounter_dpu.cpu_pim_time)
    dpu_kernel_times.append(_perfcounter_dpu.dpu_time)
    print(f"Accuracy for DPUs: {dpu_accuracy}")
    print(f"Balanced accracy score for DPUs: {dpu_balanced_accuracy_score}")
    print(f"build time for DPUs : {_perfcounter_dpu.time_taken} s")
    print(f"total time for DPUs: {toc - tic} s")

    # read CPU times
    accuracies_cpu.append(cpu_accuracy)
    balanced_accuracy_score_cpu.append(balanced_accuracy_score)
    build_times_cpu.append(_perfcounter_cpu.time_taken)
    total_times_cpu.append(cpu_total_time)

    df = pd.DataFrame(
        {
            "DPU accuracy": accuracies_dpu,
            "DPU Balanced accracy score": balanced_accuracy_score_dpu,
            "Total time on DPU": total_times_dpu,
            "Build time on DPU": build_times_dpu,
            "DPU kernel time": dpu_kernel_times,
            "DPU allocation time": init_times_dpu,
            "PIM-CPU time": pim_cpu_times,
            "Inter PIM core time": inter_pim_core_times,
            "CPU-PIM time": cpu_pim_times,
            "CPU accuracy": accuracies_cpu,
            "CPU Balanced accuracy score": balanced_accuracy_score_cpu,
            "Total time on CPU": total_times_cpu,
            "Build time on CPU": build_times_cpu,
        },
        index=ndpu_effective,
    )

    df.to_csv("criteo_results.csv")
