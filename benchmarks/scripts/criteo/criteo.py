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
from sklearn.ensemble import RandomForestClassifier
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
    os.path.join(criteo_folder, "train_day_0_" + f"to_{NR_DAYS_IN_TRAIN_SET-1}.pq")
)

print("read train dataframe")

# X = np.require(df.iloc[:, 1:].to_numpy(), dtype=np.float32, requirements=['C', 'A', 'O'])
# X_train = df_train.iloc[:, 1:].to_numpy(dtype=np.float32)
# y = np.require(df.iloc[:, 0].to_numpy(), dtype=np.float32, requirements=['C', 'A', 'O'])
# y_train = df_train.iloc[:, 0].to_numpy(dtype=np.float32)

train_size = df_train.shape[0].compute()
# train_size = df_train.shape[0]
nfeatures = df_train.shape[1]
print(f"train_size: {train_size}, nfeatures: {nfeatures}")

# X_train = np.require(X_train, dtype=np.float32, requirements=["C", "A", "O"])
# X_train = np.require(
#     df_train.iloc[:, 1:], dtype=np.float32, requirements=["C", "A", "O"]
# )
# y_train = np.require(y_train, dtype=np.float32, requirements=["C", "A", "O"])
# y_train = np.require(
#     df_train.iloc[:, 0], dtype=np.float32, requirements=["C", "A", "O"]
# )
da_train = df_train.to_dask_array().astype(np.float32)
X_train = da_train[:, 1:].compute()
y_train = da_train[:, 0].compute()

print("built train numpy arrays")

del df_train

df_test = dd.read_parquet(
    os.path.join(
        criteo_folder,
        f"test_day_{NR_DAYS_IN_TRAIN_SET}_"
        + f"to_{NR_DAYS_IN_TRAIN_SET+NR_DAYS_IN_TEST_SET-1}.pq",
    )
)

print("read test dataframe")

# X_test = df_test.iloc[:, 1:].to_numpy(dtype=np.float32)
# y_test = df_test.iloc[:, 0].to_numpy(dtype=np.float32)

# X_test = np.require(df_test.iloc[:, 1:], dtype=np.float32, requirements=["O"])
# y_test = np.require(df_test.iloc[:, 0], dtype=np.float32, requirements=["O"])

da_test = df_test.to_dask_array().astype(np.float32)
X_test = da_test[:, 1:].compute()
y_test = da_test[:, 0].compute()

print("built test numpy arrays")

del df_test

train_accuracies_dpu = []
train_accuracies_intel = []
train_accuracies_sklearn = []

train_balanced_accuracies_dpu = []
train_balanced_accuracies_intel = []
train_balanced_accuracies_sklearn = []

accuracies_dpu = []
accuracies_intel = []
accuracies_sklearn = []

balanced_accuracy_score_dpu = []
balanced_accuracy_score_intel = []
balanced_accuracy_score_sklearn = []

build_times_dpu = []
# build_times_intel = []
build_times_sklearn = []

total_times_dpu = []
total_times_intel = []
total_times_sklearn = []

init_times_dpu = []
pim_cpu_times = []
inter_pim_core_times = []
cpu_pim_times = []
dpu_kernel_times = []

ndpu_effective = []

print(f"number of points : {train_size}")
data_size = train_size * (nfeatures + 1) * 4
print(f"data size = {size(data_size)}")

clf_intel = RandomForestClassifier(
    random_state=random_state,
    criterion="gini",
    n_estimators=1,
    max_depth=10,
    bootstrap=False,
    max_features=None,
    n_jobs=-1,
)

tic = perf_counter()
clf_intel.fit(X_train, y_train)
toc = perf_counter()
print(f"Intel build time: {toc - tic}")
# export_graphviz(clf2, out_file="tree_cpu.dot")
y_pred = clf_intel.predict(X_train)
intel_train_accuracy = accuracy_score(y_train, y_pred)
intel_train_balanced_accuracy = balanced_accuracy_score(y_train, y_pred)
y_pred = clf_intel.predict(X_test)
intel_accuracy = accuracy_score(y_test, y_pred)
intel_balanced_accuracy_score = balanced_accuracy_score(y_test, y_pred)
del y_pred
del clf_intel

intel_total_time = toc - tic
print(f"Train Accuracy for Intel: {intel_train_accuracy}")
print(f"Balanced Train Accuracy for Intel: {intel_train_balanced_accuracy}")
print(f"Accuracy for Intelex: {intel_accuracy}")
print(f"Balanced accuracy score for Intelex: {intel_balanced_accuracy_score}")
# print(f"build time for Intel: {_perfcounter_cpu.time_taken} s")
print(f"total time for Intel: {toc - tic} s")

clf_sklearn = DecisionTreeClassifier(
    random_state=random_state, criterion="gini", splitter="random", max_depth=10
)

tic = perf_counter()
clf_sklearn.fit(X_train, y_train)
toc = perf_counter()
print(f"sklearn build time: {toc - tic}")
y_pred = clf_sklearn.predict(X_train)
sklearn_train_accuracy = accuracy_score(y_train, y_pred)
sklearn_train_balanced_accuracy = balanced_accuracy_score(y_train, y_pred)
y_pred = clf_sklearn.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, y_pred)
sklearn_balanced_accuracy_score = balanced_accuracy_score(y_test, y_pred)
del y_pred
del clf_sklearn

sklearn_total_time = toc - tic
print(f"Train Accuracy for sklearn: {sklearn_train_accuracy}")
print(f"Balanced Train Accuracy for sklearn: {sklearn_train_balanced_accuracy}")
print(f"Accuracy for sklearn: {sklearn_accuracy}")
print(f"Balanced accuracy score for sklearn: {sklearn_balanced_accuracy_score}")
print(f"build time for sklearn: {_perfcounter_cpu.time_taken} s")
print(f"total time for sklearn: {toc - tic} s")

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
    print(f"total time for DPU: {toc - tic}")
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X_rounded)
    dpu_train_accuracy = accuracy_score(y_rounded, y_pred)
    dpu_train_balanced_accuracy = balanced_accuracy_score(y_rounded, y_pred)
    y_pred = clf.predict(X_test)
    dpu_accuracy = accuracy_score(y_test, y_pred)
    dpu_balanced_accuracy_score = balanced_accuracy_score(y_test, y_pred)

    # read DPU times
    train_accuracies_dpu.append(dpu_train_accuracy)
    train_balanced_accuracies_dpu.append(dpu_train_balanced_accuracy)
    accuracies_dpu.append(dpu_accuracy)
    balanced_accuracy_score_dpu.append(dpu_balanced_accuracy_score)
    build_times_dpu.append(_perfcounter_dpu.time_taken)
    total_times_dpu.append(toc - tic)
    init_times_dpu.append(_perfcounter_dpu.dpu_init_time)
    pim_cpu_times.append(0.0)  # there is no final transfer for trees
    inter_pim_core_times.append(_perfcounter_dpu.time_taken - _perfcounter_dpu.dpu_time)
    cpu_pim_times.append(_perfcounter_dpu.cpu_pim_time)
    dpu_kernel_times.append(_perfcounter_dpu.dpu_time)
    print(f"Train Accuracy for DPU: {dpu_train_accuracy}")
    print(f"Balanced Train Accuracy for DPU: {dpu_train_balanced_accuracy}")
    print(f"Accuracy for DPUs: {dpu_accuracy}")
    print(f"Balanced accracy score for DPUs: {dpu_balanced_accuracy_score}")
    print(f"build time for DPUs : {_perfcounter_dpu.time_taken} s")
    print(f"total time for DPUs: {toc - tic} s")

    # read Intel times
    train_accuracies_intel.append(intel_train_accuracy)
    train_balanced_accuracies_intel.append(intel_train_balanced_accuracy)
    accuracies_intel.append(intel_accuracy)
    balanced_accuracy_score_intel.append(balanced_accuracy_score)
    # build_times_intel.append(_perfcounter_cpu.time_taken)
    total_times_intel.append(intel_total_time)

    # read sklearn times
    train_accuracies_sklearn.append(sklearn_train_accuracy)
    train_balanced_accuracies_sklearn.append(sklearn_train_balanced_accuracy)
    accuracies_sklearn.append(sklearn_accuracy)
    balanced_accuracy_score_sklearn.append(sklearn_balanced_accuracy_score)
    build_times_sklearn.append(_perfcounter_cpu.time_taken)
    total_times_sklearn.append(sklearn_total_time)

    df = pd.DataFrame(
        {
            "DPU train accuracy": train_accuracies_dpu,
            "DPU train balanced accuracy": train_balanced_accuracies_dpu,
            "DPU accuracy": accuracies_dpu,
            "DPU Balanced accracy score": balanced_accuracy_score_dpu,
            "Total time on DPU": total_times_dpu,
            "Build time on DPU": build_times_dpu,
            "DPU kernel time": dpu_kernel_times,
            "DPU allocation time": init_times_dpu,
            "PIM-CPU time": pim_cpu_times,
            "Inter PIM core time": inter_pim_core_times,
            "CPU-PIM time": cpu_pim_times,
            "Intel train accuracy": train_accuracies_intel,
            "Intel train balanced accuracy": train_balanced_accuracies_intel,
            "Intel accuracy": accuracies_intel,
            "Intel Balanced accuracy score": balanced_accuracy_score_intel,
            "Total time on Intel": total_times_intel,
            # "Build time on Intel": build_times_intel,
            "sklearn train accuracy": train_accuracies_sklearn,
            "sklearn train balanced accuracy": train_balanced_accuracies_sklearn,
            "sklearn accuracy": accuracies_sklearn,
            "sklearn Balanced accuracy score": balanced_accuracy_score_sklearn,
            "Total time on sklearn": total_times_sklearn,
            "Build time on sklearn": build_times_sklearn,
        },
        index=ndpu_effective,
    )

    df.to_csv("criteo_results.csv")
