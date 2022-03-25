from time import perf_counter

import numpy as np
import pandas as pd
from hurry.filesize import size

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC

import cudf
from cuml.ensemble import RandomForestClassifier as cuRFC

ndpu_list = [1, 4, 16, 64]
train_size_per_dpu = int(6e5)
test_size_per_dpu = int(1e5)
npoints_per_dpu = train_size_per_dpu + test_size_per_dpu
nfeatures = 16
random_state = 42

accuracies_gpu = []
accuracies_cpu = []

total_times_gpu = []
total_times_cpu = []

init_times_gpu = []
transfer_times_gpu = []

def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({"fea%d" % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf

# load empty data to the GPU to pay the initialization cost
tic = perf_counter()
gpu_data = np2cudf(np.zeros((2, 2)))
toc = perf_counter()
init_time_gpu = toc - tic
del gpu_data

for i_ndpu, ndpu in enumerate(ndpu_list):
    print(f"number of points per dpu : {train_size_per_dpu}")
    data_size = train_size_per_dpu * (nfeatures + 1) * 4
    print(f"data size per dpu= {size(data_size)}")

    ##################################################
    #                   DATA GEN                     #
    ##################################################

    X, y = make_classification(n_samples=npoints_per_dpu * ndpu, n_features=nfeatures, n_informative=4, n_redundant=4,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size_per_dpu * ndpu, shuffle=False)

    ##################################################
    #                   GPU PERF                     #
    ##################################################

    # load the data to the GPU
    tic = perf_counter()
    X_train_gpu = np2cudf(X_train)
    y_train_gpu = cudf.DataFrame(y_train)
    toc = perf_counter()
    transfer_time_gpu = toc - tic

    clf = cuRFC(n_estimators=1, random_state=random_state, split_criterion='gini', max_depth=10, bootstrap=False, max_features=1.0)

    tic = perf_counter()
    clf.fit(X_train, y_train)
    toc = perf_counter()
    # export_graphviz(clf, out_file="tree_dpu.dot")
    y_pred = clf.predict(X_test)
    gpu_accuracy = accuracy_score(y_test, y_pred)

    # read GPU times
    accuracies_gpu.append(gpu_accuracy)
    total_times_gpu.append(toc - tic)
    init_times_gpu.append(init_time_gpu)
    transfer_times_gpu.append(transfer_time_gpu)
    print(f"Accuracy for DPUs: {gpu_accuracy}")
    print(f"total time for DPUs: {toc - tic} s")

    ##################################################
    #                   CPU PERF                     #
    ##################################################

    clf2 = RFC(n_estimators=1, random_state=random_state, criterion='gini', max_depth=10, bootstrap=False, max_features=1.0)

    tic = perf_counter()
    clf2.fit(X, y)
    toc = perf_counter()
    # export_graphviz(clf2, out_file="tree_cpu.dot")
    y_pred2 = clf2.predict(X)
    cpu_accuracy = accuracy_score(y, y_pred2)

    # read CPU times
    accuracies_cpu.append(cpu_accuracy)
    total_times_cpu.append(toc - tic)
    print(f"Accuracy for CPU: {cpu_accuracy}")
    print(f"total time for CPUs: {toc - tic} s")

    df = pd.DataFrame(
        {
            "GPU_times": total_times_gpu,
            "GPU_init_time": init_times_gpu,
            "GPU_transfer_times": transfer_times_gpu,
            "CPU_times": total_times_cpu,
            "GPU_scores": gpu_accuracy,
            "CPU_scores": cpu_accuracy,
        },
        index=ndpu_list[:i_ndpu + 1])

    df.to_csv("weak_scaling_gpu.csv")
