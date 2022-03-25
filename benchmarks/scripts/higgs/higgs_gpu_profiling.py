from time import perf_counter
import cProfile
import sys

import numpy as np
import pandas as pd
from hurry.filesize import size

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.utils import shuffle

import cudf
from cuml.ensemble import RandomForestClassifier as cuRFC

train_size = int(6e5) * 256
test_size = int(1e5) * 256
npoints = train_size + test_size
nfeatures = 16
random_state = 42

##################################################
#                   DATA READ                    #
##################################################

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

print(f"number of points : {train_size}")
data_size = train_size * (nfeatures + 1) * 4
print(f"data size = {size(data_size)}")

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
cProfile.run("clf.fit(X_train, y_train)", f"higgs_gpu.prof")
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
cProfile.run("clf2.fit(X_train, y_train)", f"higgs_cpu.prof")
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
)

df.to_csv("higgs_gpu.csv")
