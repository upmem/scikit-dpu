from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from time import perf_counter
from sklearn.tree import _perfcounter as _perfcounter_cpu
from skdpu.tree import _perfcounter as _perfcounter_dpu
from hurry.filesize import size
import pandas as pd

ndpu_list = [256, 512, 1024, 2048]
npoints = int(1e5) * 256
nfeatures = 16

accuracies_dpu = []
accuracies_cpu = []
build_times_dpu = []
build_times_cpu = []
total_times_dpu = []
total_times_cpu = []

print(f"number of points : {npoints}")
data_size = npoints * (nfeatures + 1) * 4
print(f"data size = {size(data_size)}")

X, y = make_blobs(n_samples=npoints, n_features=nfeatures, centers=3, random_state=0,
                  center_box=(-1, 1))

clf2 = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='random', max_depth=10)

tic = perf_counter()
clf2.fit(X, y)
toc = perf_counter()
# export_graphviz(clf2, out_file="tree_cpu.dot")
y_pred2 = clf2.predict(X)
cpu_accuracy = accuracy_score(y, y_pred2)

cpu_total_time = toc - tic
print(f"Accuracy for CPU: {cpu_accuracy}")
print(f"build time for CPUs : {_perfcounter_cpu.time_taken} s")
print(f"total time for CPUs: {cpu_total_time} s")

for i_ndpu, ndpu in enumerate(ndpu_list):
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
        index=ndpu_list[:i_ndpu+1])

    df.to_csv("strong_scaling.csv")
