import sys
import os
from time import perf_counter
from typing import Literal, get_args

import numpy as np
import pandas as pd
import dask.dataframe as dd
from hurry.filesize import size

from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.base import BaseEstimator
from skdpu.tree import _perfcounter as _perfcounter_dpu
from skdpu.tree._classes import DecisionTreeClassifierDpu
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import _perfcounter as _perfcounter_cpu
from sklearn.ensemble import RandomForestClassifier

MAX_FEATURE_DPU = 10000000

NDPU_LIST = [256, 512, 1024, 2048]
RANDOM_STATE = 42

NR_DAYS_IN_TRAIN_SET = 1
NR_DAYS_IN_TEST_SET = 0
DO_TEST = NR_DAYS_IN_TEST_SET > 0

# define the different algorithms/accelerators
BACKENDS_NAME = Literal["sklearn", "intel", "dpu"]
BACKENDS = get_args(BACKENDS_NAME)
# define the figures of merit for the different algorithms/accelerators
FOM = {
    "common": [
        "train accuracy",
        "train balanced accuracy",
        "total time",
    ],
    "test": ["test accuracy", "test balanced accuracy"],
    "sklearn": ["build time"],
    "intel": [],
    "dpu": [
        "ndpu",
        "build time",
        "allocation time",
        "PIM-CPU time",
        "inter PIM core time",
        "CPU-PIM time",
        "DPU kernel time",
    ],
}
assert FOM.keys() - {"common", "test"} == set(BACKENDS)

KIND_NAME = Literal["train", "test"]


def get_filename(train: bool) -> str:
    start_index = 0 if train else NR_DAYS_IN_TRAIN_SET
    end_index = (
        NR_DAYS_IN_TRAIN_SET - 1
        if train
        else NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET - 1
    )
    filename_prefix = "train_" if train else "test_"
    filename = filename_prefix + f"day_{start_index}_to_{end_index}.pq"
    return filename


def get_dataset(
    criteo_folder: str, train: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    filename = get_filename(train)
    df = dd.read_parquet(os.path.join(criteo_folder, filename))
    print("read dataframe")

    da = df.to_dask_array().astype(np.float32)
    X = da[:, 1:].compute()
    y = da[:, 0].compute()

    size, nfeatures = X.shape
    nfeatures += 1  # add 1 for the target column
    if train:
        print(f"train size: {size}, nfeatures: {nfeatures}")

    # data needs to be c-contiguous for DPU
    if train:
        flags = X.flags and y.flags
        if not flags.c_contiguous or not flags.aligned or not flags.owndata:
            print("converting numpy arrays to a compatible format")
            X = np.require(X, dtype=np.float32, requirements=["C", "A", "O"])
            y = np.require(y, dtype=np.float32, requirements=["C", "A", "O"])

    print("built train numpy arrays")
    return X, y, size, nfeatures


class BenchmarkRecord:
    def __init__(self, name: BACKENDS_NAME, do_test: bool = False):
        self.name = name
        assert name in BACKENDS

        foms = FOM["common"] + (FOM["test"] if do_test else []) + FOM[name]
        self.record = {fom: [] for fom in foms}

        self.do_test = do_test

    def get_accuracies(
        self,
        kind: KIND_NAME,
        name: BACKENDS_NAME,
        clf: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        assert kind in get_args(KIND_NAME)

        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        self.record[f"{kind} accuracy"].append(accuracy)
        print(f"{kind} Accuracy for {name}: {accuracy}")

        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        self.record[f"{kind} balanced accuracy"].append(balanced_accuracy)
        print(f"Balanced {kind} Accuracy for {name}: {balanced_accuracy}")

    def benchmark_classifier(
        self,
        clf: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        *,
        export_graph: bool = False,
        ndpu: int = None,
    ) -> None:
        tic = perf_counter()
        clf.fit(X_train, y_train)
        toc = perf_counter()

        total_time = toc - tic
        self.record["total time"].append(total_time)
        print(f"total time for {self.name}: {total_time} s")

        if export_graph:
            export_graphviz(clf, out_file=f"tree_{self.name}.dot")

        self.get_accuracies("train", self.name, clf, X_train, y_train)

        if self.do_test:
            self.get_accuracies("test", self.name, clf, X_test, y_test)

        if self.name == "sklearn":
            build_time = _perfcounter_cpu.time_taken
            self.record["build time"].append(build_time)
            print(f"build time for sklearn: {build_time} s")

        elif self.name == "dpu":
            build_time = _perfcounter_dpu.time_taken
            self.record["build time"].append(build_time)
            print(f"build time for DPU: {build_time} s")

            init_time = _perfcounter_dpu.dpu_init_time
            self.record["allocation time"].append(init_time)

            pim_cpu_time = 0.0  # there is no final transfer for trees
            self.record["PIM-CPU time"].append(pim_cpu_time)

            inter_pim_core_time = (
                _perfcounter_dpu.time_taken - _perfcounter_dpu.dpu_time
            )
            self.record["inter PIM core time"].append(inter_pim_core_time)

            cpu_pim_time = _perfcounter_dpu.cpu_pim_time
            self.record["CPU-PIM time"].append(cpu_pim_time)

            dpu_kernel_time = _perfcounter_dpu.dpu_time
            self.record["DPU kernel time"].append(dpu_kernel_time)

            self.record["ndpu"].append(ndpu)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.record).add_prefix(f"{self.name} ")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        criteo_folder = sys.argv[1]
    else:
        criteo_folder = "data"
    print("reading data from ", criteo_folder)

    X_train, y_train, train_size, nfeatures = get_dataset(criteo_folder)

    if DO_TEST:
        X_test, y_test, _, nfeatures_test = get_dataset(criteo_folder, train=False)
        assert nfeatures == nfeatures_test
    else:
        X_test, y_test = None, None

    sklearn_record = BenchmarkRecord("sklearn", do_test=DO_TEST)
    intel_record = BenchmarkRecord("intel", do_test=DO_TEST)
    dpu_record = BenchmarkRecord("dpu", do_test=DO_TEST)

    print(f"number of points : {train_size}")
    data_size = train_size * (nfeatures + 1) * 4
    print(f"data size = {size(data_size)}")

    print("starting intelex benchmark")
    clf_intel = RandomForestClassifier(
        random_state=RANDOM_STATE,
        criterion="gini",
        n_estimators=1,
        max_depth=10,
        bootstrap=False,
        max_features=None,
        n_jobs=-1,
    )
    intel_record.benchmark_classifier(clf_intel, X_train, y_train, X_test, y_test)

    print("starting sklearn benchmark")
    clf_sklearn = DecisionTreeClassifier(
        random_state=RANDOM_STATE, criterion="gini", splitter="random", max_depth=10
    )
    sklearn_record.benchmark_classifier(clf_sklearn, X_train, y_train, X_test, y_test)

    print("starting DPU benchmarks")
    for i_ndpu, ndpu in enumerate(NDPU_LIST):
        print(f"\nnumber of dpus : {ndpu}")
        npoints_per_dpu = 2 * (train_size // (2 * ndpu))
        data_size = npoints_per_dpu * (nfeatures + 1) * 4
        print(f"data size per dpu= {size(data_size)}")
        if npoints_per_dpu * nfeatures > MAX_FEATURE_DPU:
            print("not enough DPUs, skipping")
            continue

        clf_dpu = DecisionTreeClassifierDpu(
            random_state=None,
            criterion="gini_dpu",
            splitter="random_dpu",
            ndpu=ndpu,
            max_depth=10,
        )

        npoints_rounded = npoints_per_dpu * ndpu
        print(f"npoints_rounded : {npoints_rounded}")
        X_rounded, y_rounded = X_train[:npoints_rounded], y_train[:npoints_rounded]

        dpu_record.benchmark_classifier(
            clf_dpu, X_rounded, y_rounded, X_test, y_test, ndpu=ndpu
        )

        # make the result DataFrame at every iteration in case we terminate early
        result = pd.concat(
            (sklearn_record.to_dataframe(), intel_record.to_dataframe()), axis=1
        )
        result = dpu_record.to_dataframe().merge(result, how="cross").set_index("dpu ndpu")

        result.to_csv("criteo_results.csv")
