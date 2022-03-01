import numpy as np
import holder
from joblib import Parallel, delayed


def core_func(i:int, array: np.ndarray):
  j = holder.counter = holder.counter + 1
  array[i] = j


if __name__ == "__main__":
    numpy_array = np.zeros(500)
    Parallel(n_jobs=4, prefer="threads")(
        delayed(core_func)(repeat_index, numpy_array)
        for repeat_index in range(len(numpy_array))
    )
    print(np.mean(numpy_array))
    print(numpy_array[:20])