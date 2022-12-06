"""
Converts the Criteo dataset CSV to a parquet file for faster access.
"""

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
import subprocess

SUBSAMPLING_FACTOR = 2

typedict = dict(
    [[i, np.float32] for i in range(14)] + [[i, str] for i in range(14, 40)]
)
# values = dict([str(i), 0] for i in range(14))

for i in range(1):
    filename = f"/scratch/sbrocard/datasets/criteo/day_{i}"
    print(f"Processing {filename}...")
    print("getting length")
    nrows = int(subprocess.check_output(["wc", "-l", filename]).decode().split()[0])
    print(f"length is {nrows}")
    df = dd.read_csv(filename, dtype=typedict, header=None, sep="\t")
    df = df.head(nrows // SUBSAMPLING_FACTOR, compute=False, npartitions=-1)
    df.columns = df.columns.astype(str)
    # df.fillna(value=values, inplace=True, axis=0)
    with ProgressBar():
        df.to_parquet(
            f"/scratch/sbrocard/datasets/criteo_serialized_half/day_{i}.pq",
            write_index=False,
            compression=None,
        )
