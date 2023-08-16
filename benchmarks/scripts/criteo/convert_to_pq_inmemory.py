"""
Converts the Criteo dataset CSV to a parquet file for faster access.
"""

import subprocess
import pandas as pd
import numpy as np

SUBSAMPLING_FACTOR = 4

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
    df = pd.read_csv(filename, dtype=typedict, header=None, sep="\t", nrows=nrows // SUBSAMPLING_FACTOR)
    # cut dataframes
    # df = df.head(df.shape[0] // SUBSAMPLING_FACTOR)
    df.columns = df.columns.astype(str)
    # df.fillna(value=values, inplace=True, axis=0)
    df.to_parquet(
        f"/scratch/sbrocard/datasets/criteo_serialized_quarter/day_{i}.pq",
        index=False,
        compression=None,
    )
