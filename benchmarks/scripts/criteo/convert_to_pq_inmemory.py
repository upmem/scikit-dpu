"""
Converts the Criteo dataset CSV to a parquet file for faster access.
"""

import pandas as pd
import numpy as np

SUBSAMPLING_FACTOR = 2

typedict = dict(
    [[i, np.float32] for i in range(14)] + [[i, str] for i in range(14, 40)]
)
# values = dict([str(i), 0] for i in range(14))

for i in range(1):
    filename = f"/scratch/sbrocard/datasets/criteo/day_{i}"
    print(f"Processing {filename}...")
    df = pd.read_csv(filename, dtype=typedict, header=None, sep="\t")
    # cut dataframes
    df = df.head(df.shape[0] // SUBSAMPLING_FACTOR)
    df.columns = df.columns.astype(str)
    # df.fillna(value=values, inplace=True, axis=0)
    df.to_parquet(
        f"/scratch/sbrocard/datasets/criteo_serialized_half/day_{i}.pq",
        index=False,
        compression=None,
    )
