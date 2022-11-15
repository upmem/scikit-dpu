"""
Converts the hashes features in the Criteo dataset to numerical features.
"""

import sys
import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from category_encoders import MEstimateEncoder

NR_DAYS_IN_TRAIN_SET = 2
NR_DAYS_IN_TEST_SET = NR_DAYS_IN_TRAIN_SET

if len(sys.argv) >= 2:
    criteo_folder = sys.argv[1]
else:
    criteo_folder = "/scratch/sbrocard/datasets/criteo_serialized"

print("converting training days")

typedict = dict(
    [[i, np.float32] for i in range(14)] + [[i, str] for i in range(14, 40)]
)

ddf = dd.concat(
    [
        dd.read_parquet(
            os.path.join(criteo_folder, f"day_{i}.pq")
        )
        for i in range(NR_DAYS_IN_TRAIN_SET)
    ],
    ignore_index=True,
)

with ProgressBar():
    df = ddf.compute()

print("read")

y = df.iloc[:, 0]

enc = MEstimateEncoder(m=0).fit(df, y)

print("fitted")

df = enc.transform(df)

values = dict([str(i), 0] for i in range(14))
df.fillna(value=values, inplace=True)

print("encoded")

df.columns = df.columns.astype(str)
df.to_parquet(
    f"/scratch/sbrocard/datasets/criteo_encoded/train_day_0_to_{NR_DAYS_IN_TRAIN_SET-1}.pq", index=False, compression=None
)

print("converting test days")

ddf = dd.concat(
    [
        dd.read_parquet(
            os.path.join(criteo_folder, f"day_{i}.pq")
        )
        for i in range(NR_DAYS_IN_TRAIN_SET, NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET)
    ],
    ignore_index=True,
)

with ProgressBar():
    df = ddf.compute()

print("read")

df = enc.transform(df)
df.fillna(value=values, inplace=True)

print("encoded")

df.columns = df.columns.astype(str)
df.to_parquet(
    f"/scratch/sbrocard/datasets/criteo_encoded/test_day_{NR_DAYS_IN_TRAIN_SET}_to_{NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET - 1}.pq",
    index=False,
    compression=None,
)
