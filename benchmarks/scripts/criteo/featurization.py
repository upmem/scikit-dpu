"""
Converts the hashes features in the Criteo dataset to numerical features.
"""

import sys
import os
import pandas as pd
import numpy as np
from category_encoders import MEstimateEncoder

NR_DAYS_IN_TRAIN_SET = 2
NR_DAYS_IN_TEST_SET = NR_DAYS_IN_TRAIN_SET

if len(sys.argv) >= 2:
    criteo_folder = sys.argv[1]
else:
    criteo_folder = "~/datasets/criteo"

print("converting training days")

typedict = dict(
    [[i, np.float32] for i in range(14)] + [[i, str] for i in range(14, 40)]
)

df = pd.concat(
    [
        pd.read_csv(
            os.path.join(criteo_folder, f"day_{i}.gz"),
            dtype=typedict,
            header=None,
            sep="\t",
            compression="gzip",
            # nrows=1000,
        )
        for i in range(NR_DAYS_IN_TRAIN_SET)
    ],
    ignore_index=True,
)

print("read")

y = df.iloc[:, 0]

enc = MEstimateEncoder(m=0).fit(df, y)
df = enc.transform(df)

values = dict([i, 0] for i in range(14))
df.fillna(value=values, inplace=True)

print("encoded")

df.columns = df.columns.astype(str)
df.to_parquet(
    f"./data/train_day_0_to_{NR_DAYS_IN_TRAIN_SET-1}.pq", index=False, compression=None
)

print("converting test days")

df = pd.concat(
    [
        pd.read_csv(
            os.path.join(criteo_folder, f"day_{i}.gz"),
            dtype=typedict,
            header=None,
            sep="\t",
            compression="gzip",
            nrows=1000,
        )
        for i in range(NR_DAYS_IN_TRAIN_SET, NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET)
    ],
    ignore_index=True,
)

df = enc.transform(df)
df.fillna(value=values, inplace=True)
df.columns = df.columns.astype(str)
df.to_parquet(
    f"./data/test_day_{NR_DAYS_IN_TRAIN_SET}_to_{NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET - 1}.pq",
    index=False,
    compression=None,
)
