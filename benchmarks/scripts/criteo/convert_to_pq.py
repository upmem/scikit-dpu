"""
Converts the Criteo dataset CSV to a parquet file for faster access.
"""

import pandas as pd
import numpy as np

for i in range(7, 14):
    filename = f"/home/upmemstaff/sbrocard/datasets/criteo/day_{i}.gz"
    df = pd.read_csv(filename, dtype=np.float32, header=None, sep="\t", usecols=range(14), compression='gzip')
    df.columns = df.columns.astype(str)
    df.fillna(0, inplace=True)
    df.to_parquet(f"./data/day_{i}.pq", index=False, compression=None)
