"""
Converts the Higgs dataset CSV to a parquet file for faster access.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/HIGGS.csv", dtype=np.float32, header=None, sep=",")
df.columns = df.columns.astype(str)
df.to_parquet("./data/higgs.pq", index=False, compression=None)
