"""
Converts the hashes features in the Criteo dataset to continuous features.
"""

import pandas as pd
import numpy as np

for i in range(1):
    filename = f"/home/upmemstaff/sbrocard/datasets/criteo/day_{i}.gz"
    df = pd.read_csv(filename, header=None, sep="\t", compression='gzip', nrows=1000)

df = pd.concat([pd.read_csv(filename, header=None, sep="\t", compression='gzip', nrows=1000) for i in range(2)], ignore_index=True)
df.columns = df.columns.astype(str)

for col in range(14, 40):
    df[f"count_{col}"] = df[str(col)].groupby(df[str(col)]).transform('count')
for col in range(14, 40):
    df[f"mean_{col}"] = df.groupby(str(col), as_index=False)['0'].transform('mean')

df.drop(columns=[str(i) for i in range(14, 40)], inplace=True)

df = df.astype(np.float32)


# TODO: replace with scikit-contrib category encoders




global_average = df['0'].mean()

values = dict()
for col in range(14):
    values[str(col)] = 0
for col in range(14, 40):
    values[f"mean_{col}"] = global_average
    values[f"count_{col}"] = 0

df.fillna(value=values, inplace=True)
