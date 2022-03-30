#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob

import pandas as pd

all_files = glob.glob("tasklets_*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, header=0, index_col=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=False)
frame.index.rename("tasklets")
frame.sort_index(inplace=True)

frame.to_csv("tasklets.csv")
frame.to_pickle("tasklets.pkl")
