"""
Converts the hashes features in the Criteo dataset to numerical features.
"""

import sys
import os
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

NR_DAYS_IN_TRAIN_SET = 2
NR_DAYS_IN_TEST_SET = NR_DAYS_IN_TRAIN_SET


def target_mean_train(data, feature_colname, target_colname):
    if data[feature_colname].dtype.kind not in ("i", "O"):
        # Non-categorical variables are kept untransformed:
        raise ValueError("Non-categorical variable encountered")

    data = data[[feature_colname, target_colname]]
    target_means = data.groupby(feature_colname).mean().astype(np.float32)
    return target_means


def target_mean_transform(data, target_means, feature_colname, target_colname="0"):
    encoded_col = dd.merge(
        data[feature_colname].to_frame(),
        target_means,
        left_on=feature_colname,
        right_index=True,
        how="left",
    )[target_colname].to_frame()
    new_colname = feature_colname + "_mean_" + target_colname
    encoded_col = encoded_col.rename(columns={target_colname: new_colname})

    # Hack: left join should preserve divisions which are required for
    # efficient downstream concat with axis=1 (but is it always true?).
    encoded_col.divisions = data.divisions
    return encoded_col


def encode_with_target_mean(data, target_means_dict, target_colname="0"):
    """Supervised encoding of categorical variables with per-group target mean.
    All columns that contain integer values are replaced by real valued data
    representing the average target value for each category.
    """
    features_data = data.drop(target_colname, axis=1)
    target_data = data[target_colname].to_frame()

    # Sequential processing of columns: there is no need to parallelize
    # column-wise as the inner-operation will already parallelize record-wise.
    encoded_columns = [
        target_mean_transform(data, target_means_dict[colname], colname, target_colname)
        if colname in target_means_dict
        else data[colname]
        for colname in features_data.columns
    ]
    return dd.concat([target_data] + encoded_columns, axis=1)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        criteo_folder = sys.argv[1]
    else:
        criteo_folder = "/scratch/sbrocard/datasets/criteo_serialized"

    print("converting training days")

    df = dd.concat(
        [
            dd.read_parquet(os.path.join(criteo_folder, f"day_{i}.pq"))
            for i in range(NR_DAYS_IN_TRAIN_SET)
        ],
        ignore_index=True,
    )

    print("read")
    df = df.fillna(0)

    target_means_dict = {
        str(i): target_mean_train(df, str(i), "0") for i in range(14, 40)
    }
    global_mean = df["0"].mean()

    print("fitted")

    df_encoded = encode_with_target_mean(df, target_means_dict, target_colname="0")

    print("encoded")

    with ProgressBar():
        df_encoded.to_parquet(
            "/scratch/sbrocard/datasets/criteo_encoded/"
            + f"train_day_0_to_{NR_DAYS_IN_TRAIN_SET-1}.pq",
            overwrite=True,
            write_index=False,
            compression=None,
        )

    del df_encoded
    del df

    print("converting test days")

    df = dd.concat(
        [
            dd.read_parquet(os.path.join(criteo_folder, f"day_{i}.pq"))
            for i in range(
                NR_DAYS_IN_TRAIN_SET, NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET
            )
        ],
        ignore_index=True,
    )

    print("read")
    df = df.fillna(0)

    df_encoded = encode_with_target_mean(df, target_means_dict, target_colname="0")
    # filling with the global target mean in case some hashes in the test set are not in the train set
    df_encoded = df_encoded.fillna(global_mean)

    with ProgressBar():
        df_encoded.to_parquet(
            "/scratch/sbrocard/datasets/criteo_encoded/"
            + f"test_day_{NR_DAYS_IN_TRAIN_SET}"
            + f"_to_{NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET - 1}.pq",
            overwrite=True,
            write_index=False,
            compression=None,
        )
