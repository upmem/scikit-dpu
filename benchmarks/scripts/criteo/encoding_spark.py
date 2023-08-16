"""
Converts the hashes features in the Criteo dataset to numerical features.
"""

import os
import sys
from functools import reduce

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

NR_DAYS_IN_TRAIN_SET = 1
NR_DAYS_IN_TEST_SET = 0

TARGET_COL = "0"

def union_all(*dfs: DataFrame) -> DataFrame:
    return reduce(DataFrame.union, dfs)


def configure_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .config("spark.driver.host", "127.0.0.1")
        # .config("spark.executor.memory", "10g")
        .config("spark.driver.memory", "256g")
        .config("spark.local.dir", "/scratch/sbrocard/sparklocaldir")
        # .config("spark.driver.maxResultSize", "150g")
        .appName("Criteo")
        .getOrCreate()
    )
    return spark


def target_mean_encoding(
    df_train: DataFrame, df_test: DataFrame, col, target
) -> tuple[DataFrame]:
    """
    :param df_train: pyspark.sql.dataframe
        dataframe to fit and apply target mean encoding
    :param df_test: pyspark.sql.dataframe
        dataframe to apply target mean encoding
    :param col: str list
        list of columns to apply target encoding
    :param unchanged_col: str list
        list of columns to keep unchanged
    :param target: str
        target column
    :return: tuple of pyspark.sql.dataframe
        dataframes with target encoded columns
    """
    # target_encoded_columns_list = []
    df_train_encoded = df_train
    df_test_encoded = df_test
    for c in col:
        print(f"column: {c}")
        means = df_train.groupby(F.col(c)).agg(
            F.mean(target).astype('float').alias(f"{c}_mean_encoding")
        )
        df_train_encoded = df_train_encoded.join(means, on=c, how="left").drop(c)
        if df_test is not None:
            df_test_encoded = df_test_encoded.join(means, on=c, how="left").drop(c)
    return (
        df_train_encoded,
        df_test_encoded,
    )


if __name__ == "__main__":
    # Set up Spark
    spark = configure_spark()

    if len(sys.argv) >= 2:
        criteo_folder = sys.argv[1]
    else:
        criteo_folder = "/scratch/sbrocard/datasets/criteo_serialized"

    print("reading training days")

    df_train = union_all(
        *[
            spark.read.parquet(os.path.join(criteo_folder, f"day_{i}.pq"))
            for i in range(NR_DAYS_IN_TRAIN_SET)
        ]
    )

    print("read")
    df_train = df_train.fillna(0)
    df_train = df_train.fillna("")

    global_mean = df_train.select(F.mean(TARGET_COL)).collect()[0][0]

    print(f"got global mean: {global_mean}")

    if NR_DAYS_IN_TEST_SET > 0:
        print("reading test days")

        df_test = union_all(
            *[
                spark.read.parquet(os.path.join(criteo_folder, f"day_{i}.pq"))
                for i in range(
                    NR_DAYS_IN_TRAIN_SET, NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET
                )
            ]
        )

        print("read")
        df_test = df_test.fillna(0)
        df_train = df_train.fillna("")
    else:
        df_test = None

    df_train_encoded, df_test_encoded = target_mean_encoding(
        df_train, df_test, [str(i) for i in range(14, 40)], TARGET_COL
    )

    df_train_encoded.repartition(200).write.parquet(
        "/scratch/sbrocard/datasets/criteo_spark_quarter/"
        + f"train_day_0_to_{NR_DAYS_IN_TRAIN_SET-1}.pq",
        mode="overwrite",
        compression="none",
    )

    if df_test is not None:
        # filling with the global target mean in case some hashes in the test set
        # are not in the train set
        df_test_encoded = df_test_encoded.fillna(global_mean)

        df_test_encoded.repartition(200).write.parquet(
            "/scratch/sbrocard/datasets/criteo_spark_quarter/"
            + f"test_day_{NR_DAYS_IN_TRAIN_SET}"
            + f"_to_{NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET - 1}.pq",
            mode="overwrite",
            compression="none",
        )
