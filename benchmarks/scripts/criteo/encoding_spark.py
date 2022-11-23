"""
Converts the hashes features in the Criteo dataset to numerical features.
"""

import os
import sys
from functools import reduce

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

NR_DAYS_IN_TRAIN_SET = 2
NR_DAYS_IN_TEST_SET = NR_DAYS_IN_TRAIN_SET


def union_all(*dfs: DataFrame) -> DataFrame:
    return reduce(DataFrame.union, dfs)


def configure_spark() -> SparkSession:
    spark = (
        SparkSession.builder.config("spark.driver.host", "127.0.0.1")
        .config("spark.executor.memory", "10g")
        .config("spark.driver.memory", "250g")
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
    df_train_enriched = df_train
    df_test_enriched = df_test
    for c in col:
        print(f"column: {c}")
        means = df_train.groupby(F.col(c)).agg(
            F.mean(target).alias(f"{c}_mean_encoding")
        )
        df_train_enriched = df_train_enriched.join(means, on=c, how="left").drop(c)
        df_test_enriched = df_test_enriched.join(means, on=c, how="left").drop(c)
        # target_encoded_columns_list.append(f"{c}_mean_encoding")
    return (
        df_train_enriched,
        df_test_enriched,
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

    global_mean = df_train.select(F.mean("0")).collect()[0][0]

    print(f"got global mean: {global_mean}")

    # del df_encoded
    # del df

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

    df_train_encoded, df_test_encoded = target_mean_encoding(
        df_train, df_test, [str(i) for i in range(14, 40)], "0"
    )

    # filling with the global target mean in case some hashes in the test set are not in the train set
    df_test_encoded = df_test_encoded.fillna(global_mean)

    df_train_encoded.write.parquet(
        "/scratch/sbrocard/datasets/criteo_spark/"
        + f"train_day_0_to_{NR_DAYS_IN_TRAIN_SET-1}.pq",
        mode="overwrite",
        compression='none',
    )

    df_test_encoded.write.parquet(
        "/scratch/sbrocard/datasets/criteo_spark/"
        + f"test_day_{NR_DAYS_IN_TRAIN_SET}"
        + f"_to_{NR_DAYS_IN_TRAIN_SET + NR_DAYS_IN_TEST_SET - 1}.pq",
        mode="overwrite",
        compression='none',
    )
