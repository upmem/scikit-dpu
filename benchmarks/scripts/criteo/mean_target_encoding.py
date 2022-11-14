# from https://gist.github.com/ogrisel/b6a97ed87939e3b559568ac2f6599cba

import os
import os.path as op
from time import time
import dask.dataframe as ddf
import dask.array as da
from distributed import Client


def make_categorical_data(n_samples=int(1e7), n_features=10, n_partitions=100):
    """Generate some random categorical data
    The default parameters should generate around 1GB of random integer data
    with increasing cardinality along with a normally distributed real valued
    target variable.
    """
    feature_names = ['f_%03d' % i for i in range(n_features)]
    features_series = [
        da.random.randint(low=0, high=(i + 1) * 10, size=n_samples,
                          chunks=n_samples // n_partitions)
        for i in range(n_features)
    ]
    features_series = [
        ddf.from_dask_array(col_data, columns=[feature_name])
        for col_data, feature_name in zip(features_series, feature_names)
    ]
    target = da.random.normal(loc=0, scale=1, size=n_samples,
                              chunks=n_samples // 10)
    target = ddf.from_dask_array(target, columns=['target'])

    data = ddf.concat(features_series + [target], axis=1)
    data = data.repartition(npartitions=n_partitions)
    return data


def target_mean_transform(data, feature_colname, target_colname):
    if data[feature_colname].dtype.kind not in ('i', 'O'):
        # Non-categorical variables are kept untransformed:
        return data[feature_colname]

    data = data[[feature_colname, target_colname]]
    target_means = data.groupby(feature_colname).mean()
    encoded_col = ddf.merge(data[feature_colname].to_frame(), target_means,
                            left_on=feature_colname, right_index=True,
                            how='left')[target_colname].to_frame()
    new_colname = feature_colname + '_mean_' + target_colname
    encoded_col = encoded_col.rename(columns={target_colname: new_colname})

    # Hack: left join should preserve divisions which are required for
    # efficient downstream concat with axis=1 (but is it always true?).
    encoded_col.divisions = data.divisions
    return encoded_col


def encode_with_target_mean(data, target_colname='target'):
    """Supervised encoding of categorical variables with per-group target mean.
    All columns that contain integer values are replaced by real valued data
    representing the average target value for each category.
    """
    features_data = data.drop(target_colname, axis=1)
    target_data = data[target_colname].to_frame()

    # Sequential processing of columns: there is no need to parallelize
    # column-wise as the inner-operation will already parallelize record-wise.
    encoded_columns = [target_mean_transform(data, colname, target_colname)
                       for colname in features_data.columns]
    return ddf.concat(encoded_columns + [target_data], axis=1)


if __name__ == '__main__':
    # make sure dask uses the distributed scheduler:
    # Start the scheduler and at least one worker with:
    #    $ dask-scheduler
    #    $ dask-worker localhost:8786
    #
    c = Client('localhost:8786')
    original_folder_name = op.abspath('random_categorical_data')
    encoded_folder_name = op.abspath('random_encoded_data')
    if not op.exists(original_folder_name):
        print("Generating random categorical data in", original_folder_name)
        os.mkdir(original_folder_name)
        data = make_categorical_data(n_partitions=10)
        ddf.to_parquet(data, original_folder_name)

    t0 = time()
    print("Using data from", original_folder_name)
    data = ddf.read_parquet(original_folder_name)

    print("Encoding categorical variables...")
    encoded = encode_with_target_mean(data, target_colname='target')

    print("Saving encoded data to", encoded_folder_name)
    # Repartition to get small parquet files in the output folder.
    # encoded = encoded.repartition(npartitions=10)
    ddf.to_parquet(encoded, encoded_folder_name)
    print("done in %0.3fs" % (time() - t0))