# Instructions for processing the Criteo dataset

The Criteo Terabyte dataset is a daily log of click-throughs.
It has one boolean target, 13 integer features, and 26 categorical features represented as hashes with unknown semantics.

In order to apply algorithms like Decision Trees, K-Means or Linear Regression to it, one needs to encode those 26 categorical features into a numerical format. The most common approach in such cases is to use one-hot encoding. However, since some columns in the Criteo dataset have more than 30 million different categories, this would result in an encoded dataset of huge dimensionality. Following the example of the xgboost paper, we instead opt for a mean target encoding. It preserves the original dataset dimensionionality and can be performed efficiently using relationial databases operators.

How you're going to perform this encoding depends on the size of the dataset you want to process and your available RAM.

## A note on disk use

All those operations are going to be very intensive on disk access. If your work folder is in a network volume (which is the case in UPMEM cloud), consider moving all the data on a NVME-mounted folder (`/scratch` in UPMEM cloud), and performing the entire pipeline there.

## First step: Download the dataset

<https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>

There's no need to download the entire dataset, just the days you plan to use for training and testing.

## Second step: Unzip the dataset

The Criteo dataset comes in gzip files. Since gzip files can only be accessed sequentially, working with them would slow down the processing considerably.

So first unzip the days from the dataset that you want to use, or all of them (`for z in *.gz; do gunzip -k "$z"; done`).

## Third step: Serialize the dataset

At this point, it is preferrable to convert the dataset to a format with faster read access than a csv file, so that subsequent steps can be iterated faster.

### In memory (Pandas)

If you have enough memory to hold an entire day of Criteo data in RAM, the script [convert_to_pq_inmemory.py](convert_to_pq_inmemory.py) will convert the csv files to parquet using base pandas.

Python requirements:

```text
pandas
pyarrow
```

### Out of memory (Dask)

If you run out of memory during this conversion, you can use dask instead of pandas for lazy evaluation with [convert_to_pq_outofmemory.py](convert_to_pq_outofmemory.py).

Python requirements:

```text
dask
pyarrow
```

## Fourth step: perform the encoding

### Constants

In either scripts, you can change the value of `NR_DAYS_IN_TRAIN_SET` and `NR_DAYS_IN_TEST_SET` to adjust the size of your train and test sets. The resulting train set will consist of the first `NR_DAYS_IN_TRAIN_SET` days in the Criteo dataset and the test set of the following `NR_DAYS_IN_TEST_SET` days.

> Note: it is important to separate both sets at this stage in the pipeline, and keep them separate afterwards. It would be illegitimate to encode the entire Criteo dataset first and then split it into train/test sets, as doing so would cause target leakage into the test set.

### In memory (Pandas) - [Not recommended]

The `category_encoders` library provides a M-estimate encoder that can be used to perform mean target encoding (which is simply a M-estimate encoding with a M value of zero). The scripts [encoding_in_memory.py](encoding_in_memory.py) shows a way to perform the encoding, with the command `python encoding_in_memory.py /path/to/criteo/folder`.

Unfortunately the memory requirement is very high (256 GiB to encode a single day of Criteo data), and the runtime is long for anything but small datasets. It is recommended to use this method only for prototyping with a small subset of the Criteo data.

Python requirements:

```text
dask
pyarrow
category_encoders
```

### Out of memory (Spark)

The script [encoding_spark.py](encoding_spark.py) performs the mean target encoding out-of-memory. It will create and use a single worker on the machine executing the script. If you have access to a Hadoop cluster, consider using it instead.  
Usage: `python encoding_spark.py /path/to/criteo/data`

Please adjust the following parameters in the script:

- `spark.driver.memory`: Since we are running spark on a single node, this is the only memory we need to care about. Make sure that it is lower than your available RAM. The default 256g is an overshoot and the process can probably run fine on 10g of memory or less.
- `spark.local.dir`: This is the directory used by spark for temporary offloads to disk. Make **absolutely sure** that this directory is mounted on a local disk (preferably NVME) with ample space and not a NFS.

Python requirements:

```text
pyspark
```

Expected runtime: ~ 40 mn

#### Expected warning messages

Since we're running in local mode, the following warning messages are expected and not cause for concern:

```text
WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
WARN SparkConf: Note that spark.local.dir will be overridden by the value set by the cluster manager (via SPARK_LOCAL_DIRS in mesos/standalone/kubernetes and LOCAL_DIRS in YARN).
```

During execution, the following message should also appear. The truncation it warns about only affects logs.

```text
WARN package: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
```

#### Monitoring encoding progress

Pyspark automatically starts a web dashboard at localhost:4040. If you're running the script on a server you can make a ssh redirection with `ssh -N -f -L localhost:4040:localhost:4040 <server>` on your local machine and follow the progression on a web browser.

#### Assumptions

There are many missing values in the Criteo dataset. Too many, in fact, to just drop the corresponding rows. Therefore we have to make assumptions as to the meaning of missing values:

- for the 13 numerical features, missing values are replaced with 0. This is somewhat justified because those features represent a count. A more prudent approach could have been to replace missing value with -1, but there already are -1 values present in the dataset.
- for the 26 categorical features, missing hashes are treated as one extra category
