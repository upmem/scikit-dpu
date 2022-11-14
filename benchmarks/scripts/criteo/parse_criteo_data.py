# from https://gist.github.com/ogrisel/b6a97ed87939e3b559568ac2f6599cba

import numpy as np
import os.path as op
import os
from gzip import GzipFile
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


DATA_FOLDER = '~/data/criteo'
OUTPUT_FOLDER = '~/data/criteo_parquet'

CHUNK_SIZE = 500 * 1024 ** 2

COLUMN_NAMES = ['label']
COLUMN_NAMES += ['num_%02d' % (i + 1) for i in range(13)]
COLUMN_NAMES += ['cat_%02d' % (i + 1) for i in range(26)]


def parse_chunk(data, previous_remainder=b"", missing_value=-1):
    """Parse a chunk of bytes"""
    lines = data.splitlines()
    remainder = lines[-1]
    lines = lines[:-1]
    lines[0] = previous_remainder + lines[0]

    # Everything can be stored as an integer, even the numerical values.
    parsed_array = np.empty(shape=(len(lines), len(COLUMN_NAMES)),
                            dtype=np.int32)
    parsed_array.fill(missing_value)
    invalid_count = 0
    for i, line in enumerate(lines):
        fields = line[:-1].split(b'\t')
        if len(fields) != 40:
            invalid_count += 1
            continue
        for j, x in enumerate(fields):
            if not x:
                # skip missing values
                continue
            parsed_array[i, j] = int(x) if j < 14 else int(x, 16)
    return parsed_array, remainder, invalid_count


def convert_file_to_parquet(filepath, output_folder, compression='SNAPPY'):
    """Convert gzipped TSV data to smaller parquet files
    Gzip does not allow for efficient seek-based file access and TSV
    is IO intensive and slow to parse.
    Converting to compressed parquet files makes it possible to efficiently
    access record-wise or column-wise chunks of the data using the
    dask.dataframe API.
    """
    invalid_lines_count = 0
    filebase, _ = op.splitext(op.basename(filepath))
    output_filepattern = op.join(output_folder, filebase + '_{:04d}.parquet')
    chunk_idx = 0
    print(f"Processing {filepath}...")
    with GzipFile(filepath) as f:
        remainder = b''
        while True:
            try:
                # Process uncompressed chunks of ~500 MiB at a time:
                bytes_chunk = f.read(CHUNK_SIZE)
            except EOFError:
                print("Warning: truncated file:", filepath)
                break
            parsed_chunk, remainder, invalid_count = parse_chunk(
                bytes_chunk, previous_remainder=remainder)
            if len(parsed_chunk) == 0:
                break
            invalid_lines_count += invalid_count
            outpath = output_filepattern.format(chunk_idx)
            df = pd.DataFrame(parsed_chunk, columns=COLUMN_NAMES)

            arrow_table = pa.Table.from_pandas(df)
            pq.write_table(arrow_table, outpath, use_dictionary=False,
                           compression=compression)
            print(f"Wrote {outpath}")
            chunk_idx += 1

    if invalid_lines_count:
        print(f"Found {invalid_lines_count} invalid lines in {filepath}")


def convert_folder_to_parquet(input_folder, output_folder):
    input_folder = op.expanduser(input_folder)
    output_folder = op.expanduser(output_folder)
    if not op.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if not filename.endswith('.gz'):
            continue
        filepath = op.join(input_folder, filename)
        convert_file_to_parquet(filepath, output_folder)


if __name__ == "__main__":
    convert_folder_to_parquet(DATA_FOLDER, OUTPUT_FOLDER)