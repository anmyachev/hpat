import h5py
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
import sys


TEST_FOLDER = os.path.dirname(os.path.abspath(__file__))
GENERATED_DATA_PATH = os.path.normpath(TEST_FOLDER + '/generated_data/')


def ensure_generated_data_folder():
    if not os.path.exists(GENERATED_DATA_PATH):
        os.mkdir(GENERATED_DATA_PATH)
        print('%s was created successfully')
        print('for removing generated data just '
              'run `python gen_test_data.py clean`')


if len(sys.argv) == 2 and sys.argv[1] == 'clean':
    os.rmdir(GENERATED_DATA_PATH)
    print('%s was removed successfully')
else:
    ensure_generated_data_folder()


class ParquetGenerator:

    @classmethod
    def gen_kde_pq(cls, file_name='kde.parquet', N=101):
        df = pd.DataFrame({'points': np.random.random(N)})
        table = pa.Table.from_pandas(df)
        row_group_size = 128
        pq.write_table(table, file_name, row_group_size)

    '''
    @classmethod
    def gen_pq_test(cls):
        if not cls.GEN_PQ_TEST_CALLED:
            df = pd.DataFrame(
                {
                    'one': [-1, np.nan, 2.5, 3., 4., 6., 10.0],
                    'two': ['foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'foo'],
                    'three': [True, False, True, True, True, False, False],
                    # float without NA
                    'four': [-1, 5.1, 2.5, 3., 4., 6., 11.0],
                    # str with NA
                    'five': ['foo', 'bar', 'baz', None, 'bar', 'baz', 'foo'],
                }
            )
            table = pa.Table.from_pandas(df)
            pq.write_table(table, GENERATED_DATA_PATH + 'example.parquet')
            pq.write_table(
                table, GENERATED_DATA_PATH + 'example2.parquet',
                row_group_size=2)
            cls.GEN_PQ_TEST_CALLED = True
    '''

    @classmethod
    def gen_parquet_from_dataframe(cls, file_name='dataframe.parquet',
                                   row_group_size=None):
        df = pd.DataFrame(
            {
                'one': [-1, np.nan, 2.5, 3., 4., 6., 10.0],
                'two': ['foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'foo'],
                'three': [True, False, True, True, True, False, False],
                # float without NA
                'four': [-1, 5.1, 2.5, 3., 4., 6., 11.0],
                # str with NA
                'five': ['foo', 'bar', 'baz', None, 'bar', 'baz', 'foo'],
            }
        )
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_name, row_group_size)

    @classmethod
    def gen_datetime64_parquet(cls, file_name='pandas_dt.parquet'):
        dt1 = pd.DatetimeIndex(['2017-03-03 03:23',
                                '1990-10-23', '1993-07-02 10:33:01'])
        df = pd.DataFrame({'DT64': dt1, 'DATE': dt1.copy()})
        df.to_parquet(file_name)

    @classmethod
    def gen_groupby_parquet(cls, file_name='groupby3.parquet'):
        df = pd.DataFrame({'A': ['bc']+["a"]*3+ ["bc"]*3+['a'], 'B': [-8,1,2,3,1,5,6,7]})
        df.to_parquet(file_name)

    @classmethod
    def gen_pivot2_parquet(cls, file_name='pivot2.parquet'):
        df = pd.DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo",
                      "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two",
                      "one", "one", "two", "two"],
                "C": ["small", "large", "large", "small",
                      "small", "large", "small", "small", "large"],
                "D": [1, 2, 2, 6, 3, 4, 5, 6, 9]
            }
        )
        df.to_parquet(file_name)

    @classmethod
    def generate_spark_parquet(cls, file_name='spark_dt.parquet'):
        import os
        import shutil
        import tarfile

        if os.path.exists('sdf_dt.pq'):
            shutil.rmtree('sdf_dt.pq')

        sdf_dt_archive = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sdf_dt.pq.bz2')
        tar = tarfile.open(sdf_dt_archive, "r:bz2")
        tar.extractall('.')
        tar.close()

    @classmethod
    def gen_asof1_parquet(cls, file_name='asof1.pq'):
        # generated data for parallel merge_asof testing
        df = pd.DataFrame({'time': pd.DatetimeIndex(
            ['2017-01-03', '2017-01-06', '2017-02-15', '2017-02-21']),
            'B': [4, 5, 9, 6]})
        df.to_parquet(file_name)

    @classmethod
    def gen_asof2_parquet(cls, file_name='asof2.pq'):
        # generated data for parallel merge_asof testing
        df = pd.DataFrame({'time': pd.DatetimeIndex(
            ['2017-01-01', '2017-01-14', '2017-01-16', '2017-02-23', '2017-02-23',
            '2017-02-25']), 'A': [2,3,7,8,9,10]})
        df.to_parquet(file_name)


def gen_lr(file_name="lr.hdf5", N=101, D=10):
    points = np.random.random((N, D))
    responses = np.random.random(N)
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("points", (N, D), dtype='f8')
    dset1[:] = points
    dset2 = f.create_dataset("responses", (N,), dtype='f8')
    dset2[:] = responses
    f.close()


def gen_group(file_name="test_group_read.hdf5", N=101):
    arr = np.arange(N)
    f = h5py.File(file_name, "w")
    g1 = f.create_group("G")
    dset1 = g1.create_dataset("data", (N,), dtype='i8')
    dset1[:] = arr
    f.close()


def gen_data1_csv(file_name="csv_data1.csv"):
    data = ("0,2.3,4.6,47736\n"
            "1,2.3,4.6,47736\n"
            "2,2.3,4.6,47736\n"
            "4,2.3,4.6,47736\n")

    with open(file_name, "w") as f:
        f.write(data)


def gen_data_infer1_csv(file_name="csv_data_infer1.csv"):
    data = ("0,2.3,4.6,47736\n"
            "1,2.3,4.6,47736\n"
            "2,2.3,4.6,47736\n"
            "4,2.3,4.6,47736\n")

    with open(file_name, "w") as f:
        f.write('A,B,C,D\n'+data)


def gen_data_date1_csv(file_name="csv_data_date1.csv"):
    data = ("0,2.3,2015-01-03,47736\n"
            "1,2.3,1966-11-13,47736\n"
            "2,2.3,1998-05-21,47736\n"
            "4,2.3,2018-07-11,47736\n")

    with open(file_name, "w") as f:
        f.write(data)
