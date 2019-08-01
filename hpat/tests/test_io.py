import unittest
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import h5py
import pyarrow.parquet as pq
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_rank,
    get_start_end)

import os

from hpat.tests.gen_test_data import (
    GENERATED_DATA_PATH, gen_lr, gen_group, gen_data1_csv,
    gen_data_infer1_csv, gen_data_date1_csv, ParquetGenerator)


class TestIO(unittest.TestCase):

    KDE_PARQUET = os.path.join(GENERATED_DATA_PATH, 'kde.parquet')
    EXAMPLE_PARQUET = os.path.join(GENERATED_DATA_PATH, 'example.parquet')
    EXAMPLE2_PARQUET = os.path.join(GENERATED_DATA_PATH, 'example2.parquet')
    PANDAS_DT_PARQUET = os.path.join(GENERATED_DATA_PATH, 'pandas_dt.parquet')
    SPARK_DT_PARQUET = os.path.join(GENERATED_DATA_PATH, 'spark_dt.parquet')
    LR_HDF5 = os.path.join(GENERATED_DATA_PATH, 'lr.hdf5')
    GROUP_HDF5 = os.path.join(GENERATED_DATA_PATH, 'test_group_read.hdf5')
    DATA1_CSV = os.path.join(GENERATED_DATA_PATH, 'csv_data1.csv')
    DATA_INFER1_CSV = os.path.join(GENERATED_DATA_PATH, 'csv_data_infer1.csv')
    DATA_DATE1_CSV = os.path.join(GENERATED_DATA_PATH, 'csv_data_date1.csv')

    @classmethod
    def setUpClass(cls):
        ParquetGenerator.gen_kde_pq(cls.KDE_PARQUET, N=101)
        ParquetGenerator.gen_parquet_from_dataframe(cls.EXAMPLE_PARQUET)
        ParquetGenerator.gen_parquet_from_dataframe(cls.EXAMPLE2_PARQUET,
                                                    row_group_size=2)
        ParquetGenerator.gen_datetime64_parquet(cls.PANDAS_DT_PARQUET)
        ParquetGenerator.generate_spark_parquet(cls.SPARK_DT_PARQUET)
        gen_lr(cls.LR_HDF5, N=101, D=10)
        gen_group(cls.GROUP_HDF5, N=101)
        gen_data1_csv(cls.DATA1_CSV)
        gen_data_infer1_csv(cls.DATA_INFER1_CSV)
        gen_data_date1_csv(cls.DATA_DATE1_CSV)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.KDE_PARQUET)
        os.remove(cls.EXAMPLE_PARQUET)
        os.remove(cls.EXAMPLE2_PARQUET)
        os.remove(cls.PANDAS_DT_PARQUET)
        os.remove(cls.SPARK_DT_PARQUET)
        os.remove(cls.LR_HDF5)
        os.remove(cls.GROUP_HDF5)
        os.remove(cls.DATA1_CSV)
        os.remove(cls.DATA_INFER1_CSV)
        os.remove(cls.DATA_DATE1_CSV)

    def setUp(self):
        if get_rank() == 0:
            # h5 filter test
            n = 11
            size = (n, 13, 21, 3)
            A = np.random.randint(0, 120, size, np.uint8)
            f = h5py.File('h5_test_filter.h5', "w")
            f.create_dataset('test', data=A)
            f.close()

            # test_csv_cat1
            data = ("2,B,SA\n"
                    "3,A,SBC\n"
                    "4,C,S123\n"
                    "5,B,BCD\n")

            with open("csv_data_cat1.csv", "w") as f:
                f.write(data)

            # test_csv_single_dtype1
            data = ("2,4.1\n"
                    "3,3.4\n"
                    "4,1.3\n"
                    "5,1.1\n")

            with open("csv_data_dtype1.csv", "w") as f:
                f.write(data)

            # test_np_io1
            n = 111
            A = np.random.ranf(n)
            A.tofile("np_file1.dat")

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_h5_read_seq(self):
        lr_hdf5 = self.LR_HDF5

        def test_impl():
            f = h5py.File(lr_hdf5, "r")
            X = f['points'][:]
            f.close()
            return X

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_allclose(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_h5_read_const_infer_seq(self):
        lr_hdf5 = self.LR_HDF5

        def test_impl():
            f = h5py.File(lr_hdf5, "r")
            s = 'po'
            X = f[s + 'ints'][:]
            f.close()
            return X

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_allclose(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_h5_read_parallel(self):
        lr_hdf5 = self.LR_HDF5

        def test_impl():
            f = h5py.File(lr_hdf5, "r")
            X = f['points'][:]
            Y = f['responses'][:]
            f.close()
            return X.sum() + Y.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl(), decimal=2)
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip("fix collective create dataset")
    def test_h5_write_parallel(self):
        def test_impl(N, D):
            points = np.ones((N,D))
            responses = np.arange(N)+1.0
            f = h5py.File("lr_w.hdf5", "w")
            dset1 = f.create_dataset("points", (N,D), dtype='f8')
            dset1[:] = points
            dset2 = f.create_dataset("responses", (N,), dtype='f8')
            dset2[:] = responses
            f.close()

        N = 101
        D = 10
        hpat_func = hpat.jit(test_impl)
        hpat_func(N, D)
        f = h5py.File("lr_w.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]
        f.close()
        np.testing.assert_almost_equal(X, np.ones((N,D)))
        np.testing.assert_almost_equal(Y, np.arange(N)+1.0)

    @unittest.skip("fix collective create dataset and group")
    def test_h5_write_group(self):
        def test_impl(n, fname):
            arr = np.arange(n)
            n = len(arr)
            f = h5py.File(fname, "w")
            g1 = f.create_group("G")
            dset1 = g1.create_dataset("data", (n,), dtype='i8')
            dset1[:] = arr
            f.close()

        n = 101
        arr = np.arange(n)
        fname = "test_group.hdf5"
        hpat_func = hpat.jit(test_impl)
        hpat_func(n, fname)
        f = h5py.File(fname, "r")
        X = f['G']['data'][:]
        f.close()
        np.testing.assert_almost_equal(X, arr)

    @unittest.skip('AssertionError - fix needed\n'
                   '2282 != 5050\n'
                   'NUMA_PES=3 build')
    def test_h5_read_group(self):
        group_hdf5 = self.GROUP_HDF5

        def test_impl():
            f = h5py.File(group_hdf5, "r")
            g1 = f['G']
            X = g1['data'][:]
            f.close()
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_h5_file_keys(self):
        group_hdf5 = self.GROUP_HDF5

        def test_impl():
            f = h5py.File(group_hdf5, "r")
            s = 0
            for gname in f.keys():
                X = f[gname]['data'][:]
                s += X.sum()
            f.close()
            return s

        hpat_func = hpat.jit(test_impl, h5_types={'X': hpat.int64[:]})
        self.assertEqual(hpat_func(), test_impl())
        # test using locals for typing
        hpat_func = hpat.jit(test_impl, locals={'X': hpat.int64[:]})
        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_h5_group_keys(self):
        group_hdf5 = self.GROUP_HDF5

        def test_impl():
            f = h5py.File(group_hdf5, "r")
            g1 = f['G']
            s = 0
            for dname in g1.keys():
                X = g1[dname][:]
                s += X.sum()
            f.close()
            return s

        hpat_func = hpat.jit(test_impl, h5_types={'X': hpat.int64[:]})
        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_h5_filter(self):
        def test_impl():
            f = h5py.File("h5_test_filter.h5", "r")
            b = np.arange(11) % 3 == 0
            X = f['test'][b,:,:,:]
            f.close()
            return X

        hpat_func = hpat.jit(locals={'X:return': 'distributed'})(test_impl)
        n = 4  # len(test_impl())
        start, end = get_start_end(n)
        np.testing.assert_allclose(hpat_func(), test_impl()[start:end])

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_read(self):
        kde_pq = self.KDE_PARQUET

        def test_impl():
            t = pq.read_table(kde_pq)
            df = t.to_pandas()
            X = df['points']
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('AssertionError - fix needed\n'
                   'Arrays are not almost equal to 7 decimals\n'
                   'ACTUAL: 59.92340551591986\n'
                   'DESIRED: 58.34405719897534\n'
                   'NUMA_PES=3 build')
    def test_pq_read_global_str1(self):
        kde_pq = self.KDE_PARQUET

        def test_impl():
            df = pd.read_parquet(kde_pq)
            X = df['points']
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_read_freevar_str1(self):
        kde_pq = self.KDE_PARQUET

        def test_impl():
            df = pd.read_parquet(kde_pq)
            X = df['points']
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pd_read_parquet(self):
        kde_pq = self.KDE_PARQUET

        def test_impl():
            df = pd.read_parquet(kde_pq)
            X = df['points']
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_str(self):
        example_pq = self.EXAMPLE_PARQUET

        def test_impl():
            df = pq.read_table(example_pq).to_pandas()
            A = df.two.values=='foo'
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_str_with_nan_seq(self):
        example_pq = self.EXAMPLE_PARQUET

        def test_impl():
            df = pq.read_table(example_pq).to_pandas()
            A = df.five.values=='foo'
            return A

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_str_with_nan_par(self):
        example_pq = self.EXAMPLE_PARQUET

        def test_impl():
            df = pq.read_table(example_pq).to_pandas()
            A = df.five.values=='foo'
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('AssertionError - fix needed\n'
                   'Arrays are not almost equal to 7 decimals\n'
                   'ACTUAL: 4625837024398916366\n'
                   'DESIRED: 2\n'
                   'NUMA_PES=3 build')
    def test_pq_str_with_nan_par_multigroup(self):
        example2_pq = self.EXAMPLE2_PARQUET

        def test_impl():
            df = pq.read_table(example2_pq).to_pandas()
            A = df.five.values=='foo'
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_bool(self):
        example_pq = self.EXAMPLE_PARQUET

        def test_impl():
            df = pq.read_table(example_pq).to_pandas()
            return df.three.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_nan(self):
        example_pq = self.EXAMPLE_PARQUET

        def test_impl():
            df = pq.read_table(example_pq).to_pandas()
            return df.one.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_float_no_nan(self):
        example_pq = self.EXAMPLE_PARQUET

        def test_impl():
            df = pq.read_table(example_pq).to_pandas()
            return df.four.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_pandas_date(self):
        pd_dt_pq = self.PANDAS_DT_PARQUET

        def test_impl():
            df = pd.read_parquet(pd_dt_pq)
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_pq_spark_date(self):
        spark_dt_pq = self.SPARK_DT_PARQUET

        def test_impl():
            df = pd.read_parquet(spark_dt_pq)
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv1(self):
        data1_csv = self.DATA1_CSV

        def test_impl():
            return pd.read_csv(data1_csv,
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_keys1(self):
        data1_csv = self.DATA1_CSV

        def test_impl():
            dtype = {'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int}
            return pd.read_csv(data1_csv,
                names=dtype.keys(),
                dtype=dtype,
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_const_dtype1(self):
        data1_csv = self.DATA1_CSV

        def test_impl():
            dtype = {'A': 'int', 'B': 'float64', 'C': 'float', 'D': 'int64'}
            return pd.read_csv(data1_csv,
                names=dtype.keys(),
                dtype=dtype,
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_infer1(self):
        data_indef1_csv = self.DATA_INFER1_CSV

        def test_impl():
            return pd.read_csv(data_indef1_csv)

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_infer_parallel1(self):
        data_indef1_csv = self.DATA_INFER1_CSV

        def test_impl():
            df = pd.read_csv(data_indef1_csv)
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_csv_skip1(self):
        data1_csv = self.DATA1_CSV

        def test_impl():
            return pd.read_csv(data1_csv,
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},
                skiprows=2,
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_infer_skip1(self):
        data_indef1_csv = self.DATA_INFER1_CSV

        def test_impl():
            return pd.read_csv(data_indef1_csv, skiprows=2)

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_infer_skip_parallel1(self):
        data_indef1_csv = self.DATA_INFER1_CSV

        def test_impl():
            df = pd.read_csv(data_indef1_csv, skiprows=2,
                names=['A', 'B', 'C', 'D'])
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_rm_dead1(self):
        data1_csv = self.DATA1_CSV

        def test_impl():
            df = pd.read_csv(data1_csv,
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},)
            return df.B.values
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_csv_date1(self):
        data_date1_csv = self.DATA_DATE1_CSV

        def test_impl():
            return pd.read_csv(data_date1_csv,
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int},
                parse_dates=[2])
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_str1(self):
        data_date1_csv = self.DATA_DATE1_CSV

        def test_impl():
            return pd.read_csv(data_date1_csv,
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_parallel1(self):
        data1_csv = self.DATA1_CSV

        def test_impl():
            df = pd.read_csv(data1_csv,
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int})
            return (df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum())
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_str_parallel1(self):
        data_date1_csv = self.DATA_DATE1_CSV

        def test_impl():
            df = pd.read_csv(data_date1_csv,
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int})
            return (df.A.sum(), df.B.sum(), (df.C == '1966-11-13').sum(),
                    df.D.sum())
        hpat_func = hpat.jit(locals={'df:return': 'distributed'})(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_csv_usecols1(self):
        data1_csv = self.DATA1_CSV

        def test_impl():
            return pd.read_csv(data1_csv,
                names=['C'],
                dtype={'C':np.float},
                usecols=[2],
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_cat1(self):
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1':np.int, 'C2': ct_dtype, 'C3':str}
            df = pd.read_csv("csv_data_cat1.csv",
                names=['C1', 'C2', 'C3'],
                dtype=dtypes,
            )
            return df.C2
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(), test_impl(), check_names=False)

    def test_csv_cat2(self):
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
            df = pd.read_csv("csv_data_cat1.csv",
                names=['C1', 'C2', 'C3'],
                dtype={'C1':np.int, 'C2': ct_dtype, 'C3':str},
            )
            return df
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_single_dtype1(self):
        def test_impl():
            df = pd.read_csv("csv_data_dtype1.csv",
                names=['C1', 'C2'],
                dtype=np.float64,
            )
            return df
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('pandas.errors.EmptyDataError - fix needed\n'
                   'No columns to parse from file\n'
                   'NUMA_PES=3 build')
    def test_write_csv1(self):
        def test_impl(df, fname):
            df.to_csv(fname)

        hpat_func = hpat.jit(test_impl)
        n = 111
        df = pd.DataFrame({'A': np.arange(n)})
        hp_fname = 'test_write_csv1_hpat.csv'
        pd_fname = 'test_write_csv1_pd.csv'
        hpat_func(df, hp_fname)
        test_impl(df, pd_fname)
        # TODO: delete files
        pd.testing.assert_frame_equal(pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_write_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({'A': np.arange(n)})
            df.to_csv(fname)

        hpat_func = hpat.jit(test_impl)
        n = 111
        hp_fname = 'test_write_csv1_hpat_par.csv'
        pd_fname = 'test_write_csv1_pd_par.csv'
        hpat_func(n, hp_fname)
        test_impl(n, pd_fname)
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        # TODO: delete files
        if get_rank() == 0:
            pd.testing.assert_frame_equal(
                pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    def test_np_io1(self):
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_np_io2(self):
        # parallel version
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_np_io3(self):
        def test_impl(A):
            if get_rank() == 0:
                A.tofile("np_file_3.dat")

        hpat_func = hpat.jit(test_impl)
        n = 111
        A = np.random.ranf(n)
        hpat_func(A)
        if get_rank() == 0:
            B = np.fromfile("np_file_3.dat", np.float64)
            np.testing.assert_almost_equal(A, B)

    def test_np_io4(self):
        # parallel version
        def test_impl(n):
            A = np.arange(n)
            A.tofile("np_file_3.dat")

        hpat_func = hpat.jit(test_impl)
        n = 111
        A = np.arange(n)
        hpat_func(n)
        B = np.fromfile("np_file_3.dat", np.int64)
        np.testing.assert_almost_equal(A, B)


if __name__ == "__main__":
    unittest.main()
