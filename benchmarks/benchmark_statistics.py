import time

import numpy as np

from pyfoamalgo.config import __NAN_DTYPES__
from pyfoamalgo import (
    nanmean, nansum, nanstd, nanvar, nanmin, nanmax, histogram1d
)


def benchmark_nan_without_axis(f_cpp, f_py, shape, dtype):
    data = np.random.randn(*shape).astype(dtype) + 1.  # shift to avoid very small mean
    data[:, :3, ::3] = np.nan

    t0 = time.perf_counter()
    ret_cpp = f_cpp(data)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    ret_py = f_py(data)
    dt_py = time.perf_counter() - t0

    np.testing.assert_allclose(ret_cpp, ret_py, rtol=1e-4)

    print(f"\nwithout axis, dtype = {dtype} - \n"
          f"dt (cpp): {dt_cpp:.4f}, "
          f"dt (numpy): {dt_py:.4f}")


def benchmark_nan_keep_zero_axis(f_cpp, f_py, shape, dtype):
    data = np.random.randn(*shape).astype(dtype=dtype) + 1.  # shift to avoid very small mean
    data[:, :3, ::3] = np.nan

    t0 = time.perf_counter()
    ret_cpp = f_cpp(data, axis=(-2, -1))
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    ret_py = f_py(data, axis=(-2, -1))
    dt_py = time.perf_counter() - t0

    np.testing.assert_allclose(ret_cpp, ret_py, rtol=1e-4)

    print(f"\nkeep zero axis, dtype = {dtype} - \n"
          f"dt (cpp): {dt_cpp:.4f}, "
          f"dt (numpy): {dt_py:.4f}")


def benchmark_histogram1d(f_cpp, f_py, shape, dtype):
    data = np.random.randn(*shape).astype(dtype=dtype)

    t0 = time.perf_counter()
    ret_cpp = f_cpp(data, bins=100)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    ret_py = f_py(data, bins=100)
    dt_py = time.perf_counter() - t0

    np.testing.assert_allclose(ret_cpp[0], ret_py[0], rtol=1e-4)
    np.testing.assert_array_almost_equal(ret_cpp[1], ret_py[1])

    print(f"\n----- {f_cpp.__name__} ------")
    print(f"\ndtype = {dtype} - \ndt (cpp): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}")


if __name__ == "__main__":
    print("*" * 80)
    print("Benchmark statistics functions")
    print("*" * 80)

    s = (32, 1096, 1120)

    for dtype in __NAN_DTYPES__:
        for f_cpp, f_py in [(nansum, np.nansum),
                            (nanmean, np.nanmean),
                            (nanmin, np.nanmin),
                            (nanmax, np.nanmax)]:
            print(f"\n----- {f_cpp.__name__} ------")
            benchmark_nan_without_axis(f_cpp, f_py, s, dtype)
            benchmark_nan_keep_zero_axis(f_cpp, f_py, s, dtype)

        for f_cpp, f_py in [(histogram1d, np.histogram)]:
            benchmark_histogram1d(f_cpp, f_py, s, dtype)
