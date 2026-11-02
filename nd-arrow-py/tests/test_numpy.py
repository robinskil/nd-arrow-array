import numpy as np
import pytest

from nd_arrow import _core

DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "datetime64[s]",
    "datetime64[ns]",
    "timedelta64[ms]",
    "<U5",
]


def sample(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "U":
        return np.array([["a", "bc", ""], ["def", "ghijk", "x"]], dtype=dtype)
    return np.arange(6).reshape(2, 3).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_roundtrip_dtypes(dtype):
    arr = sample(dtype)
    back = _core._roundtrip_numpy(arr)
    assert type(back) is np.ndarray
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_non_contiguous_keeps_logical_order():
    a = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(_core._roundtrip_numpy(a.T), a.T)
    np.testing.assert_array_equal(_core._roundtrip_numpy(a[:, ::2]), a[:, ::2])


def test_zero_d_and_empty():
    zero_d = np.array(2.5)
    back = _core._roundtrip_numpy(zero_d)
    assert back.shape == ()
    assert back == 2.5
    empty = np.zeros((0, 3), dtype="int32")
    back = _core._roundtrip_numpy(empty)
    assert back.shape == (0, 3)
    assert back.dtype == np.int32


def test_mask_becomes_null():
    arr = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    back = _core._roundtrip_numpy(arr)
    assert isinstance(back, np.ma.MaskedArray)
    np.testing.assert_array_equal(np.ma.getmaskarray(back), [False, True, False])
    assert back[0] == 1.0 and back[2] == 3.0


def test_nan_stays_value():
    back = _core._roundtrip_numpy(np.array([1.0, np.nan]))
    assert type(back) is np.ndarray
    assert np.isnan(back[1])


def test_nat_becomes_null():
    arr = np.array(["2020-01-01", "NaT"], dtype="datetime64[s]")
    back = _core._roundtrip_numpy(arr)
    assert isinstance(back, np.ma.MaskedArray)
    np.testing.assert_array_equal(np.ma.getmaskarray(back), [False, True])
    assert back[0] == arr[0]


def test_object_dtype_raises_type_error():
    with pytest.raises(TypeError):
        _core._roundtrip_numpy(np.array([{}], dtype=object))


def test_masked_non_contiguous():
    base = np.ma.array(np.arange(10.0), mask=np.arange(10) % 3 == 0)
    for view in (base[::2], base[::-1]):
        back = _core._roundtrip_numpy(view)
        np.testing.assert_array_equal(np.ma.getmaskarray(back), np.ma.getmaskarray(view))
        np.testing.assert_array_equal(back.compressed(), view.compressed())
    grid = np.ma.array(np.arange(12.0).reshape(3, 4), mask=np.arange(12).reshape(3, 4) % 5 == 0)
    back = _core._roundtrip_numpy(grid[:, ::2])
    np.testing.assert_array_equal(np.ma.getmaskarray(back), np.ma.getmaskarray(grid[:, ::2]))
