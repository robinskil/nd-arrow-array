import numpy as np
import pytest

import nd_arrow_array as nda


def arr(values, dims, attrs=None):
    return nda.NdArray.from_numpy(np.asarray(values), dims=dims, attrs=attrs)


def test_from_arrays_and_index():
    a = arr([[1.0, 2.0], [3.0, 4.0]], ["y", "x"], {"units": "K"})
    b = arr([5.0], ["x"])
    col = nda.NdColumn.from_arrays([a, b])
    assert len(col) == 2
    assert col[0] == a
    assert col[-1] == b
    assert list(col) == [a, b]


def test_null_row():
    a = arr([1.0], ["x"])
    col = nda.NdColumn.from_arrays([a, None])
    assert col[1] is None
    assert col.null_count == 1


def test_mixed_types_raise():
    with pytest.raises(nda.NdFormatError):
        nda.NdColumn.from_arrays([arr([1.0], ["x"]), arr(np.array([1], dtype="int32"), ["x"])])


def test_attr_type_conflict():
    with pytest.raises(ValueError):
        nda.NdColumn.from_arrays(
            [arr([1.0], ["x"], {"u": "K"}), arr([2.0], ["x"], {"u": 1})]
        )


def test_all_none_raises():
    with pytest.raises(ValueError):
        nda.NdColumn.from_arrays([None])


def test_index_error():
    col = nda.NdColumn.from_arrays([arr([1.0], ["x"])])
    with pytest.raises(IndexError):
        col[5]
