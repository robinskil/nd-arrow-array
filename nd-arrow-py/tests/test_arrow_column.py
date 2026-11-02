import numpy as np
import pytest

import nd_arrow as nda

pa = pytest.importorskip("pyarrow")


def make():
    return nda.NdArray.from_numpy(
        np.arange(4.0).reshape(2, 2), dims=["y", "x"], attrs={"units": "K"}
    )


def test_ndarray_from_pyarrow_values():
    a = nda.NdArray(pa.array([1.0, 2.0, 3.0, 4.0]), dims={"x": 2, "y": 2})
    assert a.shape == (2, 2)
    assert a.dims == ("x", "y")


def test_ndarray_from_pyarrow_with_attrs():
    a = nda.NdArray(pa.array([1, 2]), dims=[("x", 2)], attrs={"units": "m"})
    assert a.attrs == {"units": "m"}
    assert a.dtype == "int64"


def test_ndarray_values():
    assert make().values.equals(pa.array([0.0, 1.0, 2.0, 3.0]))


def test_column_roundtrip():
    a = make()
    col = nda.NdColumn.from_arrays([a, None])
    pa_arr = col.to_arrow()
    back = nda.NdColumn.from_arrow(pa_arr)
    assert back[0] == a
    assert back[1] is None


def test_extension_metadata_exported():
    col = nda.NdColumn.from_arrays([make()])
    schema = pa.schema([pa.field(col)])
    metadata = schema.field(0).metadata
    assert metadata[b"ARROW:extension:name"] == b"nd_arrow.array"
    assert metadata[b"ARROW:extension:metadata"] == b'{"version":1}'


def test_pa_array_of_ndarray():
    out = pa.array(make())
    assert len(out) == 1
    assert nda.NdColumn.from_arrow(out)[0] == make()


def test_from_arrow_rejects_plain_struct():
    plain = pa.array([{"a": 1}])
    with pytest.raises(nda.NdFormatError):
        nda.NdColumn.from_arrow(plain)


def test_pyarrow_scalar_attribute():
    a = make()
    a.set_attr("stamp", pa.scalar(5, type=pa.int32()))
    assert a.attrs["stamp"] == 5
    assert isinstance(a.attrs["stamp"], np.int32)


def test_attribute_without_numpy_dtype_uses_pyarrow():
    import decimal

    a = make()
    a.set_attr("d", pa.scalar(decimal.Decimal("1.50"), type=pa.decimal128(5, 2)))
    assert a.attrs["d"] == decimal.Decimal("1.50")


def test_sliced_column_with_attrs():
    a = make()
    b = make().with_attrs({"units": "degC"})
    c = make().with_attrs({})
    sliced = nda.NdColumn.from_arrays([a, b, c]).to_arrow().slice(1, 2)
    back = nda.NdColumn.from_arrow(sliced)
    assert len(back) == 2
    assert back[0] == b
    assert back[1] == c


def test_sliced_record_batch():
    import numpy as np

    rows = [
        nda.NdRecordBatch({"v": make().with_attrs({"n": i})}, attrs={"i": i})
        for i in range(3)
    ]
    rb = nda.encode_batches(rows).slice(1, 1)
    (back,) = nda.NdRecordBatch.from_arrow(rb)
    assert back.attrs == {"i": 1}
    assert back["v"].attrs == {"n": 1}
    np.testing.assert_array_equal(back["v"].to_numpy(), make().to_numpy())


def test_from_arrow_rejects_struct_with_null_rows():
    with pytest.raises((TypeError, ValueError)):
        nda.NdRecordBatch.from_arrow(pa.array([{"a": 1}, None]))
