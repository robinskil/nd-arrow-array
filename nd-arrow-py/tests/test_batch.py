import builtins
import sys

import numpy as np
import pytest

import nd_arrow_array as nda


def sst(scale=1.0):
    return nda.NdArray.from_numpy(
        np.arange(6.0).reshape(2, 3) * scale, dims=["time", "lat"], attrs={"units": "K"}
    )


def lat():
    return nda.NdArray.from_numpy(np.array([-10.0, 0.0, 10.0]), dims=["lat"])


def dataset(scale=1.0, title="demo"):
    return nda.NdRecordBatch({"sst": sst(scale), "lat": lat()}, attrs={"title": title})


def test_infer_target():
    ds = dataset()
    assert ds.target == {"time": 2, "lat": 3}
    assert ds["sst"] == sst()
    assert len(ds) == 2


def test_explicit_target_rejects_bad_column():
    with pytest.raises(ValueError):
        nda.NdRecordBatch({"lat": lat()}, target={"x": 2})


def test_explicit_target():
    ds = nda.NdRecordBatch({"lat": lat()}, target=[("time", 4), ("lat", 3)])
    assert ds.target == {"time": 4, "lat": 3}


def test_schema_order_follows_dict():
    assert list(dataset().keys()) == ["sst", "lat"]


def test_missing_key():
    with pytest.raises(KeyError):
        dataset()["nope"]


def test_attrs_getter_and_setter():
    ds = dataset()
    ds.attrs["x"] = 1
    assert ds.attrs == {"title": "demo"}
    ds.attrs = {"source": "model"}
    assert ds.attrs == {"source": "model"}


def test_global_attrs_roundtrip():
    pytest.importorskip("pyarrow")
    rb = dataset().to_arrow()
    back = nda.NdRecordBatch.from_arrow(rb)
    assert len(back) == 1
    assert back[0].attrs == {"title": "demo"}
    assert back[0]["sst"] == sst()
    assert back[0]["sst"].attrs == {"units": "K"}


def test_two_row_encode():
    pytest.importorskip("pyarrow")
    rb = nda.encode_batches([dataset(1.0, "a"), dataset(2.0, "b")])
    assert rb.num_rows == 2
    back = nda.NdRecordBatch.from_arrow(rb)
    assert [b.attrs["title"] for b in back] == ["a", "b"]
    assert back[1]["sst"] == sst(2.0)


def test_reserved_name_rejected():
    pytest.importorskip("pyarrow")
    ds = nda.NdRecordBatch({"__global_attrs__": lat()}, attrs={"t": "x"})
    with pytest.raises(ValueError):
        ds.to_arrow()


def test_stream_capsule():
    pa = pytest.importorskip("pyarrow")
    reader = pa.RecordBatchReader.from_stream(dataset())
    table = reader.read_all()
    assert table.num_rows == 1
    assert table.column_names == ["sst", "lat", "__global_attrs__"]
    back = nda.NdRecordBatch.from_arrow(table.to_batches()[0])
    assert back[0]["lat"] == lat()


def test_from_arrow_accepts_stream():
    pa = pytest.importorskip("pyarrow")
    table = pa.Table.from_batches([dataset().to_arrow()])
    back = nda.NdRecordBatch.from_arrow(table)
    assert back[0].attrs == {"title": "demo"}


def test_no_pyarrow_for_numpy_path(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m == "pyarrow" or m.startswith("pyarrow.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", blocked)

    a = nda.NdArray.from_numpy(np.zeros(2), dims=["x"])
    a.to_numpy()
    with pytest.raises(ImportError, match="pyarrow is required"):
        nda.NdRecordBatch({"a": a}).to_arrow()
