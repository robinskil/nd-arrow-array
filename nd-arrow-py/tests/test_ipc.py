from pathlib import Path

import numpy as np
import pytest

import nd_arrow as nda

pa = pytest.importorskip("pyarrow")

# The Rust test nd-arrow/tests/python_fixture.rs reads this file.
FIXTURE = Path(__file__).resolve().parents[2] / "nd-arrow" / "tests" / "fixtures" / "python.arrow"


def dataset():
    sst = nda.NdArray.from_numpy(
        np.ma.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], mask=[[0, 0, 1], [0, 0, 0]]),
        dims=["time", "lat"],
        attrs={"units": "K", "valid_range": np.array([0, 400], dtype="float32")},
    )
    lat = nda.NdArray.from_numpy(np.array([-10, 0, 10], dtype="int32"), dims=["lat"])
    time = nda.NdArray.from_numpy(
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]"), dims=["time"]
    )
    return nda.NdRecordBatch(
        {"sst": sst, "lat": lat, "time": time}, attrs={"title": "python fixture"}
    )


def write(path, batch):
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as writer:
            writer.write_batch(batch)


def test_ipc_file_roundtrip(tmp_path):
    path = tmp_path / "ds.arrow"
    write(path, dataset().to_arrow())
    with pa.memory_map(str(path)) as source:
        table = pa.ipc.open_file(source).read_all()
    (back,) = nda.NdRecordBatch.from_arrow(table)
    ds = dataset()
    assert back.attrs == ds.attrs
    assert back.keys() == ds.keys()
    for key in ds.keys():
        assert back[key] == ds[key]


def test_write_rust_fixture():
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    write(FIXTURE, dataset().to_arrow())
    assert FIXTURE.stat().st_size > 0
