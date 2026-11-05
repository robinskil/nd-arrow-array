# nd-arrow-array (Python)

Python bindings for the `nd_arrow.array` Arrow extension type.
The PyPI package is `nd-arrow-array`. The import name is `nd_arrow_array`.
The bindings move N-dimensional arrays between NumPy, pyarrow and the
format in [FORMAT.md](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md).

## Install

```bash
pip install nd-arrow-array            # NumPy only
pip install nd-arrow-array[arrow]     # with pyarrow
pip install nd-arrow-array[xarray]    # with xarray and cftime
```

To build from source, run `pip install maturin` and `maturin develop` in this directory.

## Classes

| Class | Content |
|---|---|
| `NdArray` | One array: values, named dimensions, attributes. |
| `NdColumn` | An Arrow column. Each row is one `NdArray` or `None`. |
| `NdRecordBatch` | Named arrays on one shared grid, with global attributes. |

## Example

```python
import numpy as np
import nd_arrow_array as nda

a = nda.NdArray.from_numpy(
    np.arange(6.0).reshape(2, 3),
    dims=["time", "lat"],
    attrs={"units": "K", "valid_range": np.array([0, 400], dtype="float32")},
)
a.sizes            # {'time': 2, 'lat': 3}
a.set_attr("long_name", "sea surface temperature")
a.to_numpy()

col = nda.NdColumn.from_arrays([a, None])
arr = col.to_arrow()                    # pyarrow array
nda.NdColumn.from_arrow(arr)[0] == a    # True
```

## pyarrow

`pip install nd-arrow-array[arrow]` adds pyarrow. pyarrow sees an nd column as a
`struct` column. The field metadata `ARROW:extension:name = nd_arrow.array`
marks it as an nd column. See [FORMAT.md](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md).

### Arrays and columns

```python
import pyarrow as pa

# An array from flat pyarrow values in C order.
a = nda.NdArray(pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                dims={"time": 2, "lat": 3}, attrs={"units": "K"})
a.values                         # the flat values as a pyarrow.Array

col = nda.NdColumn.from_arrays([a, None])
arr = col.to_arrow()             # pyarrow.StructArray, one row per array
nda.NdColumn.from_arrow(arr)     # back to an NdColumn
pa.field(col).metadata           # the extension name and version
```

### Record batches and tables

```python
lat = nda.NdArray.from_numpy(np.array([-10.0, 0.0, 10.0]), dims=["lat"])
batch = nda.NdRecordBatch({"sst": a, "lat": lat}, attrs={"title": "demo"})

rb = batch.to_arrow()            # pyarrow.RecordBatch with one row
table = pa.table(batch)          # pyarrow.Table, through the PyCapsule interface

# Many batches with one schema: one row each.
rows = nda.encode_batches([batch, other_batch])
batches = nda.NdRecordBatch.from_arrow(rows)   # a list, one NdRecordBatch per row
```

`NdRecordBatch.from_arrow` accepts a `RecordBatch`, a `Table` or a
`RecordBatchReader`. It returns one `NdRecordBatch` for each row.
The global attributes travel in the `__global_attrs__` column.

### Files

Parquet and Arrow IPC keep the field metadata, so the nd columns survive a
write and a read:

```python
import pyarrow.parquet as pq

pq.write_table(pa.Table.from_batches([rows]), "data.parquet")
batches = nda.NdRecordBatch.from_arrow(pq.read_table("data.parquet"))

with pa.OSFile("data.arrow", "wb") as sink, pa.ipc.new_file(sink, rows.schema) as writer:
    writer.write_batch(rows)
with pa.memory_map("data.arrow") as source:
    batches = nda.NdRecordBatch.from_arrow(pa.ipc.open_file(source).read_all())
```

### Other libraries

`NdArray`, `NdColumn` and `NdRecordBatch` implement the Arrow PyCapsule
interface (`__arrow_c_array__`, `__arrow_c_stream__`, `__arrow_c_schema__`).
pyarrow, polars, DuckDB and other Arrow libraries read them without a copy,
for example `pa.array(col)` or `pa.table(batch)`.

## xarray

`pip install nd-arrow-array[xarray]` adds xarray and cftime.

```python
import xarray as xr

ds = xr.Dataset(
    {"air": (("time", "lat"), np.zeros((2, 3)), {"units": "K"})},
    coords={"time": np.array(["2020-01-01", "2020-01-02"], "datetime64[ns]"),
            "lat": [-10.0, 0.0, 10.0]},
)
batch = nda.NdRecordBatch.from_xarray(ds)   # all variables and coordinates
batch.to_xarray()                           # an xarray.Dataset

a = nda.NdArray.from_xarray(ds["air"])      # values, dims and attrs only
a.to_xarray(name="air")                     # an xarray.DataArray
```

- `NdRecordBatch.from_xarray` accepts a `Dataset` or a `DataArray`.
  Each variable and each coordinate becomes a column.
- A 1-D column with the name of its dimension becomes an index coordinate.
- Other coordinates go into the CF `coordinates` attribute of the data
  variables. A coordinate that no data variable uses goes into the
  `coordinates` attribute of the batch.
- `NdArray.from_xarray` does not keep coordinates.
- In `to_xarray`, nulls become `NaN` or `NaT`. Integer and bool arrays with
  nulls become `float64`, as in xarray CF decoding.
- Object arrays must hold `str` or cftime dates. `None` and `NaN` become nulls.
- cftime dates become `timestamp[us]` in the standard calendar:
  - `standard`, `gregorian`, `proleptic_gregorian` and `julian` keep the
    exact instant;
  - `noleap`, `365_day`, `all_leap`, `366_day` and `360_day` keep the year,
    month, day and time. A date that the standard calendar does not have,
    such as 30 February, raises `ValueError`. Use
    `convert_calendar(..., align_on="date")` in xarray first.

## Rules

- NumPy dtypes: `bool`, signed and unsigned integers, `float16`, `float32`,
  `float64`, `datetime64` and `timedelta64` (`s`, `ms`, `us`, `ns`) and `str`.
  Other dtypes raise `TypeError`.
- A `numpy.ma.MaskedArray` mask and `NaT` become nulls. `NaN` stays a value.
- `to_numpy()` returns a `MaskedArray` when the values have nulls.
- `attrs` returns a copy. Assign a new dict, or use `set_attr` and `del_attr`.
- All classes implement the Arrow PyCapsule interface, so pyarrow, polars
  and DuckDB can read them.
- The package ships type stubs (`py.typed`), so mypy and pyright check calls.

## Test

```bash
pip install pytest mypy numpy pyarrow xarray cftime
maturin develop && pytest
python -m mypy.stubtest nd_arrow_array          # stubs match the module
mypy --strict --warn-unused-ignores tests/typing
```
