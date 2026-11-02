# nd-arrow-array (Python)

Python bindings for the `nd_arrow.array` Arrow extension type.
The bindings move N-dimensional arrays between NumPy, pyarrow and the
format in [FORMAT.md](../FORMAT.md).

## Install

```bash
pip install maturin
maturin develop            # in this directory
pip install pyarrow        # optional, for to_arrow() and values
```

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
pip install pytest mypy numpy pyarrow
maturin develop && pytest
python -m mypy.stubtest nd_arrow_array          # stubs match the module
mypy --strict --warn-unused-ignores tests/typing
```
