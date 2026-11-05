# nd-arrow-array

N-dimensional arrays as an Arrow extension type, with NumPy-style kernels.

Each row of an `nd_arrow.array` column holds one complete array: flat
values, named dimensions and optional attributes. Because it is ordinary
Arrow, the data passes through IPC, Parquet, Flight and DataFusion unchanged.

```text
Struct {
  values:    List<T>,        // flat values, C-order
  dim_names: List<Utf8>,     // one name per axis
  dim_sizes: List<UInt32>,   // one size per axis
  attrs:     Struct { .. },  // optional: one child per attribute key
}
```

Nulls in `values` are missing elements. Rank 0 is a scalar.

## Concepts

- `NdArrowArray`: one array with named `Dimensions` and `Attributes`.
- `NdColumn`: an Arrow column of the extension type.
- `NdRecordBatch`: arrays on one shared grid. Each column can use a subset of
  the dimensions, like an xarray `Dataset`. Batch-level (global) attributes
  travel in a reserved `__global_attrs__` column.

## Kernels

`nd_arrow_array::compute` follows NumPy and xarray semantics. Axes are
addressed by name, and broadcasting aligns axes by name. Reductions skip
nulls, like the NumPy `nan*` functions. The kernels cover:

- layout (broadcast, transpose, select, concat, pad);
- arithmetic, math and comparisons;
- reductions, scans, sort and search;
- interpolation, rolling windows, group-by and time resampling;
- CF decoding of fill values, packing and time.

Kernels are tested against NumPy reference results.

## Example

Create a `(time=2, lat=3)` array, subtract the mean over time, and keep the
first two latitudes:

```rust
use std::sync::Arc;

use arrow::array::Float64Array;
use nd_arrow_array::compute::{Selector, Slice};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray};

let sst = NdArrowArray::try_new(
    Arc::new(Float64Array::from(vec![10.0, 12.0, 14.0, 11.0, 13.0, 15.0])),
    Dimensions::try_new([("time", 2), ("lat", 3)])?,
)?
.with_attributes(Attributes::new().with("units", "degC"));

let anomaly = (&sst - &sst.mean(Some(&["time"]))?)?;
let north = anomaly.isel(&[("lat", Selector::Slice(Slice::range(0, 2)))])?;
println!("{north}");
```

```text
<NdArrowArray Float64 (time: 2, lat: 2)>
[[-0.5, -0.5],
 [0.5, 0.5]]
```

Run it with `cargo run --example basic`.

## Python

The `nd-arrow-py` directory holds the Python package `nd-arrow-array`
(`import nd_arrow_array`). Build it with [maturin](https://www.maturin.rs/):

```bash
cd nd-arrow-py
maturin develop
```

Create arrays from NumPy, group them in a record batch and send it to pyarrow:

```python
import numpy as np
import nd_arrow_array as nda

sst = nda.NdArray.from_numpy(
    np.arange(6.0).reshape(2, 3), dims=["time", "lat"], attrs={"units": "K"}
)
lat = nda.NdArray.from_numpy(np.array([-10.0, 0.0, 10.0]), dims=["lat"])
ds = nda.NdRecordBatch({"sst": sst, "lat": lat}, attrs={"title": "demo"})

rb = ds.to_arrow()                       # pyarrow.RecordBatch, one row
(back,) = nda.NdRecordBatch.from_arrow(rb)
back["sst"].to_numpy()                   # (2, 3) ndarray
```

See [nd-arrow-py/README.md](nd-arrow-py/README.md) for details.

## Interoperability

- Conversion to and from the Arrow canonical tensor types.
- Conversion to and from `ndarray::ArrayD` (feature `ndarray`).
- Reductions that merge exactly across chunks.

## Features

| Feature | Effect |
|---|---|
| `parallel` | Runs large kernels with rayon. |
| `ndarray` | Adds `ndarray` conversion. |

See [FORMAT.md](FORMAT.md) for the format specification, and the crate
documentation for the full API.
