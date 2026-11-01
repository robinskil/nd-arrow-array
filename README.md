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
- interpolation, rolling windows and group-by;
- CF decoding of fill values, packing and time.

```rust
let anomaly = (&sst - &sst.mean(Some(&["time"]))?)?;
let north = anomaly.isel(&[("lat", Selector::Slice(Slice::range(60, 120)))])?;
let (decoded, _) = compute::decode_cf(&packed)?;
```

Kernels are tested against NumPy reference results.

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
