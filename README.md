# nd-arrow-array

N-dimensional arrays as an Arrow extension type.

## Extension type

- Name: `nd_arrow.array` (in `ARROW:extension:name`).
- Metadata: `{"version":1}` (in `ARROW:extension:metadata`).
- Storage: a `Struct`. Each row holds one complete N-dimensional array.

```text
Struct {
  values:    List<T>,        // flat values, C-order (row-major)
  dim_names: List<Utf8>,     // one name per axis
  dim_sizes: List<UInt32>,   // one size per axis
  attrs:     Struct { .. },  // one nullable child per attribute key
}
```

### Rules

- The field order is fixed: `values`, `dim_names`, `dim_sizes`, `attrs`.
- `dim_names` and `dim_sizes` have the same length. This length is the rank.
- The length of `values` is the product of `dim_sizes`.
- Rank 0 (empty dimension lists) is a scalar with one value.
- Dimension names are not empty and are unique in a row.
- Validity of elements is in the null buffer of `values`. There is no fill value.
- An attribute can be any Arrow type. The `attrs` struct of a column holds the
  union of the attribute keys of all rows. A null child means that the row does
  not have that attribute.
- A null row decodes to a rank-0 null scalar.

### Example

A `(time=2, lat=3)` array of `Float64` with `units = "K"`:

```text
values:    [0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
dim_names: ["time", "lat"]
dim_sizes: [2, 3]
attrs:     { units: "K" }
```

## Rust API

- `NdArrowArray`: one N-dimensional array (values, `Dimensions`, `Attributes`).
- `NdArrayType`: the extension type. It implements `arrow_schema::extension::ExtensionType`.
- `NdColumn` and `NdColumnBuilder`: an Arrow column of the extension type.
- `NdRecordBatch`: nd arrays over one shared target grid. Each column can use a
  subset of the target dimensions. One nd batch encodes to one row of a
  `RecordBatch` of nd columns.

```rust
use std::sync::Arc;
use arrow::array::Float64Array;
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdColumn};

let sst = NdArrowArray::try_new(
    Arc::new(Float64Array::from(vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2])),
    Dimensions::try_new([("time", 2), ("lat", 3)])?,
)?
.with_attributes(Attributes::new().with("units", "K"));

let column = NdColumn::try_from_arrays([&sst])?;
let field = column.to_field("sst");
assert_eq!(column.value(0), sst);
```

## Kernels (`nd_arrow_array::compute`)

The kernels follow NumPy and xarray semantics. Axes are addressed by
dimension name, and broadcasting aligns axes by name. Nulls are missing values:
reductions skip them, like the NumPy `nan*` functions.

| Group | Kernels |
|---|---|
| Layout | `broadcast_to`, `broadcast_arrays`, `transpose`, `transpose_reversed`, `move_dim`, `squeeze`, `expand_dims`, `rename_dims`, `isel`, `take`, `flip`, `roll`, `concat`, `stack` |
| Arithmetic | `add`, `sub`, `mul`, `div` (true division), `rem`, `pow`, `neg`, `abs`, operators `+ - * / % & \| ^ ! -` on `&NdArrowArray` |
| Comparison and logic | `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `logical_and`, `logical_or`, `logical_xor`, `logical_not` |
| Math | `sqrt`, `cbrt`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, trigonometric and hyperbolic functions, `floor`, `ceil`, `trunc`, `round`, `deg2rad`, `rad2deg`, `isnan`, `isinf`, `isfinite` |
| Missing values | `fill_null`, `where_`, `clip`, `nan_to_null`, `is_null_mask`, `is_not_null_mask`, `cast` |
| CF conventions | `decode_cf` (mask `_FillValue`, `missing_value`, valid range; unpack `scale_factor` and `add_offset`), `encode_cf` |
| Reductions | `sum`, `prod`, `mean`, `min`, `max`, `count`, `var`, `std`, `median`, `quantile`, `any`, `all`, `argmin`, `argmax`, `reduce` |
| Scans | `cumsum`, `cumprod`, `diff` |
| Search and sort | `searchsorted`, `sort`, `argsort`, `unique` |
| Resampling | `interp` (linear), `rolling` (sum, mean, min, max, count), `groupby_reduce` |
| Record batch | `materialize`, `isel`, `sel_range`, `sel_nearest`, `interp`, `coordinate` |

Rules:

- Type promotion follows `numpy.result_type`. Integer overflow wraps.
- Sums and products of integers widen to `Int64` or `UInt64`.
- Statistics (`mean`, `var`, `std`, `median`, `quantile`) compute in `Float64`.
  The result is `Float32` for `Float32` input.
- Floats order NaN above all numbers, like Arrow. Use `nan_to_null` to treat NaN
  as missing.
- Attributes follow xarray defaults. Layout kernels, `cast`, `where_`,
  `fill_null` and `clip` keep them. Arithmetic, comparisons, math functions
  and reductions drop them.

```rust
use nd_arrow_array::compute::{self, Selector, Slice};

let anomaly = (&sst - &compute::mean(&sst, Some(&["time"]))?)?;
let tropics = compute::isel(&anomaly, &[("lat", Selector::Slice(Slice::range(60, 120)))])?;
let (decoded, encoding) = compute::decode_cf(&packed)?;
```

## Tests

- Unit tests cover each kernel.
- `tests/numpy.rs` compares the kernels with NumPy reference results in
  `tests/fixtures/numpy.json`. Regenerate the file with
  `python3 tests/fixtures/gen_numpy.py` (needs NumPy).
- `cargo bench` runs the Criterion benchmarks in `benches/kernels.rs`.
