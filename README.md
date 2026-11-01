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
  attrs:     Struct { .. },  // optional: one nullable child per attribute key
}
```

A column without attributes leaves out the `attrs` field. Readers also accept
an `attrs` struct with zero fields, and treat it as no attributes.

### Rules

- The field order is fixed: `values`, `dim_names`, `dim_sizes`, then `attrs` if present.
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

## Global attributes

Attributes that describe a whole dataset (`title`, `institution`,
`Conventions`) belong to the batch, not to one array. `NdRecordBatch` holds
them, like xarray `Dataset.attrs`, and the batch kernels keep them.

An encoded record batch stores them in one extra column:

```text
__global_attrs__: Struct { title: Utf8, institution: Utf8, ... }
  ARROW:extension:name = nd_arrow.global_attributes
```

- One row per nd batch, so each row keeps its own global attributes when
  encoded batches are concatenated.
- The struct holds the union of keys. A null means that the row does not have
  that key.
- The column is left out when no batch has global attributes.
- Decoders leave it out of the logical schema, and `materialize` does not output it.

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
| Layout | `broadcast_to`, `broadcast_arrays`, `transpose`, `transpose_reversed`, `move_dim`, `squeeze`, `expand_dims`, `rename_dims`, `isel`, `take`, `flip`, `roll`, `concat`, `stack`, `pad`, `tile`, `repeat` |
| Arithmetic | `add`, `sub`, `mul`, `div` (true division), `floor_divide`, `mod_`, `rem` (like `fmod`), `pow`, `neg`, `abs`, operators `+ - * / % & \| ^ ! -` on `&NdArrowArray` |
| Comparison and logic | `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `logical_and`, `logical_or`, `logical_xor`, `logical_not` |
| Math | `sqrt`, `cbrt`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, trigonometric and hyperbolic functions, `floor`, `ceil`, `trunc`, `round`, `deg2rad`, `rad2deg`, `isnan`, `isinf`, `isfinite` |
| Missing values | `fill_null`, `where_`, `clip`, `nan_to_null`, `is_null_mask`, `is_not_null_mask`, `cast` |
| CF conventions | `decode_cf` (mask `_FillValue`, `missing_value`, valid range; unpack `scale_factor` and `add_offset`), `encode_cf` |
| CF time | `decode_cf_time` (`"days since 1950-01-01"` to `Timestamp(Microsecond)`), `encode_cf_time`, `floor_time`, `time_component` |
| Reductions | `sum`, `prod`, `mean`, `min`, `max`, `count`, `var`, `std`, `median`, `quantile`, `any`, `all`, `argmin`, `argmax`, `ptp`, `reduce` |
| Block and weighted | `coarsen`, `weighted_sum`, `weighted_mean`, `dot` |
| Mergeable states | `PartialReduction` (`new`, `merge`, `finish`), `merge_all`: exact results across chunks |
| Scans | `cumsum`, `cumprod`, `cummin`, `cummax`, `diff` |
| Search and sort | `searchsorted`, `sort`, `argsort`, `unique`, `argwhere`, `isin`, `digitize` |
| Counting | `histogram`, `bincount` |
| Resampling | `interp` and `interp_with` (linear, nearest), `rolling` (sum, mean, min, max, count), `groupby_reduce` |
| Whole columns | `column_unary`, `column_binary`, `column_cast`, `column_decode_cf`, `row_reduce`, `row_reduce_axes` |
| Record batch | `materialize`, `isel`, `sel_range`, `sel_nearest`, `interp`, `coordinate` |

Common kernels are also methods, for example `a.sum(Some(&["time"]))` and
`a.transpose(&["lat", "time"])`. `NdArrowArray` prints like NumPy with
`{}`, and summarizes arrays with more than 1000 values.

Rules:

- Type promotion follows `numpy.result_type`. Integer overflow wraps.
- Sums and products of integers widen to `Int64` or `UInt64`.
- Statistics (`mean`, `var`, `std`, `median`, `quantile`) compute in `Float64`.
  The result is `Float32` for `Float32` input.
- Float sums over a contiguous axis use NumPy's pairwise summation, so they
  round exactly like NumPy.
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

## Interoperability

- `tensor::to_variable_shape_tensor` and `tensor::to_fixed_shape_tensor`
  convert an nd column to the Arrow canonical types
  `arrow.variable_shape_tensor` and `arrow.fixed_shape_tensor`.
  `tensor::column_from_field` reads nd and tensor columns. These types need
  one set of dimension names per column, no attributes and no null elements.
- With the `ndarray` feature: `NdArrowArray::to_ndarray` and
  `NdArrowArray::from_ndarray`.

## Features

| Feature | Effect |
|---|---|
| `parallel` | Runs large gathers and contiguous reductions with rayon. |
| `ndarray` | Adds conversion to and from `ndarray::ArrayD`. |

## Tests

- Unit tests cover each kernel.
- `tests/numpy.rs` compares the kernels with NumPy reference results in
  `tests/fixtures/numpy.json`. Regenerate the file with
  `python3 tests/fixtures/gen_numpy.py` (needs NumPy).
- `tests/properties.rs` checks kernel identities on random shapes with proptest.
- `cargo bench` runs the Criterion benchmarks in `benches/kernels.rs`.
- CI (`.github/workflows/ci.yml`) runs `fmt`, `clippy`, the tests with and
  without features, the docs, and the NumPy comparison against the latest
  NumPy release.
