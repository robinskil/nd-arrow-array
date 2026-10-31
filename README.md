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

## Roadmap

- Compute kernels (broadcast, slice, transpose, reductions), based on NumPy semantics.
