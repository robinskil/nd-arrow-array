# nd_arrow.array format

Version 1

This document specifies how N-dimensional arrays are stored in Apache Arrow.
Any Arrow implementation can read and write the format with these rules.
The Rust crate in this repository is the reference implementation.

The words "must", "must not" and "can" have their usual meaning.
A writer must obey each "must" rule. A reader can reject data that breaks one.

## 1. Overview

An nd column is an Arrow extension type over a `Struct` storage type.
Each row of the column holds one complete N-dimensional array:

- the flat values, in C order;
- the name and size of each dimension;
- optional attributes, such as `units`.

The rows of one column can have different shapes and different dimensions.

## 2. Extension type

The field of an nd column has these two metadata keys:

| Key | Value |
|---|---|
| `ARROW:extension:name` | `nd_arrow.array` |
| `ARROW:extension:metadata` | `{"version":1}` |

The metadata value is a JSON object. Its `version` member is the format version.

- A writer must write `{"version":1}`.
- A reader must treat absent metadata, an empty string and `{}` as version 1.
- A reader must reject a version that it does not know.

The field can have other metadata keys. Readers must keep them.

## 3. Storage type

```text
Struct {
  values:    List<T>,       // not nullable
  dim_names: List<Utf8>,    // not nullable, items not nullable
  dim_sizes: List<UInt32>,  // not nullable, items not nullable
  attrs:     Struct { .. }, // optional, not nullable
}
```

Rules for the storage type:

1. The field names and the field order are fixed: `values`, `dim_names`,
   `dim_sizes`, then `attrs` if it is present.
2. `values` is a `List` of any Arrow type `T`. A writer names the item field
   `item` and makes it nullable.
3. The item field of `values` can have its own metadata, for example a nested
   extension type. Readers must keep this metadata.
4. `dim_names` is a `List<Utf8>`. `dim_sizes` is a `List<UInt32>`.
5. A column without attributes must not have the `attrs` field. This rule
   prevents a struct with zero fields, which some storage formats reject.
6. A reader must accept an `attrs` struct with zero fields. It means
   no attributes.

Version 1 uses `List` with 32-bit offsets. Thus one column chunk holds at most
2,147,483,647 values in total, over all rows.

## 4. Rows

Each non-null row must obey these rules:

1. The `values`, `dim_names` and `dim_sizes` lists are not null.
2. `dim_names` and `dim_sizes` have the same length. This length is the rank.
3. The `values` list holds exactly the product of `dim_sizes` items.
4. Each dimension name is not empty. The names in one row are unique.
5. The items of `dim_names` and `dim_sizes` are not null.
6. The values are in C order (row-major). The first dimension is the outermost.

Special cases:

- **Scalar:** a row with rank 0 has empty `dim_names` and `dim_sizes`. It holds
  exactly one value.
- **Empty array:** a dimension of size 0 gives a row with zero values.
- **Missing elements:** the null buffer of the `values` items marks missing
  elements. The format has no fill value and no separate mask. A writer must
  convert fill values to nulls.
- **Null row:** a null row of the struct is a missing array. A reader must
  decode it as a rank-0 null scalar. The child lists of a null row can hold
  any data; readers must ignore it.

### Element position

For a row with sizes `s[0], ..., s[n-1]`, the element at index
`(i[0], ..., i[n-1])` is at this flat position in `values`:

```text
position = i[0] * stride[0] + ... + i[n-1] * stride[n-1]
stride[n-1] = 1
stride[k]   = stride[k+1] * s[k+1]
```

## 5. Attributes

Attributes are metadata of one array, like netCDF or CF attributes.

1. The `attrs` struct has one child field per attribute key.
2. Each child field is nullable. Its type can be any Arrow type, for example
   `Utf8`, `Float64` or `List<Float32>` for `valid_range`.
3. The child fields hold the union of the keys of all rows in the column.
4. A null child value means that the row does not have that key.
5. One key has one type in a column. A writer must reject two types for one key.
6. The order of the keys has no meaning for equality.

## 6. Example

A `(time=2, lat=3)` array of `Float64` with `units = "K"`:

```text
values:    [0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
dim_names: ["time", "lat"]
dim_sizes: [2, 3]
attrs:     { units: "K" }
```

The element at `(time=1, lat=2)` is at position `1 * 3 + 2 = 5`, so its
value is `1.2`.

A second row of the same column can be a scalar without `units`:

```text
values:    [273.15]
dim_names: []
dim_sizes: []
attrs:     { units: null }
```

## 7. Record batches of nd arrays

A dataset, like an xarray `Dataset`, has several arrays on one shared grid.
The format stores a sequence of datasets in one Arrow record batch:

1. Each row of the record batch is one dataset.
2. Each nd column is one variable of the dataset.
3. All variables in one row must broadcast onto one shared grid. Each
   dimension of a variable has the grid size or size 1.
4. A variable can use a subset of the grid dimensions. For example, a
   coordinate uses one dimension, and a scalar uses none.

The format does not store the grid. A reader infers it from the variables:

1. The variable with the highest rank sets the dimension order.
2. A dimension that only lower-rank variables have goes at the end, in order
   of first appearance.
3. A size-1 dimension takes the larger size of another variable.

A **coordinate** is a variable with one dimension and the same name as that
dimension, as in xarray.

### Field and schema metadata

- The nd field of a variable keeps the metadata of the logical field. The
  extension keys take priority over equal keys.
- The record batch keeps the schema metadata unchanged.

## 8. Global attributes

Global attributes describe a whole dataset, like xarray `Dataset.attrs`.
They are stored in one extra column of the record batch:

| Item | Value |
|---|---|
| Column name | `__global_attrs__` |
| Storage type | `Struct { .. }`, not nullable |
| `ARROW:extension:name` | `nd_arrow.global_attributes` |
| `ARROW:extension:metadata` | `{"version":1}` |

Rules:

1. The column has one row per dataset.
2. The struct fields obey the attribute rules of section 5.
3. A writer must leave out the column when no dataset has global attributes.
4. A record batch has at most one global attributes column.
5. The name `__global_attrs__` is reserved. A variable must not use it.

## 9. Conformance

A conforming writer:

- writes the extension name and the metadata of section 2;
- writes the storage type of section 3;
- writes rows that obey section 4;
- leaves out empty `attrs` structs and empty global attributes columns.

A conforming reader:

- accepts all data that a conforming writer can write;
- accepts an `attrs` struct with zero fields;
- rejects an unknown version;
- decodes a null row as a rank-0 null scalar.

## 10. Future versions

A new version is necessary for a change that a version 1 reader cannot read.
For example, `LargeList` storage for `values` needs a new version.
A change that only adds optional metadata keys keeps version 1.
