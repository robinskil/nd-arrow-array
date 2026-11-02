//! Kernels on a whole [`NdColumn`]: one kernel call for all rows.
//!
//! The `values` child of a column holds the values of all rows, one after
//! another. An elementwise kernel can run on that child once and keep the
//! offsets and dimensions. This replaces one call per row with one call per
//! column.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ListArray, StructArray, UInt64Array, new_null_array};
use arrow::compute::concat;
use arrow::datatypes::{DataType, Field, Int64Type};

use crate::column::{NdColumn, NdColumnBuilder};
use crate::compute::cf::decode_cf;
use crate::compute::reduce::{Reduction, reduce};
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};
use crate::extension::{ITEM_FIELD, NdArrayType};
use crate::{Attributes, NdArrowArray};

/// The values child of all rows as one 1-D array along dimension `n`.
fn flat_values(column: &NdColumn) -> Result<NdArrowArray> {
    let values = column.values_list();
    let first = values.value_offsets()[0] as usize;
    let last = *values.value_offsets().last().unwrap() as usize;
    let child = values.values().slice(first, last - first);
    NdArrowArray::try_new(child.clone(), Dimensions::try_new([("n", child.len())])?)
}

/// Replace the values child, keep offsets and dimensions. Without
/// `keep_attributes`, the `attrs` field goes away.
fn with_values(column: &NdColumn, values: ArrayRef, keep_attributes: bool) -> Result<NdColumn> {
    let list = column.values_list();
    let first = list.value_offsets()[0];
    let offsets = if first == 0 {
        list.offsets().clone()
    } else {
        arrow::buffer::OffsetBuffer::new(list.offsets().iter().map(|o| o - first).collect())
    };
    let item = Arc::new(Field::new(ITEM_FIELD, values.data_type().clone(), true));
    let new_list = ListArray::try_new(item.clone(), offsets, values, list.nulls().cloned())?;
    let attrs = if keep_attributes {
        column.extension_type().attribute_fields().clone()
    } else {
        Default::default()
    };
    let extension = NdArrayType::with_value_field(item, attrs);
    let storage = column.storage();
    let mut children: Vec<ArrayRef> = storage.columns()[..3].to_vec();
    children[0] = Arc::new(new_list);
    if extension.has_attributes() {
        children.push(storage.column(3).clone());
    }
    let storage = StructArray::try_new(
        extension.storage_fields(),
        children,
        storage.nulls().cloned(),
    )?;
    Ok(NdColumn::from_parts_unchecked(storage, extension))
}

/// Apply an elementwise unary kernel to all rows in one call, for example
/// `column_unary(&c, compute::sqrt, false)`.
///
/// The kernel must keep the number of values. `keep_attributes` follows the
/// attribute rule of the kernel: `true` for `cast` and `fill_null`, `false`
/// for math functions.
pub fn column_unary(
    column: &NdColumn,
    kernel: impl Fn(&NdArrowArray) -> Result<NdArrowArray>,
    keep_attributes: bool,
) -> Result<NdColumn> {
    let flat = flat_values(column)?;
    let out = kernel(&flat)?;
    if out.len() != flat.len() {
        return Err(nd_err!(
            InvalidArgument,
            "column kernel changed the number of values from {} to {}",
            flat.len(),
            out.len()
        ));
    }
    with_values(column, out.values().clone(), keep_attributes)
}

/// Cast the values of all rows. Keeps the attributes.
pub fn column_cast(column: &NdColumn, to: &DataType) -> Result<NdColumn> {
    column_unary(column, |a| crate::compute::cast(a, to), true)
}

/// Apply an elementwise binary kernel to two columns, for example `compute::add`.
///
/// When every row has the same dimensions in both columns, the kernel runs
/// once on the two values children. Otherwise it runs per row, with
/// broadcasting. A null row in either column gives a null row.
pub fn column_binary(
    a: &NdColumn,
    b: &NdColumn,
    kernel: impl Fn(&NdArrowArray, &NdArrowArray) -> Result<NdArrowArray>,
) -> Result<NdColumn> {
    if a.len() != b.len() {
        return Err(nd_err!(
            ShapeMismatch,
            "columns have {} and {} rows",
            a.len(),
            b.len()
        ));
    }
    let same_layout = (0..a.len()).all(|row| {
        a.is_null(row) == b.is_null(row) && a.dimensions_at(row) == b.dimensions_at(row)
    }) && a.values_list().value_offsets() == b.values_list().value_offsets();
    if same_layout {
        let out = kernel(&flat_values(a)?, &flat_values(b)?)?;
        return with_values(a, out.values().clone(), false);
    }
    let mut builder: Option<NdColumnBuilder> = None;
    let mut pending_nulls = 0usize;
    for row in 0..a.len() {
        match (a.get(row), b.get(row)) {
            (Some(x), Some(y)) => {
                let out = kernel(&x, &y)?;
                let builder = builder.get_or_insert_with(|| {
                    let mut b = NdColumnBuilder::with_value_type(out.data_type().clone());
                    (0..pending_nulls).for_each(|_| b.append_null());
                    b
                });
                builder.append(&out)?;
            }
            _ => match &mut builder {
                Some(builder) => builder.append_null(),
                None => pending_nulls += 1,
            },
        }
    }
    match builder {
        Some(builder) => builder.finish(),
        None => Ok(NdColumn::new_null(
            NdArrayType::new(a.value_type().clone(), Default::default()),
            a.len(),
        )),
    }
}

/// Decode CF packing and fill values of every row, see [`decode_cf`].
///
/// When all rows share the same CF attributes, the decode runs once on the
/// values of all rows. The CF keys leave the attributes.
pub fn column_decode_cf(column: &NdColumn) -> Result<NdColumn> {
    let rows: Vec<Option<NdArrowArray>> = column.iter().collect();
    let cf_of = |a: &NdArrowArray| -> Attributes {
        let mut kept = Attributes::new();
        for (key, value) in a.attributes().iter() {
            if crate::compute::cf::is_cf_key(key) {
                kept.try_insert_array(key, value.clone()).expect("length 1");
            }
        }
        kept
    };
    let present: Vec<&NdArrowArray> = rows.iter().flatten().collect();
    let shared = present.first().map(|a| cf_of(a));
    let uniform = shared
        .as_ref()
        .is_some_and(|first| present.iter().all(|a| cf_of(a) == *first));
    if uniform {
        let template = present[0];
        let flat = flat_values(column)?.with_attributes(template.attributes().clone());
        let (decoded, _) = decode_cf(&flat)?;
        let mut builder = NdColumnBuilder::with_value_type(decoded.data_type().clone());
        let offsets = column.values_list().value_offsets();
        let base = offsets[0] as usize;
        for (row, array) in rows.iter().enumerate() {
            match array {
                None => builder.append_null(),
                Some(array) => {
                    let start = offsets[row] as usize - base;
                    let len = offsets[row + 1] as usize - offsets[row] as usize;
                    let mut attrs = array.attributes().clone();
                    for (key, _) in cf_of(array).iter() {
                        attrs.remove(key);
                    }
                    let values = decoded.values().slice(start, len);
                    builder.append(
                        &NdArrowArray::try_new(values, array.dims().clone())?
                            .with_attributes(attrs),
                    )?;
                }
            }
        }
        return builder.finish();
    }
    let mut builder: Option<NdColumnBuilder> = None;
    let decoded: Vec<Option<NdArrowArray>> = rows
        .iter()
        .map(|r| r.as_ref().map(|a| decode_cf(a).map(|(d, _)| d)).transpose())
        .collect::<Result<_>>()?;
    for row in &decoded {
        let b = builder.get_or_insert_with(|| {
            let value_type = decoded
                .iter()
                .flatten()
                .next()
                .map_or(column.value_type().clone(), |a| a.data_type().clone());
            NdColumnBuilder::with_value_type(value_type)
        });
        b.append_option(row.as_ref())?;
    }
    builder
        .unwrap_or_else(|| NdColumnBuilder::with_value_type(column.value_type().clone()))
        .finish()
}

/// Reduce every row over all its axes: one value per row, like a SQL aggregate
/// per row. A null row gives null.
pub fn row_reduce(column: &NdColumn, op: Reduction) -> Result<ArrayRef> {
    if op == Reduction::Count {
        return Ok(Arc::new(row_counts(column)));
    }
    let parts: Vec<Option<ArrayRef>> = column
        .iter()
        .map(|row| {
            row.map(|a| reduce(&a, None, op).map(|r| r.values().clone()))
                .transpose()
        })
        .collect::<Result<_>>()?;
    let out_type = parts
        .iter()
        .flatten()
        .next()
        .map_or(DataType::Null, |p| p.data_type().clone());
    let null = new_null_array(&out_type, 1);
    let refs: Vec<&dyn Array> = parts
        .iter()
        .map(|p| p.as_ref().map_or(null.as_ref(), |p| p.as_ref()))
        .collect();
    if refs.is_empty() {
        return Ok(new_null_array(&out_type, 0));
    }
    Ok(concat(&refs)?)
}

/// Non-null values per row, computed from offsets and the null buffer only.
fn row_counts(column: &NdColumn) -> arrow::array::PrimitiveArray<Int64Type> {
    let list = column.values_list();
    let values = list.values();
    let offsets = list.value_offsets();
    (0..column.len())
        .map(|row| {
            (!column.is_null(row)).then(|| {
                let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
                let nulls = values
                    .logical_nulls()
                    .map_or(0, |n| (start..end).filter(|&i| n.is_null(i)).count());
                (end - start - nulls) as i64
            })
        })
        .collect()
}

/// Reduce every row over the named axes. Rows without an axis keep it out of
/// the reduction. Returns a new column.
pub fn row_reduce_axes(column: &NdColumn, axes: &[&str], op: Reduction) -> Result<NdColumn> {
    let mut reduced: Vec<Option<NdArrowArray>> = Vec::with_capacity(column.len());
    for array in column.iter() {
        reduced.push(
            array
                .map(|a| {
                    let own: Vec<&str> = axes
                        .iter()
                        .copied()
                        .filter(|name| a.dims().contains(name))
                        .collect();
                    reduce(&a, Some(&own), op)
                })
                .transpose()?,
        );
    }
    let value_type = reduced
        .iter()
        .flatten()
        .next()
        .map(|a| a.data_type().clone())
        .unwrap_or(DataType::Float64);
    let mut builder = NdColumnBuilder::with_value_type(value_type);
    for row in &reduced {
        builder.append_option(row.as_ref())?;
    }
    builder.finish()
}

/// Indices of non-null rows. Helper for callers that filter columns.
pub fn valid_rows(column: &NdColumn) -> UInt64Array {
    (0..column.len() as u64)
        .filter(|&row| !column.is_null(row as usize))
        .collect()
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Int16Array, Int32Array};
    use arrow::datatypes::{Float32Type, Float64Type, Int32Type};

    use super::*;
    use crate::compute::cf::{ADD_OFFSET, SCALE_FACTOR};
    use crate::compute::{add, sqrt};

    fn nd(values: Vec<i32>, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(
            Arc::new(Int32Array::from(values)),
            Dimensions::try_new(dims.iter().copied()).unwrap(),
        )
        .unwrap()
    }

    fn column(rows: &[Option<NdArrowArray>]) -> NdColumn {
        let mut b = NdColumnBuilder::with_value_type(DataType::Int32);
        for r in rows {
            b.append_option(r.as_ref()).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn unary_runs_once_and_keeps_layout() {
        let c = column(&[
            Some(nd(vec![1, 4], &[("x", 2)])),
            None,
            Some(nd(vec![9, 16, 25, 36], &[("y", 2), ("x", 2)])),
        ])
        .slice(0, 3);
        let out = column_unary(&c, sqrt, false).unwrap();
        assert_eq!(out.value_type(), &DataType::Float64);
        assert!(out.is_null(1));
        let row2 = out.value(2);
        assert_eq!(row2.dims().names(), vec!["y", "x"]);
        assert_eq!(
            row2.values().as_primitive::<Float64Type>().values(),
            &[3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn unary_on_a_sliced_column() {
        let c = column(&[Some(nd(vec![1], &[])), Some(nd(vec![4, 9], &[("x", 2)]))]).slice(1, 1);
        let out = column_unary(&c, sqrt, false).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out.value(0).values().as_primitive::<Float64Type>().values(),
            &[2.0, 3.0]
        );
    }

    #[test]
    fn binary_fast_and_per_row_paths() {
        let a = column(&[Some(nd(vec![1, 2], &[("x", 2)])), Some(nd(vec![3], &[]))]);
        let same = column_binary(&a, &a, add).unwrap();
        assert_eq!(
            same.value(0).values().as_primitive::<Int32Type>().values(),
            &[2, 4]
        );
        let b = column(&[Some(nd(vec![10], &[])), None]);
        let mixed = column_binary(&a, &b, add).unwrap();
        assert_eq!(
            mixed.value(0).values().as_primitive::<Int32Type>().values(),
            &[11, 12]
        );
        assert!(mixed.is_null(1));
    }

    #[test]
    fn row_reductions() {
        let c = column(&[
            Some(nd(vec![1, 2, 3, 4], &[("y", 2), ("x", 2)])),
            None,
            Some(nd(vec![5], &[])),
        ]);
        let sums = row_reduce(&c, Reduction::Sum).unwrap();
        let sums: Vec<Option<i64>> = sums.as_primitive::<Int64Type>().iter().collect();
        assert_eq!(sums, vec![Some(10), None, Some(5)]);
        let counts = row_reduce(&c, Reduction::Count).unwrap();
        assert_eq!(counts.as_primitive::<Int64Type>().value(0), 4);
        assert!(counts.is_null(1));

        let per_y = row_reduce_axes(&c, &["x"], Reduction::Max).unwrap();
        assert_eq!(per_y.value(0).dims().names(), vec!["y"]);
        assert_eq!(
            per_y.value(0).values().as_primitive::<Int32Type>().values(),
            &[2, 4]
        );
        assert!(per_y.value(2).is_scalar());
        assert_eq!(valid_rows(&c).values(), &[0, 2]);
    }

    #[test]
    fn decode_cf_uniform_and_mixed() {
        let packed = |values: Vec<i16>, scale: f32| {
            NdArrowArray::try_new(
                Arc::new(Int16Array::from(values)),
                Dimensions::try_new([("x", 2)]).unwrap(),
            )
            .unwrap()
            .with_attributes(
                Attributes::new()
                    .with(SCALE_FACTOR, scale)
                    .with(ADD_OFFSET, 1.0f32)
                    .with("units", "K"),
            )
        };
        let uniform =
            NdColumn::try_from_arrays([&packed(vec![1, 2], 0.5), &packed(vec![3, 4], 0.5)])
                .unwrap();
        let out = column_decode_cf(&uniform).unwrap();
        assert_eq!(out.value_type(), &DataType::Float32);
        assert_eq!(
            out.value(1).values().as_primitive::<Float32Type>().values(),
            &[2.5, 3.0]
        );
        assert_eq!(out.value(1).attributes().get_str("units"), Some("K"));
        assert!(!out.value(1).attributes().contains(SCALE_FACTOR));

        let mixed = NdColumn::try_from_arrays([&packed(vec![1, 2], 0.5), &packed(vec![1, 2], 2.0)])
            .unwrap();
        let out = column_decode_cf(&mixed).unwrap();
        assert_eq!(
            out.value(1).values().as_primitive::<Float32Type>().values(),
            &[3.0, 5.0]
        );
    }
}
