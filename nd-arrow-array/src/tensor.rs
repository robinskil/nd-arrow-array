//! Conversion between nd columns and the Arrow canonical tensor types
//! `arrow.fixed_shape_tensor` and `arrow.variable_shape_tensor`.
//!
//! The canonical types keep the dimension names in the field metadata, so all
//! rows of a column must share their names and rank. They have no attributes,
//! and their elements cannot be null: fill nulls before a conversion.
//!
//! The canonical `dim_names` and `shape` describe the physical row-major
//! layout, which is the nd layout. A `permutation` is only a view hint, so a
//! conversion from a tensor ignores it.
//!
//! This module writes and reads the metadata JSON itself, by the Arrow
//! specification. It reads both `permutation` and `permutations` keys.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeListArray, Int32Array, ListArray, MutableArrayData, StructArray, UInt64Array,
    make_array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Fields};
use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
use serde::{Deserialize, Serialize};

use crate::NdArrowArray;
use crate::column::{NdColumn, NdColumnBuilder};
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};
use crate::extension::is_nd_field;

/// Extension name of the Arrow canonical fixed shape tensor.
pub const FIXED_SHAPE_TENSOR: &str = "arrow.fixed_shape_tensor";
/// Extension name of the Arrow canonical variable shape tensor.
pub const VARIABLE_SHAPE_TENSOR: &str = "arrow.variable_shape_tensor";

#[derive(Debug, Default, Serialize, Deserialize)]
struct TensorMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shape: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dim_names: Option<Vec<String>>,
    #[serde(
        default,
        alias = "permutations",
        skip_serializing_if = "Option::is_none"
    )]
    permutation: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    uniform_shape: Option<Vec<Option<i32>>>,
}

fn tensor_field(
    name: &str,
    data_type: DataType,
    extension: &str,
    metadata: &TensorMetadata,
) -> Field {
    let json = serde_json::to_string(metadata).expect("metadata serializes");
    Field::new(name, data_type, true).with_metadata(HashMap::from([
        (EXTENSION_TYPE_NAME_KEY.to_string(), extension.to_string()),
        (EXTENSION_TYPE_METADATA_KEY.to_string(), json),
    ]))
}

fn read_metadata(field: &Field) -> Result<TensorMetadata> {
    match field.metadata().get(EXTENSION_TYPE_METADATA_KEY) {
        None => Ok(TensorMetadata::default()),
        Some(json) if json.trim().is_empty() => Ok(TensorMetadata::default()),
        Some(json) => serde_json::from_str(json)
            .map_err(|e| nd_err!(InvalidEncoding, "invalid tensor metadata: {e}")),
    }
}

/// The dimensions that all non-null rows share in name (and in size, if `sizes`).
fn shared_dims(column: &NdColumn, sizes: bool) -> Result<Dimensions> {
    let mut shared: Option<Dimensions> = None;
    for row in (0..column.len()).filter(|&r| !column.is_null(r)) {
        let dims = column.dimensions_at(row);
        match &shared {
            None => shared = Some(dims),
            Some(first) if first.names() != dims.names() || (sizes && *first != dims) => {
                return Err(nd_err!(
                    NotSupported,
                    "tensor types need one set of dimensions, got {first} and {dims}"
                ));
            }
            Some(_) => {}
        }
    }
    Ok(shared.unwrap_or_default())
}

fn check_no_null_elements(column: &NdColumn) -> Result<()> {
    if column.values_list().values().null_count() > 0 {
        return Err(nd_err!(
            NotSupported,
            "tensor elements cannot be null; fill the nulls first"
        ));
    }
    Ok(())
}

fn element_field(value_type: &DataType) -> Arc<Field> {
    Arc::new(Field::new_list_field(value_type.clone(), false))
}

fn names_of(dims: &Dimensions) -> Option<Vec<String>> {
    (dims.rank() > 0).then(|| dims.names().iter().map(|n| n.to_string()).collect())
}

/// Convert to an `arrow.variable_shape_tensor` field named `name` and its
/// storage. The values are not copied.
pub fn to_variable_shape_tensor(column: &NdColumn, name: &str) -> Result<(Field, StructArray)> {
    check_no_null_elements(column)?;
    let dims = shared_dims(column, false)?;
    let rank = dims.rank();

    let mut shapes: Vec<i32> = Vec::with_capacity(column.len() * rank);
    let mut uniform: Vec<Option<i32>> = Vec::new();
    for row in 0..column.len() {
        let row_dims = (!column.is_null(row)).then(|| column.dimensions_at(row));
        for axis in 0..rank {
            let size = match &row_dims {
                Some(d) => i32::try_from(d.as_slice()[axis].size())
                    .map_err(|_| nd_err!(NotSupported, "dimension size exceeds Int32"))?,
                None => 0,
            };
            shapes.push(size);
            if row_dims.is_some() {
                match uniform.get_mut(axis) {
                    None => uniform.push(Some(size)),
                    Some(slot) if *slot != Some(size) => *slot = None,
                    Some(_) => {}
                }
            }
        }
    }

    let values = column.values_list();
    let data = ListArray::try_new(
        element_field(column.value_type()),
        values.offsets().clone(),
        values.values().clone(),
        None,
    )?;
    let shape = FixedSizeListArray::try_new(
        Arc::new(Field::new_list_field(DataType::Int32, false)),
        rank as i32,
        Arc::new(Int32Array::from(shapes)),
        None,
    )?;
    let fields = Fields::from(vec![
        Field::new("data", data.data_type().clone(), false),
        Field::new("shape", shape.data_type().clone(), false),
    ]);
    let storage = StructArray::try_new(
        fields,
        vec![Arc::new(data), Arc::new(shape)],
        column.storage().nulls().cloned(),
    )?;
    let metadata = TensorMetadata {
        dim_names: names_of(&dims),
        uniform_shape: (rank > 0 && uniform.len() == rank).then_some(uniform),
        ..Default::default()
    };
    let field = tensor_field(
        name,
        storage.data_type().clone(),
        VARIABLE_SHAPE_TENSOR,
        &metadata,
    );
    Ok((field, storage))
}

/// Convert to an `arrow.fixed_shape_tensor` field named `name` and its
/// storage. All non-null rows need the same dimensions.
pub fn to_fixed_shape_tensor(column: &NdColumn, name: &str) -> Result<(Field, FixedSizeListArray)> {
    check_no_null_elements(column)?;
    let dims = shared_dims(column, true)?;
    let size = dims.num_elements();

    let values = column.values_list();
    let data = values.values().to_data();
    let offsets = values.value_offsets();
    let mut out = MutableArrayData::new(vec![&data], true, column.len() * size);
    for row in 0..column.len() {
        if column.is_null(row) {
            out.extend_nulls(size);
        } else {
            out.extend(0, offsets[row] as usize, offsets[row + 1] as usize);
        }
    }
    let array = FixedSizeListArray::try_new(
        element_field(column.value_type()),
        size as i32,
        make_array(out.freeze()),
        column.storage().nulls().cloned(),
    )?;
    let metadata = TensorMetadata {
        shape: Some(dims.shape()),
        dim_names: names_of(&dims),
        ..Default::default()
    };
    let field = tensor_field(
        name,
        array.data_type().clone(),
        FIXED_SHAPE_TENSOR,
        &metadata,
    );
    Ok((field, array))
}

fn dimension_names(names: Option<Vec<String>>, rank: usize) -> Result<Vec<String>> {
    match names {
        Some(names) if names.len() == rank => Ok(names),
        Some(names) => Err(nd_err!(
            InvalidEncoding,
            "tensor has {} dimension names for rank {rank}",
            names.len()
        )),
        None => Ok((0..rank).map(|i| format!("dim_{i}")).collect()),
    }
}

fn from_variable_shape_tensor(storage: &StructArray, metadata: TensorMetadata) -> Result<NdColumn> {
    let (Some(data), Some(shape)) = (
        storage.column(0).as_any().downcast_ref::<ListArray>(),
        storage
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>(),
    ) else {
        return Err(nd_err!(
            InvalidEncoding,
            "variable shape tensor storage must be Struct{{data: List, shape: FixedSizeList}}"
        ));
    };
    let rank = shape.value_length() as usize;
    let names = dimension_names(metadata.dim_names, rank)?;
    let sizes = cast(shape.values().as_ref(), &DataType::UInt64)?;
    let sizes = sizes
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cast to UInt64");

    let value_type = match data.data_type() {
        DataType::List(item) => item.data_type().clone(),
        _ => unreachable!(),
    };
    let mut builder = NdColumnBuilder::with_value_type(value_type);
    for row in 0..storage.len() {
        if storage.is_null(row) {
            builder.append_null();
            continue;
        }
        let base = shape.value_offset(row) as usize;
        let dims = Dimensions::try_new(
            names
                .iter()
                .enumerate()
                .map(|(axis, name)| (name.as_str(), sizes.value(base + axis) as usize)),
        )?;
        builder.append(&NdArrowArray::try_new(data.value(row), dims)?)?;
    }
    builder.finish()
}

fn from_fixed_shape_tensor(
    array: &FixedSizeListArray,
    metadata: TensorMetadata,
) -> Result<NdColumn> {
    let shape = metadata
        .shape
        .ok_or_else(|| nd_err!(InvalidEncoding, "fixed shape tensor metadata has no shape"))?;
    let names = dimension_names(metadata.dim_names, shape.len())?;
    let dims = Dimensions::try_new(names.iter().map(String::as_str).zip(shape))?;
    if dims.num_elements() != array.value_length() as usize {
        return Err(nd_err!(
            InvalidEncoding,
            "fixed shape tensor shape {dims} does not match list size {}",
            array.value_length()
        ));
    }
    let value_type = array.value_type();
    let mut builder = NdColumnBuilder::with_value_type(value_type);
    for row in 0..array.len() {
        if array.is_null(row) {
            builder.append_null();
        } else {
            builder.append(&NdArrowArray::try_new(array.value(row), dims.clone())?)?;
        }
    }
    builder.finish()
}

/// Read a column that is an nd column or a canonical tensor column, by the
/// extension name on `field`.
pub fn column_from_field(field: &Field, array: &dyn Array) -> Result<NdColumn> {
    if is_nd_field(field) {
        return NdColumn::try_from_array(array);
    }
    match field.extension_type_name() {
        Some(VARIABLE_SHAPE_TENSOR) => {
            let storage = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    nd_err!(
                        InvalidEncoding,
                        "variable shape tensor storage must be a Struct"
                    )
                })?;
            from_variable_shape_tensor(storage, read_metadata(field)?)
        }
        Some(FIXED_SHAPE_TENSOR) => {
            let storage = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    nd_err!(
                        InvalidEncoding,
                        "fixed shape tensor storage must be a FixedSizeList"
                    )
                })?;
            from_fixed_shape_tensor(storage, read_metadata(field)?)
        }
        other => Err(nd_err!(
            InvalidEncoding,
            "field '{}' has extension type {other:?}, not an nd or tensor type",
            field.name()
        )),
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float32Array};
    use arrow::datatypes::Float32Type;

    use super::*;

    fn nd(values: Vec<f32>, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(
            Arc::new(Float32Array::from(values)),
            Dimensions::try_new(dims.iter().copied()).unwrap(),
        )
        .unwrap()
    }

    fn column(arrays: &[Option<NdArrowArray>]) -> NdColumn {
        let mut builder = NdColumnBuilder::with_value_type(DataType::Float32);
        for a in arrays {
            builder.append_option(a.as_ref()).unwrap();
        }
        builder.finish().unwrap()
    }

    #[test]
    fn variable_shape_round_trip() {
        let a = nd(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[("y", 2), ("x", 3)]);
        let b = nd(vec![7.0, 8.0], &[("y", 1), ("x", 2)]);
        let col = column(&[Some(a.clone()), None, Some(b.clone())]);
        let (field, storage) = to_variable_shape_tensor(&col, "t").unwrap();
        assert_eq!(
            field.metadata()[EXTENSION_TYPE_METADATA_KEY],
            r#"{"dim_names":["y","x"],"uniform_shape":[null,null]}"#
        );
        let back = column_from_field(&field, &storage).unwrap();
        assert_eq!(back.value(0), a);
        assert!(back.is_null(1));
        assert_eq!(back.value(2), b);
    }

    #[test]
    fn fixed_shape_round_trip() {
        let a = nd(vec![1.0, 2.0, 3.0, 4.0], &[("y", 2), ("x", 2)]);
        let col = column(&[Some(a.clone()), None, Some(a.clone())]);
        let (field, array) = to_fixed_shape_tensor(&col, "t").unwrap();
        assert_eq!(array.value_length(), 4);
        let back = column_from_field(&field, &array).unwrap();
        assert_eq!(back.value(2), a);
        assert!(back.is_null(1));
        assert_eq!(
            back.value(0)
                .values()
                .as_primitive::<Float32Type>()
                .values(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn reads_metadata_from_other_writers() {
        let array = FixedSizeListArray::try_new(
            element_field(&DataType::Float32),
            2,
            Arc::new(Float32Array::from(vec![1.0, 2.0])),
            None,
        )
        .unwrap();
        let field =
            Field::new("t", array.data_type().clone(), true).with_metadata(HashMap::from([
                (
                    EXTENSION_TYPE_NAME_KEY.to_string(),
                    FIXED_SHAPE_TENSOR.to_string(),
                ),
                (
                    EXTENSION_TYPE_METADATA_KEY.to_string(),
                    r#"{"shape":[2],"permutation":[0]}"#.to_string(),
                ),
            ]));
        let col = column_from_field(&field, &array).unwrap();
        assert_eq!(col.value(0).dims().names(), vec!["dim_0"]);
    }

    #[test]
    fn mismatched_rows_and_null_elements_rejected() {
        let a = nd(vec![1.0, 2.0], &[("x", 2)]);
        let b = nd(vec![1.0, 2.0], &[("y", 2)]);
        assert!(to_variable_shape_tensor(&column(&[Some(a.clone()), Some(b)]), "t").is_err());
        let c = nd(vec![1.0, 2.0, 3.0], &[("x", 3)]);
        assert!(to_fixed_shape_tensor(&column(&[Some(a), Some(c)]), "t").is_err());
        let with_null = NdArrowArray::try_new(
            Arc::new(Float32Array::from(vec![Some(1.0), None])),
            Dimensions::try_new([("x", 2)]).unwrap(),
        )
        .unwrap();
        assert!(to_variable_shape_tensor(&column(&[Some(with_null)]), "t").is_err());
    }
}
