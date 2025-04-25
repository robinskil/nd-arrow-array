use std::sync::Arc;

use arrow::{
    array::{Array, StructArray},
    datatypes::Field,
};

use crate::consts;

use super::{default::DefaultNdArrowArray, NdArrowArray};

#[derive(Debug, Clone, thiserror::Error)]
pub enum ArrowParseError {
    #[error("No dimension sizes found.")]
    NoDimensionSizes,
    #[error("No inner values array found.")]
    NoValues,
    #[error("Shape and inner array length do not match: expected {expected}, got {got}")]
    ShapeAndArrayLengthMismatch { expected: usize, got: usize },
    #[error("Failed to downcast Arrow Array to Encoded ListArray.")]
    ListDowncastError,
}

pub fn try_from_arrow_array(array: &StructArray) -> Result<Arc<dyn NdArrowArray>, ArrowParseError> {
    let dimension_sizes = array
        .column_by_name(consts::DIMENSION_SIZES)
        .map(|col| col.as_any().downcast_ref::<arrow::array::UInt32Array>())
        .flatten()
        .ok_or(ArrowParseError::NoDimensionSizes)?;

    let shape = dimension_sizes
        .iter()
        .filter_map(|v| v.map(|u| u as usize))
        .collect::<Vec<_>>();

    let list_data_array = array
        .column_by_name(consts::VALUES)
        .ok_or(ArrowParseError::NoValues)?;

    let data_array = list_data_array
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .ok_or(ArrowParseError::ListDowncastError)?
        .value(0);

    if shape.iter().product::<usize>() != data_array.len() {
        return Err(ArrowParseError::ShapeAndArrayLengthMismatch {
            expected: shape.iter().product(),
            got: data_array.len(),
        });
    }

    Ok(Arc::new(DefaultNdArrowArray::new(
        data_array.clone(),
        shape,
        None,
    )))
}

pub fn to_arrow_array<A: NdArrowArray + ?Sized>(
    array: &A,
) -> Result<Arc<dyn Array>, ArrowParseError> {
    let shape = array.shape();
    let inner_array = array.values();
    let inner_array_field = Field::new(consts::VALUES, inner_array.data_type().clone(), false);

    let inner_array = Arc::new(arrow::array::ListArray::new(
        Arc::new(inner_array_field.clone()),
        arrow::buffer::OffsetBuffer::<i32>::from_lengths(vec![inner_array.len()]),
        inner_array,
        None,
    ));

    let shape_array = Arc::new(arrow::array::UInt32Array::from_iter(
        shape.iter().copied().map(|s| s as u32),
    ));

    let shape_field = Field::new(
        consts::DIMENSION_SIZES,
        arrow::datatypes::DataType::UInt32,
        false,
    );

    let parsed_arrow_array = StructArray::new(
        vec![shape_field, inner_array_field].into(),
        vec![shape_array, inner_array],
        None,
    );

    Ok(Arc::new(parsed_arrow_array))
}
