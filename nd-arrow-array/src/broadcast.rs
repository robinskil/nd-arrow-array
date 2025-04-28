use std::sync::Arc;

use arrow::array::{Array, ArrowPrimitiveType, Int8Array, PrimitiveArray};

use crate::nd_array::{default::DefaultNdArrowArray, dimension::Dimension, NdArrowArray};

#[derive(Debug, Clone, thiserror::Error)]
pub enum BroadcastingError {
    #[error("Invalid shapes for broadcasting: {0:?} to {1:?}")]
    InvalidShapes(Vec<Dimension>, Vec<Dimension>),
    #[error("Unable to find/determine broadcast shape from shapes: {0:?}")]
    UnableToFindBroadcastShape(Vec<Vec<Dimension>>),
    #[error("Unsupported arrow data type: {0:?}")]
    UnsupportedArrowDataType(arrow::datatypes::DataType),
}

pub type BroadcastResult<T> = Result<T, BroadcastingError>;

fn find_subslice<T: PartialEq>(haystack: &[T], needle: &[T]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub fn find_broadcast_dimensions<S: AsRef<[Dimension]>>(
    dimensions: &[S],
) -> Option<Vec<Dimension>> {
    //Find the biggest dimensions array
    let dimensions: Vec<Vec<Dimension>> = dimensions
        .into_iter()
        .map(|dim| {
            dim.as_ref()
                .into_iter()
                .map(|d| d.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut max_dimension_array: Option<Vec<Dimension>> = None;
    for dim in dimensions {
        if max_dimension_array.is_none() || dim.len() > max_dimension_array.as_ref().unwrap().len()
        {
            max_dimension_array = Some(dim);
        }
    }

    max_dimension_array
}

fn broadcast_reshape_args(
    dimensions: &[Dimension],
    target_dimensions: &[Dimension],
) -> Option<(usize, usize)> {
    if dimensions == target_dimensions {
        return Some((1, 1));
    } else {
        if dimensions.len() > target_dimensions.len() {
            return None;
        }
        // If the dimensions are empty (scalar), we can return the product of the target dimensions
        if dimensions.len() == 0 {
            return Some((1, target_dimensions.iter().map(|d| d.size()).product()));
        }
        match find_subslice(&target_dimensions, &dimensions) {
            Some(start_loc) => {
                let repeat_slice_count = target_dimensions[..start_loc]
                    .iter()
                    .map(|d| d.size())
                    .product::<usize>();
                let repeat_element_counter = target_dimensions[start_loc + dimensions.len()..]
                    .iter()
                    .map(|d| d.size())
                    .product::<usize>();

                Some((repeat_slice_count, repeat_element_counter))
            }
            None => {
                return None;
            }
        }
    }
}

fn broadcast_primitive_array<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> BroadcastResult<PrimitiveArray<T>> {
    Ok(PrimitiveArray::<T>::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_count), //      `n` times
            ),
    ))
}

fn broadcast_boolean_array(
    array: &arrow::array::BooleanArray,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> BroadcastResult<arrow::array::BooleanArray> {
    Ok(arrow::array::BooleanArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_count), //      `n` times
            ),
    ))
}

fn broadcast_string_array(
    array: &arrow::array::StringArray,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> BroadcastResult<arrow::array::StringArray> {
    Ok(arrow::array::StringArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_count), //      `n` times
            ),
    ))
}

fn broadcast_byte_array(
    array: &arrow::array::BinaryArray,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> BroadcastResult<arrow::array::BinaryArray> {
    Ok(arrow::array::BinaryArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_count), //      `n` times
            ),
    ))
}

pub fn broadcast_array<S: NdArrowArray + ?Sized>(
    array: &S,
    target_dimensions: &[Dimension],
) -> BroadcastResult<Arc<dyn NdArrowArray>> {
    let dimensions = array.dimensions();
    let (repeat_slice, repeat_element) =
        broadcast_reshape_args(dimensions, target_dimensions).ok_or(
            BroadcastingError::InvalidShapes(dimensions.to_vec(), target_dimensions.to_vec()),
        )?;

    let broadcasted_array =
        broadcast_array_impl(array.array().as_ref(), repeat_element, repeat_slice)?;

    Ok(Arc::new(DefaultNdArrowArray::new(
        broadcasted_array,
        target_dimensions.to_vec(),
    )))
}

pub fn broadcast_array_impl(
    array: &dyn Array,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> BroadcastResult<Arc<dyn Array>> {
    match array.data_type() {
        arrow::datatypes::DataType::Int8 => {
            let array = array.as_any().downcast_ref::<Int8Array>().unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int16Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt8Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt16Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            let result = broadcast_boolean_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Timestamp(unit, _) => match unit {
            arrow::datatypes::TimeUnit::Second => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampSecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Millisecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampMillisecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Microsecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Nanosecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampNanosecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
        },
        arrow::datatypes::DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let result = broadcast_string_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::BinaryArray>()
                .unwrap();
            let result = broadcast_byte_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        dtype => Err(BroadcastingError::UnsupportedArrowDataType(dtype.clone())),
    }
}
