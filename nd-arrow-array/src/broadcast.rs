use std::sync::Arc;

use arrow::array::{Array, ArrowPrimitiveType, Int8Array, PrimitiveArray};

use crate::nd_array::{default::DefaultNdArrowArray, NdArrowArray};

#[derive(Debug, Clone, thiserror::Error)]
pub enum BroadcastingError {
    #[error("Invalid shapes for broadcasting: {0:?} to {1:?}")]
    InvalidShapes(Vec<usize>, Vec<usize>),
    #[error("Unable to find/determine broadcast shape from shapes: {0:?}")]
    UnableToFindBroadcastShape(Vec<Vec<usize>>),
    #[error("Unsupported arrow data type: {0:?}")]
    UnsupportedArrowDataType(arrow::datatypes::DataType),
}

pub type BroadcastResult<T> = Result<T, BroadcastingError>;

fn find_subslice<T: PartialEq>(haystack: &[T], needle: &[T]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub fn find_broadcast_shape<S: AsRef<[usize]>>(shapes: &[S]) -> Option<Vec<usize>> {
    //Remove all 1s from the shapes as they are not relevant for broadcasting
    let shapes = shapes
        .iter()
        .map(|shape| {
            shape
                .as_ref()
                .iter()
                .copied()
                .filter(|&x| x != 1)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    //Find the biggest shape
    let max_shape = shapes.into_iter().max_by_key(|shape| shape.len());

    return max_shape;
}

fn broadcast_reshape_args(shape: &[usize], target_shape: &[usize]) -> Option<(usize, usize)> {
    //Remove all 1s from the shape and target_shape as they are not relevant for broadcasting
    let cleaned_shape = shape
        .iter()
        .copied()
        .filter(|&x| x != 1)
        .collect::<Vec<_>>();
    let cleaned_target_shape = target_shape
        .iter()
        .copied()
        .filter(|&x| x != 1)
        .collect::<Vec<_>>();

    if cleaned_shape == cleaned_target_shape {
        return Some((1, 1));
    } else {
        if cleaned_shape.len() > cleaned_target_shape.len() {
            return None;
        }
        match find_subslice(&shape, &target_shape) {
            Some(start_loc) => {
                let repeat_slice_count = target_shape[..start_loc].iter().product::<usize>();
                let repeat_element_counter = target_shape[start_loc + shape.len()..]
                    .iter()
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
    shape: &[usize],
    target_shape: &[usize],
) -> BroadcastResult<PrimitiveArray<T>> {
    let (repeat_slice_count, repeat_element_counter) = broadcast_reshape_args(shape, target_shape)
        .ok_or(BroadcastingError::InvalidShapes(
            shape.to_vec(),
            target_shape.to_vec(),
        ))?;

    // Verify length equality
    assert_eq!(
        array.len() * repeat_slice_count * repeat_element_counter,
        target_shape.iter().product::<usize>(),
        "Length mismatch: {} * {} * {} != {}",
        array.len(),
        repeat_slice_count,
        repeat_element_counter,
        target_shape.iter().product::<usize>()
    );

    Ok(PrimitiveArray::<T>::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_counter), //      `n` times
            ),
    ))
}

fn broadcast_boolean_array(
    array: &arrow::array::BooleanArray,
    shape: &[usize],
    target_shape: &[usize],
) -> BroadcastResult<arrow::array::BooleanArray> {
    let (repeat_slice_count, repeat_element_counter) = broadcast_reshape_args(shape, target_shape)
        .ok_or(BroadcastingError::InvalidShapes(
            shape.to_vec(),
            target_shape.to_vec(),
        ))?;

    // Verify length equality
    assert_eq!(
        array.len() * repeat_slice_count * repeat_element_counter,
        target_shape.iter().product::<usize>(),
        "Length mismatch: {} * {} * {} != {}",
        array.len(),
        repeat_slice_count,
        repeat_element_counter,
        target_shape.iter().product::<usize>()
    );

    Ok(arrow::array::BooleanArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_counter), //      `n` times
            ),
    ))
}

fn broadcast_string_array(
    array: &arrow::array::StringArray,
    shape: &[usize],
    target_shape: &[usize],
) -> BroadcastResult<arrow::array::StringArray> {
    let (repeat_slice_count, repeat_element_counter) = broadcast_reshape_args(shape, target_shape)
        .ok_or(BroadcastingError::InvalidShapes(
            shape.to_vec(),
            target_shape.to_vec(),
        ))?;

    // Verify length equality
    assert_eq!(
        array.len() * repeat_slice_count * repeat_element_counter,
        target_shape.iter().product::<usize>(),
        "Length mismatch: {} * {} * {} != {}",
        array.len(),
        repeat_slice_count,
        repeat_element_counter,
        target_shape.iter().product::<usize>()
    );

    Ok(arrow::array::StringArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_counter), //      `n` times
            ),
    ))
}

fn broadcast_byte_array(
    array: &arrow::array::BinaryArray,
    shape: &[usize],
    target_shape: &[usize],
) -> BroadcastResult<arrow::array::BinaryArray> {
    let (repeat_slice_count, repeat_element_counter) = broadcast_reshape_args(shape, target_shape)
        .ok_or(BroadcastingError::InvalidShapes(
            shape.to_vec(),
            target_shape.to_vec(),
        ))?;

    // Verify length equality
    assert_eq!(
        array.len() * repeat_slice_count * repeat_element_counter,
        target_shape.iter().product::<usize>(),
        "Length mismatch: {} * {} * {} != {}",
        array.len(),
        repeat_slice_count,
        repeat_element_counter,
        target_shape.iter().product::<usize>()
    );

    Ok(arrow::array::BinaryArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat(item).take(repeat_element_counter), //      `n` times
            ),
    ))
}

pub fn broadcast_arrays(
    nd_arrays: &[Arc<dyn NdArrowArray>],
) -> BroadcastResult<Vec<Arc<dyn Array>>> {
    let shapes = nd_arrays
        .iter()
        .map(|nd_array| nd_array.shape())
        .collect::<Vec<_>>();

    let target_shape = find_broadcast_shape(&shapes);

    match target_shape {
        Some(shape) => {
            let mut reshaped_arrays = Vec::new();
            for nd_array in nd_arrays {
                let array = nd_array.values();
                let target_shape = shape.clone();
                let reshaped_array =
                    broadcast_array_impl(array.as_ref(), nd_array.shape(), &target_shape)?;
                // Do something with reshaped_array
                reshaped_arrays.push(reshaped_array);
            }

            Ok(reshaped_arrays)
        }
        None => {
            return Err(BroadcastingError::UnableToFindBroadcastShape(
                shapes.iter().map(|s| s.to_vec()).collect(),
            ))
        }
    }
}

pub fn broadcast_array<S: NdArrowArray + ?Sized>(
    array: &S,
    target_shape: &[usize],
) -> BroadcastResult<Arc<dyn NdArrowArray>> {
    let shape = array.shape();
    let broadcasted_array = broadcast_array_impl(array.values().as_ref(), shape, target_shape)?;

    Ok(Arc::new(DefaultNdArrowArray::new(
        broadcasted_array,
        target_shape.to_vec(),
        None,
    )))
}

pub fn broadcast_array_impl(
    array: &dyn Array,
    shape: &[usize],
    target_shape: &[usize],
) -> BroadcastResult<Arc<dyn Array>> {
    match array.data_type() {
        arrow::datatypes::DataType::Int8 => {
            let array = array.as_any().downcast_ref::<Int8Array>().unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int16Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt8Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt16Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            let result = broadcast_primitive_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            let result = broadcast_boolean_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Timestamp(unit, _) => match unit {
            arrow::datatypes::TimeUnit::Second => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampSecondArray>()
                    .unwrap();
                let result = broadcast_primitive_array(array, shape, target_shape)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Millisecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampMillisecondArray>()
                    .unwrap();
                let result = broadcast_primitive_array(array, shape, target_shape)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Microsecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    .unwrap();
                let result = broadcast_primitive_array(array, shape, target_shape)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Nanosecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampNanosecondArray>()
                    .unwrap();
                let result = broadcast_primitive_array(array, shape, target_shape)?;
                Ok(Arc::new(result))
            }
        },
        arrow::datatypes::DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let result = broadcast_string_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::BinaryArray>()
                .unwrap();
            let result = broadcast_byte_array(array, shape, target_shape)?;
            Ok(Arc::new(result))
        }
        dtype => Err(BroadcastingError::UnsupportedArrowDataType(dtype.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {}
}
