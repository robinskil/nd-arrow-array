use std::{ops::Deref, sync::Arc};

use arrow::{
    array::{array, Array, ArrayRef, BooleanArray, NullArray, PrimitiveArray, StringArray},
    buffer::{BooleanBuffer, MutableBuffer, NullBuffer},
};
use explode::ExplodeArgs;
use shape::Shape;

pub mod dimension;
pub mod explode;
pub mod shape;

pub mod prelude {
    pub use super::dimension::Dimension;
    pub use super::explode::ExplodeArgs;
    pub use super::shape::Shape;
    pub use super::NdArrowArray;
}

#[derive(Debug, Clone)]
pub struct NdArrowArray {
    arrow_array: arrow::array::ArrayRef,
    shape: shape::Shape,
}

impl Deref for NdArrowArray {
    type Target = arrow::array::ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.arrow_array
    }
}

impl AsRef<NdArrowArray> for NdArrowArray {
    fn as_ref(&self) -> &NdArrowArray {
        self
    }
}

impl AsMut<NdArrowArray> for NdArrowArray {
    fn as_mut(&mut self) -> &mut NdArrowArray {
        self
    }
}

impl NdArrowArray {
    pub fn find_broadcast_shape<A: AsRef<NdArrowArray>, V: AsRef<[A]>>(
        arrays: V,
    ) -> Result<shape::Shape, String> {
        let arrays = arrays.as_ref();
        let mut max_shape = arrays[0].as_ref().shape().clone();
        for array in arrays.iter().skip(1) {
            match max_shape.partial_cmp(&array.as_ref().shape()) {
                Some(std::cmp::Ordering::Less) => {
                    max_shape = array.as_ref().shape().clone();
                }
                Some(std::cmp::Ordering::Greater) => {}
                Some(std::cmp::Ordering::Equal) => {}
                None => {
                    return Err("Shapes are not comparable".to_string());
                }
            }
        }

        Ok(max_shape)
    }
    /// Creates a new NdArrowArray from an Arrow array reference and a specified shape.
    ///
    /// # Arguments
    ///
    /// * `arrow_array` - Reference to the underlying Arrow array data.
    /// * `shape` - The shape describing the dimensions of the array.
    ///
    /// # Returns
    ///
    /// A new `NdArrowArray` instance wrapping the provided array data and associated shape.
    pub fn new(arrow_array: arrow::array::ArrayRef, shape: shape::Shape) -> Self {
        Self { arrow_array, shape }
    }
    /// Returns the shape for this `NdArrowArray`.
    ///
    /// # Returns
    ///
    /// A cloned `Shape` describing the dimensions of the array.
    pub fn shape(&self) -> &shape::Shape {
        &self.shape
    }
    /// Reshapes this `NdArrowArray` to the specified `target_shape`.
    /// If the array is scalar, it expands the single value to match the new shape. Otherwise, it
    /// explodes the data according to the difference in shape dimensions.
    ///
    /// # Arguments
    ///
    /// * `target_shape` - The new shape to apply.
    ///
    /// # Returns
    ///
    /// A new `NdArrowArray` instance with the updated shape.
    pub fn broadcast(&self, target_shape: &shape::Shape) -> Self {
        if self.shape.is_scalar() {
            let arrow_array = fast_scalar_explode(&self.arrow_array, target_shape);
            Self::new(arrow_array, target_shape.clone())
        } else {
            let explode_args = self.shape.explode_diff(target_shape).unwrap();

            let arrow_array = explode(&self.arrow_array, explode_args);
            Self::new(arrow_array, target_shape.clone())
        }
    }
    /// Checks whether this `NdArrowArray` is scalar, i.e., if it has only a single element.
    ///
    /// # Returns
    ///
    /// `true` if the array shape indicates a scalar value, otherwise `false`.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    /// Consumes this `NdArrowArray` and returns the underlying Arrow array reference.
    pub fn into_arrow_array(self) -> arrow::array::ArrayRef {
        self.arrow_array
    }
}

fn explode_primitive_impl<T: arrow::array::ArrowPrimitiveType>(
    array: PrimitiveArray<T>,
    explode_args: ExplodeArgs,
) -> PrimitiveArray<T> {
    let mut builder = arrow::array::PrimitiveBuilder::<T>::with_capacity(
        array.len() * explode_args.repeat_elems * explode_args.repeat_slices,
    );

    for _ in 0..explode_args.repeat_slices {
        for v in array.iter() {
            for _ in 0..explode_args.repeat_elems {
                builder.append_option(v);
            }
        }
    }

    builder.finish()
}

fn explode_null_buffer(buffer: &NullBuffer, explode_args: ExplodeArgs) -> NullBuffer {
    NullBuffer::new(explode_boolean_buffer(buffer.inner(), explode_args))
}

fn explode_boolean_buffer(buffer: &BooleanBuffer, explode_args: ExplodeArgs) -> BooleanBuffer {
    let repeated_elems = buffer
        .into_iter()
        .map(|v| std::iter::repeat(v).take(explode_args.repeat_elems));

    let repeated_slices = repeated_elems
        .flat_map(|v| std::iter::repeat(v).take(explode_args.repeat_slices))
        .flatten();

    BooleanBuffer::from_iter(repeated_slices)
}

fn explode_boolean(array: &BooleanArray, explode_args: ExplodeArgs) -> BooleanArray {
    let new_buffer = explode_boolean_buffer(array.values(), explode_args.clone());
    let new_null_buffer = array
        .nulls()
        .map(|buffer| explode_null_buffer(buffer, explode_args));

    BooleanArray::new(new_buffer, new_null_buffer)
}

fn explode_string(array: &StringArray, explode_args: ExplodeArgs) -> ArrayRef {
    let string_iter: StringArray = array
        .into_iter()
        .flat_map(|v| std::iter::repeat(v).take(explode_args.repeat_elems))
        .collect();

    let arrow_arr = Arc::new(string_iter) as ArrayRef;

    let vectorized_arrow_arr =
        arrow::compute::concat(&vec![&*arrow_arr; explode_args.repeat_slices]).unwrap();

    vectorized_arrow_arr
}

macro_rules! explode_primitive {
    ($array:expr, $explode_args: expr, $array_type:ty) => {{
        let original_array = $array.as_any().downcast_ref::<$array_type>().unwrap();

        let target_buffer = explode_primitive_impl(original_array.clone(), $explode_args);
        Arc::new(target_buffer) as Arc<dyn Array>
    }};
}

fn explode(array: &arrow::array::ArrayRef, explode_args: ExplodeArgs) -> ArrayRef {
    match array.data_type() {
        arrow::datatypes::DataType::Null => {
            let array = Arc::new(NullArray::new(
                array.len() * explode_args.repeat_elems * explode_args.repeat_slices,
            )) as Arc<dyn Array>;
            array
        }
        arrow::datatypes::DataType::Boolean => {
            let original_array = array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();

            let target_buffer = explode_boolean(original_array, explode_args);
            Arc::new(target_buffer) as Arc<dyn Array>
        }
        arrow::datatypes::DataType::Int8 => {
            explode_primitive!(array, explode_args, arrow::array::Int8Array)
        }
        arrow::datatypes::DataType::Int16 => {
            explode_primitive!(array, explode_args, arrow::array::Int16Array)
        }
        arrow::datatypes::DataType::Int32 => {
            explode_primitive!(array, explode_args, arrow::array::Int32Array)
        }
        arrow::datatypes::DataType::Int64 => {
            explode_primitive!(array, explode_args, arrow::array::Int64Array)
        }
        arrow::datatypes::DataType::UInt8 => {
            explode_primitive!(array, explode_args, arrow::array::UInt8Array)
        }
        arrow::datatypes::DataType::UInt16 => {
            explode_primitive!(array, explode_args, arrow::array::UInt16Array)
        }
        arrow::datatypes::DataType::UInt32 => {
            explode_primitive!(array, explode_args, arrow::array::UInt32Array)
        }
        arrow::datatypes::DataType::UInt64 => {
            explode_primitive!(array, explode_args, arrow::array::UInt64Array)
        }
        arrow::datatypes::DataType::Float32 => {
            explode_primitive!(array, explode_args, arrow::array::Float32Array)
        }
        arrow::datatypes::DataType::Float64 => {
            explode_primitive!(array, explode_args, arrow::array::Float64Array)
        }
        arrow::datatypes::DataType::Timestamp(time_unit, _) => match time_unit {
            arrow::datatypes::TimeUnit::Second => {
                explode_primitive!(array, explode_args, arrow::array::TimestampSecondArray)
            }
            arrow::datatypes::TimeUnit::Millisecond => {
                explode_primitive!(array, explode_args, arrow::array::TimestampMillisecondArray)
            }
            arrow::datatypes::TimeUnit::Microsecond => {
                explode_primitive!(array, explode_args, arrow::array::TimestampMicrosecondArray)
            }
            arrow::datatypes::TimeUnit::Nanosecond => {
                explode_primitive!(array, explode_args, arrow::array::TimestampNanosecondArray)
            }
        },
        arrow::datatypes::DataType::Utf8 => {
            let original_array = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();

            explode_string(original_array, explode_args)
        }
        _ => panic!("Unsupported data type for explode"),
    }
}

#[inline(always)]
fn fast_null_buffer_scalar_explode(buffer: &NullBuffer, new_len: usize) -> NullBuffer {
    NullBuffer::new(fast_boolean_scalar_explode(buffer.inner(), new_len))
}

#[inline(always)]
fn fast_boolean_scalar_explode(buffer: &BooleanBuffer, new_len: usize) -> BooleanBuffer {
    if buffer.is_empty() {
        return BooleanBuffer::new_unset(new_len);
    }

    if buffer.value(0) {
        BooleanBuffer::new_set(new_len)
    } else {
        BooleanBuffer::new_unset(new_len)
    }
}

fn fast_primitive_scalar_explode_impl<T: arrow::array::ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    new_len: usize,
) -> PrimitiveArray<T> {
    if array.is_empty() || array.is_null(0) {
        return PrimitiveArray::new_null(new_len);
    }
    let value = array.value(0);
    let null_buffer = array
        .nulls()
        .map(|buffer| fast_null_buffer_scalar_explode(buffer, new_len));
    let mut_buffer = MutableBuffer::from(vec![value; new_len]);

    PrimitiveArray::new(mut_buffer.into(), null_buffer)
}

macro_rules! fast_primitive_scalar_explode {
    ($array:expr, $target_size: expr, $array_type:ty) => {{
        let original_array = $array.as_any().downcast_ref::<$array_type>().unwrap();

        let target_buffer = fast_primitive_scalar_explode_impl(original_array, $target_size);
        Arc::new(target_buffer) as Arc<dyn Array>
    }};
}

fn string_scalar_explode(array: &StringArray, new_len: usize) -> StringArray {
    if array.is_empty() || array.is_null(0) {
        StringArray::new_null(new_len)
    } else {
        let value = array.value(0);
        StringArray::from_iter_values(std::iter::repeat(value).take(new_len))
    }
}

fn fast_scalar_explode(array: &arrow::array::ArrayRef, target_shape: &Shape) -> Arc<dyn Array> {
    let mut target_size = target_shape.shape_size.iter().product::<usize>();
    //If everything is a scalar, we need to make sure we have at least one element
    if target_size == 0 {
        target_size = 1;
    }

    match array.data_type() {
        arrow::datatypes::DataType::Null => {
            let array = Arc::new(NullArray::new(target_size)) as Arc<dyn Array>;
            array
        }
        arrow::datatypes::DataType::Boolean => {
            let original_array = array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();

            let target_buffer = fast_boolean_scalar_explode(original_array.values(), target_size);
            let null_buffer = original_array
                .nulls()
                .map(|buffer| fast_null_buffer_scalar_explode(buffer, target_size));

            Arc::new(BooleanArray::new(target_buffer, null_buffer)) as Arc<dyn Array>
        }
        arrow::datatypes::DataType::Int8 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::Int8Array)
        }
        arrow::datatypes::DataType::Int16 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::Int16Array)
        }
        arrow::datatypes::DataType::Int32 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::Int32Array)
        }
        arrow::datatypes::DataType::Int64 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::Int64Array)
        }
        arrow::datatypes::DataType::UInt8 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::UInt8Array)
        }
        arrow::datatypes::DataType::UInt16 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::UInt16Array)
        }
        arrow::datatypes::DataType::UInt32 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::UInt32Array)
        }
        arrow::datatypes::DataType::UInt64 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::UInt64Array)
        }
        arrow::datatypes::DataType::Float32 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::Float32Array)
        }
        arrow::datatypes::DataType::Float64 => {
            fast_primitive_scalar_explode!(array, target_size, arrow::array::Float64Array)
        }
        arrow::datatypes::DataType::Timestamp(time_unit, _) => match time_unit {
            arrow::datatypes::TimeUnit::Second => {
                fast_primitive_scalar_explode!(
                    array,
                    target_size,
                    arrow::array::TimestampSecondArray
                )
            }
            arrow::datatypes::TimeUnit::Millisecond => {
                fast_primitive_scalar_explode!(
                    array,
                    target_size,
                    arrow::array::TimestampMillisecondArray
                )
            }
            arrow::datatypes::TimeUnit::Microsecond => {
                fast_primitive_scalar_explode!(
                    array,
                    target_size,
                    arrow::array::TimestampMicrosecondArray
                )
            }
            arrow::datatypes::TimeUnit::Nanosecond => {
                fast_primitive_scalar_explode!(
                    array,
                    target_size,
                    arrow::array::TimestampNanosecondArray
                )
            }
        },
        arrow::datatypes::DataType::Utf8 => {
            let original_array = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();

            let target_buffer = string_scalar_explode(original_array, target_size);
            Arc::new(target_buffer) as Arc<dyn Array>
        }
        _ => panic!("Unsupported data type for scalar explode"),
    }
}
