use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayRef, AsArray, BinaryArray, BooleanArray, NullArray, PrimitiveArray, StringArray,
    },
    datatypes::Int8Type,
};

use crate::nd_array::{dimension::Dimension, NdArrowArray};

pub mod chunker;
pub mod compute;

pub mod prelude {
    pub use super::chunker::*;
    pub use super::compute::*;
    pub use super::stream_arrays_to_chunked;
    pub use super::Chunk;
    pub use super::ChunkedNdArrowArray;
}

pub fn stream_arrays_to_chunked<I: Iterator<Item = (Arc<dyn NdArrowArray>, Vec<usize>)>>(
    arrays: I,
) -> impl Iterator<Item = Result<ChunkedNdArrowArray, String>> {
    arrays.map(|(array, chunk_shape)| ChunkedNdArrowArray::from_nd_arrow(array, &chunk_shape))
}

pub struct ChunkedNdArrowArray {
    pub origin_dimensions: Vec<Dimension>,
    pub chunks: Vec<Chunk>,
}

impl ChunkedNdArrowArray {
    pub fn from_nd_arrow(
        array: Arc<dyn NdArrowArray>,
        chunk_shape: &[usize],
    ) -> Result<Self, String> {
        // Validate chunk shape against the array's dimensions
        if chunk_shape.len() != array.dimensions().len() {
            return Err(format!(
                "Chunk shape length {} does not match array dimensions length {}",
                chunk_shape.len(),
                array.dimensions().len()
            ));
        }

        let inner_array = array.array();
        let array_shape = array.shape();

        let chunks = match inner_array.data_type() {
            arrow::datatypes::DataType::Null => {
                let null_array = inner_array.as_any().downcast_ref::<NullArray>().unwrap();
                chunker::chunk_nd_array_with_meta::<NullArray>(
                    null_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Boolean => {
                let boolean_array = inner_array.as_boolean();
                chunker::chunk_nd_array_with_meta::<BooleanArray>(
                    boolean_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Int8 => {
                let int8_array = inner_array.as_primitive::<Int8Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<Int8Type>>(
                    int8_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Int16 => {
                let int16_array = inner_array.as_primitive::<arrow::datatypes::Int16Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::Int16Type>>(
                    int16_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Int32 => {
                let int32_array = inner_array.as_primitive::<arrow::datatypes::Int32Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::Int32Type>>(
                    int32_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Int64 => {
                let int64_array = inner_array.as_primitive::<arrow::datatypes::Int64Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::Int64Type>>(
                    int64_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::UInt8 => {
                let uint8_array = inner_array.as_primitive::<arrow::datatypes::UInt8Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::UInt8Type>>(
                    uint8_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::UInt16 => {
                let uint16_array = inner_array.as_primitive::<arrow::datatypes::UInt16Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::UInt16Type>>(
                    uint16_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::UInt32 => {
                let uint32_array = inner_array.as_primitive::<arrow::datatypes::UInt32Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::UInt32Type>>(
                    uint32_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::UInt64 => {
                let uint64_array = inner_array.as_primitive::<arrow::datatypes::UInt64Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::UInt64Type>>(
                    uint64_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Float32 => {
                let float32_array = inner_array.as_primitive::<arrow::datatypes::Float32Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::Float32Type>>(
                    float32_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Float64 => {
                let float64_array = inner_array.as_primitive::<arrow::datatypes::Float64Type>();
                chunker::chunk_nd_array_with_meta::<PrimitiveArray<arrow::datatypes::Float64Type>>(
                    float64_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Timestamp(unit, _) => match unit {
                arrow::datatypes::TimeUnit::Second => {
                    let timestamp_array =
                        inner_array.as_primitive::<arrow::datatypes::TimestampSecondType>();
                    chunker::chunk_nd_array_with_meta::<
                        PrimitiveArray<arrow::datatypes::TimestampSecondType>,
                    >(timestamp_array, array_shape.as_slice(), chunk_shape)
                }
                arrow::datatypes::TimeUnit::Millisecond => {
                    let timestamp_array =
                        inner_array.as_primitive::<arrow::datatypes::TimestampMillisecondType>();
                    chunker::chunk_nd_array_with_meta::<
                        PrimitiveArray<arrow::datatypes::TimestampMillisecondType>,
                    >(timestamp_array, array_shape.as_slice(), chunk_shape)
                }
                arrow::datatypes::TimeUnit::Microsecond => {
                    let timestamp_array =
                        inner_array.as_primitive::<arrow::datatypes::TimestampMicrosecondType>();
                    chunker::chunk_nd_array_with_meta::<
                        PrimitiveArray<arrow::datatypes::TimestampMicrosecondType>,
                    >(timestamp_array, array_shape.as_slice(), chunk_shape)
                }
                arrow::datatypes::TimeUnit::Nanosecond => {
                    let timestamp_array =
                        inner_array.as_primitive::<arrow::datatypes::TimestampNanosecondType>();
                    chunker::chunk_nd_array_with_meta::<
                        PrimitiveArray<arrow::datatypes::TimestampNanosecondType>,
                    >(timestamp_array, array_shape.as_slice(), chunk_shape)
                }
            },
            arrow::datatypes::DataType::Utf8 => {
                let string_array = inner_array.as_string();
                chunker::chunk_nd_array_with_meta::<StringArray>(
                    string_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            arrow::datatypes::DataType::Binary => {
                let binary_array = inner_array.as_binary();
                chunker::chunk_nd_array_with_meta::<BinaryArray>(
                    binary_array,
                    array_shape.as_slice(),
                    chunk_shape,
                )
            }
            dtype => return Err(format!("Unsupported data type for chunking: {:?}", dtype)),
        };

        Ok(Self {
            origin_dimensions: array.dimensions().to_vec(),
            chunks,
        })
    }
}

pub struct Chunk {
    pub chunk_indices: Vec<usize>,
    pub shape: Vec<usize>,
    pub slices: Vec<(usize, usize)>,
    pub array: ArrayRef,
}
