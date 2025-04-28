use std::sync::Arc;

use arrow::{
    array::{Array, ArrowPrimitiveType, PrimitiveArray, Scalar, StringArray},
    datatypes::ArrowTimestampType,
};

use super::{dimension::Dimension, NdArrowArray};

pub const SCALAR_DIMENSION: Vec<Dimension> = vec![];

/// A multi-dimensional Arrow array implementation
pub struct DefaultNdArrowArray {
    /// The underlying Arrow array
    inner_array: Arc<dyn Array>,
    /// The dimensions defining the shape of this array
    dimensions: Vec<Dimension>,
}

impl DefaultNdArrowArray {
    /// Creates a new scalar array with a single value and no dimensions
    pub fn new_scalar(inner_array: Scalar<Arc<dyn Array>>) -> Self {
        Self {
            inner_array: inner_array.into_inner(),
            dimensions: vec![],
        }
    }

    /// Creates a new multi-dimensional array from an Arrow array and dimensions
    ///
    /// # Panics
    /// Panics if the product of dimension sizes does not match the array length
    pub fn new(inner_array: Arc<dyn Array>, dimensions: Vec<impl Into<Dimension>>) -> Self {
        let dimensions: Vec<Dimension> =
            dimensions.into_iter().map(|d| d.into()).collect::<Vec<_>>();
        // Ensure the shape and length of the inner array match
        let total_size: usize = dimensions.iter().map(|dim| dim.size()).product();

        // Check if the inner array is a scalar (length 1 and no dimensions)
        if !(total_size == 1 && dimensions.is_empty()) {
            assert_eq!(
                total_size,
                inner_array.len(),
                "Shape and inner array length do not match: expected {}, got {}",
                total_size,
                inner_array.len()
            );
        }

        Self {
            inner_array,
            dimensions,
        }
    }

    /// Creates a new array from a vector of primitive values
    pub fn from_vec<T: ArrowPrimitiveType>(
        vec: Vec<Option<T::Native>>,
        dimensions: Vec<impl Into<Dimension>>,
    ) -> Self
    where
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        let primitive_array = PrimitiveArray::<T>::from(vec);
        let inner_array: Arc<dyn Array> = Arc::new(primitive_array);
        Self::new(inner_array, dimensions)
    }

    /// Creates a new array from a vector of chrono datetime values
    pub fn from_vec_chrono(
        vec: Vec<Option<chrono::NaiveDateTime>>,
        dimensions: Vec<impl Into<Dimension>>,
    ) -> Self {
        //Convert vec to ArrowTimestampType
        let timestamp_array =
            arrow::array::PrimitiveArray::<arrow::datatypes::Time64NanosecondType>::from(
                vec.iter()
                    .map(|dt| dt.map(|dt| dt.timestamp_nanos()))
                    .collect::<Vec<_>>(),
            );
        let inner_array: Arc<dyn Array> = Arc::new(timestamp_array);

        Self::new(inner_array, dimensions)
    }

    /// Creates a new array from a vector of Arrow timestamp values
    pub fn from_vec_timestamp<T: ArrowTimestampType>(
        vec: Vec<Option<T::Native>>,
        dimensions: Vec<impl Into<Dimension>>,
    ) -> Self
    where
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        let timestamp_array = arrow::array::PrimitiveArray::<T>::from(vec);
        let inner_array: Arc<dyn Array> = Arc::new(timestamp_array);
        Self::new(inner_array, dimensions)
    }

    /// Creates a new array from a vector of string references
    pub fn from_vec_str(vec: Vec<Option<&str>>, dimensions: Vec<impl Into<Dimension>>) -> Self {
        let string_array = StringArray::from(vec);
        let inner_array: Arc<dyn Array> = Arc::new(string_array);
        Self::new(inner_array, dimensions)
    }

    /// Creates a new array from a vector of byte slice references
    pub fn from_byte_slice(vec: Vec<Option<&[u8]>>, dimensions: Vec<impl Into<Dimension>>) -> Self {
        let byte_array = arrow::array::BinaryArray::from(vec);
        let inner_array: Arc<dyn Array> = Arc::new(byte_array);

        Self::new(inner_array, dimensions)
    }
}

impl NdArrowArray for DefaultNdArrowArray {
    /// Returns the shape of the array as a vector of dimension sizes
    fn shape(&self) -> Vec<usize> {
        self.dimensions
            .iter()
            .map(|dim| dim.size())
            .collect::<Vec<_>>()
    }

    /// Returns a slice containing the array's dimensions
    fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// Returns a clone of the inner Arrow array
    fn array(&self) -> Arc<dyn Array> {
        self.inner_array.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Int32Type;

    #[test]
    fn test_new_scalar() {
        let array = Arc::new(PrimitiveArray::<Int32Type>::from(vec![Some(42)])) as Arc<dyn Array>;
        let scalar = Scalar::new(array);
        let nd_array = DefaultNdArrowArray::new_scalar(scalar);
        assert_eq!(nd_array.shape(), Vec::<usize>::new());
        assert_eq!(nd_array.dimensions().len(), 0);
    }

    #[test]
    fn test_new_with_dimensions() {
        let array = Arc::new(PrimitiveArray::<Int32Type>::from(vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
        ]));
        let dims = vec![("x", 2), ("y", 2)];
        let nd_array = DefaultNdArrowArray::new(array, dims);
        assert_eq!(nd_array.shape(), vec![2, 2]);
        assert_eq!(nd_array.dimensions().len(), 2);
    }

    #[test]
    #[should_panic]
    fn test_new_invalid_dimensions() {
        let array = Arc::new(PrimitiveArray::<Int32Type>::from(vec![Some(1), Some(2)]));
        let dims = vec![("x", 2), ("y", 2)]; // Should panic - dimensions imply 4 elements but array has 2
        DefaultNdArrowArray::new(array, dims);
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![Some(1), Some(2), Some(3), Some(4)];
        let dims = vec![("x", 2), ("y", 2)];
        let nd_array = DefaultNdArrowArray::from_vec::<Int32Type>(vec, dims);
        assert_eq!(nd_array.shape(), vec![2, 2]);
        assert_eq!(nd_array.dimensions().len(), 2);
    }

    #[test]
    fn test_from_vec_str() {
        let vec = vec![Some("a"), Some("b"), Some("c"), Some("d")];
        let dims = vec![("x", 2), ("y", 2)];
        let nd_array = DefaultNdArrowArray::from_vec_str(vec, dims);
        assert_eq!(nd_array.shape(), vec![2, 2]);
        assert_eq!(nd_array.dimensions().len(), 2);
    }

    #[test]
    fn test_from_byte_slice() {
        let vec = vec![
            Some("a".as_bytes()),
            Some("b".as_bytes()),
            Some("c".as_bytes()),
            Some("d".as_bytes()),
        ];
        let dims = vec![("x", 2), ("y", 2)];
        let nd_array = DefaultNdArrowArray::from_byte_slice(vec, dims);
        assert_eq!(nd_array.shape(), vec![2, 2]);
        assert_eq!(nd_array.dimensions().len(), 2);
    }
}
