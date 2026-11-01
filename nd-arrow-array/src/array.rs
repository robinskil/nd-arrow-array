//! A single N-dimensional Arrow array.

use arrow::array::{Array, ArrayRef, new_null_array};
use arrow::datatypes::DataType;

use crate::attributes::Attributes;
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

/// A flat Arrow array plus named C-order dimensions and attributes.
///
/// Invariant: `dims.num_elements() == values.len()`. Validity travels in the
/// null buffer of the values. There is no separate mask or fill value, so
/// readers map fill values to nulls when they build the values.
#[derive(Debug, Clone)]
pub struct NdArrowArray {
    values: ArrayRef,
    dims: Dimensions,
    attrs: Attributes,
}

impl PartialEq for NdArrowArray {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims
            && self.values.as_ref() == other.values.as_ref()
            && self.attrs == other.attrs
    }
}

impl NdArrowArray {
    /// Create an array. The length of `values` must equal the number of elements of `dims`.
    pub fn try_new(values: ArrayRef, dims: Dimensions) -> Result<Self> {
        if dims.checked_num_elements() != Some(values.len()) {
            return Err(nd_err!(
                ShapeMismatch,
                "values length {} does not match dimensions {dims} ({} elements)",
                values.len(),
                dims.num_elements()
            ));
        }
        Ok(Self::new_unchecked(values, dims, Attributes::new()))
    }

    /// A rank-0 array. `value` must have length 1.
    pub fn try_new_scalar(value: ArrayRef) -> Result<Self> {
        Self::try_new(value, Dimensions::scalar())
    }

    /// A rank-0 array holding one null of `data_type`.
    pub fn new_null_scalar(data_type: &DataType) -> Self {
        Self::new_unchecked(
            new_null_array(data_type, 1),
            Dimensions::scalar(),
            Attributes::new(),
        )
    }

    pub(crate) fn new_unchecked(values: ArrayRef, dims: Dimensions, attrs: Attributes) -> Self {
        debug_assert_eq!(dims.num_elements(), values.len());
        Self {
            values,
            dims,
            attrs,
        }
    }

    /// Replace the attributes.
    pub fn with_attributes(mut self, attrs: Attributes) -> Self {
        self.attrs = attrs;
        self
    }

    /// The flat values in C order.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// The named dimensions.
    pub fn dims(&self) -> &Dimensions {
        &self.dims
    }

    /// The attributes.
    pub fn attributes(&self) -> &Attributes {
        &self.attrs
    }

    /// The attributes, for changes in place.
    pub fn attributes_mut(&mut self) -> &mut Attributes {
        &mut self.attrs
    }

    /// The Arrow type of the values.
    pub fn data_type(&self) -> &DataType {
        self.values.data_type()
    }

    /// The size of each axis.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.shape()
    }

    /// The number of axes.
    pub fn rank(&self) -> usize {
        self.dims.rank()
    }

    /// Number of logical elements, equal to the length of the values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// True for rank 0.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_scalar()
    }

    /// Replace the dimensions without a copy of the values.
    /// The new dimensions must hold the same number of elements.
    pub fn reshape(self, dims: Dimensions) -> Result<Self> {
        let attrs = self.attrs;
        Ok(Self::try_new(self.values, dims)?.with_attributes(attrs))
    }

    /// Split into values, dimensions and attributes.
    pub fn into_parts(self) -> (ArrayRef, Dimensions, Attributes) {
        (self.values, self.dims, self.attrs)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int32Array;

    use super::*;

    #[test]
    fn len_mismatch_rejected() {
        let dims = Dimensions::try_new([("x", 2)]).unwrap();
        let result = NdArrowArray::try_new(Arc::new(Int32Array::from(vec![1, 2, 3])), dims);
        assert!(result.is_err());
    }

    #[test]
    fn reshape_keeps_values_and_attributes() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Dimensions::try_new([("x", 6)]).unwrap(),
        )
        .unwrap()
        .with_attributes(Attributes::new().with("units", "m"));

        let b = a
            .clone()
            .reshape(Dimensions::try_new([("y", 2), ("x", 3)]).unwrap())
            .unwrap();
        assert_eq!(b.shape(), vec![2, 3]);
        assert_eq!(b.values(), a.values());
        assert_eq!(b.attributes().get_str("units"), Some("m"));
        assert!(a.reshape(Dimensions::try_new([("x", 4)]).unwrap()).is_err());
    }
}
